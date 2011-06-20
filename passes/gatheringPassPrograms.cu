/*
 * =====================================================================================
 *
 *       Filename:  
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  2011-04-18 22:13:37
 *
 *         Author:  Chun-Wei Huang (LittleCVR), 
 *        Company:  Communication & Multimedia Laboratory,
 *                  Department of Computer Science & Information Engineering,
 *                  National Taiwan University
 *
 * =====================================================================================
 */

/*-----------------------------------------------------------------------------
 *  Header files from OptiX
 *-----------------------------------------------------------------------------*/
#include    <optix_world.h>

/*-----------------------------------------------------------------------------
 *  header files of our own
 *-----------------------------------------------------------------------------*/
#include    "global.h"
#include    "sampler.h"
#include    "utility.h"
#include    "PPMRenderer.h"

/*-----------------------------------------------------------------------------
 *  namespace
 *-----------------------------------------------------------------------------*/
using namespace optix;
using namespace MaoPPM;



/*-----------------------------------------------------------------------------
 *  buffers
 *-----------------------------------------------------------------------------*/
rtBuffer<float4,                   2>  outputBuffer;
rtBuffer<Light,                    1>  lightList;
rtBuffer<PPMRenderer::PixelSample, 2>  pixelSampleList;
rtBuffer<PPMRenderer::Photon,      1>  photonMap;

/*-----------------------------------------------------------------------------
 *  variables
 *-----------------------------------------------------------------------------*/
rtDeclareVariable(float,    rayEpsilon, , );
rtDeclareVariable(rtObject, rootObject, , );

rtDeclareVariable(uint2, launchIndex, rtLaunchIndex, );
rtDeclareVariable(uint2, launchSize ,              , );

rtDeclareVariable(uint, nEmittedPhotons, , );



/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  generateRay
 *  Description:  The ray generaton function of the pixel sampling program.
 * =====================================================================================
 */
RT_PROGRAM void generateRay()
{
    PPMRenderer::PixelSample & pixelSample = pixelSampleList[launchIndex];

    // Compute direct illumination.
    float3 direct = make_float3(0.0f);
    if (pixelSample.flags & PIXEL_SAMPLE_HIT) {
        // Sample 1 light.
        const Light & light = lightList[0];

        float3 direction = light.position - pixelSample.position;
        float3 normalizedDirection = normalize(direction);
        float distanceSquared = dot(direction, direction);
        float distance = sqrtf(distanceSquared);
        Ray ray(pixelSample.position, normalizedDirection, PPMRenderer::GatheringRay, rayEpsilon, distance-rayEpsilon);

        PPMRenderer::GatheringRayPayload payload;
        payload.attenuation = 1.0f;
        rtTrace(rootObject, ray, payload);

        float3 attenuation = make_float3(0.0f);
        if (pixelSample.material == MaoPPM::MatteMaterial)
            attenuation = pixelSample.Kd;
        else if (pixelSample.material == MaoPPM::PlasticMaterial) {
            float3 half = normalize(normalize(direction) + pixelSample.incidentDirection);
            attenuation = pixelSample.Kd +
                (pixelSample.exponent + 2.0f) / (2.0f * M_PIf) *
                powf(dot(half, pixelSample.normal), pixelSample.exponent) * pixelSample.Ks;
        }

        direct = payload.attenuation * pairwiseMul(light.flux, attenuation) *
            fmaxf(0.0f, dot(pixelSample.normal, normalizedDirection)) /
            (4.0f * M_PIf * distanceSquared);
    }

    // Compute indirect illumination.
    float3 indirect = make_float3(0.0f);
    float3 flux     = make_float3(0.0f);
    uint nAccumulatedPhotons = 0u;
    if (pixelSample.flags & PIXEL_SAMPLE_HIT) {
        uint stack[32];
        uint stackPosition = 0u;
        uint stackNode     = 0u;
        stack[stackPosition++] = 0u;
        do {
            const PPMRenderer::Photon & photon = photonMap[stackNode];
            if (photon.axis == PHOTON_NULL)
                stackNode = stack[--stackPosition];
            else {
                float3 diff = pixelSample.position - photon.position;
                float distanceSquared = dot(diff, diff);
                if (distanceSquared < pixelSample.radiusSquared) {
                    float3 attenuation = make_float3(1.0f);
                    if (pixelSample.material == MaoPPM::MatteMaterial)
                        attenuation = pixelSample.Kd;
                    else if (pixelSample.material == MaoPPM::PlasticMaterial) {
                        float3 half = normalize(pixelSample.incidentDirection + photon.incidentDirection);
                        attenuation = pixelSample.Kd +
                            (pixelSample.exponent + 2.0f) / (2.0f * M_PIf) *
                            powf(dot(half, pixelSample.normal), pixelSample.exponent) * pixelSample.Ks;
                    }

                    flux += pairwiseMul(attenuation, photon.flux) *
                        dot(photon.normal, pixelSample.normal);
                    ++nAccumulatedPhotons;
                }

                if(photon.axis == PHOTON_LEAF)
                    stackNode = stack[--stackPosition];
                else {
                    float d;
                    if      (photon.axis == AXIS_X)  d = diff.x;
                    else if (photon.axis == AXIS_Y)  d = diff.y;
                    else                             d = diff.z;

                    // Calculate the next child selector. 0 is left, 1 is right.
                    int selector = d < 0.0f ? 0 : 1;
                    if (d*d < pixelSample.radiusSquared)
                        stack[stackPosition++] = (stackNode << 1) + 2 - selector;
                    stackNode = (stackNode << 1) + 1 + selector;
                }
            }
        } while (stackNode != 0) ;

        // Compute new N, R.
        float alpha = 0.7f;
        float R2 = pixelSample.radiusSquared;
        float N = pixelSample.nPhotons;
        float M = static_cast<float>(nAccumulatedPhotons) ;
        float newN = N + alpha*M;
        pixelSample.nPhotons = newN;

        float reductionFactor2 = 1.0f;
        float newR2 = R2;
        if (M != 0) {
            reductionFactor2 = (N + alpha*M) / (N + M);
            newR2 = R2 * reductionFactor2;
            pixelSample.radiusSquared = newR2;
        }

        // Compute indirect flux.
        float3 newFlux = (pixelSample.flux + flux) * reductionFactor2;
        pixelSample.flux = newFlux;
        indirect = newFlux / (M_PIf * newR2) / nEmittedPhotons;
    }

    float3 finalColor = direct + indirect;
    outputBuffer[launchIndex] = make_float4(finalColor.x, finalColor.y, finalColor.z, 1.0f);
}   /* -----  end of function generateRay  ----- */



/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  handleException
 *  Description:  
 * =====================================================================================
 */
RT_PROGRAM void handleException()
{
}   /* -----  end of function handleException  ----- */



/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  handleGatheringRayMiss
 *  Description:  
 * =====================================================================================
 */
RT_PROGRAM void handleGatheringRayMiss()
{
}   /* -----  end of function handleGatheringRayMiss  ----- */
