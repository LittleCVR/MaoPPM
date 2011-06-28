/*
 * =============================================================================
 *
 *       Filename:  PPMRenderer.cu
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  2011-06-27 11:09:25
 *
 *         Author:  Chun-Wei Huang (LittleCVR), 
 *        Company:  Communication & Multimedia Laboratory,
 *                  Department of Computer Science & Information Engineering,
 *                  National Taiwan University
 *
 * =============================================================================
 */

#include    "PPMRenderer.h"

/*----------------------------------------------------------------------------
 *  Header files from OptiX
 *----------------------------------------------------------------------------*/
#include    <optix_world.h>

/*----------------------------------------------------------------------------
 *  header files of our own
 *----------------------------------------------------------------------------*/
#include    "global.h"
#include    "payload.h"
#include    "utility.h"
#include    "BSDF.h"
#include    "Light.h"

/*----------------------------------------------------------------------------
 *  namespace
 *----------------------------------------------------------------------------*/
using namespace optix;
using namespace MaoPPM;



typedef PPMRenderer::PixelSample  PixelSample;
typedef PPMRenderer::Photon       Photon;



rtDeclareVariable(uint2, launchIndex, rtLaunchIndex, );
rtDeclareVariable(uint2, launchSize ,              , );

rtBuffer<float4,      2>  outputBuffer;
rtBuffer<Light,       1>  lightList;
rtBuffer<PixelSample, 2>  pixelSampleList;
rtBuffer<Photon,      1>  photonList;
rtBuffer<Photon,      1>  photonMap;
rtBuffer<float,       1>  sampleList;

rtDeclareVariable(uint, frameCount         , , );
rtDeclareVariable(uint, nSamplesPerThread  , , );
rtDeclareVariable(uint, nPhotonsPerThread  , , );
rtDeclareVariable(uint, nEmittedPhotons    , , );
rtDeclareVariable(uint, maxRayDepth        , , );

rtDeclareVariable(rtObject, rootObject, , );
rtDeclareVariable(float,    rayEpsilon, , );

rtDeclareVariable(float3, cameraPosition, , );
rtDeclareVariable(float3, cameraU       , , );
rtDeclareVariable(float3, cameraV       , , );
rtDeclareVariable(float3, cameraW       , , );

rtDeclareVariable(NormalRayPayload, normalRayPayload, rtPayload, );
rtDeclareVariable(ShadowRayPayload, shadowRayPayload, rtPayload, );



/* 
 * ===  FUNCTION  ==============================================================
 *         Name:  generatePixelSamples
 *  Description:  
 * =============================================================================
 */
RT_PROGRAM void generatePixelSamples()
{
    if (frameCount != 0) return;

    // Clear output buffer.
    if (frameCount == 0)
        outputBuffer[launchIndex] = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
    // Clear pixel sample.
    PixelSample & pixelSample = pixelSampleList[launchIndex];
    pixelSample.reset();

    /* TODO: move this task to the camera class */
    // Generate camera ray.
    Ray ray;
    {
        float2 screenSize = make_float2(outputBuffer.size());
        float2 sample = make_float2(0.5f, 0.5f); 
        float2 cameraRayDirection = (make_float2(launchIndex) + sample) / screenSize * 2.0f - 1.0f;
        float3 worldRayDirection = normalize(cameraRayDirection.x*cameraU + cameraRayDirection.y*cameraV + cameraW);
        ray = Ray(cameraPosition, worldRayDirection, NormalRay, rayEpsilon);
    }

    // Intersect with the scene.
    NormalRayPayload payload;
    payload.reset();
    rtTrace(rootObject, ray, payload);
    pixelSample.isHit = payload.isHit;
    if (!pixelSample.isHit) return;

    // Fill pixel sample data if hit.
    pixelSample.intersection   = payload.intersection();
    pixelSample.wo             = -ray.direction;
    pixelSample.direct         = make_float3(0.0f);
    pixelSample.flux           = make_float3(0.0f);
    pixelSample.nPhotons       = 0;
    /* TODO: hard coded */
    pixelSample.radiusSquared  = 32.0f;

    /* TODO: move this task to the light class */
    // Evaluate direct illumination.
    float3 Li;
    Intersection * intersection = pixelSample.intersection;
    {
        const Light * light = &lightList[0];
        float3 shadowRayDirection = light->position - intersection->dg()->point;
        float3 wi = normalize(shadowRayDirection);
        float distanceSquared = dot(shadowRayDirection, shadowRayDirection);
        float distance = sqrtf(distanceSquared);

//        /* TODO: remove these debug lines */
//        if (launchIndex.x == 449 && launchIndex.y == 252) {
//            rtPrintf("normal "); dump(intersection.dg()->normal); rtPrintf("\n");
//            rtPrintf("dpdu "); dump(intersection.dg()->dpdu); rtPrintf("\n");
//            rtPrintf("dpdv "); dump(intersection.dg()->dpdv); rtPrintf("\n");
//        }
//        //outputBuffer[launchIndex] = make_float4(intersection.dg()->normal / 2.0f + 0.5f, 1.0f);
//        //outputBuffer[launchIndex] = make_float4(normalize(intersection.dg()->dpdu) / 2.0f + 0.5f, 1.0f);
//        //outputBuffer[launchIndex] = make_float4(normalize(intersection.dg()->dpdv) / 2.0f + 0.5f, 1.0f);
//        outputBuffer[launchIndex] = make_float4(1.0f, 0.0f, 0.0f, 1.0f);

        ShadowRayPayload shadowRayPayload;
        shadowRayPayload.reset();
        ray = Ray(intersection->dg()->point, wi, ShadowRay, rayEpsilon, distance-rayEpsilon);
        rtTrace(rootObject, ray, shadowRayPayload);
        if (shadowRayPayload.isHit) return;

//        BSDF * bsdf = intersection->bsdf();
        BSDF bsdf; intersection->getBSDF(&bsdf);
        float3 f = bsdf.f(pixelSample.wo, wi);
        Li = f * light->flux / (4.0f * M_PIf * distanceSquared);
    }

    pixelSample.direct = Li;
}   /* -----  end of function generatePixelSamples  ----- */



/* 
 * ===  FUNCTION  ==============================================================
 *         Name:  shootPhotons
 *  Description:  
 * =============================================================================
 */
RT_PROGRAM void shootPhotons()
{
    uint offset = LAUNCH_OFFSET_2D(launchIndex, launchSize);
    uint sampleIndex = nSamplesPerThread * offset;
    uint photonIndex = nPhotonsPerThread * offset;

    // Clear photon list.
    for (uint i = 0; i < nPhotonsPerThread; i++)
        photonList[photonIndex+i].reset();

    Ray ray;
    NormalRayPayload payload;
    uint depth = 0;
    float3 wo, wi, flux;
    Intersection * intersection  = NULL;
//    BSDF         * bsdf          = NULL;
    BSDF bsdf;
    for (uint i = 0; i < nPhotonsPerThread; i++) {
        // starts from lights
        if (depth == 0) {
            /* TODO: sample random light */
            // sample light
            const Light & light = lightList[0];
            flux = light.flux;
            // sample direction
            float2 sample = GET_2_SAMPLES(sampleList, sampleIndex);
            wo = sampleUniformSphere(sample);
            ray = Ray(light.position, wo, NormalRay, rayEpsilon);
        }
        // starts from surface
        else {
            float  probability;
            float3 sample = GET_3_SAMPLES(sampleList, sampleIndex);
            // remember that we are now shooting rays from a light
            // thus wo and wi must be swapped
            float3 f = bsdf.sampleF(wi, &wo, sample, &probability);
            if (probability == 0.0f) continue;
            flux = f * flux * fabsf(dot(wo, intersection->dg()->normal)) / probability;
            // transform from object to world
            // remember that this transform's transpose is actually its inverse
            ray = Ray(intersection->dg()->point, wo, NormalRay, rayEpsilon);
        }

        // trace ray
        payload.reset();
        rtTrace(rootObject, ray, payload);
        if (!payload.isHit) continue;
        wi = -wo;
        intersection  = payload.intersection();
        intersection->getBSDF(&bsdf);

        // create photon
        Photon & photon = photonList[photonIndex+i];
        if (depth == 0)
            photon.flags |= Photon::Direct;
        else
            photon.flags |= Photon::Indirect;
        photon.position = intersection->dg()->point;
        photon.wi       = wi;
        photon.flux     = flux;

        // Increase depth, reset if necessary.
        ++depth;
        if (depth % maxRayDepth == 0)
            depth = 0;
    }
}   /* -----  end of function shootPhotons  ----- */



/* 
 * ===  FUNCTION  ==============================================================
 *         Name:  
 *  Description:  
 * =============================================================================
 */
RT_PROGRAM void estimateDensity()
{
    // Do not have to gather photons if pixel sample was not hit.
    PixelSample & pixelSample = pixelSampleList[launchIndex];
    if (!pixelSample.isHit) return;

    /* TODO: move this task to the KdTree class */
    // Compute indirect illumination.
    float3 flux = make_float3(0.0f);
    uint nAccumulatedPhotons = 0;
    {
        uint stack[32];
        uint stackPosition = 0;
        uint stackNode     = 0;
        stack[stackPosition++] = 0;
        do {
            const Photon & photon = photonMap[stackNode];
            if (photon.flags == Photon::Null)
                stackNode = stack[--stackPosition];
            else {
                Intersection * intersection  = pixelSample.intersection;
                BSDF bsdf; intersection->getBSDF(&bsdf);
                float3 diff = intersection->dg()->point - photon.position;
                float distanceSquared = dot(diff, diff);
                // Do not gather direct photons.
                if (!(photon.flags & Photon::Direct) &&
                    distanceSquared < pixelSample.radiusSquared)
                {
                    float3 f = bsdf.f(pixelSample.wo, photon.wi);
//                    float  s = 1.0f - distanceSquared / pixelSample.radiusSquared;
//                    float  k = 3.0f * s * s / M_PIf;
//                    flux += k * f * photon.flux;
                    flux += f * photon.flux;
                    ++nAccumulatedPhotons;
                }

                if (photon.flags & Photon::Leaf)
                    stackNode = stack[--stackPosition];
                else {
                    float d;
                    if      (photon.flags & Photon::AxisX)  d = diff.x;
                    else if (photon.flags & Photon::AxisY)  d = diff.y;
                    else                                     d = diff.z;

                    // Calculate the next child selector. 0 is left, 1 is right.
                    int selector = d < 0.0f ? 0 : 1;
                    if (d*d < pixelSample.radiusSquared)
                        stack[stackPosition++] = (stackNode << 1) + 2 - selector;
                    stackNode = (stackNode << 1) + 1 + selector;
                }
            }
        } while (stackNode != 0) ;

        // Compute new N, R.
        /* TODO: let alpha be configurable */
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
    }

    float3 indirect = pixelSample.flux / (M_PIf * pixelSample.radiusSquared) / nEmittedPhotons;
    outputBuffer[launchIndex] = make_float4(pixelSample.direct + indirect, 1.0f);
}   /* -----  end of function gatherPhotons  ----- */
