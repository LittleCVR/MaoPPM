/*
 * =============================================================================
 *
 *       Filename:  IGPPMRenderer.cu
 *
 *    Description:  The Importons Guided Progressive Photon Map Renderer.
 *
 *        Version:  1.0
 *        Created:  2011-06-21 08:02:37
 *
 *         Author:  Chun-Wei Huang (LittleCVR), 
 *        Company:  Communication & Multimedia Laboratory,
 *                  Department of Computer Science & Information Engineering,
 *                  National Taiwan University
 *
 * =============================================================================
 */

#include    "IGPPMRenderer.h"

/*----------------------------------------------------------------------------
 *  Header files from OptiX
 *----------------------------------------------------------------------------*/
#include    <optix_world.h>

/*----------------------------------------------------------------------------
 *  header files of our own
 *----------------------------------------------------------------------------*/
#include    "global.h"
#include    "payload.h"
#include    "reflection.h"
#include    "utility.h"
#include    "Light.h"

/*----------------------------------------------------------------------------
 *  namespace
 *----------------------------------------------------------------------------*/
using namespace optix;
using namespace MaoPPM;



typedef IGPPMRenderer::PixelSample  PixelSample;
typedef IGPPMRenderer::Importon     Importon;
typedef IGPPMRenderer::Photon       Photon;



rtDeclareVariable(uint2, launchIndex, rtLaunchIndex, );
rtDeclareVariable(uint2, launchSize ,              , );

rtBuffer<float4,      2>  outputBuffer;
rtBuffer<Light,       1>  lightList;
rtBuffer<PixelSample, 2>  pixelSampleList;
rtBuffer<Importon,    1>  importonList;
rtBuffer<Photon,      1>  photonMap;
rtDeclareVariable(uint, nSamplesPerThread  , , );
rtDeclareVariable(uint, nImportonsPerThread, , );
rtDeclareVariable(uint, nPhotonsPerThread  , , );
rtDeclareVariable(uint, nEmittedPhotons    , , ) = 0;

rtDeclareVariable(rtObject, rootObject, , );
rtDeclareVariable(float3, geometricNormal, attribute geometric_normal, ); 
rtDeclareVariable(float3, shadingNormal  , attribute shading_normal  , ); 

rtDeclareVariable(float3, cameraPosition, , );
rtDeclareVariable(float3, cameraU       , , );
rtDeclareVariable(float3, cameraV       , , );
rtDeclareVariable(float3, cameraW       , , );

rtDeclareVariable(float, rayEpsilon, , );
rtDeclareVariable(NormalRayPayload, normalRayPayload, rtPayload, );
rtDeclareVariable(ShadowRayPayload, shadowRayPayload, rtPayload, );



/* 
 * ===  FUNCTION  ==============================================================
 *         Name:  estimateDensity
 *  Description:  
 * =============================================================================
 */
RT_PROGRAM void gather()
{
    uint offset = launchIndexOffset(launchIndex, launchSize);
    uint importonIndex = nImportonsPerThread * offset;

    // Compute indirect illumination.
    for (uint i = 0; i < nImportonsPerThread; ++i) {
        Importon & importon = importonList[importonIndex++];
        float3 flux = make_float3(0.0f);
        uint nAccumulatedPhotons = 0;
        if (launchIndex.x == 64 && launchIndex.y == 64) {
            rtPrintf("gather isHit: %d\n", importon.isHit);
        }
        if (importon.isHit) {
            uint stack[32];
            uint stackPosition = 0;
            uint stackNode     = 0;
            stack[stackPosition++] = 0;
            do {
                const Photon & photon = photonMap[stackNode];
                if (launchIndex.x == 64 && launchIndex.y == 64) {
                    rtPrintf("gather photon flags: %d\n", photon.flags);
                }
                if (photon.flags == Photon::Null)
                    stackNode = stack[--stackPosition];
                else {
                    float3 diff = importon.intersection.dg()->point - photon.position;
                    float distanceSquared = dot(diff, diff);
                    if (launchIndex.x == 64 && launchIndex.y == 64) {
                        rtPrintf("gather distance2: %f\n", distanceSquared);
                    }
                    if (distanceSquared < importon.radiusSquared) {
                        Intersection & intersection = importon.intersection;
                        BSDF * bsdf = intersection.bsdf();
                        Matrix4x4 * worldToObject = intersection.worldToObject();
                        float3 wo = transformVector(*worldToObject, importon.wo);
                        float3 wi = transformVector(*worldToObject, photon.wi);
                        flux += pairwiseMul(bsdf->f(wo, wi), photon.flux) *
                            fmaxf(0.0f, dot(photon.normal, intersection.dg()->normal));
                        ++nAccumulatedPhotons;
                        if (launchIndex.x == 64 && launchIndex.y == 64) {
                            rtPrintf("gather accu: %d\n", nAccumulatedPhotons);
                        }
                    }

                    if(photon.flags == Photon::Leaf)
                        stackNode = stack[--stackPosition];
                    else {
                        float d;
                        if      (photon.flags == Photon::AxisX)  d = diff.x;
                        else if (photon.flags == Photon::AxisY)  d = diff.y;
                        else                                     d = diff.z;

                        // Calculate the next child selector. 0 is left, 1 is right.
                        int selector = d < 0.0f ? 0 : 1;
                        if (d*d < importon.radiusSquared)
                            stack[stackPosition++] = (stackNode << 1) + 2 - selector;
                        stackNode = (stackNode << 1) + 1 + selector;
                    }
                }
            } while (stackNode != 0) ;

            // Compute new N, R.
            float alpha = 0.7f;
            float R2 = importon.radiusSquared;
            float N = importon.nPhotons;
            float M = static_cast<float>(nAccumulatedPhotons) ;
            float newN = N + alpha*M;
            importon.nPhotons = newN;

            float reductionFactor2 = 1.0f;
            float newR2 = R2;
            if (M != 0) {
                reductionFactor2 = (N + alpha*M) / (N + M);
                newR2 = R2 * reductionFactor2;
                importon.radiusSquared = newR2;
            }

            // Compute indirect flux.
            float3 newFlux = (importon.flux + flux) * reductionFactor2;
            importon.flux = newFlux;
        }
    }

    PixelSample & pixelSample = pixelSampleList[launchIndex];
    Intersection & intersection = pixelSample.intersection;
    BSDF * bsdf = intersection.bsdf();
    Matrix4x4 * worldToObject = intersection.worldToObject();
    float3 indirect = make_float3(0.0f);
    for (uint i = 0; i < nImportonsPerThread; i++) {
        Importon & importon = importonList[importonIndex++];
        if (importon.isHit) {
            float3 wo = transformVector(*worldToObject, pixelSample.wo);
            float3 wi = transformVector(*worldToObject, -importon.wo);
            indirect += importon.weight
                * pairwiseMul(bsdf->f(wo, wi), importon.flux)
                / (2.0f * M_PIf * importon.radiusSquared);
        }
    }
    indirect /= nEmittedPhotons;

//    outputBuffer[launchIndex] = make_float4(indirect, 1.0f);
}   /* -----  end of function estimateDensity  ----- */



/* 
 * ===  FUNCTION  ==============================================================
 *         Name:  handleNormalRayMiss
 *  Description:  
 * =============================================================================
 */
RT_PROGRAM void handleNormalRayMiss()
{
    normalRayPayload.isHit = false;
}   /* -----  end of function handleNormalRayMiss  ----- */
