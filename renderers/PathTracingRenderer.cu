/*
 * =============================================================================
 *
 *       Filename:  PathTracingRenderer.cu
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  2011-06-24 23:20:49
 *
 *         Author:  Chun-Wei Huang (LittleCVR), 
 *        Company:  Communication & Multimedia Laboratory,
 *                  Department of Computer Science & Information Engineering,
 *                  National Taiwan University
 *
 * =============================================================================
 */

#include    "PathTracingRenderer.h"

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



rtDeclareVariable(uint2, launchIndex, rtLaunchIndex, );
rtDeclareVariable(uint2, launchSize ,              , );

rtBuffer<float4,      2>  outputBuffer;
rtBuffer<float,       1>  sampleList;

rtDeclareVariable(uint, nSamplesPerThread  , , );
rtDeclareVariable(uint, maxRayDepth        , , );
rtDeclareVariable(uint, frameCount         , , );

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
RT_PROGRAM void trace()
{
    uint offset = LAUNCH_OFFSET_2D(launchIndex, launchSize);
    uint sampleIndex = nSamplesPerThread * offset;

    Ray ray;
    NormalRayPayload payload;
    uint depth = 0;
    float3 wo, wi, L = make_float3(0.0f), throughput = make_float3(1.0f);
    /* TODO */
    Intersection * intersection = NULL;
    BSDF bsdf;
    for (uint i = 0; i < maxRayDepth; ++i) {
        // Start from camera.
        if (depth == 0) {
            /* TODO: move this task to the camera class */
            // Generate camera ray.
            float2 screenSize = make_float2(outputBuffer.size());
            float2 sample = GET_2_SAMPLES(sampleList, sampleIndex);
            float2 cameraRayDirection = (make_float2(launchIndex) + sample) / screenSize * 2.0f - 1.0f;
            wi = normalize(cameraRayDirection.x*cameraU + cameraRayDirection.y*cameraV + cameraW);
            ray = Ray(cameraPosition, wi, NormalRay, rayEpsilon);
        }
        // Start from surface.
        else {
            /*TODO*/
            float  probability;
            float3 sample = GET_3_SAMPLES(sampleList, sampleIndex);
            float3 f = bsdf.sampleF(wo, &wi, sample, &probability);
            if (probability == 0.0f) continue;
            throughput = f * throughput * fabsf(dot(wi, intersection->dg()->normal)) / probability;
            ray = Ray(intersection->dg()->point, wi, NormalRay, rayEpsilon);
        }

        // trace ray
        payload.reset();
        rtTrace(rootObject, ray, payload);
        if (!payload.isHit) continue;

        ++depth;
        wo = -wi;
        intersection = payload.intersection();
        intersection->getBSDF(&bsdf);

        // Evaluate radiance.
        L += throughput * estimateAllDirectLighting(
                intersection->dg()->point, bsdf, wo);
    }

    if (frameCount == 0)
        outputBuffer[launchIndex] = make_float4(0.0f);
    float frame = static_cast<float>(frameCount);
    outputBuffer[launchIndex] = (1.0f / (frame + 1.0f)) * make_float4(L, 1.0f) +
        (frame / (frame + 1.0f)) * outputBuffer[launchIndex];
}   /* -----  end of function generatePixelSamples  ----- */
