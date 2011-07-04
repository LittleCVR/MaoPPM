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
#include    "Camera.h"
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

rtDeclareVariable(Camera, camera, , );

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
    // Clear output buffer.
    if (frameCount == 0)
        outputBuffer[launchIndex] = make_float4(0.0f);

    uint offset = LAUNCH_OFFSET_2D(launchIndex, launchSize);
    uint sampleIndex = nSamplesPerThread * offset;

    Ray ray;
    BSDF bsdf;
    Intersection intersection;
    float3 L = make_float3(0.0f);
    float3 throughput = make_float3(1.0f);
    for (unsigned int depth = 0; depth < maxRayDepth; /* EMPTY */) {
        // Start from camera.
        if (depth == 0) {
            float2 sample = GET_2_SAMPLES(sampleList, sampleIndex);
            ray = camera.generateCameraRay(
                    launchIndex.x, launchIndex.y, sample, NormalRay, rayEpsilon);
        }
        // Start from surface.
        else {
            float  probability;
            float3 sample = GET_3_SAMPLES(sampleList, sampleIndex);
            // remember that we are now shooting rays from a light
            // thus wo and wi must be swapped
            if (!bounce(&ray, *intersection.dg(), bsdf, sample, &probability, &throughput))
                break;
        }

        // trace ray
        float3 wo;
        if (!traceUntilNonSpecularSurface(&ray, maxRayDepth, &depth,
                    &intersection, &bsdf, &wo, &throughput))
        {
            break;
        }

        // Evaluate radiance.
        L += throughput * estimateAllDirectLighting(intersection.dg()->point, bsdf, wo);
    }

    float frame = static_cast<float>(frameCount);
    outputBuffer[launchIndex] = (1.0f / (frame + 1.0f)) * make_float4(L, 1.0f) +
        (frame / (frame + 1.0f)) * outputBuffer[launchIndex];
}   /* -----  end of function generatePixelSamples  ----- */
