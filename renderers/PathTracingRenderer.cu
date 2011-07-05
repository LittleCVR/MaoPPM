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



typedef PathTracingRenderer::SamplePoint SamplePoint;



rtDeclareVariable(uint2, launchIndex, rtLaunchIndex, );
rtDeclareVariable(uint2, launchSize ,              , );

rtBuffer<float4,      2>  outputBuffer;
rtBuffer<float,       1>  sampleList;
rtBuffer<SamplePoint, 1>  samplePointList;

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

    unsigned int offset = LAUNCH_OFFSET_2D(launchIndex, launchSize);
    unsigned int sampleIndex = nSamplesPerThread * offset;
    unsigned int samplePointIndex = maxRayDepth * offset;
    unsigned int eyePathMaxDepth   = maxRayDepth / 2;
    unsigned int lightPathMaxDepth = maxRayDepth / 2;

    Intersection * intersectionList =
        LOCAL_HEAP_GET_OBJECT_POINTER(Intersection, maxRayDepth * sizeof(Intersection) * offset);
    for (unsigned int i = 0; i < maxRayDepth; ++i) {
        SamplePoint & samplePoint = samplePointList[samplePointIndex + i];
        samplePoint.reset();
        samplePoint.setIntersection(intersectionList + i);
    }

    Ray ray;
    BSDF bsdf;
    Intersection * intersection = NULL;

    // Trace eye paths.
    unsigned int depth = 0;
    float3 throughput = make_float3(1.0f);
    for (unsigned int i = 0; i < eyePathMaxDepth; ++i) {
        SamplePoint & samplePoint = samplePointList[samplePointIndex + i];

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
            if (!bounce(&ray, *intersection->dg(), bsdf, sample, &probability, &throughput))
                break;
        }

        // trace ray
        float3 wo;
        if (!traceUntilNonSpecularSurface(&ray, maxRayDepth, &depth,
                    samplePoint.intersection(), &bsdf, &wo, &throughput))
        {
            break;
        }

        samplePoint.flags |= SamplePoint::isHit;
        samplePoint.setBSDF(bsdf);
        samplePoint.wo         = wo;
        samplePoint.throughput = throughput;

        intersection = samplePoint.intersection();
    }

    // Trace light paths.
    depth = 0;
    for (unsigned int i = 0; i < lightPathMaxDepth; ++i) {
        SamplePoint & samplePoint = samplePointList[samplePointIndex + eyePathMaxDepth + i];

        // Start from light.
        if (depth == 0) {
            // sample light
            float lightSample = GET_1_SAMPLE(sampleList, sampleIndex);
            const Light * light = sampleOneLightUniformly(lightSample);
            // sample direction
            float3 wo;
            float  probability;
            float2 sample = GET_2_SAMPLES(sampleList, sampleIndex);
            throughput = light->sampleL(sample, &wo, &probability);
            ray = Ray(light->position, wo, NormalRay, rayEpsilon);
        }
        // Start from surface.
        else {
            float  probability;
            float3 sample = GET_3_SAMPLES(sampleList, sampleIndex);
            if (!bounce(&ray, *intersection->dg(), bsdf, sample, &probability, &throughput))
                break;
        }

        // trace ray
        float3 wo;
        if (!traceUntilNonSpecularSurface(&ray, maxRayDepth, &depth,
                    samplePoint.intersection(), &bsdf, &wo, &throughput))
        {
            break;
        }

        samplePoint.flags |= SamplePoint::isHit;
        samplePoint.setBSDF(bsdf);
        samplePoint.wo         = wo;
        samplePoint.throughput = throughput;

        intersection = samplePoint.intersection();
    }

    float3 L = make_float3(0.0f);
    for (unsigned int i = 0; i < eyePathMaxDepth; ++i) {
        SamplePoint & eyePathSamplePoint = samplePointList[samplePointIndex + i];
        if (!(eyePathSamplePoint.flags & SamplePoint::isHit)) continue;

        // Estimate direct lighting.
        L += eyePathSamplePoint.throughput * estimateAllDirectLighting(
                eyePathSamplePoint.intersection()->dg()->point, *eyePathSamplePoint.bsdf(), eyePathSamplePoint.wo);

        // Link every possible path.
        for (unsigned int j = 0; j < lightPathMaxDepth; ++j) {
            SamplePoint & lightPathSamplePoint = samplePointList[samplePointIndex + eyePathMaxDepth + j];
            if (!(lightPathSamplePoint.flags & SamplePoint::isHit)) continue;

            // Test visibility.
            float3 direction, normalizedDirection;
            float distance, distanceSquared;
            if (!isVisible(eyePathSamplePoint.intersection()->dg()->point,
                           lightPathSamplePoint.intersection()->dg()->point,
                           &direction, &normalizedDirection, &distance, &distanceSquared))
            {
                continue;
            }

            // Compute radiance.
            L += lightPathSamplePoint.throughput *
                fabsf(dot(lightPathSamplePoint.wo, lightPathSamplePoint.intersection()->dg()->normal)) *
                lightPathSamplePoint.bsdf()->f(lightPathSamplePoint.wo, -normalizedDirection) *
                eyePathSamplePoint.bsdf()->f(normalizedDirection, eyePathSamplePoint.wo) *
                fabsf(dot(eyePathSamplePoint.wo, eyePathSamplePoint.intersection()->dg()->normal)) *
                eyePathSamplePoint.throughput /
                (4.0f * M_PIf * distanceSquared);
        }
    }

    float frame = static_cast<float>(frameCount);
    outputBuffer[launchIndex] = (1.0f / (frame + 1.0f)) * make_float4(L, 1.0f) +
        (frame / (frame + 1.0f)) * outputBuffer[launchIndex];
}   /* -----  end of function generatePixelSamples  ----- */
