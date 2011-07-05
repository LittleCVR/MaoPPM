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

rtBuffer<float4,       2>  outputBuffer;
rtBuffer<float,        1>  sampleList;
rtBuffer<SamplePoint,  1>  samplePointList;
rtBuffer<unsigned int, 1>  pathCountList;

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
    /* TODO: This function is too long, must refactor it. */

    // Clear output buffer.
    if (frameCount == 0)
        outputBuffer[launchIndex] = make_float4(0.0f);

    unsigned int offset = LAUNCH_OFFSET_2D(launchIndex, launchSize);
    unsigned int sampleIndex = nSamplesPerThread * offset;
    unsigned int samplePointIndex = maxRayDepth * offset;
    unsigned int pathCountIndex   = maxRayDepth * offset;
    float splitSample = GET_1_SAMPLE(sampleList, sampleIndex);
    unsigned int lightPathMaxDepth = min(maxRayDepth-1,
            static_cast<unsigned int>(floorf(splitSample*maxRayDepth)));
    unsigned int eyePathMaxDepth   = maxRayDepth - lightPathMaxDepth;

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

    /* TODO: Tracing eye path and tracing light path shares a lot of same code fragment.
             Should refactor it in the future. */

    // Trace eye paths.
    unsigned int depth = 0;
    float3 throughput = make_float3(1.0f);
    for (unsigned int i = 0; i < eyePathMaxDepth; ++i) {
        SamplePoint & samplePoint = samplePointList[samplePointIndex + depth];

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
        if (!trace(ray, samplePoint.intersection(), &bsdf, &wo))
            break;
        ++depth;

        samplePoint.flags |= SamplePoint::isHit;
        samplePoint.setBSDF(bsdf);
        samplePoint.wo         = wo;
        samplePoint.throughput = throughput;

        intersection = samplePoint.intersection();
    }

    // Trace light paths.
    depth = 0;
    for (unsigned int i = 0; i < lightPathMaxDepth; ++i) {
        SamplePoint & samplePoint = samplePointList[samplePointIndex + eyePathMaxDepth + depth];

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
        if (!trace(ray, samplePoint.intersection(), &bsdf, &wo))
            break;
        ++depth;

        samplePoint.flags |= SamplePoint::isHit;
        samplePoint.setBSDF(bsdf);
        samplePoint.wo         = wo;
        samplePoint.throughput = throughput;

        intersection = samplePoint.intersection();
    }

    // Sum path count.
    {
        for (unsigned int i = 0; i < maxRayDepth; ++i)
            pathCountList[pathCountIndex + i] = 0;
        // pathCountList[0] correspond to path length == 2,
        // which means it must be a path like: eye <--> bounce point <--> light.
        // So every valid sample point is also a valid bounce point except the case
        // it is on a specular surface.
        for (unsigned int i = 0; i < maxRayDepth; ++i)
            if (samplePointList[samplePointIndex+i].flags & SamplePoint::isHit &&
                    !samplePointList[samplePointIndex+i].bsdf()->isSpecular())
            {
                ++pathCountList[pathCountIndex];
            }
        // Do not recount i == 0.
        for (unsigned int i = 1; i < eyePathMaxDepth; ++i)
            if (samplePointList[samplePointIndex+i].flags & SamplePoint::isHit &&
                    !samplePointList[samplePointIndex+i].bsdf()->isSpecular())
            {
                ++pathCountList[pathCountIndex+i];
            }
        for (unsigned int i = 0; i < eyePathMaxDepth; ++i) {
            SamplePoint & eyePathSamplePoint = samplePointList[samplePointIndex + i];
            if (!(eyePathSamplePoint.flags & SamplePoint::isHit)) break;
            for (unsigned int j = 0; j < lightPathMaxDepth; ++j) {
                SamplePoint & lightPathSamplePoint = samplePointList[samplePointIndex + eyePathMaxDepth + j];
                if (!(lightPathSamplePoint.flags & SamplePoint::isHit)) break;
                // Add path count.
                if (!eyePathSamplePoint.bsdf()->isSpecular() && !lightPathSamplePoint.bsdf()->isSpecular())
                    pathCountList[pathCountIndex + i + j + 1] += 1;
            }
        }
    }

    /*
       Let s = the index of vertex from eye path,
           t = the index of vertex from light path.
       There are four cases:
           s = 0, t = 0: emission from light directly to the eye
           s > 0, t = 0: the classical path tracing algorithm
           s > 0, t > 0: estimation of the radiance by connecting
                         s from the eye path and t from the light path
           s = 0, t > 0: this is the most tricky one,
                         since it may contribute to other pixels on the image plane
                         therefore, it needs to be synchronized with other thread
                         we will ignore this case for now
    */

    float3 L = make_float3(0.0f);
    for (unsigned int i = 0; i < eyePathMaxDepth; ++i) {
        SamplePoint & eyePathSamplePoint = samplePointList[samplePointIndex + i];
        // Do not have to calculate radiance if it's not valid or it's on a specular surface.
        if (!(eyePathSamplePoint.flags & SamplePoint::isHit))
            break;
        if (eyePathSamplePoint.bsdf()->isSpecular())
            continue;

        // Estimate direct lighting. s > 0, t = 0
        L += eyePathSamplePoint.throughput * estimateAllDirectLighting(
             eyePathSamplePoint.intersection()->dg()->point, *eyePathSamplePoint.bsdf(), eyePathSamplePoint.wo) /
             (pathCountList[pathCountIndex+i]);

        // Link every possible path.
        for (unsigned int j = 0; j < lightPathMaxDepth; ++j) {
            SamplePoint & lightPathSamplePoint = samplePointList[samplePointIndex + eyePathMaxDepth + j];
            if (!(lightPathSamplePoint.flags & SamplePoint::isHit))
                break;
            if (lightPathSamplePoint.bsdf()->isSpecular())
                continue;

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
            float3 fs = eyePathSamplePoint.bsdf()->f(normalizedDirection, eyePathSamplePoint.wo);
            float coss = fabsf(dot(eyePathSamplePoint.wo, eyePathSamplePoint.intersection()->dg()->normal));
            float3 ft = lightPathSamplePoint.bsdf()->f(lightPathSamplePoint.wo, -normalizedDirection);
            float cost = fabsf(dot(lightPathSamplePoint.wo, lightPathSamplePoint.intersection()->dg()->normal));
            if (!isBlack(fs) && !isBlack(ft) && coss != 0.0f && cost != 0.0f)
                L += fs * ft * coss * cost *
                     lightPathSamplePoint.throughput * eyePathSamplePoint.throughput /
                     (distanceSquared * pathCountList[pathCountIndex+i+j+1]);
        }
    }

    float frame = static_cast<float>(frameCount);
    outputBuffer[launchIndex] = (1.0f / (frame + 1.0f)) * make_float4(L, 1.0f) +
        (frame / (frame + 1.0f)) * outputBuffer[launchIndex];
}   /* -----  end of function generatePixelSamples  ----- */
