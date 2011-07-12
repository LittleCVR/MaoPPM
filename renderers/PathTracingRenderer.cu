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
rtBuffer<float3,       1>  radianceList;

rtDeclareVariable(uint, nSamplesPerThread  , , );
rtDeclareVariable(uint, maxRayDepth        , , );
rtDeclareVariable(uint, frameCount         , , );

rtDeclareVariable(Camera, camera, , );

rtDeclareVariable(NormalRayPayload, normalRayPayload, rtPayload, );
rtDeclareVariable(ShadowRayPayload, shadowRayPayload, rtPayload, );



/* 
 * ===  FUNCTION  ==============================================================
 *         Name:  
 *  Description:  
 * =============================================================================
 */
RT_PROGRAM void clear()
{
    unsigned int offset = LAUNCH_OFFSET_2D(launchIndex, launchSize);
    unsigned int index  = maxRayDepth * offset;
    // Clear path count.
    for (unsigned int i = 0; i < maxRayDepth; ++i) {
        pathCountList[index + i] = 0;
        radianceList [index + i] = make_float3(0.0f);
    }
}   /* -----  end of function clear  ----- */



/* 
 * ===  FUNCTION  ==============================================================
 *         Name:  
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
    unsigned int radianceIndex    = maxRayDepth * offset;
    float splitSample = GET_1_SAMPLE(sampleList, sampleIndex);
//    unsigned int lightPathMaxDepth = min(maxRayDepth,
//            static_cast<unsigned int>(floorf(splitSample*(maxRayDepth+1))));
//    unsigned int eyePathMaxDepth   = maxRayDepth - lightPathMaxDepth;
    unsigned int lightPathMaxDepth = maxRayDepth / 2;
    unsigned int eyePathMaxDepth   = maxRayDepth - lightPathMaxDepth;
    unsigned int eyePathSamplePointIndex   = samplePointIndex;
    unsigned int lightPathSamplePointIndex = eyePathSamplePointIndex + eyePathMaxDepth;

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
        SamplePoint & samplePoint = samplePointList[eyePathSamplePointIndex + depth];

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
        SamplePoint & samplePoint = samplePointList[lightPathSamplePointIndex + depth];

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
            if (probability == 0.0f) break;
            throughput /= probability;
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

    // Note that pathCountList[0] correspond to path length == 2.
    {
        // s = 0, t = 0: not necessary to calculate.
        // s > 0, t = 0
        for (unsigned int i = 0; i < eyePathMaxDepth; ++i) {
            SamplePoint & eyePathSamplePoint = samplePointList[eyePathSamplePointIndex + i];
            // Do not have to calculate radiance if it's not valid or it's on a specular surface.
            if (!(eyePathSamplePoint.flags & SamplePoint::isHit))
                break;
            if (eyePathSamplePoint.bsdf()->isSpecular())
                continue;
            // Add path count.
            atomicAdd(&pathCountList[pathCountIndex+i], 1);
            // Calculate radiance.
            float3 L = eyePathSamplePoint.throughput * estimateAllDirectLighting(
                    eyePathSamplePoint.intersection()->dg()->point, *eyePathSamplePoint.bsdf(), eyePathSamplePoint.wo);
            // Add contribution.
            atomicAdd(&radianceList[radianceIndex+i].x, L.x);
            atomicAdd(&radianceList[radianceIndex+i].y, L.y);
            atomicAdd(&radianceList[radianceIndex+i].z, L.z);
        }
//        // s > 0, t > 0
//        for (unsigned int i = 0; i < eyePathMaxDepth; ++i) {
//            SamplePoint & eyePathSamplePoint = samplePointList[eyePathSamplePointIndex + i];
//            // Do not have to calculate radiance if it's not valid or it's on a specular surface.
//            if (!(eyePathSamplePoint.flags & SamplePoint::isHit))
//                break;
//            if (eyePathSamplePoint.bsdf()->isSpecular())
//                continue;
//
//            // Link every possible path.
//            for (unsigned int j = 0; j < lightPathMaxDepth; ++j) {
//                SamplePoint & lightPathSamplePoint = samplePointList[lightPathSamplePointIndex + j];
//                if (!(lightPathSamplePoint.flags & SamplePoint::isHit))
//                    break;
//                if (lightPathSamplePoint.bsdf()->isSpecular())
//                    continue;
//
//                // Add path count.
//                atomicAdd(&pathCountList[pathCountIndex + i+j+1], 1);
//
//                // Test visibility.
//                float3 direction, normalizedDirection;
//                float distance, distanceSquared;
//                if (!isVisible(eyePathSamplePoint.intersection()->dg()->point,
//                            lightPathSamplePoint.intersection()->dg()->point,
//                            &direction, &normalizedDirection, &distance, &distanceSquared))
//                {
//                    continue;
//                }
//
//                // Compute radiance.
//                float3 fc = eyePathSamplePoint.bsdf()->f(eyePathSamplePoint.wo, normalizedDirection);
//                float3 fl = lightPathSamplePoint.bsdf()->f(-normalizedDirection, lightPathSamplePoint.wo);
//                if (isBlack(fc) || isBlack(fl))
//                    continue;
//                float G = fabsf(dot(normalizedDirection, eyePathSamplePoint.intersection()->dg()->normal)) *
//                    fabsf(dot(-normalizedDirection, lightPathSamplePoint.intersection()->dg()->normal)) /
//                    distanceSquared;
//                float3 L = fc * fl * G *
//                    lightPathSamplePoint.throughput * eyePathSamplePoint.throughput /
//                    pathCountList[pathCountIndex+i+j+1];
//                // Add contribution.
//                atomicAdd(&radianceList[radianceIndex + i+j+1].x, L.x);
//                atomicAdd(&radianceList[radianceIndex + i+j+1].y, L.y);
//                atomicAdd(&radianceList[radianceIndex + i+j+1].z, L.z);
//            }
//        }
        // s = 0, t > 0, the most tricky one
        for (unsigned int j = 0; j < lightPathMaxDepth; ++j) {
            SamplePoint & lightPathSamplePoint = samplePointList[lightPathSamplePointIndex + j];
            if (!(lightPathSamplePoint.flags & SamplePoint::isHit))
                break;
            if (lightPathSamplePoint.bsdf()->isSpecular())
                continue;

            // Must first project the point onto the image plane.
            float3 position = lightPathSamplePoint.intersection()->dg()->point;
            float3 pos = transformPoint(camera.worldToRaster(), position);
            uint2  ras = make_uint2(pos.x, pos.y);
            // Add path count.
            if (ras.x < camera.width && ras.y < camera.height) {
                unsigned int offset = LAUNCH_OFFSET_2D(ras, launchSize);
                unsigned int index  = maxRayDepth * offset;
                // Add path count.
                atomicAdd(&pathCountList[index+j], 1);
                // Calculate radiance.
                float3 diff = camera.position - lightPathSamplePoint.intersection()->dg()->point;
                float3 wo = normalize(diff);
                float distanceSquared = dot(diff, diff);
                if (!isVisible(camera.position, lightPathSamplePoint.intersection()->dg()->point))
                    continue;
                float3 L = lightPathSamplePoint.throughput *
                    lightPathSamplePoint.bsdf()->f(wo, lightPathSamplePoint.wo) *
                    fabsf(dot(wo, lightPathSamplePoint.intersection()->dg()->normal)) /
                    distanceSquared;
                // Add contribution.
                atomicAdd(&radianceList[index+j].x, L.x);
                atomicAdd(&radianceList[index+j].y, L.y);
                atomicAdd(&radianceList[index+j].z, L.z);
            }
        }
    }
}   /* -----  end of function trace  ----- */



/* 
 * ===  FUNCTION  ==============================================================
 *         Name:  
 *  Description:  
 * =============================================================================
 */
RT_PROGRAM void sum()
{
    unsigned int offset = LAUNCH_OFFSET_2D(launchIndex, launchSize);
    unsigned int index  = maxRayDepth * offset;

    float3 L = make_float3(0.0f);
    for (unsigned int i = 0; i < maxRayDepth; ++i)
        if (pathCountList[index+i] != 0)
            L += radianceList[index+i] / pathCountList[index+i];

    float frame = static_cast<float>(frameCount);
    outputBuffer[launchIndex] = (1.0f / (frame + 1.0f)) * make_float4(L, 1.0f) +
        (frame / (frame + 1.0f)) * outputBuffer[launchIndex];
}   /* -----  end of function clear  ----- */
