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
rtBuffer<float,       1>  sampleList;
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
 *         Name:  generatePixelSamples
 *  Description:  
 * =============================================================================
 */
RT_PROGRAM void generatePixelSamples()
{
    outputBuffer[launchIndex] = make_float4(0.0f, 0.0f, 0.0f, 1.0f);

    /* TODO: give this task to the camera class */
    // Generate camera ray.
    float2 screenSize = make_float2(outputBuffer.size());
    float2 sample = make_float2(0.5f, 0.5f); 
    float2 cameraRayDirection = (make_float2(launchIndex) + sample) / screenSize * 2.0f - 1.0f;
    float3 worldRayDirection = normalize(cameraRayDirection.x*cameraU + cameraRayDirection.y*cameraV + cameraW);

    // Intersect with the scene.
    Ray ray(cameraPosition, worldRayDirection, NormalRay, rayEpsilon);
    NormalRayPayload payload;
    rtTrace(rootObject, ray, payload);

    // Record pixel sample.
    PixelSample & pixelSample = pixelSampleList[launchIndex];
    pixelSample.isHit         = payload.isHit;
    if (!pixelSample.isHit) return;
    pixelSample.intersection  = payload.intersection;
    pixelSample.wo            = -worldRayDirection;
    pixelSample.direct        = make_float3(0.0f);

    /* TODO: move this task to the light class */
    // Evaluate direct illumination.
    const Light & light = lightList[0];
    Intersection & intersection = pixelSample.intersection;
    float3 shadowRayDirection = light.position - intersection.dg()->point;
    float3 normalizedShadowRayDirection = normalize(shadowRayDirection);
    float distanceSquared = dot(shadowRayDirection, shadowRayDirection);
    float distance = sqrtf(distanceSquared);

    ShadowRayPayload shadowRayPayload;
    shadowRayPayload.isHit = 0;
    ray = Ray(intersection.dg()->point, normalizedShadowRayDirection, ShadowRay, rayEpsilon, distance-rayEpsilon);
    rtTrace(rootObject, ray, shadowRayPayload);

    if (shadowRayPayload.isHit)
        outputBuffer[launchIndex] = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
    else {
        BSDF * bsdf = intersection.bsdf();
        Matrix4x4 * transform = intersection.worldToObject();
        float3 wo = transformVector(*transform, -worldRayDirection);
        float3 wi = transformVector(*transform, normalizedShadowRayDirection);
        float3 f = bsdf->f(wo, wi);
        pixelSample.direct = pairwiseMul(f, light.flux) / (4.0f * M_PIf * distanceSquared);
        outputBuffer[launchIndex] = make_float4(pixelSample.direct, 1.0f);
    }
}   /* -----  end of function generatePixelSamples  ----- */
