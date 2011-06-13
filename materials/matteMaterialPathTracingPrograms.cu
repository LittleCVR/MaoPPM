/*
 * =============================================================================
 *
 *       Filename:  matteMaterialPathTracingPrograms.cu
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  2011-03-29 15:46:55
 *
 *         Author:  Chun-Wei Huang (LittleCVR), 
 *        Company:  Communication & Multimedia Laboratory,
 *                  Department of Computer Science & Information Engineering,
 *                  National Taiwan University
 *
 * =============================================================================
 */

/*----------------------------------------------------------------------------
 *  Header files from OptiX
 *----------------------------------------------------------------------------*/
#include    <optix_world.h>

/*----------------------------------------------------------------------------
 *  header files of our own
 *----------------------------------------------------------------------------*/
#include    "global.h"
#include    "math.cu"
#include    "samplers.cu"

/*----------------------------------------------------------------------------
 *  namespace
 *----------------------------------------------------------------------------*/
using namespace optix;
using namespace MaoPPM;



rtBuffer<PixelSample, 2>  pixelSampleList;
rtBuffer<float,       1>  sampleList;
rtBuffer<Light,       1>  lightList;

rtDeclareVariable(float,    rayEpsilon, , );
rtDeclareVariable(rtObject, rootObject, , );

rtDeclareVariable(Ray  , currentRay, rtCurrentRay          , );
rtDeclareVariable(float, tHit      , rtIntersectionDistance, );

rtDeclareVariable(RadianceRayPayload      , radianceRayPayload      , rtPayload, );
rtDeclareVariable(ShadowRayPayload        , shadowRayPayload        , rtPayload, );

rtDeclareVariable(uint2, launchIndex, rtLaunchIndex, );
rtDeclareVariable(uint2, launchSize ,              , );

rtDeclareVariable(float3, geometricNormal, attribute geometric_normal, ); 
rtDeclareVariable(float3, shadingNormal  , attribute shading_normal  , ); 

rtDeclareVariable(float3, Kd, , );



/* 
 * ===  FUNCTION  ==============================================================
 *         Name:  handleRadianceRayClosestHit
 *  Description:  
 * =============================================================================
 */
RT_PROGRAM void handleRadianceRayClosestHit()
{
    float3 origin    = currentRay.origin;
    float3 direction = currentRay.direction;
    float3 worldShadingNormal   = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shadingNormal  ));
    float3 worldGeometricNormal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, geometricNormal));
    float3 ffnormal  = faceforward(worldShadingNormal, -direction, worldGeometricNormal);
    float3 diff      = tHit * direction;
    float3 hitPoint  = origin + diff;

    // Sample 1 light.
    const Light & light = lightList[0];

    float3 shadowRayDirection = light.position - hitPoint;
    float3 normalizedShadowRayDirection = normalize(shadowRayDirection);
    float distanceSquared = dot(shadowRayDirection, shadowRayDirection);
    float distance = sqrtf(distanceSquared);
    Ray ray(hitPoint, normalizedShadowRayDirection, ShadowRay, rayEpsilon, distance-rayEpsilon);

    ShadowRayPayload payload;
    payload.attenuation = 1.0f;
    rtTrace(rootObject, ray, payload);

    radianceRayPayload.radiance = payload.attenuation * pairwiseMul(light.flux, Kd) *
        fmaxf(0.0f, dot(ffnormal, normalizedShadowRayDirection)) /
        (4.0f * M_PIf * distanceSquared);
}   /* -----  end of function handleRadianceRayClosestHit  ----- */



/* 
 * ===  FUNCTION  ==============================================================
 *         Name:  handleShadowRayClosestHit
 *  Description:  
 * =============================================================================
 */
RT_PROGRAM void handleShadowRayAnyHit()
{
    shadowRayPayload.attenuation = 0.0f;
}   /* -----  end of function handleShadowRayAnyHit  ----- */
