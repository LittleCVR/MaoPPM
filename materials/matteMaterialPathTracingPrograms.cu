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
#include    "sampler.h"
#include    "utility.h"

/*----------------------------------------------------------------------------
 *  namespace
 *----------------------------------------------------------------------------*/
using namespace optix;
using namespace MaoPPM;



rtBuffer<float,       1>  sampleList;

rtDeclareVariable(float,    rayEpsilon, , );
rtDeclareVariable(rtObject, rootObject, , );

rtDeclareVariable(Ray  , currentRay, rtCurrentRay          , );
rtDeclareVariable(float, tHit      , rtIntersectionDistance, );

rtDeclareVariable(RadianceRayPayload, radianceRayPayload, rtPayload, );
rtDeclareVariable(ShadowRayPayload  , shadowRayPayload  , rtPayload, );

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

    // Sample one light.
    const Light & light = lightList[0];

    float3 shadowRayDirection = light.position - hitPoint;
    float3 normalizedShadowRayDirection = normalize(shadowRayDirection);
    float distanceSquared = dot(shadowRayDirection, shadowRayDirection);
    float distance = sqrtf(distanceSquared);
    Ray ray(hitPoint, normalizedShadowRayDirection, ShadowRay, rayEpsilon, distance-rayEpsilon);

    ShadowRayPayload shadowRayPayload;
    shadowRayPayload.attenuation = 1.0f;
    rtTrace(rootObject, ray, shadowRayPayload);

    float3 radiance = shadowRayPayload.attenuation * pairwiseMul(light.flux, Kd) *
        fmaxf(0.0f, dot(ffnormal, normalizedShadowRayDirection)) /
        (4.0f * M_PIf * distanceSquared);

    RadianceRayPayload & payload = radianceRayPayload;
    // check if there're enough photons or the ray depth is too deep
    if (payload.depth >= MAX_RAY_DEPTH)
        return;
    ++payload.depth;

    // otherwise, samples a new direction
    float3 W = normalize(ffnormal);
    float3 U = cross(W, make_float3(0.0f, 1.0f, 0.0f));
    if (fabsf(U.x) < 0.001f && fabsf(U.y) < 0.001f && fabsf(U.z) < 0.001f)
        U = cross(W, make_float3(1.0f, 0.0f, 0.0f));
    U = normalize(U);
    float3 V = cross(W, U);

    uint sampleIndex = payload.sampleIndexBase;
    float2 sample = make_float2(sampleList[sampleIndex], sampleList[sampleIndex+1]);
    payload.sampleIndexBase += 2;

    // Tweak sample according to cosine term.
    sample.x = asinf(sample.x * 2.0f - 1.0f) / M_PIf + 0.5f;
    sample.y = asinf(sample.y * 2.0f - 1.0f) / M_PIf + 0.5f;
    float3 newDirection = sampleHemisphereUniformly(sample);
    newDirection = newDirection.x*U + newDirection.y*V + newDirection.z*W;

    Ray newRay(hitPoint, newDirection, RadianceRay, rayEpsilon);
    rtTrace(rootObject, newRay, payload);
    payload.radiance *= dot(-direction, newDirection) / M_PIf;
    payload.radiance += radiance;
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
