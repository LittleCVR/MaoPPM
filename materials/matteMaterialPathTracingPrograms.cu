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

#include    "Matte.h"
#include    "PathTracingRenderer.h"

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
#include    "Light.h"

/*----------------------------------------------------------------------------
 *  namespace
 *----------------------------------------------------------------------------*/
using namespace optix;
using namespace MaoPPM;



rtDeclareVariable(uint2, launchIndex, rtLaunchIndex, );
rtDeclareVariable(uint2, launchSize ,              , );

rtBuffer<char,  1>  heap;
rtBuffer<Light, 1>  lightList;
rtBuffer<float, 1>  sampleList;

rtDeclareVariable(rtObject, rootObject, , );
rtDeclareVariable(float3, geometricNormal, attribute geometric_normal, ); 
rtDeclareVariable(float3, shadingNormal  , attribute shading_normal  , ); 

rtDeclareVariable(float , rayEpsilon ,                       , );
rtDeclareVariable(Ray   , currentRay , rtCurrentRay          , );
rtDeclareVariable(float , tHit       , rtIntersectionDistance, );
rtDeclareVariable(PathTracingRenderer::RadianceRayPayload, radianceRayPayload, rtPayload, );
rtDeclareVariable(PathTracingRenderer::ShadowRayPayload  , shadowRayPayload  , rtPayload, );

rtDeclareVariable(HeapIndex, materialIndex, , );



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
    Ray ray(hitPoint, normalizedShadowRayDirection, PathTracingRenderer::ShadowRay, rayEpsilon, distance-rayEpsilon);

    PathTracingRenderer::ShadowRayPayload shadowRayPayload;
    shadowRayPayload.attenuation = 1.0f;
    rtTrace(rootObject, ray, shadowRayPayload);

    Matte & material = reinterpret_cast<Matte &>(heap[materialIndex]);

    float3 radiance = shadowRayPayload.attenuation *                // visibility
        pairwiseMul(light.flux, material.m_kd) *                    // BRDF
        fmaxf(0.0f, dot(ffnormal, normalizedShadowRayDirection)) /  // cosine term
        (4.0f * M_PIf * distanceSquared);                           // solid angle and area

    PathTracingRenderer::RadianceRayPayload & payload = radianceRayPayload;
    // check if there're enough photons or the ray depth is too deep
    if (payload.depth >= DEFAULT_MAX_RAY_DEPTH)
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
    float3 wi = sampleHemisphereUniformly(sample);
    wi = wi.x*U + wi.y*V + wi.z*W;

    Ray newRay(hitPoint, wi, PathTracingRenderer::RadianceRay, rayEpsilon);
    rtTrace(rootObject, newRay, payload);
    payload.radiance  = pairwiseMul(material.m_kd, payload.radiance);
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
