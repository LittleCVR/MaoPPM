/*
 * =============================================================================
 *
 *       Filename:  Matte.cu
 *
 *    Description:  Matte material device codes.
 *
 *        Version:  1.0
 *        Created:  2011-06-21 12:59:55
 *
 *         Author:  Chun-Wei Huang (LittleCVR), 
 *        Company:  Communication & Multimedia Laboratory,
 *                  Department of Computer Science & Information Engineering,
 *                  National Taiwan University
 *
 * =============================================================================
 */

#include    "Matte.h"

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
#include    "DifferentialGeometry.h"
#include    "Intersection.h"

/*----------------------------------------------------------------------------
 *  namespace
 *----------------------------------------------------------------------------*/
using namespace optix;
using namespace MaoPPM;



rtDeclareVariable(uint2, launchIndex, rtLaunchIndex, );
rtDeclareVariable(uint2, launchSize ,              , );

rtBuffer<char,  1>  heap;
rtBuffer<float, 1>  sampleList;

rtDeclareVariable(rtObject, rootObject, , );
rtDeclareVariable(float3, geometricNormal, attribute geometric_normal, ); 
rtDeclareVariable(float3, shadingNormal  , attribute shading_normal  , ); 

rtDeclareVariable(float , rayEpsilon ,                       , );
rtDeclareVariable(Ray   , currentRay , rtCurrentRay          , );
rtDeclareVariable(float , tHit       , rtIntersectionDistance, );
rtDeclareVariable(NormalRayPayload, normalRayPayload, rtPayload, );
rtDeclareVariable(ShadowRayPayload, shadowRayPayload, rtPayload, );

rtDeclareVariable(Index, materialIndex, , );



/* 
 * ===  FUNCTION  ==============================================================
 *         Name:  handleNormalRayClosestHit
 *  Description:  
 * =============================================================================
 */
RT_PROGRAM void handleNormalRayClosestHit()
{
    /*TODO*/
    float3 wo = currentRay.direction;
    float3 worldShadingNormal   = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shadingNormal  ));
    float3 worldGeometricNormal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, geometricNormal));
    float3 ffnormal   = faceforward(worldShadingNormal, -wo, worldGeometricNormal);
    float3 difference = tHit * wo;
    float3 hitPoint   = currentRay.origin + difference;

    normalRayPayload.isHit = true;
    // Intersection
    Intersection & intersection = normalRayPayload.intersection;
    // differential geometry
    DifferentialGeometry * dg = intersection.dg();
    dg->point  = hitPoint;
    dg->normal = ffnormal;

    /*TODO*/
    // otherwise, samples a new direction
    float3 W = normalize(ffnormal);
    float3 U = cross(W, make_float3(0.0f, 1.0f, 0.0f));
    if (fabsf(U.x) < 0.001f && fabsf(U.y) < 0.001f && fabsf(U.z) < 0.001f)
        U = cross(W, make_float3(1.0f, 0.0f, 0.0f));
    U = normalize(U);
    float3 V = cross(W, U);

    Matrix4x4 * worldToObject = intersection.worldToObject();
    float * w2o = worldToObject->getData();
    w2o[ 0] = U.x ; w2o[ 1] = V.x ; w2o[ 2] = W.x ; w2o[ 3] = 0.0f;
    w2o[ 4] = U.y ; w2o[ 5] = V.y ; w2o[ 6] = W.y ; w2o[ 7] = 0.0f;
    w2o[ 8] = U.z ; w2o[ 9] = V.z ; w2o[10] = W.z ; w2o[11] = 0.0f;
    w2o[12] = 0.0f; w2o[13] = 0.0f; w2o[14] = 0.0f; w2o[15] = 1.0f;

    // BSDF
    Matte & material = reinterpret_cast<Matte &>(heap[materialIndex]);
    DifferentialGeometry dgShading;
    dgShading.point  = hitPoint;
    dgShading.normal = ffnormal;
    BSDF * bsdf = intersection.bsdf();
    *bsdf = BSDF(dgShading, ffnormal);

    /*TODO*/
    bsdf->m_nBxDFs = 1;
    Lambertian * bxdf = reinterpret_cast<Lambertian *>(bsdf->bxdfList());
    *bxdf = Lambertian(material.m_kd);
}   /* -----  end of function handleNormalRayClosestHit  ----- */



/* 
 * ===  FUNCTION  ==============================================================
 *         Name:  handleShadowRayClosestHit
 *  Description:  
 * =============================================================================
 */
RT_PROGRAM void handleShadowRayAnyHit()
{
    shadowRayPayload.isHit = 1;
}   /* -----  end of function handleShadowRayAnyHit  ----- */
