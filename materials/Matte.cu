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
rtDeclareVariable(DifferentialGeometry, geometricDG, attribute differential_geometry, ); 

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
    normalRayPayload.isHit = true;

    Intersection & intersection = normalRayPayload.intersection;
    DifferentialGeometry * dg = intersection.dg();

    *dg = geometricDG;
//    if (launchIndex.x == 449 && launchIndex.y == 252) {
//        rtPrintf("before\n");
//        rtPrintf("point "); dump(dg->point); rtPrintf("\n");
//        rtPrintf("normal "); dump(dg->normal); rtPrintf("\n");
//    }
    dg->point  = rtTransformPoint(RT_OBJECT_TO_WORLD, dg->point);
    dg->normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, dg->normal));
    dg->normal = faceforward(dg->normal, -currentRay.direction, dg->normal);
//    if (launchIndex.x == 449 && launchIndex.y == 252) {
//        rtPrintf("after\n");
//        rtPrintf("point "); dump(dg->point); rtPrintf("\n");
//        rtPrintf("normal "); dump(dg->normal); rtPrintf("\n");
//    }

    const float3 & nn = dg->normal;
    const float3 & sn = normalize(dg->dpdu);
    const float3 tn = cross(nn, sn);
//    if (launchIndex.x == 449 && launchIndex.y == 252) {
//        rtPrintf("sn "); dump(sn); rtPrintf("\n");
//        rtPrintf("tn "); dump(tn); rtPrintf("\n");
//        rtPrintf("nn "); dump(nn); rtPrintf("\n");
//    }
    float * w2o = intersection.worldToObject()->getData();
    w2o[ 0] = sn.x; w2o[ 1] = sn.y; w2o[ 2] = sn.z; w2o[ 3] = 0.0f;
    w2o[ 4] = tn.x; w2o[ 5] = tn.y; w2o[ 6] = tn.z; w2o[ 7] = 0.0f;
    w2o[ 8] = nn.x; w2o[ 9] = nn.y; w2o[10] = nn.z; w2o[11] = 0.0f;
    w2o[12] = 0.0f; w2o[13] = 0.0f; w2o[14] = 0.0f; w2o[15] = 1.0f;

    // BSDF
    Matte & material = GET_MATERIAL(Matte, materialIndex);
    BSDF * bsdf = intersection.bsdf();
    *bsdf = BSDF(*dg, geometricDG.normal);

    // BxDFs
    /*TODO*/
    bsdf->m_nBxDFs = 1;
    Lambertian * bxdf = reinterpret_cast<Lambertian *>(bsdf->bxdfList());
    *bxdf = Lambertian(material.m_kd);
}   /* -----  end of function handleNormalRayClosestHit  ----- */
