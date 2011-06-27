/*
 * =============================================================================
 *
 *       Filename:  Plastic.cu
 *
 *    Description:  
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

#include    "Plastic.h"

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
#include    "DifferentialGeometry.h"
#include    "Intersection.h"

/*----------------------------------------------------------------------------
 *  namespace
 *----------------------------------------------------------------------------*/
using namespace optix;
using namespace MaoPPM;



rtDeclareVariable(uint2, launchIndex, rtLaunchIndex, );
rtDeclareVariable(uint2, launchSize ,              , );

rtBuffer<float,  1>  sampleList;
rtBuffer<char,   1>  inputHeap;

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

    Index index = LOCAL_HEAP_ALLOC(Intersection);
    Intersection * intersection = LOCAL_HEAP_GET_OBJECT_POINTER(Intersection, index);
    normalRayPayload.m_intersection = intersection;

    intersection->m_material = GET_MATERIAL(Plastic, materialIndex);

    // Differential geometry.
    DifferentialGeometry * dg = intersection->dg();
    *dg = geometricDG;
    dg->point  = rtTransformPoint(RT_OBJECT_TO_WORLD, dg->point);
    dg->normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, dg->normal));
    dg->normal = faceforward(dg->normal, -currentRay.direction, dg->normal);
}   /* -----  end of function handleNormalRayClosestHit  ----- */
