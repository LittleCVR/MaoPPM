/*
 * =============================================================================
 *
 *       Filename:  Glass.cu
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  2011-06-28 22:13:53
 *
 *         Author:  Chun-Wei Huang (LittleCVR), 
 *        Company:  Communication & Multimedia Laboratory,
 *                  Department of Computer Science & Information Engineering,
 *                  National Taiwan University
 *
 * =============================================================================
 */

#include    "Glass.h"

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
    // Intersection
    Intersection * intersection = normalRayPayload.intersection();
    // Differential geometry.
    *(intersection->dg()) = geometricDG;
    // Material
    intersection->setMaterialPointer(GET_MATERIAL_POINTER(materialIndex));
}   /* -----  end of function handleNormalRayClosestHit  ----- */
