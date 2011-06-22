/*
 * =============================================================================
 *
 *       Filename:  ray.cu
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  2011-06-22 12:22:38
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
#include    "payload.h"

/*----------------------------------------------------------------------------
 *  namespace
 *----------------------------------------------------------------------------*/
using namespace optix;
using namespace MaoPPM;



rtDeclareVariable(NormalRayPayload, normalRayPayload, rtPayload, );
rtDeclareVariable(ShadowRayPayload, shadowRayPayload, rtPayload, );



/* 
 * ===  FUNCTION  ==============================================================
 *         Name:  handleNormalRayMiss
 *  Description:  
 * =============================================================================
 */
RT_PROGRAM void handleNormalRayMiss()
{
    normalRayPayload.isHit = false;
}   /* -----  end of function handleNormalRayMiss  ----- */
