/*
 * =====================================================================================
 *
 *       Filename:  payload.h
 *
 *    Description:  Ray payload declarations.
 *
 *        Version:  1.0
 *        Created:  2011-06-21 08:16:17
 *
 *         Author:  Chun-Wei Huang (LittleCVR), 
 *        Company:  Communication & Multimedia Laboratory,
 *                  Department of Computer Science & Information Engineering,
 *                  National Taiwan University
 *
 * =====================================================================================
 */

#ifndef IGPPM_CORE_PAYLOAD_H
#define IGPPM_CORE_PAYLOAD_H

/*----------------------------------------------------------------------------
 *  header files of our own
 *----------------------------------------------------------------------------*/
#include    "reflection.h"
#include    "Intersection.h"



namespace MaoPPM {

static const unsigned int  N_RAY_TYPES  = 2;

enum RayType {
    NormalRay,  // Should be bind to closest hit program.
                // And the material should return a proper BSDF for this ray.
    ShadowRay   // Should be bind to any hit program.
                // And the material does not have to return a proper BSDF.
};

typedef struct NormalRayPayload {
    unsigned int  isHit;
    Intersection  intersection;
} NormalRayPayload ;

typedef struct ShadowRayPayload {
    unsigned int  isHit;
} ShadowRayPayload ;

}   /* ----------  end of namespace MaoPPM  ---------- */

#endif  /* ----- #ifndef IGPPM_CORE_PAYLOAD_H  ----- */
