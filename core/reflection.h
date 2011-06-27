/*
 * =====================================================================================
 *
 *       Filename:  reflection.h
 *
 *    Description:  Some common functions for reflection models.
 *
 *        Version:  1.0
 *        Created:  2011-06-19 14:06:23
 *
 *         Author:  Chun-Wei Huang (LittleCVR), 
 *        Company:  Communication & Multimedia Laboratory,
 *                  Department of Computer Science & Information Engineering,
 *                  National Taiwan University
 *
 * =====================================================================================
 */

#ifndef IGPPM_CORE_REFLECTION_H
#define IGPPM_CORE_REFLECTION_H

/*-----------------------------------------------------------------------------
 *  header files from OptiX
 *-----------------------------------------------------------------------------*/
#include    <optix_world.h>

/*----------------------------------------------------------------------------
 *  header files of our own
 *----------------------------------------------------------------------------*/
#include    "global.h"



namespace MaoPPM {
#ifdef __CUDACC__
__device__ __inline__ float cosTheta(
        const optix::float3 & w)
{
    return w.z;
}
__device__ __inline__ bool sameHemisphere(
        const optix::float3 & wo, const optix::float3 & wi)
{
    return wo.z * wi.z > 0.0f;
}
#endif  /* -----  #ifdef __CUDACC__  ----- */
}   /* -----  end of namespace MaoPPM  ----- */

#endif  /* -----  #ifndef IGPPM_CORE_REFLECTION_H  ----- */
