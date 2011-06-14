/*
 * =====================================================================================
 *
 *       Filename:  math.cu
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  2011-05-23 15:36:38
 *
 *         Author:  Chun-Wei Huang (LittleCVR), 
 *        Company:  Communication & Multimedia Laboratory,
 *                  Department of Computer Science & Information Engineering,
 *                  National Taiwan University
 *
 * =====================================================================================
 */

#ifndef UTILITY_H
#define UTILITY_H

/*-----------------------------------------------------------------------------
 *  Header files from OptiX
 *-----------------------------------------------------------------------------*/
#include    <optix_world.h>

/*-----------------------------------------------------------------------------
 *  namespace
 *-----------------------------------------------------------------------------*/
using namespace optix;



/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  pairwiseMul
 *  Description:  Multiplies two vectors pairwisely,
 *                e.g. result = (v1.x*v2.x, v1.y*v2.y, v1.z*v2.z)
 * =====================================================================================
 */
__device__ __inline__ float2 pairwiseMul(const float2 & v1, const float2 & v2)
{
    return make_float2(v1.x*v2.x, v1.y*v2.y);
}   /* -----  end of function pairwiseMul  ----- */

__device__ __inline__ float3 pairwiseMul(const float3 & v1, const float3 & v2)
{
    return make_float3(v1.x*v2.x, v1.y*v2.y, v1.z*v2.z);
}   /* -----  end of function pairwiseMul  ----- */

__device__ __inline__ float4 pairwiseMul(const float4 & v1, const float4 & v2)
{
    return make_float4(v1.x*v2.x, v1.y*v2.y, v1.z*v2.z, v1.w*v2.w);
}   /* -----  end of function pairwiseMul  ----- */

#endif  /* -----  #ifndef UTILITY_H  ----- */
