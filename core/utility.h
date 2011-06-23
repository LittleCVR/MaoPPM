/*
 * =============================================================================
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
 * =============================================================================
 */

#ifndef UTILITY_H
#define UTILITY_H

/*----------------------------------------------------------------------------
 *  Header files from OptiX
 *----------------------------------------------------------------------------*/
#include    <optix_world.h>



namespace MaoPPM {

#define GET_MATERIAL(type, index) \
    reinterpret_cast<type &>(heap[index])

#define GET_2_SAMPLES(sampleList, sampleIndex) \
    make_float2(sampleList[sampleIndex+0], sampleList[sampleIndex+1]); \
    sampleIndex += 2

#define LAUNCH_OFFSET_2D(launchIndex, launchSize) \
    (launchIndex.y * launchSize.x + launchIndex.x)

/* 
 * ===  FUNCTION  ==============================================================
 *         Name:  pairwiseMul
 *  Description:  Multiplies two vectors pairwisely,
 *                e.g. result = (v1.x*v2.x, v1.y*v2.y, v1.z*v2.z)
 * =============================================================================
 */
__device__ __inline__ optix::float2 pairwiseMul(
        const optix::float2 & v1, const optix::float2 & v2)
{
    return optix::make_float2(v1.x*v2.x, v1.y*v2.y);
}

__device__ __inline__ optix::float3 pairwiseMul(
        const optix::float3 & v1, const optix::float3 & v2)
{
    return optix::make_float3(v1.x*v2.x, v1.y*v2.y, v1.z*v2.z);
}

__device__ __inline__ optix::float4 pairwiseMul(
        const optix::float4 & v1, const optix::float4 & v2)
{
    return optix::make_float4(v1.x*v2.x, v1.y*v2.y, v1.z*v2.z, v1.w*v2.w);
}

/* 
 * ===  FUNCTION  ==============================================================
 *         Name:  
 *  Description:  
 * =============================================================================
 */
__device__ __inline__ float3 transformVector(const optix::Matrix4x4 & m, const float3 & v)
{
    return optix::make_float3(m * optix::make_float4(v, 0.0f));
}

/* 
 * ===  FUNCTION  ==============================================================
 *         Name:  
 *  Description:  
 * =============================================================================
 */
__device__ __inline__ void createCoordinateSystem(const optix::float3 & U,
        optix::float3 * V, optix::float3 * W)
{
    if (fabsf(U.x) > fabsf(U.y)) {
        float invLen = 1.f / sqrtf(U.x*U.x + U.z*U.z);
        *V = optix::make_float3(-U.z * invLen, 0.f, U.x * invLen);
    } else {
        float invLen = 1.f / sqrtf(U.y*U.y + U.z*U.z);
        *V = optix::make_float3(0.f, U.z * invLen, -U.y * invLen);
    }
    *W = optix::cross(U, *V);
}

/* 
 * ===  FUNCTION  ==============================================================
 *         Name:  
 *  Description:  
 * =============================================================================
 */
__device__ __inline__ void dump(const optix::float3 & v)
{
    rtPrintf("%+4.4f %+4.4f %+4.4f", v.x, v.y, v.z);
}

}   /* -----  end of namespace MaoPPM  ----- */

#endif  /* -----  #ifndef UTILITY_H  ----- */
