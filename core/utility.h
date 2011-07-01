/*
 * =============================================================================
 *
 *       Filename:  utility.cu
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

/*---------------------------------------------------------------------------
 *  header files of our own
 *---------------------------------------------------------------------------*/
#include    "global.h"



#ifdef __CUDACC__
rtBuffer<char ,         1>  localHeap;
rtBuffer<MaoPPM::Index, 1>  localHeapPointer;
#endif  /* -----  #ifdef __CUDACC__  ----- */



namespace MaoPPM {

#ifdef __CUDACC__

template<typename T>
__device__ __inline__ void swap(T & t1, T & t2)
{
    T tmp = t1;
    t1 = t2;
    t2 = tmp;
}

#define LOCAL_HEAP_ALLOC_SIZE(size) \
    atomicAdd(&localHeapPointer[0], size)

#define LOCAL_HEAP_ALLOC_TYPE(type) \
    LOCAL_HEAP_ALLOC_SIZE(sizeof(type))

#define LOCAL_HEAP_GET_OBJECT_POINTER(type, index) \
    reinterpret_cast<type *>(&localHeap[index]);

#define LOCAL_HEAP_ALLOC_TYPE_AND_GET_OBJECT_POINTER(type) \
    LOCAL_HEAP_GET_OBJECT_POINTER(type, LOCAL_HEAP_ALLOC_TYPE(type))

#define GET_1_SAMPLE(sampleList, sampleIndex) \
    sampleList[sampleIndex]; \
    sampleIndex += 1

#define GET_2_SAMPLES(sampleList, sampleIndex) \
    make_float2(sampleList[sampleIndex], \
            sampleList[sampleIndex+1]); \
    sampleIndex += 2

#define GET_3_SAMPLES(sampleList, sampleIndex) \
    make_float3(sampleList[sampleIndex], \
            sampleList[sampleIndex+1], \
            sampleList[sampleIndex+2]); \
    sampleIndex += 3

#define LAUNCH_OFFSET_2D(launchIndex, launchSize) \
    (launchIndex.y * launchSize.x + launchIndex.x)

__device__ __inline__ bool isBlack(const optix::float3 & color)
{
    return (color.x == 0.0f && color.y == 0.0f && color.z == 0.0f);
}

__device__ __inline__ bool solveQuadraticEquation(
        float A, float B, float C, float * t0, float * t1)
{
    // Find quadratic discriminant
    float discrim = B * B - 4.f * A * C;
    if (discrim <= 0.f) return false;
    float rootDiscrim = sqrtf(discrim);

    // Compute quadratic _t_ values
    float q;
    if (B < 0) q = -.5f * (B - rootDiscrim);
    else       q = -.5f * (B + rootDiscrim);
    *t0 = q / A;
    *t1 = C / q;
    if (*t0 > *t1) swap(*t0, *t1);
    return true;
}

/* 
 * ===  FUNCTION  ==============================================================
 *         Name:  createCoordinateSystem
 *  Description:  Given $U, generates $V and $W that are perpendicular to $U,
 *                also $V and $W are perpendicular to each other, too.
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
 *         Name:  createCoordinateSystemTransform
 *  Description:  Given $s, $t, and $n, create a transform that will transform
 *                the regular coordinate system to $s, $t, and $n. That is,
 *                if you use this transform to trasform (1, 0, 0), you get $s,
 *                if you use this transform to trasform (0, 1, 0), you get $t,
 *                and if you use this transform to transform (0, 0, 1), you get
 *                $n.
 * =============================================================================
 */
__device__ __inline__ optix::Matrix4x4 createCoordinateSystemTransform(
        const optix::float3 & s, const optix::float3 & t, const optix::float3 & n)
{
    float d[16];
    d[ 0] =  s.x; d[ 1] =  s.y; d[ 2] =  s.z; d[ 3] = 0.0f;
    d[ 4] =  t.x; d[ 5] =  t.y; d[ 6] =  t.z; d[ 7] = 0.0f;
    d[ 8] =  n.x; d[ 9] =  n.y; d[10] =  n.z; d[11] = 0.0f;
    d[12] = 0.0f; d[13] = 0.0f; d[14] = 0.0f; d[15] = 1.0f;
    return optix::Matrix4x4(d);
}

/* 
 * ===  FUNCTION  ==============================================================
 *         Name:  printFloat3
 *  Description:  Helper function to dump float3.
 * =============================================================================
 */
__device__ __inline__ void printFloat3(const optix::float3 & v)
{
    rtPrintf("%+4.4f %+4.4f %+4.4f", v.x, v.y, v.z);
}

#endif  /* -----  #ifdef __CUDACC__  ----- */

/* 
 * ===  FUNCTION  ==============================================================
 *         Name:  
 *  Description:  
 * =============================================================================
 */
__host__ __device__ __inline__ optix::float3 transformPoint(
        const optix::Matrix4x4 & m, const optix::float3 & p)
{
    optix::float4 np = m * optix::make_float4(p, 1.0f);
    np.x = np.x / np.w;
    np.y = np.y / np.w;
    np.z = np.z / np.w;
    return make_float3(np);
}

__host__ __device__ __inline__ optix::float3 transformVector(
        const optix::Matrix4x4 & m, const optix::float3 & v)
{
    return optix::make_float3(m * optix::make_float4(v, 0.0f));
}

}   /* -----  end of namespace MaoPPM  ----- */

#endif  /* -----  #ifndef UTILITY_H  ----- */
