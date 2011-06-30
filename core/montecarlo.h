/*
 * =====================================================================================
 *
 *       Filename:  montecarlo.h
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  2011-05-23 15:13:38
 *
 *         Author:  Chun-Wei Huang (LittleCVR), 
 *        Company:  Communication & Multimedia Laboratory,
 *                  Department of Computer Science & Information Engineering,
 *                  National Taiwan University
 *
 * =====================================================================================
 */

#ifndef IGPPM_CORE_SAMPLER_H
#define IGPPM_CORE_SAMPLER_H

/*-----------------------------------------------------------------------------
 *  Header files from OptiX
 *-----------------------------------------------------------------------------*/
#include    <optix_world.h>

/*----------------------------------------------------------------------------
 *  header files of our own
 *----------------------------------------------------------------------------*/
#include    "global.h"



namespace MaoPPM {

/*----------------------------------------------------------------------------
 *  prototypes
 *----------------------------------------------------------------------------*/
__device__ __inline__ void sampleConcentricDisk(float u1, float u2, float *dx, float *dy);
__device__ __inline__ optix::float3 sampleCosineWeightedHemisphere(const optix::float2 & sample);
__device__ __inline__ optix::float3 sampleUniformHemisphere(const optix::float2 & sample);
__device__ __inline__ optix::float3 sampleUniformSphere(const optix::float2 & sample);

__device__ __inline__ void sampleConcentricDisk(float u1, float u2, float *dx, float *dy) {
    float r, theta;
    // Map uniform random numbers to $[-1,1]^2$
    float sx = 2 * u1 - 1;
    float sy = 2 * u2 - 1;

    // Map square to $(r,\theta)$

    // Handle degeneracy at the origin
    if (sx == 0.0 && sy == 0.0) {
        *dx = 0.0;
        *dy = 0.0;
        return;
    }
    if (sx >= -sy) {
        if (sx > sy) {
            // Handle first region of disk
            r = sx;
            if (sy > 0.0) theta = sy/r;
            else          theta = 8.0f + sy/r;
        }
        else {
            // Handle second region of disk
            r = sy;
            theta = 2.0f - sx/r;
        }
    }
    else {
        if (sx <= sy) {
            // Handle third region of disk
            r = -sx;
            theta = 4.0f - sy/r;
        }
        else {
            // Handle fourth region of disk
            r = -sy;
            theta = 6.0f + sx/r;
        }
    }
    theta *= M_PI / 4.f;
    *dx = r * cosf(theta);
    *dy = r * sinf(theta);
}

/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  sampleCosineWeightedHemisphere
 *  Description:  Sample a direction on an unit hemisphere by cosine weighted function,
 *                using the input sample.
 * =====================================================================================
 */
__device__ __inline__ optix::float3 sampleCosineWeightedHemisphere(const optix::float2 & sample)
{
    optix::float3 ret;
    sampleConcentricDisk(sample.x, sample.y, &ret.x, &ret.y);
    ret.z = sqrtf(fmaxf(0.f, 1.f - ret.x*ret.x - ret.y*ret.y));
    return ret;
//    optix::float2 s = sample;
//    s.x = asinf(s.x * 2.0f - 1.0f) / (M_PIf / 2.0f) / 2.0f + 0.5f;
//    return sampleUniformHemisphere(s);
}   /* -----  end of function sampleCosineWeightedHemisphere  ----- */

/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  sampleUniformHemisphere
 *  Description:  Sample a direction on an unit hemisphere uniformly using the input
 *                sample.
 * =====================================================================================
 */
__device__ __inline__ optix::float3 sampleUniformHemisphere(const optix::float2 & sample)
{
    float z = sample.x;
    float r = sqrtf(optix::fmaxf(0.0f, 1.0f - z*z));
    float phi = 2.0f * M_PIf * sample.y;
    float x = r * cosf(phi);
    float y = r * sinf(phi);
    return optix::make_float3(x, y, z);
}   /* -----  end of function sampleUniformHemisphere  ----- */

/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  sampleUniformSphere
 *  Description:  Sample a direction on an unit sphere uniformly using the input sample.
 * =====================================================================================
 */
__device__ __inline__ optix::float3 sampleUniformSphere(const optix::float2 & sample)
{
    float z = 1.0f - 2.0f * sample.x;
    float r = sqrtf(1.0f - z*z);
    float phi = 2.0f * M_PIf * sample.y;
    float x = r * cosf(phi);
    float y = r * sinf(phi);
    return optix::make_float3(x, y, z);
}   /* -----  end of function sampleUniformSphere  ----- */

}   /* -----  end of namespace MaoPPM  ----- */

#endif  /* -----  #ifndef IGPPM_CORE_SAMPLER_H  ----- */
