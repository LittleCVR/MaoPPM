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
__device__ __inline__ optix::float3 sampleCosineWeightedHemisphere(const optix::float2 & sample);
__device__ __inline__ optix::float3 sampleUniformHemisphere(const optix::float2 & sample);
__device__ __inline__ optix::float3 sampleUniformSphere(const optix::float2 & sample);

/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  sampleCosineWeightedHemisphere
 *  Description:  Sample a direction on an unit hemisphere by cosine weighted function,
 *                using the input sample.
 * =====================================================================================
 */
__device__ __inline__ optix::float3 sampleCosineWeightedHemisphere(const optix::float2 & sample)
{
    optix::float2 s = sample;
    s.x = asinf(s.x) / (M_PIf / 2.0f);
    return sampleUniformHemisphere(s);
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
    float phi = 2 * M_PIf * sample.y;
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
