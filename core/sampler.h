/*
 * =====================================================================================
 *
 *       Filename:  samplers.cu
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

#ifndef MAOPPM_SAMPLERS_H
#define MAOPPM_SAMPLERS_H

/*-----------------------------------------------------------------------------
 *  Header files from OptiX
 *-----------------------------------------------------------------------------*/
#include    <optix_world.h>

/*----------------------------------------------------------------------------
 *  header files of our own
 *----------------------------------------------------------------------------*/
#include    "global.h"

/*-----------------------------------------------------------------------------
 *  namespace
 *-----------------------------------------------------------------------------*/
using namespace optix;
using namespace MaoPPM;



/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  sampleSphereUniformly
 *  Description:  Sample a direction on an unit sphere uniformly using the input sample.
 * =====================================================================================
 */
__device__ float3 sampleSphereUniformly(const float2 & sample)
{
    float z = 1.0f - 2.0f * sample.x;
    float r = sqrtf(1.0f - z*z);
    float phi = 2.0f * M_PIf * sample.y;
    float x = r * cosf(phi);
    float y = r * sinf(phi);
    return make_float3(x, y, z);
}   /* -----  end of function sampleSphereUniformly  ----- */



/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  sampleHemisphereUniformly
 *  Description:  Sample a direction on an unit hemisphere uniformly using the input
 *                sample.
 * =====================================================================================
 */
__device__ float3 sampleHemisphereUniformly(const float2 & sample)
{
    float phi = 2.0f * M_PIf*sample.x;
    float r = sqrtf(sample.y);
    float x = r * cosf(phi);
    float y = r * sinf(phi);
    float z = 1.0f - x*x -y*y;
    z = z > 0.0f ? sqrtf(z) : 0.0f;
    return make_float3(x, y, z);
}   /* -----  end of function sampleHemisphereUniformly  ----- */

#endif  /* -----  #ifndef MAOPPM_SAMPLERS_H  ----- */
