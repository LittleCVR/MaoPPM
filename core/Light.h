/*
 * =============================================================================
 *
 *       Filename:  Light.h
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  2011-06-20 17:54:38
 *
 *         Author:  Chun-Wei Huang (LittleCVR), 
 *        Company:  Communication & Multimedia Laboratory,
 *                  Department of Computer Science & Information Engineering,
 *                  National Taiwan University
 *
 * =============================================================================
 */

#ifndef MAOPPM_CORE_LIGHT_H
#define MAOPPM_CORE_LIGHT_H

/* #####   PROTOTYPE           ############################################## */

/*----------------------------------------------------------------------------
 *  header files from OptiX
 *----------------------------------------------------------------------------*/
#include    <optix_world.h>

/*----------------------------------------------------------------------------
 *  header files of our own
 *----------------------------------------------------------------------------*/
#include    "global.h"
#include    "transport.h"
#include    "utility.h"
#include    "Intersection.h"



namespace MaoPPM {

#define N_THETA  8
#define N_PHI    16

/*
 * =============================================================================
 *        Class:  Light
 *  Description:  
 * =============================================================================
 */
class Light {
    public:
        Light() { }
        ~Light() { }

        optix::float3   position;
        optix::float3   flux;

        float           pdf [N_THETA * N_PHI];
        float           cdf [N_THETA * N_PHI];

#ifdef __CUDACC__
    public:
        __device__ optix::float3 estimateDirectLighting(
                const optix::float3 & point, const BSDF & bsdf,
                const optix::float3 & wo) const;
#endif  /* -----  #ifdef __CUDACC__  ----- */

    public:
        __host__ __device__ __inline__ float area(
                unsigned int thetaIndex, unsigned int phiIndex) const
        {
            float thetaMin = static_cast<float>(thetaIndex+0) * M_PIf / N_THETA;
            float thetaMax = static_cast<float>(thetaIndex+1) * M_PIf / N_THETA;
            return (2.0f * M_PIf / static_cast<float>(N_PHI)) *
                (cosf(thetaMin) - cosf(thetaMax));
        }

        __host__ __device__ __inline__ float normalizedArea(
                unsigned int thetaIndex, unsigned int phiIndex) const
        {
            return area(thetaIndex, phiIndex) / (4.0f * M_PIf);
        }
};  /* -----  end of class Light  ----- */

}   /* -----  end of namespace MaoPPM  ----- */



/* #####   IMPLEMETATION       ############################################## */

#ifdef __CUDACC__
rtBuffer<MaoPPM::Light, 1>  lightList;
#endif  /* -----  #ifdef __CUDACC__  ----- */



namespace MaoPPM {

#ifdef __CUDACC__

__device__ optix::float3 estimateAllDirectLighting(
        const optix::float3 & point, const BSDF & bsdf,
        const optix::float3 & wo)
{
    optix::float3 L = make_float3(0.0f);
    optix::uint nLights = lightList.size();
    for (optix::uint i = 0; i < nLights; ++i)
        L += lightList[i].estimateDirectLighting(point, bsdf, wo);
    return L;
}

__device__ optix::float3 Light::estimateDirectLighting(
        const optix::float3 & point, const BSDF & bsdf,
        const optix::float3 & wo) const
{
    float distance, distanceSquared;
    optix::float3 shadowRayDirection, normalizedShadowRayDirection;

    if (!isVisible(point, position,
                &shadowRayDirection, &normalizedShadowRayDirection,
                &distance, &distanceSquared))
    {
        return optix::make_float3(0.0f);
    }
    else
    {
        return bsdf.f(wo, normalizedShadowRayDirection) * flux *
            fabsf(optix::dot(bsdf.m_nn, normalizedShadowRayDirection)) /
            distanceSquared;
    }

}

#endif  /* -----  #ifdef __CUDACC__  ----- */

}   /* -----  end of namespace MaoPPM  ----- */

#endif  /* -----  #ifndef MAOPPM_CORE_LIGHT_H  ----- */
