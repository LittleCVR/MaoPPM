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
        optix::float3   intensity;

        float           pdf [N_THETA * N_PHI];
        float           cdf [N_THETA * N_PHI];

#ifdef __CUDACC__
    public:
        __device__ __forceinline__ optix::float3 flux() const
        {
            return 4.0f * M_PIf * intensity;
        }

        __device__ __forceinline__ optix::float3 estimateDirectLighting(
                const optix::float3 & point, const BSDF & bsdf,
                const optix::float3 & wo) const;

        __device__ __forceinline__ optix::float3 sampleL(
                const optix::float2 & sample,
                optix::float3 * wo, float * probability) const;
        __device__ __forceinline__ optix::float3 sampleL(
                const optix::float3 & sample,
                optix::float3 * wo, float * probability,
                unsigned int * thetaBin, unsigned int * phiBin) const;
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

__device__ __forceinline__ Light * sampleOneLightUniformly(float sample)
{
    unsigned int size = lightList.size();
    unsigned int index = min(size - 1,
            static_cast<unsigned int>(floorf(sample * size)));
    return &lightList[index];
}

__device__ __forceinline__ optix::float3 estimateAllDirectLighting(
        const optix::float3 & point, const BSDF & bsdf,
        const optix::float3 & wo)
{
    optix::float3 L = make_float3(0.0f);
    optix::uint nLights = lightList.size();
    for (optix::uint i = 0; i < nLights; ++i)
        L += lightList[i].estimateDirectLighting(point, bsdf, wo);
    return L;
}

__device__ __forceinline__ optix::float3 Light::estimateDirectLighting(
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
        return bsdf.f(wo, normalizedShadowRayDirection) * intensity *
            fabsf(optix::dot(bsdf.m_nn, normalizedShadowRayDirection)) /
            distanceSquared;
    }

}

__device__ __forceinline__ optix::float3 Light::sampleL(
        const optix::float2 & sample,
        optix::float3 * wo, float * probability) const
{
    *probability = 1.0f / (4.0f * M_PIf);
    *wo = sampleUniformSphere(sample);
    return intensity;
}
__device__ __forceinline__ optix::float3 Light::sampleL(
        const optix::float3 & sample,
        optix::float3 * wo, float * probability,
        unsigned int * thetaBin, unsigned int * phiBin) const
{
    // Sample a bin.
    unsigned int index = 0;
    for (unsigned int i = 0; i < N_THETA*N_PHI; ++i)
        if (sample.z <= cdf[i]) {
            index = i;
            break;
        }
    *thetaBin = index / N_PHI;
    *phiBin   = index % N_PHI;
    rtPrintf("thetaBin: %u, phiBin: %u\n", *thetaBin, *phiBin);

    *probability = (index == 0 ? cdf[index] : cdf[index] - cdf[index-1]) /
        area(*thetaBin, *phiBin);

    // Sample a direction in the bin.
    float zMax = cosf(static_cast<float>(*thetaBin+0) * M_PIf / N_THETA);
    float zMin = cosf(static_cast<float>(*thetaBin+1) * M_PIf / N_THETA);
    float pMax = static_cast<float>(*phiBin+1) / N_PHI;
    float pMin = static_cast<float>(*phiBin+0) / N_PHI;
    optix::float2 s = optix::make_float2(sample);
    s.x = s.x * (zMax-zMin) + zMin;
    s.y = s.y * (pMax-pMin) + pMin;
    *wo = sampleUniformSphere(s);

    return intensity;

//    if (launchIndex.x == 128 && launchIndex.y == 128) {
//        float theta = acosf(wo.z);
//        float phi   = acosf(wo.x);
//        if (wo.y < 0.0f) phi += M_PIf;
//        theta = theta * 180.0f / M_PIf;
//        phi   = phi   * 180.0f / M_PIf;
//        rtPrintf("tb: %u, pb: %u, zMin: %f, zMax: %f, pMin: %f, pMax: %f, ",
//                thetaBin, phiBin, zMin, zMax, pMin, pMax);
//        rtPrintf("s.x: %f, s.y: %f, theta: %f, phi: %f, flux: %f %f %f\n",
//                s.x, s.y, theta, phi, flux.x, flux.y, flux.z);
//    }
}

#endif  /* -----  #ifdef __CUDACC__  ----- */

}   /* -----  end of namespace MaoPPM  ----- */

#endif  /* -----  #ifndef MAOPPM_CORE_LIGHT_H  ----- */
