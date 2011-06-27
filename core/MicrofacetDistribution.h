/*
 * =====================================================================================
 *
 *       Filename:  MicrofacetDistribution.h
 *
 *    Description:  
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

#ifndef IGPPM_CORE_MICROFACET_DISTRIBUTION_H
#define IGPPM_CORE_MICROFACET_DISTRIBUTION_H

/*-----------------------------------------------------------------------------
 *  header files from OptiX
 *-----------------------------------------------------------------------------*/
#include    <optix_world.h>

/*----------------------------------------------------------------------------
 *  header files of our own
 *----------------------------------------------------------------------------*/
#include    "global.h"
#include    "reflection.h"



namespace MaoPPM {
class MicrofacetDistribution {
    public:
        enum Type {
            Blinn  = 1 << 0,
        };

#ifdef __CUDACC__
    public:
        __device__ __inline__ MicrofacetDistribution(Type type) : m_type(type) { /* EMPTY */ }

    public:
        __device__ __inline__ Type type() const { return m_type; }
#endif  /* -----  #ifdef __CUDACC__  ----- */

    private:
        Type  m_type;
};

class Blinn : public MicrofacetDistribution {
#ifdef __CUDACC__
    public:
        __device__ __inline__ Blinn(float e) :
            MicrofacetDistribution(MicrofacetDistribution::Blinn)
        {
            if (e > 10000.0f || isnan(e)) e = 10000.0f;
            exponent = e;
        }

    public:
        __device__ __inline__ float D(const optix::float3 & wh) const
        {
            float costhetah = fabsf(cosTheta(wh));
            return (exponent + 2.0f) / (2.0f * M_PIf) * powf(costhetah, exponent);
        }
        __device__ __inline__ void sampleF(const optix::float3 & wo, optix::float3 * wi,
                optix::float2 sample, float * pdf) const
        {
            // Compute sampled half-angle vector $\wh$ for Blinn distribution
            float costheta = powf(sample.x, 1.f / (exponent + 1.f));
            float sintheta = sqrtf(max(0.f, 1.f - costheta*costheta));
            float phi = sample.y * 2.f * M_PIf;
            optix::float3 wh = optix::make_float3(
                    sintheta * cosf(phi),
                    sintheta * sinf(phi),
                    costheta);
            if (!sameHemisphere(wo, wh)) wh = -wh;

            // Compute incident direction by reflecting about $\wh$
            *wi = -wo + 2.f * optix::dot(wo, wh) * wh;

            // Compute PDF for $\wi$ from Blinn distribution
            float blinn_pdf = ((exponent + 1.f) * powf(costheta, exponent)) /
                              (2.f * M_PI * 4.f * optix::dot(wo, wh));
            if (optix::dot(wo, wh) <= 0.f) blinn_pdf = 0.f;
            *pdf = blinn_pdf;
        }
        __device__ __inline__ float probability(const optix::float3 & wi, const optix::float3 & wo) const
        {
            optix::float3 wh = optix::normalize(wo + wi);
            float costheta = fabsf(cosTheta(wh));
            // Compute PDF for $\wi$ from Blinn distribution
            float blinn_pdf = ((exponent + 1.f) * powf(costheta, exponent)) /
                              (2.f * M_PI * 4.f * optix::dot(wo, wh));
            if (optix::dot(wo, wh) <= 0.f) blinn_pdf = 0.f;
            return blinn_pdf;
        }
#endif  /* -----  #ifdef __CUDACC__  ----- */

    private:
        float exponent;
};
}   /* -----  end of namespace MaoPPM  ----- */

#endif  /* -----  #ifndef IGPPM_CORE_MICROFACET_DISTRIBUTION_H  ----- */
