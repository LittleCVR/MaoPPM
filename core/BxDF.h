/*
 * =====================================================================================
 *
 *       Filename:  BxDF.h
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

#ifndef IGPPM_CORE_BXDF_H
#define IGPPM_CORE_BXDF_H

/*-----------------------------------------------------------------------------
 *  header files from OptiX
 *-----------------------------------------------------------------------------*/
#include    <optix_world.h>

/*----------------------------------------------------------------------------
 *  header files of our own
 *----------------------------------------------------------------------------*/
#include    "global.h"
#include    "montecarlo.h"
#include    "reflection.h"
#include    "Fresnel.h"
#include    "MicrofacetDistribution.h"



namespace MaoPPM {
/*
 * =============================================================================
 *         Name:  BxDF
 *  Description:  
 * =============================================================================
 */
class BxDF {
    public:
        enum Type {
            // basic types,
            Reflection      = 1 << 31,
            Transmission    = 1 << 30,
            Diffuse         = 1 << 29,
            Glossy          = 1 << 28,
            Specular        = 1 << 27,
            AllType         = Diffuse | Glossy | Specular,
            AllReflection   = Reflection | AllType,
            AllTransmission = Transmission | AllType,
            All             = AllReflection | AllTransmission,
            // BxDF types
            Lambertian      = 1 << 0,
            Microfacet      = 1 << 1
        };  /* -----  end of enum BxDF::Type  ----- */

#ifdef __CUDACC__
    public:
        __device__ __inline__ BxDF(Type type) : m_type(type) { /* EMPTY */ }

    public:
        __device__ __inline__ Type type() const { return m_type; }

    public:
        #define BxDF_f \
        __device__ __inline__ optix::float3 f( \
                const optix::float3 & wo, const optix::float3 & wi) const \
        { \
            return optix::make_float3(0.0f); \
        }
        BxDF_f

        #define BxDF_probability \
        __device__ __inline__ float probability( \
                const optix::float3 & wo, const optix::float3 & wi) const \
        { \
            return sameHemisphere(wo, wi) ? fabsf(cosTheta(wi)) / M_PIf : 0.0f; \
        }
        BxDF_probability

        #define BxDF_sampleF \
        __device__ __inline__ optix::float3 sampleF( \
                const optix::float3 & wo, optix::float3 * wi, \
                const optix::float2 & sample, float * prob) const \
        { \
            *wi = sampleCosineWeightedHemisphere(sample); \
            *prob = probability(wo, *wi); \
            return f(wo, *wi); \
        }
        BxDF_sampleF
#endif  /* -----  #ifdef __CUDACC__  ----- */

    private:
        Type  m_type;
};  /* -----  end of class BxDF  ----- */

/*
 * =============================================================================
 *         Name:  Lambertian
 *  Description:  
 * =============================================================================
 */
class Lambertian : public BxDF {
#ifdef __CUDACC__
    public:
        __device__ __inline__ Lambertian(const optix::float3 & reflectance) :
            BxDF(BxDF::Type(BxDF::Lambertian | Reflection | Diffuse)),
            m_reflectance(reflectance) { /* EMPTY */ }

        __device__ __inline__ optix::float3 f(
                const optix::float3 & wo, const optix::float3 & wi) const
        {
            return m_reflectance / M_PIf;
        }

        BxDF_probability
        BxDF_sampleF
#endif  /* -----  #ifdef __CUDACC__  ----- */

    private:
        optix::float3  m_reflectance;
};  /* -----  end of class Lambertian  ----- */

/*
 * =============================================================================
 *         Name:  Microfacet
 *  Description:  
 * =============================================================================
 */
class Microfacet : public BxDF {
    public:
        static const unsigned int MAX_FRESNEL_SIZE                  = sizeof(FresnelDielectric);
        static const unsigned int MAX_MICROFACET_DISTRIBUTION_SIZE  = sizeof(Blinn);

#ifdef __CUDACC__
    public:
        __device__ __inline__ Microfacet(const optix::float3 & reflectance) :
            BxDF(Type(BxDF::Microfacet | Reflection | Glossy)), R(reflectance) { /* EMPTY */ }

    public:
        __device__ __inline__ MicrofacetDistribution * distribution() 
        {
            return reinterpret_cast<MicrofacetDistribution *>(m_distribution);
        }
        __device__ __inline__ const MicrofacetDistribution * distribution() const
        {
            return reinterpret_cast<const MicrofacetDistribution *>(m_distribution);
        }
        __device__ __inline__ Fresnel * fresnel() 
        {
            return reinterpret_cast<Fresnel *>(m_fresnel);
        }
        __device__ __inline__ const Fresnel * fresnel() const
        {
            return reinterpret_cast<const Fresnel *>(m_fresnel);
        }

    public:
        __device__ __inline__ optix::float3 f(const optix::float3 & wo, const optix::float3 & wi) const
        {
            float cosThetaO = fabsf(cosTheta(wo));
            float cosThetaI = fabsf(cosTheta(wi));
            if (cosThetaI == 0.f || cosThetaO == 0.f) return optix::make_float3(0.f);
            optix::float3 wh = wi + wo;
            if (wh.x == 0. && wh.y == 0. && wh.z == 0.) return optix::make_float3(0.f);
            wh = optix::normalize(wh);
            float cosThetaH = optix::dot(wi, wh);

            // Fresnel.
            optix::float3 F;
            if (fresnel()->type() & Fresnel::NoOp)
                F = reinterpret_cast<const FresnelNoOp *>(fresnel())->evaluate(cosThetaH);
            else if (fresnel()->type() & Fresnel::Dielectric)
                F = reinterpret_cast<const FresnelDielectric *>(fresnel())->evaluate(cosThetaH);

            // Distribution.
            float D;
            if (distribution()->type() & MicrofacetDistribution::Blinn)
                D = reinterpret_cast<const Blinn *>(distribution())->D(wh);

            return R * D * G(wo, wi, wh) * F / (4.f * cosThetaI * cosThetaO);
        }
        __device__ __inline__ float G(const optix::float3 &wo, const optix::float3 &wi, const optix::float3 &wh) const {
            float NdotWh = fabsf(cosTheta(wh));
            float NdotWo = fabsf(cosTheta(wo));
            float NdotWi = fabsf(cosTheta(wi));
            float WOdotWh = fabs(optix::dot(wo, wh));
            return min(1.f, min((2.f * NdotWh * NdotWo / WOdotWh),
                                (2.f * NdotWh * NdotWi / WOdotWh)));
        }
        __device__ __inline__ optix::float3 sampleF(const optix::float3 & wo, optix::float3 * wi,
                const optix::float2 & sample, float * prob) const
        {
            // Distribution.
            if (distribution()->type() & MicrofacetDistribution::Blinn)
                reinterpret_cast<const Blinn *>(distribution())->sampleF(wo, wi, sample, prob);
            if (!sameHemisphere(wo, *wi)) return optix::make_float3(0.f);
            return f(wo, *wi);
        }
        __device__ __inline__ float probability(const optix::float3 & wo, const optix::float3 & wi) const
        {
            if (!sameHemisphere(wo, wi)) return 0.f;
            // Distribution.
            float prob;
            if (distribution()->type() & MicrofacetDistribution::Blinn)
                prob = reinterpret_cast<const Blinn *>(distribution())->probability(wo, wi);
            return prob;
        }
#endif  /* -----  #ifdef __CUDACC__  ----- */

    private:
        optix::float3  R;
        char           m_fresnel[MAX_FRESNEL_SIZE];
        char           m_distribution[MAX_MICROFACET_DISTRIBUTION_SIZE];
};

static const unsigned int  MAX_BXDF_SIZE  = sizeof(Microfacet);

#define CALL_BXDF_CONST_VIRTUAL_FUNCTION(lvalue, op, bxdf, function, ...) \
    if (bxdf->type() & BxDF::Lambertian) \
        lvalue op reinterpret_cast<const Lambertian *>(bxdf)->function(__VA_ARGS__); \
    else if (bxdf->type() & BxDF::Microfacet) \
        lvalue op reinterpret_cast<const Microfacet *>(bxdf)->function(__VA_ARGS__);

}   /* -----  end of namespace MaoPPM  ----- */

#endif  /* -----  #ifndef IGPPM_CORE_BXDF_H  ----- */
