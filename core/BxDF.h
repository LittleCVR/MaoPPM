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

#define CALL_FRESNEL_CONST_VIRTUAL_FUNCTION(lvalue, op, fresnel, function, ...) \
    if (fresnel->type() & Fresnel::NoOp) \
        lvalue op reinterpret_cast<const FresnelNoOp *>(fresnel)->function(__VA_ARGS__); \
    else if (fresnel->type() & Fresnel::Dielectric) \
        lvalue op reinterpret_cast<const FresnelDielectric *>(fresnel)->function(__VA_ARGS__);

#define CALL_MICROFACET_DISTRIBUTION_CONST_VIRTUAL_FUNCTION(lvalue, op, distribution, function, ...) \
    if (distribution->type() & MicrofacetDistribution::Blinn) \
        lvalue op reinterpret_cast<const Blinn *>(distribution)->function(__VA_ARGS__);

/*
 * =============================================================================
 *         Name:  BxDF
 *  Description:  
 * =============================================================================
 */
class BxDF {
    public:
        enum Type {
            Null            = 0,
            // basic types,
            Reflection      = 1 << 0,
            Transmission    = 1 << 1,
            Diffuse         = 1 << 2,
            Glossy          = 1 << 3,
            Specular        = 1 << 4,
            AllType         = Diffuse | Glossy | Specular,
            AllReflection   = Reflection | AllType,
            AllTransmission = Transmission | AllType,
            All             = AllReflection | AllTransmission,
            // BxDF types
            Lambertian            = 1 << 5,
            SpecularReflection    = 1 << 6,
            SpecularTransmission  = 1 << 7,
            Microfacet            = 1 << 8
        };  /* -----  end of enum BxDF::Type  ----- */

#ifdef __CUDACC__
    public:
        __device__ __forceinline__ BxDF(Type type) : m_type(type) { /* EMPTY */ }

    public:
        __device__ __forceinline__ Type type() const { return m_type; }

        // Because we compress the basic types and BxDF types in a single
        // $m_type variable, it is necessary to AND All first.
        __device__ __forceinline__ bool matchFlags(Type type) const
        {
            return (m_type & All & type) == (m_type & All);
        }

    public:
        #define BxDF_f \
        __device__ __forceinline__ optix::float3 f( \
                const optix::float3 & wo, const optix::float3 & wi) const \
        { \
            return optix::make_float3(0.0f); \
        }
        BxDF_f

        #define BxDF_probability \
        __device__ __forceinline__ float probability( \
                const optix::float3 & wo, const optix::float3 & wi) const \
        { \
            return sameHemisphere(wo, wi) ? fabsf(cosTheta(wi)) / M_PIf : 0.0f; \
        }
        BxDF_probability

        #define BxDF_sampleF \
        __device__ __forceinline__ optix::float3 sampleF( \
                const optix::float3 & wo, optix::float3 * wi, \
                const optix::float2 & sample, float * prob) const \
        { \
            *wi = sampleCosineWeightedHemisphere(sample); \
            if (wo.z < 0.0f) wi->z *= -1.0f; \
            *prob = probability(wo, *wi); \
            return f(wo, *wi); \
        }
        BxDF_sampleF

        #define BxDF_rho \
        __device__ __forceinline__ optix::float3 rho(unsigned int nSamples, \
                const float * samples1, const float * samples2) const \
        { \
            optix::float3 r = optix::make_float3(0.0f); \
            for (unsigned int i = 0; i < nSamples; ++i) { \
                optix::float3 wo, wi; \
                wo = sampleUniformSphere(optix::make_float2(samples1[2*i], samples1[2*i+1])); \
                float pdf_o = (2.0f * M_PIf), pdf_i = 0.f; \
                optix::float3 f = sampleF(wo, &wi, \
                    optix::make_float2(samples2[2*i], samples2[2*i+1]), &pdf_i); \
                if (pdf_i > 0.0f) \
                    r += f * fabsf(cosTheta(wi)) * fabsf(cosTheta(wo)) / (pdf_o * pdf_i); \
            } \
            return r / (M_PIf*nSamples); \
        }
        BxDF_rho
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
        __device__ __forceinline__ Lambertian(const optix::float3 & reflectance) :
            BxDF(BxDF::Type(BxDF::Lambertian | BxDF::Reflection | BxDF::Diffuse)),
            m_reflectance(reflectance) { /* EMPTY */ }

        BxDF_probability
        BxDF_sampleF

        __device__ __forceinline__ optix::float3 f(
                const optix::float3 & wo, const optix::float3 & wi) const
        {
            return m_reflectance / M_PIf;
        }

        __device__ __forceinline__ optix::float3 rho(unsigned int nSamples,
                const float * samples1, const float * samples2) const
        {
            return m_reflectance;
        }
#endif  /* -----  #ifdef __CUDACC__  ----- */

    private:
        optix::float3  m_reflectance;
};  /* -----  end of class Lambertian  ----- */

class SpecularReflection : public BxDF {
#ifdef __CUDACC__
    public:
        __device__ __forceinline__ SpecularReflection(const optix::float3 & reflectance) :
            BxDF(BxDF::Type(BxDF::SpecularReflection | BxDF::Reflection | BxDF::Specular)),
              m_reflectance(reflectance) { /* EMPTY */ }

    public:
        __device__ __forceinline__ Fresnel * fresnel() 
        {
            return reinterpret_cast<Fresnel *>(m_fresnel);
        }
        __device__ __forceinline__ const Fresnel * fresnel() const
        {
            return reinterpret_cast<const Fresnel *>(m_fresnel);
        }

    public:
        __device__ __forceinline__ optix::float3 f(
                const optix::float3 & /* wo */, const optix::float3 & /* wi */) const
        {
            return optix::make_float3(0.0f);
        }

        __device__ __forceinline__ float probability(const optix::float3 & wo, const optix::float3 & wi) const
        {
            return 0.0f;
        }

        __device__ __forceinline__ optix::float3 sampleF(
                const optix::float3 & wo, optix::float3 * wi,
                const optix::float2 & sample, float * prob) const
        {
            *wi = optix::make_float3(-wo.x, -wo.y, wo.z);
            *prob = 1.0f;
            optix::float3 F;
            CALL_FRESNEL_CONST_VIRTUAL_FUNCTION(F, =, fresnel(), evaluate, cosTheta(wo));
            F = F * m_reflectance / fabsf(cosTheta(*wi));
            return F;
        }
#endif  /* -----  #ifdef __CUDACC__  ----- */

    private:
        optix::float3  m_reflectance;
        char           m_fresnel[MAX_FRESNEL_SIZE];
};

class SpecularTransmission : public BxDF {
#ifdef __CUDACC__
    public:
        __device__ __forceinline__ SpecularTransmission(const optix::float3 & transmittance, float ei, float et) :
            BxDF(BxDF::Type(BxDF::SpecularTransmission | BxDF::Transmission | BxDF::Specular)),
            m_transmittance(transmittance), m_fresnel(ei, et) { /* EMPTY */ }

    public:
        __device__ __forceinline__ FresnelDielectric * fresnel() { return &m_fresnel; }
        __device__ __forceinline__ const FresnelDielectric * fresnel() const { return &m_fresnel; }

    public:
        __device__ __forceinline__ optix::float3 f(
                const optix::float3 & /* wo */, const optix::float3 & /* wi */) const
        {
            return optix::make_float3(0.0f);
        }

        __device__ __forceinline__ float probability(const optix::float3 & wo, const optix::float3 & wi) const
        {
            return 0.0f;
        }

        __device__ __forceinline__ optix::float3 sampleF(
                const optix::float3 & wo, optix::float3 * wi,
                const optix::float2 & sample, float * prob) const
        {
            // Figure out which $\eta$ is incident and which is transmitted
            bool entering = cosTheta(wo) > 0.0f;
            float ei = fresnel()->eta_i, et = fresnel()->eta_t;
            if (!entering) swap(ei, et);

            // Compute transmitted ray direction
            float sini2 = sinThetaSquared(wo);
            float eta = ei / et;
            float sint2 = eta * eta * sini2;

            // Handle total internal reflection for transmission
            if (sint2 >= 1.f) return optix::make_float3(0.f);
            float cost = sqrtf(max(0.f, 1.f - sint2));
            if (entering) cost = -cost;
            float sintOverSini = eta;
            *wi = optix::make_float3(sintOverSini * -wo.x, sintOverSini * -wo.y, cost);
            *prob = 1.f;
            optix::float3 F = fresnel()->evaluate(cosTheta(wo));
            return (optix::make_float3(1.f) - F) * m_transmittance / fabsf(cosTheta(*wi));
        }
#endif  /* -----  #ifdef __CUDACC__  ----- */

    private:
        optix::float3      m_transmittance;
        FresnelDielectric  m_fresnel;
};

/*
 * =============================================================================
 *         Name:  Microfacet
 *  Description:  
 * =============================================================================
 */
class Microfacet : public BxDF {
    public:
        static const unsigned int MAX_MICROFACET_DISTRIBUTION_SIZE  = sizeof(Blinn);

#ifdef __CUDACC__
    public:
        __device__ __forceinline__ Microfacet(const optix::float3 & reflectance) :
            BxDF(Type(BxDF::Microfacet | Reflection | Glossy)), R(reflectance) { /* EMPTY */ }

    public:
        __device__ __forceinline__ MicrofacetDistribution * distribution() 
        {
            return reinterpret_cast<MicrofacetDistribution *>(m_distribution);
        }
        __device__ __forceinline__ const MicrofacetDistribution * distribution() const
        {
            return reinterpret_cast<const MicrofacetDistribution *>(m_distribution);
        }
        __device__ __forceinline__ Fresnel * fresnel() 
        {
            return reinterpret_cast<Fresnel *>(m_fresnel);
        }
        __device__ __forceinline__ const Fresnel * fresnel() const
        {
            return reinterpret_cast<const Fresnel *>(m_fresnel);
        }

    public:
        __device__ __forceinline__ optix::float3 f(const optix::float3 & wo, const optix::float3 & wi) const
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
            CALL_FRESNEL_CONST_VIRTUAL_FUNCTION(F, =, fresnel(), evaluate, cosThetaH);

            // Distribution.
            float D;
            CALL_MICROFACET_DISTRIBUTION_CONST_VIRTUAL_FUNCTION(D, =, distribution(), D, wh);

            return R * D * G(wo, wi, wh) * F / (4.f * cosThetaI * cosThetaO);
        }
        __device__ __forceinline__ float G(const optix::float3 &wo, const optix::float3 &wi, const optix::float3 &wh) const {
            float NdotWh = fabsf(cosTheta(wh));
            float NdotWo = fabsf(cosTheta(wo));
            float NdotWi = fabsf(cosTheta(wi));
            float WOdotWh = fabs(optix::dot(wo, wh));
            return min(1.f, min((2.f * NdotWh * NdotWo / WOdotWh),
                                (2.f * NdotWh * NdotWi / WOdotWh)));
        }
        __device__ __forceinline__ optix::float3 sampleF(const optix::float3 & wo, optix::float3 * wi,
                const optix::float2 & sample, float * prob) const
        {
            // Distribution.
            CALL_MICROFACET_DISTRIBUTION_CONST_VIRTUAL_FUNCTION(, , distribution(), sampleF, wo, wi, sample, prob);
            if (!sameHemisphere(wo, *wi)) return optix::make_float3(0.f);
            return f(wo, *wi);
        }
        __device__ __forceinline__ float probability(const optix::float3 & wo, const optix::float3 & wi) const
        {
            if (!sameHemisphere(wo, wi)) return 0.f;
            // Distribution.
            float prob;
            CALL_MICROFACET_DISTRIBUTION_CONST_VIRTUAL_FUNCTION(prob, =, distribution(), probability, wo, wi);
            return prob;
        }
#endif  /* -----  #ifdef __CUDACC__  ----- */

    private:
        optix::float3  R;
        char           m_fresnel[MAX_FRESNEL_SIZE];
        char           m_distribution[MAX_MICROFACET_DISTRIBUTION_SIZE];
};

static const unsigned int  MAX_BXDF_SIZE  = sizeof(Microfacet);

}   /* -----  end of namespace MaoPPM  ----- */

#endif  /* -----  #ifndef IGPPM_CORE_BXDF_H  ----- */
