/*
 * =====================================================================================
 *
 *       Filename:  reflection.h
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

#ifndef IGPPM_CORE_REFLECTION_H
#define IGPPM_CORE_REFLECTION_H

/*-----------------------------------------------------------------------------
 *  header files from OptiX
 *-----------------------------------------------------------------------------*/
#include    <optix_world.h>

/*----------------------------------------------------------------------------
 *  header files of our own
 *----------------------------------------------------------------------------*/
#include    "global.h"
#include    "montecarlo.h"
#include    "utility.h"
#include    "DifferentialGeometry.h"



namespace MaoPPM {
#ifdef __CUDACC__
__device__ __inline__ float cosTheta(
        const optix::float3 & w)
{
    return w.z;
}
__device__ __inline__ bool sameHemisphere(
        const optix::float3 & wo, const optix::float3 & wi)
{
    return wo.z * wi.z > 0.0f;
}
#endif  /* -----  #ifdef __CUDACC__  ----- */



/* TODO: figure out WTF is Fresnel */
class Fresnel {
    public:
        enum Type {
            NoOp        = 1 << 0,
            Conductor   = 1 << 1,
            Dielectric  = 1 << 2
        };

#ifdef __CUDACC__
    public:
        __device__ __inline__ Fresnel(Type type) : m_type(type) { /* EMPTY */ }

    public:
        __device__ __inline__ Type type() const { return m_type; }

    public:
        // virtual optix::float3 evaluate(float cosi) const =0;
#endif  /* -----  #ifdef __CUDACC__  ----- */

    private:
        Type  m_type;
};
class FresnelNoOp : public Fresnel {
#ifdef __CUDACC__
    public:
        __device__ __inline__ FresnelNoOp() : Fresnel(NoOp) { /* EMPTY */ }

    public:
        __device__ __inline__ optix::float3 evaluate(float) const
        {
            return optix::make_float3(1.0f);
        }
#endif  /* -----  #ifdef __CUDACC__  ----- */
};
//class FresnelConductor : public Fresnel {
//#ifdef __CUDACC__
//    public:
//        __device__ __inline__ FresnelConductor(
//                const optix::float3 & eta, const optix::float3 & k) : Fresnel(Fresnel::Conductor),
//            m_eta(eta), m_k(k) { /* EMPTY */ }   
//
//        __device__ __inline__ optix::float3 evaluate(float cosi) const
//        {
//            cosi = fabsf(cosi);
//            const optix::float3 & eta = m_eta;
//            const optix::float3 & k   = m_k;
//            optix::float3 tmp = (eta*eta + k*k) * cosi*cosi;
//            optix::float3 Rparl2 = (tmp - (2.f * eta * cosi) + 1.f) /
//                (tmp + (2.f * eta * cosi) + 1.f); 
//            optix::float3 tmp_f = eta*eta + k*k;
//            optix::float3 Rperp2 =
//                (tmp_f - (2.f * eta * cosi) + cosi*cosi) /
//                (tmp_f + (2.f * eta * cosi) + cosi*cosi);
//            return (Rparl2 + Rperp2) / 2.f;
//        }
//#endif  /* -----  #ifdef __CUDACC__  ----- */
//
//    private:
//        optix::float3  m_eta;
//        optix::float3  m_k;
//};
class FresnelDielectric : public Fresnel {
#ifdef __CUDACC__
    public:
        __device__ __inline__ FresnelDielectric(float ei, float et) : Fresnel(Fresnel::Dielectric),
            eta_i(ei), eta_t(et) { /* EMPTY */ }

        __device__ __inline__ optix::float3 evaluate(float cosi) const 
        {
            // Compute Fresnel reflectance for dielectric
            cosi = optix::clamp(cosi, -1.0f, 1.0f);

            // Compute indices of refraction for dielectric
            bool entering = cosi > 0.0f;
            float ei = eta_i, et = eta_t;
            if (!entering) swap(ei, et);

            // Compute _sint_ using Snell's law
            float sint = ei/et * sqrtf(max(0.0f, 1.0f - cosi*cosi));
            if (sint >= 1.0f) {
                // Handle total internal reflection
                return optix::make_float3(1.0f);
            } else {
                cosi = fabsf(cosi);
                float cost = sqrtf(max(0.0f, 1.0f - sint*sint));
                optix::float3 Rparl = optix::make_float3(
                        ((et * cosi) - (ei * cost)) /
                        ((et * cosi) + (ei * cost)));
                optix::float3 Rperp = optix::make_float3(
                        ((ei * cosi) - (et * cost)) /
                        ((ei * cosi) + (et * cost)));
                return (Rparl*Rparl + Rperp*Rperp) / 2.0f;
            }
        }
#endif  /* -----  #ifdef __CUDACC__  ----- */

    private:
        float eta_i, eta_t;
};



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

        __device__ __inline__ ~Lambertian() { /* EMPTY */ }

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

    public:
        // float D(const optix::float3 & wh) const = 0;
        // void sampleF(const optix::float3 & wo, optix::float3 * wi,
        //         const optix::float2 & sample, float * pdf) const = 0;
        // float probability(const optix::float3 & wo, const optix::float3 & wi) const = 0;
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



/*
 * =============================================================================
 *         Name:  BSDF
 *  Description:  
 * =============================================================================
 */
class BSDF {
    public:
        static const unsigned int  MAX_N_BXDFS  = 2;

#ifdef __CUDACC__
    public:
        __device__ __inline__ BSDF() { /* EMPTY */ }
        __device__ __inline__ BSDF(
                const DifferentialGeometry & dgShading,
                const optix::float3 & geometricNormal, const float eta = 1.0f)
        {
            m_gn = geometricNormal;
            m_nn = dgShading.normal;
            m_sn = optix::normalize(dgShading.dpdu);
            m_tn = optix::cross(m_nn, m_sn);
            m_nBxDFs = 0;
        }

        __device__ __inline__ unsigned int nBxDFs() const { return m_nBxDFs; }
        __device__ __inline__ BxDF * bxdfAt(const Index & index)
        {
            return reinterpret_cast<BxDF *>(&m_bxdfList[index * MAX_BXDF_SIZE]);
        }
        __device__ __inline__ const BxDF * bxdfAt(const Index & index) const
        {
            return reinterpret_cast<const BxDF *>(&m_bxdfList[index * MAX_BXDF_SIZE]);
        }

    public:
        __device__ __inline__ optix::float3 f(const optix::float3 & worldWo,
                const optix::float3 & worldWi, BxDF::Type type = BxDF::All) const
        {
            optix::float3 wo, wi;
            worldToLocal(worldWo, &wo);
            worldToLocal(worldWi, &wi);
            // Calculate f.
            optix::float3 totalF = optix::make_float3(0.0f);
            for (unsigned int i = 0; i < m_nBxDFs; i++) {
                const BxDF * bxdf = bxdfAt(i);
                // Skip unmatched BxDF.
                if (!(bxdf->type() & type)) continue;
                // Determine real BxDF type.
                if (bxdf->type() & BxDF::Lambertian)
                    totalF += reinterpret_cast<const Lambertian *>(bxdf)->f(wo, wi);
                else if (bxdf->type() & BxDF::Microfacet)
                    totalF += reinterpret_cast<const Microfacet *>(bxdf)->f(wo, wi);
            }
            return totalF;
        }

        __device__ __inline__ optix::float3 sampleF(const optix::float3 & worldWo,
                optix::float3 * worldWi, const optix::float3 & sample, float * probability,
                const BxDF::Type type = BxDF::All)
        {
            optix::float3 wo;
            worldToLocal(worldWo, &wo);
            /* TODO: count type */
            // Sample BxDF.
            unsigned int index = min(m_nBxDFs-1,
                    static_cast<unsigned int>(floorf(sample.x * static_cast<float>(m_nBxDFs))));
            const BxDF * bxdf = bxdfAt(index);
            *probability = 1.0f / static_cast<float>(m_nBxDFs);
            // Sample f.
            float prob;
            optix::float3 f;
            optix::float3 wi;
            optix::float2 s = optix::make_float2(sample.y, sample.z);
            if (bxdf->type() & BxDF::Lambertian)
                f = reinterpret_cast<const Lambertian *>(bxdf)->sampleF(wo, &wi, s, &prob);
            else if (bxdf->type() & BxDF::Microfacet)
                f = reinterpret_cast<const Microfacet *>(bxdf)->sampleF(wo, &wi, s, &prob);
            *probability *= prob;
            localToWorld(wi, worldWi);
            return f;
        }

        __device__ __inline__ optix::Matrix4x4 localToWorld() const
        {
            return worldToLocal().transpose();
        }

        __device__ __inline__ optix::Matrix4x4 worldToLocal() const
        {
            return createCoordinateSystemTransform(m_sn, m_tn, m_nn);
        }

    private:
        __device__ __inline__ void localToWorld(
                const optix::float3 & localW, optix::float3 * worldW) const
        {
            *worldW = transformVector(localToWorld(), localW);
        }

        __device__ __inline__ void worldToLocal(
                const optix::float3 & worldW, optix::float3 * localW) const
        {
            *localW = transformVector(worldToLocal(), worldW);
        }
#endif  /* -----  #ifdef __CUDACC__  ----- */

    public:  // should be private
        optix::float3         m_sn;
        optix::float3         m_tn;
        optix::float3         m_nn;
        optix::float3         m_gn;
        unsigned int          m_nBxDFs;
        char                  m_bxdfList [MAX_N_BXDFS * MAX_BXDF_SIZE];
};  /* -----  end of class BSDF  ----- */

}   /* -----  end of namespace MaoPPM  ----- */

#endif  /* -----  #ifndef IGPPM_CORE_REFLECTION_H  ----- */
