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
#include    "DifferentialGeometry.h"



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
        };  /* -----  end of enum BxDF::Type  ----- */

    public:
        __device__ BxDF(Type type) : m_type(type) { /* EMPTY */ }
        __device__ ~BxDF() { /* EMPTY */ }

    public:
        __device__ Type type() const { return m_type; }

    public:
        #define BxDF_f \
        __device__ optix::float3 f( \
                const optix::float3 & wo, const optix::float3 & wi) const \
        { \
            return optix::make_float3(0.0f); \
        }
        BxDF_f

        #define BxDF_probability \
        __device__ float probability( \
                const optix::float3 & wo, const optix::float3 & wi) const \
        { \
            return sameHemisphere(wo, wi) ? fabsf(cosTheta(wi)) : 0.0f; \
        }
        BxDF_probability

        #define BxDF_sampleF \
        __device__ optix::float3 sampleF( \
                const optix::float3 & wo, optix::float3 * wi, \
                const optix::float2 & sample, float * prob) const \
        { \
            *wi = sampleCosineWeightedHemisphere(sample); \
            if (prob != NULL) \
                *prob = probability(wo, *wi); \
            return f(wo, *wi); \
        }
        BxDF_sampleF

    protected:  // methods
        __device__ __inline__ float cosTheta(
                const optix::float3 & w) const
        {
            return w.z;
        }

        __device__ __inline__ bool sameHemisphere(
                const optix::float3 & wo, const optix::float3 & wi) const
        {
            return wo.z * wi.z > 0.0f;
        }

    private:
        Type    m_type;
};  /* -----  end of class BxDF  ----- */



/*
 * =============================================================================
 *         Name:  Lambertian
 *  Description:  
 * =============================================================================
 */
class Lambertian : public BxDF {
    public:
        __device__ Lambertian(const optix::float3 & reflectance) :
            BxDF(BxDF::Type(BxDF::Lambertian | Reflection | Diffuse)),
            m_reflectance(reflectance) { /* EMPTY */ }

        __device__ ~Lambertian() { /* EMPTY */ }

        __device__ optix::float3 f(
                const optix::float3 & wo, const optix::float3 & wi) const
        {
            return m_reflectance * (1.0f / M_PIf);
        }

        BxDF_probability
        BxDF_sampleF

    private:
        optix::float3  m_reflectance;
};  /* -----  end of class Lambertian  ----- */



/*
 * =============================================================================
 *         Name:  BSDF
 *  Description:  
 * =============================================================================
 */
class BSDF {
    public:
        static const unsigned int  MAX_N_BXDFS    = 2;
        // If added some new BxDF in the future,
        // change this to the biggest BxDF class.
        static const unsigned int  MAX_BXDF_SIZE  = sizeof(Lambertian);

    public:
        __device__ __inline__ BSDF(
                const DifferentialGeometry & dgShading,
                const optix::float3 & geometricNormal, const float eta = 1.0f)
//            : m_dgShading(dgShading), m_eta(eta)
        {
//            m_gn = geometricNormal;
//            m_nn = dgShading.normal;
            m_nBxDFs = 0;
        }

        __device__ __inline__ ~BSDF() { /* EMPTY */ }

        __device__ __inline__ void * bxdfList()
        {
            return reinterpret_cast<void *>(m_bxdfList);
        }

        __device__ __inline__ unsigned int nBxDFs() const { return m_nBxDFs; }

    public:
        __device__ __inline__ optix::float3 f(const optix::float3 & wo,
                const optix::float3 & wi, BxDF::Type type = BxDF::All) const
        {
            Index index = 0;
            optix::float3 totalF = optix::make_float3(0.0f);
            for (unsigned int i = 0; i < m_nBxDFs; i++) {
                const BxDF & bxdf = reinterpret_cast<const BxDF &>(m_bxdfList[index]);
                if (!(bxdf.type() & type))
                    continue;
                if (bxdf.type() & BxDF::Lambertian) {
                    const Lambertian & b = reinterpret_cast<const Lambertian &>(bxdf);
                    totalF += b.f(wo, wi);
                    index += sizeof(Lambertian);
                }
            }
            return totalF;
        }

//        __device__ __inline__ DifferentialGeometry * dgShading()
//        {
//            return &m_dgShading;
//        }

//    private:
//        DifferentialGeometry  m_dgShading;
//        optix::float3         m_gn, m_nn;
//        float                 m_eta;
        unsigned int          m_nBxDFs;
        char                  m_bxdfList [MAX_N_BXDFS * sizeof(Lambertian)];
};  /* -----  end of class BSDF  ----- */

}   /* -----  end of namespace MaoPPM  ----- */

#endif  /* -----  #ifndef IGPPM_CORE_REFLECTION_H  ----- */
