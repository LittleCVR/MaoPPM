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

#ifndef MAOPPM_REFLECTION_H
#define MAOPPM_REFLECTION_H

/*-----------------------------------------------------------------------------
 *  header files from OptiX
 *-----------------------------------------------------------------------------*/
#include    <optix_world.h>

/*----------------------------------------------------------------------------
 *  header files of our own
 *----------------------------------------------------------------------------*/
#include    "sampler.h"



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
            Reflection      = 1 << 0,
            Transmission    = 1 << 1,
            Diffuse         = 1 << 2,
            Glossy          = 1 << 3,
            Specular        = 1 << 4,
            AllType         = Diffuse | Glossy | Specular,
            AllReflection   = Reflection | AllType,
            AllTransmission = Transmission | AllType,
            All             = AllReflection | AllTransmission
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
            return make_float3(0.0f); \
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
            optix::float2 s = sample; \
            s.x = asinf(s.x * 2.0f - 1.0f) / M_PIf + 0.5f; \
            s.y = asinf(s.y * 2.0f - 1.0f) / M_PIf + 0.5f; \
            *wi = sampleHemisphereUniformly(s); \
            if (prob != NULL) \
                *prob = probability(wo, *wi); \
            return f(wo, *wi); \
        }
        BxDF_sampleF

    protected:  // methods
        __device__ float cosTheta(
                const optix::float3 & w) const
        {
            return w.z;
        }

        __device__ bool sameHemisphere(
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
            BxDF(BxDF::Type(Reflection | Diffuse)), m_reflectance(reflectance) { /* EMPTY */ }
        __device__ ~Lambertian() { /* EMPTY */ }

        __device__ optix::float3 f(
                const optix::float3 & wo, const optix::float3 & wi) const
        {
            return m_reflectance * M_1_PIf;
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
        __device__ BSDF() { /* EMPTY */ }
        __device__ ~BSDF() { /* EMPTY */ }

    private:
        HeapIndex  m_nBxDFs;
        HeapIndex  m_BxDFList[4];
};  /* -----  end of class BSDF  ----- */

}   /* -----  end of namespace MaoPPM  ----- */

#endif  /* -----  #ifndef MAOPPM_REFLECTION_H  ----- */
