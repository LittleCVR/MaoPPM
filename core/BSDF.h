/*
 * =====================================================================================
 *
 *       Filename:  BSDF.h
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

#ifndef IGPPM_CORE_BSDF_H
#define IGPPM_CORE_BSDF_H

/*-----------------------------------------------------------------------------
 *  header files from OptiX
 *-----------------------------------------------------------------------------*/
#include    <optix_world.h>

/*----------------------------------------------------------------------------
 *  header files of our own
 *----------------------------------------------------------------------------*/
#include    "global.h"
#include    "reflection.h"
#include    "utility.h"
#include    "BxDF.h"
#include    "DifferentialGeometry.h"



namespace MaoPPM {

#define CALL_BXDF_CONST_VIRTUAL_FUNCTION(lvalue, op, bxdf, function, ...) \
    if (bxdf->type() & BxDF::Lambertian) \
        lvalue op reinterpret_cast<const Lambertian *>(bxdf)->function(__VA_ARGS__); \
    else if (bxdf->type() & BxDF::SpecularReflection) \
        lvalue op reinterpret_cast<const SpecularReflection *>(bxdf)->function(__VA_ARGS__); \
    else if (bxdf->type() & BxDF::SpecularTransmission) \
        lvalue op reinterpret_cast<const SpecularTransmission *>(bxdf)->function(__VA_ARGS__); \
    else if (bxdf->type() & BxDF::Microfacet) \
        lvalue op reinterpret_cast<const Microfacet *>(bxdf)->function(__VA_ARGS__);

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
        __device__ __forceinline__ BSDF() { /* EMPTY */ }
        __device__ __forceinline__ BSDF(
                const DifferentialGeometry & dgShading,
                const optix::float3 & geometricNormal, const float eta = 1.0f)
        {
            m_gn = geometricNormal;
            m_nn = dgShading.normal;
            m_sn = optix::normalize(dgShading.dpdu);
            m_tn = optix::cross(m_nn, m_sn);
            m_nBxDFs = 0;
        }

        __device__ __forceinline__ unsigned int nBxDFs() const { return m_nBxDFs; }

        __device__ __forceinline__ unsigned int nBxDFs(BxDF::Type type) const
        {
            unsigned int count = 0;
            for (unsigned int i = 0; i < nBxDFs(); ++i)
                if (bxdfAt(i)->matchFlags(type))
                    ++count;
            return count;
        }

        __device__ __inline__ BxDF * bxdfAt(const Index & index)
        {
            const BSDF * bsdf = this;
            return const_cast<BxDF *>(bsdf->bxdfAt(index));
        }
        __device__ __inline__ const BxDF * bxdfAt(const Index & index) const
        {
            return reinterpret_cast<const BxDF *>(&m_bxdfList[index * MAX_BXDF_SIZE]);
        }
        __device__ __inline__ BxDF * bxdfAt(const Index & index, BxDF::Type type)
        {
            const BSDF * bsdf = this;
            return const_cast<BxDF *>(bsdf->bxdfAt(index, type));
        }
        __device__ __inline__ const BxDF * bxdfAt(const Index & index, BxDF::Type type) const
        {
            unsigned int count = index;
            for (unsigned int i = 0; i < nBxDFs(); ++i) {
                if (bxdfAt(i)->matchFlags(type))
                    if (count != 0)
                        --count;
                    else
                        return bxdfAt(i);
            }
            return NULL;
        }

        __device__ __forceinline__ bool isSpecular() const
        {
            return (nBxDFs(BxDF::Type(BxDF::All & ~BxDF::Specular)) == 0);
        }

    public:
        __device__ optix::float3 f(const optix::float3 & worldWo,
                const optix::float3 & worldWi, BxDF::Type sampleType = BxDF::All) const
        {
            optix::float3 wo, wi;
            worldToLocal(worldWo, &wo);
            worldToLocal(worldWi, &wi);
            // Calculate f.
            optix::float3 f = optix::make_float3(0.0f);
            if (optix::dot(m_gn, worldWi) * optix::dot(m_gn, worldWo) >= 0.0f)  // ignore BTDF
                sampleType = BxDF::Type(sampleType & ~BxDF::Transmission);
            else                                                                // ignore BRDF
                sampleType = BxDF::Type(sampleType & ~BxDF::Reflection);
            for (unsigned int i = 0; i < nBxDFs(); ++i)
                if (bxdfAt(i)->matchFlags(sampleType))
                    CALL_BXDF_CONST_VIRTUAL_FUNCTION(f, +=, bxdfAt(i), f, wo, wi);
            return f;
        }

        __device__ optix::float3 sampleF(const optix::float3 & worldWo,
                optix::float3 * worldWi, const optix::float3 & sample, float * prob,
                BxDF::Type sampleType = BxDF::All, BxDF::Type * sampledType = NULL) const
        {
            // Count matched componets.
            unsigned int nMatched = nBxDFs(sampleType);
            if (nMatched == 0) {
                *prob = 0.0f;
                if (sampledType) *sampledType = BxDF::Null;
                return optix::make_float3(0.0f);
            }

            // Sample BxDF.
            unsigned int index = min(nMatched-1,
                    static_cast<unsigned int>(floorf(sample.x * static_cast<float>(nMatched))));
            const BxDF * bxdf = bxdfAt(index, sampleType);
            if (bxdf == NULL) {
                *prob = 0.0f;
                if (sampledType) *sampledType = BxDF::Null;
                return optix::make_float3(0.0f);
            }

            // Transform.
            optix::float3 wo;
            worldToLocal(worldWo, &wo);

            // Sample f.
            optix::float3 f;
            optix::float3 wi;
            optix::float2 s = optix::make_float2(sample.y, sample.z);
            CALL_BXDF_CONST_VIRTUAL_FUNCTION(f, =, bxdf, sampleF, wo, &wi, s, prob);
            // Rejected.
            if (*prob == 0.0f) {
                if (sampledType) *sampledType = BxDF::Null;
                return optix::make_float3(0.0f);
            }
            // Otherwise.
            if (sampledType) *sampledType = bxdf->type();
            localToWorld(wi, worldWi);

            // If not specular, sum all non-specular BxDF's probability.
            if (!(bxdf->type() & BxDF::Specular) && nMatched > 1) {
                *prob = 1.0f;
                for (unsigned int i = 0; i < 1; i++)
                    if (bxdfAt(i)->matchFlags(sampleType))
                        CALL_BXDF_CONST_VIRTUAL_FUNCTION(*prob, +=, bxdfAt(i), probability, wo, wi);
            }
            // Remember to divide component count.
            if (nMatched > 1)
                *prob /= static_cast<float>(nMatched);
            // If not specular, sum all f.
            if (!(bxdf->type() & BxDF::Specular)) {
                f = make_float3(0.0f);
                // Cannot use sameHemisphere(wo, *wi) here,
                // do not confuse with the geometric normal and the shading normal.
                if (optix::dot(m_gn, *worldWi) * optix::dot(m_gn, worldWo) >= 0.0f)  // ignore BTDF
                    sampleType = BxDF::Type(sampleType & ~BxDF::Transmission);
                else                                                                 // ignore BRDF
                    sampleType = BxDF::Type(sampleType & ~BxDF::Reflection);
                for (unsigned int i = 0; i < nBxDFs(); ++i)
                    if (bxdfAt(i)->matchFlags(sampleType))
                        CALL_BXDF_CONST_VIRTUAL_FUNCTION(f, +=, bxdfAt(i), f, wo, wi);
            }

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

#endif  /* -----  #ifndef IGPPM_CORE_BSDF_H  ----- */
