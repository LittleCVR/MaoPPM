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
                CALL_BXDF_CONST_VIRTUAL_FUNCTION(totalF, +=, bxdf, f, wo, wi);
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
            CALL_BXDF_CONST_VIRTUAL_FUNCTION(f, =, bxdf, sampleF, wo, &wi, s, &prob);
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

#endif  /* -----  #ifndef IGPPM_CORE_BSDF_H  ----- */
