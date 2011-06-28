/*
 * =====================================================================================
 *
 *       Filename:  Glass.h
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  2011-06-28 22:06:54
 *
 *         Author:  YOUR NAME (), 
 *        Company:  
 *
 * =====================================================================================
 */

#ifndef MAOPPM_MATERIALS_GLASS_H
#define MAOPPM_MATERIALS_GLASS_H

/*----------------------------------------------------------------------------
 *  header files from OptiX
 *----------------------------------------------------------------------------*/
#include    <optix_world.h>

/*----------------------------------------------------------------------------
 *  header files of our own
 *----------------------------------------------------------------------------*/
#include    "global.h"
#include    "BSDF.h"
#include    "Material.h"



namespace MaoPPM {
/*
 * =============================================================================
 *        Class:  Glass
 *  Description:  
 * =============================================================================
 */
class Glass : public Material {
    public:
        Glass(const optix::float3 & kr, const optix::float3 & kt, float index) :
            Material(Material::Glass), m_kr(kr), m_kt(kt), m_index(index) { /* EMPTY */ }

#ifdef __CUDACC__
    public:
        __device__ __inline__ void getBSDF(const DifferentialGeometry & dg, BSDF * bsdf) const
        {
            *bsdf = BSDF(dg, dg.normal);
            unsigned int offset = 0;
            if (!isBlack(m_kr)) {
                // SpecularReflection.
                SpecularReflection * ref = reinterpret_cast<SpecularReflection *>(bsdf->bxdfAt(offset));
                *ref = SpecularReflection(m_kr);
                FresnelDielectric * fresnel = reinterpret_cast<FresnelDielectric *>(ref->fresnel());
                *fresnel = FresnelDielectric(1.0f, m_index);
                ++offset;
            }
            if (!isBlack(m_kt)) {
                // SpecularTransmission.
                SpecularTransmission * tran = reinterpret_cast<SpecularTransmission *>(bsdf->bxdfAt(offset));
                *tran = SpecularTransmission(m_kt, 1.0f, m_index);
                ++offset;
            }
            bsdf->m_nBxDFs = offset;
        }
#endif  /* -----  #ifdef __CUDACC__  ----- */

    public:
        optix::float3   m_kr;
        optix::float3   m_kt;
        float           m_index;
};  /* -----  end of class Glass  ----- */
}   /* -----  end of namespace MaoPPM  ----- */

#endif  /* -----  #ifndef MAOPPM_MATERIALS_GLASS_H  ----- */
