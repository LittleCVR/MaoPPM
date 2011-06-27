/*
 * =============================================================================
 *
 *       Filename:  Plastic.h
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  2011-06-27 17:09:25
 *
 *         Author:  Chun-Wei Huang (LittleCVR), 
 *        Company:  Communication & Multimedia Laboratory,
 *                  Department of Computer Science & Information Engineering,
 *                  National Taiwan University
 *
 * =============================================================================
 */

#ifndef MAOPPM_MATERIALS_PLASTIC_H
#define MAOPPM_MATERIALS_PLASTIC_H

/*----------------------------------------------------------------------------
 *  header files from OptiX
 *----------------------------------------------------------------------------*/
#include    <optix_world.h>

/*----------------------------------------------------------------------------
 *  header files of our own
 *----------------------------------------------------------------------------*/
#include    "global.h"
#include    "reflection.h"
#include    "Material.h"



namespace MaoPPM {
/*
 * =============================================================================
 *        Class:  Plastic
 *  Description:  
 * =============================================================================
 */
class Plastic : public Material {
    public:
        Plastic(const optix::float3 & kd, const optix::float3 ks, const float roughness) :
            Material(Material::Plastic), m_kd(kd), m_ks(ks), m_roughness(roughness) { /* EMPTY */ }

#ifdef __CUDACC__
    public:
        __device__ __inline__ void getBSDF(const DifferentialGeometry & dg, BSDF * bsdf) const
        {
            *bsdf = BSDF(dg, dg.normal);
            bsdf->m_nBxDFs = 2;

            Lambertian * lambertian = reinterpret_cast<Lambertian *>(bsdf->bxdfAt(0));
            *lambertian = Lambertian(m_kd);

            Microfacet * microfacet = reinterpret_cast<Microfacet *>(bsdf->bxdfAt(1));
            *microfacet = Microfacet(m_ks);
            FresnelDielectric * fresnel = reinterpret_cast<FresnelDielectric *>(microfacet->fresnel());
            *fresnel = FresnelDielectric(1.5f, 1.0f);
            Blinn * blinn = reinterpret_cast<Blinn *>(microfacet->distribution());
            *blinn = Blinn(1.0f / m_roughness);
        }
#endif  /* -----  #ifdef __CUDACC__  ----- */

    public:
        optix::float3  m_kd;
        optix::float3  m_ks;
        float          m_roughness;
};  /* -----  end of class METAL  ----- */
}   /* -----  end of namespace MaoPPM  ----- */

#endif  /* -----  #ifndef MAOPPM_MATERIALS_PLASTIC_H  ----- */
