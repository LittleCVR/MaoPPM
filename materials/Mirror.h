/*
 * =============================================================================
 *
 *       Filename:  Mirror.h
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  2011-06-28 18:52:22
 *
 *         Author:  Chun-Wei Huang (LittleCVR), 
 *        Company:  Communication & Multimedia Laboratory,
 *                  Department of Computer Science & Information Engineering,
 *                  National Taiwan University
 *
 * =============================================================================
 */

#ifndef MAOPPM_MATERIALS_MIRROR_H
#define MAOPPM_MATERIALS_MIRROR_H

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
 *        Class:  Mirror
 *  Description:  
 * =============================================================================
 */
class Mirror : public Material {
    public:
        Mirror(const optix::float3 & kr) : Material(Material::Mirror), m_kr(kr) { /* EMPTY */ }
        ~Mirror() { /* EMPTY */ }

#ifdef __CUDACC__
    public:
        __device__ __inline__ void getBSDF(const DifferentialGeometry & dg, BSDF * bsdf) const
        {
            *bsdf = BSDF(dg, dg.normal);
            bsdf->m_nBxDFs = 1;
            SpecularReflection * spec = reinterpret_cast<SpecularReflection *>(bsdf->bxdfAt(0));
            *spec = SpecularReflection(m_kr);
            FresnelNoOp * fresnel = reinterpret_cast<FresnelNoOp *>(spec->fresnel());
            *fresnel = FresnelNoOp();
        }
#endif  /* -----  #ifdef __CUDACC__  ----- */

    private:
        optix::float3  m_kr;
};  /* -----  end of class Mirror  ----- */
}   /* -----  end of namespace MaoPPM  ----- */

#endif  /* -----  #ifndef MAOPPM_MATERIALS_MIRROR_H  ----- */
