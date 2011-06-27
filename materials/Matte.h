/*
 * =============================================================================
 *
 *       Filename:  Matte.h
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  2011-06-20 17:59:51
 *
 *         Author:  Chun-Wei Huang (LittleCVR), 
 *        Company:  Communication & Multimedia Laboratory,
 *                  Department of Computer Science & Information Engineering,
 *                  National Taiwan University
 *
 * =============================================================================
 */

#ifndef MAOPPM_MATERIALS_MATTE_H
#define MAOPPM_MATERIALS_MATTE_H

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
 *        Class:  Matte
 *  Description:  
 * =============================================================================
 */
class Matte : public Material {
    public:
        Matte(optix::float3 kd) : Material(Material::Matte), m_kd(kd) { /* EMPTY */ }
        ~Matte() { /* EMPTY */ }

#ifdef __CUDACC__
    public:
        __device__ __inline__ void getBSDF(const DifferentialGeometry & dg, BSDF * bsdf) const
        {
            *bsdf = BSDF(dg, dg.normal);
            bsdf->m_nBxDFs = 1;
            Lambertian * lambertian = reinterpret_cast<Lambertian *>(bsdf->bxdfAt(0));
            *lambertian = Lambertian(m_kd);
        }
#endif  /* -----  #ifdef __CUDACC__  ----- */

    public:
        optix::float3   m_kd;
};  /* -----  end of class Matte  ----- */
}   /* -----  end of namespace MaoPPM  ----- */

#endif  /* -----  #ifndef MAOPPM_MATERIALS_MATTE_H  ----- */
