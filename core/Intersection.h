/*
 * =============================================================================
 *
 *       Filename:  Intersection.h
 *
 *    Description:  Intersection class file.
 *
 *        Version:  1.0
 *        Created:  2011-06-19 14:57:06
 *
 *         Author:  Chun-Wei Huang (LittleCVR), 
 *        Company:  Communication & Multimedia Laboratory,
 *                  Department of Computer Science & Information Engineering,
 *                  National Taiwan University
 *
 * =============================================================================
 */

#ifndef MAOPPM_CORE_INTERSECTION_H
#define MAOPPM_CORE_INTERSECTION_H

/*----------------------------------------------------------------------------
 *  header files from OptiX
 *----------------------------------------------------------------------------*/
#include    <optix_world.h>

/*---------------------------------------------------------------------------
 *  header files of our own
 *---------------------------------------------------------------------------*/
#include    "global.h"
#include    "utility.h"
#include    "BSDF.h"
#include    "DifferentialGeometry.h"
#include    "Material.h"
#include    "Matte.h"
#include    "Plastic.h"



namespace MaoPPM {

#define CALL_MATERIAL_VIRTUAL_FUNCTION(lvalue, op, material, function, ...) \
    if (material->type() & Material::Matte) \
        lvalue op reinterpret_cast<Matte *>(material)->function(__VA_ARGS__); \
    else if (material->type() & Material::Plastic) \
        lvalue op reinterpret_cast<Plastic *>(material)->function(__VA_ARGS__);

class Intersection {
#ifdef __CUDACC__
    public:
        __device__ __inline__ DifferentialGeometry * dg() { return &m_dg; }

        __device__ __inline__ void getBSDF(BSDF * bsdf)
        {
            CALL_MATERIAL_VIRTUAL_FUNCTION( , , m_material, getBSDF, m_dg, bsdf);
        }

        __device__ __inline__ optix::Matrix4x4 worldToObject() const
        {
            optix::float3 n = m_dg.normal;
            optix::float3 s = optix::normalize(m_dg.dpdu);
            optix::float3 t = optix::cross(n, s);
            return createCoordinateSystemTransform(s, t, n);
        }
#endif  /* -----  #ifdef __CUDACC__  ----- */

//    private:
        DifferentialGeometry  m_dg;
        Material *            m_material;
};  /* -----  end of class Intersection  ----- */
}   /* -----  end of namespace MaoPPM  ----- */

#endif  /* -----  #ifndef MAOPPM_CORE_INTERSECTION_H  ----- */
