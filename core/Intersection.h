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
#include    "reflection.h"
#include    "utility.h"
#include    "Material.h"
#include    "Matte.h"



namespace MaoPPM {
class Intersection {
#ifdef __CUDACC__
    public:
        __device__ __inline__ DifferentialGeometry * dg() { return &m_dg; }

        __device__ __inline__ BSDF * bsdf()
        {
            return reinterpret_cast<BSDF *>(m_bsdf);
        }

        __device__ __inline__ optix::Matrix4x4 worldToObject() const
        {
            optix::float3 n = m_dg.normal;
            optix::float3 s = optix::normalize(m_dg.dpdu);
            optix::float3 t = optix::cross(n, s);
            return createCoordinateSystemTransform(s, t, n);
        }
#endif  /* -----  #ifdef __CUDACC__  ----- */

    private:
        DifferentialGeometry  m_dg;
        // Can't use BSDF directly here because nvcc would say:
        // can't generate code for non empty constructors or destructors on device
        char  m_bsdf [sizeof(BSDF)];
};  /* -----  end of class Intersection  ----- */
}   /* -----  end of namespace MaoPPM  ----- */

#endif  /* -----  #ifndef MAOPPM_CORE_INTERSECTION_H  ----- */
