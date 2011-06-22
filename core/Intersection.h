/*
 * =====================================================================================
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
 * =====================================================================================
 */

#ifndef MAOPPM_CORE_INTERSECTION_H
#define MAOPPM_CORE_INTERSECTION_H

/*-----------------------------------------------------------------------------
 *  header files from OptiX
 *-----------------------------------------------------------------------------*/
#include    <optix_world.h>

/*----------------------------------------------------------------------------
 *  header files of our own
 *----------------------------------------------------------------------------*/
#include    "global.h"
#include    "reflection.h"



namespace MaoPPM {
class Intersection {
    public:
        __device__ __inline__ DifferentialGeometry * dg() { return &m_dg; }

        __device__ __inline__ BSDF * bsdf()
        {
            return reinterpret_cast<BSDF *>(m_bsdf);
        }

        __device__ __inline__ optix::Matrix4x4 * worldToObject()
        {
            return reinterpret_cast<optix::Matrix4x4 *>(m_worldToObject);
        }

    private:
        DifferentialGeometry  m_dg;
        // Can't use BSDF directly here because nvcc would say:
        // can't generate code for non empty constructors or destructors on device
        char  m_bsdf[sizeof(BSDF)];
        // We do not store objectToWorld transform,
        // because for this transform, its inverse is equal to its transpose.
        char  m_worldToObject[sizeof(optix::Matrix4x4)];
};  /* -----  end of class Intersection  ----- */
}   /* -----  end of namespace MaoPPM  ----- */

#endif  /* -----  #ifndef MAOPPM_CORE_INTERSECTION_H  ----- */
