/*
 * =============================================================================
 *
 *       Filename:  Material.h
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  2011-06-20 17:49:44
 *
 *         Author:  Chun-Wei Huang (LittleCVR), 
 *        Company:  Communication & Multimedia Laboratory,
 *                  Department of Computer Science & Information Engineering,
 *                  National Taiwan University
 *
 * =============================================================================
 */

#ifndef MAOPPM_CORE_MATERIAL_H
#define MAOPPM_CORE_MATERIAL_H

/*----------------------------------------------------------------------------
 *  header files from OptiX
 *----------------------------------------------------------------------------*/
#include    <optix_world.h>

/*----------------------------------------------------------------------------
 *  header files of our own
 *----------------------------------------------------------------------------*/
#include    "global.h"



namespace MaoPPM {
/*
 * =============================================================================
 *        Class:  Material
 *  Description:  
 * =============================================================================
 */
class Material {
    public:
        enum Type {
            Matte  = 1 << 0
        };

    public:
        Material(Type type) : m_type(type) { /* EMPTY */ }
        ~Material() { /* EMPTY */ }

#ifdef __CUDACC__
    public:
        __device__ __inline__ Type type() const { return m_type; }
#endif  /* -----  #ifdef __CUDACC__  ----- */

    private:
        Type  m_type;
};  /* -----  end of class Material  ----- */
}   /* -----  end of namespace MaoPPM  ----- */

#endif  /* -----  #ifndef MAOPPM_CORE_MATERIAL_H  ----- */
