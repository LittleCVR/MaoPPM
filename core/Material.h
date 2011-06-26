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
        Material() { /* EMPTY */ }
        ~Material() { /* EMPTY */ }
};  /* -----  end of class Material  ----- */
}   /* -----  end of namespace MaoPPM  ----- */

#endif  /* -----  #ifndef MAOPPM_CORE_MATERIAL_H  ----- */
