/*
 * =============================================================================
 *
 *       Filename:  Light.h
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  2011-06-20 17:54:38
 *
 *         Author:  Chun-Wei Huang (LittleCVR), 
 *        Company:  Communication & Multimedia Laboratory,
 *                  Department of Computer Science & Information Engineering,
 *                  National Taiwan University
 *
 * =============================================================================
 */

#ifndef MAOPPM_CORE_LIGHT_H
#define MAOPPM_CORE_LIGHT_H

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
 *        Class:  Light
 *  Description:  
 * =============================================================================
 */
class Light {
    public:
        Light() { }
        ~Light() { }

        optix::float3   position;
        optix::float3   flux;
};  /* -----  end of class Light  ----- */
}   /* -----  end of namespace MaoPPM  ----- */

#endif  /* -----  #ifndef MAOPPM_CORE_LIGHT_H  ----- */
