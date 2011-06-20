/*
 * =====================================================================================
 *
 *       Filename:  geometry.h
 *
 *    Description:  
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

#ifndef MAOPPM_CORE_GEOMETRY_H
#define MAOPPM_CORE_GEOMETRY_H

/*-----------------------------------------------------------------------------
 *  header files from OptiX
 *-----------------------------------------------------------------------------*/
#include    <optix_world.h>



namespace MaoPPM {
class DifferentialGeometry {
    public:
        optix::float3   point;
        optix::float3   normal;
};  /* -----  end of class DifferentialGeometry  ----- */
}   /* -----  end of namespace MaoPPM  ----- */

#endif  /* -----  #ifndef MAOPPM_CORE_GEOMETRY_H  ----- */
