/*
 * =====================================================================================
 *
 *       Filename:  DifferentialGeometry.h
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

#ifndef MAOPPM_CORE_DIFFERENTIAL_GEOMETRY_H
#define MAOPPM_CORE_DIFFERENTIAL_GEOMETRY_H

/*-----------------------------------------------------------------------------
 *  header files from OptiX
 *-----------------------------------------------------------------------------*/
#include    <optix_world.h>

/*----------------------------------------------------------------------------
 *  header files of our own
 *----------------------------------------------------------------------------*/
#include    "global.h"



namespace MaoPPM {
typedef struct DifferentialGeometry {
    optix::float3  point;
    optix::float3  normal;
    optix::float3  dpdu, dpdv;
} DifferentialGeometry ;  /* -----  end of struct DifferentialGeometry  ----- */
}   /* -----  end of namespace MaoPPM  ----- */

#endif  /* -----  #ifndef MAOPPM_CORE_DIFFERENTIAL_GEOMETRY_H  ----- */
