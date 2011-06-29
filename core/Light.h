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

#define N_THETA  8
#define N_PHI    16

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

        float           pdf [N_THETA * N_PHI];
        float           cdf [N_THETA * N_PHI];

    public:
        __device__ __inline__ float area(
                unsigned int thetaIndex, unsigned int phiIndex) const
        {
            float thetaMin = static_cast<float>(thetaIndex+0) * M_PIf / N_THETA;
            float thetaMax = static_cast<float>(thetaIndex+1) * M_PIf / N_THETA;
            return (2.0f * M_PIf / static_cast<float>(N_PHI)) *
                (cosf(thetaMin) - cosf(thetaMax));
        }

        __device__ __inline__ float normalizedArea(
                unsigned int thetaIndex, unsigned int phiIndex) const
        {
            return area(thetaIndex, phiIndex) / (4.0f * M_PIf);
        }
};  /* -----  end of class Light  ----- */
}   /* -----  end of namespace MaoPPM  ----- */

#endif  /* -----  #ifndef MAOPPM_CORE_LIGHT_H  ----- */
