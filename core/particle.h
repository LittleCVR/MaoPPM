/*
 * =============================================================================
 *
 *       Filename:  particle.h
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  2011-06-29 12:00:54
 *
 *         Author:  Chun-Wei Huang (LittleCVR), 
 *        Company:  Communication & Multimedia Laboratory,
 *                  Department of Computer Science & Information Engineering,
 *                  National Taiwan University
 *
 * =============================================================================
 */

#ifndef IGPPM_CORE_PARTICLE_H
#define IGPPM_CORE_PARTICLE_H

/*----------------------------------------------------------------------------
 *  header files of our own
 *----------------------------------------------------------------------------*/
#include    "global.h"
#include    "KdTree.h"



namespace MaoPPM {

class Photon {
    public:
        unsigned int   flags;     // various
        optix::float3  position;  // photon position
        optix::float3  wi;        // incident direction
        optix::float3  flux;      // photon flux

        enum Flag {
            Direct    = KdTree<Photon>::User << 0,
            Indirect  = KdTree<Photon>::User << 1,
            User      = KdTree<Photon>::User << 2,
        };

        __device__ __inline__ void reset()
        {
            flags  = 0;
            flux   = optix::make_float3(0.0f);
        }
};

class GatherPoint {
    public:
        optix::float3  flux;
        unsigned int   nPhotons;
        float          radiusSquared;
};

}   /* ----------  end of namespace MaoPPM  ---------- */

#endif  /* ----- #ifndef IGPPM_CORE_PARTICLE_H  ----- */
