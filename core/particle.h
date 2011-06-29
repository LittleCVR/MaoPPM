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
#include    "utility.h"
#include    "BSDF.h"
#include    "KdTree.h"



#ifdef __CUDACC__
rtDeclareVariable(uint, nPhotonsUsed, , );
#endif  /* -----  #ifdef __CUDACC__  ----- */



namespace MaoPPM {

class HitPoint {
    public:
        enum Flag {
            Null   = 0,
            isHit  = 1 << 1,
            User   = 1 << 2
        };

    public:
        unsigned int    flags;

    public:
        __device__ __inline__ void reset()
        {
            flags = Null;
        }
        __device__ __inline__ Intersection * intersection()
        {
            return m_intersection;
        }
        __device__ __inline__ void setIntersection(Intersection * intersection)
        {
            m_intersection = intersection;
        }

    public:  // should be private
        Intersection *  m_intersection;
};

class Photon {
    public:
        enum Flag {
            Direct    = KdTree::User << 0,
            Indirect  = KdTree::User << 1,
            User      = KdTree::User << 2,
        };

    public:
        unsigned int   flags;     // various
        optix::float3  position;  // photon position
        optix::float3  wi;        // incident direction
        optix::float3  flux;      // photon flux

    public:
        __device__ __inline__ void reset()
        {
            flags  = 0;
            flux   = optix::make_float3(0.0f);
        }
};

class GatherPoint : public HitPoint {
    public:
        optix::float3  flux;
        unsigned int   nPhotons;
        float          radiusSquared;

    public:
        __device__ __inline__ void reset()
        {
            HitPoint::reset();
            flux = optix::make_float3(0.0f);
            nPhotons = 0;
        }
};

class GatheredPhoton {
    public:
        float           distanceSquared;
        const Photon *  photon;

#ifdef __CUDACC__
    public:
        __device__ __inline__ GatheredPhoton(float d, const Photon * p) :
            distanceSquared(d), photon(p) { /* EMPTY */ }

    public:
        __device__ __inline__ bool operator<(const GatheredPhoton & p)
        {
            return distanceSquared < p.distanceSquared;
        }
#endif  /* -----  #ifdef __CUDACC__  ----- */
};

class PhotonGatherer {
    public:
        unsigned int           nFound;
        optix::float3          flux;
        const optix::float3 *  wo;
        const BSDF *           bsdf;

#ifdef __CUDACC__
    public:
        __device__ __inline__ PhotonGatherer(const optix::float3 * w, const BSDF * b) :
            nFound(0), wo(w), bsdf(b)
        {
            flux = optix::make_float3(0.0f);
        }

    public:
        __device__ __inline__ void gather(const optix::float3 & point,
                const Photon * photon, float distanceSquared, float * maxDistanceSquared)
        {
            optix::float3 f = bsdf->f(*wo, photon->wi);
            if (!isBlack(f))
                flux += f * photon->flux;
        }
#endif  /* -----  #ifdef __CUDACC__  ----- */
};

class LimitedPhotonGatherer {
    public:
        unsigned int      nFound;
        GatheredPhoton *  gatheredPhotonList;

#ifdef __CUDACC__
    public:
        __device__ __inline__ LimitedPhotonGatherer(GatheredPhoton * list) :
            nFound(0), gatheredPhotonList(list) { /* EMPTY */ }

    public:
        __device__ __inline__ void gather(const optix::float3 & point,
                const Photon * photon, float distanceSquared, float * maxDistanceSquared)
        {
            if (nFound < nPhotonsUsed)
                gatheredPhotonList[nFound++] = GatheredPhoton(distanceSquared, photon);
            else {
                float maxD = 0.0f, maxD2 = 0.0f;
                unsigned int index = 0;
                for (unsigned int i = 0; i < nFound; ++i)
                    if (gatheredPhotonList[i].distanceSquared > maxD) {
                        index = i;
                        maxD2 = maxD;
                        maxD = gatheredPhotonList[i].distanceSquared;
                    }
                gatheredPhotonList[index] = GatheredPhoton(distanceSquared, photon);
                *maxDistanceSquared = fmaxf(maxD2, distanceSquared);
            }
        }
#endif  /* -----  #ifdef __CUDACC__  ----- */
};

}   /* ----------  end of namespace MaoPPM  ---------- */

#endif  /* ----- #ifndef IGPPM_CORE_PARTICLE_H  ----- */
