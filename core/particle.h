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
#include    "Light.h"



#ifdef __CUDACC__
rtDeclareVariable(unsigned int, nPhotonsUsed, , );
#endif  /* -----  #ifdef __CUDACC__  ----- */



namespace MaoPPM {

static const float  DEFAULT_RADIUS  = 16.0f;

class HitPoint {
    public:
        enum Flag {
            Null   = 0,
            isHit  = 1 << 1,
            User   = 1 << 2
        };

    public:
        unsigned int  flags;

    public:
        __device__ __forceinline__ void reset()
        {
            flags = Null;
        }
        __device__ __forceinline__ Intersection * intersection()
        {
            return m_intersection;
        }
        __device__ __forceinline__ void setIntersection(Intersection * intersection)
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
            Caustic   = KdTree::User << 1,
            Indirect  = KdTree::User << 2,
            All       = Direct | Caustic | Indirect,
            User      = KdTree::User << 3,
        };

    public:
        unsigned int   flags;     // various
        optix::float3  position;  // photon position
        optix::float3  wi;        // incident direction
        optix::float3  flux;      // photon flux

    public:
        __device__ __forceinline__ void reset()
        {
            flags  = 0;
            flux   = optix::make_float3(0.0f);
        }
};

class RadiancePhoton {
    public:
        optix::float3  position;
        optix::float3  normal;
        optix::float3  radiance;

    public:
        __device__ __forceinline__ void reset()
        {
            radiance = optix::make_float3(0.0f);
        }
};

class GatherPoint : public HitPoint {
    public:
        unsigned int   nPhotons;
        float          radiusSquared;
        optix::float3  flux;

#ifdef __CUDACC__
    public:
        __device__ __forceinline__ void reset()
        {
            HitPoint::reset();
            flux = optix::make_float3(0.0f);
            nPhotons = 0;
        }

    public:
        __device__ __forceinline__ void shrinkRadius(
                const optix::float3 & f, unsigned int newPhotons,
                float * reductionFactorSquared = NULL)
        {
            /* TODO: let alpha be configurable */
            float alpha = 0.7f;
            float R2 = radiusSquared;
            float N = nPhotons;
            float M = static_cast<float>(newPhotons) ;
            float newN = N + alpha*M;
            nPhotons = newN;

            float reductionFactor2 = 1.0f;
            float newR2 = R2;
            if (M != 0) {
                reductionFactor2 = (N + alpha*M) / (N + M);
                newR2 = R2 * reductionFactor2;
                radiusSquared = newR2;
            }
            if (reductionFactorSquared)
                *reductionFactorSquared = reductionFactor2;

            flux = (flux + f) * reductionFactor2;
        }
#endif  /* -----  #ifdef __CUDACC__  ----- */
};

class GatheredPhoton {
    public:
        float           distanceSquared;
        const Photon *  photon;

#ifdef __CUDACC__
    public:
        __device__ __forceinline__ GatheredPhoton(float d, const Photon * p) :
            distanceSquared(d), photon(p) { /* EMPTY */ }

    public:
        __device__ __forceinline__ bool operator<(const GatheredPhoton & p)
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
        Photon::Flag           condition;

        Light *                light;
        float                  weight;

#ifdef __CUDACC__
    public:
        __device__ __forceinline__ PhotonGatherer(
                const optix::float3 * w, const BSDF * b, Photon::Flag cond, Light * l = NULL, float ww = 0.0f) :
            nFound(0), wo(w), bsdf(b), condition(cond), light(l), weight(ww)
        {
            flux = optix::make_float3(0.0f);
        }

    public:
        __device__ __forceinline__ static optix::float3 accumulateFlux(
                const optix::float3 & point, const optix::float3 & wo, const BSDF * bsdf,
                const Photon * photonMap, float * maxDistanceSquared, unsigned int * nAccumulatedPhotons,
                Photon::Flag cond = Photon::Flag(~0), Light * l = NULL, float ww = 0.0f)
        {
            PhotonGatherer gatherer(&wo, bsdf, cond, l, ww);
            KdTree::find(point, &photonMap[0], &gatherer, maxDistanceSquared);
            *nAccumulatedPhotons = gatherer.nFound;
            return gatherer.flux;
        }

    public:
        __device__ __forceinline__ void gather(const optix::float3 & point,
                const Photon * photon, float distanceSquared, float * maxDistanceSquared)
        {
            if (!(photon->flags & condition))
                return;
            optix::float3 f = bsdf->f(*wo, photon->wi);
            if (!isBlack(f))
                flux += f * photon->flux;
            /* TODO */
            unsigned int thetaBin = (photon->flags >> 24) & 0xFF;
            unsigned int phiBin   = (photon->flags >> 16) & 0xFF;
            light->pdf[thetaBin*N_PHI + phiBin] += weight;
        }
#endif  /* -----  #ifdef __CUDACC__  ----- */
};

class LimitedPhotonGatherer {
    public:
        unsigned int      nFound;
        Photon::Flag      condition;
        GatheredPhoton *  gatheredPhotonList;

#ifdef __CUDACC__
    public:
        __device__ __forceinline__ LimitedPhotonGatherer(
                GatheredPhoton * list, Photon::Flag cond) :
            nFound(0), condition(cond), gatheredPhotonList(list) { /* EMPTY */ }

    public:
        __device__ __forceinline__ static optix::float3 accumulateFlux(
                const optix::float3 & point, const optix::float3 & wo, const BSDF * bsdf,
                const Photon * photonMap, float * maxDistanceSquared, unsigned int * nAccumulatedPhotons,
                GatheredPhoton * gatheredPhotonList, Photon::Flag cond = Photon::Flag(~0))
        {
            optix::float3 flux = optix::make_float3(0.0f);
            LimitedPhotonGatherer gatherer(gatheredPhotonList, cond);
            KdTree::find(point, photonMap, &gatherer, maxDistanceSquared);
            // Accumulate flux.
            *nAccumulatedPhotons = gatherer.nFound;
            for (unsigned int i = 0; i < gatherer.nFound; ++i) {
                const Photon * photon = gatherer.gatheredPhotonList[i].photon;
                float3 f = bsdf->f(wo, photon->wi);
                if (!isBlack(f))
                    flux += f * photon->flux;
            }
            // Return.
            return flux;
        }

    public:
        __device__ __forceinline__ void gather(const optix::float3 & point,
                const Photon * photon, float distanceSquared, float * maxDistanceSquared)
        {
            if (!(photon->flags & condition))
                return;
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
