/*
 * =============================================================================
 *
 *       Filename:  PPMRenderer.h
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  2011-06-08 12:00:58
 *
 *         Author:  Chun-Wei Huang (LittleCVR), 
 *        Company:  Communication & Multimedia Laboratory,
 *                  Department of Computer Science & Information Engineering,
 *                  National Taiwan University
 *
 * =============================================================================
 */

#ifndef IGPPM_RENDERER_PPM_RENDERER_H
#define IGPPM_RENDERER_PPM_RENDERER_H

/*----------------------------------------------------------------------------
 *  header files from OptiX
 *----------------------------------------------------------------------------*/
#include    <optix_world.h>

/*----------------------------------------------------------------------------
 *  header files of our own
 *----------------------------------------------------------------------------*/
#include    "global.h"
#include    "reflection.h"
#include    "DifferentialGeometry.h"
#include    "Intersection.h"
#include    "Renderer.h"



namespace MaoPPM {
/*
 * =============================================================================
 *        Class:  PPMRenderer
 *  Description:  
 * =============================================================================
 */
class PPMRenderer : public Renderer {
    public:
        static const unsigned int  DEFAULT_N_PHOTONS_WANTED  = 256*256*4;
        static const unsigned int  DEFAULT_PHOTON_SHOOTING_PASS_LAUNCH_WIDTH   = 256;
        static const unsigned int  DEFAULT_PHOTON_SHOOTING_PASS_LAUNCH_HEIGHT  = 256;

    public:     // methods
        PPMRenderer(Scene * scene = NULL);
        ~PPMRenderer();

    public:
        static const unsigned int  N_PASSES  = 3;
        enum Pass {
            PixelSamplingPass, PhotonShootingPass, DensityEstimationPass
        };

        typedef struct PixelSample {
            unsigned int    isHit;
            Intersection *  intersection;
            optix::float3   wo;
            optix::float3   direct;
            optix::float3   flux;
            unsigned int    nPhotons;
            float           radiusSquared;

            __device__ __inline__ void reset()
            {
                isHit   = false;
                direct  = optix::make_float3(0.0f);
            }
        } PixelSample ;

        typedef struct Photon {
            optix::float3  position;  // photon position
            optix::float3  wi;        // incident direction
            optix::float3  flux;      // photon flux

            enum Flags {
                Null      = 0,
                Leaf      = 1 << 0,
                AxisX     = 1 << 1,
                AxisY     = 1 << 2,
                AxisZ     = 1 << 3,
                Direct    = 1 << 4,
                Indirect  = 1 << 5
            };
            unsigned int   flags;     // for KdTree

            __device__ __inline__ void reset()
            {
                flags  = Null;
                flux   = optix::make_float3(0.0f);
            }

            static bool positionXComparator(const Photon & photon1, const Photon & photon2)
            {
                return photon1.position.x < photon2.position.x;
            }
            static bool positionYComparator(const Photon & photon1, const Photon & photon2)
            {
                return photon1.position.y < photon2.position.y;
            }
            static bool positionZComparator(const Photon & photon1, const Photon & photon2)
            {
                return photon1.position.z < photon2.position.z;
            }
        } Photon ;

    private:
        void createPhotonMap();
        void buildPhotonMapAcceleration(Photon * photonList,
                optix::uint start, optix::uint end, Photon * photonMap,
                optix::uint root, optix::float3 bbMin, optix::float3 bbMax);

    public:
        void    init();
        void    resize(unsigned int width, unsigned int height);
        void    render(const Scene::RayGenCameraData & cameraData);

    private:
        unsigned int   m_nPhotonsWanted;
        unsigned int   m_nPhotonsPerThread;
        unsigned int   m_photonShootingPassLaunchWidth;
        unsigned int   m_photonShootingPassLaunchHeight;
        unsigned int   m_nEmittedPhotons;
        optix::Buffer  m_pixelSampleList;
        optix::Buffer  m_photonMap;
        unsigned int   m_pixelSamplingPassLocalHeapSize;
        unsigned int   m_photonShootingPassLocalHeapOffset;
        unsigned int   m_photonShootingPassLocalHeapSize;
        unsigned int   m_demandLocalHeapSize;
        unsigned int   m_frame;
};  /* -----  end of class PPMRenderer  ----- */
}   /* -----  end of namespace MaoPPM  ----- */

#endif  /* -----  #ifndef IGPPM_RENDERER_PPM_RENDERER_H  ----- */
