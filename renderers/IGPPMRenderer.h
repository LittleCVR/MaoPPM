/*
 * =============================================================================
 *
 *       Filename:  IGPPMRenderer.h
 *
 *    Description:  The Importons Guided Progressive Photon Map Renderer.
 *
 *        Version:  1.0
 *        Created:  2011-06-21 07:35:37
 *
 *         Author:  Chun-Wei Huang (LittleCVR), 
 *        Company:  Communication & Multimedia Laboratory,
 *                  Department of Computer Science & Information Engineering,
 *                  National Taiwan University
 *
 * =============================================================================
 */

#ifndef IGPPM_RENDERERS_IGPPM_RENDERER_H
#define IGPPM_RENDERERS_IGPPM_RENDERER_H

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
 *        Class:  IGPPMRenderer
 *  Description:  The Importons Guided Progressive Photon Map Renderer class.
 * =============================================================================
 */
class IGPPMRenderer : public Renderer {
    public:
        static const unsigned int  DEFAULT_N_IMPORTONS_PER_THREAD  = 8;
        static const unsigned int  DEFAULT_N_PHOTONS_WANTED        = 256*256*2;
        static const unsigned int  DEFAULT_PHOTON_SHOOTING_PASS_LAUNCH_WIDTH   = 256;
        static const unsigned int  DEFAULT_PHOTON_SHOOTING_PASS_LAUNCH_HEIGHT  = 256;

    public:
        IGPPMRenderer(Scene * scene = NULL);
        ~IGPPMRenderer();

    public:
        static const unsigned int  N_PASSES  = 4;
        enum Pass {
            PixelSamplingPass, ImportonShootingPass,
            PhotonShootingPass, FinalGatheringPass
        };

        typedef struct PixelSample {
            unsigned int   isHit;
            Intersection   intersection;
            optix::float3  wo;
            optix::float3  direct;
            optix::float3  indirect;

            __device__ __inline__ void reset()
            {
                isHit = false;
            }
        } PixelSample ;

        typedef struct Importon {
            unsigned int   isHit;
            float          weight;
            Intersection   intersection;
            optix::float3  wo;
            optix::float3  flux;
            unsigned int   nPhotons;
            float          radiusSquared;

            __device__ __inline__ void reset()
            {
                isHit = false;
            }
        } Importon ;

        typedef struct Photon {
            optix::float3  position;  // photon position
            optix::float3  normal;    // surface normal
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
        unsigned int   m_nImportonsPerThread;
        unsigned int   m_nPhotonsWanted;
        unsigned int   m_nPhotonsPerThread;
        unsigned int   m_nEmittedPhotons;
        optix::Buffer  m_pixelSampleList;
        optix::Buffer  m_importonList;
        optix::Buffer  m_photonMap;
};  /* -----  end of class IGPPMRenderer  ----- */
}   /* -----  end of namespace MaoPPM  ----- */

#endif  /* -----  #ifndef IGPPM_RENDERERS_IGPPM_RENDERER_H  ----- */
