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
#include    "particle.h"
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
        static const unsigned int  DEFAULT_N_IMPORTONS_PER_THREAD  = 1;
        static const unsigned int  DEFAULT_N_PHOTONS_WANTED        = 256*256*4;
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

        class PixelSample : public HitPoint {
            public:
                enum Flag {
                    Finished  = HitPoint::User << 0
                };

            public:
                unsigned int    nEmittedPhotonsOffset;
                unsigned int    nSampled;
                optix::float3   radiance;
                optix::float3   wo;
                optix::float3   throughput;
                optix::float3   direct;

                __device__ __inline__ void reset()
                {
                    HitPoint::reset();
                    direct  = optix::make_float3(0.0f);
                }
        };

        class Importon : public GatherPoint {
            public:
                optix::float3   weight;
                optix::float3   wo;

                __device__ __inline__ void reset()
                {
                    GatherPoint::reset();
                }
        };

        class Photon : public MaoPPM::Photon {
            public:
                __device__ __inline__ unsigned int thetaBin() const
                {
                    return (flags >> 24) & 0xFF;
                }
                __device__ __inline__ unsigned int phiBin() const
                {
                    return (flags >> 16) & 0xFF;
                }
        };

    private:
        void createPhotonMap();

    public:
        void    init();
        void    resize(unsigned int width, unsigned int height);
        void    render(const Scene::RayGenCameraData & cameraData);

    private:
        unsigned int   m_nImportonsPerThread;
        unsigned int   m_nPhotonsWanted;
        unsigned int   m_nPhotonsPerThread;
        unsigned int   m_photonShootingPassLaunchWidth;
        unsigned int   m_photonShootingPassLaunchHeight;
        unsigned int   m_nEmittedPhotons;
        optix::Buffer  m_pixelSampleList;
        optix::Buffer  m_importonList;
        optix::Buffer  m_photonMap;
        unsigned int   m_pixelSamplingPassLocalHeapSize;
        unsigned int   m_importonShootingPassLocalHeapOffset;
        unsigned int   m_importonShootingPassLocalHeapSize;
        unsigned int   m_photonShootingPassLocalHeapOffset;
        unsigned int   m_photonShootingPassLocalHeapSize;
        unsigned int   m_demandLocalHeapSize;
        unsigned int   m_frame;
};  /* -----  end of class IGPPMRenderer  ----- */
}   /* -----  end of namespace MaoPPM  ----- */

#endif  /* -----  #ifndef IGPPM_RENDERERS_IGPPM_RENDERER_H  ----- */
