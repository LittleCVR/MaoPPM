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
#include    "particle.h"
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
        static const unsigned int  DEFAULT_N_PHOTONS_USED    = 32;
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

        class PixelSample : public GatherPoint {
            public:
                optix::float3  throughput;
                optix::float3  wo;

                __device__ __inline__ void reset()
                {
                    GatherPoint::reset();
                    throughput = optix::make_float3(1.0f);
                }
        };

    private:
        void createPhotonMap();

    public:
        void    init();
        void    resize(unsigned int width, unsigned int height);
        void    render(const Scene::RayGenCameraData & cameraData);

    private:
        unsigned int   m_nPhotonsUsed;
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
        unsigned int   m_densityEstimationPassLocalHeapOffset;
        unsigned int   m_densityEstimationPassLocalHeapSize;
        unsigned int   m_demandLocalHeapSize;
        unsigned int   m_frame;
};  /* -----  end of class PPMRenderer  ----- */
}   /* -----  end of namespace MaoPPM  ----- */

#endif  /* -----  #ifndef IGPPM_RENDERER_PPM_RENDERER_H  ----- */
