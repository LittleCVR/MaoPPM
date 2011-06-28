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
#include    "KdTree.h"
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
        static const unsigned int  DEFAULT_N_IMPORTONS_PER_THREAD  = 4;
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

        typedef struct PixelSample {
            unsigned int    isHit;
            Intersection *  intersection;
            optix::float3   wo;
            optix::float3   direct;

            __device__ __inline__ void reset()
            {
                isHit   = false;
                direct  = optix::make_float3(0.0f);
            }
        } PixelSample ;

        typedef struct Importon {
            unsigned int    isHit;
            optix::float3   weight;
            Intersection *  intersection;
            optix::float3   wo;
            optix::float3   flux;
            unsigned int    nPhotons;
            float           radiusSquared;

            __device__ __inline__ void reset()
            {
                isHit  = false;
            }
        } Importon ;

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
