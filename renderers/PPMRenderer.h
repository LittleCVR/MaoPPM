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

#ifndef PPM_RENDERER_H
#define PPM_RENDERER_H

/*----------------------------------------------------------------------------
 *  header files from OptiX
 *----------------------------------------------------------------------------*/
#include    <optix_world.h>

/*----------------------------------------------------------------------------
 *  header files of our own
 *----------------------------------------------------------------------------*/
#include    "global.h"
#include    "Renderer.h"



namespace MaoPPM {
/*
 * =============================================================================
 *        Class:  PPMRenderer
 *  Description:  
 * =============================================================================
 */
class PPMRenderer : public Renderer {
    public:     // methods
        PPMRenderer(Scene * scene);
        ~PPMRenderer();

    public:     // methods
        void    init();
        void    render();
        void    resize(unsigned int width, unsigned int height);

    private:    // methods
        void    initPixelSamplingPassData();
        void    initImportonShootingPassData();
        void    initPhotonShootingPassData();

    private:
        void createPhotonMap();
        void buildPhotonMapAcceleration(MaoPPM::Photon * photonList,
                optix::uint start, optix::uint end, MaoPPM::Photon * photonMap,
                optix::uint root, optix::float3 bbMin, optix::float3 bbMax);

    private:    // attributes
        optix::Buffer           m_pixelSampleList;
        optix::Buffer           m_importonMap;
        optix::uint             m_nEmittedPhotons;
        optix::Buffer           m_photonList;
        optix::Buffer           m_photonMap;
};  /* -----  end of class PPMRenderer  ----- */
}   /* -----  end of namespace MaoPPM  ----- */

#endif  /* -----  #ifndef PPM_RENDERER_H  ----- */
