/*
 * =====================================================================================
 *
 *       Filename:  Scene.h
 *
 *    Description:  Scene class header file.
 *
 *        Version:  1.0
 *        Created:  2011-03-23 14:38:48
 *
 *         Author:  Chun-Wei Huang (LittleCVR), 
 *        Company:  Communication & Multimedia Laboratory,
 *                  Department of Computer Science & Information Engineering,
 *                  National Taiwan University
 *
 * =====================================================================================
 */

#ifndef SCENE_H
#define SCENE_H

/*-----------------------------------------------------------------------------
 *  header files from OptiX
 *-----------------------------------------------------------------------------*/
#include    <optix_world.h>

/*-----------------------------------------------------------------------------
 *  header files from sutil
 *-----------------------------------------------------------------------------*/
#include    <SampleScene.h>

/*-----------------------------------------------------------------------------
 *  header files of our own
 *-----------------------------------------------------------------------------*/
#include    "global.h"



namespace MaoPPM {
/*
 * =====================================================================================
 *        Class:  Scene
 *  Description:  
 * =====================================================================================
 */
class Scene : public SampleScene {
    friend class SceneBuilder;

    public:
        /* ====================  LIFECYCLE     ======================================= */
        Scene();
        ~Scene();

        /* ====================  ACCESSORS     ======================================= */
        optix::Buffer getOutputBuffer() { return m_outputBuffer; }
        inline bool isCameraChanged() const { return _camera_changed; }

        /* ====================  MUTATORS      ======================================= */

        /* ====================  OPERATORS     ======================================= */
        void doResize(unsigned int width, unsigned int height);
        void initScene(InitialCameraData & cameraData);
        void trace(const RayGenCameraData & cameraData);

    protected:
        inline void setIsCameraChanged(bool isCameraChanged) { _camera_changed = isCameraChanged; }

#ifdef DEBUG
    private:    // methods
        void initDebug();
#endif

    private:    // methods
        void initPixelSamplingPassData();
        void initImportonShootingPassData();
        void initPhotonShootingPassData();
        void setPrograms(const std::string & cuFileName,
                unsigned int entryPointIndex,
                const std::string & rayGenerationProgramName  = "generateRay",
                const std::string & missProgramName           = "handleMiss",
                const std::string & exceptionProgramName      = "handleException");

    private:
        void createPhotonMap();
        void buildPhotonMapAcceleration(MaoPPM::Photon * photonList,
                optix::uint start, optix::uint end, MaoPPM::Photon * photonMap,
                optix::uint root, optix::float3 bbMin, optix::float3 bbMax);
        void generateSamples(const optix::uint nSamples, optix::Buffer & sampleList);

    private:    // attributes
        InitialCameraData       m_initialCameraData;

        unsigned int            m_width;                /* screen width */
        unsigned int            m_height;               /* screen height */
        optix::Buffer           m_outputBuffer;         /* pixel buffer */

        optix::Buffer           m_pixelSampleList;
        optix::Buffer           m_importonMap;
        optix::uint             m_nEmittedPhotons;
        optix::Buffer           m_photonList;
        optix::Buffer           m_photonMap;

        optix::GeometryGroup    m_rootObject;
        optix::Buffer           m_lightList;
        optix::Buffer           m_sampleList;   /* random numbers */
};  /* -----  end of class Scene  ----- */
}   /* -----  end of namespace MaoPPM  ----- */



#endif  /* -----  #ifndef SCENE_H  ----- */
