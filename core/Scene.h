/*
 * =============================================================================
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
 * =============================================================================
 */

#ifndef SCENE_H
#define SCENE_H

/*----------------------------------------------------------------------------
 *  header files from OptiX
 *----------------------------------------------------------------------------*/
#include    <optix_world.h>

/*----------------------------------------------------------------------------
 *  header files from sutil
 *----------------------------------------------------------------------------*/
#include    <SampleScene.h>

/*----------------------------------------------------------------------------
 *  header files of our own
 *----------------------------------------------------------------------------*/
#include    "global.h"



namespace MaoPPM {
/*
 * =============================================================================
 *        Class:  Scene
 *  Description:  
 * =============================================================================
 */
class Scene : public SampleScene {
    friend class Renderer;
    /* TODO: remove */
    friend class IGPPMRenderer;
    friend class SceneBuilder;

    public:
        Scene();
        ~Scene();

        optix::Context      context();
        inline bool         isCameraChanged() const { return _camera_changed; }
        inline void         setIsCameraChanged(bool isCameraChanged) { _camera_changed = isCameraChanged; }
        inline Renderer *   renderer() const { return m_renderer; }
        void                setRenderer(Renderer * renderer);

        optix::Buffer       inputHeap() const { return m_inputHeap; }
        /*
         *----------------------------------------------------------------------
         *       Class:  Scene
         *      Method:  Scene :: copyToHeap
         * Description:  Copy $size bytes from $data to heap. Probabily expands
         *               the heap size if necessary. Returns the start position
         *               of copied data on the heap.
         *----------------------------------------------------------------------
         */
        Index               copyToHeap(void * data, unsigned int size);

        /*--------------------------------------------------------------------
         *  Methods from base class.
         *--------------------------------------------------------------------*/
        optix::Buffer       getOutputBuffer();
        void                cleanUp();
        void                doResize(unsigned int width, unsigned int height);
        void                initScene(InitialCameraData & cameraData);
        void                trace(const RayGenCameraData & cameraData);

#ifndef NDEBUG
    private:    // methods
        void                initDebug();
#endif  /* -----  end of #ifndef NDEBUG  ----- */

    private:    // attributes
        Renderer *         m_renderer;
        InitialCameraData  m_initialCameraData;
        RayGenCameraData   m_rayGenCameraData;
        optix::Group       m_rootObject;
        optix::Buffer      m_lightList;
        optix::Buffer      m_inputHeap;
        Index              m_inputHeapPointer;
};  /* -----  end of class Scene  ----- */
}   /* -----  end of namespace MaoPPM  ----- */

#endif  /* -----  #ifndef SCENE_H  ----- */
