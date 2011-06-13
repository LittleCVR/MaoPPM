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
    friend class Renderer;
    friend class SceneBuilder;

    public:
        Scene();
        ~Scene();

        optix::Buffer       getOutputBuffer();
        inline bool         isCameraChanged() const { return _camera_changed; }
        inline void         setIsCameraChanged(bool isCameraChanged) { _camera_changed = isCameraChanged; }
        inline Renderer *   renderer() const { return m_renderer; }
        void                setRenderer(Renderer * renderer);

        void                doResize(unsigned int width, unsigned int height);
        void                initScene(InitialCameraData & cameraData);
        void                trace(const RayGenCameraData & cameraData);

    private:    // attributes
        Renderer *              m_renderer;
        InitialCameraData       m_initialCameraData;
        RayGenCameraData        m_rayGenCameraData;
        optix::GeometryGroup    m_rootObject;
        optix::Buffer           m_lightList;
};  /* -----  end of class Scene  ----- */
}   /* -----  end of namespace MaoPPM  ----- */



#endif  /* -----  #ifndef SCENE_H  ----- */
