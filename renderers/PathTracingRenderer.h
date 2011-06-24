/*
 * =============================================================================
 *
 *       Filename:  PathTracingRenderer.h
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

#ifndef PATH_TRACING_RENDERER_H
#define PATH_TRACING_RENDERER_H

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
 *        Class:  PathTracingRenderer
 *  Description:  
 * =============================================================================
 */
class PathTracingRenderer : public Renderer {
    public:     // methods
        PathTracingRenderer(Scene * scene = NULL);
        ~PathTracingRenderer();

    public:
        static const unsigned int  N_PASSES  = 1;
        enum Pass { PathTracingPass };

    public:     // methods
        void    init();
        void    resize(unsigned int width, unsigned int height);
        void    render(const Scene::RayGenCameraData & cameraData);

    private:    // attributes
        unsigned int    m_frame;
};  /* -----  end of class PathTracingRenderer  ----- */
}   /* -----  end of namespace MaoPPM  ----- */

#endif  /* -----  #ifndef PATH_TRACING_RENDERER_H  ----- */
