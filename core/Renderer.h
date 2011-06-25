/*
 * =============================================================================
 *
 *       Filename:  Renderer.h
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  2011-06-08 14:31:22
 *
 *         Author:  Chun-Wei Huang (LittleCVR), 
 *        Company:  Communication & Multimedia Laboratory,
 *                  Department of Computer Science & Information Engineering,
 *                  National Taiwan University
 *
 * =============================================================================
 */

#ifndef RENDERER_H
#define RENDERER_H

/*----------------------------------------------------------------------------
 *  header files from OptiX
 *----------------------------------------------------------------------------*/
#include    <optix_world.h>

/*----------------------------------------------------------------------------
 *  header files of our own
 *----------------------------------------------------------------------------*/
#include    "global.h"
#include    "Scene.h"



namespace MaoPPM {
/*
 * =============================================================================
 *        Class:  Renderer
 *  Description:  
 * =============================================================================
 */
class Renderer {
    public:
        Renderer(Scene * scene = NULL);
        ~Renderer();

    public:
        inline optix::Buffer  outputBuffer() { return m_outputBuffer; }
        inline unsigned int   width() const { return m_width; }
        inline unsigned int   height() const { return m_height; }
        inline Scene *        scene() const { return m_scene; }
        void                  setScene(Scene * scene);

        virtual void          init();
        virtual void          render(const Scene::RayGenCameraData & cameraData) = 0;
        virtual void          resize(unsigned int width, unsigned int height);

    protected:  // methods
        /*
         *----------------------------------------------------------------------
         *       Class:  Renderer
         *      Method:  Renderer :: context
         * Description:  Convenient method to get scene's context.
         *----------------------------------------------------------------------
         */
        optix::Context  context();
        /*
         *----------------------------------------------------------------------
         *       Class:  Renderer
         *      Method:  Renderer :: sampleList
         * Description:  Derived classes could use this method to get sample
         *               list. And use generateSamples(...) to generate randon
         *               samples.
         *----------------------------------------------------------------------
         */
        optix::Buffer  sampleList() const { return m_sampleList; }
        /*
         *----------------------------------------------------------------------
         *       Class:  Renderer
         *      Method:  Renderer :: generateSamples
         * Description:  Derived classes could use this method to generate
         *               random samples, and use sampleList() to get the
         *               generated sample list.
         *----------------------------------------------------------------------
         */
        void  generateSamples(const optix::uint nSamples);

        Index localHeapPointer();
        void  resetLocalHeapPointer();
        void  setLocalHeapPointer(const Index & index);

        void  setExceptionProgram(unsigned int entryPointIndex,
                const std::string & cuFileName = "exception.cu",
                const std::string & exceptionProgramName = "handleException");
        void  setMissProgram(unsigned int rayType,
                const std::string & cuFileName,
                const std::string & missProgramName);
        void  setRayGenerationProgram(unsigned int entryPointIndex,
                const std::string & cuFileName,
                const std::string & rayGenerationProgramName  = "generateRay");

    private:    // attributes
        Scene *        m_scene;
        unsigned int   m_width;         /* screen width */
        unsigned int   m_height;        /* screen height */
        optix::Buffer  m_outputBuffer;
        optix::Buffer  m_sampleList;
        optix::Buffer  m_localHeap;
        optix::Buffer  m_localHeapPointer;
};  /* -----  end of class Renderer  ----- */
}   /* -----  end of namespace MaoPPM  ----- */



#endif  /* -----  #ifndef RENDERER_H  ----- */
