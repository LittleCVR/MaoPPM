/*
 * =====================================================================================
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
 * =====================================================================================
 */

#ifndef RENDERER_H
#define RENDERER_H

/*-----------------------------------------------------------------------------
 *  header files from OptiX
 *-----------------------------------------------------------------------------*/
#include    <optix_world.h>

/*-----------------------------------------------------------------------------
 *  header files of our own
 *-----------------------------------------------------------------------------*/
#include    "global.h"



namespace MaoPPM {
/*
 * =====================================================================================
 *        Class:  Renderer
 *  Description:  
 * =====================================================================================
 */
class Renderer {
    public:
        Renderer(Scene * scene = NULL);
        ~Renderer();

    public:
        inline optix::Buffer    getOutputBuffer() { return m_outputBuffer; }
        inline unsigned int     width() const { return m_width; }
        inline unsigned int     height() const { return m_height; }
        inline Scene *          scene() const { return m_scene; }
        void                    setScene(Scene * scene);

        virtual void            init();
        virtual void            render() = 0;
        virtual void            resize(unsigned int width, unsigned int height);

#ifdef DEBUG
    private:    // methods
        void                    initDebug();
#endif

    protected:  // methods
        optix::Context          getContext();
        optix::Buffer           sampleList() const { return m_sampleList; }
        void                    generateSamples(const optix::uint nSamples);

        void setPrograms(const std::string & cuFileName,
                unsigned int entryPointIndex,
                const std::string & rayGenerationProgramName  = "generateRay",
                const std::string & missProgramName           = "handleMiss",
                const std::string & exceptionProgramName      = "handleException");

    private:    // attributes
        Scene *                 m_scene;
        unsigned int            m_width;                /* screen width */
        unsigned int            m_height;               /* screen height */
        optix::Buffer           m_outputBuffer;
        optix::Buffer           m_sampleList;
};  /* -----  end of class Renderer  ----- */
}   /* -----  end of namespace MaoPPM  ----- */



#endif  /* -----  #ifndef RENDERER_H  ----- */