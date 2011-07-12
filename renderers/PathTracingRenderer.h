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
#include    "particle.h"
#include    "BSDF.h"
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
        static const unsigned int  N_PASSES  = Renderer::N_PASSES + 3;
        enum Pass {
            CleaningPass  = Renderer::UserPass + 0,
            TracingPass   = Renderer::UserPass + 1,
            SummingPass   = Renderer::UserPass + 2
        };

        class SamplePoint : public HitPoint {
            public:
                optix::float3  throughput;
                optix::float3  wo;

                __device__ __forceinline__ void reset()
                {
                    HitPoint::reset();
                    throughput = optix::make_float3(0.0f);
                }

                __device__ __forceinline__ BSDF * bsdf()
                {
                    return reinterpret_cast<BSDF *>(m_bsdf);
                }
                __device__ __forceinline__ void setBSDF(const BSDF & b)
                {
                    *bsdf() = b;
                }

            private:
                char           m_bsdf[sizeof(BSDF)];
        };

    public:     // methods
        void    init();
        void    resize(unsigned int width, unsigned int height);
        void    render(const Scene::RayGenCameraData & cameraData);

    private:    // attributes
        unsigned int    m_frame;
        unsigned int    m_demandLocalHeapSize;
        unsigned int    m_maxRayDepth;
        optix::Buffer   m_samplePointList;
        optix::Buffer   m_pathCountList;
        optix::Buffer   m_radianceList;
};  /* -----  end of class PathTracingRenderer  ----- */
}   /* -----  end of namespace MaoPPM  ----- */

#endif  /* -----  #ifndef PATH_TRACING_RENDERER_H  ----- */
