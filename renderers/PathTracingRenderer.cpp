/*
 * =============================================================================
 *
 *       Filename:  PathTracingRenderer.cpp
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  2011-06-08 19:09:58
 *
 *         Author:  Chun-Wei Huang (LittleCVR), 
 *        Company:  Communication & Multimedia Laboratory,
 *                  Department of Computer Science & Information Engineering,
 *                  National Taiwan University
 *
 * =============================================================================
 */

#include    "PathTracingRenderer.h"

/*----------------------------------------------------------------------------
 *  header files from std C/C++
 *----------------------------------------------------------------------------*/
#include    <algorithm>
#include    <cstdio>
#include    <ctime>
#include    <iostream>
#include    <limits>

/*----------------------------------------------------------------------------
 *  header files of our own
 *----------------------------------------------------------------------------*/
#include    "payload.h"
#include    "Scene.h"

/*----------------------------------------------------------------------------
 *  using namespaces
 *----------------------------------------------------------------------------*/
using namespace std;
using namespace optix;
using namespace MaoPPM;



PathTracingRenderer::PathTracingRenderer(Scene * scene) : Renderer(scene)
{
    /* EMPTY */
}   /* -----  end of method PathTracingRenderer::PathTracingRenderer  ----- */



PathTracingRenderer::~PathTracingRenderer()
{
    /* EMPTY */
}   /* -----  end of method PathTracingRenderer::~PathTracingRenderer  ----- */



void PathTracingRenderer::init()
{
    Renderer::preInit();

    context()["maxRayDepth"]->setUint(DEFAULT_MAX_RAY_DEPTH);

    // Do not need this.
    setLocalHeapSize(0);

    context()->setEntryPointCount(N_PASSES);
    setExceptionProgram(PathTracingPass);
    setRayGenerationProgram(PathTracingPass, "PathTracingRenderer.cu", "trace");
    setMissProgram(NormalRay, "ray.cu", "handleNormalRayMiss");

    Renderer::postInit();
}   /* -----  end of method PathTracingRenderer::init  ----- */



void PathTracingRenderer::render(const Scene::RayGenCameraData & cameraData)
{
    resetLocalHeapPointer();

    if (scene()->isCameraChanged()) {
        scene()->setIsCameraChanged(false);
        m_frame = 0;
    }

    // Launch path tracing pass.
    generateSamples(3 * width() * height() * DEFAULT_MAX_RAY_DEPTH);
    context()["nSamplesPerThread"]->setUint(3 * DEFAULT_MAX_RAY_DEPTH);
    context()["frameCount"]->setUint(m_frame);
    context()["launchSize"]->setUint(width(), height());
    context()->launch(PathTracingPass, width(), height());

    // Add frame count.
    m_frame++;
}   /* -----  end of method PathTracingRenderer::render  ----- */



void PathTracingRenderer::resize(unsigned int width, unsigned int height)
{
    Renderer::resize(width, height);
}   /* -----  end of method PathTracingRenderer::resize  ----- */
