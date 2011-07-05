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

    // buffers
    m_samplePointList = context()->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_USER);
    m_samplePointList->setElementSize(sizeof(SamplePoint));
    m_samplePointList->setSize(width() * height() * DEFAULT_MAX_RAY_DEPTH);
    context()["samplePointList"]->set(m_samplePointList);

    m_pathCountList = context()->createBuffer(RT_BUFFER_OUTPUT, RT_FORMAT_UNSIGNED_INT);
    m_pathCountList->setSize(width() * height() * DEFAULT_MAX_RAY_DEPTH);
    context()["pathCountList"]->set(m_pathCountList);

    context()["maxRayDepth"]->setUint(DEFAULT_MAX_RAY_DEPTH);

    context()->setEntryPointCount(N_PASSES);
    setExceptionProgram(PathTracingPass);
    setRayGenerationProgram(PathTracingPass, "PathTracingRenderer.cu", "trace");
    setMissProgram(NormalRay, "ray.cu", "handleNormalRayMiss");

    Renderer::postInit();
}   /* -----  end of method PathTracingRenderer::init  ----- */



void PathTracingRenderer::render(const Scene::RayGenCameraData & cameraData)
{
    if (scene()->isCameraChanged()) {
        scene()->setIsCameraChanged(false);
        m_frame = 0;
    }

    uint  nSamplesPerThread = 0;
    uint2 launchSize = make_uint2(0, 0);

    context()["frameCount"]->setUint(m_frame);

    // Launch path tracing pass.
    setLocalHeapPointer(0);
    launchSize = make_uint2(width(), height());
    context()["launchSize"]->setUint(launchSize.x, launchSize.y);
    nSamplesPerThread = 1 + 3 * DEFAULT_MAX_RAY_DEPTH;
    generateSamples(nSamplesPerThread * launchSize.x * launchSize.y);
    context()["nSamplesPerThread"]->setUint(nSamplesPerThread);
    context()->launch(PathTracingPass, width(), height());

    // Add frame count.
    m_frame++;
}   /* -----  end of method PathTracingRenderer::render  ----- */



void PathTracingRenderer::resize(unsigned int width, unsigned int height)
{
    Renderer::resize(width, height);

    m_samplePointList->setSize(width * height * DEFAULT_MAX_RAY_DEPTH);
    debug("\033[01;33msamplePointList\033[00m resized to: \033[01;31m%10u\033[00m.\n",
            sizeof(SamplePoint) * DEFAULT_MAX_RAY_DEPTH * width * height);

    m_pathCountList->setSize(width * height * DEFAULT_MAX_RAY_DEPTH);
    debug("\033[01;33mpathCountList\033[00m resized to: \033[01;31m%10u\033[00m.\n",
            sizeof(unsigned int) * DEFAULT_MAX_RAY_DEPTH * width * height);

    m_demandLocalHeapSize = width * height *
            DEFAULT_MAX_RAY_DEPTH * sizeof(Intersection);
    setLocalHeapSize(m_demandLocalHeapSize);
}   /* -----  end of method PathTracingRenderer::resize  ----- */
