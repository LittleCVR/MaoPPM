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



PathTracingRenderer::PathTracingRenderer(Scene * scene) : Renderer(scene),
    m_maxRayDepth(DEFAULT_MAX_RAY_DEPTH)
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

    debug("sizeof(SamplePoint) = \033[01;31m%4d\033[00m.\n", sizeof(SamplePoint));

    // buffers
    m_samplePointList = context()->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_USER);
    m_samplePointList->setElementSize(sizeof(SamplePoint));
    m_samplePointList->setSize(width() * height() * m_maxRayDepth);
    context()["samplePointList"]->set(m_samplePointList);

    m_pathCountList = context()->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_UNSIGNED_INT);
    m_pathCountList->setSize(width() * height() * m_maxRayDepth);
    context()["pathCountList"]->set(m_pathCountList);

    m_radianceList = context()->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT3);
    m_radianceList->setSize(width() * height() * m_maxRayDepth);
    context()["radianceList"]->set(m_radianceList);

    context()["maxRayDepth"]->setUint(m_maxRayDepth);

    context()->setEntryPointCount(N_PASSES);
    setExceptionProgram(CleaningPass);
    setExceptionProgram(TracingPass);
    setExceptionProgram(SummingPass);
    setRayGenerationProgram(CleaningPass, "PathTracingRenderer.cu", "clear");
    setRayGenerationProgram(TracingPass,  "PathTracingRenderer.cu", "trace");
    setRayGenerationProgram(SummingPass,  "PathTracingRenderer.cu", "sum");
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
    nSamplesPerThread = 1 + 3 * m_maxRayDepth;
    generateSamples(nSamplesPerThread * launchSize.x * launchSize.y);
    context()["nSamplesPerThread"]->setUint(nSamplesPerThread);
    context()->launch(CleaningPass, width(), height());
    context()->launch(TracingPass,  width(), height());
    context()->launch(SummingPass,  width(), height());

    // Add frame count.
    m_frame++;
}   /* -----  end of method PathTracingRenderer::render  ----- */



void PathTracingRenderer::resize(unsigned int width, unsigned int height)
{
    Renderer::resize(width, height);

    m_samplePointList->setSize(width * height * m_maxRayDepth);
    debug("\033[01;33msamplePointList\033[00m resized to: \033[01;31m%10u\033[00m.\n",
            sizeof(SamplePoint) * m_maxRayDepth * width * height);

    m_pathCountList->setSize(width * height * m_maxRayDepth);
    debug("\033[01;33mpathCountList\033[00m resized to: \033[01;31m%10u\033[00m.\n",
            sizeof(unsigned int) * m_maxRayDepth * width * height);

    m_demandLocalHeapSize = width * height *
            m_maxRayDepth * sizeof(Intersection);
    setLocalHeapSize(m_demandLocalHeapSize);
}   /* -----  end of method PathTracingRenderer::resize  ----- */



void PathTracingRenderer::parseArguments(vector<char *> argumentList)
{
    int argc = argumentList.size();
    for (vector<char *>::iterator it = argumentList.begin();
         it != argumentList.end(); ++it)
    {
        std::string arg(*it);
        if (arg == "--max-ray-depth") {
            if (++it != argumentList.end()) {
                m_maxRayDepth = atoi(*it);
                cerr << "Set max ray depth to " << m_maxRayDepth << endl;
            } else {
                std::cerr << "Missing argument to " << arg << std::endl;
                exit(EXIT_FAILURE);
            }
        }
        // otherwise
        else {
            std::cerr << "Unknown option: '" << arg << std::endl;
            exit(EXIT_FAILURE);
        }
    }
}   /* -----  end of function PathTracingRenderer::parseArguments  ----- */



void PathTracingRenderer::printUsageAndExit(bool doExit)
{
    std::cerr
        << "BDPT options:" << std::endl
        << "     | --max-ray-depth <int>  Set IGPPM to use N importons per thread."               << std::endl
        << std::endl;

    if (doExit)
        exit(EXIT_FAILURE);
}   /* -----  end of function PathTracingRenderer::printUsageAndExit  ----- */
