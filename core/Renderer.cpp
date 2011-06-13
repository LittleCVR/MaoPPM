/*
 * =============================================================================
 *
 *       Filename:  Renderer.cpp
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

#include    "Renderer.h"

/*----------------------------------------------------------------------------
 *  header files from std
 *----------------------------------------------------------------------------*/
#include    <cstdlib>
#include    <iostream>

/*----------------------------------------------------------------------------
 *  header files of our own
 *----------------------------------------------------------------------------*/
#include    "Scene.h"

/*----------------------------------------------------------------------------
 *  using namespaces
 *----------------------------------------------------------------------------*/
using namespace std;
using namespace optix;
using namespace MaoPPM;



Renderer::Renderer(Scene * scene) :
    m_scene(NULL), m_width(INITIAL_WIDTH), m_height(INITIAL_HEIGHT)
{
    setScene(scene);
}   /* -----  end of method Renderer::Renderer  ----- */



Renderer::~Renderer()
{
    /* EMPTY */
}   /* -----  end of method Renderer::~Renderer  ----- */



void Renderer::init()
{
#ifdef DEBUG
    initDebug();
#endif

    context()->setStackSize(STACK_SIZE);

    // create output buffer
    m_outputBuffer = scene()->createOutputBuffer(RT_FORMAT_FLOAT4, m_width, m_height);
    context()["outputBuffer"]->set(m_outputBuffer);

    // initialize sample buffer
    m_sampleList = context()->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT);
    m_sampleList->setSize(0);
    context()["sampleList"]->set(m_sampleList);
}   /* -----  end of method PPMRenderer::initPPMRenderer  ----- */



#ifdef DEBUG
void Renderer::initDebug()
{
    // Get device compute capability and determine if we can enable the
    // rtPrintf functionality. Because rtPrintf cannot be enabled for devices
    // that have compute capability less than SM11.
    int2 computeCapability;

    RTresult rc = rtDeviceGetAttribute(0,                 // first device
            RT_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY,
            sizeof(computeCapability),
            reinterpret_cast<void *>(&computeCapability));
    if (rc != RT_SUCCESS)
        throw Exception::makeException(rc, context()->get());

    cerr << "Compute capability is SM" << computeCapability.x << computeCapability.y << ", ";
    if (computeCapability.x < 1 || computeCapability.y < 1)
        cerr << "debug mode cannot be enabled." << endl;
    else {
        cerr << "debug mode enabled." << endl;
        context()->setPrintEnabled(true);
    }
}   /* -----  end of method Renderer::initDebug  ----- */
#endif



void Renderer::generateSamples(const uint nSamples)
{
    // For convenience.
    Buffer & sampleList = m_sampleList;

    // Check size.
    RTsize N = 0;
    sampleList->getSize(N);
    // Expande if necessary.
    if (N < nSamples)
        sampleList->setSize(nSamples);

    // Generate samples.
    float * sampleListPtr = static_cast<float *>(sampleList->map());
    for (uint i = 0; i < nSamples; i++)
        sampleListPtr[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    sampleList->unmap();
}   /* -----  end of method Renderer::generateSamples  ----- */



Context Renderer::context()
{
    return scene()->getContext();
}   /* -----  enf of method Renderer::context  ----- */



void Renderer::resize(unsigned int width, unsigned int height)
{
    m_width = width; m_height = height;
    m_outputBuffer->setSize(m_width, m_height);
}   /* -----  end of method Renderer::resize  ----- */



void Renderer::setEntryPointPrograms(const std::string & cuFileName,
        unsigned int entryPointIndex,
        const std::string & rayGenerationProgramName,
        const std::string & exceptionProgramName)
{
    std::string ptxPath = scene()->ptxpath("MaoPPM", cuFileName);

    Program rayGenerationProgram = context()->createProgramFromPTXFile(ptxPath, rayGenerationProgramName);
    context()->setRayGenerationProgram(entryPointIndex, rayGenerationProgram);

    Program exceptionProgram = context()->createProgramFromPTXFile(ptxPath, exceptionProgramName);
    context()->setExceptionProgram(entryPointIndex, exceptionProgram);
}   /* -----  end of method Renderer::setEntryPointPrograms  ----- */



void Renderer::setMissProgram(const std::string & cuFileName,
        unsigned int rayType,
        const std::string & missProgramName)
{
    std::string ptxPath = scene()->ptxpath("MaoPPM", cuFileName);

    Program missProgram = context()->createProgramFromPTXFile(ptxPath, missProgramName);
    context()->setMissProgram(rayType, missProgram);
}   /* -----  end of method Renderer::setMissProgram */



void Renderer::setScene(Scene * scene)
{
    if (m_scene == scene)
        return;

    m_scene = scene;
    m_scene->setRenderer(this);
}   /* -----  end of method Renderer::setScene  ----- */
