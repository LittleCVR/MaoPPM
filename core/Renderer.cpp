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
    m_scene(NULL), m_width(DEFAULT_WIDTH), m_height(DEFAULT_HEIGHT)
{
    setScene(scene);
}   /* -----  end of method Renderer::Renderer  ----- */



Renderer::~Renderer()
{
    /* EMPTY */
}   /* -----  end of method Renderer::~Renderer  ----- */



void Renderer::init()
{
    // create output buffer
    m_outputBuffer = scene()->createOutputBuffer(RT_FORMAT_FLOAT4, width(), height());
    context()["outputBuffer"]->set(m_outputBuffer);

    // initialize sample buffer
    m_sampleList = context()->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT);
    m_sampleList->setSize(0);
    context()["sampleList"]->set(m_sampleList);

    // Local heap for GPU.
    /* TODO: remove hard coded size */
    m_localHeap = context()->createBuffer(
            RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_BYTE, 512 * 1024 * 1024);
    context()["localHeap"]->set(m_localHeap);
    m_localHeapPointer = context()->createBuffer(
            RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_USER, 1);
    m_localHeapPointer->setElementSize(sizeof(Index));
    context()["localHeapPointer"]->set(m_localHeapPointer);
}   /* -----  end of method PPMRenderer::initPPMRenderer  ----- */



void Renderer::generateSamples(const uint nSamples)
{
    // For convenience.
    Buffer & sampleList = m_sampleList;

    // Check size.
    RTsize N = 0;
    sampleList->getSize(N);
    // Expande if necessary.
    if (N < nSamples) {
        sampleList->setSize(nSamples);
        debug("\033[01;33msampleList\033[00m expanded, size = \033[01;31m%u\033[00m.\n", sizeof(float) * nSamples);
    }

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
    debug("window resized to: \033[01;31m%u\033[00m x \033[01;31m%u\033[00m.\n", width, height);
    m_width = width; m_height = height;
    debug("\033[01;33moutputBuffer\033[00m resized to: \033[01;31m%u\033[00m.\n",
            sizeof(float4) * width * height);
    m_outputBuffer->setSize(width, height);
}   /* -----  end of method Renderer::resize  ----- */



Index Renderer::localHeapPointer()
{
    Index * localHeapPointer = static_cast<Index *>(m_localHeapPointer->map());
    Index index = localHeapPointer[0];
    m_localHeapPointer->unmap();
    return index;
}



void Renderer::resetLocalHeapPointer()
{
    setLocalHeapPointer(0);
}



void Renderer::setLocalHeapPointer(const Index & index)
{
    static bool firstTimeEntered = true;
    Index * localHeapPointer = static_cast<Index *>(m_localHeapPointer->map());
    if (firstTimeEntered) localHeapPointer[0] = 0;
    debug("\033[01;33mlocalHeapPointer\033[00m at: \033[01;31m%u\033[00m, set to: \033[01;31m%u\033[00m\n",
            localHeapPointer[0], index);
    localHeapPointer[0] = index;
    m_localHeapPointer->unmap();
    firstTimeEntered = false;
}



void Renderer::setRayGenerationProgram(unsigned int entryPointIndex,
        const std::string & cuFileName,
        const std::string & rayGenerationProgramName)
{
    std::string ptxPath = scene()->ptxpath("MaoPPM", cuFileName);
    Program rayGenerationProgram = context()->createProgramFromPTXFile(ptxPath, rayGenerationProgramName);
    context()->setRayGenerationProgram(entryPointIndex, rayGenerationProgram);
}   /* -----  end of method Renderer::setEntryPointPrograms  ----- */



void Renderer::setExceptionProgram(unsigned int entryPointIndex,
        const std::string & cuFileName,
        const std::string & exceptionProgramName)
{
    std::string ptxPath = scene()->ptxpath("MaoPPM", cuFileName);
    Program exceptionProgram = context()->createProgramFromPTXFile(ptxPath, exceptionProgramName);
    context()->setExceptionProgram(entryPointIndex, exceptionProgram);
}   /* -----  end of method Renderer::setEntryPointPrograms  ----- */



void Renderer::setMissProgram(unsigned int rayType,
        const std::string & cuFileName, const std::string & missProgramName)
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
