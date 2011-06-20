/*
 * =============================================================================
 *
 *       Filename:  Scene.cpp
 *
 *    Description:  Scene class implementation.
 *
 *        Version:  1.0
 *        Created:  2011-03-23 14:47:34
 *
 *         Author:  Chun-Wei Huang (LittleCVR), 
 *        Company:  Communication & Multimedia Laboratory,
 *                  Department of Computer Science & Information Engineering,
 *                  National Taiwan University
 *
 * =============================================================================
 */

#include    "Scene.h"

/*-----------------------------------------------------------------------------
 *  header files from std C/C++
 *-----------------------------------------------------------------------------*/
#include    <cstring>

/*----------------------------------------------------------------------------
 *  header files of our own
 *----------------------------------------------------------------------------*/
#include    "Light.h"
#include    "Renderer.h"
#include    "SceneBuilder.h"

/*----------------------------------------------------------------------------
 *  using namespaces
 *----------------------------------------------------------------------------*/
using namespace std;
using namespace optix;
using namespace MaoPPM;



Scene::Scene() : SampleScene()
{
    /* EMPTY */
}   /* -----  end of method Scene::Scene  ----- */



Scene::~Scene()
{
    /* EMPTY */
}   /* -----  end of method Scene::~Scene  ----- */



Context Scene::context()
{
    return getContext();
}   /* -----  end of method Scene::context  ----- */



Buffer Scene::getOutputBuffer()
{
    return m_renderer->outputBuffer();
}   /* -----  end of method Scene::getOutputBuffer  ----- */



void Scene::setRenderer(Renderer * renderer)
{
    if (m_renderer == renderer)
        return;

    m_renderer = renderer;
    m_renderer->setScene(this);
}   /* -----  end of method Scene::setRenderer  ----- */



void Scene::cleanUp()
{
    SampleScene::cleanUp();
}   /* -----  end of method Scene::cleanUp  ----- */



HeapIndex Scene::copyToHeap(void * data, unsigned int size)
{
    do {
        // Check heap size.
        RTsize heapSize;
        m_heap->getSize(heapSize);
        // Expand it if necessary.
        if (m_heapPointer + size <= heapSize)
            break;
        else {
            void * dst = malloc(heapSize);
            void * src = m_heap->map();
            memcpy(dst, src, heapSize);
            m_heap->unmap();
            m_heap->setSize(2 * heapSize);
            src = m_heap->map();
            memcpy(src, dst, heapSize);
            m_heap->unmap();
            free(dst);
        }
    } while (true);

    // Copy data.
    char * dst = static_cast<char *>(m_heap->map());
    memcpy(dst + m_heapPointer, data, size);
    m_heap->unmap();

    // Returns address.
    unsigned int pos = m_heapPointer;
    m_heapPointer += size;
    return pos;
}   /* -----  end of method Scene::copyToHeap  ----- */



void Scene::doResize(unsigned int width, unsigned int height)
{
    m_renderer->resize(width, height);
}   /* -----  end of method Scene::doResize  ----- */



void Scene::initScene(InitialCameraData & cameraData)
{
    context()->setStackSize(DEFAULT_STACK_SIZE);
    context()->setRayTypeCount(m_renderer->nRayTypes());
    context()->setEntryPointCount(m_renderer->nPasses());

    context()["rayEpsilon"]->setFloat(DEFAULT_RAY_EPSILON);

    // Initialize heap.
    m_heapPointer = 0;
    m_heap = context()->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_BYTE);
    m_heap->setSize(DEFAULT_HEAP_SIZE);
    context()["heap"]->set(m_heap);

    // Initialize root object.
    m_rootObject = context()->createGeometryGroup();
    context()["rootObject"]->set(m_rootObject);
    // Acceleration.
    Acceleration acceleration = context()->createAcceleration("Bvh", "Bvh");
    m_rootObject->setAcceleration(acceleration);

    // Initialize light buffer.
    m_lightList = context()->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER);
    m_lightList->setElementSize(sizeof(Light));
    m_lightList->setSize(0);
    context()["lightList"]->set(m_lightList);

    // Set camera data.
    SceneBuilder sceneBuilder;
    sceneBuilder.parse(this);
    cameraData = m_initialCameraData;

    // Initialize renderer.
    m_renderer->init();
}   /* -----  end of method Scene::initScene  ----- */



void Scene::trace(const RayGenCameraData & cameraData)
{
    m_rayGenCameraData = cameraData;
    m_renderer->render(cameraData);
}   /* -----  end of method Scene::trace  ----- */
