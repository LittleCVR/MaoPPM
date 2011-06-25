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
#include    "payload.h"
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



Index Scene::copyToHeap(void * data, unsigned int size)
{
    do {
        // Check heap size.
        RTsize heapSize;
        m_inputHeap->getSize(heapSize);
        // Expand it if necessary.
        if (m_inputHeapPointer + size <= heapSize)
            break;
        else {
            void * dst = malloc(heapSize);
            void * src = m_inputHeap->map();
            memcpy(dst, src, heapSize);
            m_inputHeap->unmap();
            debug("\033[01;33mheap\033[00m expanded, size = \033[01;31m%u\033[00m.\n", 2 * heapSize);
            m_inputHeap->setSize(2 * heapSize);
            src = m_inputHeap->map();
            memcpy(src, dst, heapSize);
            m_inputHeap->unmap();
            free(dst);
        }
    } while (true);

    // Copy data.
    char * dst = static_cast<char *>(m_inputHeap->map());
    memcpy(dst + m_inputHeapPointer, data, size);
    m_inputHeap->unmap();

    // Returns address.
    unsigned int pos = m_inputHeapPointer;
    m_inputHeapPointer += size;
    return pos;
}   /* -----  end of method Scene::copyToHeap  ----- */



void Scene::doResize(unsigned int width, unsigned int height)
{
    m_renderer->resize(width, height);
}   /* -----  end of method Scene::doResize  ----- */



#ifndef NDEBUG
void Scene::initDebug()
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

    if (computeCapability.x < 1 || computeCapability.y < 1)
        debug("Compute capability is SM %d.%d, debug mode cannot be enabled.\n",
                computeCapability.x, computeCapability.y);
    else {
        debug("Compute capability is SM %d.%d, debug mode enabled.\n",
                computeCapability.x, computeCapability.y);
        context()->setPrintEnabled(true);
    }

    // some useful messages
    debug("sizeof(NormalRayPayload)     = \033[01;31m%4d\033[00m.\n", sizeof(NormalRayPayload));
    debug("sizeof(ShadowRayPayload)     = \033[01;31m%4d\033[00m.\n", sizeof(ShadowRayPayload));
    debug("sizeof(DifferentialGeometry) = \033[01;31m%4d\033[00m.\n", sizeof(DifferentialGeometry));
    debug("sizeof(Intersection)         = \033[01;31m%4d\033[00m.\n", sizeof(Intersection));
    debug("sizeof(BSDF)                 = \033[01;31m%4d\033[00m.\n", sizeof(BSDF));
}   /* -----  end of method Scene::initDebug  ----- */
#endif  /* -----  end of #ifndef NDEBUG  ----- */



void Scene::initScene(InitialCameraData & cameraData)
{
#ifndef NDEBUG
    initDebug();
#endif  /* -----  end of #ifndef NDEBUG  ----- */

    debug("Set stack size to %u.\n", DEFAULT_STACK_SIZE);
    context()->setStackSize(DEFAULT_STACK_SIZE);
    context()->setRayTypeCount(N_RAY_TYPES);

    debug("Default ray epsilon is %f.\n", DEFAULT_RAY_EPSILON);
    context()["rayEpsilon"]->setFloat(DEFAULT_RAY_EPSILON);

    // Initialize heap.
    m_inputHeapPointer = 0;
    m_inputHeap = context()->createBuffer(
            RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL, RT_FORMAT_BYTE);
    m_inputHeap->setSize(DEFAULT_INPUT_HEAP_SIZE);
    context()["inputHeap"]->set(m_inputHeap);

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
    // Camera.
    m_rayGenCameraData = cameraData;
    // Render.
    m_renderer->render(cameraData);
}   /* -----  end of method Scene::trace  ----- */
