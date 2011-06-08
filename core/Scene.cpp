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

/*----------------------------------------------------------------------------
 *  header files of our own
 *----------------------------------------------------------------------------*/
#include    "Renderer.h"
#include    "SceneBuilder.h"

/*----------------------------------------------------------------------------
 *  using namespaces
 *----------------------------------------------------------------------------*/
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



Buffer Scene::getOutputBuffer()
{
    return m_renderer->getOutputBuffer();
}   /* -----  end of method Scene::getOutputBuffer  ----- */



void Scene::setRenderer(Renderer * renderer)
{
    if (m_renderer == renderer)
        return;

    m_renderer = renderer;
    m_renderer->setScene(this);
}   /* -----  end of method Scene::setRenderer  ----- */



void Scene::doResize(unsigned int width, unsigned int height)
{
    m_renderer->resize(width, height);
}   /* -----  end of method Scene::doResize  ----- */



void Scene::initScene(InitialCameraData & cameraData)
{
    getContext()->setRayTypeCount(nRayTypes);
    getContext()->setEntryPointCount(nPasses);

    // Initialize root object.
    m_rootObject = getContext()->createGeometryGroup();
    getContext()["rootObject"]->set(m_rootObject);
    // Acceleration.
    Acceleration acceleration = getContext()->createAcceleration("Bvh", "Bvh");
    m_rootObject->setAcceleration(acceleration);

    // Initialize light buffer.
    m_lightList = getContext()->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER);
    m_lightList->setElementSize(sizeof(Light));
    m_lightList->setSize(0);
    getContext()["lightList"]->set(m_lightList);

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
    m_renderer->render();
}   /* -----  end of method Scene::trace  ----- */
