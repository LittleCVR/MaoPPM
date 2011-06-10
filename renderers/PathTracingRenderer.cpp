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
#include    "Scene.h"
#include    "SceneBuilder.h"

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
    Renderer::init();
    setEntryPointPrograms("pathTracingPassPrograms.cu", PathTracingPass);
    setMissProgram("pathTracingPassPrograms.cu", RadianceRay, "handleRadianceRayMiss");
}   /* -----  end of method PathTracingRenderer::init  ----- */



void PathTracingRenderer::render()
{
    // For convenience.
    Scene::RayGenCameraData & cameraData = scene()->m_rayGenCameraData;

    if (scene()->isCameraChanged()) {
        scene()->setIsCameraChanged(false);
        getContext()["cameraPosition"]->setFloat(cameraData.eye);
        getContext()["cameraU"]->setFloat(cameraData.U);
        getContext()["cameraV"]->setFloat(cameraData.V);
        getContext()["cameraW"]->setFloat(cameraData.W);
    }

    getContext()["launchSize"]->setUint(width(), height());
    getContext()->launch(PathTracingPass, width(), height());
}   /* -----  end of method PathTracingRenderer::render  ----- */
