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



void PathTracingRenderer::render(const Scene::RayGenCameraData & cameraData)
{
    if (scene()->isCameraChanged()) {
        scene()->setIsCameraChanged(false);
        context()["cameraPosition"]->setFloat(cameraData.eye);
        context()["cameraU"]->setFloat(cameraData.U);
        context()["cameraV"]->setFloat(cameraData.V);
        context()["cameraW"]->setFloat(cameraData.W);
    }

    context()["launchSize"]->setUint(width(), height());
    context()->launch(PathTracingPass, width(), height());
}   /* -----  end of method PathTracingRenderer::render  ----- */



void PathTracingRenderer::setMaterialPrograms(const std::string & name,
        optix::Material & material)
{
    if (name == "matte") {
        string ptxPath = scene()->ptxpath("MaoPPM", "matteMaterialPathTracingPrograms.cu");
        material->setClosestHitProgram(RadianceRay,
                scene()->getContext()->createProgramFromPTXFile(ptxPath, "handleRadianceRayClosestHit"));
        material->setAnyHitProgram(ShadowRay,
                scene()->getContext()->createProgramFromPTXFile(ptxPath, "handleShadowRayAnyHit"));
    }
    else if (name == "plastic") {
    }
}   /* -----  end of method PathTracingRenderer::setMaterialPrograms  ----- */
