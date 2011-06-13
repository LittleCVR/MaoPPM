/*
 * =============================================================================
 *
 *       Filename:  PPMRenderer.cpp
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  2011-06-08 12:00:58
 *
 *         Author:  Chun-Wei Huang (LittleCVR), 
 *        Company:  Communication & Multimedia Laboratory,
 *                  Department of Computer Science & Information Engineering,
 *                  National Taiwan University
 *
 * =============================================================================
 */

#include    "PPMRenderer.h"

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



PPMRenderer::PPMRenderer(Scene * scene) : Renderer(scene)
{
    /* EMPTY */
}   /* -----  end of method PPMRenderer::PPMRenderer  ----- */



PPMRenderer::~PPMRenderer()
{
    /* EMPTY */
}   /* -----  end of method PPMRenderer::~PPMRenderer  ----- */



void PPMRenderer::buildPhotonMapAcceleration(MaoPPM::Photon * photonList,
        uint start, uint end, MaoPPM::Photon * photonMap,
        uint root, float3 bbMin, float3 bbMax)
{
    if (end - start == 0) {
        // Make a fake photon.
        photonMap[root].axis = PHOTON_NULL;
        photonMap[root].flux = make_float3(0.0f);
        return;
    }
    if (end - start == 1) {
        // Create a leaf photon.
        photonList[start].axis = PHOTON_LEAF;
        photonMap[root] = photonList[start];
        return;
    }

    // Choose the longest axis.
    uint axis = 0;
    float3 bbDiff = bbMax - bbMin;
    if (bbDiff.x > bbDiff.y) {
        if (bbDiff.x > bbDiff.z)
            axis = AXIS_X;
        else
            axis = AXIS_Z;
    } else {
        if (bbDiff.y > bbDiff.z)
            axis = AXIS_Y;
        else
            axis = AXIS_Z;
    }

    // Partition the photon list.
    uint median = (start + end) / 2;
    if (axis == AXIS_X)
        nth_element(&photonList[start], &photonList[median], &photonList[end], Photon::positionXComparator);
    else if (axis == AXIS_Y)
        nth_element(&photonList[start], &photonList[median], &photonList[end], Photon::positionYComparator);
    else if (axis == AXIS_Z)
        nth_element(&photonList[start], &photonList[median], &photonList[end], Photon::positionZComparator);
    else
        assert(false);  // This should never happen.
    photonList[median].axis = axis;
    photonMap[root] = photonList[median];

    // Calculate new bounding box.
    float3 leftMax  = bbMax;
    float3 rightMin = bbMin;
    float3 midPoint = photonMap[root].position;
    switch (axis) {
        case AXIS_X:
            rightMin.x = midPoint.x;
            leftMax.x  = midPoint.x;
            break;
        case AXIS_Y:
            rightMin.y = midPoint.y;
            leftMax.y  = midPoint.y;
            break;
        case AXIS_Z:
            rightMin.z = midPoint.z;
            leftMax.z  = midPoint.z;
            break;
        default:
            assert(false);
            break;
    }

    buildPhotonMapAcceleration(photonList, start, median, photonMap, 2*root+1, bbMin,  leftMax);
    buildPhotonMapAcceleration(photonList, median+1, end, photonMap, 2*root+2, rightMin, bbMax);
}   /* -----  end of method PPMRenderer::buildPhotonMapAcceleration ----- */



void PPMRenderer::createPhotonMap()
{
    RTsize photonListSize = 0;
    m_photonList->getSize(photonListSize);
    Photon * validPhotonList = new Photon [photonListSize];

    // count valid photons & build bounding box
    uint nValidPhotons = 0;
    float3 bbMin = make_float3(+std::numeric_limits<float>::max());
    float3 bbMax = make_float3(-std::numeric_limits<float>::max());
    Photon * photonListPtr = static_cast<Photon *>(m_photonList->map());
    for (uint i = 0; i < static_cast<uint>(photonListSize); i++)
        if (fmaxf(photonListPtr[i].flux) > 0.0f) {
            validPhotonList[nValidPhotons] = photonListPtr[i];
            bbMin = fminf(bbMin, validPhotonList[nValidPhotons].position);
            bbMax = fmaxf(bbMax, validPhotonList[nValidPhotons].position);
            ++nValidPhotons;
        }
    m_photonList->unmap();
    m_nEmittedPhotons += nValidPhotons;

    // build acceleration
    Photon * photonMapPtr  = static_cast<Photon *>(m_photonMap->map());
    buildPhotonMapAcceleration(validPhotonList, 0, nValidPhotons, photonMapPtr, 0, bbMin, bbMax);
    m_photonMap->unmap();

    delete [] validPhotonList;
}   /* -----  end of method PPMRenderer::createPhotonMap  ----- */



void PPMRenderer::init()
{
    Renderer::init();

    context()["nEmittedPhotons"]->setUint(0u);

    // init pixel sampling data, importon shooting data, and photon shooting data
    initPixelSamplingPassData();
    initImportonShootingPassData();
    initPhotonShootingPassData();
    setEntryPointPrograms("gatheringPassPrograms.cu", GatheringPass);
    setMissProgram("gatheringPassPrograms.cu", GatheringRay, "handleGatheringRayMiss");
}   /* -----  end of method PPMRenderer::init  ----- */



void PPMRenderer::initImportonShootingPassData()
{
}   /* -----  end of method PPMRenderer::initImportonShootingData  ----- */



void PPMRenderer::initPhotonShootingPassData()
{
    uint size = PHOTON_WIDTH * PHOTON_HEIGHT * PHOTON_COUNT;

    // create photon buffer
    m_photonList = context()->createBuffer(RT_BUFFER_OUTPUT);
    m_photonList->setFormat(RT_FORMAT_USER);
    m_photonList->setElementSize(sizeof(Photon));
    m_photonList->setSize(size);
    context()["photonList"]->set(m_photonList);

    // create photon acceleration buffer
    m_photonMap = context()->createBuffer(RT_BUFFER_INPUT);
    m_photonMap->setFormat(RT_FORMAT_USER);
    m_photonMap->setElementSize(sizeof(Photon));
    m_photonMap->setSize(size);
    context()["photonMap"]->set(m_photonMap);

    // create photon shooting programs
    setEntryPointPrograms("photonShootingPassPrograms.cu", PhotonShootingPass);
    setMissProgram("photonShootingPassPrograms.cu", PhotonShootingRay, "handlePhotonShootingRayMiss");
}   /* -----  end of method PPMRenderer::initPhotonShootingPassData  ----- */



void PPMRenderer::initPixelSamplingPassData()
{
    // create pixel sample buffer
    m_pixelSampleList = context()->createBuffer(RT_BUFFER_OUTPUT);
    m_pixelSampleList->setFormat(RT_FORMAT_USER);
    m_pixelSampleList->setElementSize(sizeof(PixelSample));
    m_pixelSampleList->setSize(width(), height());
    context()["pixelSampleList"]->set(m_pixelSampleList);

    // create pixel sampling programs
    setEntryPointPrograms("pixelSamplingPassPrograms.cu", PixelSamplingPass);
    setMissProgram("pixelSamplingPassPrograms.cu", PixelSamplingRay, "handlePixelSamplingRayMiss");
}   /* -----  end of method PPMRenderer::initPixelSamplingPassData  ----- */



void PPMRenderer::render()
{
    // For convenience.
    Scene::RayGenCameraData & cameraData = scene()->m_rayGenCameraData;

    // re-run the pixel sampling pass if camera has changed
    if (scene()->isCameraChanged()) {
        m_nEmittedPhotons = 0u;
        scene()->setIsCameraChanged(false);
        context()["cameraPosition"]->setFloat(cameraData.eye);
        context()["cameraU"]->setFloat(cameraData.U);
        context()["cameraV"]->setFloat(cameraData.V);
        context()["cameraW"]->setFloat(cameraData.W);
        context()["launchSize"]->setUint(width(), height());
        context()->launch(PixelSamplingPass, width(), height());
    }

    generateSamples(PHOTON_WIDTH * PHOTON_HEIGHT * PHOTON_COUNT * 2);
    context()["launchSize"]->setUint(PHOTON_WIDTH, PHOTON_HEIGHT);
    context()->launch(PhotonShootingPass, PHOTON_WIDTH, PHOTON_HEIGHT);
    createPhotonMap();

    context()["launchSize"]->setUint(width(), height());
    context()["nEmittedPhotons"]->setUint(m_nEmittedPhotons);
    context()->launch(GatheringPass, width(), height());
}   /* -----  end of method PPMRenderer::render  ----- */



void PPMRenderer::resize(unsigned int width, unsigned int height)
{
    // Call parent's resize(...).
    Renderer::resize(width, height);
    m_pixelSampleList->setSize(width, height);
}   /* -----  end of method PPMRenderer::doResize  ----- */
