/*
 * =====================================================================================
 *
 *       Filename:  IGPPMRenderer.cpp
 *
 *    Description:  The Importons Guided Progressive Photon Map Renderer.
 *
 *        Version:  1.0
 *        Created:  2011-06-21 07:37:54
 *
 *         Author:  Chun-Wei Huang (LittleCVR), 
 *        Company:  Communication & Multimedia Laboratory,
 *                  Department of Computer Science & Information Engineering,
 *                  National Taiwan University
 *
 * =====================================================================================
 */

#include    "IGPPMRenderer.h"

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



IGPPMRenderer::IGPPMRenderer(Scene * scene) : Renderer(scene),
    m_nImportonsPerThread(DEFAULT_N_IMPORTONS_PER_THREAD),
    m_nPhotonsPerThread(DEFAULT_N_PHOTONS_PER_THREAD)
{
    /* EMPTY */
}   /* -----  end of method IGPPMRenderer::IGPPMRenderer  ----- */



IGPPMRenderer::~IGPPMRenderer()
{
    /* EMPTY */
}   /* -----  end of method IGPPMRenderer::~IGPPMRenderer  ----- */



void IGPPMRenderer::init()
{
    Renderer::init();

    debug("sizeof(PixelSample) = \033[01;31m%4d\033[00m.\n", sizeof(PixelSample));
    debug("sizeof(Importon)    = \033[01;31m%4d\033[00m.\n", sizeof(Importon));
    debug("sizeof(Photon)      = \033[01;31m%4d\033[00m.\n", sizeof(Photon));

    // buffers
    m_pixelSampleList = context()->createBuffer(RT_BUFFER_OUTPUT, RT_FORMAT_USER);
    m_pixelSampleList->setElementSize(sizeof(PixelSample));
    m_pixelSampleList->setSize(width(), height());
    context()["pixelSampleList"]->set(m_pixelSampleList);

    m_importonList = context()->createBuffer(RT_BUFFER_OUTPUT, RT_FORMAT_USER);
    m_importonList->setElementSize(sizeof(Importon));
    m_importonList->setSize(m_nImportonsPerThread * width() * height());
    context()["importonList"]->set(m_importonList);

    m_photonMap = context()->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_USER);
    m_photonMap->setElementSize(sizeof(Photon));
    m_photonMap->setSize(m_nPhotonsPerThread * width() * height());
    context()["photonList"]->set(m_photonMap);
    context()["photonMap"]->set(m_photonMap);

    // variables
    context()["nImportonsPerThread"]->setUint(m_nImportonsPerThread);
    context()["nPhotonsPerThread"]->setUint(m_nPhotonsPerThread);

    // programs
    context()->setEntryPointCount(N_PASSES);
    setExceptionProgram(PixelSamplingPass);
    setExceptionProgram(ImportonShootingPass);
    setExceptionProgram(PhotonShootingPass);
    setExceptionProgram(FinalGatheringPass);
    setRayGenerationProgram(PixelSamplingPass, "IGPPMRendererPixelSamplingPass.cu", "generatePixelSamples");
    setRayGenerationProgram(ImportonShootingPass, "IGPPMRendererImportonShootingPass.cu", "shootImportons");
    setRayGenerationProgram(PhotonShootingPass, "IGPPMRendererPhotonShootingPass.cu", "shootPhotons");
    setRayGenerationProgram(FinalGatheringPass, "IGPPMRendererFinalGatheringPass.cu", "gather");
    setMissProgram(NormalRay, "ray.cu", "handleNormalRayMiss");
}   /* -----  end of method IGPPMRenderer::init  ----- */



void IGPPMRenderer::render(const Scene::RayGenCameraData & cameraData)
{
    uint  nSamplesPerThread = 0;
    uint2 launchSize = make_uint2(0, 0);

    if (scene()->isCameraChanged()) {
        m_nEmittedPhotons = 0;

        scene()->setIsCameraChanged(false);
        context()["cameraPosition"]->setFloat(cameraData.eye);
        context()["cameraU"]->setFloat(cameraData.U);
        context()["cameraV"]->setFloat(cameraData.V);
        context()["cameraW"]->setFloat(cameraData.W);

        // pixel sample
        launchSize = make_uint2(width(), height());
        context()["launchSize"]->setUint(launchSize.x, launchSize.y);
        nSamplesPerThread = 2;
        generateSamples(nSamplesPerThread * launchSize.x * launchSize.y);
        context()["nSamplesPerThread"]->setUint(nSamplesPerThread);
        context()->launch(PixelSamplingPass, launchSize.x, launchSize.y);

        // importon
        launchSize = make_uint2(width(), height());
        context()["launchSize"]->setUint(launchSize.x, launchSize.y);
        nSamplesPerThread = 2 * m_nImportonsPerThread;
        generateSamples(nSamplesPerThread * launchSize.x * launchSize.y);
        context()["nSamplesPerThread"]->setUint(nSamplesPerThread);
        context()->launch(ImportonShootingPass, launchSize.x, launchSize.y);
    }

//    // photon
//    launchSize = make_uint2(width(), height());
//    context()["launchSize"]->setUint(launchSize.x, launchSize.y);
//    nSamplesPerThread = 2 * m_nPhotonsPerThread;
//    generateSamples(nSamplesPerThread * launchSize.x * launchSize.y);
//    context()["nSamplesPerThread"]->setUint(nSamplesPerThread);
//    context()->launch(PhotonShootingPass, launchSize.x, launchSize.y);
//    createPhotonMap();
//
//    // gathering
//    context()["nEmittedPhotons"]->setUint(m_nEmittedPhotons);
//    launchSize = make_uint2(width(), height());
//    context()["launchSize"]->setUint(launchSize.x, launchSize.y);
//    context()->launch(FinalGatheringPass, launchSize.x, launchSize.y);
}   /* -----  end of method IGPPMRenderer::render  ----- */



void IGPPMRenderer::resize(unsigned int width, unsigned int height)
{
    Renderer::resize(width, height);

    m_pixelSampleList->setSize(width, height);
    context()["pixelSampleList"]->set(m_pixelSampleList);
    debug("\033[01;33mpixelSampleList\033[00m resized to: \033[01;31m%10u\033[00m.\n",
            sizeof(PixelSample) * width * height);
    m_importonList->setSize(m_nImportonsPerThread * width * height);
    context()["importonList"]->set(m_importonList);
    debug("\033[01;33mimportonList\033[00m    resized to: \033[01;31m%10u\033[00m.\n",
            sizeof(Importon) * m_nImportonsPerThread * width * height);
    m_photonMap->setSize(m_nPhotonsPerThread * width * height);
    debug("\033[01;33mphotonMap\033[00m       resized to: \033[01;31m%10u\033[00m.\n",
            sizeof(Photon) * m_nPhotonsPerThread * width * height);
}   /* -----  end of method IGPPMRenderer::resize  ----- */



void IGPPMRenderer::buildPhotonMapAcceleration(Photon * photonList,
        uint start, uint end, Photon * photonMap,
        uint root, float3 bbMin, float3 bbMax)
{
    if (end - start == 0) {
        // Make a fake photon.
        photonMap[root].flags = Photon::Null;
        photonMap[root].flux  = make_float3(0.0f);
        return;
    }
    if (end - start == 1) {
        // Create a leaf photon.
        photonList[start].flags = Photon::Leaf;
        photonMap[root] = photonList[start];
        return;
    }

    // Choose the longest axis.
    uint axis = 0;
    float3 bbDiff = bbMax - bbMin;
    if (bbDiff.x > bbDiff.y) {
        if (bbDiff.x > bbDiff.z)
            axis = Photon::AxisX;
        else
            axis = Photon::AxisZ;
    } else {
        if (bbDiff.y > bbDiff.z)
            axis = Photon::AxisY;
        else
            axis = Photon::AxisZ;
    }

    // Partition the photon list.
    uint median = (start + end) / 2;
    if (axis == Photon::AxisX)
        nth_element(&photonList[start], &photonList[median], &photonList[end], Photon::positionXComparator);
    else if (axis == Photon::AxisY)
        nth_element(&photonList[start], &photonList[median], &photonList[end], Photon::positionYComparator);
    else if (axis == Photon::AxisZ)
        nth_element(&photonList[start], &photonList[median], &photonList[end], Photon::positionZComparator);
    else
        assert(false);  // This should never happen.
    photonList[median].flags = axis;
    photonMap[root] = photonList[median];

    // Calculate new bounding box.
    float3 leftMax  = bbMax;
    float3 rightMin = bbMin;
    float3 midPoint = photonMap[root].position;
    switch (axis) {
        case Photon::AxisX:
            rightMin.x = midPoint.x;
            leftMax.x  = midPoint.x;
            break;
        case Photon::AxisY:
            rightMin.y = midPoint.y;
            leftMax.y  = midPoint.y;
            break;
        case Photon::AxisZ:
            rightMin.z = midPoint.z;
            leftMax.z  = midPoint.z;
            break;
        default:
            assert(false);
            break;
    }

    buildPhotonMapAcceleration(photonList, start, median, photonMap, 2*root+1, bbMin,  leftMax);
    buildPhotonMapAcceleration(photonList, median+1, end, photonMap, 2*root+2, rightMin, bbMax);
}   /* -----  end of method IGPPMRenderer::buildPhotonMapAcceleration ----- */



void IGPPMRenderer::createPhotonMap()
{
    RTsize photonListSize = 0;
    m_photonMap->getSize(photonListSize);
    Photon * validPhotonList = new Photon [photonListSize];

    // count valid photons & build bounding box
    uint nValidPhotons = 0;
    float3 bbMin = make_float3(+std::numeric_limits<float>::max());
    float3 bbMax = make_float3(-std::numeric_limits<float>::max());
    Photon * photonListPtr = static_cast<Photon *>(m_photonMap->map());
    for (uint i = 0; i < static_cast<uint>(photonListSize); i++)
        if (fmaxf(photonListPtr[i].flux) > 0.0f) {
            validPhotonList[nValidPhotons] = photonListPtr[i];
            bbMin = fminf(bbMin, validPhotonList[nValidPhotons].position);
            bbMax = fmaxf(bbMax, validPhotonList[nValidPhotons].position);
            ++nValidPhotons;
        }
    m_photonMap->unmap();
    m_nEmittedPhotons += nValidPhotons;

    // build acceleration
    Photon * photonMapPtr  = static_cast<Photon *>(m_photonMap->map());
    buildPhotonMapAcceleration(validPhotonList, 0, nValidPhotons, photonMapPtr, 0, bbMin, bbMax);
    m_photonMap->unmap();

    delete [] validPhotonList;
}   /* -----  end of method IGPPMRenderer::createPhotonMap  ----- */
