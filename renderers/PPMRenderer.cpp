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



PPMRenderer::PPMRenderer(Scene * scene) : Renderer(scene),
    m_radius(DEFAULT_RADIUS),
    m_nPhotonsUsed(DEFAULT_N_PHOTONS_USED),
    m_nPhotonsWanted(DEFAULT_N_PHOTONS_WANTED),
    m_photonShootingPassLaunchWidth(DEFAULT_PHOTON_SHOOTING_PASS_LAUNCH_WIDTH),
    m_photonShootingPassLaunchHeight(DEFAULT_PHOTON_SHOOTING_PASS_LAUNCH_HEIGHT)
{
    m_nPhotonsPerThread = m_nPhotonsWanted /
        m_photonShootingPassLaunchWidth /
        m_photonShootingPassLaunchHeight;
}   /* -----  end of method PPMRenderer::PPMRenderer  ----- */



PPMRenderer::~PPMRenderer()
{
    /* EMPTY */
}   /* -----  end of method PPMRenderer::~PPMRenderer  ----- */



void PPMRenderer::init()
{
    Renderer::preInit();

    debug("sizeof(PixelSample) = \033[01;31m%4d\033[00m.\n", sizeof(PixelSample));
    debug("sizeof(Photon)      = \033[01;31m%4d\033[00m.\n", sizeof(Photon));

    // buffers
    m_pixelSampleList = context()->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_USER);
    m_pixelSampleList->setElementSize(sizeof(PixelSample));
    m_pixelSampleList->setSize(width(), height());
    context()["pixelSampleList"]->set(m_pixelSampleList);

    m_photonMap = context()->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_USER);
    m_photonMap->setElementSize(sizeof(Photon));
    m_photonMap->setSize(m_nPhotonsWanted);
    context()["photonList"]->set(m_photonMap);
    context()["photonMap"]->set(m_photonMap);
    debug("\033[01;33mphotonMap\033[00m consumes: \033[01;31m%10u\033[00m.\n",
            sizeof(Photon) * m_nPhotonsWanted);

    // user variables
    context()["radiusSquared"]->setFloat(m_radius * m_radius);
    context()["maxRayDepth"]->setUint(DEFAULT_MAX_RAY_DEPTH);
    context()["nPhotonsUsed"]->setUint(m_nPhotonsUsed);
    // auto generated variables
    context()["frameCount"]->setUint(0);
    context()["launchSize"]->setUint(0, 0);
    context()["nPhotonsPerThread"]->setUint(m_nPhotonsPerThread);
    context()["nEmittedPhotons"]->setUint(0);
    context()["nSamplesPerThread"]->setUint(0);

    // programs
    context()->setEntryPointCount(N_PASSES);
    setExceptionProgram(PixelSamplingPass);
    setExceptionProgram(PhotonShootingPass);
    setExceptionProgram(DensityEstimationPass);
    setRayGenerationProgram(PixelSamplingPass,     "PPMRenderer.cu", "generatePixelSamples");
    setRayGenerationProgram(PhotonShootingPass,    "PPMRenderer.cu", "shootPhotons");
    setRayGenerationProgram(DensityEstimationPass, "PPMRenderer.cu", "estimateDensity");
    setMissProgram(NormalRay, "ray.cu", "handleNormalRayMiss");

    Renderer::postInit();
}   /* -----  end of method PPMRenderer::init  ----- */



void PPMRenderer::render(const Scene::RayGenCameraData & cameraData)
{
    clearOutputBuffer();

    uint  nSamplesPerThread = 0;
    uint2 launchSize = make_uint2(0, 0);

    if (scene()->isCameraChanged()) {
        m_frame = 0;
        m_nEmittedPhotons = 0;
        context()["frameCount"]->setUint(m_frame);

        scene()->setIsCameraChanged(false);

        // pixel sample
        debug("\033[01;36mPrepare to launch pixel sampling pass\033[00m\n");
        setLocalHeapPointer(0);
        launchSize = make_uint2(width(), height());
        context()["launchSize"]->setUint(launchSize.x, launchSize.y);
        nSamplesPerThread = 2;
        generateSamples(nSamplesPerThread * launchSize.x * launchSize.y);
        context()["nSamplesPerThread"]->setUint(nSamplesPerThread);
        context()->launch(PixelSamplingPass, launchSize.x, launchSize.y);
    }

    context()["frameCount"]->setUint(m_frame++);

//    // Dump average radius.
//    const PixelSample * pixelSampleList = static_cast<PixelSample *>(m_pixelSampleList->map());
//    float averageRadiusSquared = 0.0f;
//    uint  nPixelSamples        = 0;
//    for (unsigned int i = 0; i < launchSize.x * launchSize.y; ++i) {
//        if (pixelSampleList[i].isHit) {
//            averageRadiusSquared += pixelSampleList[i].radiusSquared;
//            ++nPixelSamples;
//        }
//    }
//    debug("\033[01;33maverageRadiusSquared\033[00m: \033[01;31m%f\033[00m\n",
//            averageRadiusSquared / static_cast<float>(nPixelSamples));
//    m_pixelSampleList->unmap();

    // photon
    debug("\033[01;36mPrepare to launch photon shooting pass\033[00m\n");
    setLocalHeapPointer(m_photonShootingPassLocalHeapOffset);
    launchSize = make_uint2(
            m_photonShootingPassLaunchWidth,
            m_photonShootingPassLaunchHeight);
    context()["launchSize"]->setUint(launchSize.x, launchSize.y);
    nSamplesPerThread = 4 * m_nPhotonsPerThread;
    generateSamples(nSamplesPerThread * launchSize.x * launchSize.y);
    context()["nSamplesPerThread"]->setUint(nSamplesPerThread);
    context()->launch(PhotonShootingPass, launchSize.x, launchSize.y);
    createPhotonMap();

    // gathering
    debug("\033[01;36mPrepare to launch density estimation pass\033[00m\n");
    setLocalHeapPointer(m_densityEstimationPassLocalHeapOffset);
    context()["nEmittedPhotons"]->setUint(m_nEmittedPhotons);
    launchSize = make_uint2(width(), height());
    context()["launchSize"]->setUint(launchSize.x, launchSize.y);
    context()->launch(DensityEstimationPass, launchSize.x, launchSize.y);
}   /* -----  end of method PPMRenderer::render  ----- */



void PPMRenderer::resize(unsigned int width, unsigned int height)
{
    // Call parent's resize(...).
    Renderer::resize(width, height);
    m_pixelSampleList->setSize(width, height);

    // Local heap.
    m_pixelSamplingPassLocalHeapSize =
        width * height * sizeof(Intersection);
    debug("PixelSamplingPass     demands \033[01;33m%u\033[00m bytes of memory on localHeap.\n",
            m_pixelSamplingPassLocalHeapSize);

    m_photonShootingPassLocalHeapSize =
        m_photonShootingPassLaunchWidth * m_photonShootingPassLaunchHeight * sizeof(Intersection);
    debug("PhotonShootingPass    demands \033[01;33m%u\033[00m bytes of memory on localHeap.\n",
            m_photonShootingPassLocalHeapSize);
    m_photonShootingPassLocalHeapOffset = m_pixelSamplingPassLocalHeapSize;

    m_densityEstimationPassLocalHeapSize =
        width * height * m_nPhotonsUsed * sizeof(GatheredPhoton);
    debug("DensityEstimationPass demands \033[01;33m%u\033[00m bytes of memory on localHeap.\n",
            m_densityEstimationPassLocalHeapSize);
    m_densityEstimationPassLocalHeapOffset =
        m_photonShootingPassLocalHeapOffset + m_photonShootingPassLocalHeapSize;

    m_demandLocalHeapSize =
        m_densityEstimationPassLocalHeapOffset + m_densityEstimationPassLocalHeapSize;
    setLocalHeapSize(m_demandLocalHeapSize);
}   /* -----  end of method PPMRenderer::doResize  ----- */



void PPMRenderer::parseArguments(vector<char *> argumentList)
{
    int argc = argumentList.size();
    for (vector<char *>::iterator it = argumentList.begin();
         it != argumentList.end(); ++it)
    {
        std::string arg(*it);
        if (arg == "--radius") {
            if (++it != argumentList.end()) {
                m_radius = atof(*it);
                cerr << "Set radius to " << m_radius << endl;
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
}   /* -----  end of function PPMRenderer::parseArguments  ----- */



void PPMRenderer::printUsageAndExit(bool doExit)
{
    std::cerr
        << "PPM options:" << std::endl
        << "     | --raduis <float>    Set PPM's maximum gathering distance." << std::endl
        << std::endl;

    if (doExit)
        exit(EXIT_FAILURE);
}   /* -----  end of function PPMRenderer::printUsageAndExit  ----- */



void PPMRenderer::createPhotonMap()
{
    RTsize photonListSize = 0;
    m_photonMap->getSize(photonListSize);
    Photon * validPhotonList = new Photon [photonListSize];

    // count valid photons & build bounding box
    uint nValidPhotons = 0, nDirectPhotons = 0;
    float3 bbMin = make_float3(+std::numeric_limits<float>::max());
    float3 bbMax = make_float3(-std::numeric_limits<float>::max());
    Photon * photonListPtr = static_cast<Photon *>(m_photonMap->map());
    for (uint i = 0; i < static_cast<uint>(photonListSize); i++) {
        // Do not add direct photons.
        if (photonListPtr[i].flags & Photon::Direct) {
            ++nDirectPhotons;
            continue;
        }
        if (fmaxf(photonListPtr[i].flux) > 0.0f) {
            validPhotonList[nValidPhotons] = photonListPtr[i];
            bbMin = fminf(bbMin, validPhotonList[nValidPhotons].position);
            bbMax = fmaxf(bbMax, validPhotonList[nValidPhotons].position);
            ++nValidPhotons;
        }
    }
    m_nEmittedPhotons += nDirectPhotons;
    debug("direct photons: \033[01;31m%u\033[00m\n", nDirectPhotons);
    debug("valid  photons: \033[01;31m%u\033[00m\n", nValidPhotons);

    // build acceleration
    Photon * photonMapPtr = photonListPtr;
    KdTree::build(validPhotonList, 0, nValidPhotons, photonMapPtr, 0, bbMin, bbMax);
    m_photonMap->unmap();

    delete [] validPhotonList;
}   /* -----  end of method PPMRenderer::createPhotonMap  ----- */
