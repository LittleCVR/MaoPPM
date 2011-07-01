/*
 * =============================================================================
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
 * =============================================================================
 */

#include    "IGPPMRenderer.h"

/*----------------------------------------------------------------------------
 *  header files from std C/C++
 *----------------------------------------------------------------------------*/
#include    <cstdio>
#include    <ctime>
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
    m_guidedByImportons(true),
    m_nImportonsPerThread(DEFAULT_N_IMPORTONS_PER_THREAD),
    m_nPhotonsUsed(DEFAULT_N_PHOTONS_USED),
    m_nPhotonsWanted(DEFAULT_N_PHOTONS_WANTED),
    m_photonShootingPassLaunchWidth(DEFAULT_PHOTON_SHOOTING_PASS_LAUNCH_WIDTH),
    m_photonShootingPassLaunchHeight(DEFAULT_PHOTON_SHOOTING_PASS_LAUNCH_HEIGHT)
{
    m_nPhotonsPerThread = m_nPhotonsWanted /
        m_photonShootingPassLaunchWidth /
        m_photonShootingPassLaunchHeight;
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
    m_pixelSampleList = context()->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_USER);
    m_pixelSampleList->setElementSize(sizeof(PixelSample));
    m_pixelSampleList->setSize(width(), height());
    context()["pixelSampleList"]->set(m_pixelSampleList);

    m_importonList = context()->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_USER);
    m_importonList->setElementSize(sizeof(Importon));
    m_importonList->setSize(m_nImportonsPerThread * width() * height());
    context()["importonList"]->set(m_importonList);

    m_photonMap = context()->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_USER);
    m_photonMap->setElementSize(sizeof(Photon));
    m_photonMap->setSize(m_nPhotonsWanted);
    context()["photonList"]->set(m_photonMap);
    context()["photonMap"]->set(m_photonMap);
    debug("\033[01;33mphotonMap\033[00m consumes: \033[01;31m%10u\033[00m.\n",
            sizeof(Photon) * m_nPhotonsWanted);

    // variables
    context()["maxRayDepth"]->setUint(DEFAULT_MAX_RAY_DEPTH);
    context()["guidedByImportons"]->setUint(m_guidedByImportons);
    context()["nImportonsPerThread"]->setUint(m_nImportonsPerThread);
    context()["nPhotonsUsed"]->setUint(m_nPhotonsUsed);
    context()["nPhotonsPerThread"]->setUint(m_nPhotonsPerThread);
    context()["nEmittedPhotons"]->setUint(0);

    // programs
    context()->setEntryPointCount(N_PASSES);
    setExceptionProgram(PixelSamplingPass);
    setExceptionProgram(ImportonShootingPass);
    setExceptionProgram(PhotonShootingPass);
    setExceptionProgram(FinalGatheringPass);
    setRayGenerationProgram(PixelSamplingPass,    "IGPPMRenderer.cu", "generatePixelSamples");
    setRayGenerationProgram(ImportonShootingPass, "IGPPMRenderer.cu", "shootImportons");
    setRayGenerationProgram(PhotonShootingPass,   "IGPPMRenderer.cu", "shootPhotons");
    setRayGenerationProgram(FinalGatheringPass,   "IGPPMRenderer.cu", "gatherPhotons");
    setMissProgram(NormalRay, "ray.cu", "handleNormalRayMiss");
}   /* -----  end of method IGPPMRenderer::init  ----- */



void IGPPMRenderer::render(const Scene::RayGenCameraData & cameraData)
{
    bool reset = false;
    if (scene()->isCameraChanged()) {
        reset = true;
        m_frame = 0;
        m_nEmittedPhotons = 0;
        context()["cameraPosition"]->setFloat(cameraData.eye);
        context()["cameraU"]->setFloat(cameraData.U);
        context()["cameraV"]->setFloat(cameraData.V);
        context()["cameraW"]->setFloat(cameraData.W);
        scene()->setIsCameraChanged(false);
    }

    // Timing.
    clock_t startClock, endClock;

    uint  nSamplesPerThread = 0;
    uint2 launchSize = make_uint2(0, 0);

    context()["frameCount"]->setUint(m_frame++);
    // ImportonShootingPass needs this.
    context()["nEmittedPhotons"]->setUint(m_nEmittedPhotons);

    if (reset) {
        // pixel sample
        debug("\033[01;36mPrepare to launch pixel sampling pass\033[00m\n");
        setLocalHeapPointer(0);
        launchSize = make_uint2(width(), height());
        context()["launchSize"]->setUint(launchSize.x, launchSize.y);
        nSamplesPerThread = 2;
        generateSamples(nSamplesPerThread * launchSize.x * launchSize.y);
        context()["nSamplesPerThread"]->setUint(nSamplesPerThread);
        startClock  = clock();
        context()->launch(PixelSamplingPass, launchSize.x, launchSize.y);
        endClock    = clock();
        debug("\033[01;36mFinished launching pixel sampling pass in %f secs.\033[00m\n",
                static_cast<float>(endClock-startClock) / CLOCKS_PER_SEC);

        Light * lightList = static_cast<Light *>(scene()->m_lightList->map());
        Light * light = &lightList[0];
        float accumulated = 0.0f;
        for (unsigned int i = 0; i < N_THETA*N_PHI; ++i) {
            light->pdf[i] = light->area(i/N_PHI, i%N_PHI);
            accumulated += light->pdf[i];
            light->cdf[i] = accumulated / (4.0f * M_PIf);
        }
        scene()->m_lightList->unmap();
    }

    // importon
    debug("\033[01;36mPrepare to launch importon shooting pass\033[00m\n");
    setLocalHeapPointer(m_importonShootingPassLocalHeapOffset);
    launchSize = make_uint2(width(), height());
    context()["launchSize"]->setUint(launchSize.x, launchSize.y);
    nSamplesPerThread = 3 * m_nImportonsPerThread;
    generateSamples(nSamplesPerThread * launchSize.x * launchSize.y);
    context()["nSamplesPerThread"]->setUint(nSamplesPerThread);
    startClock  = clock();
    context()->launch(ImportonShootingPass, launchSize.x, launchSize.y);
    endClock    = clock();
    debug("\033[01;36mFinished launching importon shooting pass in %f secs.\033[00m\n",
            static_cast<float>(endClock-startClock) / CLOCKS_PER_SEC);

//    // Dump average radius.
//    const Importon * importonList = static_cast<Importon *>(m_importonList->map());
//    float averageRadiusSquared = 0.0f;
//    uint  nImportons           = 0;
//    for (unsigned int i = 0; i < launchSize.x * launchSize.y * m_nImportonsPerThread; ++i) {
//        if (importonList[i].isHit) {
//            averageRadiusSquared += importonList[i].radiusSquared;
//            ++nImportons;
//        }
//    }
//    debug("\033[01;33maverageRadiusSquared\033[00m: \033[01;31m%f\033[00m\n",
//            averageRadiusSquared / static_cast<float>(nImportons));
//    m_importonList->unmap();

    Light * lightList = static_cast<Light *>(scene()->m_lightList->map());
    Light * light = &lightList[0];
    float total = 0.0f, accumulated = 0.0f;
    for (unsigned int i = 0; i < N_THETA*N_PHI; ++i) {
        total += light->pdf[i];
    }
    debug("total: %f\n", total);
    debug("light PDF:");
    for (unsigned int i = 0; i < N_THETA*N_PHI; ++i) {
        light->pdf[i] /= total;
        accumulated += light->pdf[i];
        if (i % N_THETA == 0) fprintf(stderr, "\n");
        fprintf(stderr, "%8.4f ", light->pdf[i]);
        if (total != 0.0f)
            light->cdf[i] = 0.50f * light->cdf[i] + 0.49f * accumulated + 0.01f;
        light->pdf[i] = 0.0f;
    }
    fprintf(stderr, "\n");
    debug("light CDF:");
    for (unsigned int i = 0; i < N_THETA*N_PHI; ++i) {
        light->cdf[i] /= light->cdf[N_THETA*N_PHI-1];
        if (i % N_THETA == 0) fprintf(stderr, "\n");
        fprintf(stderr, "%4.4f ", light->cdf[i]);
    }
    fprintf(stderr, "\n");
    scene()->m_lightList->unmap();

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
    startClock  = clock();
    context()->launch(PhotonShootingPass, launchSize.x, launchSize.y);
    endClock    = clock();
    debug("\033[01;36mFinished launching photon shooting pass in %f secs.\033[00m\n",
            static_cast<float>(endClock-startClock) / CLOCKS_PER_SEC);

    debug("\033[01;36mPrepare to build photon map.\033[00m\n");
    startClock  = clock();
    createPhotonMap();
    endClock    = clock();
    debug("\033[01;36mFinished building photon map in %f secs.\033[00m\n",
            static_cast<float>(endClock-startClock) / CLOCKS_PER_SEC);

    // gathering
    debug("\033[01;36mPrepare to launch final gathering pass\033[00m\n");
    context()["nEmittedPhotons"]->setUint(m_nEmittedPhotons);
    launchSize = make_uint2(width(), height());
    context()["launchSize"]->setUint(launchSize.x, launchSize.y);
    startClock  = clock();
    context()->launch(FinalGatheringPass, launchSize.x, launchSize.y);
    endClock    = clock();
    debug("\033[01;36mFinished launching final gathering pass in %f secs.\033[00m\n",
            static_cast<float>(endClock-startClock) / CLOCKS_PER_SEC);
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

    // Local heap.
    m_pixelSamplingPassLocalHeapSize =
        width * height * sizeof(Intersection);
    debug("PixelSamplingPass    demands \033[01;33m%u\033[00m bytes of memory on localHeap.\n",
            m_pixelSamplingPassLocalHeapSize);

    m_importonShootingPassLocalHeapSize =
        width * height * m_nImportonsPerThread * sizeof(Intersection);
    debug("ImportonShootingPass demands \033[01;33m%u\033[00m bytes of memory on localHeap.\n",
            m_importonShootingPassLocalHeapSize);
    m_importonShootingPassLocalHeapOffset = m_pixelSamplingPassLocalHeapSize;

    m_photonShootingPassLocalHeapSize =
        m_photonShootingPassLaunchWidth * m_photonShootingPassLaunchHeight *
        m_nPhotonsPerThread * sizeof(Intersection);
    debug("PhotonShootingPass   demands \033[01;33m%u\033[00m bytes of memory on localHeap.\n",
            m_photonShootingPassLocalHeapSize);
    m_photonShootingPassLocalHeapOffset =
        m_importonShootingPassLocalHeapOffset + m_importonShootingPassLocalHeapSize;

    m_finalGatheringPassLocalHeapSize =
        width * height * m_nPhotonsUsed * sizeof(GatheredPhoton);
    debug("FinalGatheringPass   demands \033[01;33m%u\033[00m bytes of memory on localHeap.\n",
            m_finalGatheringPassLocalHeapSize);
    m_finalGatheringPassLocalHeapOffset =
        m_photonShootingPassLocalHeapOffset + m_photonShootingPassLocalHeapSize;

    m_demandLocalHeapSize =
        m_finalGatheringPassLocalHeapOffset + m_finalGatheringPassLocalHeapSize;
    setLocalHeapSize(m_demandLocalHeapSize);
}   /* -----  end of method IGPPMRenderer::resize  ----- */



void IGPPMRenderer::setGuidedByImportons(bool guided)
{
    m_guidedByImportons = guided;
}



void IGPPMRenderer::createPhotonMap()
{
    RTsize photonListSize = 0;
    m_photonMap->getSize(photonListSize);
    Photon * validPhotonList = new Photon [photonListSize];

    // count valid photons & build bounding box
    uint nValidPhotons = 0, nDirectPhotons = 0;
    float3 bbMin = make_float3(+std::numeric_limits<float>::max());
    float3 bbMax = make_float3(-std::numeric_limits<float>::max());
    Photon * photonListPtr = static_cast<Photon *>(m_photonMap->map());
    for (uint i = 0; i < static_cast<uint>(photonListSize); i++)
        if (fmaxf(photonListPtr[i].flux) > 0.0f) {
            validPhotonList[nValidPhotons] = photonListPtr[i];
            bbMin = fminf(bbMin, validPhotonList[nValidPhotons].position);
            bbMax = fmaxf(bbMax, validPhotonList[nValidPhotons].position);
            // PPM does not have to store direct photons, but IGPPM has to.
            if (photonListPtr[i].flags & Photon::Direct)
                ++nDirectPhotons;
            ++nValidPhotons;
        }
    m_nEmittedPhotons += nDirectPhotons;
    debug("direct photons: \033[01;31m%u\033[00m\n", nDirectPhotons);
    debug("valid  photons: \033[01;31m%u\033[00m\n", nValidPhotons);

    // build acceleration
    Photon * photonMapPtr = photonListPtr;
    KdTree::build(validPhotonList, 0, nValidPhotons, photonMapPtr, 0, bbMin, bbMax);
    m_photonMap->unmap();

    delete [] validPhotonList;
}   /* -----  end of method IGPPMRenderer::createPhotonMap  ----- */
