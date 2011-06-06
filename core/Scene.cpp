/*
 * =====================================================================================
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
 * =====================================================================================
 */





/* #####   HEADER FILE INCLUDES   ################################################### */

/*-----------------------------------------------------------------------------
 *  header files from std C/C++
 *-----------------------------------------------------------------------------*/
#include    <algorithm>
#include    <cstdio>
#include    <ctime>
#include    <iostream>
#include    <limits>

/*-----------------------------------------------------------------------------
 *  header files of our own
 *-----------------------------------------------------------------------------*/
#include    "Scene.h"
#include    "SceneBuilder.h"

/*-----------------------------------------------------------------------------
 *  namespace
 *-----------------------------------------------------------------------------*/
using namespace std;
using namespace optix;
using namespace MaoPPM;



/* #####   MACROS  -  LOCAL TO THIS SOURCE FILE   ################################### */

/* #####   TYPE DEFINITIONS  -  LOCAL TO THIS SOURCE FILE   ######################### */

/* #####   DATA TYPES  -  LOCAL TO THIS SOURCE FILE   ############################### */

/* #####   VARIABLES  -  LOCAL TO THIS SOURCE FILE   ################################ */

/* #####   PROTOTYPES  -  LOCAL TO THIS SOURCE FILE   ############################### */

/* #####   FUNCTION DEFINITIONS  -  EXPORTED FUNCTIONS   ############################ */

/* #####   FUNCTION DEFINITIONS  -  LOCAL TO THIS SOURCE FILE   ##################### */

Scene::Scene() : SampleScene(),
    m_width(INITIAL_WIDTH), m_height(INITIAL_HEIGHT)
{
    /* EMPTY */
}   /* -----  end of method Scene::Scene  ----- */



Scene::~Scene()
{
    /* EMPTY */
}   /* -----  end of method Scene::~Scene  ----- */



void Scene::doResize(unsigned int width, unsigned int height)
{
    m_width = width; m_height = height;

    m_outputBuffer->setSize(m_width, m_height);
    m_pixelSampleList->setSize(m_width, m_height);
}   /* -----  end of method Scene::doResize  ----- */



#ifdef DEBUG
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
        throw Exception::makeException(rc, getContext()->get());

    cerr << "Compute capability is SM" << computeCapability.x << computeCapability.y << ", ";
    if (computeCapability.x < 1 || computeCapability.y < 1)
        cerr << "debug mode cannot be enabled." << endl;
    else {
        cerr << "debug mode enabled." << endl;
        getContext()->setPrintEnabled(true);
    }
}   /* -----  end of method Scene::initDebug  ----- */
#endif



void Scene::initImportonShootingPassData()
{
}   /* -----  end of method Scene::initImportonShootingData  ----- */



void Scene::initPhotonShootingPassData()
{
    uint size = PHOTON_WIDTH * PHOTON_HEIGHT * PHOTON_COUNT;

    // create photon buffer
    m_photonList = getContext()->createBuffer(RT_BUFFER_OUTPUT);
    m_photonList->setFormat(RT_FORMAT_USER);
    m_photonList->setElementSize(sizeof(Photon));
    m_photonList->setSize(size);
    getContext()["photonList"]->set(m_photonList);

    // create photon acceleration buffer
    m_photonMap = getContext()->createBuffer(RT_BUFFER_INPUT);
    m_photonMap->setFormat(RT_FORMAT_USER);
    m_photonMap->setElementSize(sizeof(Photon));
    m_photonMap->setSize(size);
    getContext()["photonMap"]->set(m_photonMap);

    // create photon shooting programs
    setPrograms("photonShootingPassPrograms.cu", PhotonShootingPass);
}   /* -----  end of method Scene::initPhotonShootingPassData  ----- */



void Scene::initPixelSamplingPassData()
{
    // create pixel sample buffer
    m_pixelSampleList = getContext()->createBuffer(RT_BUFFER_OUTPUT);
    m_pixelSampleList->setFormat(RT_FORMAT_USER);
    m_pixelSampleList->setElementSize(sizeof(PixelSample));
    m_pixelSampleList->setSize(m_width, m_height);
    getContext()["pixelSampleList"]->set(m_pixelSampleList);

    // create pixel sampling programs
    setPrograms("pixelSamplingPassPrograms.cu", PixelSamplingPass);
}   /* -----  end of method Scene::initPixelSamplingPassData  ----- */



void Scene::initScene(InitialCameraData & cameraData)
{
#ifdef DEBUG
    initDebug();
#endif

    getContext()->setRayTypeCount(nRayTypes);
    getContext()->setEntryPointCount(nPasses);
    getContext()->setStackSize(STACK_SIZE);
    getContext()["rayEpsilon"]->setFloat(RAY_EPSILON);
    getContext()["nEmittedPhotons"]->setUint(0u);

    // create output buffer
    m_outputBuffer = createOutputBuffer(RT_FORMAT_FLOAT4, m_width, m_height);
    getContext()["outputBuffer"]->set(m_outputBuffer);

    // init pixel sampling data, importon shooting data, and photon shooting data
    initPixelSamplingPassData();
    initImportonShootingPassData();
    initPhotonShootingPassData();
    setPrograms("gatheringPassPrograms.cu", GatheringPass);

    // initialize root object
    m_rootObject = getContext()->createGeometryGroup();
    getContext()["rootObject"]->set(m_rootObject);
    Acceleration acceleration = getContext()->createAcceleration("Bvh", "Bvh");
    m_rootObject->setAcceleration(acceleration);

    // initialize light buffer
    m_lightList = getContext()->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER);
    m_lightList->setElementSize(sizeof(Light));
    m_lightList->setSize(0);
    getContext()["lightList"]->set(m_lightList);

    // initialize sample buffer
    srand(time(NULL));
    m_sampleList = getContext()->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT);
    m_sampleList->setSize(0);
    getContext()["sampleList"]->set(m_sampleList);

    // build scene world
    SceneBuilder sceneBuilder;
    sceneBuilder.parse(this, &cameraData);
}   /* -----  end of method Scene::initScene  ----- */



void Scene::setPrograms(const std::string & cuFileName,
        unsigned int entryPointIndex,
        const std::string & rayGenerationProgramName,
        const std::string & missProgramName,
        const std::string & exceptionProgramName)
{
    // create pixel sampling programs
    std::string ptxPath = ptxpath("MaoPPM", cuFileName);

    Program rayGenerationProgram = getContext()->createProgramFromPTXFile(ptxPath, rayGenerationProgramName);
    getContext()->setRayGenerationProgram(entryPointIndex, rayGenerationProgram);

    Program exceptionProgram = getContext()->createProgramFromPTXFile(ptxPath, exceptionProgramName);
    getContext()->setExceptionProgram(entryPointIndex, exceptionProgram);

    Program missProgram = getContext()->createProgramFromPTXFile(ptxPath, missProgramName);
    getContext()->setMissProgram(entryPointIndex, missProgram);
}   /* -----  end of method Scene::setPrograms  ----- */



void Scene::createPhotonMap()
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
}   /* -----  end of method Scene::createPhotonMap  ----- */



void Scene::buildPhotonMapAcceleration(MaoPPM::Photon * photonList,
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
}   /* -----  end of method Scene::buildPhotonMapAcceleration ----- */



void Scene::generateSamples(const uint nSamples, optix::Buffer & sampleList)
{
    // Check size.
    RTsize N = 0;
    sampleList->getSize(N);
    if (N < nSamples)
        sampleList->setSize(nSamples);

    // Generate samples.
    float * sampleListPtr = static_cast<float *>(sampleList->map());
    for (uint i = 0; i < nSamples; i++)
        sampleListPtr[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    sampleList->unmap();
}   /* -----  end of method Scene::generateSamples  ----- */



void Scene::trace(const RayGenCameraData & cameraData)
{
    // re-run the pixel sampling pass if camera has changed
    if (isCameraChanged()) {
        m_nEmittedPhotons = 0u;
        setIsCameraChanged(false);
        getContext()["cameraPosition"]->setFloat(cameraData.eye);
        getContext()["cameraU"]->setFloat(cameraData.U);
        getContext()["cameraV"]->setFloat(cameraData.V);
        getContext()["cameraW"]->setFloat(cameraData.W);
        getContext()["launchSize"]->setUint(m_width, m_height);
        getContext()->launch(PixelSamplingPass, m_width, m_height);
    }

    generateSamples(PHOTON_WIDTH * PHOTON_HEIGHT * PHOTON_COUNT * 2, m_sampleList);
    getContext()["launchSize"]->setUint(PHOTON_WIDTH, PHOTON_HEIGHT);
    getContext()->launch(PhotonShootingPass, PHOTON_WIDTH, PHOTON_HEIGHT);
    createPhotonMap();

    getContext()["launchSize"]->setUint(m_width, m_height);
    getContext()["nEmittedPhotons"]->setUint(m_nEmittedPhotons);
    getContext()->launch(GatheringPass, m_width, m_height);
}   /* -----  end of method Scene::trace  ----- */
