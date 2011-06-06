/*
 * =====================================================================================
 *
 *       Filename:  matteMaterialPrograms.cu
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  2011-03-29 15:46:55
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
 *  Header files from OptiX
 *-----------------------------------------------------------------------------*/
#include    <optix_world.h>

/*-----------------------------------------------------------------------------
 *  header files of our own
 *-----------------------------------------------------------------------------*/
#include    "global.h"
#include    "math.cu"
#include    "samplers.cu"

/*-----------------------------------------------------------------------------
 *  namespace
 *-----------------------------------------------------------------------------*/
using namespace optix;
using namespace MaoPPM;





/* #####   MACROS  -  LOCAL TO THIS SOURCE FILE   ################################### */

/* #####   TYPE DEFINITIONS  -  LOCAL TO THIS SOURCE FILE   ######################### */

/* #####   DATA TYPES  -  LOCAL TO THIS SOURCE FILE   ############################### */

/* #####   VARIABLES  -  LOCAL TO THIS SOURCE FILE   ################################ */

rtBuffer<PixelSample, 2>  pixelSampleList;
rtBuffer<Photon,      1>  photonList;
rtBuffer<float,       1>  sampleList;

rtDeclareVariable(float,    rayEpsilon, , );
rtDeclareVariable(rtObject, rootObject, , );

rtDeclareVariable(Ray  , currentRay, rtCurrentRay          , );
rtDeclareVariable(float, tHit      , rtIntersectionDistance, );

rtDeclareVariable(PixelSamplingRayPayload , pixelSamplingRayPayload , rtPayload, );
rtDeclareVariable(PhotonShootingRayPayload, photonShootingRayPayload, rtPayload, );
rtDeclareVariable(GatheringRayPayload     , gatheringRayPayload     , rtPayload, );

rtDeclareVariable(uint2, launchIndex, rtLaunchIndex, );
rtDeclareVariable(uint2, launchSize ,              , );

rtDeclareVariable(float3, geometricNormal, attribute geometric_normal, ); 
rtDeclareVariable(float3, shadingNormal  , attribute shading_normal  , ); 

rtDeclareVariable(float3, Kd, , );





/* #####   PROTOTYPES  -  LOCAL TO THIS SOURCE FILE   ############################### */

/* #####   FUNCTION DEFINITIONS  -  EXPORTED FUNCTIONS   ############################ */

/* #####   FUNCTION DEFINITIONS  -  LOCAL TO THIS SOURCE FILE   ##################### */

/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  handlePixelSamplingRayClosestHit
 *  Description:  
 * =====================================================================================
 */
RT_PROGRAM void handlePixelSamplingRayClosestHit()
{
    float3 origin    = currentRay.origin;
    float3 direction = currentRay.direction;
    float3 worldShadingNormal   = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shadingNormal  ));
    float3 worldGeometricNormal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, geometricNormal));
    float3 ffnormal  = faceforward(worldShadingNormal, -direction, worldGeometricNormal);
    float3 diff      = tHit * direction;
    float3 hitPoint  = origin + diff;

    PixelSample & pixelSample = pixelSampleList[launchIndex];
    pixelSample.flags             |= PIXEL_SAMPLE_HIT;
    pixelSample.position           = hitPoint; 
    pixelSample.incidentDirection  = -direction;
    pixelSample.normal             = ffnormal;

    pixelSample.material  = 1;
    pixelSample.Kd        = Kd;
}   /* -----  end of function handlePixelSamplingRayClosestHit  ----- */



/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  handlePhotonShootingRayClosestHit
 *  Description:  
 * =====================================================================================
 */
RT_PROGRAM void handlePhotonShootingRayClosestHit()
{
    float3 origin    = currentRay.origin;
    float3 direction = currentRay.direction;
    float3 worldShadingNormal   = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shadingNormal  ));
    float3 worldGeometricNormal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, geometricNormal));
    float3 ffnormal  = faceforward(worldShadingNormal, -direction, worldGeometricNormal);
    float3 hitPoint  = origin + tHit*direction;

    // record this photon if it had bounced at least once
    PhotonShootingRayPayload & payload = photonShootingRayPayload;
    if (payload.depth > 0u) {
        Photon & photon = photonList[payload.photonIndexBase];
        photon.position          = hitPoint;
        photon.flux              = payload.flux;
        photon.normal            = ffnormal;
        photon.incidentDirection = -direction;
        ++payload.nPhotons;
        ++payload.photonIndexBase;
    }

    // check if there're enough photons or the ray depth is too deep
    if (payload.nPhotons >= PHOTON_COUNT || payload.depth >= MAX_RAY_DEPTH)
        return;

    // otherwise, samples a new direction
    float3 W = normalize(ffnormal);
    float3 U = cross(W, make_float3(0.0f, 1.0f, 0.0f));
    if (fabsf(U.x) < 0.001f && fabsf(U.y) < 0.001f && fabsf(U.z) < 0.001f)
        U = cross(W, make_float3(1.0f, 0.0f, 0.0f));
    U = normalize(U);
    float3 V = cross(W, U);

    uint sampleIndex = payload.sampleIndexBase;
    float2 sample = make_float2(sampleList[sampleIndex], sampleList[sampleIndex+1]);
    payload.sampleIndexBase += 2;

    // Tweak sample according to cosine term.
    sample.x = asinf(sample.x * 2.0f - 1.0f) / M_PIf + 0.5f;
    sample.y = asinf(sample.y * 2.0f - 1.0f) / M_PIf + 0.5f;
    float3 newDirection = sampleHemisphereUniformly(sample);
    newDirection = newDirection.x*U + newDirection.y*V + newDirection.z*W;

    Ray ray(hitPoint, newDirection, PhotonShootingRay, rayEpsilon);
    payload.depth += 1;
    payload.flux   = pairwiseMul(Kd, payload.flux);
    rtTrace(rootObject, ray, payload);
}   /* -----  end of function handlePhotonShootingRayClosestHit  ----- */



/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  handleGatheringRayAnyHit
 *  Description:  
 * =====================================================================================
 */
RT_PROGRAM void handleGatheringRayAnyHit()
{
    gatheringRayPayload.attenuation = 0.0f;
    rtTerminateRay();
}   /* -----  end of function handleGatheringRayAnyHit  ----- */
