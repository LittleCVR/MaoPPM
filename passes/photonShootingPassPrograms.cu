/*
 * =====================================================================================
 *
 *       Filename:  photonShooting.cu
 *
 *    Description:  This file contains functions to finish the photon shooting pass.
 *
 *        Version:  1.0
 *        Created:  2011-03-30 21:16:51
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

/*-----------------------------------------------------------------------------
 *  buffers
 *-----------------------------------------------------------------------------*/
rtBuffer<Photon, 1>  photonList;
rtBuffer<Light,  1>  lightList;
rtBuffer<float,  1>  sampleList;

/*-----------------------------------------------------------------------------
 *  variables
 *-----------------------------------------------------------------------------*/
rtDeclareVariable(float,    rayEpsilon, , );
rtDeclareVariable(rtObject, rootObject, , );

rtDeclareVariable(uint2, launchIndex, rtLaunchIndex, );
rtDeclareVariable(uint2, launchSize ,              , );





/* #####   PROTOTYPES  -  LOCAL TO THIS SOURCE FILE   ############################### */

/* #####   FUNCTION DEFINITIONS  -  EXPORTED FUNCTIONS   ############################ */

/* #####   FUNCTION DEFINITIONS  -  LOCAL TO THIS SOURCE FILE   ##################### */

/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  generateRay
 *  Description:  The ray generaton function of the pixel sampling program.
 * =====================================================================================
 */
RT_PROGRAM void generateRay()
{
    // clear the photon map
    uint photonIndexBase = (launchIndex.y*launchSize.x + launchIndex.x) * PHOTON_COUNT;
    for (uint i = photonIndexBase; i < photonIndexBase + PHOTON_COUNT; ++i) {
        Photon & photon = photonList[i];
        photon.flux = make_float3(0.0f);
    }

    // pick up a light
    uint sampleIndexBase = (launchIndex.y*launchSize.x + launchIndex.x) * (PHOTON_COUNT*2);
    const Light & light = lightList[0];

    // shoot N photons
    uint sampleIndex = sampleIndexBase;
    // pick a starting point on the light
    float3 position = light.position;

    // sample the direction
    float2 sample = make_float2(sampleList[sampleIndex], sampleList[sampleIndex+1]);
    sampleIndex += 2;

    // sample sphere uniformly
    float3 direction = sampleSphereUniformly(sample);

    Ray ray(position, direction, PhotonShootingRay, rayEpsilon);
    PhotonShootingRayPayload payload;
    payload.nPhotons        = 0u;
    payload.photonIndexBase = photonIndexBase;
    payload.sampleIndexBase = sampleIndex;
    payload.attenuation = 1.0f;
    payload.depth       = 0u;
    payload.flux        = light.flux;
    rtTrace(rootObject, ray, payload);
}   /* -----  end of function generateRay  ----- */



/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  handleException
 *  Description:  
 * =====================================================================================
 */
RT_PROGRAM void handleException()
{
}   /* -----  end of function handleException  ----- */



/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  handlePhotonShootingRayMiss
 *  Description:  
 * =====================================================================================
 */
RT_PROGRAM void handlePhotonShootingRayMiss()
{
}   /* -----  end of function handlePhotonShootingRayMiss  ----- */
