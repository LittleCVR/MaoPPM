/*
 * =============================================================================
 *
 *       Filename:  pixelSamplingProgram.cu
 *
 *    Description:  This file contains functions to finish the pixel sampling
 *                  pass.
 *
 *        Version:  1.0
 *        Created:  2011-03-29 15:46:55
 *
 *         Author:  Chun-Wei Huang (LittleCVR), 
 *        Company:  Communication & Multimedia Laboratory,
 *                  Department of Computer Science & Information Engineering,
 *                  National Taiwan University
 *
 * =============================================================================
 */

/*----------------------------------------------------------------------------
 *  Header files from OptiX
 *----------------------------------------------------------------------------*/
#include    <optix_world.h>

/*----------------------------------------------------------------------------
 *  header files of our own
 *----------------------------------------------------------------------------*/
#include    "global.h"
#include    "PPMRenderer.h"

/*----------------------------------------------------------------------------
 *  namespace
 *----------------------------------------------------------------------------*/
using namespace optix;
using namespace MaoPPM;



/*----------------------------------------------------------------------------
 *  buffers
 *----------------------------------------------------------------------------*/
rtBuffer<float4,                   2>      outputBuffer;
rtBuffer<PPMRenderer::PixelSample, 2>      pixelSampleList;

/*----------------------------------------------------------------------------
 *  variables
 *----------------------------------------------------------------------------*/
rtDeclareVariable(float,    rayEpsilon, , );
rtDeclareVariable(rtObject, rootObject, , );

rtDeclareVariable(float3, cameraPosition, , );
rtDeclareVariable(float3, cameraU       , , );
rtDeclareVariable(float3, cameraV       , , );
rtDeclareVariable(float3, cameraW       , , );

rtDeclareVariable(Ray  , currentRay, rtCurrentRay          , );
rtDeclareVariable(float, tHit      , rtIntersectionDistance, );
rtDeclareVariable(PPMRenderer::PixelSamplingRayPayload,  currentRayPayload,  rtPayload, );

rtDeclareVariable(uint2 ,   launchIndex    ,    rtLaunchIndex, );
rtDeclareVariable(float3,   geometricNormal,    attribute geometric_normal, ); 
rtDeclareVariable(float3,   shadingNormal  ,    attribute shading_normal  , ); 



/* 
 * ===  FUNCTION  ==============================================================
 *         Name:  generateRay
 *  Description:  The ray generaton function of the pixel sampling program.
 * =============================================================================
 */
RT_PROGRAM void generateRay()
{
    // clear previous buffer
    PPMRenderer::PixelSample & pixelSample = pixelSampleList[launchIndex];
    pixelSample.flux          = make_float3(0.0f);
    pixelSample.flags         = 0u;
    pixelSample.nPhotons      = 0u;
    pixelSample.radiusSquared = 15.0f;

    /* :TODO:2011-05-16 14:48:04:: should make this a random sample */
    float2 screenSize = make_float2(outputBuffer.size());
    float2 sample = make_float2(0.5f, 0.5f); 

    float2 cameraRayDirection = (make_float2(launchIndex) + sample) / screenSize * 2.0f - 1.0f;
    float3 worldRayDirection = normalize(cameraRayDirection.x*cameraU + cameraRayDirection.y*cameraV + cameraW);

    Ray ray(cameraPosition, worldRayDirection, PPMRenderer::PixelSamplingRay, rayEpsilon);
    PPMRenderer::PixelSamplingRayPayload payload;
    payload.attenuation = 1.0f;
    rtTrace(rootObject, ray, payload);
}   /* -----  end of function generateRay  ----- */



/* 
 * ===  FUNCTION  ==============================================================
 *         Name:  handleException
 *  Description:  
 * =============================================================================
 */
RT_PROGRAM void handleException()
{
}   /* -----  end of function handleException  ----- */



/* 
 * ===  FUNCTION  ==============================================================
 *         Name:  handlePixelSamplingRayMiss
 *  Description:  
 * =============================================================================
 */
RT_PROGRAM void handlePixelSamplingRayMiss()
{
    currentRayPayload.attenuation = 0.0f;
}   /* -----  end of function handlePixelSamplingMiss  ----- */
