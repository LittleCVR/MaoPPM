/*
 * =============================================================================
 *
 *       Filename:  pathTracingPassPrograms.cu
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  2011-06-08 19:19:55
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
 *  Header files from OptiX
 *----------------------------------------------------------------------------*/
#include    <optix_world.h>

/*----------------------------------------------------------------------------
 *  header files of our own
 *----------------------------------------------------------------------------*/
#include    "global.h"

/*----------------------------------------------------------------------------
 *  namespace
 *----------------------------------------------------------------------------*/
using namespace optix;
using namespace MaoPPM;



rtDeclareVariable(uint2, launchIndex, rtLaunchIndex, );
rtDeclareVariable(uint2, launchSize ,              , );

rtBuffer<float4, 2>  outputBuffer;

rtDeclareVariable(rtObject, rootObject, , );
rtDeclareVariable(float3, geometricNormal, attribute geometric_normal, ); 
rtDeclareVariable(float3, shadingNormal  , attribute shading_normal  , ); 

rtDeclareVariable(float3, cameraPosition, , );
rtDeclareVariable(float3, cameraU       , , );
rtDeclareVariable(float3, cameraV       , , );
rtDeclareVariable(float3, cameraW       , , );

rtDeclareVariable(float, rayEpsilon , , );
rtDeclareVariable(PathTracingRenderer::RadianceRayPayload, radianceRayPayload, rtPayload, );
rtDeclareVariable(PathTracingRenderer::ShadowRayPayload  , shadowRayPayload  , rtPayload, );

rtDeclareVariable(uint, frameCount, , );



/* 
 * ===  FUNCTION  ==============================================================
 *         Name:  generateRay
 *  Description:  The ray generaton function of the pixel sampling program.
 * =============================================================================
 */
RT_PROGRAM void generateRay()
{
    /* :TODO:2011-05-16 14:48:04:: should make this a random sample */
    float2 screenSize = make_float2(outputBuffer.size());
    float2 sample = make_float2(0.5f, 0.5f); 

    float2 cameraRayDirection = (make_float2(launchIndex) + sample) / screenSize * 2.0f - 1.0f;
    float3 worldRayDirection = normalize(cameraRayDirection.x*cameraU + cameraRayDirection.y*cameraV + cameraW);

    Ray ray(cameraPosition, worldRayDirection, PathTracingRenderer::RadianceRay, rayEpsilon);
    PathTracingRenderer::RadianceRayPayload payload;
    payload.depth           = 0;
    payload.radiance        = make_float3(0.0f);
    payload.sampleIndexBase = 2 * DEFAULT_MAX_RAY_DEPTH* ((launchIndex.y * launchSize.x) + launchIndex.x);
    rtTrace(rootObject, ray, payload);

    if (frameCount == 0)
        outputBuffer[launchIndex] = make_float4(0.0f);
    float frame = static_cast<float>(frameCount);
    outputBuffer[launchIndex] = (1.0f / (frame + 1.0f)) * make_float4(payload.radiance, 1.0f) +
        (frame / (frame + 1.0f)) * outputBuffer[launchIndex];
}   /* -----  end of function generateRay  ----- */



/* 
 * ===  FUNCTION  ==============================================================
 *         Name:  handleRadianceRayMiss
 *  Description:  
 * =============================================================================
 */
RT_PROGRAM void handleRadianceRayMiss()
{
    radianceRayPayload.radiance = make_float3(0.0f);
}   /* -----  end of function handleRadianceRayMiss  ----- */
