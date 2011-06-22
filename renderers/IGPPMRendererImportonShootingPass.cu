/*
 * =============================================================================
 *
 *       Filename:  IGPPMRenderer.cu
 *
 *    Description:  The Importons Guided Progressive Photon Map Renderer.
 *
 *        Version:  1.0
 *        Created:  2011-06-21 08:02:37
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
 *  Header files from OptiX
 *----------------------------------------------------------------------------*/
#include    <optix_world.h>

/*----------------------------------------------------------------------------
 *  header files of our own
 *----------------------------------------------------------------------------*/
#include    "global.h"
#include    "payload.h"
#include    "reflection.h"
#include    "utility.h"
#include    "Light.h"

/*----------------------------------------------------------------------------
 *  namespace
 *----------------------------------------------------------------------------*/
using namespace optix;
using namespace MaoPPM;



typedef IGPPMRenderer::PixelSample  PixelSample;
typedef IGPPMRenderer::Importon     Importon;
typedef IGPPMRenderer::Photon       Photon;



rtDeclareVariable(uint2, launchIndex, rtLaunchIndex, );
rtDeclareVariable(uint2, launchSize ,              , );

rtBuffer<float4,      2>  outputBuffer;
rtBuffer<PixelSample, 2>  pixelSampleList;
rtBuffer<Importon,    1>  importonList;
rtBuffer<float,       1>  sampleList;
rtDeclareVariable(uint, nSamplesPerThread  , , );
rtDeclareVariable(uint, nImportonsPerThread, , );

rtDeclareVariable(rtObject, rootObject, , );

rtDeclareVariable(float, rayEpsilon, , );
rtDeclareVariable(NormalRayPayload, normalRayPayload, rtPayload, );
rtDeclareVariable(ShadowRayPayload, shadowRayPayload, rtPayload, );



/* 
 * ===  FUNCTION  ==============================================================
 *         Name:  shootImportons
 *  Description:  
 * =============================================================================
 */
RT_PROGRAM void shootImportons()
{
    outputBuffer[launchIndex] = make_float4(0.0f);

    PixelSample & pixelSample = pixelSampleList[launchIndex];
    if (!pixelSample.isHit) return;

    uint offset = launchIndexOffset(launchIndex, launchSize);
    uint sampleIndex   = offset * nSamplesPerThread;
    uint importonIndex = offset * nImportonsPerThread;

    /*TODO: write sampleF */
    Intersection & intersection = pixelSample.intersection;
    BSDF * bsdf = intersection.bsdf();
    Lambertian & bxdf = reinterpret_cast<Lambertian &>(bsdf->m_bxdfList[0]);
    Matrix4x4 * worldToObject = intersection.worldToObject();
    float3 wo = transformVector(*worldToObject, pixelSample.wo);
    for (uint i = 0; i < nImportonsPerThread; i++) {
        float3 wi;
        float  probability;
        float2 sample = GET_SAMPLES(sampleList, sampleIndex, sample);
        bxdf.sampleF(wo, &wi, sample, &probability);
        float3 wwi = transformVector(worldToObject->transpose(), wi);
        if (launchIndex.x == 100 && launchIndex.y == 20) {
            rtPrintf("launch index = (%u, %u)\n", launchIndex.x, launchIndex.y);
            rtPrintf("importon probability: %f\n", probability);
            rtPrintf("importon: world wo = (%+4.4f, %+4.4f, %+4.4f), wo = (%+4.4f, %+4.4f, %+4.4f),\n"
                     "          world wi = (%+4.4f, %+4.4f, %+4.4f), wi = (%+4.4f, %+4.4f, %+4.4f)\n",
                    pixelSample.wo.x, pixelSample.wo.y, pixelSample.wo.z, wo.x, wo.y, wo.z,
                    wwi.x, wwi.y, wwi.z, wi.x, wi.y, wi.z);
        }

        // trace
        Ray ray(intersection.dg()->point, wwi, NormalRay, rayEpsilon);
        NormalRayPayload payload;
        rtTrace(rootObject, ray, payload);

        // importon
        Importon & importon = importonList[importonIndex++];
        importon.isHit      = payload.isHit;
        if (!importon.isHit) continue;
        importon.weight         = 1.0f / probability;
        importon.intersection   = payload.intersection;
        importon.wo             = -wwi;
        importon.flux           = make_float3(0.0f);
        importon.nPhotons       = 0;
        importon.radiusSquared  = 64.0f;
        outputBuffer[launchIndex] = make_float4(probability);
    }
}   /* -----  end of function shootImportons  ----- */
