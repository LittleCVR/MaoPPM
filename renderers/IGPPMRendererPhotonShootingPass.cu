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

rtBuffer<Light,       1>  lightList;
rtBuffer<Photon,      1>  photonList;
rtBuffer<Photon,      1>  photonMap;
rtBuffer<float,       1>  sampleList;
rtDeclareVariable(uint, nSamplesPerThread  , , );
rtDeclareVariable(uint, nImportonsPerThread, , );
rtDeclareVariable(uint, nPhotonsPerThread  , , );
rtDeclareVariable(uint, nEmittedPhotons    , , ) = 0;

rtDeclareVariable(rtObject, rootObject, , );
rtDeclareVariable(float3, geometricNormal, attribute geometric_normal, ); 
rtDeclareVariable(float3, shadingNormal  , attribute shading_normal  , ); 

rtDeclareVariable(float3, cameraPosition, , );
rtDeclareVariable(float3, cameraU       , , );
rtDeclareVariable(float3, cameraV       , , );
rtDeclareVariable(float3, cameraW       , , );

rtDeclareVariable(float, rayEpsilon, , );
rtDeclareVariable(NormalRayPayload, normalRayPayload, rtPayload, );
rtDeclareVariable(ShadowRayPayload, shadowRayPayload, rtPayload, );



/* 
 * ===  FUNCTION  ==============================================================
 *         Name:  shootPhotons
 *  Description:  
 * =============================================================================
 */
RT_PROGRAM void shootPhotons()
{
    uint offset = launchIndexOffset(launchIndex, launchSize);
    uint sampleIndex = nSamplesPerThread * offset;
    uint photonIndex = nPhotonsPerThread * offset;

    // reset photon list
    for (uint i = 0; i < nPhotonsPerThread; i++) {
        Photon & photon = photonList[photonIndex+i];
        photon.flux = make_float3(0.0f);
    }

    Ray ray;
    float3 wo, wi, flux;
    NormalRayPayload payload;
    bool hitAtLeastOnce = false;
    for (uint i = 0; i < nPhotonsPerThread; i++) {
        // starts from lights
        if (!hitAtLeastOnce) {
            // sample light
            const Light & light = lightList[0];
            flux = light.flux;
            // sample direction
            float2 sample = make_float2(
                    sampleList[sampleIndex+0], sampleList[sampleIndex+1]);
            sampleIndex += 2;
            wo = sampleUniformSphere(sample);
            ray = Ray(light.position, wo, NormalRay, rayEpsilon);
        }
        // starts from surface
        else {
            /*TODO*/
            Intersection & intersection = payload.intersection;
            BSDF * bsdf = intersection.bsdf();
            Lambertian & bxdf = reinterpret_cast<Lambertian &>(bsdf->m_bxdfList[0]);
            Matrix4x4 * worldToObject = intersection.worldToObject();
            wi = transformVector(*worldToObject, -wo);
            float  probability;
            float2 sample = make_float2(
                    sampleList[sampleIndex+0], sampleList[sampleIndex+1]);
            sampleIndex += 2;
            // remember that we are now shooting rays from a light
            // thus wo and wi must be swapped
            float3 f = bxdf.sampleF(wi, &wo, sample, &probability);
            flux = pairwiseMul(f, flux);
            if (launchIndex.x == 64 && launchIndex.y == 64) {
                rtPrintf("photon: f = (%4.4f, %4.4f, %4.4f)\n", f.x, f.y, f.z);
            }

            // transform from object to world
            // remember that this transform's transpose is actually its inverse
            wo = transformVector(worldToObject->transpose(), wo);
            ray = Ray(intersection.dg()->point, wo, NormalRay, rayEpsilon);
        }

        // trace ray
        rtTrace(rootObject, ray, payload);
        if (payload.isHit)
            hitAtLeastOnce = true;
        else {
            // give up this sample pair and continue
            sampleIndex += 2;
            continue;
        }

        // create photon
        Intersection & intersection = payload.intersection;
        Photon & photon = photonList[photonIndex++];
        photon.position = intersection.dg()->point;
        photon.normal   = intersection.dg()->normal;
        photon.wi       = -wo;
        photon.flux     = flux;
    }
}   /* -----  end of function shootPhotons  ----- */
