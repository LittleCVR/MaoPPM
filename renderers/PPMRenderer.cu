/*
 * =============================================================================
 *
 *       Filename:  PPMRenderer.cu
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  2011-06-27 11:09:25
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
 *  Header files from OptiX
 *----------------------------------------------------------------------------*/
#include    <optix_world.h>

/*----------------------------------------------------------------------------
 *  header files from std C/C++
 *----------------------------------------------------------------------------*/
#include    <limits>

/*----------------------------------------------------------------------------
 *  header files of our own
 *----------------------------------------------------------------------------*/
#include    "global.h"
#include    "payload.h"
#include    "utility.h"
#include    "BSDF.h"
#include    "Light.h"

/*----------------------------------------------------------------------------
 *  namespace
 *----------------------------------------------------------------------------*/
using namespace optix;
using namespace MaoPPM;



typedef PPMRenderer::PixelSample  PixelSample;
typedef MaoPPM::Photon            Photon;



rtDeclareVariable(uint2, launchIndex, rtLaunchIndex, );
rtDeclareVariable(uint2, launchSize ,              , );

rtBuffer<float4,      2>  outputBuffer;
rtBuffer<Light,       1>  lightList;
rtBuffer<PixelSample, 2>  pixelSampleList;
rtBuffer<Photon,      1>  photonList;
rtBuffer<Photon,      1>  photonMap;
rtBuffer<float,       1>  sampleList;

rtDeclareVariable(uint, frameCount         , , );
rtDeclareVariable(uint, nSamplesPerThread  , , );
rtDeclareVariable(uint, nPhotonsPerThread  , , );
rtDeclareVariable(uint, nEmittedPhotons    , , );
rtDeclareVariable(uint, maxRayDepth        , , );

rtDeclareVariable(rtObject, rootObject, , );
rtDeclareVariable(float,    rayEpsilon, , );

rtDeclareVariable(float3, cameraPosition, , );
rtDeclareVariable(float3, cameraU       , , );
rtDeclareVariable(float3, cameraV       , , );
rtDeclareVariable(float3, cameraW       , , );

rtDeclareVariable(NormalRayPayload, normalRayPayload, rtPayload, );
rtDeclareVariable(ShadowRayPayload, shadowRayPayload, rtPayload, );



/* 
 * ===  FUNCTION  ==============================================================
 *         Name:  generatePixelSamples
 *  Description:  
 * =============================================================================
 */
RT_PROGRAM void generatePixelSamples()
{
    if (frameCount != 0) return;

    // Clear output buffer.
    if (frameCount == 0)
        outputBuffer[launchIndex] = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
    // Clear pixel sample.
    PixelSample & pixelSample = pixelSampleList[launchIndex];
    pixelSample.reset();

    /* TODO: move this task to the camera class */
    // Generate camera ray.
    Ray ray;
    {
        float2 screenSize = make_float2(outputBuffer.size());
        float2 sample = make_float2(0.5f, 0.5f); 
        float2 cameraRayDirection = (make_float2(launchIndex) + sample) / screenSize * 2.0f - 1.0f;
        float3 worldRayDirection = normalize(cameraRayDirection.x*cameraU + cameraRayDirection.y*cameraV + cameraW);
        ray = Ray(cameraPosition, worldRayDirection, NormalRay, rayEpsilon);
    }

    // Intersect with the scene.
    NormalRayPayload payload;
    payload.reset();
    rtTrace(rootObject, ray, payload);
    if (!payload.isHit) return;
    pixelSample.flags |= PixelSample::isHit;

    // Fill pixel sample data if hit.
    pixelSample.setIntersection(payload.intersection());
    pixelSample.wo        = -ray.direction;
    pixelSample.direct    = make_float3(0.0f);
    pixelSample.flux      = make_float3(0.0f);
    pixelSample.nPhotons  = 0;
    /* TODO: hard coded */
    pixelSample.radiusSquared  = 32.0f;

    /* TODO: move this task to the light class */
    // Evaluate direct illumination.
    float3 Li;
    Intersection * intersection = pixelSample.intersection();
    {
        const Light * light = &lightList[0];
        float3 shadowRayDirection = light->position - intersection->dg()->point;
        float3 wi = normalize(shadowRayDirection);
        float distanceSquared = dot(shadowRayDirection, shadowRayDirection);
        float distance = sqrtf(distanceSquared);

        ShadowRayPayload shadowRayPayload;
        shadowRayPayload.reset();
        ray = Ray(intersection->dg()->point, wi, ShadowRay, rayEpsilon, distance-rayEpsilon);
        rtTrace(rootObject, ray, shadowRayPayload);
        if (shadowRayPayload.isHit) return;

        BSDF bsdf; intersection->getBSDF(&bsdf);
        float3 f = bsdf.f(pixelSample.wo, wi);
        Li = f * light->flux  * fabsf(dot(wi, intersection->dg()->normal))
            / (4.0f * M_PIf * distanceSquared);
    }

    pixelSample.direct = Li;
}   /* -----  end of function generatePixelSamples  ----- */



/* 
 * ===  FUNCTION  ==============================================================
 *         Name:  shootPhotons
 *  Description:  
 * =============================================================================
 */
RT_PROGRAM void shootPhotons()
{
    uint offset = LAUNCH_OFFSET_2D(launchIndex, launchSize);
    uint sampleIndex = nSamplesPerThread * offset;
    uint photonIndex = nPhotonsPerThread * offset;

    // Clear photon list.
    for (uint i = 0; i < nPhotonsPerThread; i++)
        photonList[photonIndex+i].reset();

    Ray ray;
    NormalRayPayload payload;
    uint depth = 0;
    float3 wo, wi, flux;
    Intersection * intersection  = NULL;
    BSDF bsdf;
    for (uint i = 0; i < nPhotonsPerThread; i++) {
        // starts from lights
        if (depth == 0) {
            /* TODO: sample random light */
            // sample light
            const Light & light = lightList[0];
            flux = light.flux;
            // sample direction
            float2 sample = GET_2_SAMPLES(sampleList, sampleIndex);
            wo = sampleUniformSphere(sample);
            ray = Ray(light.position, wo, NormalRay, rayEpsilon);
        }
        // starts from surface
        else {
            float  probability;
            float3 sample = GET_3_SAMPLES(sampleList, sampleIndex);
            // remember that we are now shooting rays from a light
            // thus wo and wi must be swapped
            float3 f = bsdf.sampleF(wi, &wo, sample, &probability);
            if (probability == 0.0f) continue;
            flux = f * flux * fabsf(dot(wo, intersection->dg()->normal)) / probability;
            // transform from object to world
            // remember that this transform's transpose is actually its inverse
            ray = Ray(intersection->dg()->point, wo, NormalRay, rayEpsilon);
        }

        // trace ray
        payload.reset();
        rtTrace(rootObject, ray, payload);
        if (!payload.isHit) continue;
        wi = -wo;
        intersection  = payload.intersection();
        intersection->getBSDF(&bsdf);

        // create photon
        Photon & photon = photonList[photonIndex+i];
        photon.reset();
        if (depth == 0)
            photon.flags |= Photon::Direct;
        else
            photon.flags |= Photon::Indirect;
        photon.position = intersection->dg()->point;
        photon.wi       = wi;
        photon.flux     = flux;

        // Increase depth, reset if necessary.
        ++depth;
        if (depth % maxRayDepth == 0)
            depth = 0;
    }
}   /* -----  end of function shootPhotons  ----- */



/* 
 * ===  FUNCTION  ==============================================================
 *         Name:  
 *  Description:  
 * =============================================================================
 */
RT_PROGRAM void estimateDensity()
{
    // Do not have to gather photons if pixel sample was not hit.
    PixelSample & pixelSample = pixelSampleList[launchIndex];
    if (!(pixelSample.flags & PixelSample::isHit)) return;

    // Gather.
    Intersection * intersection = pixelSample.intersection();
    BSDF bsdf; intersection->getBSDF(&bsdf);
    uint nAccumulatedPhotons = 0;
    float3 flux = make_float3(0.0f);
    float maxDistanceSquared = pixelSample.radiusSquared;
    // First time should use LimitedPhotonGatherer to find initial radius.
    // Otherwise just gather all the photons in range.
    if (frameCount == 0) {
        Index gatheredPhotonListIndex =
            LOCAL_HEAP_ALLOC_SIZE(nPhotonsUsed * sizeof(GatheredPhoton));
        GatheredPhoton * gatheredPhotonList =
            LOCAL_HEAP_GET_OBJECT_POINTER(GatheredPhoton, gatheredPhotonListIndex);
        flux = LimitedPhotonGatherer::accumulateFlux(
                intersection->dg()->point, pixelSample.wo, &bsdf,
                &photonMap[0], &maxDistanceSquared, &nAccumulatedPhotons,
                gatheredPhotonList);
        // maxDistanceSquared may be shrinked. So write it back to pixelSample.
        pixelSample.radiusSquared = maxDistanceSquared;
    }
    else {  // frameCount != 0
        flux = PhotonGatherer::accumulateFlux(
                intersection->dg()->point, pixelSample.wo, &bsdf,
                &photonMap[0], &maxDistanceSquared, &nAccumulatedPhotons);
    }

    pixelSample.shrinkRadius(flux, nAccumulatedPhotons);

    float3 indirect = pixelSample.flux / (M_PIf * pixelSample.radiusSquared) / nEmittedPhotons;
    outputBuffer[launchIndex] = make_float4(pixelSample.direct + indirect, 1.0f);
}   /* -----  end of function gatherPhotons  ----- */
