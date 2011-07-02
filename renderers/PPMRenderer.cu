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
 *  header files of our own
 *----------------------------------------------------------------------------*/
#include    "global.h"
#include    "payload.h"
#include    "utility.h"
#include    "BSDF.h"
#include    "Camera.h"
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
rtBuffer<PixelSample, 2>  pixelSampleList;
rtBuffer<Photon,      1>  photonList;
rtBuffer<Photon,      1>  photonMap;
rtBuffer<float,       1>  sampleList;

rtDeclareVariable(uint, frameCount         , , );
rtDeclareVariable(uint, nSamplesPerThread  , , );
rtDeclareVariable(uint, nPhotonsPerThread  , , );
rtDeclareVariable(uint, nEmittedPhotons    , , );
rtDeclareVariable(uint, maxRayDepth        , , );

rtDeclareVariable(Camera, camera, , );

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
    // Clear output buffer.
    if (frameCount == 0)
        outputBuffer[launchIndex] = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
    // Clear pixel sample.
    PixelSample & pixelSample = pixelSampleList[launchIndex];
    pixelSample.reset();

    // Generate camera ray.
    unsigned int offset = LAUNCH_OFFSET_2D(launchIndex, launchSize);
    unsigned int sampleIndex = nSamplesPerThread * offset;
    float2 sample = GET_2_SAMPLES(sampleList, sampleIndex);
    Ray ray = camera.generateCameraRay(
            launchIndex.x, launchIndex.y, sample, NormalRay, rayEpsilon);

    // Allocate memory for intersection.
    Intersection * intersection = LOCAL_HEAP_GET_OBJECT_POINTER(Intersection,
            LOCAL_HEAP_GET_CURRENT_INDEX() + sizeof(Intersection) * offset);
    // Trace until non-specular surface.
    BSDF bsdf;
    uint depth = 0;
    if (!traceUntilNonSpecularSurface(&ray, maxRayDepth, &depth,
                intersection, &bsdf, &pixelSample.wo, &pixelSample.throughput))
    {
        return;
    }
    pixelSample.flags |= PixelSample::isHit;
    pixelSample.setIntersection(intersection);
}   /* -----  end of function generatePixelSamples  ----- */



/* 
 * ===  FUNCTION  ==============================================================
 *         Name:  shootPhotons
 *  Description:  
 * =============================================================================
 */
RT_PROGRAM void shootPhotons()
{
    unsigned int offset = LAUNCH_OFFSET_2D(launchIndex, launchSize);
    unsigned int sampleIndex = nSamplesPerThread * offset;
    unsigned int photonIndex = nPhotonsPerThread * offset;

    // Clear photon list.
    for (uint i = 0; i < nPhotonsPerThread; i++)
        photonList[photonIndex+i].reset();

    // Allocate memory for intersection.
    Intersection * intersection = LOCAL_HEAP_GET_OBJECT_POINTER(Intersection,
            LOCAL_HEAP_GET_CURRENT_INDEX() + offset * sizeof(Intersection));

    // Shoot photons.
    Ray ray;
    BSDF bsdf;
    float3 wo, wi, flux;
    unsigned int depth = 0;
    for (unsigned int i = 0; i < nPhotonsPerThread; i++) {
        // starts from lights
        if (depth == 0) {
            // sample light
            float lightSample = GET_1_SAMPLE(sampleList, sampleIndex);
            const Light * light = sampleOneLightUniformly(lightSample);
            flux = light->flux * 4.0f * M_PIf;
            // sample direction
            float2 sample = GET_2_SAMPLES(sampleList, sampleIndex);
            wo = sampleUniformSphere(sample);
            ray = Ray(light->position, wo, NormalRay, rayEpsilon);
        }
        // starts from surface
        else {
            float  probability;
            float3 sample = GET_3_SAMPLES(sampleList, sampleIndex);
            if (!bounce(&ray, *intersection->dg(), bsdf, sample, &probability, &flux))
                continue;
        }

        // trace ray
        if (!traceUntilNonSpecularSurface(&ray, maxRayDepth, &depth,
                    intersection, &bsdf, &wo, &flux))
        {
            continue;
        }
        wi = wo;

        // create photon
        Photon & photon = photonList[photonIndex+i];
        if (depth == 1)
            photon.flags |= Photon::Direct;
        else
            photon.flags |= Photon::Indirect;
        photon.position = intersection->dg()->point;
        photon.wi       = wi;
        photon.flux     = flux;

//        /* TODO */
//        float3 pos = transformPoint(camera.worldToRaster(), photon.position);
//        uint2  ras = make_uint2(pos.x, pos.y);
//        if (ras.x < camera.width && ras.y < camera.height) {
//            if (isVisible(camera.position, photon.position))
//                outputBuffer[ras] += make_float4(0.5f, 0.0f, 0.0f, 0.0f);
//        }

        // Reset depth if necessary.
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

    // Evaluate direct illumination.
    Intersection * intersection = pixelSample.intersection();
    BSDF bsdf; intersection->getBSDF(&bsdf);
    pixelSample.direct = pixelSample.throughput *
        estimateAllDirectLighting(intersection->dg()->point, bsdf, pixelSample.wo);

    // Gather.
    uint nAccumulatedPhotons = 0;
    float3 flux = make_float3(0.0f);
    float maxDistanceSquared = pixelSample.radiusSquared;
    // First time we should use LimitedPhotonGatherer to find initial radius.
    // Otherwise just gather all the photons in range.
    if (frameCount == 0) {
        unsigned int offset = LAUNCH_OFFSET_2D(launchIndex, launchSize);
        unsigned int gatheredPhotonListIndex = LOCAL_HEAP_GET_CURRENT_INDEX() +
            offset * nPhotonsUsed * sizeof(GatheredPhoton);
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
    // Finally shrink the radius.
    pixelSample.shrinkRadius(flux, nAccumulatedPhotons);

    // Output.
    float3 indirect = pixelSample.flux / (M_PIf * pixelSample.radiusSquared) / nEmittedPhotons;
    outputBuffer[launchIndex] = make_float4(pixelSample.direct + indirect, 1.0f);
}   /* -----  end of function gatherPhotons  ----- */
