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
#include    "utility.h"
#include    "BSDF.h"
#include    "Camera.h"
#include    "Light.h"

/*----------------------------------------------------------------------------
 *  namespace
 *----------------------------------------------------------------------------*/
using namespace optix;
using namespace MaoPPM;



typedef IGPPMRenderer::PixelSample  PixelSample;
typedef IGPPMRenderer::Importon     Importon;
#define Photon IGPPMRenderer::Photon



rtDeclareVariable(uint2, launchIndex, rtLaunchIndex, );
rtDeclareVariable(uint2, launchSize ,              , );

rtBuffer<float4,      2>  outputBuffer;
rtBuffer<PixelSample, 2>  pixelSampleList;
rtBuffer<Importon,    1>  importonList;
rtBuffer<Photon,      1>  photonList;
rtBuffer<Photon,      1>  photonMap;
rtBuffer<float,       1>  sampleList;

rtDeclareVariable(uint, frameCount         , , );
rtDeclareVariable(uint, guidedByImportons  , , );
rtDeclareVariable(uint, nSamplesPerThread  , , );
rtDeclareVariable(uint, nImportonsPerThread, , );
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
 *         Name:  shootImportons
 *  Description:  
 * =============================================================================
 */
RT_PROGRAM void shootImportons()
{
    PixelSample & pixelSample = pixelSampleList[launchIndex];
    // Does not have to shoot importons if pixel sample was not hit.
    if (!(pixelSample.flags & PixelSample::isHit))
        return;
    if (frameCount != 0 && !(pixelSample.flags & PixelSample::Regather))
        return;
    pixelSample.flags &= ~PixelSample::Regather;
    pixelSample.nEmittedPhotonsOffset = nEmittedPhotons;

    // Prepare offset variables.
    unsigned int offset = LAUNCH_OFFSET_2D(launchIndex, launchSize);
    unsigned int sampleIndex   = offset * nSamplesPerThread;
    unsigned int importonIndex = offset * nImportonsPerThread;

    for (uint i = 0; i < nImportonsPerThread; i++)
        importonList[importonIndex+i].reset();

    const Intersection * pIntersection = pixelSample.intersection();
    const BSDF pBSDF = pIntersection->getBSDF();

    // other importons
    for (uint i = 0; i < nImportonsPerThread; i++) {
        Importon & importon = importonList[importonIndex+i];
        // Allocate memory for intersection. This is for importon.
        Intersection * intersection = LOCAL_HEAP_GET_OBJECT_POINTER(Intersection,
                LOCAL_HEAP_GET_CURRENT_INDEX() + (offset + i) * sizeof(Intersection));
        BSDF bsdf;

        float probability;
        float3 sample = GET_3_SAMPLES(sampleList, sampleIndex);
        Ray ray(camera.position, -pixelSample.wo, NormalRay, rayEpsilon);
        if (!bounce(&ray, *pIntersection->dg(), pBSDF, sample, &probability, &importon.throughput))
            continue;

        uint depth = 1;
        if (!traceUntilNonSpecularSurface(&ray, maxRayDepth, &depth,
                    intersection, &bsdf, &importon.wo, &importon.throughput))
        {
            continue;
        }

        importon.flags |= Importon::isHit;
        importon.setIntersection(intersection);

//        /* TODO */
//        float3 position = intersection->dg()->point;
//        float3 pos = transformPoint(camera.worldToRaster(), position);
//        uint2  ras = make_uint2(pos.x, pos.y);
//        if (ras.x < camera.width && ras.y < camera.height) {
//            if (isVisible(camera.position, position))
//                outputBuffer[ras] += make_float4(0.5f, 0.0f, 0.0f, 0.0f);
//        }
    }
}   /* -----  end of function shootImportons  ----- */



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

    Ray ray;
    BSDF bsdf;
    uint depth = 0;
    float3 wo, wi, flux;
    bool isCausticPhoton = true;
    unsigned int binFlags = 0;
    for (uint i = 0; i < nPhotonsPerThread; i++) {
        // starts from lights
        if (depth == 0) {
            // sample light
            float lightSample = GET_1_SAMPLE(sampleList, sampleIndex);
            const Light * light = sampleOneLightUniformly(lightSample);
            // sample direction
            float  probability;
            float2 sample = GET_2_SAMPLES(sampleList, sampleIndex);
            /* TODO: add guidedByImportons */
            float3 Le = light->sampleL(sample, &wo, &probability);
            flux = Le / probability;
            ray = Ray(light->position, wo, NormalRay, rayEpsilon);
//            unsigned int thetaBin, phiBin;
//            if (!guidedByImportons) {
//                float2 s = make_float2(sample);
//                wo = sampleUniformSphere(s);
//                float theta = acosf(wo.z);
//                float phi   = acosf(wo.x);
//                if (wo.y < 0.0f) phi += M_PIf;
//                thetaBin = fminf(N_THETA-1,
//                        floorf(theta / M_PIf * static_cast<float>(N_THETA)));
//                phiBin = fminf(N_PHI-1,
//                        floorf(phi / (2.0f*M_PIf) * static_cast<float>(N_PHI)));
//                flux = light.flux * 4.0f * M_PIf;
//            } else {
//                // CDF
//                uint index = 0;
//                for (uint j = 0; j < N_THETA*N_PHI; ++j)
//                    if (sample.z <= light.cdf[j]) {
//                        index = j;
//                        break;
//                    }
//                thetaBin = index / N_PHI;
//                phiBin   = index % N_PHI;
//                float zMax = static_cast<float>(thetaBin+0) / N_THETA;
//                float zMin = static_cast<float>(thetaBin+1) / N_THETA;
//                float pMax = static_cast<float>(phiBin+0) * (2.0f * M_PIf) / N_PHI;
//                float pMin = static_cast<float>(phiBin+1) * (2.0f * M_PIf) / N_PHI;
//                float2 s = make_float2(sample);
//                s.x = s.x * (zMax-zMin) + zMin;
//                s.y = (s.y * (pMax-pMin) + pMin) / (2.0f * M_PIf);
//                wo = sampleUniformSphere(s);
//                flux = light.flux * 4.0f * M_PIf * light.normalizedArea(thetaBin, phiBin) /
//                    (index == 0 ? light.cdf[index] : (light.cdf[index]-light.cdf[index-1]));
//                if (launchIndex.x == 128 && launchIndex.y == 128) {
//                    float theta = acosf(wo.z);
//                    float phi   = acosf(wo.x);
//                    if (wo.y < 0.0f) phi += M_PIf;
//                    theta = theta * 180.0f / M_PIf;
//                    phi   = phi   * 180.0f / M_PIf;
//                    rtPrintf("tb: %u, pb: %u, zMin: %f, zMax: %f, pMin: %f, pMax: %f, ",
//                            thetaBin, phiBin, zMin, zMax, pMin, pMax);
//                    rtPrintf("s.x: %f, s.y: %f, theta: %f, phi: %f, flux: %f %f %f\n",
//                            s.x, s.y, theta, phi, flux.x, flux.y, flux.z);
//                }
//            }
//            binFlags = (thetaBin << 24) | (phiBin << 16);
        }
        // starts from surface
        else {
            float  probability;
            float3 sample = GET_3_SAMPLES(sampleList, sampleIndex);
            // remember that we are now shooting rays from a light
            // thus wo and wi must be swapped
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
        photon.flags |= binFlags;
        if (depth == 1)
            photon.flags |= Photon::Direct;
        else if (isCausticPhoton)
            photon.flags |= Photon::Caustic;
        else
            photon.flags |= Photon::Indirect;
        photon.position = intersection->dg()->point;
        photon.wi       = wi;
        photon.flux     = flux;

//        /* TODO */
//        if (photon.flags & Photon::Direct) {
//            float3 position = intersection->dg()->point;
//            float3 pos = transformPoint(camera.worldToRaster(), position);
//            uint2  ras = make_uint2(pos.x, pos.y);
//            if (ras.x < camera.width && ras.y < camera.height) {
//                if (isVisible(camera.position, position))
//                    outputBuffer[ras] += make_float4(0.5f, 0.0f, 0.0f, 0.0f);
//            }
//        }

        // After traceUntilNonSpecularSurface(),
        // photons should be all indirect now.
        isCausticPhoton = false;
        // Reset depth if necessary.
        if (depth % maxRayDepth == 0) {
            depth = 0;
            isCausticPhoton = true;
        }
    }
}   /* -----  end of function shootPhotons  ----- */



/* 
 * ===  FUNCTION  ==============================================================
 *         Name:  gatherPhotons
 *  Description:  
 * =============================================================================
 */
RT_PROGRAM void gatherPhotons()
{
    // Do not have to gather photons if pixel sample was not hit.
    PixelSample & pixelSample = pixelSampleList[launchIndex];
    if (!(pixelSample.flags & PixelSample::isHit)) return;

    uint offset = LAUNCH_OFFSET_2D(launchIndex, launchSize);
    uint importonIndex = nImportonsPerThread * offset;

    // Evaluate direct illumination.
    Intersection * intersection = pixelSample.intersection();
    BSDF bsdf; intersection->getBSDF(&bsdf);
    float3 direct = pixelSample.throughput *
        estimateAllDirectLighting(intersection->dg()->point, bsdf, pixelSample.wo);

    GatheredPhoton * gatheredPhotonList = NULL;
    if (frameCount == 0) {
        Index gatheredPhotonListIndex = LOCAL_HEAP_GET_CURRENT_INDEX() +
            offset * nPhotonsUsed * sizeof(GatheredPhoton);
        gatheredPhotonList =
            LOCAL_HEAP_GET_OBJECT_POINTER(GatheredPhoton, gatheredPhotonListIndex);
    }

    // Gather pixel sample first.
    uint nAccumulatedPhotons = 0;
    float3 flux = make_float3(0.0f);
    float maxDistanceSquared = pixelSample.radiusSquared;
//    // First time we should use LimitedPhotonGatherer to find initial radius.
//    // Otherwise just gather all the photons in range.
//    if (frameCount == 0) {
//        flux = LimitedPhotonGatherer::accumulateFlux(
//                intersection->dg()->point, pixelSample.wo, &bsdf,
//                &photonMap[0], &maxDistanceSquared, &nAccumulatedPhotons,
//                gatheredPhotonList, Photon::Flag(Photon::Caustic));
//        // maxDistanceSquared may be shrinked. So write it back to pixelSample.
//        pixelSample.radiusSquared = maxDistanceSquared;
//    }
//    else {  // frameCount != 0
//        flux = PhotonGatherer::accumulateFlux(
//                intersection->dg()->point, pixelSample.wo, &bsdf,
//                &photonMap[0], &maxDistanceSquared, &nAccumulatedPhotons,
//                Photon::Flag(Photon::Caustic));
//    }
//    pixelSample.shrinkRadius(flux, nAccumulatedPhotons);
//
//    // Caustic.
//    float3 caustic = pixelSample.throughput *
//        pixelSample.flux / (M_PIf * pixelSample.radiusSquared) /
//        nEmittedPhotons;

    // Compute indirect illumination.
    float greatestReductionFactor2 = 0.0f;
    for (uint i = 0; i < nImportonsPerThread; ++i) {
        Importon & importon = importonList[importonIndex+i];
        if (importon.flags & Importon::isHit) {
            intersection = importon.intersection();
            intersection->getBSDF(&bsdf);
            maxDistanceSquared = importon.radiusSquared;
            // First time we should use LimitedPhotonGatherer to find initial radius.
            // Otherwise just gather all the photons in range.
            if (frameCount == 0) {
                flux = LimitedPhotonGatherer::accumulateFlux(
                        intersection->dg()->point, importon.wo, &bsdf,
                        &photonMap[0], &maxDistanceSquared, &nAccumulatedPhotons,
                        gatheredPhotonList, Photon::Flag(Photon::All));
                // KdTree::find() may shrink the radius. So write it back to pixelSample.
                importon.radiusSquared = maxDistanceSquared;
            }
            else {  // frameCount != 0
                flux = PhotonGatherer::accumulateFlux(
                        intersection->dg()->point, importon.wo, &bsdf,
                        &photonMap[0], &maxDistanceSquared, &nAccumulatedPhotons,
                        Photon::Flag(Photon::All),
                        &lightList[0], dot(importon.throughput, importon.throughput));
            }

            float reductionFactor2;
            importon.shrinkRadius(flux, nAccumulatedPhotons, &reductionFactor2);
            if (greatestReductionFactor2 < reductionFactor2)
                greatestReductionFactor2 = reductionFactor2;
        }
    }

    // Indirect
    float3 indirect = make_float3(0.0f);
    unsigned int nValidImportons = 0;
    for (uint i = 0; i < nImportonsPerThread; i++) {
        Importon & importon = importonList[importonIndex+i];
        if (importon.flags & Importon::isHit) {
            ++nValidImportons;
            float3 Li = importon.flux / (M_PIf * importon.radiusSquared);
            indirect += importon.throughput * Li;
        }
    }
    if (nValidImportons != 0) {
        indirect *= pixelSample.throughput /
            (nEmittedPhotons - pixelSample.nEmittedPhotonsOffset) / nValidImportons;
    }

    /* TODO: test */
    if (nValidImportons == 0 || greatestReductionFactor2 > 0.95f)
        pixelSample.flags |= PixelSample::Regather;

    // Average.
    float nGathered = static_cast<float>(pixelSample.nGathered);
    indirect = (1.0f / (nGathered + 1.0f)) * indirect +
        (nGathered / (nGathered + 1.0f)) * pixelSample.indirect;
    if (pixelSample.flags & PixelSample::Regather) {
        pixelSample.indirect = indirect;
        ++pixelSample.nGathered;
    }
    outputBuffer[launchIndex] = make_float4(direct + indirect, 1.0f);
}   /* -----  end of function gatherPhotons  ----- */
