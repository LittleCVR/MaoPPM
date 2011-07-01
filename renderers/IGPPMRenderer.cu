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
    PixelSample & pixelSample = pixelSampleList[launchIndex];
    // Clear output buffer.
    if (frameCount == 0) {
        outputBuffer[launchIndex] = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
        pixelSample.flags    = 0;
        pixelSample.direct   = make_float3(0.0f);
        pixelSample.radiance = make_float3(0.0f);
        pixelSample.nSampled = 0;
        pixelSample.nEmittedPhotonsOffset = 0;
        pixelSample.throughput = make_float3(1.0f);
    }
    else {
        // Return immediately if not finished.
        if (!(pixelSample.flags & PixelSample::Finished))
            return;
    }
    // Clear pixel sample.
    pixelSample.flags &= ~PixelSample::isHit;

    Ray ray;
    NormalRayPayload payload;
    uint depth = 0;
    float3 wo, wi;
    Intersection * intersection  = NULL;
    BSDF bsdf;
    /* TODO: hard coded max depth */
    for (uint i = 0; i < maxRayDepth; ++i) {
        if (depth == 0) {
            /* TODO: move this task to the camera class */
            // Generate camera ray.
            float2 screenSize = make_float2(outputBuffer.size());
            float2 sample = make_float2(0.5f, 0.5f); 
            float2 cameraRayDirection = (make_float2(launchIndex) + sample) / screenSize * 2.0f - 1.0f;
            wi = normalize(cameraRayDirection.x*cameraU + cameraRayDirection.y*cameraV + cameraW);
            ray = Ray(cameraPosition, wi, NormalRay, rayEpsilon);
        }
        else {
            float  probability;
            // Do not have to use real sample since the surface is perfect specular.
            float3 sample = make_float3(0.0f);
            float3 f = bsdf.sampleF(wo, &wi, sample, &probability,
                    BxDF::Type(BxDF::Reflection | BxDF::Transmission | BxDF::Specular));
            pixelSample.throughput *= f * fabsf(dot(wi, intersection->dg()->normal)) / probability;
            ray = Ray(intersection->dg()->point, wi, NormalRay, rayEpsilon);
        }

        // Intersect with the scene.
        payload.reset();
        rtTrace(rootObject, ray, payload);
        if (!payload.isHit) return;

        ++depth;  wo = -wi;
        intersection = payload.intersection();
        intersection->getBSDF(&bsdf);
        // If the surface is not a perfect specular surface.
        if (depth == maxRayDepth ||
            bsdf.nBxDFs(BxDF::Type(BxDF::All & ~BxDF::Specular)) > 0)
        {
            pixelSample.flags |= PixelSample::isHit;
            pixelSample.setIntersection(intersection);
            pixelSample.wo = wo;
            pixelSample.radiusSquared = 32.0f;
            pixelSample.nPhotons = 0;
            pixelSample.flux = make_float3(0.0f);
            break;
        }
    }

    // Evaluate direct illumination.
    pixelSample.direct = pixelSample.throughput *
        estimateAllDirectLighting(intersection->dg()->point, bsdf, pixelSample.wo);
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
    if (frameCount != 0 && !(pixelSample.flags & PixelSample::Finished))
        return;
    pixelSample.nEmittedPhotonsOffset = nEmittedPhotons;

    // Prepare offset variables.
    uint offset = LAUNCH_OFFSET_2D(launchIndex, launchSize);
    uint sampleIndex   = offset * nSamplesPerThread;
    uint importonIndex = offset * nImportonsPerThread;

    for (uint i = 0; i < nImportonsPerThread; i++)
        importonList[importonIndex+i].reset();

    Intersection *  intersection  = pixelSample.intersection();
    BSDF bsdf; intersection->getBSDF(&bsdf);
    // other importons
    for (uint i = 0; i < nImportonsPerThread; i++) {
        Importon & importon = importonList[importonIndex++];

        Ray ray;
        NormalRayPayload payload;
        uint depth = 0;
        float probability;
        float3 wo, wi, throughput;
        for (uint j = 0; j < maxRayDepth; ++j) {
            if (depth == 0) {
                // sample direction
                wo = pixelSample.wo;
                float3 sample = GET_3_SAMPLES(sampleList, sampleIndex);
                float3 f = bsdf.sampleF(wo, &wi, sample, &probability);
                if (probability == 0.0f) break;;
                throughput = f * fabsf(dot(wi, intersection->dg()->normal)) / probability;
                ray = Ray(intersection->dg()->point, wi, NormalRay, rayEpsilon);
            }
            else {
                // Do not have to use real sample since the surface is perfect specular.
                float3 sample = make_float3(0.0f);
                float3 f = bsdf.sampleF(wo, &wi, sample, &probability,
                        BxDF::Type(BxDF::Reflection | BxDF::Transmission | BxDF::Specular));
                throughput *= f * fabsf(dot(wi, intersection->dg()->normal)) / probability;
                ray = Ray(intersection->dg()->point, wi, NormalRay, rayEpsilon);
            }

            // trace
            payload.reset();
            rtTrace(rootObject, ray, payload);
            if (!payload.isHit) break;

            ++depth;  wo = -wi;
            intersection = payload.intersection();
            intersection->getBSDF(&bsdf);
            // If the surface is not a perfect specular surface.
            if (depth == maxRayDepth ||
                bsdf.nBxDFs(BxDF::Type(BxDF::All & ~BxDF::Specular)) > 0)
            {
                importon.flags |= Importon::isHit;
                importon.setIntersection(intersection);
                importon.weight = throughput;
                importon.wo     = wo;
                /*TODO*/
                importon.radiusSquared  = 32.0f;
                break;
            }
        }
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
    bool isCausticPhoton = true;
    unsigned int binFlags = 0;
    for (uint i = 0; i < nPhotonsPerThread; i++) {
        // starts from lights
        if (depth == 0) {
            /*TODO*/
            // sample light
            const Light & light = lightList[0];
            // sample direction
            float3 sample = GET_3_SAMPLES(sampleList, sampleIndex);
            unsigned int thetaBin, phiBin;
            if (!guidedByImportons) {
                float2 s = make_float2(sample);
                wo = sampleUniformSphere(s);
                float theta = acosf(wo.z);
                float phi   = acosf(wo.x);
                if (wo.y < 0.0f) phi += M_PIf;
                thetaBin = fminf(N_THETA-1,
                        floorf(theta / M_PIf * static_cast<float>(N_THETA)));
                phiBin = fminf(N_PHI-1,
                        floorf(phi / (2.0f*M_PIf) * static_cast<float>(N_PHI)));
                flux = light.flux * 4.0f * M_PIf;
            } else {
                // CDF
                uint index = 0;
                for (uint j = 0; j < N_THETA*N_PHI; ++j)
                    if (sample.z <= light.cdf[j]) {
                        index = j;
                        break;
                    }
                thetaBin = index / N_PHI;
                phiBin   = index % N_PHI;
                float zMax = static_cast<float>(thetaBin+0) / N_THETA;
                float zMin = static_cast<float>(thetaBin+1) / N_THETA;
                float pMax = static_cast<float>(phiBin+0) * (2.0f * M_PIf) / N_PHI;
                float pMin = static_cast<float>(phiBin+1) * (2.0f * M_PIf) / N_PHI;
                float2 s = make_float2(sample);
                s.x = s.x * (zMax-zMin) + zMin;
                s.y = (s.y * (pMax-pMin) + pMin) / (2.0f * M_PIf);
                wo = sampleUniformSphere(s);
                flux = light.flux * 4.0f * M_PIf * light.normalizedArea(thetaBin, phiBin) /
                    (index == 0 ? light.cdf[index] : (light.cdf[index]-light.cdf[index-1]));
                if (launchIndex.x == 128 && launchIndex.y == 128) {
                    float theta = acosf(wo.z);
                    float phi   = acosf(wo.x);
                    if (wo.y < 0.0f) phi += M_PIf;
                    theta = theta * 180.0f / M_PIf;
                    phi   = phi   * 180.0f / M_PIf;
                    rtPrintf("tb: %u, pb: %u, zMin: %f, zMax: %f, pMin: %f, pMax: %f, ",
                            thetaBin, phiBin, zMin, zMax, pMin, pMax);
                    rtPrintf("s.x: %f, s.y: %f, theta: %f, phi: %f, flux: %f %f %f\n",
                            s.x, s.y, theta, phi, flux.x, flux.y, flux.z);
                }
            }
            binFlags = (thetaBin << 24) | (phiBin << 16);
            // Ray
            ray = Ray(light.position, wo, NormalRay, rayEpsilon);
        }
        // starts from surface
        else {
            /*TODO*/
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
        intersection = payload.intersection();
        intersection->getBSDF(&bsdf);
        if (bsdf.nBxDFs(BxDF::Type(BxDF::All & ~BxDF::Specular)) > 0)
            isCausticPhoton = false;

        // create photon
        Photon & photon = photonList[photonIndex+i];
        photon.reset();
        photon.flags |= binFlags;
        if (depth == 0)
            photon.flags |= Photon::Direct;
        else if (isCausticPhoton)
            photon.flags |= Photon::Caustic;
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
 *         Name:  gatherPhotons
 *  Description:  
 * =============================================================================
 */
RT_PROGRAM void gatherPhotons()
{
    // Do not have to gather photons if pixel sample was not hit.
    PixelSample & pixelSample = pixelSampleList[launchIndex];
    if (!(pixelSample.flags & PixelSample::isHit)) return;
    pixelSample.flags &= ~PixelSample::Finished;

    uint offset = LAUNCH_OFFSET_2D(launchIndex, launchSize);
    uint importonIndex = nImportonsPerThread * offset;

    float3 direct = pixelSample.direct;

    Index gatheredPhotonListIndex;
    GatheredPhoton * gatheredPhotonList;
    if (frameCount == 0) {
        gatheredPhotonListIndex =
            LOCAL_HEAP_ALLOC_SIZE(nPhotonsUsed * sizeof(GatheredPhoton));
        gatheredPhotonList =
            LOCAL_HEAP_GET_OBJECT_POINTER(GatheredPhoton, gatheredPhotonListIndex);
    }

    // Gather pixel sample first.
    Intersection * intersection = pixelSample.intersection();
    BSDF bsdf; intersection->getBSDF(&bsdf);
    uint nAccumulatedPhotons = 0;
    float3 flux = make_float3(0.0f);
    float maxDistanceSquared = pixelSample.radiusSquared;

    if (frameCount == 0) {
        flux = LimitedPhotonGatherer::accumulateFlux(
                intersection->dg()->point, pixelSample.wo, &bsdf,
                &photonMap[0], &maxDistanceSquared, &nAccumulatedPhotons,
                gatheredPhotonList, Photon::Flag(Photon::Caustic));
        // maxDistanceSquared may be shrinked. So write it back to pixelSample.
        pixelSample.radiusSquared = maxDistanceSquared;
    }
    else {  // frameCount != 0
        flux = PhotonGatherer::accumulateFlux(
                intersection->dg()->point, pixelSample.wo, &bsdf,
                &photonMap[0], &maxDistanceSquared, &nAccumulatedPhotons,
                Photon::Flag(Photon::Caustic));
    }
    pixelSample.shrinkRadius(flux, nAccumulatedPhotons);

    // Compute indirect illumination.
    float greatestReductionFactor2 = 0.0f;
    for (uint i = 0; i < nImportonsPerThread; ++i) {
        Importon & importon = importonList[importonIndex+i];
        if (importon.flags & Importon::isHit) {
            intersection = importon.intersection();
            intersection->getBSDF(&bsdf);
            maxDistanceSquared = importon.radiusSquared;

            if (frameCount == 0) {
                flux = LimitedPhotonGatherer::accumulateFlux(
                        intersection->dg()->point, importon.wo, &bsdf,
                        &photonMap[0], &maxDistanceSquared, &nAccumulatedPhotons,
                        gatheredPhotonList, Photon::Flag(Photon::Direct | Photon::Indirect));
                // KdTree::find() may shrink the radius. So write it back to pixelSample.
                importon.radiusSquared = maxDistanceSquared;
            }
            else {  // frameCount != 0
                flux = PhotonGatherer::accumulateFlux(
                        intersection->dg()->point, importon.wo, &bsdf,
                        &photonMap[0], &maxDistanceSquared, &nAccumulatedPhotons,
                        Photon::Flag(Photon::Direct | Photon::Indirect),
                        &lightList[0], dot(importon.weight, importon.weight));
            }

            float reductionFactor2;
            importon.shrinkRadius(flux, nAccumulatedPhotons, &reductionFactor2);
            if (greatestReductionFactor2 < reductionFactor2)
                greatestReductionFactor2 = reductionFactor2;
        }
    }

    float3 caustic = pixelSample.throughput *
        pixelSample.flux / (M_PIf * pixelSample.radiusSquared) /
        nEmittedPhotons;
    float3 indirect = make_float3(0.0f);
    unsigned int nValidImportons = 0;
    for (uint i = 0; i < nImportonsPerThread; i++) {
        Importon & importon = importonList[importonIndex+i];
        if (importon.flags & Importon::isHit) {
            ++nValidImportons;
            float3 Li = importon.flux / (M_PIf * importon.radiusSquared);
            indirect += importon.weight * Li;
        }
    }
    if (nValidImportons != 0) {
        indirect = indirect * pixelSample.throughput /
            (nEmittedPhotons - pixelSample.nEmittedPhotonsOffset) / nValidImportons;
    }

    /* TODO: test */
    if (nValidImportons == 0 || greatestReductionFactor2 > 0.95f)
        pixelSample.flags |= PixelSample::Finished;

    // Average.
    float nSampled = static_cast<float>(pixelSample.nSampled);
    indirect = (1.0f / (nSampled + 1.0f)) * indirect +
        (nSampled / (nSampled + 1.0f)) * pixelSample.radiance;
    if (pixelSample.flags & PixelSample::Finished)
    {
        pixelSample.radiance = indirect;
        ++pixelSample.nSampled;
    }
    outputBuffer[launchIndex] = make_float4(direct + caustic + indirect, 1.0f);
}   /* -----  end of function gatherPhotons  ----- */
