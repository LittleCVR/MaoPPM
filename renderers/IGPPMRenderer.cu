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
rtBuffer<Light,       1>  lightList;
rtBuffer<PixelSample, 2>  pixelSampleList;
rtBuffer<Importon,    1>  importonList;
rtBuffer<Photon,      1>  photonList;
rtBuffer<Photon,      1>  photonMap;
rtBuffer<float,       1>  sampleList;

rtDeclareVariable(uint, frameCount         , , );
rtDeclareVariable(uint, nSamplesPerThread  , , );
rtDeclareVariable(uint, nImportonsPerThread, , );
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
    for (uint i = 0; i < DEFAULT_MAX_RAY_DEPTH; ++i) {
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
        if (!payload.isHit)
            return;

        ++depth;
        wo = -wi;

        intersection = payload.intersection();
        intersection->getBSDF(&bsdf);
        if (launchIndex.x == 92 && launchIndex.y == 60) {
            rtPrintf("depth: %u, material type: %u\n", depth, intersection->m_material->type());
        }
        // If the surface is not a perfect specular surface.
        if (depth == DEFAULT_MAX_RAY_DEPTH ||
            bsdf.nBxDFs(BxDF::Type(BxDF::All & ~BxDF::Specular)) > 0)
        {
            pixelSample.flags |= PixelSample::isHit;
            pixelSample.setIntersection(intersection);
            pixelSample.wo = wo;
            break;
        }
    }

    /* TODO: move this task to the light class */
    // Evaluate direct illumination.
    float3 Li;
    intersection = pixelSample.intersection();
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

        intersection->getBSDF(&bsdf);
        float3 f = bsdf.f(pixelSample.wo, wi);
        Li = f * light->flux  * fabsf(dot(wi, intersection->dg()->normal))
            / (4.0f * M_PIf * distanceSquared);
    }

    pixelSample.direct = Li * pixelSample.throughput;
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
        // do not re-shoot if this importon is valid
        Importon & importon = importonList[importonIndex++];
        if (importon.flags & Importon::isHit) continue;

        // sample direction
        float3 wi;
        float  probability;
        float3 sample = GET_3_SAMPLES(sampleList, sampleIndex);
        float3 f = bsdf.sampleF(pixelSample.wo, &wi, sample, &probability);
        if (probability == 0.0f) continue;

        // trace
        Ray ray(intersection->dg()->point, wi, NormalRay, rayEpsilon);
        NormalRayPayload payload;
        payload.reset();
        rtTrace(rootObject, ray, payload);
        if (!payload.isHit)
            continue;
        else
            importon.flags |= Importon::isHit;

        // importon
        importon.setIntersection(payload.intersection());
        importon.weight    = f / probability;
        importon.wo        = -wi;
        /*TODO*/
        importon.radiusSquared  = 32.0f;
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
    unsigned int binFlags = 0;
    for (uint i = 0; i < nPhotonsPerThread; i++) {
        // starts from lights
        if (depth == 0) {
            /*TODO*/
            // sample light
            const Light & light = lightList[0];
            flux = light.flux;
            // sample direction
            float2 sample = GET_2_SAMPLES(sampleList, sampleIndex);
            // PDF
            if (frameCount == 0)
                wo = sampleUniformSphere(sample);
            else {
                wo = sampleUniformSphere(sample);
            }
            float theta = acosf(wo.z);
            float phi   = acosf(wo.x);
            if (wo.y < 0.0f) phi += M_PIf;
            unsigned int thetaBin = fmaxf(N_THETA-1,
                    floorf(theta / M_PIf * static_cast<float>(N_THETA)));
            unsigned int phiBin = fmaxf(N_PHI-1,
                    floorf(phi / (2.0f*M_PIf) * static_cast<float>(N_PHI)));
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

        // create photon
        Photon & photon = photonList[photonIndex+i];
        photon.reset();
        photon.flags |= binFlags;
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

    /* TODO: move this task to the KdTree class */
    // Compute indirect illumination.
    float greatestReductionFactor2 = 0.0f;
    for (uint i = 0; i < nImportonsPerThread; ++i) {
        Importon & importon = importonList[importonIndex+i];
        float3 flux = make_float3(0.0f);
        uint nAccumulatedPhotons = 0;
        if (importon.flags & Importon::isHit) {
            uint stack[32];
            uint stackPosition = 0;
            uint stackNode     = 0;
            stack[stackPosition++] = 0;
            do {
                const Photon & photon = photonMap[stackNode];
                if (photon.flags == KdTree::Null)
                    stackNode = stack[--stackPosition];
                else {
                    Intersection * intersection  = importon.intersection();
                    BSDF bsdf; intersection->getBSDF(&bsdf);
                    float3 diff = intersection->dg()->point - photon.position;
                    float distanceSquared = dot(diff, diff);
                    if (distanceSquared < importon.radiusSquared) {
                        float3 f = bsdf.f(importon.wo, photon.wi);
                        if (!isBlack(f)) {
                            flux += f * photon.flux;
                            ++nAccumulatedPhotons;
                        }
                    }

                    if (photon.flags & KdTree::Leaf)
                        stackNode = stack[--stackPosition];
                    else {
                        float d;
                        if      (photon.flags & KdTree::AxisX)  d = diff.x;
                        else if (photon.flags & KdTree::AxisY)  d = diff.y;
                        else                                    d = diff.z;

                        // Calculate the next child selector. 0 is left, 1 is right.
                        int selector = d < 0.0f ? 0 : 1;
                        if (d*d < importon.radiusSquared)
                            stack[stackPosition++] = (stackNode << 1) + 2 - selector;
                        stackNode = (stackNode << 1) + 1 + selector;
                    }
                }
            } while (stackNode != 0) ;

            // Compute new N, R.
            /* TODO: let alpha be configurable */
            float alpha = 0.7f;
            float R2 = importon.radiusSquared;
            float N = importon.nPhotons;
            float M = static_cast<float>(nAccumulatedPhotons) ;
            float newN = N + alpha*M;
            importon.nPhotons = newN;

            float reductionFactor2 = 1.0f;
            float newR2 = R2;
            if (M != 0) {
                reductionFactor2 = (N + alpha*M) / (N + M);
                newR2 = R2 * reductionFactor2;
                importon.radiusSquared = newR2;
            }

            if (greatestReductionFactor2 < reductionFactor2)
                greatestReductionFactor2 = reductionFactor2;

            // Compute indirect flux.
            float3 newFlux = (importon.flux + flux) * reductionFactor2;
            importon.flux = newFlux;
        }
    }

    Intersection * intersection = pixelSample.intersection();
    BSDF bsdf; intersection->getBSDF(&bsdf);
    unsigned int nValidImportons = 0;
    float3 indirect = make_float3(0.0f);
    for (uint i = 0; i < nImportonsPerThread; i++) {
        Importon & importon = importonList[importonIndex+i];
        if (importon.flags & Importon::isHit) {
            ++nValidImportons;
            float3 Li = importon.flux / (M_PIf * importon.radiusSquared);
            indirect += importon.weight * Li * fabsf(dot(intersection->dg()->normal, importon.wo));
        }
    }
    if (nValidImportons != 0)
        indirect = indirect * pixelSample.throughput /
            (nEmittedPhotons - pixelSample.nEmittedPhotonsOffset) / nValidImportons;

    /* TODO: test */
    if (nValidImportons == 0 || greatestReductionFactor2 > 0.9f)
        pixelSample.flags |= PixelSample::Finished;

    // Average.
    float3 color = indirect;
    float nSampled = static_cast<float>(pixelSample.nSampled);
    color = (1.0f / (nSampled + 1.0f)) * color +
        (nSampled / (nSampled + 1.0f)) * pixelSample.radiance;
    if (pixelSample.flags & PixelSample::Finished)
    {
        pixelSample.radiance = color;
        ++pixelSample.nSampled;
    }
    outputBuffer[launchIndex] = make_float4(pixelSample.direct + color, 1.0f);
}   /* -----  end of function gatherPhotons  ----- */
