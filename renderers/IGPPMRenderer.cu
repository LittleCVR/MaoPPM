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
#include    "Light.h"

/*----------------------------------------------------------------------------
 *  namespace
 *----------------------------------------------------------------------------*/
using namespace optix;
using namespace MaoPPM;



typedef IGPPMRenderer::PixelSample  PixelSample;
typedef IGPPMRenderer::Importon     Importon;
typedef PPMRenderer::Photon         Photon;



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
    pixelSample.isHit = payload.isHit;
    if (!pixelSample.isHit) return;

    // Fill pixel sample data if hit.
    pixelSample.intersection  = payload.intersection();
    pixelSample.wo            = -ray.direction;
    pixelSample.direct        = make_float3(0.0f);

    /* TODO: move this task to the light class */
    // Evaluate direct illumination.
    float3 Li;
    Intersection * intersection = pixelSample.intersection;
    {
        const Light * light = &lightList[0];
        float3 shadowRayDirection = light->position - intersection->dg()->point;
        float3 wi = normalize(shadowRayDirection);
        float distanceSquared = dot(shadowRayDirection, shadowRayDirection);
        float distance = sqrtf(distanceSquared);

//        /* TODO: remove these debug lines */
//        if (launchIndex.x == 449 && launchIndex.y == 252) {
//            rtPrintf("normal "); dump(intersection.dg()->normal); rtPrintf("\n");
//            rtPrintf("dpdu "); dump(intersection.dg()->dpdu); rtPrintf("\n");
//            rtPrintf("dpdv "); dump(intersection.dg()->dpdv); rtPrintf("\n");
//        }
//        //outputBuffer[launchIndex] = make_float4(intersection.dg()->normal / 2.0f + 0.5f, 1.0f);
//        //outputBuffer[launchIndex] = make_float4(normalize(intersection.dg()->dpdu) / 2.0f + 0.5f, 1.0f);
//        //outputBuffer[launchIndex] = make_float4(normalize(intersection.dg()->dpdv) / 2.0f + 0.5f, 1.0f);
//        outputBuffer[launchIndex] = make_float4(1.0f, 0.0f, 0.0f, 1.0f);

        ShadowRayPayload shadowRayPayload;
        shadowRayPayload.reset();
        ray = Ray(intersection->dg()->point, wi, ShadowRay, rayEpsilon, distance-rayEpsilon);
        rtTrace(rootObject, ray, shadowRayPayload);
        if (shadowRayPayload.isHit) return;

        BSDF bsdf; intersection->getBSDF(&bsdf);
        float3 f = bsdf.f(pixelSample.wo, wi);
        Li = f * light->flux / (4.0f * M_PIf * distanceSquared);
    }

    pixelSample.direct = Li;
}   /* -----  end of function generatePixelSamples  ----- */



/* 
 * ===  FUNCTION  ==============================================================
 *         Name:  shootImportons
 *  Description:  
 * =============================================================================
 */
RT_PROGRAM void shootImportons()
{
    if (frameCount != 0) return;

    // Does not have to shoot importons if pixel sample was not hit.
    PixelSample & pixelSample = pixelSampleList[launchIndex];
    if (!pixelSample.isHit) return;

    // Prepare offset variables.
    uint offset = LAUNCH_OFFSET_2D(launchIndex, launchSize);
    uint sampleIndex   = offset * nSamplesPerThread;
    uint importonIndex = offset * nImportonsPerThread;

    for (uint i = 0; i < nImportonsPerThread; i++)
        importonList[importonIndex+i].reset();

    Intersection *  intersection  = pixelSample.intersection;
    BSDF bsdf; intersection->getBSDF(&bsdf);
    // other importons
    for (uint i = 0; i < nImportonsPerThread; i++) {
        // do not re-shoot if this importon is valid
        Importon & importon = importonList[importonIndex++];
        if (importon.isHit) continue;

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
        importon.isHit = payload.isHit;
        if (!importon.isHit) continue;

        // importon
        importon.weight         = f / probability;
        importon.intersection   = payload.intersection();
        importon.wo             = -wi;
        importon.flux           = make_float3(0.0f);
        importon.nPhotons       = 0;
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
    for (uint i = 0; i < nPhotonsPerThread; i++) {
        // starts from lights
        if (depth == 0) {
            /*TODO*/
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
        intersection  = payload.intersection();
        intersection->getBSDF(&bsdf);

        // create photon
        Photon & photon = photonList[photonIndex+i];
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
    if (!pixelSample.isHit) return;

    uint offset = LAUNCH_OFFSET_2D(launchIndex, launchSize);
    uint importonIndex = nImportonsPerThread * offset;

    /* TODO: move this task to the KdTree class */
    // Compute indirect illumination.
    for (uint i = 0; i < nImportonsPerThread; ++i) {
        Importon & importon = importonList[importonIndex+i];
        float3 flux = make_float3(0.0f);
        uint nAccumulatedPhotons = 0;
        if (importon.isHit) {
            uint stack[32];
            uint stackPosition = 0;
            uint stackNode     = 0;
            stack[stackPosition++] = 0;
            do {
                const Photon & photon = photonMap[stackNode];
                if (photon.flags == KdTree<Photon>::Null)
                    stackNode = stack[--stackPosition];
                else {
                    Intersection * intersection  = importon.intersection;
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

                    if (photon.flags & KdTree<Photon>::Leaf)
                        stackNode = stack[--stackPosition];
                    else {
                        float d;
                        if      (photon.flags & KdTree<Photon>::AxisX)  d = diff.x;
                        else if (photon.flags & KdTree<Photon>::AxisY)  d = diff.y;
                        else                                            d = diff.z;

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

            // Compute indirect flux.
            float3 newFlux = (importon.flux + flux) * reductionFactor2;
            importon.flux = newFlux;

//            /* TODO: remove these debug lines */
//            if (launchIndex.x == 449 && launchIndex.y == 252) {
//                rtPrintf("import flux: "); dump(importon.flux); rtPrintf("\n");
//            }
        }
    }

    Intersection * intersection  = pixelSample.intersection;
//    BSDF         * bsdf          = intersection->bsdf();
    BSDF bsdf; intersection->getBSDF(&bsdf);
    unsigned int nValidImportons = 0;
    float3 indirect = make_float3(0.0f);
    for (uint i = 0; i < nImportonsPerThread; i++) {
        Importon & importon = importonList[importonIndex+i];
        if (importon.isHit) {
            ++nValidImportons;
            float3 Li = importon.flux / (M_PIf * importon.radiusSquared);
            indirect += importon.weight * Li * fabsf(dot(intersection->dg()->normal, importon.wo));
//            /*TODO*/
//            if (importon.isHit == 99) {
//                float nImportons = static_cast<float>(pixelSample.nImportons);
//                pixelSample.indirect = (nImportons / (nImportons + 1.0f)) * pixelSample.indirect +
//                    (1.0f / (nImportons + 1.0f)) * L;
//                importon.isHit = false;
//            } else {
//                indirect += L;
//            }
//            /*TODO*/
//            if (launchIndex.x == 449 && launchIndex.y == 252) {
//                rtPrintf("indirect: "); dump(indirect); rtPrintf("\n");
//            }
        }
    }
    if (nValidImportons != 0)
        indirect = indirect / nEmittedPhotons / nValidImportons;
    /*TODO*/
//    if (launchIndex.x == 449 && launchIndex.y == 252) {
//        rtPrintf("indirect: "); dump(indirect); rtPrintf("\n");
//    }

//    /*TODO*/
//    if (frameCount % 5 == 4) {
//        float  frame = static_cast<float>(frameCount / 5);
//        float4 color = make_float4(pixelSample.direct + indirect, 1.0f);
//        outputBuffer[launchIndex] = (1.0f / (frame + 1.0f)) * color +
//            (frame / (frame + 1.0f)) * outputBuffer[launchIndex];
//    }
    outputBuffer[launchIndex] = make_float4(pixelSample.direct + indirect, 1.0f);
}   /* -----  end of function gatherPhotons  ----- */
