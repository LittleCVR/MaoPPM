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
rtBuffer<Light,       1>  lightList;
rtBuffer<PixelSample, 2>  pixelSampleList;
rtBuffer<Importon,    1>  importonList;
rtBuffer<Photon,      1>  photonList;
rtBuffer<Photon,      1>  photonMap;
rtBuffer<float,       1>  sampleList;

rtDeclareVariable(uint, resetImporton      , , ) = true;
rtDeclareVariable(uint, nSamplesPerThread  , , );
rtDeclareVariable(uint, nImportonsPerThread, , );
rtDeclareVariable(uint, nPhotonsPerThread  , , );
rtDeclareVariable(uint, nEmittedPhotons    , , ) = 0;

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
    // Clear output buffer and pixel sample.
    outputBuffer[launchIndex] = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
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
    rtTrace(rootObject, ray, payload);
    pixelSample.isHit = payload.isHit;
    if (!pixelSample.isHit) return;

    // Fill pixel sample data if hit.
    pixelSample.intersection  = payload.intersection;
    pixelSample.wo            = -ray.direction;
    pixelSample.direct        = make_float3(0.0f);

    /* TODO: move this task to the light class */
    // Evaluate direct illumination.
    {
        const Light & light = lightList[0];
        Intersection & intersection = pixelSample.intersection;
        float3 shadowRayDirection = light.position - intersection.dg()->point;
        float3 normalizedShadowRayDirection = normalize(shadowRayDirection);
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
        shadowRayPayload.isHit = 0;
        ray = Ray(intersection.dg()->point, normalizedShadowRayDirection, ShadowRay, rayEpsilon, distance-rayEpsilon);
        rtTrace(rootObject, ray, shadowRayPayload);
        if (shadowRayPayload.isHit) return;

        BSDF * bsdf = intersection.bsdf();
        Matrix4x4 * transform = intersection.worldToObject();
        float3 wo = transformVector(*transform, pixelSample.wo);
        float3 wi = transformVector(*transform, normalizedShadowRayDirection);
        float3 f = bsdf->f(wo, wi);
        pixelSample.direct = pairwiseMul(f, light.flux) / (4.0f * M_PIf * distanceSquared);
    }
}   /* -----  end of function generatePixelSamples  ----- */



/* 
 * ===  FUNCTION  ==============================================================
 *         Name:  shootImportons
 *  Description:  
 * =============================================================================
 */
RT_PROGRAM void shootImportons()
{
    /*TODO*/
//    if (launchIndex.x == 449 && launchIndex.y == 252)
//        rtPrintf("========== importon pass\n");

    PixelSample & pixelSample = pixelSampleList[launchIndex];
    if (!pixelSample.isHit) return;

    uint offset = LAUNCH_OFFSET_2D(launchIndex, launchSize);
    uint sampleIndex   = offset * nSamplesPerThread;
    uint importonIndex = offset * nImportonsPerThread;

    if (resetImporton)
        for (uint i = 0; i < nImportonsPerThread; i++)
            importonList[importonIndex+i].isHit = false;

    /*TODO: write sampleF */
    Intersection & intersection = pixelSample.intersection;
    BSDF * bsdf = intersection.bsdf();
    Lambertian & bxdf = reinterpret_cast<Lambertian &>(bsdf->m_bxdfList[0]);
    Matrix4x4 * worldToObject = intersection.worldToObject();
    float3 wo = transformVector(*worldToObject, pixelSample.wo);
    // other importons
    for (uint i = 0; i < nImportonsPerThread; i++) {
        // do not re-shoot if this importon is valid
        Importon & importon = importonList[importonIndex++];
        if (importon.isHit) continue;

        // sample direction
        float3 wi;
        float  probability;
        float2 sample = GET_2_SAMPLES(sampleList, sampleIndex);
        bxdf.sampleF(wo, &wi, sample, &probability);
        if (probability == 0.0f) continue;

        // trace
        float3 wwi = transformVector(worldToObject->transpose(), wi);
//        if (launchIndex.x == 449 && launchIndex.y == 252) {
//            rtPrintf("importon: world wo = (%+4.4f, %+4.4f, %+4.4f), wo = (%+4.4f, %+4.4f, %+4.4f),\n"
//                     "          world wi = (%+4.4f, %+4.4f, %+4.4f), wi = (%+4.4f, %+4.4f, %+4.4f)\n",
//                    pixelSample.wo.x, pixelSample.wo.y, pixelSample.wo.z, wo.x, wo.y, wo.z,
//                    wwi.x, wwi.y, wwi.z, wi.x, wi.y, wi.z);
//            rtPrintf("importon probability: %f\n", probability);
//        }
        Ray ray(intersection.dg()->point, wwi, NormalRay, rayEpsilon);
        NormalRayPayload payload;
        rtTrace(rootObject, ray, payload);
        importon.isHit = payload.isHit;
        if (!importon.isHit) continue;

        // importon
        importon.weight         = 1.0f / probability;
        importon.intersection   = payload.intersection;
        importon.wo             = -wwi;
        importon.flux           = make_float3(0.0f);
        importon.nPhotons       = 0;
        /*TODO*/
        importon.radiusSquared  = 64.0f;
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

    // reset photon list
    for (uint i = 0; i < nPhotonsPerThread; i++)
        photonList[photonIndex+i].init();

    Ray ray;
    float3 wo, wi, flux;
    NormalRayPayload payload;
    bool hitAtLeastOnce = false;
    for (uint i = 0; i < nPhotonsPerThread; i++) {
        // starts from lights
        if (!hitAtLeastOnce) {
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
            Intersection & intersection = payload.intersection;
            BSDF * bsdf = intersection.bsdf();
            Lambertian & bxdf = reinterpret_cast<Lambertian &>(bsdf->m_bxdfList[0]);
            Matrix4x4 * worldToObject = intersection.worldToObject();
            wi = transformVector(*worldToObject, -wo);
            float  probability;
            float2 sample = GET_2_SAMPLES(sampleList, sampleIndex);
            // remember that we are now shooting rays from a light
            // thus wo and wi must be swapped
            float3 f = bxdf.sampleF(wi, &wo, sample, &probability);
            flux = pairwiseMul(f, flux) / probability;
            // transform from object to world
            // remember that this transform's transpose is actually its inverse
            wo = transformVector(worldToObject->transpose(), wo);
            ray = Ray(intersection.dg()->point, wo, NormalRay, rayEpsilon);
        }

        // trace ray
        payload.reset();
        rtTrace(rootObject, ray, payload);
        if (payload.isHit)
            hitAtLeastOnce = true;
        else
            continue;

        // create photon
        Intersection & intersection = payload.intersection;
        Photon & photon = photonList[photonIndex+i];
        photon.position = intersection.dg()->point;
        photon.normal   = intersection.dg()->normal;
        photon.wi       = -wo;
        photon.flux     = flux;
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
    // Do not have to gather photons if pixel sample is not hit.
    PixelSample & pixelSample = pixelSampleList[launchIndex];
    if (!pixelSample.isHit) return;

    uint offset = LAUNCH_OFFSET_2D(launchIndex, launchSize);
    uint importonIndex = nImportonsPerThread * offset;

    /* TODO: move this task to the KdTree class */
    // Compute indirect illumination.
    uint nValidImportons = 0;
    for (uint i = 0; i < nImportonsPerThread; ++i) {
        Importon & importon = importonList[importonIndex+i];
        float3 flux = make_float3(0.0f);
        uint nAccumulatedPhotons = 0;
        if (importon.isHit) {
            ++nValidImportons;
            uint stack[32];
            uint stackPosition = 0;
            uint stackNode     = 0;
            stack[stackPosition++] = 0;
            do {
                const Photon & photon = photonMap[stackNode];
                if (photon.flags == Photon::Null)
                    stackNode = stack[--stackPosition];
                else {
                    Intersection & intersection = importon.intersection;
                    float3 diff = intersection.dg()->point - photon.position;
                    float distanceSquared = dot(diff, diff);
                    if (distanceSquared < importon.radiusSquared) {
                        BSDF * bsdf = intersection.bsdf();
                        Matrix4x4 * worldToObject = intersection.worldToObject();
                        float3 wo = transformVector(*worldToObject, importon.wo);
                        float3 wi = transformVector(*worldToObject, photon.wi);
                        flux += pairwiseMul(bsdf->f(wo, wi), photon.flux) *
                            fmaxf(0.0f, dot(photon.normal, intersection.dg()->normal));
                        ++nAccumulatedPhotons;
                    }

                    if(photon.flags == Photon::Leaf)
                        stackNode = stack[--stackPosition];
                    else {
                        float d;
                        if      (photon.flags == Photon::AxisX)  d = diff.x;
                        else if (photon.flags == Photon::AxisY)  d = diff.y;
                        else                                     d = diff.z;

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
    if (nValidImportons == 0) return;

    Intersection & intersection = pixelSample.intersection;
    BSDF * bsdf = intersection.bsdf();
    Matrix4x4 * worldToObject = intersection.worldToObject();
    float3 indirect = make_float3(0.0f);
    for (uint i = 0; i < nImportonsPerThread; i++) {
        Importon & importon = importonList[importonIndex+i];
        if (importon.isHit) {
            float3 wo = transformVector(*worldToObject, pixelSample.wo);
            float3 wi = transformVector(*worldToObject, -importon.wo);
            indirect = importon.weight *
                pairwiseMul(bsdf->f(wo, wi), importon.flux) /
                (M_PIf * importon.radiusSquared);
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
    indirect /= nEmittedPhotons;
//    /*TODO*/
//    if (launchIndex.x == 449 && launchIndex.y == 252) {
//        rtPrintf("indirect: "); dump(indirect); rtPrintf("\n");
//    }

    /*TODO*/
//    float nImportons = static_cast<float>(pixelSample.nImportons);
//    float validImportons = static_cast<float>(nValidImportons);
//    float3 finalColor = (nImportons / (nImportons + validImportons)) * pixelSample.indirect +
//        (validImportons / (nImportons + validImportons)) * indirect;
    //outputBuffer[launchIndex] = make_float4(pixelSample.direct + indirect, 1.0f);
    //outputBuffer[launchIndex] = make_float4(finalColor, 1.0f);
    outputBuffer[launchIndex] = make_float4(indirect, 1.0f);
}   /* -----  end of function gatherPhotons  ----- */
