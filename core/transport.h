/*
 * =============================================================================
 *
 *       Filename:  transport.cu
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  2011-07-01 01:12:51
 *
 *         Author:  Chun-Wei Huang (LittleCVR), 
 *        Company:  Communication & Multimedia Laboratory,
 *                  Department of Computer Science & Information Engineering,
 *                  National Taiwan University
 *
 * =============================================================================
 */

#ifndef IGPPM_CORE_TRANSPORT_H
#define IGPPM_CORE_TRANSPORT_H

/*----------------------------------------------------------------------------
 *  Header files from OptiX
 *----------------------------------------------------------------------------*/
#include    <optix_world.h>

/*---------------------------------------------------------------------------
 *  header files of our own
 *---------------------------------------------------------------------------*/
#include    "global.h"
#include    "payload.h"



#ifdef __CUDACC__
rtDeclareVariable(rtObject, rootObject, , );
rtDeclareVariable(float,    rayEpsilon, , );
#endif  /* -----  #ifdef __CUDACC__  ----- */



namespace MaoPPM {

#ifdef __CUDACC__

__device__ bool isVisible(
        const optix::float3 & p1, const optix::float3 & p2,
        optix::float3 * direction, optix::float3 * normalizedDirection,
        float * distance, float * distanceSquared)
{
    *direction = p2 - p1;
    *normalizedDirection = optix::normalize(*direction);
    *distanceSquared = optix::dot(*direction, *direction);
    *distance = sqrtf(*distanceSquared);

    ShadowRayPayload payload;
    payload.reset();
    optix::Ray ray(p1, *normalizedDirection, ShadowRay, rayEpsilon, *distance - rayEpsilon);
    rtTrace(rootObject, ray, payload);

    return !payload.isHit;
}

__device__ __inline__ bool traceUntilNonSpecularSurface(
        optix::Ray * ray, unsigned int maxDepth, unsigned int * depth,
        Intersection * intersection, BSDF * bsdf,
        optix::float3 * wo, optix::float3 * throughput)
{
    /* TODO */
    rtPrintf("traceUntilNonSpecularSurface()\n");

    *depth = 0;
    *throughput = optix::make_float3(1.0f);
    NormalRayPayload payload;
    payload.setIntersectionBuffer(intersection);
    optix::float3 wi = ray->direction;
    for (unsigned int i = 0; i < maxDepth; ++i) {
        /* TODO */
        rtPrintf("    loop %u\n", i);

        // Intersect with the scene.
        payload.reset();
        rtTrace(rootObject, *ray, payload);
        // Terminate immediately if not hit.
        if (!payload.isHit)
            return false;
        rtPrintf("    hit\n");

        // Increase depth, forward to the next intersection, swap wo and wi.
        ++(*depth);
        *wo = -wi;
        intersection->getBSDF(bsdf);
        // If the surface is not a perfect specular surface.
        if (bsdf->nBxDFs(BxDF::Type(BxDF::All & ~BxDF::Specular)) > 0)
            return true;
        // Terminate if it's too deep.
        if (*depth == maxDepth)
            return false;

        // Compute throughput and make new ray.
        float  probability;
        // Do not have to use real sample since the surface is perfect specular.
        optix::float3 sample = optix::make_float3(0.0f);
        optix::float3 f = bsdf->sampleF(*wo, &wi, sample, &probability,
                BxDF::Type(BxDF::Reflection | BxDF::Transmission | BxDF::Specular));
        *throughput *= f * fabsf(optix::dot(wi, intersection->dg()->normal)) / probability;
        *ray = optix::Ray(intersection->dg()->point, wi, NormalRay, rayEpsilon);
    }

    return false;
}

#endif  /* -----  #ifdef __CUDACC__  ----- */

}   /* -----  end of namespace MaoPPM  ----- */

#endif  /* -----  #ifndef IGPPM_CORE_TRANSPORT_H  ----- */
