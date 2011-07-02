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
#include    "BSDF.h"
#include    "Intersection.h"



#ifdef __CUDACC__
rtDeclareVariable(rtObject, rootObject, , );
rtDeclareVariable(float,    rayEpsilon, , );
#endif  /* -----  #ifdef __CUDACC__  ----- */



namespace MaoPPM {

#ifdef __CUDACC__

__device__ __forceinline__ bool isVisible(
        const optix::float3 & p1, const optix::float3 & p2,
        optix::float3 * direction = NULL, optix::float3 * normalizedDirection = NULL,
        float * distance = NULL, float * distanceSquared = NULL)
{
    optix::float3 d = p2 - p1;
    optix::float3 nd = optix::normalize(d);
    float dist2 = optix::dot(d, d);
    float dist = sqrtf(dist2);

    if (direction) *direction = d;
    if (normalizedDirection) *normalizedDirection = nd;
    if (distanceSquared) *distanceSquared = dist2;
    if (distance) *distance = dist;

    ShadowRayPayload payload;
    payload.reset();
    optix::Ray ray(p1, nd, ShadowRay, rayEpsilon, dist- rayEpsilon);
    rtTrace(rootObject, ray, payload);

    return !payload.isHit;
}

__device__ __forceinline__ bool bounce(
        optix::Ray * ray, const DifferentialGeometry & dg, const BSDF & bsdf,
        const optix::float3 & sample, float * probability, optix::float3 * throughput,
        BxDF::Type type = BxDF::All)
{
    optix::float3 wi, wo = -ray->direction;
    optix::float3 f = bsdf.sampleF(wo, &wi, sample, probability, type);
    if (*probability == 0.0f || isBlack(f))
        return false;

    *throughput *= f * fabsf(optix::dot(wi, dg.normal)) / (*probability);
    *ray = optix::Ray(dg.point, wi, NormalRay, rayEpsilon);
    return true;
}

/* 
 * ===  FUNCTION  ==============================================================
 *         Name:  traceUntilNonSpecularSurface
 *  Description:  Trace $ray until it intersects a non-specular surface or
 *                $depth reaches the $maxDepth.
 *
 *                Note that the caller is responsible for allocating memory
 *                blocks for those pointers. The function would write its
 *                temporary and final results to those pointers.
 *
 *                And also, the caller should initialize the $depth variable
 *                properly, e.g. set it to 0, or some number else if you're
 *                tracing a ray from an intermidiate stage of your algorithm.
 *
 *                For example: in the photon shooting pass, you may want to
 *                store 2 photons. You can set $depth to 0 when starting to
 *                trace the first photon, and then save the intersection
 *                information to the photon. Then you call the bounce() function
 *                to find the new ray, and now you can call this function
 *                the second time without setting any value to $depth, because
 *                it was previously set properly by this function. The function
 *                will start tracing the ray from the previous depth, so it
 *                won't exceed the $maxDepth value.
 * =============================================================================
 */
__device__ __forceinline__ bool traceUntilNonSpecularSurface(
        optix::Ray * ray, unsigned int maxDepth, unsigned int * depth,
        Intersection * intersection, BSDF * bsdf,
        optix::float3 * wo, optix::float3 * throughput)
{
    if (*depth == maxDepth)
        return false;

    NormalRayPayload payload;
    payload.setIntersectionBuffer(intersection);
    optix::float3 wi = ray->direction;
    for (unsigned int i = 0; i < maxDepth; ++i) {
        // Intersect with the scene.
        payload.reset();
        rtTrace(rootObject, *ray, payload);
        // Terminate immediately if not hit.
        if (!payload.isHit)
            return false;

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
        if (!bounce(ray, *intersection->dg(), *bsdf, sample, &probability, throughput,
                BxDF::Type(BxDF::Reflection | BxDF::Transmission | BxDF::Specular)))
        {
            return false;
        }
        wi = ray->direction;
    }

    return false;
}

#endif  /* -----  #ifdef __CUDACC__  ----- */

}   /* -----  end of namespace MaoPPM  ----- */

#endif  /* -----  #ifndef IGPPM_CORE_TRANSPORT_H  ----- */
