/*
 * =============================================================================
 *
 *       Filename:  Sphere.cu
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  2011-06-28 14:25:00
 *
 *         Author:  Chun-Wei Huang (LittleCVR), 
 *        Company:  Communication & Multimedia Laboratory,
 *                  Department of Computer Science & Information Engineering,
 *                  National Taiwan University
 *
 * =============================================================================
 */

/*-----------------------------------------------------------------------------
 *  Header files from OptiX
 *-----------------------------------------------------------------------------*/
#include    <optix_world.h>

/*-----------------------------------------------------------------------------
 *  header files of our own
 *-----------------------------------------------------------------------------*/
#include    "global.h"
#include    "utility.h"
#include    "DifferentialGeometry.h"

/*-----------------------------------------------------------------------------
 *  namespace
 *-----------------------------------------------------------------------------*/
using namespace optix;
using namespace MaoPPM;



rtDeclareVariable(float, radius, , ); 
rtDeclareVariable(float, phiMax, , ); 
rtDeclareVariable(float, zMin  , , ); 
rtDeclareVariable(float, zMax  , , ); 
rtDeclareVariable(Matrix4x4, worldToObject, , ); 
rtDeclareVariable(Matrix4x4, objectToWorld, , ); 
rtDeclareVariable(DifferentialGeometry, geometricDG, attribute differential_geometry, ); 
rtDeclareVariable(optix::Ray, currentRay, rtCurrentRay, );



/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  boundingBox
 *  Description:  
 * =====================================================================================
 */
RT_PROGRAM void boundingBox(int primitiveIndex, float result[6])
{
    float3 origin = transformPoint(objectToWorld, make_float3(0.0f, 0.0f, 0.0f));
    result[0] = origin.x - radius;
    result[1] = origin.y - radius;
    result[2] = origin.z - radius;
    result[3] = origin.x + radius;
    result[4] = origin.y + radius;
    result[5] = origin.z + radius;
}   /* -----  end of function boundingBox  ----- */



/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  intersect
 *  Description:  
 * =====================================================================================
 */
RT_PROGRAM void intersect(int primitiveIndex)
{
    float3 o = transformPoint(worldToObject, currentRay.origin);
    float3 d = transformVector(worldToObject, currentRay.direction);
    // Compute quadratic sphere coefficients
    float A = d.x * d.x + d.y * d.y + d.z * d.z;
    float B = 2 * (d.x * o.x + d.y * o.y + d.z * o.z);
    float C = o.x*o.x + o.y*o.y + o.z*o.z - radius*radius;

    // Solve quadratic equation for _t_ values
    float t0, t1;
    if (!solveQuadraticEquation(A, B, C, &t0, &t1))
        return;

    // Compute intersection distance along ray
    if (t0 > currentRay.tmax || t1 < currentRay.tmin)
        return;
    float thit = t0;
    if (t0 < currentRay.tmin) {
        thit = t1;
        if (thit > currentRay.tmax) return;
    }

    // Compute sphere hit position and $\phi$
    float3 phit = o + thit*d;
    if (phit.x == 0.f && phit.y == 0.f) phit.x = 1e-5f * radius;
    float phi = atan2f(phit.y, phit.x);
    if (phi < 0.f) phi += 2.f*M_PIf;

//    rtPrintf("zMin: %f, zMax: %f, phi: %f, phiMax: %f, phit: (%f, %f, %f)\n",
//            zMin, zMax, phi, phiMax, phit.x, phit.y, phit.z);
    // Test sphere intersection against clipping parameters
    if ((zMin > -radius && phit.z < zMin) ||
        (zMax <  radius && phit.z > zMax) || phi > phiMax)
    {
        if (thit == t1) return;
        if (t1 > currentRay.tmax) return;
        thit = t1;
        // Compute sphere hit position and $\phi$
        phit = o + thit*d;
        if (phit.x == 0.f && phit.y == 0.f) phit.x = 1e-5f * radius;
        phi = atan2f(phit.y, phit.x);
        if (phi < 0.f) phi += 2.f*M_PIf;
        if ((zMin > -radius && phit.z < zMin) ||
            (zMax <  radius && phit.z > zMax) || phi > phiMax)
            return;
    }

    // Intersect ray with triangle.
    if (rtPotentialIntersection(thit)) {
        // Find parametric representation of sphere hit
        float theta = acosf(clamp(phit.z / radius, -1.f, 1.f));
        float thetaMin = acosf(clamp(zMin/radius, -1.f, 1.f));
        float thetaMax = acosf(clamp(zMax/radius, -1.f, 1.f));

        // Compute sphere $\dpdu$ and $\dpdv$
        float zradius = sqrtf(phit.x*phit.x + phit.y*phit.y);
        float invzradius = 1.f / zradius;
        float cosphi = phit.x * invzradius;
        float sinphi = phit.y * invzradius;
        float3 dpdu = make_float3(-phiMax * phit.y, phiMax * phit.x, 0);
        float3 dpdv = (thetaMax-thetaMin) *
            make_float3(phit.z * cosphi, phit.z * sinphi, -radius * sinf(theta));
        float3 N = normalize(cross(dpdu, dpdv));

        // Update _tHit_ for quadric intersection
        geometricDG.point  = transformPoint(objectToWorld, phit);
        geometricDG.dpdu   = transformVector(objectToWorld, dpdu);
        geometricDG.dpdv   = transformVector(objectToWorld, dpdv);
        // Normals must be transformed by the inverse transpose.
        geometricDG.normal = transformVector(worldToObject.transpose(), N);

        rtReportIntersection(0);    // we have only one material
    }
}   /* -----  end of function intersect  ----- */
