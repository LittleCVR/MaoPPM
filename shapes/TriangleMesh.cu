/*
 * =====================================================================================
 *
 *       Filename:  TriangleMesh.cu
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  2011-04-18 15:15:26
 *
 *         Author:  Chun-Wei Huang (LittleCVR), 
 *        Company:  Communication & Multimedia Laboratory,
 *                  Department of Computer Science & Information Engineering,
 *                  National Taiwan University
 *
 * =====================================================================================
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



rtBuffer<float3, 1>  vertexList;
rtBuffer<uint3 , 1>  vertexIndexList;
rtDeclareVariable(DifferentialGeometry, geometricDG, attribute differential_geometry, ); 
rtDeclareVariable(optix::Ray, currentRay, rtCurrentRay, );



__device__ __inline__ void getUVs(float uvs[3][2])
{
    /* TODO: use input UV array if possible */
    uvs[0][0] = 0.0f; uvs[0][1] = 0.0f;
    uvs[1][0] = 1.0f; uvs[1][1] = 0.0f;
    uvs[2][0] = 1.0f; uvs[2][1] = 1.0f;
}



/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  boundingBox
 *  Description:  Create bounding box for the specific triangle.
 * =====================================================================================
 */
RT_PROGRAM void boundingBox(int primitiveIndex, float result[6])
{
    uint3 vertexIndex = vertexIndexList[primitiveIndex];

    float3 v0 = vertexList[vertexIndex.x];
    float3 v1 = vertexList[vertexIndex.y];
    float3 v2 = vertexList[vertexIndex.z];

    result[0] = fmin(fmin(v0.x, v1.x), v2.x);
    result[1] = fmin(fmin(v0.y, v1.y), v2.y);
    result[2] = fmin(fmin(v0.z, v1.z), v2.z);
    result[3] = fmax(fmax(v0.x, v1.x), v2.x);
    result[4] = fmax(fmax(v0.y, v1.y), v2.y);
    result[5] = fmax(fmax(v0.z, v1.z), v2.z);
}   /* -----  end of function boundingBox  ----- */



/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  intersect
 *  Description:  Intersect a ray with an triangle.
 * =====================================================================================
 */
RT_PROGRAM void intersect(int primitiveIndex)
{
    uint3 vertexIndex = vertexIndexList[primitiveIndex];

    float3 p1 = vertexList[vertexIndex.x];
    float3 p2 = vertexList[vertexIndex.y];
    float3 p3 = vertexList[vertexIndex.z];
    float3 e1 = p2 - p1;
    float3 e2 = p3 - p1;
    float3 s1 = cross(currentRay.direction, e2);
    float divisor = dot(s1, e1);
    if (divisor == 0.0f)
        return;
    float invDivisor = 1.0f / divisor;

    // Compute first barycentric coordinate
    float3 d = currentRay.origin - p1;
    float b1 = dot(d, s1) * invDivisor;
    if (b1 < 0.0f || b1 > 1.0f)
        return;

    // Compute second barycentric coordinate
    float3 s2 = cross(d, e1);
    float b2 = dot(currentRay.direction, s2) * invDivisor;
    if (b2 < 0.0f || b1 + b2 > 1.0f)
        return;

    // Compute t to intersection point
    float t = dot(e2, s2) * invDivisor;

    // Intersect ray with triangle.
    if (rtPotentialIntersection(t)) {
        // compute dpdu, dpdv
        float uvs[3][2]; getUVs(uvs);
        float du1 = uvs[0][0] - uvs[2][0];
        float du2 = uvs[1][0] - uvs[2][0];
        float dv1 = uvs[0][1] - uvs[2][1];
        float dv2 = uvs[1][1] - uvs[2][1];
        float3 dp1 = p1 - p3, dp2 = p2 - p3;
        float determinant = du1 * dv2 - dv1 * du2;
        if (determinant == 0.0f)
            createCoordinateSystem(cross(e2, e1), &geometricDG.dpdu, &geometricDG.dpdv);
        else {
            float invdet = 1.0f / determinant;
            geometricDG.dpdu = ( dv2 * dp1 - dv1 * dp2) * invdet;
            geometricDG.dpdv = (-du2 * dp1 + du1 * dp2) * invdet;
        }

        // hit point and normal
        /* TODO: interpolate normal */
        geometricDG.point  = currentRay.origin + t*currentRay.direction;
        geometricDG.normal = normalize(cross(geometricDG.dpdu, geometricDG.dpdv));

        rtReportIntersection(0);    // we have only one material
    }
}   /* -----  end of function intersect  ----- */
