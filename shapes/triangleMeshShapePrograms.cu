/*
 * =====================================================================================
 *
 *       Filename:  triangleMeshShapePrograms.cu
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

/*-----------------------------------------------------------------------------
 *  namespace
 *-----------------------------------------------------------------------------*/
using namespace optix;
using namespace MaoPPM;



rtBuffer<float3, 1>  vertexList;
rtBuffer<uint3 , 1>  vertexIndexList;
rtDeclareVariable(float3, geometricNormal,  attribute geometric_normal, ); 
rtDeclareVariable(float3, shadingNormal  ,  attribute shading_normal  , ); 
rtDeclareVariable(optix::Ray, currentRay, rtCurrentRay, );



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

    float3 v0 = vertexList[vertexIndex.x];
    float3 v1 = vertexList[vertexIndex.y];
    float3 v2 = vertexList[vertexIndex.z];

    // Intersect ray with triangle.
    float3 normal;
    float  tHit, beta, gamma;
    if (intersect_triangle(currentRay, v0, v1, v2, normal, tHit, beta, gamma)) {
        if (rtPotentialIntersection(tHit)) {
            shadingNormal   = normal;
            geometricNormal = normal;
            rtReportIntersection(0);    // we have only one material
        }
    }
}   /* -----  end of function intersect  ----- */
