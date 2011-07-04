/*
 * =============================================================================
 *
 *       Filename:  Camera.h
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  2011-07-02 15:30:48
 *
 *         Author:  Chun-Wei Huang (LittleCVR), 
 *        Company:  Communication & Multimedia Laboratory,
 *                  Department of Computer Science & Information Engineering,
 *                  National Taiwan University
 *
 * ============================================================================
 */

#ifndef MAOPPM_CORE_CAMERA_H
#define MAOPPM_CORE_CAMERA_H

/* #####   PROTOTYPE           ############################################## */

/*----------------------------------------------------------------------------
 *  header files from OptiX
 *----------------------------------------------------------------------------*/
#include    <optix_world.h>

/*----------------------------------------------------------------------------
 *  header files of our own
 *----------------------------------------------------------------------------*/
#include    "global.h"
#include    "transport.h"
#include    "utility.h"
#include    "Intersection.h"



namespace MaoPPM {
/*
 * =============================================================================
 *        Class:  Camera
 *  Description:  
 * =============================================================================
 */
class Camera {
#ifdef __CUDACC__
    public:
        __device__ __forceinline__ float focalLength() const
        {
            return optix::length(W);
        }
        __device__ __forceinline__ float vFOV() const
        {
            return 2.0f * atan2f(optix::length(V), optix::length(W));
        }
        __device__ __forceinline__ float hFOV() const
        {
            return 2.0f * atan2f(optix::length(U), optix::length(W));
        }

        __device__ __forceinline__ optix::Matrix4x4 worldToCamera() const;
        __device__ __forceinline__ optix::Matrix4x4 cameraToWorld() const;
        __device__ __forceinline__ optix::Matrix4x4 cameraToScreen() const;
        __device__ __forceinline__ optix::Matrix4x4 screenToRaster() const;
        __device__ __forceinline__ optix::Matrix4x4 worldToRaster() const;

        __device__ __forceinline__ optix::Ray generateCameraRay(
                unsigned int x, unsigned int y, const optix::float2 & sample,
                RayType rayType, float epsilon) const;

#endif  /* -----  #ifdef __CUDACC__  ----- */

    public:  // should be private
        optix::float3  position;
        optix::float3  U;
        optix::float3  V;
        optix::float3  W;
        unsigned int   width;
        unsigned int   height;
};  /* -----  end of class Camera  ----- */
}   /* -----  end of namespace MaoPPM  ----- */



/* #####   IMPLEMETATION       ############################################## */

namespace MaoPPM {

#ifdef __CUDACC__

__device__ __forceinline__ optix::Matrix4x4 Camera::worldToCamera() const
{
    float m[16] = { 1, 0, 0, -position.x,
                    0, 1, 0, -position.y,
                    0, 0, 1, -position.z,
                    0, 0, 0,           1, };
    optix::float3 X = optix::normalize(U);
    optix::float3 Y = optix::normalize(V);
    optix::float3 Z = optix::normalize(W);
    return createCoordinateSystemTransform(X, Y, Z) * optix::Matrix4x4(m);
}
__device__ __forceinline__ optix::Matrix4x4 Camera::cameraToWorld() const
{
    return worldToCamera().transpose();
}
__device__ __forceinline__ optix::Matrix4x4 Camera::cameraToScreen() const
{
    float n = focalLength();
    /* TODO: hard coded fat clipping plane */
    float f = 1.0e+6;
    float a = 1.0f / tanf(hFOV() / 2.0f);
    float b = 1.0f / tanf(vFOV() / 2.0f);
    // Perform projective divide
    float m[16] = { a, 0,           0,              0,
                    0, b,           0,              0,
                    0, 0, f / (f - n), -f*n / (f - n),
                    0, 0,           1,              0 };
    return optix::Matrix4x4(m);
}
__device__ __forceinline__ optix::Matrix4x4 Camera::screenToRaster() const
{
    float a = static_cast<float>(width) / 2;
    float b = static_cast<float>(height) / 2;
    float m[16] = { a, 0, 0, a,
                    0, b, 0, b,
                    0, 0, 1, 0,
                    0, 0, 0, 1 };
    return optix::Matrix4x4(m);
}
__device__ __forceinline__ optix::Matrix4x4 Camera::worldToRaster() const
{
    return screenToRaster() * cameraToScreen() * worldToCamera();
}

__device__ __forceinline__ optix::Ray Camera::generateCameraRay(
        unsigned int x, unsigned int y, const optix::float2 & sample,
        RayType rayType, float epsilon) const
{
    optix::float2 pixel = optix::make_float2(x, y);
    optix::float2 screen = optix::make_float2(width, height);
    optix::float2 d = (pixel + sample) / screen * 2.0f - 1.0f;
    optix::float3 worldD = optix::normalize(d.x*U + d.y*V + W);
    return optix::Ray(position, worldD, rayType, epsilon);
}

#endif  /* -----  #ifdef __CUDACC__  ----- */

}   /* -----  end of namespace MaoPPM  ----- */

#endif  /* -----  #ifndef MAOPPM_CORE_CAMERA_H  ----- */
