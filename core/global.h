/*
 * =====================================================================================
 *
 *       Filename:  global.h
 *
 *    Description:  This file contains global settings and class declarations.
 *
 *        Version:  1.0
 *        Created:  2011/3/29 16:04:54
 *
 *         Author:  Chun-Wei Huang (LittleCVR), 
 *        Company:  Communication & Multimedia Laboratory,
 *                  Department of Computer Science & Information Engineering,
 *                  National Taiwan University
 *
 * =====================================================================================
 */

#ifndef GLOBAL_H
#define GLOBAL_H





/* #####   HEADER FILE INCLUDES   ################################################### */

/*-----------------------------------------------------------------------------
 *  header files from OptiX
 *-----------------------------------------------------------------------------*/
#include    <optix_world.h>





/* #####   MACROS  -  LOCAL TO THIS SOURCE FILE   ################################### */

#define DEBUG 1

#define STACK_SIZE          2000

#define INITIAL_WIDTH       128
#define INITIAL_HEIGHT      128
#define PHOTON_WIDTH        128
#define PHOTON_HEIGHT       128

#define RAY_EPSILON         0.001f
#define MAX_RAY_DEPTH       4

#define PIXEL_SAMPLE_HIT    1

#define PHOTON_COUNT        4
#define PHOTON_NULL         0
#define PHOTON_LEAF         1
#define AXIS_X              2 
#define AXIS_Y              4
#define AXIS_Z              8





namespace MaoPPM {

enum Pass {
    PixelSamplingPass,
    ImportonShootingPass,
    PhotonShootingPass,
    GatheringPass,
    nPasses
};  /* ----------  end of enum Pass  ---------- */



enum RayType {
    PixelSamplingRay,
    ImportonShootingRay,
    PhotonShootingRay,
    GatheringRay,
    nRayTypes
};  /* ----------  end of enum RayType  ---------- */



typedef struct Light {
    optix::float3   position;
    optix::float3   flux;
} Light ;



typedef struct PixelSample {
    optix::uint     flags;
    optix::float3   position;
    optix::float3   incidentDirection;
    optix::float3   normal;
    optix::float3   flux;
    int             material;
    optix::float3   Kd;
    optix::float3   Ks;
    float           invRoughness;
    optix::uint     nPhotons;
    float           radiusSquared;
} PixelSample ; /* ----------  end of struct PixelSample  ---------- */



typedef struct PixelSamplingRayPayload {
    float           attenuation;
} PixelSamplingRayPayload ; /* ----------  end of struct PixelSamplingRayPayload  ---------- */



typedef struct PhotonShootingRayPayload {
    optix::uint     nPhotons;
    optix::uint     photonIndexBase;
    optix::uint     sampleIndexBase;
    float           attenuation;
    optix::uint     depth;
    optix::float3   flux;
} PhotonShootingRayPayload ; /* ----------  end of struct PhotonShootingRayPayload  ---------- */



typedef struct GatheringRayPayload {
    float   attenuation;
} GatheringRayPayload ; /* ----------  end of struct PixelSample  ---------- */



typedef struct Photon {
    optix::float3   position;
    optix::float3   flux;
    optix::float3   normal;                     /* the surface normal */
    optix::float3   incidentDirection;          /* pointed outward */
    optix::uint     axis;

    static bool positionXComparator(const Photon & photon1, const Photon & photon2)
    {
        return photon1.position.x < photon2.position.x;
    }
    static bool positionYComparator(const Photon & photon1, const Photon & photon2)
    {
        return photon1.position.y < photon2.position.y;
    }
    static bool positionZComparator(const Photon & photon1, const Photon & photon2)
    {
        return photon1.position.z < photon2.position.z;
    }
} Photon ;  /* ----------  end of struct Photon  ---------- */



class Scene;
class SceneBuilder;



/*-----------------------------------------------------------------------------
 *  Helper functions for debugging.
 *
 *  fatal(...) and critical(...) are used to issue error messages, with the
 *  only difference that fatal() will exit the program immediatly after the
 *  message was print out, where critical(...) will not.
 *
 *  warning(...) and debug(...) are for debugging use only. Both of them won't
 *  do anything if the DEBUG macro was not defined.
 *-----------------------------------------------------------------------------*/

void fatal    (const char * message, ... );
void critical (const char * message, ... );

#ifdef DEBUG
    void warning (const char * message, ... );
    void debug   (const char * message, ... );
#else
    inline void warning (const char * message, ... ) { /* nothing to do */ }
    inline void debug   (const char * message, ... ) { /* nothing to do */ }
#endif

}   /* ----------  end of namespace MaoPPM  ---------- */





#endif  /* ----- #ifndef GLOBAL_H  ----- */
