/*
 * =============================================================================
 *
 *       Filename:  PPMRenderer.h
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  2011-06-08 12:00:58
 *
 *         Author:  Chun-Wei Huang (LittleCVR), 
 *        Company:  Communication & Multimedia Laboratory,
 *                  Department of Computer Science & Information Engineering,
 *                  National Taiwan University
 *
 * =============================================================================
 */

#ifndef PPM_RENDERER_H
#define PPM_RENDERER_H

/*----------------------------------------------------------------------------
 *  header files from OptiX
 *----------------------------------------------------------------------------*/
#include    <optix_world.h>

/*----------------------------------------------------------------------------
 *  header files of our own
 *----------------------------------------------------------------------------*/
#include    "global.h"
#include    "Renderer.h"



namespace MaoPPM {
/*
 * =============================================================================
 *        Class:  PPMRenderer
 *  Description:  
 * =============================================================================
 */
class PPMRenderer : public Renderer {
    public:     // methods
        PPMRenderer(Scene * scene = NULL);
        ~PPMRenderer();

    public:
        enum Pass { PixelSamplingPass, PhotonShootingPass, GatheringPass };
        unsigned int nPasses() const { return 3; }

        enum RayType { PixelSamplingRay, PhotonShootingRay, GatheringRay };
        unsigned int nRayTypes() const { return 3; }

#define PHOTON_WIDTH        128
#define PHOTON_HEIGHT       128

#define PIXEL_SAMPLE_HIT    1

#define PHOTON_COUNT        4
#define PHOTON_NULL         0
#define PHOTON_LEAF         1
#define AXIS_X              2 
#define AXIS_Y              4
#define AXIS_Z              8

        typedef struct PixelSample {
            optix::uint     flags;
            optix::float3   position;
            optix::float3   incidentDirection;
            optix::float3   normal;
            optix::float3   flux;
            int             material;
            optix::float3   Kd;
            optix::float3   Ks;
            float           exponent;
            optix::uint     nPhotons;
            float           radiusSquared;
        } PixelSample ;

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
        } Photon ;

        typedef struct PixelSamplingRayPayload {
            float           attenuation;
        } PixelSamplingRayPayload ;

        typedef struct PhotonShootingRayPayload {
            optix::uint     nPhotons;
            optix::uint     photonIndexBase;
            optix::uint     sampleIndexBase;
            float           attenuation;
            optix::uint     depth;
            optix::float3   flux;
        } PhotonShootingRayPayload ;

        typedef struct GatheringRayPayload {
            float   attenuation;
        } GatheringRayPayload ;

        void setMaterialPrograms(const std::string & name,
                optix::Material & material);

    public:     // methods
        void    init();
        void    render(const Scene::RayGenCameraData & cameraData);
        void    resize(unsigned int width, unsigned int height);

    private:    // methods
        void    initPixelSamplingPassData();
        void    initImportonShootingPassData();
        void    initPhotonShootingPassData();

    private:
        void createPhotonMap();
        void buildPhotonMapAcceleration(Photon * photonList,
                optix::uint start, optix::uint end, Photon * photonMap,
                optix::uint root, optix::float3 bbMin, optix::float3 bbMax);

    private:    // attributes
        optix::Buffer           m_pixelSampleList;
        optix::Buffer           m_importonMap;
        optix::uint             m_nEmittedPhotons;
        optix::Buffer           m_photonList;
        optix::Buffer           m_photonMap;
};  /* -----  end of class PPMRenderer  ----- */
}   /* -----  end of namespace MaoPPM  ----- */

#endif  /* -----  #ifndef PPM_RENDERER_H  ----- */
