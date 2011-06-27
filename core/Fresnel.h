/*
 * =====================================================================================
 *
 *       Filename:  Fresnel.h
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  2011-06-19 14:06:23
 *
 *         Author:  Chun-Wei Huang (LittleCVR), 
 *        Company:  Communication & Multimedia Laboratory,
 *                  Department of Computer Science & Information Engineering,
 *                  National Taiwan University
 *
 * =====================================================================================
 */

#ifndef IGPPM_CORE_FRESNEL_H
#define IGPPM_CORE_FRESNEL_H

/*-----------------------------------------------------------------------------
 *  header files from OptiX
 *-----------------------------------------------------------------------------*/
#include    <optix_world.h>

/*----------------------------------------------------------------------------
 *  header files of our own
 *----------------------------------------------------------------------------*/
#include    "global.h"
#include    "reflection.h"
#include    "utility.h"



namespace MaoPPM {
/* TODO: figure out WTF is Fresnel */
class Fresnel {
    public:
        enum Type {
            NoOp        = 1 << 0,
            Conductor   = 1 << 1,
            Dielectric  = 1 << 2
        };

#ifdef __CUDACC__
    public:
        __device__ __inline__ Fresnel(Type type) : m_type(type) { /* EMPTY */ }

    public:
        __device__ __inline__ Type type() const { return m_type; }

    public:
        // virtual optix::float3 evaluate(float cosi) const =0;
#endif  /* -----  #ifdef __CUDACC__  ----- */

    private:
        Type  m_type;
};

class FresnelNoOp : public Fresnel {
#ifdef __CUDACC__
    public:
        __device__ __inline__ FresnelNoOp() : Fresnel(NoOp) { /* EMPTY */ }

    public:
        __device__ __inline__ optix::float3 evaluate(float) const
        {
            return optix::make_float3(1.0f);
        }
#endif  /* -----  #ifdef __CUDACC__  ----- */
};

//class FresnelConductor : public Fresnel {
//#ifdef __CUDACC__
//    public:
//        __device__ __inline__ FresnelConductor(
//                const optix::float3 & eta, const optix::float3 & k) : Fresnel(Fresnel::Conductor),
//            m_eta(eta), m_k(k) { /* EMPTY */ }   
//
//        __device__ __inline__ optix::float3 evaluate(float cosi) const
//        {
//            cosi = fabsf(cosi);
//            const optix::float3 & eta = m_eta;
//            const optix::float3 & k   = m_k;
//            optix::float3 tmp = (eta*eta + k*k) * cosi*cosi;
//            optix::float3 Rparl2 = (tmp - (2.f * eta * cosi) + 1.f) /
//                (tmp + (2.f * eta * cosi) + 1.f); 
//            optix::float3 tmp_f = eta*eta + k*k;
//            optix::float3 Rperp2 =
//                (tmp_f - (2.f * eta * cosi) + cosi*cosi) /
//                (tmp_f + (2.f * eta * cosi) + cosi*cosi);
//            return (Rparl2 + Rperp2) / 2.f;
//        }
//#endif  /* -----  #ifdef __CUDACC__  ----- */
//
//    private:
//        optix::float3  m_eta;
//        optix::float3  m_k;
//};

class FresnelDielectric : public Fresnel {
#ifdef __CUDACC__
    public:
        __device__ __inline__ FresnelDielectric(float ei, float et) : Fresnel(Fresnel::Dielectric),
            eta_i(ei), eta_t(et) { /* EMPTY */ }

        __device__ __inline__ optix::float3 evaluate(float cosi) const 
        {
            // Compute Fresnel reflectance for dielectric
            cosi = optix::clamp(cosi, -1.0f, 1.0f);

            // Compute indices of refraction for dielectric
            bool entering = cosi > 0.0f;
            float ei = eta_i, et = eta_t;
            if (!entering) swap(ei, et);

            // Compute _sint_ using Snell's law
            float sint = ei/et * sqrtf(max(0.0f, 1.0f - cosi*cosi));
            if (sint >= 1.0f) {
                // Handle total internal reflection
                return optix::make_float3(1.0f);
            } else {
                cosi = fabsf(cosi);
                float cost = sqrtf(max(0.0f, 1.0f - sint*sint));
                optix::float3 Rparl = optix::make_float3(
                        ((et * cosi) - (ei * cost)) /
                        ((et * cosi) + (ei * cost)));
                optix::float3 Rperp = optix::make_float3(
                        ((ei * cosi) - (et * cost)) /
                        ((ei * cosi) + (et * cost)));
                return (Rparl*Rparl + Rperp*Rperp) / 2.0f;
            }
        }
#endif  /* -----  #ifdef __CUDACC__  ----- */

    private:
        float eta_i, eta_t;
};
}   /* -----  end of namespace MaoPPM  ----- */

#endif  /* -----  #ifndef IGPPM_CORE_FRESNEL_H  ----- */
