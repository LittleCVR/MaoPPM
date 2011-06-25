/*
 * =============================================================================
 *
 *       Filename:  global.h
 *
 *    Description:  This file contains default global settings and class forward
 *                  declarations.
 *
 *        Version:  1.0
 *        Created:  2011-03-29 16:04:54
 *
 *         Author:  Chun-Wei Huang (LittleCVR), 
 *        Company:  Communication & Multimedia Laboratory,
 *                  Department of Computer Science & Information Engineering,
 *                  National Taiwan University
 *
 * =============================================================================
 */

#ifndef IGPPM_CORE_GLOBAL_H
#define IGPPM_CORE_GLOBAL_H

namespace MaoPPM {

/*----------------------------------------------------------------------------
 *  default global settings
 *----------------------------------------------------------------------------*/
static const unsigned int  DEFAULT_STACK_SIZE       = 4096;
static const unsigned int  DEFAULT_WIDTH            = 512;
static const unsigned int  DEFAULT_HEIGHT           = 512;
static const float         DEFAULT_TIMEOUT          = 0.0f;    // forever
static const unsigned int  DEFAULT_INPUT_HEAP_SIZE  = 32;      // 32 bytes
static const float         DEFAULT_RAY_EPSILON      = 1.0e-2f;
static const unsigned int  DEFAULT_MAX_RAY_DEPTH    = 4;

/*----------------------------------------------------------------------------
 *  typedefs
 *----------------------------------------------------------------------------*/
typedef unsigned int Index;

/*----------------------------------------------------------------------------
 *  class forward declarations
 *----------------------------------------------------------------------------*/
class BxDF;
class BSDF;
class DifferentialGeometry;
class Intersection;
class Renderer;
class Scene;
class SceneBuilder;
class System;

/*----------------------------------------------------------------------------
 *  Helper functions for debugging.
 *
 *  fatal(...) and critical(...) are used to issue error messages, with the
 *  only difference that fatal() will exit the program immediatly after the
 *  message was print out, where critical(...) will not.
 *
 *  warning(...) and debug(...) are for debugging use only. Both of them won't
 *  do anything if the NDEBUG macro was defined.
 *----------------------------------------------------------------------------*/

void fatal    (const char * message, ... );
void critical (const char * message, ... );

#ifndef NDEBUG
    void warning (const char * message, ... );
    void debug   (const char * message, ... );
#else   /* -----  else of #ifndef NDEBUG  ----- */
    inline void warning (const char * message, ... ) { /* nothing to do */ }
    inline void debug   (const char * message, ... ) { /* nothing to do */ }
#endif  /* -----  end of #ifndef NDEBUG  ----- */

}   /* ----------  end of namespace MaoPPM  ---------- */

#endif  /* ----- #ifndef IGPPM_CORE_GLOBAL_H  ----- */
