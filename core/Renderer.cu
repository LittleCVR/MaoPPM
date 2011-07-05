/*
 * =============================================================================
 *
 *       Filename:  Renderer.cu
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  2011-07-04 22:09:38
 *
 *         Author:  Chun-Wei Huang (LittleCVR), 
 *        Company:  Communication & Multimedia Laboratory,
 *                  Department of Computer Science & Information Engineering,
 *                  National Taiwan University
 *
 * =============================================================================
 */

/*----------------------------------------------------------------------------
 *  Header files from OptiX
 *----------------------------------------------------------------------------*/
#include    <optix_world.h>

/*----------------------------------------------------------------------------
 *  header files of our own
 *----------------------------------------------------------------------------*/
#include    "global.h"

/*----------------------------------------------------------------------------
 *  namespace
 *----------------------------------------------------------------------------*/
using namespace optix;
using namespace MaoPPM;



rtDeclareVariable(uint2, launchIndex, rtLaunchIndex, );

rtBuffer<float4, 2>  outputBuffer;



/* 
 * ===  FUNCTION  ==============================================================
 *         Name:  clearOutputBuffer
 *  Description:  
 * =============================================================================
 */
RT_PROGRAM void clearOutputBuffer()
{
    outputBuffer[launchIndex] = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
}   /* -----  end of function clearOutputBuffer  ----- */
