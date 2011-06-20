/*
 * =============================================================================
 *
 *       Filename:  exception.cu
 *
 *    Description:  A generic exception handling program.
 *
 *        Version:  1.0
 *        Created:  2011-06-21 01:19:34
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
 *  namespace
 *----------------------------------------------------------------------------*/
using namespace optix;



/* 
 * ===  FUNCTION  ==============================================================
 *         Name:  handleException
 *  Description:  
 * =============================================================================
 */
RT_PROGRAM void handleException()
{
    unsigned int code = rtGetExceptionCode();
    rtPrintExceptionDetails();
}   /* -----  end of function handleException  ----- */
