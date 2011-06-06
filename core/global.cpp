/*
 * =====================================================================================
 *
 *       Filename:  global.cpp
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  2011-05-02 14:38:43
 *
 *         Author:  Chun-Wei Huang (LittleCVR), 
 *        Company:  Communication & Multimedia Laboratory,
 *                  Department of Computer Science & Information Engineering,
 *                  National Taiwan University
 *
 * =====================================================================================
 */





/* #####   HEADER FILE INCLUDES   ################################################### */

/*-----------------------------------------------------------------------------
 *  header files from std C/C++
 *-----------------------------------------------------------------------------*/
#include    <cstdarg>
#include    <cstdio>
#include    <cstdlib>

/*-----------------------------------------------------------------------------
 *  header files of our own
 *-----------------------------------------------------------------------------*/
#include    "global.h"

/*-----------------------------------------------------------------------------
 *  namespace
 *-----------------------------------------------------------------------------*/
using namespace std;
using namespace MaoPPM;





/* #####   MACROS  -  LOCAL TO THIS SOURCE FILE   ################################### */

/* #####   TYPE DEFINITIONS  -  LOCAL TO THIS SOURCE FILE   ######################### */

/* #####   DATA TYPES  -  LOCAL TO THIS SOURCE FILE   ############################### */

/* #####   VARIABLES  -  LOCAL TO THIS SOURCE FILE   ################################ */

/* #####   PROTOTYPES  -  LOCAL TO THIS SOURCE FILE   ############################### */

/* #####   FUNCTION DEFINITIONS  -  EXPORTED FUNCTIONS   ############################ */

/* #####   FUNCTION DEFINITIONS  -  LOCAL TO THIS SOURCE FILE   ##################### */

void MaoPPM::fatal(const char * message, ... )
{
    va_list arguments;
    va_start(arguments, message);
    fprintf(stderr, "fatal: ");
    vfprintf(stderr, message, arguments);
    exit(EXIT_FAILURE);
}



void MaoPPM::critical(const char * message, ... )
{
    va_list arguments;
    va_start(arguments, message);
    fprintf(stderr, "critical: ");
    vfprintf(stderr, message, arguments);
}



#ifdef DEBUG
void MaoPPM::warning(const char * message, ... )
{
    va_list arguments;
    va_start(arguments, message);
    fprintf(stderr, "warning: ");
    vfprintf(stderr, message, arguments);
}
#endif



#ifdef DEBUG
void MaoPPM::debug(const char * message, ... )
{
    va_list arguments;
    va_start(arguments, message);
    fprintf(stderr, "debug: ");
    vfprintf(stderr, message, arguments);
}
#endif
