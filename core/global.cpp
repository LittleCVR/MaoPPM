/*
 * =============================================================================
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
 * =============================================================================
 */

#include    "global.h"

/*----------------------------------------------------------------------------
 *  header files from std C/C++
 *----------------------------------------------------------------------------*/
#include    <cstdarg>
#include    <cstdio>
#include    <cstdlib>

/*----------------------------------------------------------------------------
 *  namespace
 *----------------------------------------------------------------------------*/
using namespace std;
using namespace MaoPPM;



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



#ifndef NDEBUG
void MaoPPM::warning(const char * message, ... )
{
    va_list arguments;
    va_start(arguments, message);
    fprintf(stderr, "warning: ");
    vfprintf(stderr, message, arguments);
}
#endif



#ifndef NDEBUG
void MaoPPM::debug(const char * message, ... )
{
    va_list arguments;
    va_start(arguments, message);
    fprintf(stderr, "debug: ");
    vfprintf(stderr, message, arguments);
}
#endif
