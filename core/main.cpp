/*
 * =====================================================================================
 *
 *       Filename:  main.cpp
 *
 *    Description:  File that contains functions to control main workflow.
 *
 *        Version:  1.0
 *        Created:  2011-03-23 11:27:02
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
#include    <cstdlib>
#include    <iostream>

/*-----------------------------------------------------------------------------
 *  header files from sutil
 *-----------------------------------------------------------------------------*/
#include    <GLUTDisplay.h>

/*-----------------------------------------------------------------------------
 *  header files of our own
 *-----------------------------------------------------------------------------*/
#include    "global.h"
#include    "core/Scene.h"
#include    "SceneBuilder.h"

/*-----------------------------------------------------------------------------
 *  namespace
 *-----------------------------------------------------------------------------*/
using namespace std;
using namespace optix;
using namespace MaoPPM;




/* #####   MACROS  -  LOCAL TO THIS SOURCE FILE   ################################### */





/* #####   TYPE DEFINITIONS  -  LOCAL TO THIS SOURCE FILE   ######################### */





/* #####   DATA TYPES  -  LOCAL TO THIS SOURCE FILE   ############################### */





/* #####   VARIABLES  -  LOCAL TO THIS SOURCE FILE   ################################ */

bool    useCornellBox   = false;    // If set to true, using the cornell box scene.
float   timeout         = 0.0f;     // The program will run for only timeout secs,
                                    // set it to 0 to indicate the program should run
                                    // forever.

SceneBuilder * g_sceneBuilder = NULL;





/* #####   PROTOTYPES  -  LOCAL TO THIS SOURCE FILE   ############################### */

/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  parseAuguments
 *  Description:  Parse the input arguments and set up the corresponding state of the
 *                program.
 * =====================================================================================
 */
void    parseArguments(int argc, char ** argv);

/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  printUsageAndExit
 *  Description:  Print usage and exit the program. Should be called if an error has
 *                occured when parsing the input arguments. The fileName argument is the
 *                executable file name.
 * =====================================================================================
 */
void    printUsageAndExit(const char * fileName, bool doExit = true);





/* #####   FUNCTION DEFINITIONS  -  LOCAL TO THIS SOURCE FILE   ##################### */

int main(int argc, char ** argv)
{
    GLUTDisplay::init(argc, argv);

    if (!GLUTDisplay::isBenchmark())
        printUsageAndExit(argv[0], false);

//    extern int yydebug;
//    yydebug = 1;

    parseArguments(argc, argv);

    try {
        Scene * scene = new Scene;
        /* :TODO:2011/3/28 18:00:54:: Try to understand what consequenses this line will cause. */
        GLUTDisplay::setUseSRGB(true);
        GLUTDisplay::setProgressiveDrawingTimeout(timeout);
        GLUTDisplay::run("MaoPPM", scene, GLUTDisplay::CDProgressive);
    } catch(const Exception & e) {
        sutilReportError(e.getErrorString().c_str());
        exit(EXIT_FAILURE);
    }

    return EXIT_SUCCESS;
}     /* ----------  end of function main  ---------- */



void parseArguments(int argc, char ** argv)
{
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        // -h or --help
        if (arg == "--help" || arg == "-h")
            printUsageAndExit(argv[0]);
        // -t or --timeout
        else if (arg == "--timeout" || arg == "-t") {
            if (++i < argc) {
                timeout = static_cast<float>(atof(argv[i]));
                cerr << "Timeout " << timeout << " seconds." << endl;
            } else {
                std::cerr << "Missing argument to " << arg << std::endl;
                printUsageAndExit(argv[0]);
            }
        }
        // otherwise
        else {
            std::cerr << "Unknown option: '" << arg << std::endl;
            printUsageAndExit(argv[0]);
        }
    }
}   /* -----  end of function parseArguments  ----- */



void printUsageAndExit(const char * fileName, bool doExit)
{
    std::cerr
        << "Usage  : " << fileName << " [options]" << std::endl
        << "App options:" << std::endl
        << "    -h | --help             Print this usage message"                                     << std::endl
        << "    -t | --timeout <sec>    Seconds before stopping rendering. Set to 0 for no stopping." << std::endl
        << std::endl;

    GLUTDisplay::printUsage();

    if (doExit)
        exit(EXIT_FAILURE);
}   /* -----  end of function printUsageAndExit  ----- */
