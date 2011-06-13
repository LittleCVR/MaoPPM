/*
 * =============================================================================
 *
 *       Filename:  System.cpp
 *
 *    Description:  MaoPPM rendering system implementation file.
 *
 *        Version:  1.0
 *        Created:  2011-06-07 19:31:25
 *
 *         Author:  Chun-Wei Huang (LittleCVR), 
 *        Company:  Communication & Multimedia Laboratory,
 *                  Department of Computer Science & Information Engineering,
 *                  National Taiwan University
 *
 * =============================================================================
 */

#include    "System.h"

/*-----------------------------------------------------------------------------
 *  header files from std C/C++
 *-----------------------------------------------------------------------------*/
#include    <cstdlib>
#include	<ctime>
#include    <iostream>

/*-----------------------------------------------------------------------------
 *  header files from sutil
 *-----------------------------------------------------------------------------*/
#include    <GLUTDisplay.h>

/*-----------------------------------------------------------------------------
 *  header files of our own
 *-----------------------------------------------------------------------------*/
#include    "PathTracingRenderer.h"
#include    "PPMRenderer.h"
#include    "Renderer.h"
#include    "Scene.h"
#include    "SceneBuilder.h"

/*-----------------------------------------------------------------------------
 *  namespace
 *-----------------------------------------------------------------------------*/
using namespace std;
using namespace optix;
using namespace MaoPPM;



SceneBuilder * g_sceneBuilder = NULL;



System::System(int argc, char ** argv) :
    m_timeout(0.0f), m_scene(NULL), m_renderer(NULL)
{
    GLUTDisplay::init(argc, argv);

    if (!GLUTDisplay::isBenchmark())
        printUsageAndExit(argv[0], false);

    parseArguments(argc, argv);
}   /* -----  end of System::System  ----- */



System::~System()
{
    /* EMPTY */
}   /* -----  end of System::~System  ----- */



int System::exec()
{
    srand(static_cast<unsigned int>(time(NULL)));

    m_scene = new Scene;
    if (!m_renderer)
        m_renderer = new PPMRenderer(m_scene);
    else
        m_renderer->setScene(m_scene);

    try {
        /* :TODO:2011/3/28 18:00:54:: Try to understand what consequenses this line will cause. */
        GLUTDisplay::setUseSRGB(true);
        GLUTDisplay::setProgressiveDrawingTimeout(m_timeout);
        GLUTDisplay::run("MaoPPM", m_scene, GLUTDisplay::CDProgressive);
    } catch(const Exception & e) {
        sutilReportError(e.getErrorString().c_str());
        exit(EXIT_FAILURE);
    }
    delete m_scene;
    delete m_renderer;

    return EXIT_SUCCESS;
}   /* -----  end of method System::exec  ----- */



void System::parseArguments(int argc, char ** argv)
{
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        // -h or --help
        if (arg == "--help" || arg == "-h")
            printUsageAndExit(argv[0]);
        // -t or --timeout
        else if (arg == "--timeout" || arg == "-t") {
            if (++i < argc) {
                m_timeout = static_cast<float>(atof(argv[i]));
                cerr << "Timeout " << m_timeout << " seconds." << endl;
            } else {
                std::cerr << "Missing argument to " << arg << std::endl;
                printUsageAndExit(argv[0]);
            }
        }
        else if (arg == "--renderer" || arg == "-r") {
            if (++i < argc) {
                string rendererType(argv[i]);
                cerr << "Specified renderer: " << rendererType << "." << endl;
                if (rendererType == "PathTracer")
                    m_renderer = new PathTracingRenderer;
                else if (rendererType == "PPM")
                    m_renderer = new PPMRenderer;
                else {
                    cerr << "Unknown renderer: " << rendererType << endl;
                    printUsageAndExit(argv[0]);
                }
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
}   /* -----  end of function System::parseArguments  ----- */



void System::printUsageAndExit(const char * fileName, bool doExit)
{
    std::cerr
        << "Usage  : " << fileName << " [options]" << std::endl
        << "App options:" << std::endl
        << "  -h | --help             Print this usage message"                                                        << std::endl
        << "  -t | --timeout <sec>    Seconds before stopping rendering. Set to 0 for no stopping."                    << std::endl
        << "  -r | --renderer <type>  Specify renderer, available renderers are: PathTracer, PPM. PPM is the default." << std::endl
        << std::endl;

    GLUTDisplay::printUsage();

    if (doExit)
        exit(EXIT_FAILURE);
}   /* -----  end of function System::printUsageAndExit  ----- */
