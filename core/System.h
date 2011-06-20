/*
 * =============================================================================
 *
 *       Filename:  System.h
 *
 *    Description:  MaoPPM rendering system header file.
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

#ifndef SYSTEM_H
#define SYSTEM_H

/*----------------------------------------------------------------------------
 *  header files of our own
 *----------------------------------------------------------------------------*/
#include    "global.h"



namespace MaoPPM {
/*
 * =============================================================================
 *         Name:  System
 *  Description:  Manage the workflow of the rendering system, and provide some
 *                utility functions ready to be use globally.
 * =============================================================================
 */
class System {
    public:     // methods
        System(int argc, char ** argv);
        ~System();

    public:
        int     exec();

    private:    // methods
        /*
         * ---------------------------------------------------------------------
         *        Class:  System
         *       Method:  System :: parseArguments
         *  Description:  Parse the input arguments and set up the corresponding
         *                state of the program.
         * ---------------------------------------------------------------------
         */
        void    parseArguments(int argc, char ** argv);

        /*
         * ---------------------------------------------------------------------
         *        Class:  System
         *       Method:  System :: printUsageAndExit
         *  Description:  Print usage and exit the program. Should be called if
         *                an error has occured when parsing the input arguments.
         *                The fileName argument is the executable file name.
         * ---------------------------------------------------------------------
         */
        void    printUsageAndExit(const char * fileName, bool doExit = true);

    private:    // members
        float       m_timeout;  // The program will run for only timeout secs,
                                // set it to 0 to indicate the program should run
                                // forever.
        Scene *     m_scene;
        Renderer *  m_renderer;
};  /* -----  end of class System  ----- */
}   /* -----  end of namespace MaoPPM  ----- */

#endif  /* -----  #ifndef SYSTEM_H  ----- */
