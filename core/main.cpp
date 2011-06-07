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

/*----------------------------------------------------------------------------
 *  header files of our own
 *----------------------------------------------------------------------------*/
#include    "System.h"

/*----------------------------------------------------------------------------
 *  using namespaces
 *----------------------------------------------------------------------------*/
using namespace MaoPPM;



int main(int argc, char ** argv)
{
    System system(argc, argv);
    return system.exec();
}     /* ----------  end of function main  ---------- */
