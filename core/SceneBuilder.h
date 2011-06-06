/*
 * =====================================================================================
 *
 *       Filename:  SceneBuilder.h
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  2011-04-11 12:04:50
 *
 *         Author:  Chun-Wei Huang (LittleCVR), 
 *        Company:  Communication & Multimedia Laboratory,
 *                  Department of Computer Science & Information Engineering,
 *                  National Taiwan University
 *
 * =====================================================================================
 */


#ifndef SCENE_BUILDER_H
#define SCENE_BUILDER_H





/* #####   HEADER FILE INCLUDES   ################################################### */

/*-----------------------------------------------------------------------------
 *  header files from std
 *-----------------------------------------------------------------------------*/
#include    <map>
#include    <stack>

/*-----------------------------------------------------------------------------
 *  header files from OptiX
 *-----------------------------------------------------------------------------*/
#include    <optix_world.h>

/*-----------------------------------------------------------------------------
 *  header files of our own
 *-----------------------------------------------------------------------------*/
#include    "global.h"
#include    "ParameterVector.h"
#include    "Scene.h"






/* #####   MACROS  -  LOCAL TO THIS SOURCE FILE   ################################### */

namespace MaoPPM {

/*
 * =====================================================================================
 *        Class:  SceneBuilder
 *  Description:  
 * =====================================================================================
 */
class SceneBuilder
{
    public:
        struct State {
            optix::Matrix4x4    transform;
            optix::Material     material;
        };

    public:
        /* ====================  LIFECYCLE     ======================================= */
        SceneBuilder();
        ~SceneBuilder();

        /* ====================  ACCESSORS     ======================================= */

        /* ====================  MUTATORS      ======================================= */

        /* ====================  OPERATORS     ======================================= */
        void parse(Scene * scene, Scene::InitialCameraData * cameraData);

        void lookAt(float eyeX, float eyeY, float eyeZ,
                float atX, float atY, float atZ,
                float upX, float upY, float upZ);
        void translate(float dx, float dy, float dz);
        void rotate(float angle, float x, float y, float z);
        void scale(float sx, float sy, float sz);

        void coordinateSystemTransform(const char * name);

        void attributeBegin();
        void attributeEnd();
        void worldBegin();
        void worldEnd();

        void areaLightSource(const char * type, ParameterVector * parameterVector);
        void lightSource(const char * type, ParameterVector * parameterVector);

        void material(const char * type, ParameterVector * parameterVector);
        void shape(const char * type, ParameterVector * parameterVector);

    protected:
        /* ====================  DATA MEMBERS  ======================================= */

    private:    // functions
        ParameterVector * findByTypeAndName(const char * type, const char * name,
                const ParameterVector & parameterVector);

        /*
         *--------------------------------------------------------------------------------------
         *       Class:  SceneBuilder
         *      Method:  SceneBuilder :: deleteParameterVector
         * Description:  Delete the input ParameterVector and all its children.
         *--------------------------------------------------------------------------------------
         */
        void deleteParameterVector(ParameterVector * parameterVector);

        void dumpParameterVector(const ParameterVector & parameterVector);



    private:    // members
        /* ====================  DATA MEMBERS  ======================================= */
        Scene *                     m_scene;
        Scene::InitialCameraData *  m_cameraData;

        std::stack<State>   m_stateStack;
        State               m_currentState;

        std::map<std::string, optix::Matrix4x4> m_namedCoordinateSystemList;
};  /* -----  end of class SceneBuilder  ----- */

}   /* -----  end of namespace MaoPPm */





#endif  /* -----  not SCENE_BUILDER_H  ----- */
