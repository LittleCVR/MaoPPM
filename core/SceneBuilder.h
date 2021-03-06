/*
 * =============================================================================
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
 * =============================================================================
 */

#ifndef SCENE_BUILDER_H
#define SCENE_BUILDER_H

/*----------------------------------------------------------------------------
 *  header files from std
 *----------------------------------------------------------------------------*/
#include    <map>
#include    <stack>

/*----------------------------------------------------------------------------
 *  header files from OptiX
 *----------------------------------------------------------------------------*/
#include    <optix_world.h>

/*----------------------------------------------------------------------------
 *  header files of our own
 *----------------------------------------------------------------------------*/
#include    "global.h"
#include    "utility.h"
#include    "ParameterVector.h"



namespace MaoPPM {
/*
 * =============================================================================
 *        Class:  SceneBuilder
 *  Description:  
 * =============================================================================
 */
class SceneBuilder
{
    public:
        struct State {
            optix::Matrix4x4    transform;
            optix::Material     material;
        };

    public:
        SceneBuilder();
        ~SceneBuilder();

        void parse(Scene * scene);

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

    private:    // functions
        void setMaterialPrograms(optix::Material material, const char * cuFileName);

        float findOneFloat(const char * name, const ParameterVector & parameterVector,
                const float defaultValue);
        optix::float3 findOneColor(const char * name, const ParameterVector & parameterVector,
                const optix::float3 defaultValue);
        // Why does this function return float* not int*?
        // Because ParameterVector does not store integer, it stores only float.
        float * findIntegerList(const char * name, const ParameterVector & parameterVector,
                unsigned int * nFound);
        float * findPointList(const char * name, const ParameterVector & parameterVector,
                unsigned int * nFound);
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



    private:
        Scene *             m_scene;

        std::stack<State>   m_stateStack;
        State               m_currentState;

        std::map<std::string, optix::Matrix4x4> m_namedCoordinateSystemList;
};  /* -----  end of class SceneBuilder  ----- */
}   /* -----  end of namespace MaoPPm */

#endif  /* -----  not SCENE_BUILDER_H  ----- */
