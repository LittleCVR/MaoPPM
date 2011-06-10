/*
 * =====================================================================================
 *
 *       Filename:  SceneBuilder.cpp
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  2011-04-13 14:32:38
 *
 *         Author:  Chun-Wei Huang (LittleCVR), 
 *        Company:  Communication & Multimedia Laboratory,
 *                  Department of Computer Science & Information Engineering,
 *                  National Taiwan University
 *
 * =====================================================================================
 */

#include    "SceneBuilder.h"

/*-----------------------------------------------------------------------------
 *  header files from std C/C++
 *-----------------------------------------------------------------------------*/
#include    <cstdio>
#include    <iostream>

/*-----------------------------------------------------------------------------
 *  namespace
 *-----------------------------------------------------------------------------*/
using namespace std;
using namespace optix;
using namespace MaoPPM;



extern int pbrtparse();



SceneBuilder::SceneBuilder()
{
    m_currentState.transform = Matrix4x4::identity();
}



SceneBuilder::~SceneBuilder()
{
    /* EMPTY */
}



void SceneBuilder::parse(Scene * scene)
{
    m_scene = scene;
    extern SceneBuilder * g_sceneBuilder;
    g_sceneBuilder = this;

    extern FILE * pbrtin;
    pbrtin = stdin;
    pbrtparse();
}



void SceneBuilder::lookAt(float eyeX, float eyeY, float eyeZ,
        float atX, float atY, float atZ,
        float upX, float upY, float upZ)
{
    m_scene->m_initialCameraData.eye    = make_float3(eyeX, eyeY, eyeZ);
    m_scene->m_initialCameraData.lookat = make_float3( atX,  atY,  atZ);
    m_scene->m_initialCameraData.up     = make_float3( upX,  upY,  upZ);
    m_scene->m_initialCameraData.vfov   = 60.0;
}



void SceneBuilder::translate(float dx, float dy, float dz)
{
    float m[16];
    m[ 0] = 1.0f; m[ 1] = 0.0f; m[ 2] = 0.0f; m[ 3] = dx;
    m[ 4] = 0.0f; m[ 5] = 1.0f; m[ 6] = 0.0f; m[ 7] = dy;
    m[ 8] = 0.0f; m[ 9] = 0.0f; m[10] = 1.0f; m[11] = dz;
    m[12] = 0.0f; m[13] = 0.0f; m[14] = 0.0f; m[15] = 1.0f;
    m_currentState.transform = Matrix4x4(m) * m_currentState.transform;
}



void SceneBuilder::rotate(float angle, float x, float y, float z)
{
}



void SceneBuilder::scale(float sx, float sy, float sz)
{
    float m[16];
    m[ 0] =  sx ; m[ 1] = 0.0f; m[ 2] = 0.0f; m[ 3] = 0.0f;
    m[ 4] = 0.0f; m[ 5] =  sy ; m[ 6] = 0.0f; m[ 7] = 0.0f;
    m[ 8] = 0.0f; m[ 9] = 0.0f; m[10] =  sz ; m[11] = 0.0f;
    m[12] = 0.0f; m[13] = 0.0f; m[14] = 0.0f; m[15] = 1.0f;
    m_currentState.transform = Matrix4x4(m) * m_currentState.transform;
}



void SceneBuilder::coordinateSystemTransform(const char * name)
{
}



void SceneBuilder::attributeBegin()
{
    m_stateStack.push(m_currentState);
}



void SceneBuilder::attributeEnd()
{
    // check if the state stack is empty
    if (m_stateStack.empty())
        MaoPPM::fatal("Unexpected AttributeEnd.");

    // pop previous state
    m_currentState = m_stateStack.top();
    m_stateStack.pop();
}



void SceneBuilder::worldBegin()
{
}



void SceneBuilder::worldEnd()
{
}



void SceneBuilder::areaLightSource(const char * type, ParameterVector * parameterVector)
{
}



void SceneBuilder::lightSource(const char * type, ParameterVector * parameterVector)
{
    RTsize nLights;
    m_scene->m_lightList->getSize(nLights);
    m_scene->m_lightList->setSize(nLights + 1);

    Light light;
    if (strcmp(type, "point") == 0) {
        ParameterVector * colorVector = findByTypeAndName("color", "I", *parameterVector);
        if (colorVector == NULL) {
            cerr << "Light \"point\" must contains color I." << endl;
            exit(EXIT_FAILURE);
        }
        float * color = static_cast<float *>(colorVector->data);
        light.flux = make_float3(color[0], color[1], color[2]);
        float4 position = m_currentState.transform * make_float4(0.0f, 0.0f, 0.0f, 1.0f);
        light.position = make_float3(position.x / position.w, position.y / position.w, position.z / position.w);
    }

    Light * data = static_cast<Light *>(m_scene->m_lightList->map());
    data[nLights] = light;
    m_scene->m_lightList->unmap();

    // delete
    deleteParameterVector(parameterVector);
}



void SceneBuilder::material(const char * type, ParameterVector * parameterVector)
{
    optix::Material material = m_scene->getContext()->createMaterial();
    if (strcmp(type, "matte") == 0) {
        ParameterVector * colorVector = findByTypeAndName("color", "Kd", *parameterVector);
        if (colorVector == NULL) {
            cerr << "Material \"matte\" must contains color Kd." << endl;
            exit(EXIT_FAILURE);
        }
        string ptxPath = m_scene->ptxpath("MaoPPM", "matteMaterialPrograms.cu");
        material["Kd"]->setFloat(make_float3(
                    static_cast<float *>(colorVector->data)[0],
                    static_cast<float *>(colorVector->data)[1],
                    static_cast<float *>(colorVector->data)[2]));
        material->setClosestHitProgram(RadianceRay,
                m_scene->getContext()->createProgramFromPTXFile(ptxPath, "handleRadianceRayClosestHit"));
        material->setAnyHitProgram(ShadowRay,
                m_scene->getContext()->createProgramFromPTXFile(ptxPath, "handleShadowRayAnyHit"));
        material->setClosestHitProgram(PixelSamplingRay,
                m_scene->getContext()->createProgramFromPTXFile(ptxPath, "handlePixelSamplingRayClosestHit"));
        material->setClosestHitProgram(PhotonShootingRay,
                m_scene->getContext()->createProgramFromPTXFile(ptxPath, "handlePhotonShootingRayClosestHit"));
        material->setAnyHitProgram(GatheringRay,
                m_scene->getContext()->createProgramFromPTXFile(ptxPath, "handleGatheringRayAnyHit"));
    }
    else if (strcmp(type, "plastic") == 0) {
        ParameterVector * Kd = findByTypeAndName("color", "Kd", *parameterVector);
        ParameterVector * Ks = findByTypeAndName("color", "Ks", *parameterVector);
        ParameterVector * roughness = findByTypeAndName("float", "roughness", *parameterVector);

        string ptxPath = m_scene->ptxpath("MaoPPM", "plasticMaterialPrograms.cu");
        material["Kd"]->setFloat(make_float3(
                    static_cast<float *>(Kd->data)[0],
                    static_cast<float *>(Kd->data)[1],
                    static_cast<float *>(Kd->data)[2]));
        material["Ks"]->setFloat(make_float3(
                    static_cast<float *>(Ks->data)[0],
                    static_cast<float *>(Ks->data)[1],
                    static_cast<float *>(Ks->data)[2]));
        material["exponent"]->setFloat(1.0f / static_cast<float *>(roughness->data)[0]);
        material->setClosestHitProgram(RadianceRay,
                m_scene->getContext()->createProgramFromPTXFile(ptxPath, "handleRadianceRayClosestHit"));
        material->setAnyHitProgram(ShadowRay,
                m_scene->getContext()->createProgramFromPTXFile(ptxPath, "handleShadowRayAnyHit"));
        material->setClosestHitProgram(PixelSamplingRay,
                m_scene->getContext()->createProgramFromPTXFile(ptxPath, "handlePixelSamplingRayClosestHit"));
        material->setClosestHitProgram(PhotonShootingRay,
                m_scene->getContext()->createProgramFromPTXFile(ptxPath, "handlePhotonShootingRayClosestHit"));
        material->setAnyHitProgram(GatheringRay,
                m_scene->getContext()->createProgramFromPTXFile(ptxPath, "handleGatheringRayAnyHit"));
    }
    m_currentState.material = material;

    // delete
    deleteParameterVector(parameterVector);
}



void SceneBuilder::shape(const char * type, ParameterVector * parameterVector)
{
    Geometry geometry = m_scene->getContext()->createGeometry();
    if (strcmp(type, "trianglemesh") == 0) {
        ParameterVector * pointList = findByTypeAndName("point", "P", *parameterVector);
        ParameterVector * indexList = findByTypeAndName("integer", "indices", *parameterVector);
        if (pointList == NULL) {
            cerr << "Shape \"trianglemesh\" must contains point P." << endl;
            exit(EXIT_FAILURE);
        }
        if (indexList == NULL) {
            cerr << "Shape \"trianglemesh\" must contains integer indices." << endl;
            exit(EXIT_FAILURE);
        }

        geometry->setPrimitiveCount(indexList->nElements / 3);

        std::string ptxPath = m_scene->ptxpath("MaoPPM", "triangleMeshShapePrograms.cu");
        Program boundingBoxProgram = m_scene->getContext()->createProgramFromPTXFile(ptxPath, "boundingBox");
        geometry->setBoundingBoxProgram(boundingBoxProgram);
        Program intersectProgram = m_scene->getContext()->createProgramFromPTXFile(ptxPath, "intersect");
        geometry->setIntersectionProgram(intersectProgram);

        Buffer vertexList = m_scene->getContext()->createBuffer(RT_BUFFER_INPUT);
        vertexList->setFormat(RT_FORMAT_FLOAT3);
        vertexList->setSize(pointList->nElements / 3);
        {
            float3 * dst = static_cast<float3 *>(vertexList->map());
            float  * src = static_cast<float  *>(pointList->data);
            for (int i = 0; i < pointList->nElements / 3; ++i) {
                dst[i].x = src[3*i+0];
                dst[i].y = src[3*i+1];
                dst[i].z = src[3*i+2];
            }
            vertexList->unmap();
        }
        geometry["vertexList"]->set(vertexList);

        Buffer vertexIndexList = m_scene->getContext()->createBuffer(RT_BUFFER_INPUT);
        vertexIndexList->setFormat(RT_FORMAT_UNSIGNED_INT3);
        vertexIndexList->setSize(indexList->nElements / 3);
        {
            uint3 * dst = static_cast<uint3 *>(vertexIndexList->map());
            float * src = static_cast<float *>(indexList->data);
            for (int i = 0; i < indexList->nElements / 3; ++i) {
                dst[i].x = static_cast<uint>(src[3*i+0]);
                dst[i].y = static_cast<uint>(src[3*i+1]);
                dst[i].z = static_cast<uint>(src[3*i+2]);
            }
            vertexIndexList->unmap();
        }
        geometry["vertexIndexList"]->set(vertexIndexList);

        GeometryInstance geometryInstance = m_scene->getContext()->createGeometryInstance();
        geometryInstance->setGeometry(geometry);
        geometryInstance->setMaterialCount(1);
        geometryInstance->setMaterial(0, m_currentState.material);

        m_scene->m_rootObject->setChildCount(m_scene->m_rootObject->getChildCount() + 1);
        m_scene->m_rootObject->setChild(m_scene->m_rootObject->getChildCount() - 1, geometryInstance);
    }

    // delete
    deleteParameterVector(parameterVector);
}



ParameterVector * SceneBuilder::findByTypeAndName(const char * type, const char * name,
        const ParameterVector & parameterVector)
{
    ParameterVector * result = NULL;
    for (int i = 0; i < parameterVector.nElements; i++) {
        size_t length = strlen(parameterVector.name[i]);
        char * t = new char [length];
        char * n = new char [length];

        sscanf(parameterVector.name[i], "%s%s", t, n);
        if (strcmp(t, type) == 0 && strcmp(n, name) == 0) {
            result = static_cast<ParameterVector **>(parameterVector.data)[i];
            break;
        }

        delete [] t;
        delete [] n;
    }
    return result;
}



void SceneBuilder::deleteParameterVector(ParameterVector * parameterVector)
{
    if (parameterVector->type == ParameterVector::Parameter) {
        ParameterVector ** data = static_cast<ParameterVector **>(parameterVector->data);
        for (int i = 0; i < parameterVector->nElements; i++)
            deleteParameterVector(data[i]);
    }
    delete parameterVector;
}



void SceneBuilder::dumpParameterVector(const ParameterVector & parameterVector)
{
    for (int i = 0; i < parameterVector.nElements; i++) {
        const ParameterVector * pv = static_cast<ParameterVector **>(parameterVector.data)[i];
        cout << parameterVector.name[i] << endl;
        cout << "    ";
        for (int j = 0; j < pv->nElements; j++) {
            if (pv->type == ParameterVector::RealNumber)
                cout << static_cast<float *>(pv->data)[j] << "\t";
        }
        cout << endl;
    }
}
