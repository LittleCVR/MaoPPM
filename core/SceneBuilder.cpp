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

/*----------------------------------------------------------------------------
 *  header files of our own
 *----------------------------------------------------------------------------*/
#include    "payload.h"
#include    "utility.h"
#include    "Light.h"
#include    "Glass.h"
#include    "Matte.h"
#include    "Mirror.h"
#include    "Plastic.h"
#include    "Renderer.h"

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
        light.intensity = findOneColor("I", *parameterVector, make_float3(1.0f));
        light.position = transformPoint(m_currentState.transform, make_float3(0.0f));
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
        float3 kd = findOneColor("Kd", *parameterVector, make_float3(0.25f));
        // material
        Matte matte(kd);
        Index index = m_scene->copyToHeap(&matte, sizeof(matte));
        material["materialIndex"]->setUserData(sizeof(index), &index);
        // program
        setMaterialPrograms(material, "Matte.cu");
    }
    else if (strcmp(type, "plastic") == 0) {
        float3 kd = findOneColor("Kd", *parameterVector, make_float3(0.25f));
        float3 ks = findOneColor("Ks", *parameterVector, make_float3(0.25f));
        float  roughness = findOneFloat("roughness", *parameterVector, 0.1f);
        // material
        Plastic plastic(kd, ks, roughness);
        Index index = m_scene->copyToHeap(&plastic, sizeof(plastic));
        material["materialIndex"]->setUserData(sizeof(index), &index);
        // program
        setMaterialPrograms(material, "Plastic.cu");
    }
    else if (strcmp(type, "mirror") == 0) {
        float3 kr = findOneColor("Kr", *parameterVector, make_float3(0.9f));
        // material
        Mirror mirror(kr);
        Index index = m_scene->copyToHeap(&mirror, sizeof(mirror));
        material["materialIndex"]->setUserData(sizeof(index), &index);
        // program
        setMaterialPrograms(material, "Mirror.cu");
    }
    else if (strcmp(type, "glass") == 0) {
        float3 kr = findOneColor("Kr", *parameterVector, make_float3(1.0f));
        float3 kt = findOneColor("Kt", *parameterVector, make_float3(1.0f));
        float  in = findOneFloat("index", *parameterVector, 1.5f);
        // material
        Glass glass(kr, kt, in);
        Index index = m_scene->copyToHeap(&glass, sizeof(glass));
        material["materialIndex"]->setUserData(sizeof(index), &index);
        // program
        setMaterialPrograms(material, "Glass.cu");
    }
    m_currentState.material = material;

    // delete
    deleteParameterVector(parameterVector);
}



void SceneBuilder::shape(const char * type, ParameterVector * parameterVector)
{
    Geometry geometry = m_scene->getContext()->createGeometry();
    if (strcmp(type, "trianglemesh") == 0) {
        unsigned int nVertices;
        unsigned int nIndices;
        float * vertexList = findPointList("P", *parameterVector, &nVertices);
        float * indexList  = findIntegerList("indices", *parameterVector, &nIndices);
        unsigned int nTriangles = nIndices / 3;
        if (vertexList == NULL) {
            cerr << "Shape \"trianglemesh\" must contain point P." << endl;
            exit(EXIT_FAILURE);
        }
        if (indexList == NULL) {
            cerr << "Shape \"trianglemesh\" must contain integer indices." << endl;
            exit(EXIT_FAILURE);
        }

        // Geometry.
        geometry->setPrimitiveCount(nTriangles);
        // Vertices.
        Buffer vertexBuffer = m_scene->getContext()->createBuffer(RT_BUFFER_INPUT);
        vertexBuffer->setFormat(RT_FORMAT_FLOAT3);
        vertexBuffer->setSize(nVertices / 3);
        {
            float3 * dst = static_cast<float3 *>(vertexBuffer->map());
            for (int i = 0; i < nVertices / 3; ++i) {
                dst[i] = transformPoint(m_currentState.transform,
                        make_float3(vertexList[3*i+0], vertexList[3*i+1], vertexList[3*i+2]));
            }
            vertexBuffer->unmap();
        }
        geometry["vertexList"]->set(vertexBuffer);
        // Indices.
        Buffer indexBuffer = m_scene->getContext()->createBuffer(RT_BUFFER_INPUT);
        indexBuffer->setFormat(RT_FORMAT_UNSIGNED_INT3);
        indexBuffer->setSize(nIndices / 3);
        {
            uint3 * dst = static_cast<uint3 *>(indexBuffer->map());
            for (int i = 0; i < nIndices / 3; ++i) {
                dst[i].x = static_cast<uint>(indexList[3*i+0]);
                dst[i].y = static_cast<uint>(indexList[3*i+1]);
                dst[i].z = static_cast<uint>(indexList[3*i+2]);
            }
            indexBuffer->unmap();
        }
        geometry["vertexIndexList"]->set(indexBuffer);

        // Programs
        std::string ptxPath = m_scene->ptxpath("MaoPPM", "TriangleMesh.cu");
        Program boundingBoxProgram = m_scene->getContext()->createProgramFromPTXFile(ptxPath, "boundingBox");
        geometry->setBoundingBoxProgram(boundingBoxProgram);
        Program intersectProgram = m_scene->getContext()->createProgramFromPTXFile(ptxPath, "intersect");
        geometry->setIntersectionProgram(intersectProgram);
    }
    else if (strcmp(type, "sphere") == 0) {
        float  radius = findOneFloat("radius", *parameterVector, 1.0f);
        float  phiMax = findOneFloat("phimax", *parameterVector, 360.0f);
        float  zMin   = findOneFloat("zmin",   *parameterVector, -radius);
        float  zMax   = findOneFloat("zmax",   *parameterVector,  radius);
        phiMax = phiMax / 180.0 * M_PIf;

        geometry->setPrimitiveCount(1);
        geometry["radius"]->setFloat(radius);
        geometry["phiMax"]->setFloat(phiMax);
        geometry["zMin"]->setFloat(zMin);
        geometry["zMax"]->setFloat(zMax);

        // Programs
        std::string ptxPath = m_scene->ptxpath("MaoPPM", "Sphere.cu");
        Program boundingBoxProgram = m_scene->getContext()->createProgramFromPTXFile(ptxPath, "boundingBox");
        geometry->setBoundingBoxProgram(boundingBoxProgram);
        Program intersectProgram = m_scene->getContext()->createProgramFromPTXFile(ptxPath, "intersect");
        geometry->setIntersectionProgram(intersectProgram);
    }

    GeometryInstance geometryInstance = m_scene->context()->createGeometryInstance();
    geometryInstance->setGeometry(geometry);
    geometryInstance->setMaterialCount(1);
    geometryInstance->setMaterial(0, m_currentState.material);

    GeometryGroup geometryGroup = m_scene->context()->createGeometryGroup();
    geometryGroup->setChildCount(1);
    geometryGroup->setChild(0, geometryInstance);
    // Acceleration.
    Acceleration acceleration = m_scene->context()->createAcceleration("Bvh", "Bvh");
    geometryGroup->setAcceleration(acceleration);

    // Why don't we use optix::Transform here?
    // Because it would cause OptiX to crash, OptiX sucks.
    // We will only use it to compute the inverse.
    float matrix[16], inverse[16];
    Transform transform = m_scene->context()->createTransform();
    transform->setMatrix(false, m_currentState.transform.getData(), NULL);
    transform->getMatrix(false, matrix, inverse);
    geometry["worldToObject"]->setMatrix4x4fv(false, inverse);
    geometry["objectToWorld"]->setMatrix4x4fv(false, matrix);

    m_scene->m_rootObject->setChildCount(m_scene->m_rootObject->getChildCount() + 1);
    m_scene->m_rootObject->setChild(m_scene->m_rootObject->getChildCount() - 1, geometryGroup);

    // delete
    deleteParameterVector(parameterVector);
}



void SceneBuilder::setMaterialPrograms(optix::Material material, const char * cuFileName)
{
    std::string ptxPath = m_scene->ptxpath("MaoPPM", cuFileName);
    Program program = m_scene->getContext()->createProgramFromPTXFile(ptxPath, "handleNormalRayClosestHit");
    material->setClosestHitProgram(NormalRay, program);
    ptxPath = m_scene->ptxpath("MaoPPM", "ray.cu");
    Program program2 = m_scene->getContext()->createProgramFromPTXFile(ptxPath, "handleShadowRayAnyHit");
    material->setAnyHitProgram(ShadowRay, program2);
}



float SceneBuilder::findOneFloat(const char * name,
        const ParameterVector & parameterVector,
        const float defaultValue)
{
    ParameterVector * v = findByTypeAndName("float", name, parameterVector);

    if (v == NULL)
        return defaultValue;
    else {
        float * p = static_cast<float *>(v->data);
        return p[0];
    }
}



optix::float3 SceneBuilder::findOneColor(const char * name,
        const ParameterVector & parameterVector,
        const optix::float3 defaultValue)
{
    ParameterVector * v = findByTypeAndName("color", name, parameterVector);
    if (v == NULL)
        v = findByTypeAndName("spectrum", name, parameterVector);

    if (v == NULL)
        return defaultValue;
    else {
        float * p = static_cast<float *>(v->data);
        return make_float3(p[0], p[1], p[2]);
    }
}



float * SceneBuilder::findIntegerList(const char * name,
        const ParameterVector & parameterVector,
        unsigned int * nFound)
{
    ParameterVector * v = findByTypeAndName("integer", name, parameterVector);
    if (v == NULL) {
        *nFound = 0;
        return NULL;
    } else {
        *nFound = v->nElements;
        return static_cast<float *>(v->data);
    }
}



float * SceneBuilder::findPointList(const char * name,
        const ParameterVector & parameterVector,
        unsigned int * nFound)
{
    ParameterVector * v = findByTypeAndName("point", name, parameterVector);
    if (v == NULL) {
        *nFound = 0;
        return NULL;
    } else {
        *nFound = v->nElements;
        return static_cast<float *>(v->data);
    }
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
