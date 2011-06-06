%{

#include    <cstdio>
#include    <cstdlib>
#include    <cstring>
#include    "ParameterVector.h"
#include    "SceneBuilder.h"

/*-----------------------------------------------------------------------------
 *  must have this in order to cooperate with flex
 *-----------------------------------------------------------------------------*/
extern int yylex();
extern void yyinPush(const char * fileName);

/*-----------------------------------------------------------------------------
 *  bison needs this, a simple error handling function
 *-----------------------------------------------------------------------------*/
void yyerror(const char * message);

/*-----------------------------------------------------------------------------
 *  my scene builder
 *-----------------------------------------------------------------------------*/
extern MaoPPM::SceneBuilder * g_sceneBuilder;

%}



/*-----------------------------------------------------------------------------
 *  YYSTYPE
 *-----------------------------------------------------------------------------*/

%union {
    /* primitive types */
    float   realNumber;
    /* 1024: hard coded string length limit */
    char    string[1024];
    /* array and parameter */
    struct ParameterVector * parameterVector;
}



/*-----------------------------------------------------------------------------
 *  tokens & types
 *-----------------------------------------------------------------------------*/
%token  INCLUDE
%token  LOOK_AT
%token  ACCELERATOR
%token  WORLD_BEGIN WORLD_END
%token  ATTRIBUTE_BEGIN ATTRIBUTE_END 
%token  TRANSFORM TRANSLATE ROTATE SCALE COORDINATE_SYSTEM_TRANSFORM
%token  AREA_LIGHT_SOURCE LIGHT_SOURCE
%token  SHAPE MATERIAL
%token  LEFT_BRACKET RIGHT_BRACKET

%token  <realNumber>    REAL_NUMBER
%token  <string>        STRING

%type   <parameterVector>   array
%type   <parameterVector>   string_array
%type   <parameterVector>   string_list
%type   <parameterVector>   single_element_string_array
%type   <parameterVector>   real_number_array
%type   <parameterVector>   real_number_list
%type   <parameterVector>   single_element_real_number_array
%type   <parameterVector>   parameter_list




%%





start   : statement_list
        ;



/*-----------------------------------------------------------------------------
 *  array
 *-----------------------------------------------------------------------------*/

array   : real_number_array
        {
            $$ = $1;
        }
        | string_array
        {
            $$ = $1;
        }
        ;



string_array
        : LEFT_BRACKET string_list RIGHT_BRACKET
        {
            $$ = $2;
        }
        | single_element_string_array
        {
            $$ = $1;
        }
        ;



single_element_string_array
        : STRING
        {
            $$ = new ParameterVector($1);
        }
        ;



string_list
        : string_list STRING
        {
            $1->append($2);
            $$ = $1;
        }
        | STRING
        {
            $$ = new ParameterVector($1);
        }
        ;



real_number_array
        : LEFT_BRACKET real_number_list RIGHT_BRACKET
        {
            $$ = $2;
        }
        | single_element_real_number_array
        {
            $$ = $1;
        }
        ;



single_element_real_number_array
        : REAL_NUMBER
        {
            $$ = new ParameterVector($1);
        }
        ;



real_number_list
        : real_number_list REAL_NUMBER
        {
            $1->append($2);
            $$ = $1;
        }
        | REAL_NUMBER
        {
            $$ = new ParameterVector($1);
        }
        ;



/*-----------------------------------------------------------------------------
 *  parameter list
 *-----------------------------------------------------------------------------*/

parameter_list
        : /* EMPTY */
        {
            $$ = new ParameterVector;
            $$->type = ParameterVector::Parameter;
            $$->elementSize = sizeof(ParameterVector *);
        }
        | parameter_list STRING array
        {
            $1->append($2, $3);
            $$ = $1;
        }
        ;



/*-----------------------------------------------------------------------------
 *  statement
 *-----------------------------------------------------------------------------*/

statement_list
        : statement_list statement
        | statement
        ;



statement
        /*: ACCELERATOR STRING parameter_list
        {
            ParamSet params;
            InitParamSet(params, SPECTRUM_REFLECTANCE);
            pbrtAccelerator($2, params);
            FreeArgs();
        }*/
        /*| ACTIVETRANSFORM ENDTIME
        {
            pbrtActiveTransformEndTime();
        }*/
        /*| ACTIVETRANSFORM STARTTIME
        {
            pbrtActiveTransformStartTime();
        }*/
        : AREA_LIGHT_SOURCE STRING parameter_list
        {
            g_sceneBuilder->areaLightSource($2, $3);
        }
        | ATTRIBUTE_BEGIN
        {
            g_sceneBuilder->attributeBegin();
        }
        | ATTRIBUTE_END
        {
            g_sceneBuilder->attributeEnd();
        }
        /*| CAMERA STRING parameter_list
        {
            ParamSet params;
            InitParamSet(params, SPECTRUM_REFLECTANCE);
            pbrtCamera($2, params);
            FreeArgs();
        }*/
        /*| CONCATTRANSFORM num_array
        {
            if (VerifyArrayLength($2, 16, "ConcatTransform"))
                pbrtConcatTransform((float *) $2->array);
            ArrayFree($2);
        }*/
        /*| COORDINATESYSTEM STRING
        {
            pbrtCoordinateSystem($2);
        }*/
        | COORDINATE_SYSTEM_TRANSFORM STRING
        {
            g_sceneBuilder->coordinateSystemTransform($2);
        }
        /*| FILM STRING parameter_list
        {
            ParamSet params;
            InitParamSet(params, SPECTRUM_REFLECTANCE);
            pbrtFilm($2, params);
            FreeArgs();
        }*/
        /*| IDENTITY
        {
            pbrtIdentity();
        }*/
        | INCLUDE STRING
        {
            yyinPush($2);
        }
        | LIGHT_SOURCE STRING parameter_list
        {
            g_sceneBuilder->lightSource($2, $3);
        }
        | LOOK_AT REAL_NUMBER REAL_NUMBER REAL_NUMBER REAL_NUMBER REAL_NUMBER REAL_NUMBER REAL_NUMBER REAL_NUMBER REAL_NUMBER
        {
            g_sceneBuilder->lookAt($2, $3, $4, $5, $6, $7, $8, $9, $10);
        }
        /*| MAKENAMEDMATERIAL STRING parameter_list
        {
            ParamSet params;
            InitParamSet(params, SPECTRUM_REFLECTANCE);
            pbrtMakeNamedMaterial($2, params);
            FreeArgs();
        }*/
        | MATERIAL STRING parameter_list
        {
            g_sceneBuilder->material($2, $3);
        }
        /*| NAMEDMATERIAL STRING
        {
            pbrtNamedMaterial($2);
        }*/
        /*| OBJECTBEGIN STRING
        {
            pbrtObjectBegin($2);
        }*/
        /*| OBJECTEND
        {
            pbrtObjectEnd();
        }*/
        /*| OBJECTINSTANCE STRING
        {
            pbrtObjectInstance($2);
        }*/
        /*| PIXELFILTER STRING parameter_list
        {
            ParamSet params;
            InitParamSet(params, SPECTRUM_REFLECTANCE);
            pbrtPixelFilter($2, params);
            FreeArgs();
        }*/
        /*| RENDERER STRING parameter_list
        {
            ParamSet params;
            InitParamSet(params, SPECTRUM_REFLECTANCE);
            pbrtRenderer($2, params);
            FreeArgs();
        }*/
        /*| REVERSEORIENTATION
        {
            pbrtReverseOrientation();
        }*/
        | ROTATE REAL_NUMBER REAL_NUMBER REAL_NUMBER REAL_NUMBER
        {
            g_sceneBuilder->rotate($2, $3, $4, $5);
        }
        /*| SAMPLER STRING parameter_list
        {
            ParamSet params;
            InitParamSet(params, SPECTRUM_REFLECTANCE);
            pbrtSampler($2, params);
            FreeArgs();
        }*/
        | SCALE REAL_NUMBER REAL_NUMBER REAL_NUMBER
        {
            g_sceneBuilder->scale($2, $3, $4);
        }
        | SHAPE STRING parameter_list
        {
            g_sceneBuilder->shape($2, $3);
        }
        /*| SURFACEINTEGRATOR STRING parameter_list
        {
            ParamSet params;
            InitParamSet(params, SPECTRUM_REFLECTANCE);
            pbrtSurfaceIntegrator($2, params);
            FreeArgs();
        }*/
        /*| TEXTURE STRING STRING STRING parameter_list
        {
            ParamSet params;
            InitParamSet(params, SPECTRUM_REFLECTANCE);
            pbrtTexture($2, $3, $4, params);
            FreeArgs();
        }*/
        /*| TRANSFORMBEGIN
        {
            pbrtTransformBegin();
        }*/
        /*| TRANSFORMEND
        {
            pbrtTransformEnd();
        }*/
        /*| TRANSFORMTIMES NUM NUM
        {
            pbrtTransformTimes($2, $3);
        }*/
        /*| TRANSFORM num_array
        {
            if (VerifyArrayLength( $2, 16, "Transform" ))
                pbrtTransform( (float *) $2->array );
            ArrayFree($2);
        }*/
        | TRANSLATE REAL_NUMBER REAL_NUMBER REAL_NUMBER
        {
            g_sceneBuilder->translate($2, $3, $4);
        }
        /*| VOLUMEINTEGRATOR STRING parameter_list
        {
            ParamSet params;
            InitParamSet(params, SPECTRUM_REFLECTANCE);
            pbrtVolumeIntegrator($2, params);
            FreeArgs();
        }*/
        /*| VOLUME STRING parameter_list
        {
            ParamSet params;
            InitParamSet(params, SPECTRUM_REFLECTANCE);
            pbrtVolume($2, params);
            FreeArgs();
        }*/
        | WORLD_BEGIN
        {
            g_sceneBuilder->worldBegin();
        }
        | WORLD_END
        {
            g_sceneBuilder->worldEnd();
        }
        ;





%%





void yyerror(const char * message)
{
    /* :TODO:2011-04-28 20:53:55:: should implement a fatal function for this */
    fprintf(stderr, "fatal: parsing error: %s\n", message);
    exit(EXIT_FAILURE);
}
