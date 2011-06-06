%{

/*-----------------------------------------------------------------------------
 *  from std
 *-----------------------------------------------------------------------------*/
#include    <cstdlib>
#include    <string>
#include    <vector>

/*-----------------------------------------------------------------------------
 *  from bison
 *-----------------------------------------------------------------------------*/
#include    "global.h"
#include    "parser.hh"

/*-----------------------------------------------------------------------------
 *  Code to handle the string input.
 *-----------------------------------------------------------------------------*/
static int stringPosition;
static void append(char c);

/*-----------------------------------------------------------------------------
 *  The input file stack.
 *-----------------------------------------------------------------------------*/
struct YyinInfo {
    std::string     fileName;
    YY_BUFFER_STATE bufferState;
    int             lineNumber;
};
std::string yyinCurrentFileName;
std::vector<YyinInfo> yyinStack;
void yyinPush(const char * fileName);
void yyinPop();

%}



/*-----------------------------------------------------------------------------
 *  In order to let this scanner work under Windows, we have to tell flex that
 *  do not include unistd.h and do not use isatty() function.
 *-----------------------------------------------------------------------------*/
%option nounistd
%option never-interactive

WHITESPACE      [ \t\r]
REAL_NUMBER     [-+]?([0-9]+|(([0-9]+\.[0-9]*)|(\.[0-9]+)))([eE][-+]?[0-9]+)?
IDENTITY        [a-zA-Z_][a-zA-Z_0-9]*

%x STRING_CONDITION COMMENT_CONDITION





%%





{WHITESPACE}+           { /* ignore */ }
"\n"                    { yylineno++; }

"#"                     { BEGIN COMMENT_CONDITION; }
<COMMENT_CONDITION>.    { /* ignore */ }
<COMMENT_CONDITION>\n   { yylineno++; BEGIN INITIAL; }

"Accelerator"           { return ACCELERATOR; }
%{ /*ActiveTransform         { return ACTIVETRANSFORM; }*/ %}
%{ /*All                     { return ALL; }*/ %}
"AreaLightSource"       { return AREA_LIGHT_SOURCE; }
"AttributeBegin"        { return ATTRIBUTE_BEGIN; }
"AttributeEnd"          { return ATTRIBUTE_END; }
%{ /*Camera                  { return CAMERA; }*/ %}
%{ /*ConcatTransform         { return CONCATTRANSFORM; }*/ %}
%{ /*CoordinateSystem        { return COORDINATESYSTEM; }*/ %}
"CoordSysTransform"     { return COORDINATE_SYSTEM_TRANSFORM; }
%{ /*EndTime                 { return ENDTIME; }*/ %}
%{ /*Film                    { return FILM; }*/ %}
%{ /*Identity                { return IDENTITY; }*/ %}
"Include"               { return INCLUDE; }
"LightSource"           { return LIGHT_SOURCE; }
"LookAt"                { return LOOK_AT; }
%{ /*MakeNamedMaterial       { return MAKENAMEDMATERIAL; }*/ %}
"Material"              { return MATERIAL; }
%{ /*NamedMaterial           { return NAMEDMATERIAL; }*/ %}
%{ /*ObjectBegin             { return OBJECTBEGIN; }*/ %}
%{ /*ObjectEnd               { return OBJECTEND; }*/ %}
%{ /*ObjectInstance          { return OBJECTINSTANCE; }*/ %}
%{ /*PixelFilter             { return PIXELFILTER; }*/ %}
%{ /*Renderer                { return RENDERER; }*/ %}
%{ /*ReverseOrientation      { return REVERSEORIENTATION; }*/ %}
"Rotate"                { return ROTATE; }
%{ /*Sampler                 { return SAMPLER; }*/ %}
"Scale"                 { return SCALE; }
"Shape"                 { return SHAPE; }
%{ /*StartTime               { return STARTTIME; }*/ %}
%{ /*SurfaceIntegrator       { return SURFACEINTEGRATOR; }*/ %}
%{ /*Texture                 { return TEXTURE; }*/ %}
%{ /*TransformBegin          { return TRANSFORMBEGIN; }*/ %}
%{ /*TransformEnd            { return TRANSFORMEND; }*/ %}
%{ /*TransformTimes          { return TRANSFORMTIMES; }*/ %}
"Transform"             { return TRANSFORM; }
"Translate"             { return TRANSLATE; }
%{ /*Volume                  { return VOLUME; }*/ %}
%{ /*VolumeIntegrator        { return VOLUMEINTEGRATOR; }*/ %}
"WorldBegin"            { return WORLD_BEGIN; }
"WorldEnd"              { return WORLD_END; }

"["                     { return LEFT_BRACKET; }
"]"                     { return RIGHT_BRACKET; }

{REAL_NUMBER}   {
    yylval.realNumber = static_cast<float>(atof(yytext));
    return REAL_NUMBER;
}

"\""                    { BEGIN STRING_CONDITION; stringPosition = 0; }
<STRING_CONDITION>\\n   { append('\n'); }
<STRING_CONDITION>\\t   { append('\t'); }
<STRING_CONDITION>\\r   { append('\r'); }
<STRING_CONDITION>\\b   { append('\b'); }
<STRING_CONDITION>\\f   { append('\f'); }
<STRING_CONDITION>\\\"  { append('\"'); }
<STRING_CONDITION>\\\\  { append('\\'); }
<STRING_CONDITION>\\[0-9]{3}    {
    int value = atoi(yytext+1);
    while (value > 256)
        value -= 256;
    append(static_cast<char>(value));
}
<STRING_CONDITION>\\\n  { yylineno++; }
<STRING_CONDITION>\\.   { append(yytext[1]); }
<STRING_CONDITION>\"    { BEGIN INITIAL; return STRING; }
<STRING_CONDITION>.     { append(yytext[0]); }
<STRING_CONDITION>\n    {
    MaoPPM::fatal("Unterminated string in file %s, line %d.\n",
            yyinCurrentFileName.c_str(), yylineno);
}

.   {
    MaoPPM::fatal("Illegal character: %c (0x%x)",
            yytext[0], int(yytext[0]));
}





%%





static void append(char c)
{
    /* :TODO:2011-04-30 20:53:30:: hard coding 1024 chars for now */
    if (stringPosition >= 1024)
        MaoPPM::fatal("The scene file cannot contains string longer than 1024 characters!");
    yylval.string[stringPosition] = c;
    ++stringPosition;
    yylval.string[stringPosition] = '\0';
}



void yyinPush(const char * fileName)
{
    // save current buffer
    YyinInfo info;
    info.fileName = yyinCurrentFileName;
    info.bufferState = YY_CURRENT_BUFFER;
    info.lineNumber = yylineno;
    yyinStack.push_back(info);

    // create new buffer
    yyinCurrentFileName = std::string(fileName);
    yylineno = 1;
    yyin = fopen(yyinCurrentFileName.c_str(), "r");
    if (!yyin)
        MaoPPM::fatal("Unable to open included scene file \"%s\"", yyinCurrentFileName.c_str());
    yy_switch_to_buffer(yy_create_buffer(yyin, YY_BUF_SIZE));
}



void yyinPop()
{
    // clear current buffer
    fclose(yyin);
    yy_delete_buffer(YY_CURRENT_BUFFER);

    // switch to previous buffer 
    YyinInfo info = yyinStack.back();
    yyinCurrentFileName = info.fileName;
    yylineno = info.lineNumber;
    yy_switch_to_buffer(info.bufferState);
    yyinStack.pop_back();
}



int yywrap()
{
    if (yyinStack.size() == 0)
        return 1;
    else {
        yyinPop();
        return 0;
    }
}
