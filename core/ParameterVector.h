/*
 * =====================================================================================
 *
 *       Filename:  ParameterVector.h
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  04/13/2011 08:15:50 PM
 *
 *         Author:  YOUR NAME (), 
 *        Company:  
 *
 * =====================================================================================
 */

#ifndef PARAMETER_VECTOR_H
#define PARAMETER_VECTOR_H





#include    <cstdlib>
#include    <cstring>





typedef struct ParameterVector {
    enum Type {
        Unknown, RealNumber, String, Parameter
    };

    Type    type;
    int     allocated;
    int     nElements;
    int     elementSize;

    const char **   name;   /* this var is useless if type != Parameter */
    void *          data;

    ParameterVector() :
        type(Unknown), allocated(0), nElements(0), elementSize(0),
        name(NULL), data(NULL) { }
    ParameterVector(float f) :
        type(RealNumber), allocated(1), nElements(1), elementSize(sizeof(f))
    {
        data = malloc(sizeof(f));
        static_cast<float *>(data)[0] = f;
    }
    ParameterVector(const char * s) :
        type(String), allocated(1), nElements(1), elementSize(sizeof(s))
    {
        data = malloc(sizeof(s));
        static_cast<const char **>(data)[0] = strdup(s);
    }
    ParameterVector(const char * n, ParameterVector * array) :
        type(Parameter), allocated(1), nElements(1), elementSize(sizeof(array))
    {
        name = static_cast<const char **>(malloc(sizeof(name[0])));
        name[0] = strdup(n);
        data = malloc(sizeof(array));
        static_cast<ParameterVector **>(data)[0] = array;
    }

    void append(float f)
    {
        expandIfNecesssary();
        static_cast<float *>(data)[nElements] = f;
        ++nElements;
    }
    void append(const char * s)
    {
        expandIfNecesssary();
        static_cast<const char **>(data)[nElements] = strdup(s);
        ++nElements;
    }
    void append(const char * n, ParameterVector * array)
    {
        expandIfNecesssary();
        static_cast<const char **>(name)[nElements] = strdup(n);
        static_cast<ParameterVector **>(data)[nElements] = array;
        ++nElements;
    }

    void expandIfNecesssary()
    {
        if (nElements >= allocated) {
            if (allocated == 0)
                allocated = 1;
            else
                allocated *= 2;
            data = realloc(data, elementSize * allocated);
            if (type == Parameter)
                name = static_cast<const char **>(realloc(name, sizeof(const char *) * allocated));
        }
    }
} ParameterVector ;



#endif  /* -----  not PARAMETER_VECTOR_H  ----- */
