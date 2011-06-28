if(GLUT_FOUND AND OPENGL_FOUND)
  include_directories(${GLUT_INCLUDE_DIR} . core materials parsers passes renderers shapes)
  add_definitions(-DGLUT_FOUND -DGLUT_NO_LIB_PRAGMA)

  # See top level CMakeLists.txt file for documentation of OPTIX_add_sample_executable.
  OPTIX_add_sample_executable(MaoPPM
    # header files
    core/global.h
    core/DifferentialGeometry.h
    core/Intersection.h
    core/Light.h
    core/Material.h
    core/montecarlo.h
    core/ParameterVector.h
    core/payload.h
    core/reflection.h
    core/BSDF.h
    core/BxDF.h
    core/Fresnel.h
    core/MicrofacetDistribution.h
    core/Renderer.h
    core/SceneBuilder.h
    core/Scene.h
    core/System.h
    core/utility.h
    materials/Glass.h
    materials/Matte.h
    materials/Mirror.h
    materials/Plastic.h
    renderers/IGPPMRenderer.h
    renderers/PathTracingRenderer.h
    renderers/PPMRenderer.h
    # source files
    core/global.cpp
    core/main.cpp
    core/Renderer.cpp
    core/Scene.cpp
    core/SceneBuilder.cpp
    core/System.cpp
    parsers/pbrtParser.cc
    parsers/pbrtScanner.cc
    renderers/IGPPMRenderer.cpp
    renderers/PathTracingRenderer.cpp
    renderers/PPMRenderer.cpp
    # CUDA source files
    core/exception.cu
    core/ray.cu
    materials/Glass.cu
    materials/Matte.cu
    materials/Mirror.cu
    materials/Plastic.cu
    renderers/IGPPMRenderer.cu
    renderers/PathTracingRenderer.cu
    renderers/PPMRenderer.cu
    shapes/TriangleMesh.cu
    shapes/Sphere.cu
    )
else()
  # GLUT or OpenGL not found
  message("Disabling progressive_photon, which requires glut and opengl.")
endif()
