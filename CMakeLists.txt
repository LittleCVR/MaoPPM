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
    core/Renderer.h
    core/SceneBuilder.h
    core/Scene.h
    core/System.h
    core/utility.h
    materials/Matte.h
    renderers/IGPPMRenderer.h
#    renderers/PathTracingRenderer.h
#    renderers/PPMRenderer.h
    # source files
    core/global.cpp
    core/main.cpp
    core/Renderer.cpp
    core/Scene.cpp
    core/SceneBuilder.cpp
    core/System.cpp
    parsers/pbrtParser.cc
    parsers/pbrtScanner.cc
#    renderers/PathTracingRenderer.cpp
#    renderers/PPMRenderer.cpp
    renderers/IGPPMRenderer.cpp
    # CUDA source files
    core/exception.cu
    core/ray.cu
    materials/Matte.cu
#    materials/matteMaterialPathTracingPrograms.cu
#    materials/matteMaterialPPMPrograms.cu
#    materials/plasticMaterialPathTracingPrograms.cu
#    materials/plasticMaterialPPMPrograms.cu
#    passes/pathTracingPassPrograms.cu
#    passes/pixelSamplingPassPrograms.cu
#    passes/importonShootingPassPrograms.cu
#    passes/photonShootingPassPrograms.cu
#    passes/gatheringPassPrograms.cu
    renderers/IGPPMRenderer.cu
    shapes/triangleMeshShapePrograms.cu
    )
else()
  # GLUT or OpenGL not found
  message("Disabling progressive_photon, which requires glut and opengl.")
endif()
