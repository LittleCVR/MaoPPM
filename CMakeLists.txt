if(GLUT_FOUND AND OPENGL_FOUND)
  include_directories(${GLUT_INCLUDE_DIR} . core materials parsers passes renderers shapes)
  add_definitions(-DGLUT_FOUND -DGLUT_NO_LIB_PRAGMA)

  # See top level CMakeLists.txt file for documentation of OPTIX_add_sample_executable.
  OPTIX_add_sample_executable(MaoPPM
    # header files
    core/global.h
    core/ParameterVector.h
    core/Renderer.h
	core/Scene.h
    core/SceneBuilder.h
    core/System.h
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
    renderers/PPMRenderer.cpp
    # CUDA source files
    core/samplers.cu
    core/math.cu
	materials/matteMaterialPrograms.cu
	materials/plasticMaterialPrograms.cu
	passes/pixelSamplingPassPrograms.cu
	passes/importonShootingPassPrograms.cu
	passes/photonShootingPassPrograms.cu
	passes/gatheringPassPrograms.cu
	shapes/triangleMeshShapePrograms.cu
    )
else()
  # GLUT or OpenGL not found
  message("Disabling progressive_photon, which requires glut and opengl.")
endif()



