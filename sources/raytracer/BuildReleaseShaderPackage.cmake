if(NOT BUILD_CONFIG STREQUAL "Release")
  return()
endif()

file(MAKE_DIRECTORY "${SHADER_PACKAGE_DIRECTORY}")
execute_process(
  COMMAND "${RAYTRACER_EXECUTABLE}"
    --build-shader-package "${SHADER_PACKAGE}"
    --shader-backend "${SHADER_BACKEND}"
    --shader-source-root "${SHADER_SOURCE_ROOT}"
  WORKING_DIRECTORY "${RAYTRACER_WORKING_DIRECTORY}"
  RESULT_VARIABLE result
)
if(NOT result EQUAL 0)
  message(FATAL_ERROR "Release shader package generation failed with exit code ${result}")
endif()
