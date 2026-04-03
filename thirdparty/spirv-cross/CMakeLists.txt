add_library(spirv-cross-core STATIC
  GLSL.std.450.h
  NonSemanticShaderDebugInfo100.h
  spirv.h
  spirv.hpp
  spirv_cfg.cpp
  spirv_cfg.hpp
  spirv_common.hpp
  spirv_cross.cpp
  spirv_cross.hpp
  spirv_cross_containers.hpp
  spirv_cross_error_handling.hpp
  spirv_cross_parsed_ir.cpp
  spirv_cross_parsed_ir.hpp
  spirv_parser.cpp
  spirv_parser.hpp
)
target_include_directories(spirv-cross-core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")

add_library(spirv-cross-glsl STATIC
  spirv_glsl.cpp
  spirv_glsl.hpp
)
target_include_directories(spirv-cross-glsl PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(spirv-cross-glsl PUBLIC spirv-cross-core)

add_library(spirv-cross-msl STATIC
  spirv_msl.cpp
  spirv_msl.hpp
)
target_include_directories(spirv-cross-msl PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(spirv-cross-msl PUBLIC spirv-cross-glsl)
