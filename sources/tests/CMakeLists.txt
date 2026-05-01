add_executable(bsdf_validation
  bsdf_validation.cxx
)

set(ETX_UNITY_STD_HEADER "${CMAKE_SOURCE_DIR}/sources/etx/std.hxx")

if(MSVC)
  target_compile_options(bsdf_validation PRIVATE "/FI${ETX_UNITY_STD_HEADER}")
else()
  target_compile_options(bsdf_validation PRIVATE "-include" "${ETX_UNITY_STD_HEADER}")
endif()

target_link_libraries(bsdf_validation
  PRIVATE imgui sokol etx-core etx-render etx-rt etx-engine etx-rhi tinyexr stb_image bluenoise
)

target_include_directories(bsdf_validation
  PRIVATE "${CMAKE_SOURCE_DIR}/sources"
  PRIVATE "${CMAKE_SOURCE_DIR}/sources/etx/render"
)

set_target_properties(bsdf_validation PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY_DEBUG "${root_folder}/bin"
  RUNTIME_OUTPUT_DIRECTORY_RELEASE "${root_folder}/bin"
  RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${root_folder}/bin"
  RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL "${root_folder}/bin"
  FOLDER "tests"
)

if(MSVC)
  target_link_options(bsdf_validation PRIVATE "/NODEFAULTLIB:LIBCMT")
endif()
