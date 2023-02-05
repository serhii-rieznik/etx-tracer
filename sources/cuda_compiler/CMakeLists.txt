find_package(CUDAToolkit REQUIRED)

add_library(cuda-compiler-lib STATIC 
  cuda_compiler_lib.hxx
  cuda_compiler_lib.cxx
)

get_filename_component(ms_build_path ${CMAKE_C_COMPILER} DIRECTORY)

target_link_libraries(cuda-compiler-lib PUBLIC etx-log CUDA::nvrtc)

target_include_directories(cuda-compiler-lib PUBLIC "${CMAKE_CURRENT_LIST_DIR}")
target_include_directories(cuda-compiler-lib PRIVATE ${OptiX_INCLUDE} ${CUDA_TOOLKIT_ROOT_DIR}/include ${root_folder}/sources)

target_compile_definitions(cuda-compiler-lib PRIVATE "-DETX_INCLUDES=R\"(${root_folder}/sources/)\"")
target_compile_definitions(cuda-compiler-lib PRIVATE "-DETX_OPTIX_INCLUDES=R\"(${OptiX_INCLUDE}/)\"")
target_compile_definitions(cuda-compiler-lib PRIVATE "-DETX_CUDA_INCLUDES=R\"(${CUDA_TOOLKIT_ROOT_DIR}/include)\"")
target_compile_definitions(cuda-compiler-lib PRIVATE "-DETX_MSBUILD_PATH=R\"(${ms_build_path}/)\"")
target_compile_definitions(cuda-compiler-lib PRIVATE "-DETX_CUDA_COMPILER=R\"(${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc.exe)\"")
set_property(TARGET cuda-compiler-lib PROPERTY FOLDER "etx")

add_executable(cuda-compiler cuda_compiler.cxx)
target_link_libraries(cuda-compiler cuda-compiler-lib)

set_target_properties(cuda-compiler PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY_DEBUG "${root_folder}/bin"
  RUNTIME_OUTPUT_DIRECTORY_RELEASE "${root_folder}/bin"
  RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${root_folder}/bin"
  RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL "${root_folder}/bin"
)
set_property(TARGET cuda-compiler PROPERTY FOLDER "etx")
