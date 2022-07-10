function(etx_build_pack target folder)
  if(NOT IS_DIRECTORY "${folder}")
    message(ERROR "Unable to find folder ${folder}")
    return()
  endif()

  file(GLOB_RECURSE hxx_sources "${folder}/*.hxx")
  list(APPEND sources ${hxx_sources})

  file(GLOB_RECURSE cxx_sources "${folder}/*.cxx")
  list(APPEND sources ${cxx_sources})

  source_group(TREE "${folder}/../" FILES ${sources})

# #[[
  set(lib_source "${CMAKE_CURRENT_BINARY_DIR}/../compile_packs/${target}.pack.cpp")
  file(WRITE ${lib_source})

  foreach(source ${cxx_sources})
    target_sources(${target} PRIVATE ${source})
    set_property(SOURCE ${source} PROPERTY HEADER_FILE_ONLY TRUE)
    file(APPEND ${lib_source} "#include \"${source}\"\n")
  endforeach()

  list(APPEND packed_sources ${lib_source})
  target_sources(${target} PRIVATE ${hxx_sources})
  target_sources(${target} PRIVATE ${packed_sources})
  source_group("" FILES ${lib_source})
  target_sources(${target} PRIVATE ${headers})
# #]]

#  target_sources(${target} PRIVATE ${headers})
#  target_sources(${target} PRIVATE ${sources})

endfunction(etx_build_pack)

function(create_library folder)
  message(STATUS "Creating etx library `etx-${folder}` ...")
  set(lib_name "etx-${folder}")
  string(REPLACE "/" "-" lib_name ${lib_name})

  add_library(${lib_name} STATIC)
  etx_build_pack(${lib_name} "${CMAKE_CURRENT_LIST_DIR}/${folder}")

  target_compile_options(${lib_name} PRIVATE "/Zl")
  target_compile_options(${lib_name} PRIVATE -DNOMINMAX=1 -D_CRT_SECURE_NO_WARNINGS=1 -DWIN32_LEAN_AND_MEAN=1 -D_USE_MATH_DEFINES=1)

  set_property(TARGET ${lib_name} PROPERTY FOLDER "etx")
  set_property(TARGET ${lib_name} PROPERTY LINKER_LANGUAGE CXX)
endfunction(create_library)

create_library(core)
target_link_libraries(etx-core PUBLIC jansson Dbghelp.lib)
target_include_directories(etx-core PUBLIC .. ${JANSSON_INCLUDE_DIRS})

create_library(log)
target_link_libraries(etx-log PRIVATE etx-core)

create_library(render)
target_link_libraries(etx-render PRIVATE stb_image tinyexr tiny_obj_loader mikktspace enkiTS etx-core)
target_include_directories(etx-render PRIVATE ${JANSSON_INCLUDE_DIRS})

if (WITH_OPENVDB)
  target_link_libraries(etx-render PUBLIC openvdb)
  target_compile_definitions(etx-render PUBLIC -DETX_HAVE_OPENVDB=1)
endif()

find_package(embree 3.0 REQUIRED)

set(HAS_OPTIX 0)

if(DEFINED ENV{OptiX_INSTALL_DIR})
  find_package(CUDA QUIET)
  if (CUDA_FOUND)
    # enable_language(CUDA)
    # set(CMAKE_CUDA_STANDARD 20)
    # set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(OptiX_FIND_QUIETLY YES)
    find_package(OptiX QUIET)
    if (OptiX_INCLUDE) 
      set(HAS_OPTIX 1)
    else()
      message(WARNING "OptiX not found, GPU rendering will not be available.")
    endif()
  else()
    message(WARNING "CUDA not found, GPU rendering will not be available.")
  endif()
else()
  message(WARNING "OptiX_INSTALL_DIR environment variable not found, GPU rendering will not be available.")
endif()

if (HAS_OPTIX)
  create_library(gpu)
  target_link_libraries(etx-gpu PUBLIC etx-core etx-log cuda-compiler-lib)
  target_link_libraries(etx-gpu PUBLIC ${optix_LIBRARY} ${CUDA_LIBRARIES} ${CUDA_CUDA_LIBRARY})
  target_include_directories(etx-gpu PUBLIC ${OptiX_INCLUDE} ${CUDA_TOOLKIT_ROOT_DIR}/include)
else()
  create_library(gpu-off)
  target_link_libraries(etx-gpu-off PRIVATE etx-core etx-log)
endif()

create_library(rt)
target_link_libraries(etx-rt PRIVATE embree etx-core etx-render)
target_compile_definitions(etx-rt PUBLIC -DETX_HAVE_GPU_RENDER=${ETX_GPU_RENDER})

if(HAS_OPTIX)
  set(ETX_GPU_RENDER 1 PARENT_SCOPE)
  target_link_libraries(etx-rt PUBLIC etx-gpu)
else()
  set(ETX_GPU_RENDER 0 PARENT_SCOPE)
  target_link_libraries(etx-rt PUBLIC etx-gpu-off)
endif()

