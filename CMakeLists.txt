cmake_minimum_required(VERSION 3.5)
cmake_policy(SET CMP0048 NEW)
cmake_policy(SET CMP0074 NEW)
cmake_policy(SET CMP0091 NEW)
cmake_policy(SET CMP0056 NEW)
cmake_policy(SET CMP0066 NEW)
set(CMAKE_OSX_DEPLOYMENT_TARGET "13.0")

project(etx-tracer VERSION 0.0.1)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set_property(GLOBAL PROPERTY PREDEFINED_TARGETS_FOLDER "CMake")

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if (MSVC) 
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNOMINMAX /MP")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DNOMINMAX /MP")
endif()

if (APPLE) 
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-deprecated")
endif()

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
add_definitions(-D_CRT_SECURE_NO_WARNINGS)

set(root_folder ${CMAKE_CURRENT_LIST_DIR})

add_subdirectory(thirdparty)
add_subdirectory(sources)

set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT raytracer)

set(bin_dir "${CMAKE_CURRENT_LIST_DIR}/bin")
if (EXISTS "${bin_dir}")
  if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.12")
    file(GLOB_RECURSE bin_files CONFIGURE_DEPENDS
      "${bin_dir}/*.json"
      "${bin_dir}/*.mtl"
      "${bin_dir}/*.cu"
      "${bin_dir}/*.spd"
      "${bin_dir}/*.materials")
  else()
    file(GLOB_RECURSE bin_files
      "${bin_dir}/*.json"
      "${bin_dir}/*.mtl"
      "${bin_dir}/*.cu"
      "${bin_dir}/*.spd"
      "${bin_dir}/*.materials")
  endif()
  if (bin_files)
    add_custom_target(data SOURCES ${bin_files})
    source_group(TREE "${bin_dir}" PREFIX "data" FILES ${bin_files})
    set_property(TARGET data PROPERTY EXCLUDE_FROM_ALL TRUE)
  endif()
endif()