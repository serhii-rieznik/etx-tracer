cmake_minimum_required(VERSION 3.20)
cmake_policy(SET CMP0048 NEW)
cmake_policy(SET CMP0074 NEW)
cmake_policy(SET CMP0077 NEW)
cmake_policy(SET CMP0091 NEW)
cmake_policy(SET CMP0056 NEW)
cmake_policy(SET CMP0066 NEW)

project(etx-tracer VERSION 0.0.1)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set_property(GLOBAL PROPERTY PREDEFINED_TARGETS_FOLDER "CMake")

set(CMAKE_OSX_DEPLOYMENT_TARGET "13.3")
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake)
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

# Find or install DXC
find_package(DXC REQUIRED)

add_subdirectory(thirdparty)
add_subdirectory(sources)

set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT raytracer)

set(bin_dir "${CMAKE_CURRENT_LIST_DIR}/bin")
set(render_dir "${CMAKE_CURRENT_LIST_DIR}/sources/etx/render")
set(bin_files "")
set(render_files "")

if (EXISTS "${render_dir}")
  if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.12")
    file(GLOB_RECURSE render_files CONFIGURE_DEPENDS
      "${render_dir}/*.hlsl"
      "${render_dir}/*.hxx"
    )
  else()
    file(GLOB_RECURSE render_files
      "${render_dir}/*.hlsl"
      "${render_dir}/*.hxx"
    )
  endif()
endif()

if (EXISTS "${bin_dir}")
  if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.12")
    file(GLOB_RECURSE bin_files CONFIGURE_DEPENDS
      "${bin_dir}/*.json"
      "${bin_dir}/*.mtl"
      "${bin_dir}/*.cu"
      "${bin_dir}/*.spd"
      "${bin_dir}/*.materials"
      "${bin_dir}/*.hlsl"
      "${bin_dir}/*.hxx"
    )
  else()
    file(GLOB_RECURSE bin_files
      "${bin_dir}/*.json"
      "${bin_dir}/*.mtl"
      "${bin_dir}/*.cu"
      "${bin_dir}/*.spd"
      "${bin_dir}/*.materials"
      "${bin_dir}/*.hlsl"
      "${bin_dir}/*.hxx"
    )
  endif()
  list(FILTER bin_files EXCLUDE REGEX "[/\\\\]assets_testing[/\\\\]")

  set(data_hlsl_files "")
  set(render_hlsl_files ${render_files})
  list(FILTER render_hlsl_files INCLUDE REGEX "\\.hlsl$")
  list(APPEND data_hlsl_files ${render_hlsl_files})

  set(bin_hlsl_files ${bin_files})
  list(FILTER bin_hlsl_files INCLUDE REGEX "\\.hlsl$")
  list(APPEND data_hlsl_files ${bin_hlsl_files})

  if (data_hlsl_files)
    list(REMOVE_DUPLICATES data_hlsl_files)
    set_source_files_properties(${data_hlsl_files} PROPERTIES HEADER_FILE_ONLY TRUE)
    if (MSVC)
      set_source_files_properties(${data_hlsl_files} PROPERTIES VS_TOOL_OVERRIDE "None")
    endif()
  endif()

  if (bin_files OR render_files)
    add_custom_target(data SOURCES ${bin_files} ${render_files})
    if (bin_files)
      source_group(TREE "${bin_dir}" PREFIX "data/bin" FILES ${bin_files})
    endif()
    if (render_files)
      source_group(TREE "${render_dir}" PREFIX "data/render" FILES ${render_files})
    endif()
    set_property(TARGET data PROPERTY EXCLUDE_FROM_ALL TRUE)
  endif()
elseif (render_files)
  set(data_hlsl_files "")
  set(render_hlsl_files ${render_files})
  list(FILTER render_hlsl_files INCLUDE REGEX "\\.hlsl$")
  list(APPEND data_hlsl_files ${render_hlsl_files})

  if (data_hlsl_files)
    list(REMOVE_DUPLICATES data_hlsl_files)
    set_source_files_properties(${data_hlsl_files} PROPERTIES HEADER_FILE_ONLY TRUE)
    if (MSVC)
      set_source_files_properties(${data_hlsl_files} PROPERTIES VS_TOOL_OVERRIDE "None")
    endif()
  endif()

  add_custom_target(data SOURCES ${render_files})
  source_group(TREE "${render_dir}" PREFIX "data/render" FILES ${render_files})
  set_property(TARGET data PROPERTY EXCLUDE_FROM_ALL TRUE)
endif()
