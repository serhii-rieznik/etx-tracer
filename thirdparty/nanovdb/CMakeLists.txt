# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: MPL-2.0
#
#[=======================================================================[

  CMake Configuration for NanoVDB

#]=======================================================================]

cmake_minimum_required(VERSION 3.18)
project(NanoVDB LANGUAGES C CXX)

include(GNUInstallDirs)

###############################################################################

message(STATUS "----------------------------------------------------")
message(STATUS "--------------- Configuring NanoVDB ----------------")
message(STATUS "----------------------------------------------------")

###############################################################################
# add options
###############################################################################

option(NANOVDB_BUILD_TOOLS "Build command-line tools" OFF)
option(NANOVDB_BUILD_UNITTESTS "Build Unit tests" OFF)
option(NANOVDB_BUILD_EXAMPLES "Build examples" OFF)
option(NANOVDB_BUILD_BENCHMARK "Build benchmark in examples" OFF)

option(NANOVDB_USE_INTRINSICS "Build with hardware intrinsics support" OFF)
option(NANOVDB_CUDA_KEEP_PTX "Keep CUDA PTX" OFF)

option(NANOVDB_USE_BLOSC "Build with BLOSC support" ON)
option(NANOVDB_USE_ZLIB "Build with ZLIB support" OFF)
option(NANOVDB_USE_MAGICAVOXEL "Build with MagicaVoxel support" OFF)
option(NANOVDB_ALLOW_FETCHCONTENT "Allow FetchContent to download missing dependencies" OFF)

# Add our cmake modules

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/../../cmake")

if(UNIX)
  # For CMake's find Threads module which brings in pthread - This flag
  # forces the compiler -pthread flag vs -lpthread
  set(THREADS_PREFER_PTHREAD_FLAG TRUE)
  find_package(Threads REQUIRED)
endif()

add_subdirectory(blosc)

if(NANOVDB_USE_ZLIB AND NOT TARGET ZLIB::ZLIB)
  find_package(ZLIB REQUIRED)
endif()

###############################################################################
# Installation
###############################################################################

# NanoVDB header files
set(NANOVDB_INCLUDE_FILES
  CNanoVDB.h
  NanoVDB.h
  PNanoVDB.h
)

# NanoVDB util header files
set(NANOVDB_INCLUDE_UTILFILES
  util/CSampleFromVoxels.h
  util/CudaDeviceBuffer.h
  util/DitherLUT.h
  util/ForEach.h
  util/GridBuilder.h
  util/GridChecksum.h
  util/GridHandle.h
  util/GridStats.h
  util/GridValidator.h
  util/HDDA.h
  util/HostBuffer.h
  util/Invoke.h
  util/IO.h
  util/NanoToOpenVDB.h
  util/NodeManager.h
  util/OpenToNanoVDB.h
  util/Primitives.h
  util/Range.h
  util/Ray.h
  util/Reduce.h
  util/SampleFromVoxels.h
  util/Stencils.h
)

add_library(nanovdb INTERFACE ${NANOVDB_INCLUDE_FILES} ${NANOVDB_INCLUDE_UTILFILES})
target_include_directories(nanovdb INTERFACE ../)
target_compile_options(nanovdb INTERFACE
  "$<$<COMPILE_LANG_AND_ID:CXX,GNU>:-Wno-invalid-offsetof>"
  "$<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/bigobj>")
set_target_properties(nanovdb PROPERTIES FOLDER "thirdparty")

if(WIN32)
  target_compile_definitions(nanovdb INTERFACE -DNOMINMAX -D_USE_MATH_DEFINES)
endif()

if(NANOVDB_USE_INTRINSICS)
  target_compile_definitions(nanovdb INTERFACE -DNANOVDB_USE_INTRINSICS)
endif()

if(NANOVDB_USE_BLOSC)
  target_link_libraries(nanovdb INTERFACE Blosc::blosc)
  target_compile_definitions(nanovdb INTERFACE -DNANOVDB_USE_BLOSC)
endif()

if(NANOVDB_USE_ZLIB)
  target_link_libraries(nanovdb INTERFACE ZLIB::ZLIB)
  target_compile_definitions(nanovdb INTERFACE -DNANOVDB_USE_ZIP)
endif()

if(TARGET Threads::Threads)
  target_link_libraries(nanovdb INTERFACE Threads::Threads)
endif()

set(NANOVDB_INSTALL_INCLUDEDIR ${CMAKE_INSTALL_INCLUDEDIR}/nanovdb)
set(NANOVDB_INSTALL_UTILDIR ${NANOVDB_INSTALL_INCLUDEDIR}/util)

install(FILES ${NANOVDB_INCLUDE_FILES} DESTINATION ${NANOVDB_INSTALL_INCLUDEDIR})
install(FILES ${NANOVDB_INCLUDE_UTILFILES} DESTINATION ${NANOVDB_INSTALL_UTILDIR})

