# CMakeList.txt : CMake project for bcdec, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

project ("bcdec" LANGUAGES C)

option(BCDEC_BUILD_TESTS "Build test executable" OFF)
option(BCDEC_STATIC_LIB "Build as static library instead of interface library" OFF)

if(BCDEC_STATIC_LIB)
  # Static library - compiled implementation
  add_library(bcdec STATIC bcdec.h bcdec.c)
  target_include_directories(bcdec PUBLIC ${CMAKE_CURRENT_LIST_DIR})
else()
  # Interface library - header-only, requires BCDEC_IMPLEMENTATION definition
  add_library(bcdec INTERFACE bcdec.h)
  target_include_directories(bcdec INTERFACE ${CMAKE_CURRENT_LIST_DIR})
endif()

# Optionally build test executable
if(BCDEC_BUILD_TESTS)
  add_executable(bcdec_test "bcdec.h" "test.c")
endif()
