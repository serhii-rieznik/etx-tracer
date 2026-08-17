file(GLOB TINYEXR_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_LIST_DIR}/src/*.c")

add_library(tinyexr STATIC
  ${TINYEXR_SOURCES}
  deps/zstd/tinyexr_zstd.c
  include/exr.h
)

target_compile_features(tinyexr PRIVATE c_std_11)
if(MSVC)
  target_compile_options(tinyexr PRIVATE /experimental:c11atomics)
endif()
target_include_directories(tinyexr
  PUBLIC "${CMAKE_CURRENT_LIST_DIR}/include"
  PRIVATE "${CMAKE_CURRENT_LIST_DIR}/src" "${CMAKE_CURRENT_LIST_DIR}/deps/zstd"
)
set_target_properties(tinyexr PROPERTIES FOLDER "thirdparty")
