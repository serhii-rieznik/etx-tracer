add_library(nativefiledialog STATIC
  src/nfd_common.h
  src/include/nfd.h
  src/common.h
  src/nfd_cocoa.m
  src/nfd_common.c
  src/nfd_gtk.c
  src/nfd_win.cpp
  src/simple_exec.h
)

target_include_directories(nativefiledialog PUBLIC "${CMAKE_CURRENT_LIST_DIR}/src/include")
set_target_properties(nativefiledialog PROPERTIES FOLDER "thirdparty")
