add_library(nativefiledialog STATIC
  src/nfd_common.h
  src/include/nfd.h
  src/common.h
  src/nfd_common.c
  src/simple_exec.h
)

if(WIN32)
	target_sources(nativefiledialog PRIVATE src/nfd_win.cpp)
endif()

if(APPLE)
	target_sources(nativefiledialog PRIVATE src/nfd_cocoa.m)
endif()

if(LINUX)
	target_sources(nativefiledialog PRIVATE src/nfd_gtk.c)
endif()

target_include_directories(nativefiledialog PUBLIC "${CMAKE_CURRENT_LIST_DIR}/src/include")
set_target_properties(nativefiledialog PROPERTIES FOLDER "thirdparty")
