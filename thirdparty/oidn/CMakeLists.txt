add_library(oidn INTERFACE
  include/OpenImageDenoise/config.h
  include/OpenImageDenoise/oidn.h
  include/OpenImageDenoise/oidn.hpp
)

if (APPLE)
  target_link_libraries(oidn INTERFACE
    "${CMAKE_CURRENT_LIST_DIR}/lib/libOpenImageDenoise.dylib"
  )
else()
  target_link_libraries(oidn INTERFACE
    "${CMAKE_CURRENT_LIST_DIR}/lib/OpenImageDenoise.lib"
    "${CMAKE_CURRENT_LIST_DIR}/lib/OpenImageDenoise_core.lib"
  )
endif()

set_target_properties(oidn PROPERTIES LINKER_LANGUAGE CXX FOLDER "thirdparty")
target_include_directories(oidn INTERFACE "${CMAKE_CURRENT_LIST_DIR}/include")
