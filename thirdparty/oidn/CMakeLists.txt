add_library(oidn STATIC
  include/OpenImageDenoise/config.h
  include/OpenImageDenoise/oidn.h
  include/OpenImageDenoise/oidn.hpp
)

if(WIN32)
  target_link_libraries(oidn PUBLIC OpenImageDenoise.lib)
endif()

if(APPLE)
endif()

if(LINUX)
endif()

set_target_properties(oidn PROPERTIES LINKER_LANGUAGE CXX FOLDER "thirdparty")
target_include_directories(oidn PUBLIC "${CMAKE_CURRENT_LIST_DIR}/include")

