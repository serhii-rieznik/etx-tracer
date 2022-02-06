add_library(tinyexr
  tinyexr.hxx
  tinyexr.cxx
)

target_include_directories(tinyexr PUBLIC ${CMAKE_CURRENT_LIST_DIR})
set_target_properties(tinyexr PROPERTIES FOLDER "thirdparty")