add_library(stb_image
  stb_image.hxx
  stb_image.cxx
)

target_include_directories(stb_image PUBLIC ${CMAKE_CURRENT_LIST_DIR})
set_target_properties(stb_image PROPERTIES FOLDER "thirdparty")