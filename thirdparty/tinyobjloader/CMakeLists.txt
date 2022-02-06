add_library(tiny_obj_loader
  tiny_obj_loader.hxx
  tiny_obj_loader.cxx
)

target_include_directories(tiny_obj_loader PUBLIC ${CMAKE_CURRENT_LIST_DIR})
set_target_properties(tiny_obj_loader PROPERTIES FOLDER "thirdparty")
