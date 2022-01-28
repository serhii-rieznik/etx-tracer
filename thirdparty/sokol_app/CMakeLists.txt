add_library(sokol_app STATIC 
  sokol_app.h 
  sokol_app.c
)
set_target_properties(sokol_app PROPERTIES FOLDER "thirdparty")
target_include_directories(sokol_app PUBLIC ${CMAKE_CURRENT_LIST_DIR})
