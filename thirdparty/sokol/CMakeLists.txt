add_library(sokol STATIC 
  sokol_app.h
  sokol_gfx.h
  util/sokol_imgui.h
  sokol.cc
  sokol.mm
)

set_target_properties(sokol PROPERTIES FOLDER "thirdparty")
target_include_directories(sokol PUBLIC ${CMAKE_CURRENT_LIST_DIR})
target_link_libraries(sokol PUBLIC imgui)

add_library(sokol_new STATIC 
  sokol_app_new.h
  util/sokol_imgui.h
  sokol.cc
)

set_target_properties(sokol_new PROPERTIES FOLDER "thirdparty")
target_compile_definitions(sokol_new PRIVATE "ETX_SOKOL_NEW")
target_include_directories(sokol_new PUBLIC ${CMAKE_CURRENT_LIST_DIR})
target_link_libraries(sokol_new PUBLIC imgui)
