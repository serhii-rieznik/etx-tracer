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
