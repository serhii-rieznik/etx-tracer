add_library(sokol_imgui STATIC
  sokol_imgui.cpp
  sokol_imgui.h
  cimgui.cpp
  cimgui.h
)

target_link_libraries(sokol_imgui sokol_app sokol_gfx imgui)
target_include_directories(sokol_imgui PUBLIC ${CMAKE_CURRENT_LIST_DIR})
set_target_properties(sokol_imgui PROPERTIES FOLDER "thirdparty")