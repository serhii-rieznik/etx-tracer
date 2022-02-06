add_library(imgui STATIC
  imgui.cpp
  imgui_demo.cpp
  imgui_draw.cpp
  imgui_tables.cpp
  imgui_widgets.cpp
  imconfig.h
  imgui.h
  imgui_internal.h
  imstb_rectpack.h
  imstb_textedit.h
  imstb_truetype.h
)

target_include_directories(imgui PUBLIC ${CMAKE_CURRENT_LIST_DIR})
set_target_properties(imgui PROPERTIES FOLDER "thirdparty")