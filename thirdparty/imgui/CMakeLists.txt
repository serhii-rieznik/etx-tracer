add_library(imgui STATIC 
  imgui.cpp
  imgui_draw.cpp
  imgui_tables.cpp
  imgui_widgets.cpp
)

set_target_properties(imgui PROPERTIES FOLDER "thirdparty")
target_include_directories(imgui PUBLIC ${CMAKE_CURRENT_LIST_DIR})
