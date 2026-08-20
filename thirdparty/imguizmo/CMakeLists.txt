add_library(imguizmo STATIC
  ImGuizmo.cpp
  ImGuizmo.h
)

target_include_directories(imguizmo PUBLIC ${CMAKE_CURRENT_LIST_DIR})
target_link_libraries(imguizmo PUBLIC imgui)
set_target_properties(imguizmo PROPERTIES FOLDER "thirdparty")
