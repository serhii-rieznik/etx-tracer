add_library(tiny_gltf
  tiny_gltf.hxx
  tiny_gltf.cxx
)

target_link_libraries(tiny_gltf PRIVATE json)
target_include_directories(tiny_gltf PUBLIC ${CMAKE_CURRENT_LIST_DIR})
set_target_properties(tiny_gltf PROPERTIES FOLDER "thirdparty")
