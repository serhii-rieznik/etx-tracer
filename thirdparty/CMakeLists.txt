add_subdirectory(sokol_app)
add_subdirectory(sokol_gfx)
add_subdirectory(imgui)

add_subdirectory(sokol_imgui)

add_subdirectory(stb_image)
add_subdirectory(tinyexr)

add_subdirectory(jansson)

add_subdirectory(tinyobjloader)
add_subdirectory(mikktspace)

add_subdirectory(enkits)

if (WITH_OPENVDB)
  add_subdirectory(openvdb)
endif()
