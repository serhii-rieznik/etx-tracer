add_executable(raytracer 
  main.cxx
)

# set_target_properties(raytracer PROPERTIES WIN32_EXECUTABLE YES)

target_link_libraries(raytracer sokol_app etx-core etx-log etx-app etx-rt)
