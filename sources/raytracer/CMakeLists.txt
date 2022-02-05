add_executable(raytracer 
  main.cxx
  app.hxx
  app.cxx
  ui.hxx
  ui.cxx
  render.hxx
  render.cxx
  options.hxx
  camera_controller.hxx
)

target_link_libraries(raytracer sokol_imgui glm etx-core etx-log etx-render etx-rt)

set_target_properties(raytracer PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY_DEBUG "${root_folder}/bin"
  RUNTIME_OUTPUT_DIRECTORY_RELEASE "${root_folder}/bin"
  RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${root_folder}/bin"
  RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL "${root_folder}/bin"
)

