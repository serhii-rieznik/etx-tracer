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

target_link_libraries(raytracer sokol_imgui etx-core etx-log etx-render etx-rt tinyexr stb_image)

set_target_properties(raytracer PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY_DEBUG "${root_folder}/bin"
  RUNTIME_OUTPUT_DIRECTORY_RELEASE "${root_folder}/bin"
  RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${root_folder}/bin"
  RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL "${root_folder}/bin"
)

if (ETX_GPU_RENDER)
  file(GLOB_RECURSE cu_sources "${root_folder}/bin/optix/*.cu")
  source_group(TREE "${root_folder}/bin/" FILES ${cu_sources})
  target_sources(raytracer PRIVATE ${cu_sources})
  set_source_files_properties(${cu_sources} PROPERTIES HEADER_FILE_ONLY TRUE)
endif()                               