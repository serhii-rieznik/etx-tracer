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
  denoiser.cxx
  denoiser.hxx
)

target_link_libraries(raytracer PRIVATE imgui sokol etx-core etx-render etx-rt tinyexr stb_image)

if(WIN32)
  target_link_libraries(raytracer PRIVATE oidn)
endif()

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

if (APPLE)
  target_link_libraries(raytracer PRIVATE
    "-framework Foundation"
    "-framework QuartzCore"
    "-framework CoreGraphics"
    "-framework Foundation"
    "-framework Metal"
    "-framework MetalKit"
    "-framework AppKit"
    )
elseif(MSVC)
  target_link_options(raytracer PRIVATE "/NODEFAULTLIB:LIBCMT")
endif()

set_property(TARGET raytracer PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "${root_folder}/bin")
