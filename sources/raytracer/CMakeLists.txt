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

target_link_libraries(raytracer
  PRIVATE imgui sokol etx-core etx-render etx-rt etx-util tinyexr stb_image
)

set_target_properties(raytracer PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY_DEBUG "${root_folder}/bin"
  RUNTIME_OUTPUT_DIRECTORY_RELEASE "${root_folder}/bin"
  RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${root_folder}/bin"
  RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL "${root_folder}/bin"
)

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
