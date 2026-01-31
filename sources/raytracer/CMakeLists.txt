set(RAYTRACER_SHARED_SOURCES
  app.hxx
  app.cxx
  ui.hxx
  ui.cxx
  render.hxx
  render.cxx
  options.hxx
  camera_controller.hxx
  gpu_renderer.hxx
  gpu_renderer.cxx
  raster_renderer.cxx
  raster_renderer.hxx
  cpu_renderer.hxx
  cpu_renderer.cxx
)

add_executable(raytracer
  main.cxx
  ${RAYTRACER_SHARED_SOURCES}
)

target_link_libraries(raytracer
  PRIVATE imgui sokol etx-core etx-render etx-rt etx-util etx-rhi tinyexr stb_image
)

target_include_directories(raytracer
    PRIVATE "${CMAKE_SOURCE_DIR}/bin")

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
elseif (WIN32)
  find_package(Vulkan REQUIRED)
  target_link_libraries(raytracer PRIVATE Vulkan::Vulkan)
  target_include_directories(raytracer PRIVATE ${Vulkan_INCLUDE_DIRS} "${CMAKE_SOURCE_DIR}/bin")
endif()

if(MSVC)
  target_link_options(raytracer PRIVATE "/NODEFAULTLIB:LIBCMT")
endif()

set_property(TARGET raytracer PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "${root_folder}/bin")

# Copy DXC library to bin folder (we only need the DLL for runtime compilation)
if(DXC_FOUND AND DXC_LIBRARY)
  get_filename_component(DXC_LIBRARY_NAME "${DXC_LIBRARY}" NAME)
  add_custom_command(TARGET raytracer POST_BUILD 
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${DXC_LIBRARY}" "${root_folder}/bin/${DXC_LIBRARY_NAME}" 
    COMMENT "Copying DXC library to bin folder")
endif()
