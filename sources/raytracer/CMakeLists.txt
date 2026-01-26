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
  PRIVATE imgui sokol etx-core etx-render etx-rt etx-util etx-rhi tinyexr stb_image
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


add_executable(raytracer-rhi
  main_rhi.cxx
  rhi_test_app.hxx
  rhi_test_app.cxx
)

target_link_libraries(raytracer-rhi
  PRIVATE imgui sokol_new etx-core etx-render etx-rt etx-util etx-rhi tinyexr stb_image
)

if(WIN32)
  find_package(Vulkan REQUIRED)
  target_link_libraries(raytracer-rhi PRIVATE Vulkan::Vulkan)
  target_include_directories(raytracer-rhi PRIVATE ${Vulkan_INCLUDE_DIRS})
endif()

set_target_properties(raytracer-rhi PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY_DEBUG "${root_folder}/bin"
  RUNTIME_OUTPUT_DIRECTORY_RELEASE "${root_folder}/bin"
  RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${root_folder}/bin"
  RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL "${root_folder}/bin"
)

if (APPLE)
  target_link_libraries(raytracer-rhi PRIVATE
    "-framework Foundation"
    "-framework QuartzCore"
    "-framework CoreGraphics"
    "-framework Foundation"
    "-framework Metal"
    "-framework MetalKit"
    "-framework AppKit"
    )
elseif(MSVC)
  target_link_options(raytracer-rhi PRIVATE "/NODEFAULTLIB:LIBCMT")
endif()

set_property(TARGET raytracer-rhi PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "${root_folder}/bin")

# Copy DXC library to bin folder (we only need the DLL for runtime compilation)
if(DXC_FOUND AND DXC_LIBRARY)
    get_filename_component(DXC_LIBRARY_NAME "${DXC_LIBRARY}" NAME)

    add_custom_command(TARGET raytracer-rhi POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${DXC_LIBRARY}"
            "${root_folder}/bin/${DXC_LIBRARY_NAME}"
        COMMENT "Copying DXC library to bin folder"
    )
endif()
