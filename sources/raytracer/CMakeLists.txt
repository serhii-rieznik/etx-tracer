set(RAYTRACER_SHARED_SOURCES
  app.hxx
  app.cxx
  ui.hxx
  ui.cxx
  render_context.hxx
  render_context.cxx
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

set(ETX_UNITY_STD_HEADER "${CMAKE_SOURCE_DIR}/sources/etx/std.hxx")

if(MSVC)
  target_compile_options(raytracer PRIVATE "/FI${ETX_UNITY_STD_HEADER}")
else()
  target_compile_options(raytracer PRIVATE "-include" "${ETX_UNITY_STD_HEADER}")
endif()

target_link_libraries(raytracer
  PRIVATE imgui sokol etx-core etx-render etx-rt etx-util etx-rhi tinyexr stb_image
)

target_include_directories(raytracer
    PRIVATE "${CMAKE_SOURCE_DIR}/bin"
    PRIVATE "${CMAKE_SOURCE_DIR}/sources/etx/render")

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
  target_link_options(raytracer PRIVATE "-Wl,-rpath,@executable_path" "-Wl,-rpath,@loader_path")
elseif (WIN32)
  find_package(Vulkan REQUIRED)
  target_link_libraries(raytracer PRIVATE Vulkan::Vulkan)
  target_include_directories(raytracer PRIVATE ${Vulkan_INCLUDE_DIRS} "${CMAKE_SOURCE_DIR}/bin" "${CMAKE_SOURCE_DIR}/sources/etx/render")
endif()

if(MSVC)
  target_link_options(raytracer PRIVATE "/NODEFAULTLIB:LIBCMT")
endif()

set_property(TARGET raytracer PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "${root_folder}/bin")

# Copy DXC runtime library to bin folder (runtime shader compilation)
if(DXC_FOUND AND DXC_LIBRARY)
  get_filename_component(DXC_LIBRARY_NAME "${DXC_LIBRARY}" NAME)
  get_filename_component(DXC_LIBRARY_DIR "${DXC_LIBRARY}" DIRECTORY)
  add_custom_command(TARGET raytracer POST_BUILD 
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${DXC_LIBRARY}" "${root_folder}/bin/${DXC_LIBRARY_NAME}" 
    COMMENT "Copying DXC library to bin folder")

  set(DXC_COPY_SEARCH_DIRS
    "${DXC_LIBRARY_DIR}"
  )
  if(DXC_ROOT)
    list(APPEND DXC_COPY_SEARCH_DIRS
      "${DXC_ROOT}"
      "${DXC_ROOT}/bin"
      "${DXC_ROOT}/lib"
      "${DXC_ROOT}/lib64"
    )
  endif()
  list(REMOVE_DUPLICATES DXC_COPY_SEARCH_DIRS)

  if(DXC_EXECUTABLE)
    get_filename_component(DXC_EXECUTABLE_NAME "${DXC_EXECUTABLE}" NAME)
    add_custom_command(TARGET raytracer POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different "${DXC_EXECUTABLE}" "${root_folder}/bin/${DXC_EXECUTABLE_NAME}"
      COMMENT "Copying DXC executable to bin folder")
  endif()

  set(DXC_SIDECAR_LIBRARIES dxil.dll libdxil.dylib libdxil.so)
  foreach(DXC_SIDECAR_LIBRARY ${DXC_SIDECAR_LIBRARIES})
    set(DXC_SIDECAR_SOURCE "")
    foreach(DXC_SEARCH_DIR ${DXC_COPY_SEARCH_DIRS})
      if(EXISTS "${DXC_SEARCH_DIR}/${DXC_SIDECAR_LIBRARY}")
        set(DXC_SIDECAR_SOURCE "${DXC_SEARCH_DIR}/${DXC_SIDECAR_LIBRARY}")
        break()
      endif()
    endforeach()

    if(DXC_SIDECAR_SOURCE)
      add_custom_command(TARGET raytracer POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${DXC_SIDECAR_SOURCE}" "${root_folder}/bin/${DXC_SIDECAR_LIBRARY}"
        COMMENT "Copying DXC sidecar library to bin folder")
    endif()
  endforeach()
endif()
