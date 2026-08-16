set(RAYTRACER_SHARED_SOURCES
  application_control.hxx
  application_control_server.hxx
  application_control_server.cxx
  scene_dependencies.hxx
  scene_dependencies.cxx
  application_runner.hxx
  application_runner.cxx
  app.hxx
  app.cxx
  ui.hxx
  ui.cxx
  platform_ui.hxx
  platform_ui.cxx
  render_context.hxx
  render_context.cxx
  options.hxx
  image_output.hxx
  image_output.cxx
  headless_render_context.hxx
  headless_render_context.cxx
  batch_mode.hxx
  batch_mode.cxx
  bsdf_lut_generation.hxx
  bsdf_lut_generation.cxx
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

if(APPLE)
  target_sources(raytracer PRIVATE platform_ui_macos.mm)
  set_source_files_properties(platform_ui_macos.mm PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
endif()

if(APPLE)
  set(ETX_TRACER_APP_ICON "${CMAKE_CURRENT_SOURCE_DIR}/macos/AppIcon.icns")
  set(ETX_MACOS_CODE_SIGN_IDENTITY "-" CACHE STRING
    "Code-signing identity for the ETX Tracer app bundle ('-' creates a local ad-hoc signature)")
  set(ETX_MACOS_BUNDLE_VERSION "2" CACHE STRING
    "Internal macOS bundle build number; increment when publishing a new app bundle")
  set_source_files_properties("${ETX_TRACER_APP_ICON}" PROPERTIES MACOSX_PACKAGE_LOCATION "Resources")

  add_executable(raytracer_app MACOSX_BUNDLE
    main.cxx
    ${RAYTRACER_SHARED_SOURCES}
    platform_ui_macos.mm
    "${ETX_TRACER_APP_ICON}"
  )

  target_compile_options(raytracer_app PRIVATE "-include" "${CMAKE_SOURCE_DIR}/sources/etx/std.hxx")
  target_link_libraries(raytracer_app PRIVATE
    imgui sokol etx-core etx-render etx-rt etx-engine etx-rhi stb_image bluenoise
    "-framework Foundation"
    "-framework QuartzCore"
    "-framework CoreGraphics"
    "-framework Metal"
    "-framework MetalKit"
    "-framework AppKit"
  )
  target_include_directories(raytracer_app PRIVATE
    "${CMAKE_SOURCE_DIR}/sources/etx/render"
    "${CMAKE_SOURCE_DIR}/bin"
  )
  set_target_properties(raytracer_app PROPERTIES
    OUTPUT_NAME "ETX Tracer"
    RUNTIME_OUTPUT_DIRECTORY "${root_folder}/bin"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${root_folder}/bin"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${root_folder}/bin"
    RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${root_folder}/bin"
    RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL "${root_folder}/bin"
    MACOSX_BUNDLE_INFO_PLIST "${CMAKE_CURRENT_SOURCE_DIR}/macos/Info.plist.in"
    MACOSX_BUNDLE_BUNDLE_NAME "ETX Tracer"
    MACOSX_BUNDLE_GUI_IDENTIFIER "com.etxtracer.app"
    MACOSX_BUNDLE_ICON_FILE "AppIcon.icns"
    MACOSX_BUNDLE_BUNDLE_VERSION "${ETX_MACOS_BUNDLE_VERSION}"
    MACOSX_BUNDLE_SHORT_VERSION_STRING "${PROJECT_VERSION}"
    MACOSX_BUNDLE_COPYRIGHT "Copyright © 2026 ETX Tracer contributors"
    MACOSX_RPATH TRUE
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "@executable_path/../Frameworks;@loader_path/../Frameworks"
  )

  # Bundle resources are copied and the finished app is signed in POST_BUILD.
  # Make resource edits relink the executable so those steps cannot be skipped.
  file(GLOB_RECURSE ETX_TRACER_BUNDLE_RESOURCES CONFIGURE_DEPENDS
    "${root_folder}/bin/assets/*"
    "${root_folder}/bin/fonts/*"
    "${root_folder}/bin/spectrum/*"
    "${root_folder}/bin/lib/*"
    "${CMAKE_SOURCE_DIR}/sources/etx/render/shaders/*"
    "${CMAKE_SOURCE_DIR}/sources/etx/render/interop/*"
    "${CMAKE_SOURCE_DIR}/sources/etx/render/access/*"
  )
  list(APPEND ETX_TRACER_BUNDLE_RESOURCES
    "${ETX_TRACER_APP_ICON}"
    "${CMAKE_CURRENT_SOURCE_DIR}/macos/DefaultOptions.json"
    "${CMAKE_CURRENT_SOURCE_DIR}/macos/Info.plist.in"
    "${CMAKE_CURRENT_SOURCE_DIR}/macos/SignBundle.cmake"
  )
  if(DXC_FOUND AND DXC_LIBRARY)
    list(APPEND ETX_TRACER_BUNDLE_RESOURCES "${DXC_LIBRARY}")
  endif()
  set_property(TARGET raytracer_app APPEND PROPERTY LINK_DEPENDS ${ETX_TRACER_BUNDLE_RESOURCES})

  add_custom_command(TARGET raytracer_app POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E remove_directory "$<TARGET_FILE_DIR:raytracer_app>/../Resources/assets"
    COMMAND ${CMAKE_COMMAND} -E remove_directory "$<TARGET_FILE_DIR:raytracer_app>/../Resources/fonts"
    COMMAND ${CMAKE_COMMAND} -E remove_directory "$<TARGET_FILE_DIR:raytracer_app>/../Resources/spectrum"
    COMMAND ${CMAKE_COMMAND} -E remove_directory "$<TARGET_FILE_DIR:raytracer_app>/../Resources/shaders"
    COMMAND ${CMAKE_COMMAND} -E remove_directory "$<TARGET_FILE_DIR:raytracer_app>/../Resources/interop"
    COMMAND ${CMAKE_COMMAND} -E remove_directory "$<TARGET_FILE_DIR:raytracer_app>/../Resources/access"
    COMMAND ${CMAKE_COMMAND} -E remove -f "$<TARGET_FILE_DIR:raytracer_app>/../Resources/options.json"
    COMMAND ${CMAKE_COMMAND} -E remove_directory "$<TARGET_FILE_DIR:raytracer_app>/../Frameworks"
    COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:raytracer_app>/../Resources"
    COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:raytracer_app>/../Frameworks"
    COMMAND ${CMAKE_COMMAND} -E copy_directory
      "${root_folder}/bin/assets"
      "$<TARGET_FILE_DIR:raytracer_app>/../Resources/assets"
    COMMAND ${CMAKE_COMMAND} -E copy_directory
      "${root_folder}/bin/fonts"
      "$<TARGET_FILE_DIR:raytracer_app>/../Resources/fonts"
    COMMAND ${CMAKE_COMMAND} -E copy_directory
      "${root_folder}/bin/spectrum"
      "$<TARGET_FILE_DIR:raytracer_app>/../Resources/spectrum"
    COMMAND ${CMAKE_COMMAND} -E copy_directory
      "${CMAKE_SOURCE_DIR}/sources/etx/render/shaders"
      "$<TARGET_FILE_DIR:raytracer_app>/../Resources/shaders"
    COMMAND ${CMAKE_COMMAND} -E copy_directory
      "${CMAKE_SOURCE_DIR}/sources/etx/render/interop"
      "$<TARGET_FILE_DIR:raytracer_app>/../Resources/interop"
    COMMAND ${CMAKE_COMMAND} -E copy_directory
      "${CMAKE_SOURCE_DIR}/sources/etx/render/access"
      "$<TARGET_FILE_DIR:raytracer_app>/../Resources/access"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${CMAKE_CURRENT_SOURCE_DIR}/macos/DefaultOptions.json"
      "$<TARGET_FILE_DIR:raytracer_app>/../Resources/DefaultOptions.json"
    COMMAND ${CMAKE_COMMAND} -E copy_directory
      "${root_folder}/bin/lib"
      "$<TARGET_FILE_DIR:raytracer_app>/../Frameworks"
    COMMENT "Bundling ETX Tracer resources and runtime libraries"
  )

  if(DXC_FOUND AND DXC_LIBRARY)
    add_custom_command(TARGET raytracer_app POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${DXC_LIBRARY}"
        "$<TARGET_FILE_DIR:raytracer_app>/../Frameworks/"
      COMMENT "Bundling DXC runtime library"
    )
  endif()

  if(ETX_MACOS_CODE_SIGN_IDENTITY)
    add_custom_command(TARGET raytracer_app POST_BUILD
      COMMAND ${CMAKE_COMMAND}
        "-DAPP_BUNDLE=$<TARGET_BUNDLE_DIR:raytracer_app>"
        "-DSIGNING_IDENTITY=${ETX_MACOS_CODE_SIGN_IDENTITY}"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/macos/SignBundle.cmake"
      COMMENT "Code-signing ETX Tracer app bundle"
    )
  endif()
endif()

set(ETX_UNITY_STD_HEADER "${CMAKE_SOURCE_DIR}/sources/etx/std.hxx")

if(MSVC)
  target_compile_options(raytracer PRIVATE "/FI${ETX_UNITY_STD_HEADER}")
else()
  target_compile_options(raytracer PRIVATE "-include" "${ETX_UNITY_STD_HEADER}")
endif()

target_link_libraries(raytracer
  PRIVATE imgui sokol etx-core etx-render etx-rt etx-engine etx-rhi stb_image bluenoise
)

target_include_directories(raytracer
    PRIVATE "${CMAKE_SOURCE_DIR}/sources/etx/render"
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
  target_link_options(raytracer PRIVATE "-Wl,-rpath,@executable_path" "-Wl,-rpath,@loader_path")
endif()

if(MSVC)
  target_link_options(raytracer PRIVATE "/NODEFAULTLIB:LIBCMT")
endif()

set_property(TARGET raytracer PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "${root_folder}/bin")

# Runtime shader compilation must not depend on the process working directory.
# Keep the complete shader include tree next to the executable.
add_custom_target(raytracer_runtime_shaders
  COMMAND ${CMAKE_COMMAND} -E copy_directory
    "${CMAKE_SOURCE_DIR}/sources/etx/render/shaders"
    "${root_folder}/bin/shaders"
  COMMAND ${CMAKE_COMMAND} -E copy_directory
    "${CMAKE_SOURCE_DIR}/sources/etx/render/interop"
    "${root_folder}/bin/interop"
  COMMAND ${CMAKE_COMMAND} -E copy_directory
    "${CMAKE_SOURCE_DIR}/sources/etx/render/access"
    "${root_folder}/bin/access"
  COMMENT "Copying runtime shaders to bin folder"
)
add_dependencies(raytracer raytracer_runtime_shaders)
if(TARGET bsdf_validation)
  add_dependencies(bsdf_validation raytracer_runtime_shaders)
endif()

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
