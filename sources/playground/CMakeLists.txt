set(PLAYGROUND_SOURCES
  app.hxx
  app.cxx
  envmap.hxx
  envmap.cxx
  ocean.hxx
  ocean.cxx
)

add_executable(playground
  main.cxx
  ${PLAYGROUND_SOURCES}
)

set(ETX_UNITY_STD_HEADER "${CMAKE_SOURCE_DIR}/sources/etx/std.hxx")

if(MSVC)
  target_compile_options(playground PRIVATE "/FI${ETX_UNITY_STD_HEADER}")
else()
  target_compile_options(playground PRIVATE "-include" "${ETX_UNITY_STD_HEADER}")
endif()

target_link_libraries(playground
  PRIVATE imgui sokol etx-core etx-render etx-rhi etx-engine
)

target_include_directories(playground
    PRIVATE "${CMAKE_SOURCE_DIR}/bin"
    PRIVATE "${CMAKE_SOURCE_DIR}/sources/etx/render"
    PRIVATE "${CMAKE_SOURCE_DIR}/thirdparty/imgui"
    PRIVATE "${CMAKE_SOURCE_DIR}/thirdparty/sokol")

set_target_properties(playground PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY_DEBUG "${root_folder}/bin"
  RUNTIME_OUTPUT_DIRECTORY_RELEASE "${root_folder}/bin"
  RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${root_folder}/bin"
  RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL "${root_folder}/bin"
)

if (APPLE)
  set_target_properties(playground PROPERTIES OUTPUT_NAME "playground_app")
  target_link_libraries(playground PRIVATE
    "-framework Foundation"
    "-framework QuartzCore"
    "-framework CoreGraphics"
    "-framework Foundation"
    "-framework Metal"
    "-framework MetalKit"
    "-framework AppKit"
    )
  target_link_options(playground PRIVATE "-Wl,-rpath,@executable_path" "-Wl,-rpath,@loader_path")
endif()

if(MSVC)
  target_link_options(playground PRIVATE "/NODEFAULTLIB:LIBCMT")
endif()

set_property(TARGET playground PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "${root_folder}/bin")

# Copy DXC runtime library to bin folder (runtime shader compilation)
if(DXC_FOUND AND DXC_LIBRARY)
  get_filename_component(DXC_LIBRARY_NAME "${DXC_LIBRARY}" NAME)
  get_filename_component(DXC_LIBRARY_DIR "${DXC_LIBRARY}" DIRECTORY)
  add_custom_command(TARGET playground POST_BUILD 
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
    add_custom_command(TARGET playground POST_BUILD
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
      add_custom_command(TARGET playground POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${DXC_SIDECAR_SOURCE}" "${root_folder}/bin/${DXC_SIDECAR_LIBRARY}"
        COMMENT "Copying DXC sidecar library to bin folder")
    endif()
  endforeach()
endif()
