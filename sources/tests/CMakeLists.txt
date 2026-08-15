add_executable(bsdf_validation
  bsdf_validation.cxx
)

add_executable(procedural_geometry
  procedural_geometry.cxx
)

set(ETX_UNITY_STD_HEADER "${CMAKE_SOURCE_DIR}/sources/etx/std.hxx")

if(MSVC)
  target_compile_options(bsdf_validation PRIVATE "/FI${ETX_UNITY_STD_HEADER}")
  target_compile_options(procedural_geometry PRIVATE "/FI${ETX_UNITY_STD_HEADER}")
else()
  target_compile_options(bsdf_validation PRIVATE "-include" "${ETX_UNITY_STD_HEADER}")
  target_compile_options(procedural_geometry PRIVATE "-include" "${ETX_UNITY_STD_HEADER}")
endif()

target_link_libraries(bsdf_validation
  PRIVATE imgui sokol etx-core etx-render etx-rt etx-engine etx-rhi tinyexr stb_image bluenoise
)

target_link_libraries(procedural_geometry
  PRIVATE imgui sokol etx-core etx-render etx-rt etx-engine etx-rhi tinyexr stb_image bluenoise
)

target_include_directories(bsdf_validation
  PRIVATE "${CMAKE_SOURCE_DIR}/sources"
  PRIVATE "${CMAKE_SOURCE_DIR}/sources/etx/render"
)

target_include_directories(procedural_geometry
  PRIVATE "${CMAKE_SOURCE_DIR}/sources"
  PRIVATE "${CMAKE_SOURCE_DIR}/sources/etx/render"
)

set_target_properties(bsdf_validation PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY_DEBUG "${root_folder}/bin"
  RUNTIME_OUTPUT_DIRECTORY_RELEASE "${root_folder}/bin"
  RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${root_folder}/bin"
  RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL "${root_folder}/bin"
  FOLDER "tests"
)

set_target_properties(procedural_geometry PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY_DEBUG "${root_folder}/bin"
  RUNTIME_OUTPUT_DIRECTORY_RELEASE "${root_folder}/bin"
  RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${root_folder}/bin"
  RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL "${root_folder}/bin"
  FOLDER "tests"
)

if(APPLE)
  foreach(test_target bsdf_validation procedural_geometry)
    target_link_libraries(${test_target} PRIVATE
      "-framework Foundation"
      "-framework QuartzCore"
      "-framework CoreGraphics"
      "-framework Metal"
      "-framework MetalKit"
      "-framework AppKit"
    )
    target_link_options(${test_target} PRIVATE "-Wl,-rpath,@executable_path" "-Wl,-rpath,@loader_path")
  endforeach()
endif()

if(DXC_FOUND AND DXC_LIBRARY)
  get_filename_component(DXC_LIBRARY_NAME "${DXC_LIBRARY}" NAME)
  get_filename_component(DXC_LIBRARY_DIR "${DXC_LIBRARY}" DIRECTORY)
  add_custom_command(TARGET bsdf_validation POST_BUILD
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
    add_custom_command(TARGET bsdf_validation POST_BUILD
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
      add_custom_command(TARGET bsdf_validation POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${DXC_SIDECAR_SOURCE}" "${root_folder}/bin/${DXC_SIDECAR_LIBRARY}"
        COMMENT "Copying DXC sidecar library to bin folder")
    endif()
  endforeach()
endif()

if(MSVC)
  target_link_options(bsdf_validation PRIVATE "/NODEFAULTLIB:LIBCMT")
  target_link_options(procedural_geometry PRIVATE "/NODEFAULTLIB:LIBCMT")
endif()
