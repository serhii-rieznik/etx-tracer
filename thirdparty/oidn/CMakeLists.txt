set(oidn_version .)

set(include_path "${CMAKE_CURRENT_LIST_DIR}/${oidn_version}/include")
set(library_path "${CMAKE_CURRENT_LIST_DIR}/${oidn_version}/lib")
set(binary_path "${CMAKE_CURRENT_LIST_DIR}/${oidn_version}/bin")

add_library(oidn INTERFACE
  ${include_path}/OpenImageDenoise/config.h
  ${include_path}/OpenImageDenoise/oidn.h
  ${include_path}/OpenImageDenoise/oidn.hpp
)

if(APPLE)
  target_link_libraries(oidn INTERFACE
    "${library_path}/libOpenImageDenoise.dylib"
  )
else()
  target_link_libraries(oidn INTERFACE
    "${library_path}/OpenImageDenoise.lib"
    "${library_path}/OpenImageDenoise_core.lib"
  )
endif()

set_target_properties(oidn PROPERTIES LINKER_LANGUAGE CXX FOLDER "thirdparty")
target_include_directories(oidn INTERFACE "${include_path}")

if (WIN32) 
  configure_file("${binary_path}/OpenImageDenoise.dll" "${root_folder}/bin/" COPYONLY)
  configure_file("${binary_path}/OpenImageDenoise_core.dll" "${root_folder}/bin/" COPYONLY)
  configure_file("${binary_path}/OpenImageDenoise_device_cpu.dll" "${root_folder}/bin/" COPYONLY)
endif()
