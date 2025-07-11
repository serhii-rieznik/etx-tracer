set(include_path "${CMAKE_CURRENT_LIST_DIR}/include")
set(library_path "${CMAKE_CURRENT_LIST_DIR}/lib")
set(binary_path "${CMAKE_CURRENT_LIST_DIR}/bin")
set(local_library_path "${root_folder}/bin/lib/")

set(oidn_version_high "2")
set(oidn_version_low "3")
set(oidn_version_patch "3")
set(oidn_version "${oidn_version_high}.${oidn_version_low}.${oidn_version_patch}")

if(NOT EXISTS "${library_path}")
  message(FATAL_ERROR "OIDN library folder not found at ${library_path}.\n*********\n1. download OIDN: https://www.openimagedenoise.org/)\n2. unpack it completely to: ${CMAKE_CURRENT_LIST_DIR}\n*********")
endif()

if(NOT EXISTS "${include_path}")
  message(FATAL_ERROR "OIDN library folder not found at ${library_path}.\n*********\n1. download OIDN: https://www.openimagedenoise.org/)\n2. unpack it completely to: ${CMAKE_CURRENT_LIST_DIR}\n*********")
endif()

if(APPLE) 
  message(STATUS "Copying OIDN libraries to ${local_library_path}")
  configure_file("${library_path}/libOpenImageDenoise.${oidn_version_high}.dylib" "${local_library_path}" COPYONLY)
  configure_file("${library_path}/libOpenImageDenoise.${oidn_version}.dylib" "${local_library_path}" COPYONLY)
  configure_file("${library_path}/libOpenImageDenoise_core.${oidn_version}.dylib" "${local_library_path}" COPYONLY)
  configure_file("${library_path}/libOpenImageDenoise_device_cpu.${oidn_version}.dylib" "${local_library_path}" COPYONLY)
endif()

add_library(oidn INTERFACE
  ${include_path}/OpenImageDenoise/config.h
  ${include_path}/OpenImageDenoise/oidn.h
  ${include_path}/OpenImageDenoise/oidn.hpp
)

if(APPLE)
  target_link_libraries(oidn INTERFACE
    "${local_library_path}/libOpenImageDenoise.${oidn_version}.dylib"
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
