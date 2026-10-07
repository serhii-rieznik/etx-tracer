cmake_minimum_required(VERSION 3.20)

if(NOT DEFINED SOURCE_ROOT OR NOT DEFINED DESTINATION_ROOT)
  message(FATAL_ERROR "SOURCE_ROOT and DESTINATION_ROOT are required")
endif()

foreach(tree shaders interop access)
  set(source_directory "${SOURCE_ROOT}/${tree}")
  set(destination_directory "${DESTINATION_ROOT}/${tree}")
  if(NOT IS_DIRECTORY "${source_directory}")
    message(FATAL_ERROR "Runtime shader source directory is missing: ${source_directory}")
  endif()

  file(GLOB_RECURSE source_files LIST_DIRECTORIES FALSE RELATIVE "${source_directory}" "${source_directory}/*")
  file(GLOB_RECURSE destination_files LIST_DIRECTORIES FALSE RELATIVE "${destination_directory}" "${destination_directory}/*")
  foreach(path IN LISTS destination_files)
    if(NOT path IN_LIST source_files)
      file(REMOVE "${destination_directory}/${path}")
    endif()
  endforeach()

  # COPYONLY preserves destination timestamps when the contents match.
  file(MAKE_DIRECTORY "${destination_directory}")
  foreach(path IN LISTS source_files)
    configure_file("${source_directory}/${path}" "${destination_directory}/${path}" COPYONLY)
  endforeach()
endforeach()
