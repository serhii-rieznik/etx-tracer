function(etx_build_pack target folder)
  if(NOT IS_DIRECTORY "${folder}")
    message(ERROR "Unable to find folder ${folder}")
    return()
  endif()

  file(GLOB_RECURSE hxx_sources "${folder}/*.hxx")
  list(APPEND sources ${hxx_sources})

  file(GLOB_RECURSE cxx_sources "${folder}/*.cxx")
  list(APPEND sources ${cxx_sources})

  source_group(TREE "${folder}/../" FILES ${sources})

#[[
  set(lib_source "${CMAKE_CURRENT_BINARY_DIR}/../compile_packs/${target}.pack.cpp")
  file(WRITE ${lib_source})

  foreach(source ${sources})
    target_sources(${target} PRIVATE ${source})
    set_property(SOURCE ${source} PROPERTY HEADER_FILE_ONLY TRUE)
    file(APPEND ${lib_source} "#include \"${source}\"\n")
  endforeach()

  list(APPEND packed_sources ${lib_source})
  target_sources(${target} PRIVATE ${packed_sources})
  source_group("" FILES ${lib_source})
  target_sources(${target} PRIVATE ${headers})
#]]

  target_sources(${target} PRIVATE ${headers})
  target_sources(${target} PRIVATE ${sources})
endfunction(etx_build_pack)

function(create_library folder)
  message(STATUS "Creating etx library `etx-${folder}` ...")
  set(lib_name "etx-${folder}")
  string(REPLACE "/" "-" lib_name ${lib_name})

  add_library(${lib_name} STATIC)
  etx_build_pack(${lib_name} "${CMAKE_CURRENT_LIST_DIR}/${folder}")

  target_compile_options(${lib_name} PRIVATE "/Zl")
  target_compile_options(${lib_name} PRIVATE -DNOMINMAX=1 -D_CRT_SECURE_NO_WARNINGS=1 -DWIN32_LEAN_AND_MEAN=1)

  # set_target_properties(${project} PROPERTIES
  #  RUNTIME_OUTPUT_DIRECTORY_DEBUG "${et_root}/bin"
  #  RUNTIME_OUTPUT_DIRECTORY_RELEASE "${et_root}/bin"
  #  RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${et_root}/bin"
  #  RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL "${et_root}/bin"
  # )

  set_property(TARGET ${lib_name} PROPERTY FOLDER "etx")
  set_property(TARGET ${lib_name} PROPERTY LINKER_LANGUAGE CXX)
endfunction(create_library)

create_library(core)
target_include_directories(etx-core PUBLIC ..)

create_library(log)
target_link_libraries(etx-log PRIVATE etx-core)

create_library(rt)
target_link_libraries(etx-rt PRIVATE etx-core)
