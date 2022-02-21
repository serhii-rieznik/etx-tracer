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
  target_compile_options(${lib_name} PRIVATE -DNOMINMAX=1 -D_CRT_SECURE_NO_WARNINGS=1 -DWIN32_LEAN_AND_MEAN=1 -D_USE_MATH_DEFINES=1)

  set_property(TARGET ${lib_name} PROPERTY FOLDER "etx")
  set_property(TARGET ${lib_name} PROPERTY LINKER_LANGUAGE CXX)
endfunction(create_library)

create_library(core)
target_link_libraries(etx-core PUBLIC jansson)
target_include_directories(etx-core PUBLIC .. ${JANSSON_INCLUDE_DIRS})

create_library(log)
target_link_libraries(etx-log PRIVATE etx-core)

create_library(render)
target_link_libraries(etx-render PRIVATE glm stb_image tinyexr tiny_obj_loader mikktspace enkiTS etx-core)
target_include_directories(etx-render PRIVATE ${JANSSON_INCLUDE_DIRS})

if (WITH_OPENVDB)
  target_link_libraries(etx-render PUBLIC openvdb)
  target_compile_definitions(etx-render PUBLIC -DETX_HAVE_OPENVDB=1)
endif()


find_package(embree 3.0 REQUIRED)

create_library(rt)
target_link_libraries(etx-rt PRIVATE glm embree etx-core etx-render)

if(DEFINED ENV{OptiX_INSTALL_DIR})

  find_package(CUDA REQUIRED)
  find_package(OptiX REQUIRED)
  create_library(gpu)
  target_link_libraries(etx-gpu PRIVATE etx-core etx-log)
  target_include_directories(etx-gpu PUBLIC ${OptiX_INCLUDE} ${CUDA_TOOLKIT_ROOT_DIR}/include)
  target_link_libraries(etx-gpu PUBLIC tlsf ${optix_LIBRARY} ${CUDA_LIBRARIES} ${CUDA_CUDA_LIBRARY})

else()
  message(WARNING "OptiX_INSTALL_DIR environment variable not found, GPU rendering will not be available.")
endif()
