function(etx_build_pack target folder)
  if(NOT IS_DIRECTORY "${folder}")
    message(ERROR "Unable to find folder ${folder}")
    return()
  endif()

  file(GLOB_RECURSE hxx_sources "${folder}/*.hxx")
  list(APPEND sources ${hxx_sources})

  file(GLOB_RECURSE cxx_sources "${folder}/*.cxx")
  list(APPEND sources ${cxx_sources})
  
  if (WIN32)
    file(GLOB_RECURSE metal_sources "${folder}/metal/*.*")
    list(REMOVE_ITEM sources ${metal_sources})
    list(REMOVE_ITEM hxx_sources ${metal_sources})
    list(REMOVE_ITEM cxx_sources ${metal_sources})
  elseif(APPLE)
    file(GLOB_RECURSE vulkan_sources "${folder}/vulkan/*.*")
    list(REMOVE_ITEM sources ${vulkan_sources})
    list(REMOVE_ITEM hxx_sources ${vulkan_sources})
    list(REMOVE_ITEM cxx_sources ${vulkan_sources})
  endif()

  source_group(TREE "${folder}/../" FILES ${sources})

  set(lib_source "${CMAKE_CURRENT_BINARY_DIR}/../compile_packs/${target}.pack.cpp")
  set(unity_std_header "${CMAKE_CURRENT_LIST_DIR}/std.hxx")
  file(WRITE ${lib_source} "#include \"${unity_std_header}\"\n")

  foreach(source ${cxx_sources})
    target_sources(${target} PRIVATE ${source})
    set_property(SOURCE ${source} PROPERTY HEADER_FILE_ONLY TRUE)
    file(APPEND ${lib_source} "#include \"${source}\"\n")
  endforeach()

  list(APPEND packed_sources ${lib_source})
  target_sources(${target} PRIVATE ${hxx_sources})
  target_sources(${target} PRIVATE ${packed_sources})
  source_group("" FILES ${lib_source})
  target_sources(${target} PRIVATE ${headers})

endfunction(etx_build_pack)

function(create_library folder)
  message(STATUS "Creating etx library `etx-${folder}` ...")
  set(lib_name "etx-${folder}")
  string(REPLACE "/" "-" lib_name ${lib_name})

  add_library(${lib_name} STATIC)
  etx_build_pack(${lib_name} "${CMAKE_CURRENT_LIST_DIR}/${folder}")

  if(MSVC)
    target_compile_options(${lib_name} PRIVATE -DNOMINMAX=1 -D_CRT_SECURE_NO_WARNINGS=1 -DWIN32_LEAN_AND_MEAN=1 -D_USE_MATH_DEFINES=1 -DETX_LIBRARY=1)
    target_link_options(${lib_name} PRIVATE "/NODEFAULTLIB:LIBCMT")
  else()
    target_compile_options(${lib_name} PRIVATE -D_USE_MATH_DEFINES=1 -DETX_LIBRARY=1)
  endif()

  set_target_properties(${lib_name} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${root_folder}/bin"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${root_folder}/bin"
    RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${root_folder}/bin"
    RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL "${root_folder}/bin"
    LIBRARY_OUTPUT_DIRECTORY_DEBUG "${root_folder}/bin"
    LIBRARY_OUTPUT_DIRECTORY_RELEASE "${root_folder}/bin"
    LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO "${root_folder}/bin"
    LIBRARY_OUTPUT_DIRECTORY_MINSIZEREL "${root_folder}/bin"
  )

  set_property(TARGET ${lib_name} PROPERTY FOLDER "etx")
  set_property(TARGET ${lib_name} PROPERTY LINKER_LANGUAGE CXX)
endfunction(create_library)

create_library(core)
target_link_libraries(etx-core PUBLIC json nativefiledialog microprofile xxhash)
target_include_directories(etx-core PUBLIC ..)

create_library(render)
target_link_libraries(etx-render PRIVATE etx-core stb_image tinyexr tiny_obj_loader tiny_gltf mikktspace enkiTS nanovdb oidn bcdec)
target_link_libraries(etx-render PUBLIC bcdec)
target_compile_definitions(etx-render PUBLIC -DETX_HAVE_OPENVDB=1)
target_include_directories(etx-render PUBLIC "${CMAKE_SOURCE_DIR}/sources/etx/render")

create_library(util)
target_link_libraries(etx-util PRIVATE etx-core etx-render)

create_library(rhi)
target_link_libraries(etx-rhi PRIVATE etx-core imgui sokol)
target_compile_definitions(etx-rhi PRIVATE ETX_RHI_INTERNAL=1)
target_include_directories(etx-rhi PUBLIC "${CMAKE_SOURCE_DIR}/sources/etx/render")

# Add Vulkan dependency for Windows
if(WIN32)
  find_package(Vulkan REQUIRED)
  target_link_libraries(etx-rhi PRIVATE Vulkan::Vulkan)
  target_include_directories(etx-rhi PRIVATE ${Vulkan_INCLUDE_DIRS})
endif()

find_package(embree 4.0 REQUIRED)

create_library(rt)
target_include_directories(etx-rt PUBLIC ${EMBREE_INCLUDE_DIR})
target_link_libraries(etx-rt PRIVATE ${EMBREE_LIBRARY} etx-core etx-render bluenoise)

if (WIN32)
  message("Copying ${EMBREE_LIBRARY} ... to ${root_folder}/bin/")
  # Derive bin directory from the found library path
  get_filename_component(EMBREE_LIB_DIR "${EMBREE_LIBRARY}" DIRECTORY)
  get_filename_component(EMBREE_ROOT_DIR "${EMBREE_LIB_DIR}" DIRECTORY)
  set(EMBREE_BIN_DIR "${EMBREE_ROOT_DIR}/bin")

  configure_file("${EMBREE_BIN_DIR}/embree4.dll" "${root_folder}/bin/" COPYONLY)
  configure_file("${EMBREE_BIN_DIR}/tbb12.dll" "${root_folder}/bin/" COPYONLY)
endif()

