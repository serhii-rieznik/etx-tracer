add_library(etx-import STATIC
  ../etx/import/import_service.cxx
  ../etx/import/scene_dependencies.cxx
)
target_link_libraries(etx-import PUBLIC etx-render etx-core json)
target_compile_options(etx-import PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/FI${CMAKE_SOURCE_DIR}/sources/etx/std.hxx> $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-include${CMAKE_SOURCE_DIR}/sources/etx/std.hxx>)
if(WIN32)
  target_compile_definitions(etx-import PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
else()
  target_link_libraries(etx-import PRIVATE ${CMAKE_DL_LIBS})
endif()

add_library(etx-source-decoders STATIC scene_obj_loader.cxx scene_gltf_loader.cxx scene_tungsten_loader.cxx scene_tungsten_dependencies.cxx scene_pbrt_input.cxx scene_pbrt_parser.cxx scene_pbrt_mesh.cxx scene_pbrt_subdivision.cxx scene_pbrt_spectrum.cxx scene_pbrt_loader.cxx scene_mesh_loader.cxx scene_ply_loader.cxx scene_stl_loader.cxx)
target_link_libraries(etx-source-decoders PUBLIC etx-render etx-core etx-import tiny_obj_loader tiny_gltf)
target_link_libraries(etx-source-decoders PRIVATE Blosc::blosc)
target_compile_definitions(etx-source-decoders PRIVATE NOMINMAX _CRT_SECURE_NO_WARNINGS _USE_MATH_DEFINES)
foreach(format obj gltf tungsten legacy pbrt_v3 pbrt_v4 ply stl)
  add_library(etx-import-${format} MODULE importer_plugin.cxx importer_${format}.cxx)
  target_compile_definitions(etx-import-${format} PRIVATE NOMINMAX _CRT_SECURE_NO_WARNINGS WIN32_LEAN_AND_MEAN)
  target_link_libraries(etx-import-${format} PRIVATE etx-source-decoders etx-import etx-render etx-rt etx-engine etx-rhi etx-core stb_image bluenoise)
  set_target_properties(etx-import-${format} PROPERTIES PREFIX "" FOLDER "importers")
  foreach(config DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
    set_target_properties(etx-import-${format} PROPERTIES RUNTIME_OUTPUT_DIRECTORY_${config} "${root_folder}/bin/importers" LIBRARY_OUTPUT_DIRECTORY_${config} "${root_folder}/bin/importers")
  endforeach()
  if(APPLE)
    target_link_libraries(etx-import-${format} PRIVATE "-framework Foundation" "-framework QuartzCore" "-framework CoreGraphics" "-framework Metal" "-framework MetalKit" "-framework AppKit")
    set_target_properties(etx-import-${format} PROPERTIES SUFFIX ".dylib" BUILD_RPATH "@loader_path/../lib;@loader_path/../../Frameworks")
  endif()
endforeach()

add_custom_command(TARGET etx-import-legacy POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E rm -f "${root_folder}/bin/importers/etx-import-json${CMAKE_SHARED_LIBRARY_SUFFIX}"
  VERBATIM
)
