#pragma once

#include <etx/import/importer_abi.hxx>
#include <etx/import/scene_dependencies.hxx>
#include <etx/render/host/scene_representation.hxx>

namespace etx {

struct ImporterFormat {
  const char* id;
  const char* name;
  const char* extensions;
  int32_t(ETX_IMPORT_CALL* probe)(const char* source);
  SceneDependencyInspection (*inspect_dependencies)(const std::filesystem::path& file_path, std::string_view relative_path);
  SceneRepresentation::SourceDecoder decode;
  bool decode_root;
  void (*configure_scene)(const char* source, SceneRepresentation& scene);
  bool inspect_root_only = false;
};

extern const ImporterFormat kImporterFormat;

}  // namespace etx
