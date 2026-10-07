#include <etx/std.hxx>
#include <etx/core/environment.hxx>
#include "importer_plugin.hxx"
#include "scene_tungsten_dependencies.hxx"
#include "scene_obj_loader.hxx"
#include "scene_gltf_loader.hxx"
#include "scene_tungsten_loader.hxx"
#include <json.hpp>
#include <fstream>

namespace etx {
namespace {

int32_t ETX_IMPORT_CALL probe(const char* source) {
  if ((source == nullptr) || (_stricmp(get_file_ext(source), ".json") != 0))
    return 0;
  try {
    std::ifstream stream(std::filesystem::u8path(source));
    const auto json = nlohmann::json::parse(stream, nullptr, false);
    if (json.is_discarded() || (json.is_object() == false) || json.contains("etx_document") || json.contains("bsdfs") || json.contains("asset"))
      return 0;
    if ((json.contains("geometry") == false) || (json["geometry"].is_string() == false))
      return 0;
    const std::string geometry = json["geometry"].get<std::string>();
    return ((geometry.empty() == false) && (_stricmp(get_file_ext(geometry.c_str()), ".etx") != 0)) ? 1 : 0;
  } catch (...) {
    return 0;
  }
}

uint32_t decode(const char* source, const char* materials, SceneData& data, const IORDatabase& database, TaskScheduler& scheduler, Camera& camera) {
  const char* extension = get_file_ext(source);
  if (_stricmp(extension, ".obj") == 0)
    return load_from_obj_file(source, materials, data, database, scheduler);
  if ((_stricmp(extension, ".gltf") == 0) || (_stricmp(extension, ".glb") == 0))
    return load_from_gltf_file(source, _stricmp(extension, ".glb") == 0, data, scheduler, camera);
  if (_stricmp(extension, ".json") == 0)
    return load_from_tungsten_file(source, data, database, scheduler, camera);
  return SceneLoadFailed;
}

}  // namespace

const ImporterFormat kImporterFormat = {"etx.legacy-json", "Legacy ETX Scene", "json", probe, inspect_tungsten_dependencies, decode, false, nullptr};

}  // namespace etx
