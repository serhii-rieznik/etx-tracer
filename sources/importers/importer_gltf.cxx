#include <etx/std.hxx>
#include <etx/core/environment.hxx>
#include "importer_plugin.hxx"
#include "scene_gltf_loader.hxx"

namespace etx {
namespace {

int32_t ETX_IMPORT_CALL probe(const char* source) {
  if (source == nullptr)
    return 0;
  const char* extension = get_file_ext(source);
  return ((_stricmp(extension, ".gltf") == 0) || (_stricmp(extension, ".glb") == 0)) ? 1 : 0;
}

uint32_t decode(const char* source, const char*, SceneData& data, const IORDatabase&, TaskScheduler& scheduler, Camera& camera) {
  return load_from_gltf_file(source, _stricmp(get_file_ext(source), ".glb") == 0, data, scheduler, camera);
}

}  // namespace

const ImporterFormat kImporterFormat = {"etx.gltf", "glTF 2.0", "gltf,glb", probe, inspect_scene_dependencies, decode, false, nullptr};

}  // namespace etx
