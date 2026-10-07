#include <etx/std.hxx>
#include <etx/core/environment.hxx>
#include "importer_plugin.hxx"
#include "scene_mesh_loader.hxx"

namespace etx {
namespace {

int32_t ETX_IMPORT_CALL probe(const char* source) {
  return (source != nullptr) && (_stricmp(get_file_ext(source), ".stl") == 0) ? 1 : 0;
}

uint32_t decode(const char* source, const char*, SceneData& data, const IORDatabase&, TaskScheduler&, Camera&) {
  const auto path = std::filesystem::u8path(source);
  return load_mesh_document(load_stl_mesh(path), path, data);
}

}  // namespace

const ImporterFormat kImporterFormat = {"etx.stl", "STL Mesh", "stl", probe, inspect_scene_dependencies, decode, false, nullptr};

}  // namespace etx
