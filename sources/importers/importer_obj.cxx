#include <etx/std.hxx>
#include <etx/core/environment.hxx>
#include "importer_plugin.hxx"
#include "scene_obj_loader.hxx"

namespace etx {
namespace {

int32_t ETX_IMPORT_CALL probe(const char* source) {
  return (source != nullptr) && (_stricmp(get_file_ext(source), ".obj") == 0) ? 1 : 0;
}

uint32_t decode(const char* source, const char* materials, SceneData& data, const IORDatabase& database, TaskScheduler& scheduler, Camera&) {
  return load_from_obj_file(source, materials, data, database, scheduler);
}

}  // namespace

const ImporterFormat kImporterFormat = {"etx.obj", "Wavefront OBJ", "obj", probe, inspect_scene_dependencies, decode, false, nullptr};

}  // namespace etx
