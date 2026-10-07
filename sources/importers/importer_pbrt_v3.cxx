#include <etx/std.hxx>
#include "importer_plugin.hxx"
#include "scene_pbrt_loader.hxx"

namespace etx {
namespace {
uint32_t decode(const char* source, const char*, SceneData& data, const IORDatabase& database, TaskScheduler& scheduler, Camera& camera) {
  return load_pbrt_file(source, data, database, scheduler, camera, PbrtVersion::V3);
}
void configure(const char* source, SceneRepresentation& scene) {
  configure_pbrt_scene(source, scene, PbrtVersion::V3);
}
}  // namespace

const ImporterFormat kImporterFormat = {"etx.pbrt-v3", "PBRT v3", "pbrt,gz", probe_pbrt_file, inspect_pbrt_dependencies, decode, true, configure, true};
}  // namespace etx
