#include <etx/std.hxx>
#include <etx/core/environment.hxx>
#include "importer_plugin.hxx"
#include "scene_tungsten_dependencies.hxx"
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
    return is_tungsten_document(json) ? 1 : 0;
  } catch (...) {
    return 0;
  }
}

uint32_t decode(const char* source, const char*, SceneData& data, const IORDatabase& database, TaskScheduler& scheduler, Camera& camera) {
  return load_from_tungsten_file(source, data, database, scheduler, camera);
}

void configure_scene(const char* source, SceneRepresentation& scene) {
  std::ifstream stream(std::filesystem::u8path(source));
  const auto json = nlohmann::json::parse(stream);
  auto integrator = scene.integrator_data();
  if (json.contains("integrator") && json["integrator"].is_object()) {
    const auto& settings = json["integrator"];
    const std::string type = settings.value("type", "path_tracer");
    integrator.selected = type == "bidirectional_path_tracer"                       ? Integrator::Type::Bidirectional
                          : ((type == "vcm") || (type == "progressive_photon_map")) ? Integrator::Type::VCM
                          : type == "debug"                                         ? Integrator::Type::Debug
                                                                                    : Integrator::Type::PathTracing;
    scene.data().options.min_path_length = static_cast<uint32_t>(std::max<int64_t>(0, settings.value("min_bounces", int64_t(1))));
    scene.data().options.max_path_length = static_cast<uint32_t>(std::max<int64_t>(0, settings.value("max_bounces", int64_t(8))));
  }
  if (json.contains("renderer") && json["renderer"].is_object())
    scene.data().options.samples = static_cast<uint32_t>(std::max<int64_t>(1, json["renderer"].value("spp", int64_t(1))));
  scene.set_integrator_data(integrator);
}

}  // namespace

const ImporterFormat kImporterFormat = {"etx.tungsten", "Tungsten", "json", probe, inspect_tungsten_dependencies, decode, true, configure_scene};

}  // namespace etx
