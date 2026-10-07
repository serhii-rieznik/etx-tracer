#pragma once

#include "scene_pbrt_parser.hxx"
#include <etx/import/scene_dependencies.hxx>
#include <etx/render/host/scene_representation.hxx>

namespace etx {

int32_t probe_pbrt_file(const char* source);
SceneDependencyInspection inspect_pbrt_dependencies(const std::filesystem::path& file, std::string_view relative_path);
uint32_t load_pbrt_file(const char* source, SceneData& data, const IORDatabase& database, TaskScheduler& scheduler, Camera& camera, PbrtVersion version);
void configure_pbrt_scene(const char* source, SceneRepresentation& scene, PbrtVersion version);

}  // namespace etx
