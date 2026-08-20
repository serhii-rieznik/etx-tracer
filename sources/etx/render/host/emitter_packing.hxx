#pragma once

#include <etx/render/host/scene_data.hxx>

namespace etx {

struct PackedEmitterData {
  std::vector<Triangle> triangles = {};
  std::vector<EmitterProfile> emitter_profiles = {};
  std::vector<Emitter> emitter_instances = {};
  std::vector<SceneInstance> instances = {};
  std::vector<uint32_t> active_emitter_indices = {};
  Scene::EnvironmentEmitters environment_emitters = {};
};

struct PackedEmitterTopology {
  std::vector<uint32_t> area_triangle_indices = {};
  std::vector<uint32_t> mesh_area_triangle_offsets = {};
};

PackedEmitterData build_packed_emitters(const SceneData& scene_data);
PackedEmitterData build_packed_emitters(const SceneData& scene_data, PackedEmitterTopology& topology);
PackedEmitterData build_packed_emitters_for_transforms(const SceneData& scene_data, const PackedEmitterTopology& topology);

uint32_t fill_packed_emitter_distribution_entries(const std::vector<Emitter>& emitter_instances, const std::vector<uint32_t>& active_emitter_indices, Distribution::Entry* entries,
  uint32_t entry_capacity);

std::vector<Distribution::Entry> build_packed_emitter_distribution(const PackedEmitterData& packed_emitters);

}  // namespace etx
