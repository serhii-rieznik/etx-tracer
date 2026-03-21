#include <etx/render/host/emitter_packing.hxx>

#include <etx/core/profiler.hxx>

#include <cmath>

namespace etx {
namespace {

float safe_spectrum_luminance(const SceneData& scene_data, uint32_t spectrum_index) {
  if (spectrum_index >= static_cast<uint32_t>(scene_data.spectrum_values.size())) {
    return 0.0f;
  }
  return scene_data.spectrum_values[spectrum_index].luminance();
}

bool triangle_has_valid_positions(const SceneData& scene_data, const Triangle& triangle) {
  const bool i0_valid = (triangle.i[0] < scene_data.vertices.pos.size());
  const bool i1_valid = (triangle.i[1] < scene_data.vertices.pos.size());
  const bool i2_valid = (triangle.i[2] < scene_data.vertices.pos.size());
  return i0_valid && i1_valid && i2_valid;
}

void update_directional_profile_data(std::vector<EmitterProfile>& emitter_profiles) {
  for (auto& profile : emitter_profiles) {
    if (profile.cls == EmitterProfile::Class::Directional) {
      profile.directional.equivalent_disk_size = 2.0f * std::tan(profile.directional.angular_size * 0.5f);
      profile.directional.angular_size_cosine = std::cos(profile.directional.angular_size * 0.5f);
    }
  }
}

void collect_active_and_environment_emitters(PackedEmitterData& result) {
  result.active_emitter_indices.clear();
  result.active_emitter_indices.reserve(result.emitter_instances.size());
  result.environment_emitters.count = 0u;

  for (uint32_t emitter_index = 0u; emitter_index < static_cast<uint32_t>(result.emitter_instances.size()); ++emitter_index) {
    const auto& emitter = result.emitter_instances[emitter_index];
    const float total_weight = emitter.spectrum_weight * emitter.additional_weight;
    if (total_weight <= 0.0f) {
      continue;
    }

    result.active_emitter_indices.push_back(emitter_index);

    const bool is_directional = (emitter.cls == EmitterProfile::Class::Directional);
    const bool is_environment = (emitter.cls == EmitterProfile::Class::Environment);
    if ((is_directional || is_environment) && (result.environment_emitters.count < SceneLimits::MaxEnvironmentEmitters)) {
      result.environment_emitters.emitters[result.environment_emitters.count++] = emitter_index;
    }
  }
}

}  // namespace

PackedEmitterData build_packed_emitters(const SceneData& scene_data) {
  ETX_PROFILER_SCOPE();

  PackedEmitterData result = {};
  result.triangles = scene_data.triangles;
  const BoundingBox bbox = scene_data.compute_bounding_volumes();
  const float3 bounding_sphere_center = 0.5f * (bbox.p_min + bbox.p_max);
  const float bounding_sphere_radius = length(bbox.p_max - bounding_sphere_center);

  {
    ETX_PROFILER_NAMED_SCOPE("pack_emitters_reset_triangle_emitter_indices");
    for (auto& triangle : result.triangles) {
      triangle.emitter_index = kInvalidIndex;
    }
  }

  result.emitter_profiles = scene_data.emitter_profiles;
  update_directional_profile_data(result.emitter_profiles);

  {
    ETX_PROFILER_NAMED_SCOPE("pack_emitters_non_area");
    for (uint32_t profile_index = 0u; profile_index < static_cast<uint32_t>(result.emitter_profiles.size()); ++profile_index) {
      const auto& profile = result.emitter_profiles[profile_index];
      if (profile.cls == EmitterProfile::Class::Area) {
        continue;
      }

      Emitter emitter(profile.cls);
      emitter.profile = profile_index;
      emitter.triangle_index = kInvalidIndex;
      if (profile.emission.spectrum_index != kInvalidIndex) {
        emitter.spectrum_weight = safe_spectrum_luminance(scene_data, profile.emission.spectrum_index);
      }
      emitter.additional_weight =
        ((profile.cls == EmitterProfile::Class::Directional) || (profile.cls == EmitterProfile::Class::Environment)) ? (kPi * bounding_sphere_radius * bounding_sphere_radius) :
                                                                                                                          (4.0f * kPi);
      result.emitter_instances.push_back(emitter);
    }
  }

  {
    ETX_PROFILER_NAMED_SCOPE("pack_emitters_area");
    for (uint32_t triangle_index = 0u; triangle_index < static_cast<uint32_t>(scene_data.triangles.size()); ++triangle_index) {
      const Triangle& triangle = scene_data.triangles[triangle_index];
      if (triangle.emitter_index == kInvalidIndex) {
        continue;
      }
      if (triangle.emitter_index >= static_cast<uint32_t>(result.emitter_profiles.size())) {
        continue;
      }

      const auto& profile = result.emitter_profiles[triangle.emitter_index];
      if (profile.cls != EmitterProfile::Class::Area) {
        continue;
      }

      Emitter emitter(EmitterProfile::Class::Area);
      emitter.profile = triangle.emitter_index;
      emitter.triangle_index = triangle_index;

      if ((triangle.material_index < static_cast<uint32_t>(scene_data.materials.size())) && triangle_has_valid_positions(scene_data, triangle)) {
        const auto& material = scene_data.materials[triangle.material_index];

        if (profile.emission.spectrum_index != kInvalidIndex) {
          emitter.spectrum_weight = safe_spectrum_luminance(scene_data, profile.emission.spectrum_index);
        }

        const float3& v0 = scene_data.vertices.pos[triangle.i[0]];
        const float3& v1 = scene_data.vertices.pos[triangle.i[1]];
        const float3& v2 = scene_data.vertices.pos[triangle.i[2]];
        const float triangle_area = 0.5f * length(cross(v1 - v0, v2 - v0));
        emitter.triangle_area = triangle_area;
        emitter.additional_weight = (material.two_sided ? 2.0f : 1.0f) * triangle_area * kPi;
      }

      result.emitter_instances.push_back(emitter);
      result.triangles[triangle_index].emitter_index = static_cast<uint32_t>(result.emitter_instances.size() - 1u);
    }
  }

  collect_active_and_environment_emitters(result);
  return result;
}

uint32_t fill_packed_emitter_distribution_entries(const std::vector<Emitter>& emitter_instances, const std::vector<uint32_t>& active_emitter_indices,
  Distribution::Entry* entries, uint32_t entry_capacity) {
  const uint32_t active_count = static_cast<uint32_t>(active_emitter_indices.size());
  if ((entries == nullptr) || (entry_capacity < active_count)) {
    return 0u;
  }

  for (uint32_t i = 0u; i < active_count; ++i) {
    const uint32_t emitter_index = active_emitter_indices[i];
    const auto& emitter = emitter_instances[emitter_index];
    const float total_weight = emitter.spectrum_weight * emitter.additional_weight;
    entries[i] = {total_weight, 0.0f, 0.0f, emitter_index};
  }

  return active_count;
}

std::vector<Distribution::Entry> build_packed_emitter_distribution(const PackedEmitterData& packed_emitters) {
  ETX_PROFILER_SCOPE();

  const uint32_t active_count = static_cast<uint32_t>(packed_emitters.active_emitter_indices.size());
  if (active_count == 0u) {
    return {};
  }

  std::vector<Distribution::Entry> entries(active_count + 1u);
  const uint32_t written_count =
    fill_packed_emitter_distribution_entries(packed_emitters.emitter_instances, packed_emitters.active_emitter_indices, entries.data(), active_count);
  if (written_count != active_count) {
    return {};
  }

  Distribution::build(entries.data(), active_count);
  entries[active_count].reference = kInvalidIndex;

  return entries;
}

}  // namespace etx
