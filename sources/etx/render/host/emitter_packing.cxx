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

float4 normalize_quaternion(const float4& value) {
  const float length_squared = dot(value, value);
  return (length_squared > kEpsilon) ? (value / std::sqrt(length_squared)) : float4{0.0f, 0.0f, 0.0f, 1.0f};
}

float4 multiply_quaternions(const float4& a, const float4& b) {
  return {
    a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
    a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
    a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
    a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
  };
}

float4 quaternion_from_orientation(const AffineTransform& orientation) {
  const float m00 = orientation.rows[0].x;
  const float m11 = orientation.rows[1].y;
  const float m22 = orientation.rows[2].z;
  float4 result = {};
  const float trace = m00 + m11 + m22;
  if (trace > 0.0f) {
    const float scale = 2.0f * std::sqrt(trace + 1.0f);
    result = {(orientation.rows[2].y - orientation.rows[1].z) / scale, (orientation.rows[0].z - orientation.rows[2].x) / scale,
      (orientation.rows[1].x - orientation.rows[0].y) / scale, 0.25f * scale};
  } else if ((m00 > m11) && (m00 > m22)) {
    const float scale = 2.0f * std::sqrt(1.0f + m00 - m11 - m22);
    result = {0.25f * scale, (orientation.rows[0].y + orientation.rows[1].x) / scale, (orientation.rows[0].z + orientation.rows[2].x) / scale,
      (orientation.rows[2].y - orientation.rows[1].z) / scale};
  } else if (m11 > m22) {
    const float scale = 2.0f * std::sqrt(1.0f + m11 - m00 - m22);
    result = {(orientation.rows[0].y + orientation.rows[1].x) / scale, 0.25f * scale, (orientation.rows[1].z + orientation.rows[2].y) / scale,
      (orientation.rows[0].z - orientation.rows[2].x) / scale};
  } else {
    const float scale = 2.0f * std::sqrt(1.0f + m22 - m00 - m11);
    result = {(orientation.rows[0].z + orientation.rows[2].x) / scale, (orientation.rows[1].z + orientation.rows[2].y) / scale, 0.25f * scale,
      (orientation.rows[1].x - orientation.rows[0].y) / scale};
  }
  return normalize_quaternion(result);
}

void update_directional_profile_data(std::vector<EmitterProfile>& emitter_profiles) {
  for (auto& profile : emitter_profiles) {
    if (profile.cls == EmitterProfile::Class::Directional) {
      profile.directional.equivalent_disk_size = 2.0f * std::tan(profile.directional.angular_size * 0.5f);
      profile.directional.angular_size_cosine = std::cos(profile.directional.angular_size * 0.5f);
    } else if (profile.cls == EmitterProfile::Class::Environment) {
      profile.set_environment_rotation(normalize_quaternion(profile.environment_rotation()));
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

PackedEmitterTopology build_packed_emitter_topology(const SceneData& scene_data) {
  PackedEmitterTopology result = {};
  result.mesh_area_triangle_offsets.resize(scene_data.meshes.size() + 1u);

  for (uint32_t mesh_index = 0u; mesh_index < static_cast<uint32_t>(scene_data.meshes.size()); ++mesh_index) {
    result.mesh_area_triangle_offsets[mesh_index] = static_cast<uint32_t>(result.area_triangle_indices.size());
    const Mesh& mesh = scene_data.meshes[mesh_index];
    const uint32_t triangle_end = mesh.triangle_offset + mesh.triangle_count;
    if (triangle_end > scene_data.triangles.size()) {
      continue;
    }
    for (uint32_t triangle_index = mesh.triangle_offset; triangle_index < triangle_end; ++triangle_index) {
      const Triangle& triangle = scene_data.triangles[triangle_index];
      if ((triangle.emitter_index >= scene_data.emitter_profiles.size()) || (scene_data.emitter_profiles[triangle.emitter_index].cls != EmitterProfile::Class::Area)) {
        continue;
      }
      result.area_triangle_indices.push_back(triangle_index);
    }
  }

  result.mesh_area_triangle_offsets.back() = static_cast<uint32_t>(result.area_triangle_indices.size());
  return result;
}

PackedEmitterData build_packed_emitters_impl(const SceneData& scene_data, const PackedEmitterTopology& topology, bool include_triangles) {
  ETX_PROFILER_SCOPE();

  PackedEmitterData result = {};
  if (include_triangles) {
    result.triangles = scene_data.triangles;
  }
  const BoundingBox bbox = scene_data.compute_bounding_volumes();
  const float3 bounding_sphere_center = 0.5f * (bbox.p_min + bbox.p_max);
  const float bounding_sphere_radius = length(bbox.p_max - bounding_sphere_center);

  {
    ETX_PROFILER_NAMED_SCOPE("pack_emitters_reset_triangle_emitter_indices");
    if (include_triangles) {
      for (auto& triangle : result.triangles) {
        triangle.emitter_index = kInvalidIndex;
      }
    }
  }

  result.emitter_profiles = scene_data.emitter_profiles;
  update_directional_profile_data(result.emitter_profiles);

  {
    ETX_PROFILER_NAMED_SCOPE("pack_emitters_non_area");
    std::vector<bool> attached_profiles(result.emitter_profiles.size(), false);
    for (uint32_t node_index : scene_data.hierarchy.evaluation_order) {
      const SceneNode& node = scene_data.hierarchy.nodes[node_index];
      const uint32_t attachment_end = node.attachment_offset + node.attachment_count;
      if ((attachment_end > scene_data.hierarchy.attachments.size()) || (node_index >= scene_data.hierarchy.world_transforms.size())) {
        continue;
      }
      for (uint32_t attachment_index = node.attachment_offset; attachment_index < attachment_end; ++attachment_index) {
        const SceneAttachment& attachment = scene_data.hierarchy.attachments[attachment_index];
        if ((attachment.type != SceneAttachment::Type::Emitter) || (attachment.resource_index >= attached_profiles.size())) {
          continue;
        }
        attached_profiles[attachment.resource_index] = true;
        if ((node_index >= scene_data.hierarchy.effective_enabled.size()) || (scene_data.hierarchy.effective_enabled[node_index] == 0u)) {
          continue;
        }
        const EmitterProfile& source_profile = result.emitter_profiles[attachment.resource_index];
        if (source_profile.cls == EmitterProfile::Class::Area) {
          continue;
        }

        EmitterProfile profile = source_profile;
        if ((node_index >= scene_data.hierarchy.orientation_valid.size()) || (scene_data.hierarchy.orientation_valid[node_index] == 0u) ||
            (node_index >= scene_data.hierarchy.world_orientations.size())) {
          log::warning("Distant emitter %u is attached below a sheared or singular node and was skipped", attachment.resource_index);
          continue;
        }
        const AffineTransform& world_orientation = scene_data.hierarchy.world_orientations[node_index];
        if (profile.cls == EmitterProfile::Class::Directional) {
          const float3 transformed_direction = transform_vector(world_orientation, profile.directional.direction);
          if (dot(transformed_direction, transformed_direction) <= kEpsilon) {
            log::warning("Directional emitter %u has a degenerate direction and was skipped", attachment.resource_index);
            continue;
          }
          profile.directional.direction = normalize(transformed_direction);
        } else if (profile.cls == EmitterProfile::Class::Environment) {
          const float4 node_rotation = quaternion_from_orientation(world_orientation);
          profile.set_environment_rotation(normalize_quaternion(multiply_quaternions(node_rotation, profile.environment_rotation())));
        }
        const uint32_t packed_profile_index = static_cast<uint32_t>(result.emitter_profiles.size());
        result.emitter_profiles.push_back(profile);

        Emitter emitter(profile.cls);
        emitter.profile = packed_profile_index;
        emitter.triangle_index = kInvalidIndex;
        if (profile.emission.spectrum_index != kInvalidIndex) {
          emitter.spectrum_weight = safe_spectrum_luminance(scene_data, profile.emission.spectrum_index);
        }
        emitter.additional_weight = ((profile.cls == EmitterProfile::Class::Directional) || (profile.cls == EmitterProfile::Class::Environment))
                                      ? (kPi * bounding_sphere_radius * bounding_sphere_radius)
                                      : (4.0f * kPi);
        result.emitter_instances.push_back(emitter);
      }
    }

    for (uint32_t profile_index = 0u; profile_index < static_cast<uint32_t>(attached_profiles.size()); ++profile_index) {
      if (attached_profiles[profile_index]) {
        continue;
      }
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
      emitter.additional_weight = ((profile.cls == EmitterProfile::Class::Directional) || (profile.cls == EmitterProfile::Class::Environment))
                                    ? (kPi * bounding_sphere_radius * bounding_sphere_radius)
                                    : (4.0f * kPi);
      result.emitter_instances.push_back(emitter);
    }
  }

  {
    ETX_PROFILER_NAMED_SCOPE("pack_emitters_area");
    result.instances.reserve(scene_data.hierarchy.mesh_instances.size());
    for (uint32_t instance_index = 0u; instance_index < static_cast<uint32_t>(scene_data.hierarchy.mesh_instances.size()); ++instance_index) {
      const ResolvedMeshInstance& resolved = scene_data.hierarchy.mesh_instances[instance_index];
      SceneInstance& instance = result.instances.emplace_back();
      instance.object_to_world = resolved.object_to_world;
      instance.world_to_object = resolved.world_to_object;
      instance.mesh_index = resolved.mesh_index;
      instance.flags = resolved.flags;
      instance.emitter_offset = static_cast<uint32_t>(result.emitter_instances.size());

      if ((resolved.flags & ResolvedMeshInstance::Enabled) == 0u) {
        continue;
      }

      if (resolved.mesh_index >= scene_data.meshes.size()) {
        continue;
      }

      if ((resolved.mesh_index + 1u) >= topology.mesh_area_triangle_offsets.size()) {
        continue;
      }

      const uint32_t triangle_begin = topology.mesh_area_triangle_offsets[resolved.mesh_index];
      const uint32_t triangle_end = topology.mesh_area_triangle_offsets[resolved.mesh_index + 1u];
      if ((triangle_end < triangle_begin) || (triangle_end > topology.area_triangle_indices.size())) {
        continue;
      }
      for (uint32_t area_triangle_index = triangle_begin; area_triangle_index < triangle_end; ++area_triangle_index) {
        const uint32_t triangle_index = topology.area_triangle_indices[area_triangle_index];
        if (triangle_index >= scene_data.triangles.size()) {
          continue;
        }
        const Triangle& triangle = scene_data.triangles[triangle_index];
        if ((triangle.emitter_index == kInvalidIndex) || (triangle.emitter_index >= static_cast<uint32_t>(result.emitter_profiles.size()))) {
          continue;
        }

        const auto& profile = result.emitter_profiles[triangle.emitter_index];
        if (profile.cls != EmitterProfile::Class::Area) {
          continue;
        }

        Emitter emitter(EmitterProfile::Class::Area);
        emitter.profile = triangle.emitter_index;
        emitter.triangle_index = triangle_index;
        emitter.instance_index = instance_index;

        if ((triangle.material_index < static_cast<uint32_t>(scene_data.materials.size())) && triangle_has_valid_positions(scene_data, triangle)) {
          const auto& material = scene_data.materials[triangle.material_index];
          if (profile.emission.spectrum_index != kInvalidIndex) {
            emitter.spectrum_weight = safe_spectrum_luminance(scene_data, profile.emission.spectrum_index);
          }

          const float3 v0 = transform_point(resolved.object_to_world, scene_data.vertices.pos[triangle.i[0]]);
          const float3 v1 = transform_point(resolved.object_to_world, scene_data.vertices.pos[triangle.i[1]]);
          const float3 v2 = transform_point(resolved.object_to_world, scene_data.vertices.pos[triangle.i[2]]);
          const float triangle_area = 0.5f * length(cross(v1 - v0, v2 - v0));
          emitter.triangle_area = triangle_area;
          emitter.additional_weight = (material.two_sided ? 2.0f : 1.0f) * triangle_area * kPi;
        }

        result.emitter_instances.push_back(emitter);
      }

      instance.emitter_count = static_cast<uint32_t>(result.emitter_instances.size()) - instance.emitter_offset;
    }
  }

  collect_active_and_environment_emitters(result);
  return result;
}

}  // namespace

PackedEmitterData build_packed_emitters(const SceneData& scene_data) {
  PackedEmitterTopology topology = build_packed_emitter_topology(scene_data);
  return build_packed_emitters_impl(scene_data, topology, true);
}

PackedEmitterData build_packed_emitters(const SceneData& scene_data, PackedEmitterTopology& topology) {
  topology = build_packed_emitter_topology(scene_data);
  return build_packed_emitters_impl(scene_data, topology, true);
}

PackedEmitterData build_packed_emitters_for_transforms(const SceneData& scene_data, const PackedEmitterTopology& topology) {
  if (topology.mesh_area_triangle_offsets.size() != (scene_data.meshes.size() + 1u)) {
    const PackedEmitterTopology rebuilt_topology = build_packed_emitter_topology(scene_data);
    return build_packed_emitters_impl(scene_data, rebuilt_topology, false);
  }
  return build_packed_emitters_impl(scene_data, topology, false);
}

uint32_t fill_packed_emitter_distribution_entries(const std::vector<Emitter>& emitter_instances, const std::vector<uint32_t>& active_emitter_indices, Distribution::Entry* entries,
  uint32_t entry_capacity) {
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
  const uint32_t written_count = fill_packed_emitter_distribution_entries(packed_emitters.emitter_instances, packed_emitters.active_emitter_indices, entries.data(), active_count);
  if (written_count != active_count) {
    return {};
  }

  Distribution::build(entries.data(), active_count);
  entries[active_count].reference = kInvalidIndex;

  return entries;
}

}  // namespace etx
