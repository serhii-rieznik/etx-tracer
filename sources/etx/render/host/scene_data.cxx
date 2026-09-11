#include <etx/render/host/scene_data.hxx>

#include <etx/core/core.hxx>
#include <etx/core/log.hxx>
#include <etx/render/host/gpu_asset_descriptor.hxx>

namespace etx {
namespace {

uint64_t hash_payload_view(const BufferPool& buffer_pool, BufferView view, uint64_t seed) {
  uint64_t result = seed;
  result = etx_hash64_continue(&view, sizeof(view), result);
  if (view.valid() == false) {
    return result;
  }

  const void* ptr = buffer_pool.map(view);
  if (ptr == nullptr) {
    return result;
  }
  return etx_hash64_continue(ptr, view.byte_size, result);
}

uint64_t hash_images_struct_and_payload(const SceneData& scene_data) {
  uint64_t result = 0u;
  for (const auto& image : scene_data.images_vector) {
    const ::Image interop_image = make_gpu_image_descriptor(image);
    result = etx_hash64_continue(&interop_image, sizeof(::Image), result);
    result = hash_payload_view(scene_data.buffer_pool, image.data, result);
    result = hash_payload_view(scene_data.buffer_pool, image.x_distributions_storage, result);
    result = hash_payload_view(scene_data.buffer_pool, image.y_distribution_storage, result);
  }
  return result;
}

uint64_t hash_mediums_struct_and_payload(const SceneData& scene_data) {
  uint64_t result = 0u;
  for (const auto& medium : scene_data.mediums_vector) {
    const ::Medium interop_medium = make_gpu_medium_descriptor(medium);
    result = etx_hash64_continue(&interop_medium, sizeof(::Medium), result);
    result = hash_payload_view(scene_data.buffer_pool, medium.density_data, result);
  }
  return result;
}

uint64_t hash_triangle_indices(const std::vector<Triangle>& triangles) {
  uint64_t result = 0u;
  for (const auto& tri : triangles) {
    result = etx_hash64_continue(tri.i, sizeof(tri.i), result);
  }
  return result;
}

uint64_t hash_hierarchy_structure(const SceneHierarchy& hierarchy) {
  const size_t node_count = hierarchy.nodes.size();
  uint64_t result = etx_hash64_continue(&node_count, sizeof(node_count), 0u);
  for (const SceneNode& node : hierarchy.nodes) {
    result = etx_hash64_continue(&node.parent_index, sizeof(node.parent_index), result);
    result = etx_hash64_continue(&node.attachment_offset, sizeof(node.attachment_offset), result);
    result = etx_hash64_continue(&node.attachment_count, sizeof(node.attachment_count), result);
  }
  return result;
}

uint64_t hash_hierarchy_transforms(const SceneHierarchy& hierarchy) {
  uint64_t result = 0u;
  for (const SceneNode& node : hierarchy.nodes) {
    result = etx_hash64_continue(&node.local_transform, sizeof(node.local_transform), result);
    result = etx_hash64_continue(&node.flags, sizeof(node.flags), result);
  }
  return result;
}

uint64_t hash_instance_transforms(const SceneHierarchy& hierarchy) {
  const size_t instance_count = hierarchy.mesh_instances.size();
  uint64_t result = etx_hash64_continue(&instance_count, sizeof(instance_count), 0u);
  for (const ResolvedMeshInstance& instance : hierarchy.mesh_instances) {
    result = etx_hash64_continue(&instance.object_to_world, sizeof(instance.object_to_world), result);
    result = etx_hash64_continue(&instance.mesh_index, sizeof(instance.mesh_index), result);
    result = etx_hash64_continue(&instance.flags, sizeof(instance.flags), result);
  }
  return result;
}

uint64_t hash_hierarchy_attachments(const SceneHierarchy& hierarchy) {
  const size_t attachment_count = hierarchy.attachments.size();
  uint64_t result = etx_hash64_continue(&attachment_count, sizeof(attachment_count), 0u);
  for (const SceneAttachment& attachment : hierarchy.attachments) {
    result = etx_hash64_continue(&attachment.type, sizeof(attachment.type), result);
    result = etx_hash64_continue(&attachment.resource_index, sizeof(attachment.resource_index), result);
    result = etx_hash64_continue(&attachment.flags, sizeof(attachment.flags), result);
  }
  return result;
}

}  // namespace

SceneBoundingSphere compute_transport_bounding_sphere(const BoundingBox& transport_bounds, const Camera& camera, const bool has_media) {
  float3 camera_extent = {};
  if ((camera.cls == Camera::Class::Perspective) && (camera.lens_radius > kEpsilon) && (camera.focal_distance > kEpsilon)) {
    camera_extent = camera.lens_radius * float3{
                                           fabsf(camera.side.x) + fabsf(camera.up.x),
                                           fabsf(camera.side.y) + fabsf(camera.up.y),
                                           fabsf(camera.side.z) + fabsf(camera.up.z),
                                         };
  }

  float3 bounds_min = min(transport_bounds.p_min, camera.position - camera_extent);
  float3 bounds_max = max(transport_bounds.p_max, camera.position + camera_extent);
  // Only media can create primary scattering vertices beyond the geometry and camera bounds.
  // Keep those vertices inside the domain used to terminate subsequent unbounded segments.
  if (has_media && (camera.cls == Camera::Class::Perspective) && (camera.clip_far > 0.0f)) {
    const float far_horizontal_scale = camera.clip_far * camera.tan_half_fov;
    const float far_vertical_scale = far_horizontal_scale / camera.aspect;
    float3 far_extent = {
      fabsf(camera.side.x * far_horizontal_scale) + fabsf(camera.up.x * far_vertical_scale),
      fabsf(camera.side.y * far_horizontal_scale) + fabsf(camera.up.y * far_vertical_scale),
      fabsf(camera.side.z * far_horizontal_scale) + fabsf(camera.up.z * far_vertical_scale),
    };
    if ((camera.lens_radius > kEpsilon) && (camera.focal_distance > kEpsilon)) {
      far_extent += camera_extent * fabsf(1.0f - camera.clip_far / camera.focal_distance);
    }
    const float3 far_center = camera.position + camera.direction * camera.clip_far;
    bounds_min = min(bounds_min, far_center - far_extent);
    bounds_max = max(bounds_max, far_center + far_extent);
  }

  const float3 center = 0.5f * (bounds_min + bounds_max);
  const float radius = length(bounds_max - center);
  const float coordinate_scale =
    max(1.0f, max(max(max(fabsf(bounds_min.x), fabsf(bounds_min.y)), fabsf(bounds_min.z)), max(max(fabsf(bounds_max.x), fabsf(bounds_max.y)), fabsf(bounds_max.z))));
  const float padding = max(kRayEpsilon, coordinate_scale * kRayEpsilon);
  const float padded_radius = radius + padding;
  // Medium transport can occupy the full termination sphere, beyond surface bounds.
  const float3 emission_half_extent = has_media ? float3{padded_radius, padded_radius, padded_radius} : bounds_max - center + float3{padding, padding, padding};
  return {center, padded_radius, emission_half_extent};
}

SceneData::SceneData(TaskScheduler& s)
  : images(images_vector, buffer_pool)
  , mediums(mediums_vector, buffer_pool, images)
  , scheduler(s) {
}

BoundingBox SceneData::compute_bounding_volumes() const {
  BoundingBox bbox = {
    float3{kMaxFloat, kMaxFloat, kMaxFloat},
    0.0f,
    float3{-kMaxFloat, -kMaxFloat, -kMaxFloat},
    0.0f,
  };

  if (hierarchy.nodes.empty() == false) {
    for (const ResolvedMeshInstance& instance : hierarchy.mesh_instances) {
      if ((instance.flags & ResolvedMeshInstance::Enabled) == 0u) {
        continue;
      }
      bbox.p_min = min(bbox.p_min, instance.bbox_min);
      bbox.p_max = max(bbox.p_max, instance.bbox_max);
    }
  } else {
    std::vector<BoundingBox> thread_bounds(scheduler.max_thread_count(), bbox);

    scheduler.execute(triangles.size(), [&](uint32_t begin, uint32_t end, uint32_t thread_id) {
      BoundingBox& local_bounds = thread_bounds[thread_id];
      for (uint32_t i = begin; i < end; ++i) {
        const auto& tri = triangles[i];
        if (tri.i[0] < vertices.pos.size()) {
          const float3& v = vertices.pos[tri.i[0]];
          local_bounds.p_min = min(local_bounds.p_min, v);
          local_bounds.p_max = max(local_bounds.p_max, v);
        }
        if (tri.i[1] < vertices.pos.size()) {
          const float3& v = vertices.pos[tri.i[1]];
          local_bounds.p_min = min(local_bounds.p_min, v);
          local_bounds.p_max = max(local_bounds.p_max, v);
        }
        if (tri.i[2] < vertices.pos.size()) {
          const float3& v = vertices.pos[tri.i[2]];
          local_bounds.p_min = min(local_bounds.p_min, v);
          local_bounds.p_max = max(local_bounds.p_max, v);
        }
      }
    });

    for (const auto& thread_bounds_value : thread_bounds) {
      bbox.p_min = min(bbox.p_min, thread_bounds_value.p_min);
      bbox.p_max = max(bbox.p_max, thread_bounds_value.p_max);
    }
  }

  const bool has_valid_bounds = (bbox.p_min.x <= bbox.p_max.x) && (bbox.p_min.y <= bbox.p_max.y) && (bbox.p_min.z <= bbox.p_max.z);
  if (has_valid_bounds == false) {
    bbox.p_min = {-1.0f, -1.0f, -1.0f};
    bbox.p_max = {1.0f, 1.0f, 1.0f};
  }

  return bbox;
}

BoundingBox SceneData::compute_transport_bounding_volumes() const {
  BoundingBox bbox = compute_bounding_volumes();
  for (const Medium& medium : mediums_vector) {
    bbox.p_min = min(bbox.p_min, medium.bounds.p_min);
    bbox.p_max = max(bbox.p_max, medium.bounds.p_max);
  }
  return bbox;
}

SceneHashes SceneData::compute_hashes() const {
  ETX_PROFILER_SCOPE();
  SceneHashes result = {};
  result.vertices_pos_hash = xxh64(vertices.pos.data(), vertices.pos.size() * sizeof(float3));
  result.vertices_nrm_hash = xxh64(vertices.nrm.data(), vertices.nrm.size() * sizeof(float3));
  result.vertices_tan_hash = xxh64(vertices.tan.data(), vertices.tan.size() * sizeof(float3));
  result.vertices_btn_hash = xxh64(vertices.btn.data(), vertices.btn.size() * sizeof(float3));
  result.vertices_tex_hash = xxh64(vertices.tex.data(), vertices.tex.size() * sizeof(float2));
  result.triangles_hash = xxh64(triangles.data(), triangles.size() * sizeof(Triangle));
  result.meshes_hash = xxh64(meshes.data(), meshes.size() * sizeof(Mesh));
  result.hierarchy_hash = hash_hierarchy_structure(hierarchy);
  result.transforms_hash = compute_transforms_hash();
  result.instance_transforms_hash = compute_instance_transforms_hash();
  result.attachments_hash = hash_hierarchy_attachments(hierarchy);
  result.materials_hash = xxh64(materials.data(), materials.size() * sizeof(Material));
  result.spectra_hash = xxh64(spectrum_values.data(), spectrum_values.size() * sizeof(SpectralDistribution));
  result.emitter_profiles_hash = xxh64(emitter_profiles.data(), emitter_profiles.size() * sizeof(EmitterProfile));
  result.energy_compensation_interfaces_hash = xxh64(energy_compensation_interfaces.data(), energy_compensation_interfaces.size() * sizeof(Scene::EnergyCompensationInterface));
  result.pixel_filter_hash = xxh64(&pixel_filter, sizeof(PixelFilter));
  result.defaults_hash = xxh64(&defaults, sizeof(Scene::Defaults));
  result.options_hash = xxh64(&options, sizeof(Scene::Options));
  result.images_hash = hash_images_struct_and_payload(*this);
  result.mediums_hash = hash_mediums_struct_and_payload(*this);
  result.triangle_indices_hash = hash_triangle_indices(triangles);

  return result;
}

uint64_t SceneData::compute_transforms_hash() const {
  ETX_PROFILER_SCOPE();
  return hash_hierarchy_transforms(hierarchy);
}

uint64_t SceneData::compute_instance_transforms_hash() const {
  ETX_PROFILER_SCOPE();
  return hash_instance_transforms(hierarchy);
}

void SceneData::clear(TaskScheduler& scheduler) {
  images.remove_all();
  mediums.remove_all();
  buffer_pool.clear();
  vertices.pos.clear();
  vertices.nrm.clear();
  vertices.tan.clear();
  vertices.btn.clear();
  vertices.tex.clear();
  triangles.clear();
  materials.clear();
  meshes.clear();
  emitter_profiles.clear();
  emitter_names.clear();
  spectrum_values.clear();
  images_vector.clear();
  mediums_vector.clear();
  energy_compensation_interfaces.clear();
  hierarchy.clear();
  spectrum_names.clear();
  material_mapping.clear();
  mesh_mapping.clear();
  material_to_emitter_profile.clear();
  gltf_image_mapping.clear();
  gltf_material_mapping.clear();
  cameras.clear();
  json_file_name.clear();
  geometry_file_name.clear();
  materials_file_name.clear();
  images.init(1024u);
  mediums.init(1024u);
}

void SceneData::swap_contents(SceneData& other) {
  using std::swap;
  swap(vertices.pos, other.vertices.pos);
  swap(vertices.nrm, other.vertices.nrm);
  swap(vertices.tan, other.vertices.tan);
  swap(vertices.btn, other.vertices.btn);
  swap(vertices.tex, other.vertices.tex);
  swap(triangles, other.triangles);
  swap(materials, other.materials);
  swap(meshes, other.meshes);
  swap(emitter_profiles, other.emitter_profiles);
  swap(emitter_names, other.emitter_names);
  swap(spectrum_values, other.spectrum_values);
  swap(images_vector, other.images_vector);
  swap(mediums_vector, other.mediums_vector);
  swap(energy_compensation_interfaces, other.energy_compensation_interfaces);
  swap(hierarchy, other.hierarchy);
  swap(buffer_pool, other.buffer_pool);
  images.swap_contents(other.images);
  mediums.swap_contents(other.mediums);
  swap(material_mapping, other.material_mapping);
  swap(mesh_mapping, other.mesh_mapping);
  swap(spectrum_names, other.spectrum_names);
  swap(material_to_emitter_profile, other.material_to_emitter_profile);
  swap(gltf_image_mapping, other.gltf_image_mapping);
  swap(gltf_material_mapping, other.gltf_material_mapping);
  swap(cameras, other.cameras);
  swap(json_file_name, other.json_file_name);
  swap(geometry_file_name, other.geometry_file_name);
  swap(materials_file_name, other.materials_file_name);
  swap(pixel_filter, other.pixel_filter);
  swap(defaults, other.defaults);
  swap(options, other.options);
  swap(_camera_attachment_nodes_scratch, other._camera_attachment_nodes_scratch);
  swap(_medium_attachment_nodes_scratch, other._medium_attachment_nodes_scratch);
}

uint32_t SceneData::add_spectrum(const char* source_id, const SpectralDistribution& spd) {
  ETX_CRITICAL((source_id != nullptr) && (source_id[0] != 0));

  uint32_t index = uint32_t(spectrum_values.size());
  spectrum_values.emplace_back(spd);
  spectrum_names.emplace_back(source_id);
  return index;
}

uint32_t SceneData::add_spectrum(const char* id) {
  return add_spectrum(id, SpectralDistribution{});
}

uint32_t SceneData::add_spectrum() {
  char buffer[64] = {};
  snprintf(buffer, sizeof(buffer), "##spectrum%04u", uint32_t(spectrum_names.size()));
  return add_spectrum(buffer);
}

uint32_t SceneData::add_spectrum(const SpectralDistribution& spd) {
  uint32_t i = add_spectrum();
  spectrum_values[i] = spd;
  return i;
}

uint32_t SceneData::find_spectrum(const char* id) const {
  if ((id == nullptr) || (id[0] == 0))
    return kInvalidIndex;

  auto i = std::find(spectrum_names.begin(), spectrum_names.end(), id);
  if (i == spectrum_names.end())
    return kInvalidIndex;

  return uint32_t(std::distance(spectrum_names.begin(), i));
}

bool SceneData::has_material(const char* name) const {
  return material_mapping.count(name) > 0;
}

uint32_t SceneData::add_material(const char* name) {
  std::string id = (name != nullptr) && (name[0] != 0) ? name : ("material-" + std::to_string(materials.size()));
  auto i = material_mapping.find(id);
  if (i != material_mapping.end()) {
    uint32_t existing_index = i->second;
    if (existing_index >= materials.size()) {
      materials.resize(existing_index + 1);
    }
    return existing_index;
  }
  uint32_t index = static_cast<uint32_t>(materials.size());
  materials.emplace_back();
  material_mapping[id] = index;
  return index;
}

uint32_t SceneData::clone_material(const Material& src, const char* name) {
  uint32_t index = static_cast<uint32_t>(materials.size());
  materials.emplace_back(src);
  std::string id = (name != nullptr) && (name[0] != 0) ? name : ("material-" + std::to_string(index));
  if (material_mapping.count(id) > 0) {
    id += "#" + std::to_string(index);
  }
  material_mapping[id] = index;
  return index;
}

uint32_t SceneData::add_mesh(const char* name, uint32_t triangle_offset, uint32_t triangle_count, const float3& bbox_min, const float3& bbox_max) {
  const uint32_t mesh_index = add_mesh_asset(name, triangle_offset, triangle_count, bbox_min, bbox_max);
  const uint32_t node_index = hierarchy.add_node(name, kInvalidIndex, {});
  ETX_CRITICAL(node_index != kInvalidIndex);
  const SceneAttachment attachment = {SceneAttachment::Type::Mesh, mesh_index, 0u, 0u};
  ETX_CRITICAL(hierarchy.add_attachment(node_index, attachment));
  return mesh_index;
}

uint32_t SceneData::add_mesh_asset(const char* name, uint32_t triangle_offset, uint32_t triangle_count, const float3& bbox_min, const float3& bbox_max) {
  uint32_t index = static_cast<uint32_t>(meshes.size());
  auto& mesh = meshes.emplace_back();
  mesh.triangle_offset = triangle_offset;
  mesh.triangle_count = triangle_count;
  mesh.bbox_min = bbox_min;
  mesh.bbox_max = bbox_max;
  std::string mesh_name = name && name[0] ? name : ("mesh-" + std::to_string(index));
  mesh_mapping[mesh_name] = index;
  return index;
}

bool SceneData::resolve_hierarchy() {
  if (hierarchy.resolve_mesh_instances(meshes) == false) {
    return false;
  }

  _camera_attachment_nodes_scratch.assign(cameras.size(), kInvalidIndex);
  _medium_attachment_nodes_scratch.assign(mediums.array_size(), kInvalidIndex);
  for (uint32_t node_index : hierarchy.evaluation_order) {
    const SceneNode& node = hierarchy.nodes[node_index];
    const uint32_t attachment_end = node.attachment_offset + node.attachment_count;
    if (attachment_end > hierarchy.attachments.size()) {
      return false;
    }

    for (uint32_t attachment_index = node.attachment_offset; attachment_index < attachment_end; ++attachment_index) {
      const SceneAttachment& attachment = hierarchy.attachments[attachment_index];
      switch (attachment.type) {
        case SceneAttachment::Type::Mesh:
          if (attachment.resource_index >= meshes.size()) {
            return false;
          }
          break;
        case SceneAttachment::Type::Camera:
          if (attachment.resource_index >= cameras.size()) {
            return false;
          }
          if (hierarchy.effective_enabled[node_index] != 0u) {
            if (_camera_attachment_nodes_scratch[attachment.resource_index] != kInvalidIndex) {
              log::error("Camera %u is attached to multiple enabled nodes (%u and %u)", attachment.resource_index, _camera_attachment_nodes_scratch[attachment.resource_index],
                node_index);
              return false;
            }
            _camera_attachment_nodes_scratch[attachment.resource_index] = node_index;
          }
          break;
        case SceneAttachment::Type::Emitter:
          if (attachment.resource_index >= emitter_profiles.size()) {
            return false;
          }
          break;
        case SceneAttachment::Type::Medium:
          if (attachment.resource_index >= mediums.array_size()) {
            return false;
          }
          if (hierarchy.effective_enabled[node_index] != 0u) {
            if (_medium_attachment_nodes_scratch[attachment.resource_index] != kInvalidIndex) {
              log::error("Medium %u is attached to multiple enabled nodes (%u and %u)", attachment.resource_index, _medium_attachment_nodes_scratch[attachment.resource_index],
                node_index);
              return false;
            }
            _medium_attachment_nodes_scratch[attachment.resource_index] = node_index;
          }
          break;
        default:
          return false;
      }
    }
  }
  hierarchy.mark_resolved_state_current();
  return true;
}

uint32_t SceneData::add_image(const char* path, uint32_t options, const float2& offset, const float2& scale) {
  std::string id = path && path[0] ? path : ("##image-" + std::to_string(images.array_size()));
  return images.add_from_file(id, options, offset, scale);
}

uint32_t SceneData::add_image(const float4* data, const uint2& dim, uint32_t options, const float2& offset, const float2& scale) {
  return images.add_from_data(data, dim, options, offset, scale);
}

uint32_t SceneData::add_image(const Image& img) {
  return images.add_copy(img);
}

uint32_t SceneData::add_image(const char* path, uint32_t options) {
  return add_image(path, options, {}, {1.0f, 1.0f});
}

void SceneData::add_image_options(uint32_t index, uint32_t options) {
  images.add_options(index, options);
}

uint32_t SceneData::add_energy_compensation_interface(const Scene::EnergyCompensationInterface& interface_data) {
  const uint32_t index = static_cast<uint32_t>(energy_compensation_interfaces.size());
  energy_compensation_interfaces.emplace_back(interface_data);
  return index;
}

uint32_t SceneData::add_medium(Medium::Class cls, const char* name, const char* volume_file, const SpectralDistribution& s_a, const SpectralDistribution& s_t, float g,
  bool explicit_connections) {
  uint32_t absorption_index = add_spectrum(s_a);
  uint32_t scattering_index = add_spectrum(s_t);

  std::string id = name && name[0] ? name : ("medium-" + std::to_string(mediums.array_size()));
  return mediums.add(cls, id, volume_file, absorption_index, scattering_index, g, explicit_connections);
}

uint32_t SceneData::add_atmosphere_emitter(const AtmosphereEmitterParameters& params) {
  constexpr uint32_t kSkyImageBaseDimensions = 1024u;

  uint2 sky_image_dimensions = uint2{kSkyImageBaseDimensions, 2u * kSkyImageBaseDimensions};
  sky_image_dimensions.x = max(64u, uint32_t(sky_image_dimensions.x * params.quality));
  sky_image_dimensions.y = max(64u, uint32_t(sky_image_dimensions.y * params.quality));

  uint32_t atmosphere_emitter_index = uint32_t(emitter_profiles.size());

  // Create placeholder sky image (will be properly generated later in finalize_scene_loading)
  std::vector<float4> image_buffer(sky_image_dimensions.x * sky_image_dimensions.y, float4{0.0f, 0.0f, 0.0f, 1.0f});

  auto& e = emitter_profiles.emplace_back(EmitterProfile::Class::Environment);
  e.emission.spectrum_index = add_spectrum(params.env_spectrum);
  e.atmosphere.scattering = params.scattering;
  e.atmosphere.quality = params.quality;
  e.meta = EmitterProfile::Meta::Atmosphere;
  e.emission.image_index = add_image(image_buffer.data(), sky_image_dimensions, Image::BuildSamplingTable | Image::UniformSamplingTable | Image::RepeatU, {}, {1.0f, 1.0f});

  return atmosphere_emitter_index;
}

void SceneData::build_atmosphere_and_sun_images(uint32_t atmosphere_emitter_index, RHIContext& rhi, scattering::GpuContext& gpu_context) {
  auto& atmosphere_emitter = emitter_profiles[atmosphere_emitter_index];

  std::vector<uint32_t> sun_emitter_indices;
  std::vector<scattering::LightSource> light_sources;
  for (uint32_t i = 0; i < emitter_profiles.size(); ++i) {
    auto& candidate = emitter_profiles[i];
    if ((candidate.cls == EmitterProfile::Class::Directional) && (candidate.reference_emitter_index == atmosphere_emitter_index)) {
      sun_emitter_indices.push_back(i);
      scattering::LightSource sun_light = {
        spectrum_values[candidate.emission.spectrum_index],
        candidate.directional.direction,
        candidate.directional.angular_size,
        1.0f,
      };
      light_sources.push_back(sun_light);
    }
  }

  if (atmosphere_emitter.emission.image_index != kInvalidIndex) {
    images.load_images(scheduler);
    auto& img = images_vector[atmosphere_emitter.emission.image_index];
    img.options = img.options | Image::UniformSamplingTable | Image::RepeatU;
    auto ptr = buffer_pool.map<float4>(img.data);
    ETX_CRITICAL(ptr != nullptr);
    const uint2 image_dimensions = uint2{img.isize.x, img.isize.y};
    if (scattering::generate_sky_image(rhi, gpu_context, atmosphere_emitter.atmosphere.scattering, image_dimensions, light_sources, ptr) == false) {
      log::error("Failed to generate atmosphere sky image on GPU for emitter %u", atmosphere_emitter_index);
    } else {
      images.rebuild_sampling_table(atmosphere_emitter.emission.image_index, scheduler);
    }
  }

  rebuild_sun_images_for_atmosphere(atmosphere_emitter_index, sun_emitter_indices, rhi, gpu_context);
}

void SceneData::rebuild_sun_images_for_atmosphere(uint32_t atmosphere_emitter_index, const std::vector<uint32_t>& sun_emitter_indices, RHIContext& rhi,
  scattering::GpuContext& gpu_context) {
  auto& atmosphere_emitter = emitter_profiles[atmosphere_emitter_index];

  constexpr uint2 kSunImageDimensions = uint2{128u, 128u};

  for (uint32_t sun_idx : sun_emitter_indices) {
    auto& sun_emitter = emitter_profiles[sun_idx];

    std::vector<float4> sun_buffer(kSunImageDimensions.x * kSunImageDimensions.y, float4{0.0f, 0.0f, 0.0f, 0.0f});
    if (scattering::generate_sun_image(rhi, gpu_context, atmosphere_emitter.atmosphere.scattering, kSunImageDimensions, sun_emitter.directional.direction,
          sun_emitter.directional.angular_size, sun_buffer.data()) == false) {
      log::error("Failed to generate atmosphere sun image on GPU for emitter %u", sun_idx);
      continue;
    }

    if (sun_emitter.emission.image_index == kInvalidIndex) {
      sun_emitter.emission.image_index = add_image(sun_buffer.data(), kSunImageDimensions, Image::BuildSamplingTable, {}, {1.0f, 1.0f});
      images.load_images(scheduler);
    } else {
      auto& img = images_vector[sun_emitter.emission.image_index];
      if ((img.isize.x != kSunImageDimensions.x) || (img.isize.y != kSunImageDimensions.y)) {
        sun_emitter.emission.image_index = add_image(sun_buffer.data(), kSunImageDimensions, Image::BuildSamplingTable, {}, {1.0f, 1.0f});
        images.load_images(scheduler);
      } else {
        buffer_pool.write(img.data, sun_buffer.data(), sun_buffer.size() * sizeof(float4));
        images.rebuild_sampling_table(sun_emitter.emission.image_index, scheduler);
      }
    }
  }
}

void SceneData::rebuild_atmosphere_emitter(uint32_t emitter_index, RHIContext& rhi, scattering::GpuContext& gpu_context) {
  if (emitter_index >= emitter_profiles.size()) {
    return;
  }

  auto& emitter = emitter_profiles[emitter_index];

  uint32_t atmosphere_emitter_index = kInvalidIndex;
  if (((emitter.meta & EmitterProfile::Meta::Atmosphere) != 0) && (emitter.cls == EmitterProfile::Class::Environment)) {
    atmosphere_emitter_index = emitter_index;
  } else if ((emitter.cls == EmitterProfile::Class::Directional) && (emitter.reference_emitter_index != kInvalidIndex)) {
    uint32_t ref_index = emitter.reference_emitter_index;
    if (ref_index < emitter_profiles.size()) {
      auto& ref_emitter = emitter_profiles[ref_index];
      if (((ref_emitter.meta & EmitterProfile::Meta::Atmosphere) != 0) && (ref_emitter.cls == EmitterProfile::Class::Environment)) {
        atmosphere_emitter_index = ref_index;
      }
    }
  }

  if (atmosphere_emitter_index != kInvalidIndex) {
    build_atmosphere_and_sun_images(atmosphere_emitter_index, rhi, gpu_context);
  }
}

}  // namespace etx
