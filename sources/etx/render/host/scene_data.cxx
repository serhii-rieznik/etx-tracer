#include <etx/render/host/scene_data.hxx>

#include <etx/core/core.hxx>
#include <etx/core/log.hxx>
#include <etx/render/host/gpu_asset_descriptor.hxx>

#include <chrono>

namespace etx {
namespace {

double elapsed_ms(const std::chrono::steady_clock::time_point& begin, const std::chrono::steady_clock::time_point& end) {
  return std::chrono::duration<double, std::milli>(end - begin).count();
}

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

}  // namespace

SceneData::SceneData(TaskScheduler& s)
  : images(images_vector, buffer_pool)
  , mediums(mediums_vector, buffer_pool)
  , scheduler(s) {
}

BoundingBox SceneData::compute_bounding_volumes() const {
  BoundingBox bbox = {
    float3{kMaxFloat, kMaxFloat, kMaxFloat},
    0.0f,
    float3{-kMaxFloat, -kMaxFloat, -kMaxFloat},
    0.0f,
  };

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

  for (const auto& tb : thread_bounds) {
    bbox.p_min = min(bbox.p_min, tb.p_min);
    bbox.p_max = max(bbox.p_max, tb.p_max);
  }

  return bbox;
}

SceneHashes SceneData::compute_hashes() const {
  ETX_PROFILER_SCOPE();
  const auto total_begin = std::chrono::steady_clock::now();
  SceneHashes result = {};
  const auto timed_hash = [](const void* data, size_t size, uint64_t* out_result, double& out_time_ms) {
    const auto begin = std::chrono::steady_clock::now();
    *out_result = xxh64(data, size);
    const auto end = std::chrono::steady_clock::now();
    out_time_ms = elapsed_ms(begin, end);
  };

  double vertices_pos_ms = 0.0;
  double vertices_nrm_ms = 0.0;
  double vertices_tan_ms = 0.0;
  double vertices_btn_ms = 0.0;
  double vertices_tex_ms = 0.0;
  double triangles_ms = 0.0;
  double triangle_indices_ms = 0.0;
  double meshes_ms = 0.0;
  double materials_ms = 0.0;
  double spectra_ms = 0.0;
  double emitters_ms = 0.0;
  double images_ms = 0.0;
  double mediums_ms = 0.0;
  double pixel_filter_ms = 0.0;
  double defaults_ms = 0.0;
  double options_ms = 0.0;

  timed_hash(vertices.pos.data(), vertices.pos.size() * sizeof(float3), &result.vertices_pos_hash, vertices_pos_ms);
  timed_hash(vertices.nrm.data(), vertices.nrm.size() * sizeof(float3), &result.vertices_nrm_hash, vertices_nrm_ms);
  timed_hash(vertices.tan.data(), vertices.tan.size() * sizeof(float3), &result.vertices_tan_hash, vertices_tan_ms);
  timed_hash(vertices.btn.data(), vertices.btn.size() * sizeof(float3), &result.vertices_btn_hash, vertices_btn_ms);
  timed_hash(vertices.tex.data(), vertices.tex.size() * sizeof(float2), &result.vertices_tex_hash, vertices_tex_ms);
  timed_hash(triangles.data(), triangles.size() * sizeof(Triangle), &result.triangles_hash, triangles_ms);
  timed_hash(meshes.data(), meshes.size() * sizeof(Mesh), &result.meshes_hash, meshes_ms);
  timed_hash(materials.data(), materials.size() * sizeof(Material), &result.materials_hash, materials_ms);
  timed_hash(spectrum_values.data(), spectrum_values.size() * sizeof(SpectralDistribution), &result.spectra_hash, spectra_ms);
  timed_hash(emitter_profiles.data(), emitter_profiles.size() * sizeof(EmitterProfile), &result.emitter_profiles_hash, emitters_ms);
  timed_hash(&pixel_filter, sizeof(PixelFilter), &result.pixel_filter_hash, pixel_filter_ms);
  timed_hash(&defaults, sizeof(Scene::Defaults), &result.defaults_hash, defaults_ms);
  timed_hash(&options, sizeof(Scene::Options), &result.options_hash, options_ms);

  {
    const auto begin = std::chrono::steady_clock::now();
    result.images_hash = hash_images_struct_and_payload(*this);
    const auto end = std::chrono::steady_clock::now();
    images_ms = elapsed_ms(begin, end);
  }
  {
    const auto begin = std::chrono::steady_clock::now();
    result.mediums_hash = hash_mediums_struct_and_payload(*this);
    const auto end = std::chrono::steady_clock::now();
    mediums_ms = elapsed_ms(begin, end);
  }
  {
    const auto begin = std::chrono::steady_clock::now();
    result.triangle_indices_hash = hash_triangle_indices(triangles);
    const auto end = std::chrono::steady_clock::now();
    triangle_indices_ms = elapsed_ms(begin, end);
  }

  const auto total_end = std::chrono::steady_clock::now();
  log::info(
    "Scene hash recompute timing: total=%.2fms pos=%.2fms nrm=%.2fms tan=%.2fms btn=%.2fms tex=%.2fms tri=%.2fms tri_idx=%.2fms meshes=%.2fms materials=%.2fms "
    "spectra=%.2fms emitters=%.2fms images=%.2fms mediums=%.2fms pixel_filter=%.2fms defaults=%.2fms options=%.2fms",
    elapsed_ms(total_begin, total_end), vertices_pos_ms, vertices_nrm_ms, vertices_tan_ms, vertices_btn_ms, vertices_tex_ms, triangles_ms, triangle_indices_ms, meshes_ms,
    materials_ms, spectra_ms, emitters_ms, images_ms, mediums_ms, pixel_filter_ms, defaults_ms, options_ms);
  log::info(
    "Scene hash recompute sizes: vertices=%zu triangles=%zu meshes=%zu materials=%zu spectra=%zu emitters=%zu images=%zu mediums=%zu",
    vertices.pos.size(), triangles.size(), meshes.size(), materials.size(), spectrum_values.size(), emitter_profiles.size(), images_vector.size(), mediums_vector.size());

  return result;
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
  spectrum_values.clear();
  images_vector.clear();
  mediums_vector.clear();
  energy_compensation_interfaces.clear();
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
    if (scattering::generate_sky_image(rhi, gpu_context, atmosphere_emitter.atmosphere.scattering, img.isize, light_sources, ptr) == false) {
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
