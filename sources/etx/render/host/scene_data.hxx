#pragma once

#include <etx/render/shared/scene.hxx>
#include <etx/render/shared/camera.hxx>
#include <etx/render/shared/spectrum.hxx>
#include <etx/render/shared/image.hxx>
#include <etx/render/host/tasks.hxx>
#include <etx/render/shared/scattering.hxx>
#include <etx/render/host/image_pool.hxx>
#include <etx/render/host/medium_pool.hxx>
#include <etx/render/host/distribution_builder.hxx>
#include <etx/render/host/scene_representation.hxx>

#include <stb_image_write.hxx>

#include <cstdint>
#include <string>
#include <vector>

namespace etx {

struct SceneData {
  struct {
    std::vector<float3> pos;
    std::vector<float3> nrm;
    std::vector<float3> tan;
    std::vector<float3> btn;
    std::vector<float2> tex;
  } vertices;

  std::vector<Triangle> triangles;
  std::vector<uint32_t> triangle_to_emitter;
  std::vector<Material> materials;
  std::vector<Mesh> meshes;
  std::vector<EmitterProfile> emitter_profiles;
  std::vector<Emitter> emitter_instances;
  std::vector<SpectralDistribution> spectrum_values;
  std::vector<Image> images_vector;
  std::vector<ImageStorage> images_storage_vector;
  std::vector<Medium> mediums_vector;
  std::vector<Distribution::Entry> emitters_distribution_storage;

  ImagePool images;
  MediumPool mediums;
  std::vector<std::string> spectrum_names;
  using MaterialMapping = std::unordered_map<std::string, uint32_t>;
  MaterialMapping material_mapping;
  MaterialMapping mesh_mapping;
  std::unordered_map<uint32_t, uint32_t> material_to_emitter_profile;
  std::unordered_map<uint32_t, uint32_t> gltf_image_mapping;
  std::unordered_map<int32_t, uint32_t> gltf_material_mapping;

  struct CameraInfo {
    Camera cam;
    std::string id;
    bool active = false;
  };
  std::vector<CameraInfo> cameras;
  std::string json_file_name;
  std::string geometry_file_name;
  std::string materials_file_name;
  scattering::ScatteringSpectrums scattering_spectrums;
  scattering::OpticalDepthData extinction_data;
  TaskScheduler& scheduler;

  SceneData(TaskScheduler& s)
    : images(images_vector, images_storage_vector)
    , mediums(mediums_vector)
    , scheduler(s) {
  }

  void clear(TaskScheduler& scheduler) {
    images.remove_all();
    mediums.remove_all();
    vertices.pos.clear();
    vertices.nrm.clear();
    vertices.tan.clear();
    vertices.btn.clear();
    vertices.tex.clear();
    triangles.clear();
    triangle_to_emitter.clear();
    materials.clear();
    meshes.clear();
    emitter_profiles.clear();
    emitter_instances.clear();
    spectrum_values.clear();
    images_vector.clear();
    images_storage_vector.clear();
    mediums_vector.clear();
    emitters_distribution_storage.clear();
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

  SceneData(const SceneData&) = delete;
  SceneData operator=(const SceneData&) = delete;

  uint32_t add_spectrum(const char* source_id, const SpectralDistribution& spd) {
    ETX_CRITICAL((source_id != nullptr) && (source_id[0] != 0));

    uint32_t index = uint32_t(spectrum_values.size());
    spectrum_values.emplace_back(spd);
    spectrum_names.emplace_back(source_id);
    return index;
  }

  uint32_t add_spectrum(const char* id) {
    return add_spectrum(id, SpectralDistribution{});
  }

  uint32_t add_spectrum() {
    char buffer[64] = {};
    snprintf(buffer, sizeof(buffer), "##spectrum%04u", uint32_t(spectrum_names.size()));
    return add_spectrum(buffer);
  }

  uint32_t add_spectrum(const SpectralDistribution& spd) {
    uint32_t i = add_spectrum();
    spectrum_values[i] = spd;
    return i;
  }

  uint32_t find_spectrum(const char* id) const {
    if ((id == nullptr) || (id[0] == 0))
      return kInvalidIndex;

    auto i = std::find(spectrum_names.begin(), spectrum_names.end(), id);
    if (i == spectrum_names.end())
      return kInvalidIndex;

    return uint32_t(std::distance(spectrum_names.begin(), i));
  }

  bool has_material(const char* name) const {
    return material_mapping.count(name) > 0;
  }

  uint32_t add_material(const char* name) {
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

  uint32_t clone_material(const Material& src, const char* name = nullptr) {
    uint32_t index = static_cast<uint32_t>(materials.size());
    materials.emplace_back(src);
    std::string id = (name != nullptr) && (name[0] != 0) ? name : ("material-" + std::to_string(index));
    if (material_mapping.count(id) > 0) {
      id += "#" + std::to_string(index);
    }
    material_mapping[id] = index;
    return index;
  }

  uint32_t add_mesh(const char* name, uint32_t triangle_offset, uint32_t triangle_count, const float3& bbox_min, const float3& bbox_max) {
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

  uint32_t add_image(const char* path, uint32_t options, const float2& offset, const float2& scale) {
    std::string id = path && path[0] ? path : ("##image-" + std::to_string(images.array_size()));
    return images.add_from_file(id, options, offset, scale);
  }

  uint32_t add_image(const float4* data, const uint2& dim, uint32_t options, const float2& offset, const float2& scale) {
    return images.add_from_data(data, dim, options, offset, scale);
  }

  uint32_t add_image(const Image& img) {
    return images.add_copy(img);
  }

  uint32_t add_image(const char* path, uint32_t options) {
    return add_image(path, options, {}, {1.0f, 1.0f});
  }

  void add_image_options(uint32_t index, uint32_t options) {
    images.add_options(index, options);
  }

  uint32_t add_medium(Medium::Class cls, const char* name, const char* volume_file, const SpectralDistribution& s_a, const SpectralDistribution& s_t, float g,
    bool explicit_connections) {
    uint32_t absorption_index = add_spectrum(s_a);
    uint32_t scattering_index = add_spectrum(s_t);

    std::string id = name && name[0] ? name : ("medium-" + std::to_string(mediums.array_size()));
    return mediums.add(cls, id, volume_file, absorption_index, scattering_index, g, explicit_connections);
  }

  uint32_t add_atmosphere_emitter(const SceneRepresentation::AtmosphereEmitterParameters& params, Scene& scene) {
    constexpr uint32_t kSkyImageBaseDimensions = 1024u;

    uint2 sky_image_dimensions = uint2{kSkyImageBaseDimensions, 2u * kSkyImageBaseDimensions};
    sky_image_dimensions.x = max(64u, uint32_t(sky_image_dimensions.x * params.quality));
    sky_image_dimensions.y = max(64u, uint32_t(sky_image_dimensions.y * params.quality));

    auto& instance = emitter_instances.emplace_back(EmitterProfile::Class::Environment);
    uint32_t atmosphere_emitter_index = uint32_t(emitter_profiles.size());
    instance.profile = atmosphere_emitter_index;

    // Create placeholder sky image (will be properly generated later in finalize_scene_loading)
    std::vector<float4> image_buffer(sky_image_dimensions.x * sky_image_dimensions.y, float4{0.0f, 0.0f, 0.0f, 1.0f});

    auto& e = emitter_profiles.emplace_back(EmitterProfile::Class::Environment);
    e.emission.spectrum_index = add_spectrum(params.env_spectrum);
    e.atmosphere.scattering = params.scattering;
    e.atmosphere.quality = params.quality;
    e.meta = EmitterProfile::Meta::Atmosphere;
    e.emission.image_index = add_image(image_buffer.data(), sky_image_dimensions, Image::BuildSamplingTable, {}, {1.0f, 1.0f});

    scene.emitter_profiles = {emitter_profiles.data(), emitter_profiles.size()};
    scene.emitter_instances = {emitter_instances.data(), emitter_instances.size()};

    // Atmosphere images will be generated later in finalize_scene_loading when all emitters are set up
    return atmosphere_emitter_index;
  }

  void generate_atmosphere_sky_image(const scattering::Parameters& scattering_params, const std::vector<scattering::LightSource>& light_sources, float4* image_buffer,
    const uint2& dimensions) {
    scattering::generate_sky_image(scattering_params, dimensions, light_sources, extinction_data, image_buffer, scattering_spectrums, scheduler);
  }

  void build_atmosphere_and_sun_images(uint32_t atmosphere_emitter_index) {
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
      auto& img = images.get(atmosphere_emitter.emission.image_index);
      generate_atmosphere_sky_image(atmosphere_emitter.atmosphere.scattering, light_sources, img.pixels.f32.a, img.isize);
      images.rebuild_sampling_table(atmosphere_emitter.emission.image_index, scheduler);
    }

    rebuild_sun_images_for_atmosphere(atmosphere_emitter_index, sun_emitter_indices);
  }

  void rebuild_sun_images_for_atmosphere(uint32_t atmosphere_emitter_index, const std::vector<uint32_t>& sun_emitter_indices) {
    auto& atmosphere_emitter = emitter_profiles[atmosphere_emitter_index];

    constexpr uint2 kSunImageDimensions = uint2{128u, 128u};

    static uint32_t sun_image_counter = 0;

    for (uint32_t sun_idx : sun_emitter_indices) {
      auto& sun_emitter = emitter_profiles[sun_idx];

      std::vector<float4> sun_buffer(kSunImageDimensions.x * kSunImageDimensions.y, float4{0.0f, 0.0f, 0.0f, 0.0f});
      scattering::generate_sun_image(atmosphere_emitter.atmosphere.scattering, kSunImageDimensions, sun_emitter.directional.direction, sun_emitter.directional.angular_size,
        sun_buffer.data(), scattering_spectrums, scheduler);

      char tmp_path[2048] = {};
      std::string filename = std::string("sun_") + std::to_string(sun_image_counter++) + ".hdr";
      env().file_in_tmp(filename.c_str(), tmp_path, sizeof(tmp_path));
      stbi_write_hdr(tmp_path, kSunImageDimensions.x, kSunImageDimensions.y, 4, reinterpret_cast<const float*>(&sun_buffer[0]));

      if (sun_emitter.emission.image_index == kInvalidIndex) {
        sun_emitter.emission.image_index = add_image(sun_buffer.data(), kSunImageDimensions, Image::BuildSamplingTable, {}, {1.0f, 1.0f});
        images.load_images(scheduler);
      } else {
        auto& img = images.get(sun_emitter.emission.image_index);
        if ((img.isize.x != kSunImageDimensions.x) || (img.isize.y != kSunImageDimensions.y)) {
          sun_emitter.emission.image_index = add_image(sun_buffer.data(), kSunImageDimensions, Image::BuildSamplingTable, {}, {1.0f, 1.0f});
          images.load_images(scheduler);
        } else {
          memcpy(img.pixels.f32.a, sun_buffer.data(), sun_buffer.size() * sizeof(float4));
          images.rebuild_sampling_table(sun_emitter.emission.image_index, scheduler);
        }
      }
    }
  }

  void rebuild_atmosphere_emitter(uint32_t emitter_index) {
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
      build_atmosphere_and_sun_images(atmosphere_emitter_index);
    }
  }
};

void build_emitters_distribution(SceneData& scene_data, Scene& scene) {
  for (uint32_t i = 0; i < scene_data.emitter_profiles.size(); ++i) {
    auto& emitter = scene_data.emitter_profiles[i];
    if (emitter.is_distant()) {
      emitter.directional.equivalent_disk_size = 2.0f * std::tan(emitter.directional.angular_size / 2.0f);
      emitter.directional.angular_size_cosine = std::cos(emitter.directional.angular_size / 2.0f);
      float additional_weight = kPi * scene.bounding_sphere_radius * scene.bounding_sphere_radius;
      for (uint32_t j = 0; j < scene_data.emitter_instances.size(); ++j) {
        if (scene_data.emitter_instances[j].profile == i) {
          scene_data.emitter_instances[j].additional_weight = additional_weight;
        }
      }
    }
  }

  uint32_t active_emitter_count = 0;
  for (uint32_t i = 0; i < scene_data.emitter_instances.size(); ++i) {
    auto& emitter = scene_data.emitter_instances[i];

    const auto& profile = scene_data.emitter_profiles[emitter.profile];
    float spectrum_weight = (profile.emission.spectrum_index != kInvalidIndex) ? scene_data.spectrum_values[profile.emission.spectrum_index].luminance() : 0.0f;
    emitter.spectrum_weight = spectrum_weight;

    float total_weight = emitter.spectrum_weight * emitter.additional_weight;
    if (total_weight > 0.0f) {
      active_emitter_count++;
    }
  }

  log::warning("Building emitters distribution for %llu emitters (%llu active)...", scene.emitter_instances.count, active_emitter_count);

  scene.environment_emitters.count = 0;

  scene_data.emitters_distribution_storage.resize(active_emitter_count + 1);
  auto* entries = scene_data.emitters_distribution_storage.data();

  uint32_t dist_index = 0;
  for (uint32_t emitter_idx = 0; emitter_idx < scene.emitter_instances.count; ++emitter_idx) {
    auto& emitter = scene.emitter_instances[emitter_idx];
    float total_weight = emitter.spectrum_weight * emitter.additional_weight;

    if (total_weight > 0.0f) {
      entries[dist_index] = {total_weight, 0.0f, 0.0f, emitter_idx};
      dist_index++;
    }

    if (emitter.is_local()) {
      scene.triangle_to_emitter[emitter.triangle_index] = emitter_idx;
    }
  }

  for (uint32_t i = 0; i < active_emitter_count; ++i) {
    uint32_t emitter_idx = entries[i].reference;
    const auto& emitter = scene.emitter_instances[emitter_idx];
    if (emitter.is_distant()) {
      scene.environment_emitters.emitters[scene.environment_emitters.count++] = emitter_idx;
    }
  }

  float total_weight = DistributionBuilder::finalize_entries(entries, active_emitter_count);

  scene.emitters_distribution.values = {entries, active_emitter_count};
  scene.emitters_distribution.total_weight = total_weight;
}

}  // namespace etx
