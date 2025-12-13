#pragma once

#include <etx/render/shared/scene.hxx>
#include <etx/render/shared/camera.hxx>
#include <etx/render/shared/spectrum.hxx>
#include <etx/render/shared/image.hxx>
#include <etx/render/host/tasks.hxx>
#include <etx/render/shared/scattering.hxx>
#include <etx/render/host/image_pool.hxx>
#include <etx/render/host/medium_pool.hxx>
#include <etx/render/host/scene_representation.hxx>

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace etx {

struct SceneData {
  struct CameraInfo {
    Camera cam;
    std::string id;
    bool active = false;
  };

  using MaterialMapping = std::unordered_map<std::string, uint32_t>;

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
  std::vector<std::string> spectrum_names;
  std::vector<SpectralDistribution> spectrum_values;
  std::vector<CameraInfo> cameras;

  std::string json_file_name;
  std::string geometry_file_name;
  std::string materials_file_name;

  Image atmosphere_extinction;

  MaterialMapping material_mapping;
  MaterialMapping mesh_mapping;
  std::unordered_map<uint32_t, uint32_t> material_to_emitter_profile;
  std::unordered_map<uint32_t, uint32_t> gltf_image_mapping;
  std::unordered_map<int32_t, uint32_t> gltf_material_mapping;

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
};

struct SceneLoaderContext {
  SceneLoaderContext(TaskScheduler& s)
    : images(s) {
  }

  ImagePool images;
  MediumPool mediums;
  scattering::ScatteringSpectrums scattering_spectrums;

  uint32_t add_image(const char* path, uint32_t options, const float2& offset, const float2& scale) {
    std::string id = path && path[0] ? path : ("##image-" + std::to_string(images.array_size()));
    return images.add_from_file(id, options | Image::Delay, offset, scale);
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

  uint32_t add_medium(const Scene& scene, SceneData& data, Medium::Class cls, const char* name, const char* volume_file, const SpectralDistribution& s_a,
    const SpectralDistribution& s_t, float g, bool explicit_connections) {
    auto select_index = [&](const SpectralDistribution& spd, uint32_t fallback) {
      if (spd.spectral_entry_count == 0) {
        return fallback;
      }
      return data.add_spectrum(spd);
    };

    uint32_t absorption_index = select_index(s_a, scene.black_spectrum);
    uint32_t scattering_index = select_index(s_t, scene.black_spectrum);

    std::string id = name && name[0] ? name : ("medium-" + std::to_string(mediums.array_size()));
    return mediums.add(cls, id, volume_file, absorption_index, scattering_index, g, explicit_connections);
  }

  void add_atmosphere_emitter(const SceneRepresentation::AtmosphereEmitterParameters& params, SceneData& data, Scene& scene, TaskScheduler& scheduler) {
    const float3 normalized_direction = normalize(params.direction);

    constexpr uint2 kSunImageDimensions = uint2{128u, 128u};
    constexpr uint32_t kSkyImageBaseDimensions = 1024u;

    auto sun_spectrum = SpectralDistribution::from_normalized_black_body(5772.0f, 1.0f);

    uint32_t sun_emitter_index = kInvalidIndex;
    if (params.sun_scale > 0.0f) {
      auto& instance = data.emitter_instances.emplace_back(EmitterProfile::Class::Directional);
      sun_emitter_index = uint32_t(data.emitter_profiles.size());
      instance.profile = sun_emitter_index;

      auto& d = data.emitter_profiles.emplace_back(EmitterProfile::Class::Directional);
      d.emission.spectrum_index = data.add_spectrum(sun_spectrum);
      d.directional.angular_size = params.angular_diameter_degrees * kPi / 180.0f;
      d.directional.direction = normalized_direction;
      d.meta = uint32_t(EmitterProfile::Meta::Atmosphere);

      data.spectrum_values[d.emission.spectrum_index].scale(params.sun_scale);

      if (d.directional.angular_size > 0.0f) {
        d.emission.image_index = add_image(nullptr, kSunImageDimensions, Image::Delay, {}, {1.0f, 1.0f});
        auto& img = images.get(d.emission.image_index);
        scattering::generate_sun_image(params, kSunImageDimensions, normalized_direction, d.directional.angular_size, img.pixels.f32.a, scattering_spectrums, scheduler);
      }
    }

    if (params.sky_scale > 0.0f) {
      uint2 sky_image_dimensions = uint2{kSkyImageBaseDimensions, 2u * kSkyImageBaseDimensions};
      sky_image_dimensions.x = max(64u, uint32_t(sky_image_dimensions.x * params.quality));
      sky_image_dimensions.y = max(64u, uint32_t(sky_image_dimensions.y * params.quality));

      auto& instance = data.emitter_instances.emplace_back(EmitterProfile::Class::Environment);
      instance.profile = uint32_t(data.emitter_profiles.size());

      uint32_t sky_emitter_index = uint32_t(data.emitter_profiles.size());
      auto& e = data.emitter_profiles.emplace_back(EmitterProfile::Class::Environment);
      e.emission.spectrum_index = data.add_spectrum(sun_spectrum);
      e.emission.image_index = add_image(nullptr, sky_image_dimensions, Image::BuildSamplingTable | Image::Delay, {}, {1.0f, 1.0f});
      e.directional.direction = normalized_direction;
      e.meta = uint32_t(EmitterProfile::Meta::Atmosphere);
      e.reference_emitter_index = sun_emitter_index;

      if ((sun_emitter_index != kInvalidIndex) && (sun_emitter_index < data.emitter_profiles.size())) {
        data.emitter_profiles[sun_emitter_index].reference_emitter_index = sky_emitter_index;
      }

      data.spectrum_values[e.emission.spectrum_index].scale(params.sky_scale);

      auto& img = images.get(e.emission.image_index);
      scattering::generate_sky_image(params, sky_image_dimensions, normalized_direction, data.atmosphere_extinction, img.pixels.f32.a, scattering_spectrums, scheduler);
    }
  }

  void rebuild_atmosphere_emitter(uint32_t emitter_index, SceneData& data, Scene& scene, TaskScheduler& scheduler) {
    if (emitter_index >= scene.emitter_profiles.count) {
      return;
    }

    auto& emitter = scene.emitter_profiles[emitter_index];
    if ((emitter.meta & uint32_t(EmitterProfile::Meta::Atmosphere)) == 0) {
      return;
    }

    uint32_t paired_emitter_index = emitter.reference_emitter_index;
    if ((paired_emitter_index == kInvalidIndex) || (paired_emitter_index >= scene.emitter_profiles.count)) {
      return;
    }

    auto& paired_emitter = scene.emitter_profiles[paired_emitter_index];
    uint32_t sky_emitter_index = (emitter.cls == EmitterProfile::Class::Environment) ? emitter_index : paired_emitter_index;
    uint32_t sun_emitter_index = (emitter.cls == EmitterProfile::Class::Directional) ? emitter_index : paired_emitter_index;

    auto& sky_emitter = scene.emitter_profiles[sky_emitter_index];
    auto& sun_emitter = scene.emitter_profiles[sun_emitter_index];
    float3 sun_direction = sun_emitter.directional.direction;

    if (sky_emitter.emission.image_index != kInvalidIndex) {
      auto& img = images.get(sky_emitter.emission.image_index);
      scattering::generate_sky_image(sky_emitter.atmosphere.scattering, img.isize, sun_direction, data.atmosphere_extinction, img.pixels.f32.a, scattering_spectrums, scheduler);
      images.add_options(sky_emitter.emission.image_index, Image::BuildSamplingTable);
      images.load_images();
    }

    if (sun_emitter.emission.image_index != kInvalidIndex) {
      auto& sun_img = images.get(sun_emitter.emission.image_index);
      scattering::generate_sun_image(sky_emitter.atmosphere.scattering, sun_img.isize, sun_direction, sun_emitter.directional.angular_size, sun_img.pixels.f32.a,
        scattering_spectrums, scheduler);
    }
  }
};

}  // namespace etx
