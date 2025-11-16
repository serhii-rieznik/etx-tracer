#pragma once

#include <etx/render/shared/scene.hxx>
#include <etx/render/shared/camera.hxx>
#include <etx/render/shared/spectrum.hxx>
#include <etx/render/shared/image.hxx>

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

  std::vector<Vertex> vertices;
  std::vector<Triangle> triangles;
  std::vector<uint32_t> triangle_to_emitter;
  std::vector<Material> materials;
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
  std::unordered_map<uint32_t, uint32_t> material_to_emitter_profile;
  std::unordered_map<uint32_t, uint32_t> gltf_image_mapping;

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
      return i->second;
    }
    uint32_t index = static_cast<uint32_t>(materials.size());
    materials.emplace_back();
    material_mapping[id] = index;
    return index;
  }
};

struct SceneLoaderContext {
  SceneLoaderContext(TaskScheduler& s)
    : images(s) {
  }

  ImagePool images;
  MediumPool mediums;

  uint32_t add_image(const char* path, uint32_t options, const float2& offset, const float2& scale) {
    std::string id = path && path[0] ? path : ("image-" + std::to_string(images.array_size()));
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

    float max_sigma = 0.0f;
    if ((absorption_index != kInvalidIndex) && (absorption_index < data.spectrum_values.size())) {
      max_sigma += data.spectrum_values[absorption_index].maximum_spectral_power();
    }
    if ((scattering_index != kInvalidIndex) && (scattering_index < data.spectrum_values.size())) {
      max_sigma += data.spectrum_values[scattering_index].maximum_spectral_power();
    }

    std::string id = name && name[0] ? name : ("medium-" + std::to_string(mediums.array_size()));
    return mediums.add(cls, id, volume_file, absorption_index, scattering_index, max_sigma, g, explicit_connections);
  }
};

}  // namespace etx
