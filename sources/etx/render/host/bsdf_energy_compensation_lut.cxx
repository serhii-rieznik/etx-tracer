#include <etx/render/host/bsdf_energy_compensation_lut.hxx>

#include <etx/core/core.hxx>
#include <etx/core/environment.hxx>
#include <etx/core/log.hxx>
#include <etx/render/host/image_loaders.hxx>
#include <etx/render/interop/bsdf_energy_compensated_shared.hxx>
#include <etx/render/interop/bsdf_external_shared.hxx>

#include <tinyexr.hxx>

#include <chrono>
#include <filesystem>
#include <unordered_map>

namespace etx {

namespace {

constexpr uint32_t kEnergyCompensationGeneratorVersion = 24u;
constexpr uint32_t kEnergyCompensationConductorLutSize = 64u;
constexpr uint32_t kEnergyCompensationDielectricLutSize = 64u;
constexpr uint32_t kEnergyCompensationConductorSampleCount = 2048u;
constexpr uint32_t kEnergyCompensationSampleCount = 2048u;
constexpr uint32_t kEnergyCompensationDielectricMultiScatterSampleCount = 2048u;
constexpr uint32_t kEnergyCompensationDielectricBranchCount = 4u;
constexpr uint32_t kEnergyCompensationDielectricAverageWidth = 8u;
constexpr uint32_t kEnergyCompensationSpectralWavelengthCount = kBSDFEnergyCompensationSpectralWavelengthCount;
constexpr uint32_t kEnergyCompensationSpectralWavelengthGroupSize = kBSDFEnergyCompensationSpectralWavelengthGroupSize;
constexpr uint32_t kEnergyCompensationSpectralWavelengthGroupCount = kBSDFEnergyCompensationSpectralWavelengthGroupCount;

struct SpectralDirectionalAlbedoResult {
  float3 albedo = {};
  float visible_probability = 0.0f;
  float geometric_albedo = 0.0f;
};

struct DielectricDirectionalAlbedoResult {
  float3 branch_albedo[2] = {};
  float branch_visible_probability[2] = {};
  float visible_probability = 0.0f;
};

struct GeneratedInterfacePaths {
  std::filesystem::path directional;
  std::filesystem::path average;
  std::filesystem::path geometric;
  std::filesystem::path geometric_average;
  std::filesystem::path conductor_fms;
  std::filesystem::path probability;
};

float lut_parameter(uint32_t index, uint32_t size) {
  return static_cast<float>(index) / static_cast<float>(size - 1u);
}

float mu_parameter(uint32_t index, uint32_t size) {
  if (index == 0u) {
    return 0.5f / static_cast<float>(size - 1u);
  }
  return lut_parameter(index, size);
}

float alpha_parameter(uint32_t index, uint32_t size) {
  const float axis = lut_parameter(index, size);
  return kBSDFNormalDistributionMinAlpha + (1.0f - kBSDFNormalDistributionMinAlpha) * axis * axis;
}

float radical_inverse_vdc(uint32_t bits) {
  bits = (bits << 16u) | (bits >> 16u);
  bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xaaaaaaaau) >> 1u);
  bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xccccccccu) >> 2u);
  bits = ((bits & 0x0f0f0f0fu) << 4u) | ((bits & 0xf0f0f0f0u) >> 4u);
  bits = ((bits & 0x00ff00ffu) << 8u) | ((bits & 0xff00ff00u) >> 8u);
  return static_cast<float>(bits) * 2.3283064365386963e-10f;
}

float2 hammersley(uint32_t index, uint32_t count) {
  return float2{(static_cast<float>(index) + 0.5f) / static_cast<float>(count), radical_inverse_vdc(index)};
}

float quasi_random(uint32_t index, uint32_t dimension) {
  const uint32_t seed = index + 1u + dimension * 0x9e3779b9u;
  return min(1.0f - kEpsilon, max(kEpsilon, radical_inverse_vdc(seed)));
}

float2 quasi_random_2d(uint32_t index, uint32_t dimension) {
  return float2{quasi_random(index, dimension), quasi_random(index, dimension + 1u)};
}

SpectralQuery spectral_cache_query(uint32_t wavelength_index) {
  const float t = static_cast<float>(min(wavelength_index, kEnergyCompensationSpectralWavelengthCount - 1u)) /
                  static_cast<float>(kEnergyCompensationSpectralWavelengthCount - 1u);
  SpectralQuery result = {};
  result.wavelength = kShortestWavelength + (kLongestWavelength - kShortestWavelength) * t;
  result.flags = SpectralFlags::Spectral;
  return result;
}

void set_float4_channel(float4& value, uint32_t channel, float scalar) {
  if (channel == 0u) {
    value.x = scalar;
  } else if (channel == 1u) {
    value.y = scalar;
  } else if (channel == 2u) {
    value.z = scalar;
  } else {
    value.w = scalar;
  }
}

float3 incident_direction_from_mu(float mu) {
  return float3{sqrt(max(0.0f, 1.0f - mu * mu)), 0.0f, mu};
}

uint32_t dielectric_side(bool outside) {
  return outside ? 0u : 1u;
}

uint32_t dielectric_branch_index(uint32_t incident_side, uint32_t outgoing_side) {
  return incident_side * 2u + outgoing_side;
}

float3 sample_vndf_local(ETX_IN(float3, w_i), float alpha, ETX_IN(float2, rnd)) {
  const float3 w_i_11 = normalize(float3(alpha * w_i.x, alpha * w_i.y, w_i.z));
  const float2 slope_11 = bsdf_external_sample_p22_11(acos(saturate(w_i_11.z)), rnd, float2(alpha, alpha));

  const float phi = atan2(w_i_11.y, w_i_11.x);
  float2 slope = float2(cos(phi) * slope_11.x - sin(phi) * slope_11.y, sin(phi) * slope_11.x + cos(phi) * slope_11.y);
  slope.x *= alpha;
  slope.y *= alpha;

  if ((slope.x != slope.x) || (isinf(slope.x))) {
    if (w_i.z > 0.0f) {
      return float3(0.0f, 0.0f, 1.0f);
    }
    return normalize(float3(w_i.x, w_i.y, 0.0f));
  }

  return normalize(float3(-slope.x, -slope.y, 1.0f));
}

ThinfilmEval empty_thinfilm() {
  ThinfilmEval result = {};
  result.ior.cls = SpectralDistribution::Invalid;
  result.rgb_wavelengths = kRGBWavelengths;
  result.thickness = 0.0f;
  return result;
}

SpectralResponse spectrum_or_default(const SceneData& data, uint32_t index, const SpectralQuery& spect, float default_value) {
  if (index < data.spectrum_values.size()) {
    return data.spectrum_values[index].query(spect);
  }
  return SpectralResponse(spect, default_value);
}

RefractiveIndexSample sample_refractive_index(const SceneData& data, const RefractiveIndex& refractive_index, const SpectralQuery& spect) {
  RefractiveIndexSample result = {};
  result.cls = refractive_index.cls;
  result.eta = spectrum_or_default(data, refractive_index.eta_index, spect, 1.0f);
  result.k = spectrum_or_default(data, refractive_index.k_index, spect, 0.0f);
  return result;
}

ThinfilmEval sample_constant_thinfilm(const SceneData& data, const Thinfilm& thinfilm, const SpectralQuery& spect) {
  if (bsdf_resource_thinfilm_enabled(thinfilm) == false) {
    return empty_thinfilm();
  }

  ThinfilmEval result = {};
  result.ior = sample_refractive_index(data, thinfilm.ior, spect);
  result.rgb_wavelengths = kRGBWavelengths;
  result.thickness = thinfilm.min_thickness;
  return result;
}

uint32_t thinfilm_lut_slice_count(const Thinfilm& thinfilm) {
  if (bsdf_resource_thinfilm_enabled(thinfilm) == false) {
    return 1u;
  }

  const float thickness_range = fabsf(thinfilm.max_thickness - thinfilm.min_thickness);
  if (thickness_range <= kEpsilon) {
    return 1u;
  }

  const uint32_t estimated_count = static_cast<uint32_t>(ceilf(thickness_range / 50.0f)) + 1u;
  return clamp(estimated_count, 4u, 32u);
}

ThinfilmEval sample_thinfilm_slice(const SceneData& data, const Thinfilm& thinfilm, const SpectralQuery& spect, uint32_t slice_index, uint32_t slice_count) {
  ThinfilmEval result = sample_constant_thinfilm(data, thinfilm, spect);
  if ((bsdf_resource_thinfilm_enabled(thinfilm) == false) || (slice_count <= 1u)) {
    return result;
  }

  const float t = static_cast<float>(slice_index) / static_cast<float>(slice_count - 1u);
  result.thickness = thinfilm.min_thickness + (thinfilm.max_thickness - thinfilm.min_thickness) * t;
  return result;
}

uint64_t hash_spectrum(const SceneData& data, uint32_t index, uint64_t seed) {
  if (index >= data.spectrum_values.size()) {
    return etx_hash64_continue(&index, sizeof(index), seed);
  }
  return etx_hash64_continue(data.spectrum_values.data() + index, sizeof(SpectralDistribution), seed);
}

uint64_t hash_refractive_index(const SceneData& data, const RefractiveIndex& refractive_index, uint64_t seed) {
  uint64_t result = etx_hash64_continue(&refractive_index.cls, sizeof(refractive_index.cls), seed);
  result = hash_spectrum(data, refractive_index.eta_index, result);
  result = hash_spectrum(data, refractive_index.k_index, result);
  return result;
}

uint64_t hash_thinfilm(const SceneData& data, const Thinfilm& thinfilm, uint64_t seed) {
  const bool enabled = bsdf_resource_thinfilm_enabled(thinfilm);
  uint64_t result = etx_hash64_continue(&enabled, sizeof(enabled), seed);
  if (enabled == false) {
    return result;
  }

  result = hash_refractive_index(data, thinfilm.ior, result);
  result = etx_hash64_continue(&thinfilm.min_thickness, sizeof(thinfilm.min_thickness), result);
  result = etx_hash64_continue(&thinfilm.max_thickness, sizeof(thinfilm.max_thickness), result);
  result = etx_hash64_continue(&thinfilm.thinkness_image, sizeof(thinfilm.thinkness_image), result);
  const uint32_t slice_count = thinfilm_lut_slice_count(thinfilm);
  result = etx_hash64_continue(&slice_count, sizeof(slice_count), result);
  return result;
}

uint64_t hash_material_interface(const SceneData& data, const Material& material, uint32_t material_class, uint32_t cache_mode) {
  uint64_t result = 0u;
  result = etx_hash64_continue(&kEnergyCompensationGeneratorVersion, sizeof(kEnergyCompensationGeneratorVersion), result);
  result = etx_hash64_continue(&material_class, sizeof(material_class), result);
  result = etx_hash64_continue(&cache_mode, sizeof(cache_mode), result);
  result = etx_hash64_continue(&kEnergyCompensationConductorLutSize, sizeof(kEnergyCompensationConductorLutSize), result);
  result = etx_hash64_continue(&kEnergyCompensationDielectricLutSize, sizeof(kEnergyCompensationDielectricLutSize), result);
  result = etx_hash64_continue(&kEnergyCompensationConductorSampleCount, sizeof(kEnergyCompensationConductorSampleCount), result);
  result = etx_hash64_continue(&kEnergyCompensationSampleCount, sizeof(kEnergyCompensationSampleCount), result);
  result = etx_hash64_continue(&kEnergyCompensationDielectricMultiScatterSampleCount, sizeof(kEnergyCompensationDielectricMultiScatterSampleCount), result);
  result = etx_hash64_continue(&kEnergyCompensationDielectricBranchCount, sizeof(kEnergyCompensationDielectricBranchCount), result);
  result = etx_hash64_continue(&kEnergyCompensationDielectricAverageWidth, sizeof(kEnergyCompensationDielectricAverageWidth), result);
  const uint32_t thinfilm_slice_count = thinfilm_lut_slice_count(material.thinfilm);
  result = etx_hash64_continue(&thinfilm_slice_count, sizeof(thinfilm_slice_count), result);
  if (cache_mode == kBSDFEnergyCompensationCacheModeSpectralScalar) {
    result = etx_hash64_continue(&kEnergyCompensationSpectralWavelengthCount, sizeof(kEnergyCompensationSpectralWavelengthCount), result);
    result = etx_hash64_continue(&kEnergyCompensationSpectralWavelengthGroupSize, sizeof(kEnergyCompensationSpectralWavelengthGroupSize), result);
    result = etx_hash64_continue(&kShortestWavelength, sizeof(kShortestWavelength), result);
    result = etx_hash64_continue(&kLongestWavelength, sizeof(kLongestWavelength), result);
  }
  result = hash_refractive_index(data, material.ext_ior, result);
  result = hash_refractive_index(data, material.int_ior, result);
  result = hash_thinfilm(data, material.thinfilm, result);
  return result;
}

std::filesystem::path cache_directory() {
  return std::filesystem::path(env().file_in_data("cache/bsdf/energy_compensation"));
}

std::string hash_string(uint64_t hash) {
  char buffer[32] = {};
  snprintf(buffer, sizeof(buffer), "%016llx", static_cast<unsigned long long>(hash));
  return std::string(buffer);
}

GeneratedInterfacePaths interface_paths(uint32_t material_class, uint64_t hash) {
  const std::string prefix = (material_class == MaterialClass::Conductor) ? "ec_conductor_" : "ec_dielectric_";
  const std::string key = prefix + hash_string(hash);
  const std::filesystem::path directory = cache_directory();
  return {
    directory / (key + ".exr"),
    directory / (key + "_average.exr"),
    directory / (key + "_geometric.exr"),
    directory / (key + "_geometric_average.exr"),
    directory / (key + "_conductor_fms.exr"),
    directory / (key + "_probability.exr"),
  };
}

std::filesystem::path slice_path(const std::filesystem::path& path, uint32_t slice_index) {
  if (slice_index == 0u) {
    return path;
  }

  const std::string stem = path.stem().generic_string() + "_slice_" + std::to_string(slice_index);
  return path.parent_path() / (stem + path.extension().generic_string());
}

GeneratedInterfacePaths interface_slice_paths(const GeneratedInterfacePaths& paths, uint32_t slice_index) {
  return {
    slice_path(paths.directional, slice_index),
    slice_path(paths.average, slice_index),
    slice_path(paths.geometric, slice_index),
    slice_path(paths.geometric_average, slice_index),
    slice_path(paths.conductor_fms, slice_index),
    slice_path(paths.probability, slice_index),
  };
}

bool save_exr_rgba(const std::filesystem::path& path, const std::vector<float4>& pixels, uint32_t width, uint32_t height) {
  const uint64_t expected_pixel_count = static_cast<uint64_t>(width) * static_cast<uint64_t>(height);
  if (pixels.size() != expected_pixel_count) {
    log::error("Invalid energy-compensation LUT dimensions for %s", path.generic_string().c_str());
    return false;
  }

  const std::filesystem::path parent_path = path.parent_path();
  if (parent_path.empty() == false) {
    std::error_code ec = {};
    std::filesystem::create_directories(parent_path, ec);
    if (ec.value() != 0) {
      log::error("Failed to create energy-compensation LUT cache directory %s", parent_path.generic_string().c_str());
      return false;
    }
  }

  const std::string file_name = path.generic_string();
  const char* error = nullptr;
  if (SaveEXR(reinterpret_cast<const float*>(pixels.data()), static_cast<int>(width), static_cast<int>(height), 4, false, file_name.c_str(), &error) !=
      TINYEXR_SUCCESS) {
    log::error("Failed to save energy-compensation LUT %s: %s", file_name.c_str(), (error != nullptr) ? error : "unknown error");
    if (error != nullptr) {
      FreeEXRErrorMessage(error);
    }
    return false;
  }

  return true;
}

bool load_rgba32f_lut(const std::filesystem::path& path, uint32_t width, uint32_t height, std::vector<float4>& out_pixels) {
  std::vector<uint8_t> source_data;
  uint2 dimensions = {};
  const Image::Format format = load_data(path.generic_string().c_str(), source_data, dimensions);
  if ((format != Image::Format::RGBA32F) || (dimensions.x != width) || (dimensions.y != height)) {
    log::error("Failed to load energy-compensation LUT %s", path.generic_string().c_str());
    return false;
  }

  const uint64_t pixel_count = static_cast<uint64_t>(width) * static_cast<uint64_t>(height);
  if (source_data.size() != pixel_count * sizeof(float4)) {
    log::error("Invalid energy-compensation LUT size %s", path.generic_string().c_str());
    return false;
  }

  out_pixels.resize(static_cast<size_t>(pixel_count));
  memcpy(out_pixels.data(), source_data.data(), source_data.size());
  return true;
}

bool load_rgba32f_lut_slices(const GeneratedInterfacePaths& paths, uint32_t width, uint32_t height, uint32_t slice_count, std::vector<float4>& out_pixels,
  const std::filesystem::path GeneratedInterfacePaths::*member) {
  const uint64_t slice_pixel_count = static_cast<uint64_t>(width) * static_cast<uint64_t>(height);
  out_pixels.resize(static_cast<size_t>(slice_pixel_count * slice_count));
  for (uint32_t slice_index = 0u; slice_index < slice_count; ++slice_index) {
    const GeneratedInterfacePaths slice_paths_value = interface_slice_paths(paths, slice_index);
    std::vector<float4> slice_pixels;
    if (load_rgba32f_lut(slice_paths_value.*member, width, height, slice_pixels) == false) {
      return false;
    }
    std::copy(slice_pixels.begin(), slice_pixels.end(), out_pixels.begin() + static_cast<size_t>(slice_pixel_count * slice_index));
  }
  return true;
}

SpectralDirectionalAlbedoResult integrate_conductor_directional(const SpectralQuery& spect, const RefractiveIndexSample& ext_ior, const RefractiveIndexSample& int_ior,
  const ThinfilmEval& thinfilm, float mu_i, float alpha) {
  SpectralDirectionalAlbedoResult result = {};
  if (mu_i <= kEpsilon) {
    return result;
  }

  const float2 alpha2 = float2(alpha, alpha);
  const float3 w_i = incident_direction_from_mu(mu_i);
  const float lambda_i = bsdf_external_ray_info_make(w_i, alpha2).Lambda;
  const float g1_i = 1.0f / (1.0f + lambda_i);
  const auto texture = spectral_response_make(spect, 1.0f);

  for (uint32_t sample_index = 0u; sample_index < kEnergyCompensationConductorSampleCount; ++sample_index) {
    const float3 m = sample_vndf_local(w_i, alpha, hammersley(sample_index, kEnergyCompensationConductorSampleCount));
    const float i_dot_m = dot(w_i, m);
    if ((m.z <= kEpsilon) || (i_dot_m <= kEpsilon)) {
      continue;
    }

    const float3 w_o = -w_i + 2.0f * m * i_dot_m;
    if (w_o.z > 0.0f) {
      const float vndf_pdf = bsdf_energy_compensated_vndf_pdf(w_i, m, alpha);
      const float raw_specular_pdf = vndf_pdf / max(kEpsilon, 4.0f * dot(w_o, m));
      const BSDFEnergyCompensatedLobe lobe = bsdf_energy_compensated_conductor_base_lobe(spect, w_i, w_o, alpha, ext_ior, int_ior, thinfilm, texture);
      if ((raw_specular_pdf > kEpsilon) && (lobe.pdf > kEpsilon)) {
        result.albedo += lobe.bsdf.integrated / raw_specular_pdf;
        const float lambda_o = bsdf_external_ray_info_make(w_o, alpha2).Lambda;
        const float d = bsdf_external_d_ggx(m, alpha2);
        const float g2 = 1.0f / (1.0f + lambda_i + lambda_o);
        result.geometric_albedo += (d * g2 / (4.0f * w_i.z)) / raw_specular_pdf;
      }
      if (g1_i > kEpsilon) {
        result.visible_probability += 1.0f;
      }
    }
  }

  const float inv_sample_count = 1.0f / static_cast<float>(kEnergyCompensationConductorSampleCount);
  result.albedo = saturate(result.albedo * inv_sample_count);
  result.geometric_albedo = saturate(result.geometric_albedo * inv_sample_count);
  result.visible_probability = saturate(result.visible_probability * inv_sample_count);
  return result;
}

DielectricDirectionalAlbedoResult integrate_dielectric_directional(const SpectralQuery& spect, const RefractiveIndexSample& ext_ior, const RefractiveIndexSample& int_ior,
  const ThinfilmEval& thinfilm, bool incident_outside, float mu_i, float alpha) {
  DielectricDirectionalAlbedoResult result = {};
  if (mu_i <= kEpsilon) {
    return result;
  }

  const float3 w_i = incident_direction_from_mu(mu_i);
  const float eta = spectral_response_monochromatic(spectral_response_div(int_ior.eta, ext_ior.eta));
  const auto texture = spectral_response_make(spect, 1.0f);
  const uint32_t incident_side = dielectric_side(incident_outside);
  const uint32_t opposite_side = 1u - incident_side;

  const bool no_thinfilm = (thinfilm.thickness <= 0.0f) || spectral_response_is_zero(thinfilm.ior.eta);
  if ((no_thinfilm) && (abs(eta - 1.0f) <= (16.0f * kEpsilon))) {
    result.branch_albedo[opposite_side] = float3(1.0f, 1.0f, 1.0f);
    result.branch_visible_probability[opposite_side] = 1.0f;
    result.visible_probability = 1.0f;
    return result;
  }

  for (uint32_t sample_index = 0u; sample_index < kEnergyCompensationSampleCount; ++sample_index) {
    const float3 m = sample_vndf_local(w_i, alpha, hammersley(sample_index, kEnergyCompensationSampleCount));
    const float i_dot_m = dot(w_i, m);
    if (i_dot_m <= kEpsilon) {
      continue;
    }

    const auto fresnel = bsdf_fresnel_calculate(spect, i_dot_m, ext_ior, int_ior, thinfilm);
    const float fresnel_probability = spectral_response_monochromatic(fresnel);
    const float cos_theta_t2 = 1.0f - (1.0f - i_dot_m * i_dot_m) / (eta * eta);

    const float3 w_o_r = -w_i + 2.0f * m * i_dot_m;
    if (w_o_r.z > 0.0f) {
      const auto lobe = bsdf_energy_compensated_dielectric_base_lobe(spect, w_i, w_o_r, alpha, ext_ior, int_ior, thinfilm, texture);
      if ((fresnel_probability > kEpsilon) && (lobe.pdf > kEpsilon)) {
        result.branch_albedo[incident_side] += lobe.bsdf.integrated * (fresnel_probability / lobe.pdf);
        result.branch_visible_probability[incident_side] += fresnel_probability;
      }
    }

    if ((fresnel_probability < 1.0f) && (cos_theta_t2 > 0.0f)) {
      const float3 w_o_t = normalize(bsdf_external_refract(w_i, m, eta));
      if (w_o_t.z < 0.0f) {
        const auto lobe = bsdf_energy_compensated_dielectric_base_lobe(spect, w_i, w_o_t, alpha, ext_ior, int_ior, thinfilm, texture);
        const float transmission_probability = 1.0f - fresnel_probability;
        if ((transmission_probability > kEpsilon) && (lobe.pdf > kEpsilon)) {
          result.branch_albedo[opposite_side] += lobe.bsdf.integrated * (transmission_probability / lobe.pdf);
          result.branch_visible_probability[opposite_side] += transmission_probability;
        }
      }
    }
  }

  const float inv_sample_count = 1.0f / static_cast<float>(kEnergyCompensationSampleCount);
  result.branch_albedo[0] = saturate(result.branch_albedo[0] * inv_sample_count);
  result.branch_albedo[1] = saturate(result.branch_albedo[1] * inv_sample_count);
  result.branch_visible_probability[0] = saturate(result.branch_visible_probability[0] * inv_sample_count);
  result.branch_visible_probability[1] = saturate(result.branch_visible_probability[1] * inv_sample_count);
  result.visible_probability = saturate(result.branch_visible_probability[0] + result.branch_visible_probability[1]);
  return result;
}

DielectricDirectionalAlbedoResult integrate_dielectric_total_directional(const SpectralQuery& spect, const RefractiveIndexSample& ext_ior, const RefractiveIndexSample& int_ior,
  const ThinfilmEval& thinfilm, bool incident_outside, float mu_i, float alpha) {
  DielectricDirectionalAlbedoResult result = {};
  if (mu_i <= kEpsilon) {
    return result;
  }

  const float2 alpha2 = float2(alpha, alpha);
  const float3 w_i = incident_direction_from_mu(mu_i);
  const uint32_t incident_side = dielectric_side(incident_outside);
  const uint32_t opposite_side = 1u - incident_side;
  const float eta = spectral_response_monochromatic(spectral_response_div(int_ior.eta, ext_ior.eta));

  const bool no_thinfilm = (thinfilm.thickness <= 0.0f) || spectral_response_is_zero(thinfilm.ior.eta);
  if ((no_thinfilm) && (abs(eta - 1.0f) <= (16.0f * kEpsilon))) {
    result.branch_albedo[opposite_side] = float3(1.0f, 1.0f, 1.0f);
    result.visible_probability = 1.0f;
    return result;
  }

  for (uint32_t sample_index = 0u; sample_index < kEnergyCompensationDielectricMultiScatterSampleCount; ++sample_index) {
    BSDFExternalRayInfo ray = bsdf_external_ray_info_make(-w_i, alpha2);
    ray = bsdf_external_ray_info_update_height(ray, 1.0f);
    bool ray_outside = true;
    bool valid = true;
    uint32_t scattering_order = 0u;
    uint32_t dimension = 0u;

    while (valid) {
      const float sampled_height = bsdf_external_sample_height(ray, quasi_random(sample_index, dimension));
      dimension += 1u;
      if (sampled_height == kMaxFloat) {
        break;
      }

      ray = bsdf_external_ray_info_update_height(ray, sampled_height);

      const float2 rnd_slope = quasi_random_2d(sample_index, dimension);
      dimension += 2u;
      const float rnd_reflection = quasi_random(sample_index, dimension);
      dimension += 1u;
      const RefractiveIndexSample& phase_ext_ior = ray_outside ? ext_ior : int_ior;
      const RefractiveIndexSample& phase_int_ior = ray_outside ? int_ior : ext_ior;
      const BSDFExternalDielectricSample sample =
        bsdf_external_sample_phase_function_dielectric(spect, rnd_slope, rnd_reflection, -ray.w, alpha2, phase_ext_ior, phase_int_ior, thinfilm);

      if (sample.reflection) {
        ray = bsdf_external_ray_info_update_direction(ray, sample.w_o, alpha2);
        ray = bsdf_external_ray_info_update_height(ray, ray.h);
      } else {
        ray_outside = (ray_outside == false);
        ray = bsdf_external_ray_info_update_direction(ray, -sample.w_o, alpha2);
        ray = bsdf_external_ray_info_update_height(ray, -ray.h);
      }

      scattering_order += 1u;
      if ((scattering_order > kBSDFExternalScatteringOrderMax) || (ray.h != ray.h) || (ray.w.x != ray.w.x)) {
        valid = false;
      }
    }

    if (valid == false) {
      continue;
    }

    const float3 local_w_o = ray_outside ? ray.w : -ray.w;
    const uint32_t outgoing_side = (local_w_o.z > 0.0f) ? incident_side : opposite_side;
    result.branch_albedo[outgoing_side] += float3(1.0f, 1.0f, 1.0f);
    result.visible_probability += 1.0f;
  }

  const float inv_sample_count = 1.0f / static_cast<float>(kEnergyCompensationDielectricMultiScatterSampleCount);
  result.branch_albedo[0] = saturate(result.branch_albedo[0] * inv_sample_count);
  result.branch_albedo[1] = saturate(result.branch_albedo[1] * inv_sample_count);
  result.visible_probability = saturate(result.visible_probability * inv_sample_count);
  return result;
}

float3 integrate_average(const std::vector<SpectralDirectionalAlbedoResult>& directional, uint32_t alpha_index, uint32_t lut_size, uint32_t side_offset) {
  float3 total = {};
  const float h = 1.0f / static_cast<float>(lut_size - 1u);
  for (uint32_t mu_index = 0u; mu_index < (lut_size - 1u); ++mu_index) {
    const float mu = static_cast<float>(mu_index) * h;
    const float weight_0 = h * mu + h * h / 3.0f;
    const float weight_1 = h * mu + 2.0f * h * h / 3.0f;
    const float3 value_0 = directional[side_offset + alpha_index * lut_size + mu_index].albedo;
    const float3 value_1 = directional[side_offset + alpha_index * lut_size + mu_index + 1u].albedo;
    total += value_0 * weight_0 + value_1 * weight_1;
  }
  return saturate(total);
}

float3 integrate_dielectric_branch_average(const std::vector<DielectricDirectionalAlbedoResult>& directional, uint32_t alpha_index, uint32_t incident_side,
  uint32_t outgoing_side) {
  float3 total = {};
  const float h = 1.0f / static_cast<float>(kEnergyCompensationDielectricLutSize - 1u);
  const uint32_t side_entry_count = kEnergyCompensationDielectricLutSize * kEnergyCompensationDielectricLutSize;
  const uint32_t side_offset = incident_side * side_entry_count;
  for (uint32_t mu_index = 0u; mu_index < (kEnergyCompensationDielectricLutSize - 1u); ++mu_index) {
    const float mu = static_cast<float>(mu_index) * h;
    const float weight_0 = h * mu + h * h / 3.0f;
    const float weight_1 = h * mu + 2.0f * h * h / 3.0f;
    const float3 value_0 = directional[side_offset + alpha_index * kEnergyCompensationDielectricLutSize + mu_index].branch_albedo[outgoing_side];
    const float3 value_1 = directional[side_offset + alpha_index * kEnergyCompensationDielectricLutSize + mu_index + 1u].branch_albedo[outgoing_side];
    total += value_0 * weight_0 + value_1 * weight_1;
  }
  return saturate(total);
}

void clamp_dielectric_residual_row(float3 residual_average[kEnergyCompensationDielectricBranchCount], const float3 side_residual[2]) {
  for (uint32_t incident_side = 0u; incident_side < 2u; ++incident_side) {
    const uint32_t branch_0 = dielectric_branch_index(incident_side, 0u);
    const uint32_t branch_1 = dielectric_branch_index(incident_side, 1u);
    const float3 residual_sum = residual_average[branch_0] + residual_average[branch_1];
    const float3 scale = min(float3(1.0f, 1.0f, 1.0f), side_residual[incident_side] / max(float3(kEpsilon, kEpsilon, kEpsilon), residual_sum));
    residual_average[branch_0] *= scale;
    residual_average[branch_1] *= scale;
  }
}

float integrate_geometric_average(const std::vector<SpectralDirectionalAlbedoResult>& directional, uint32_t alpha_index, uint32_t lut_size) {
  float total = 0.0f;
  const float h = 1.0f / static_cast<float>(lut_size - 1u);
  for (uint32_t mu_index = 0u; mu_index < (lut_size - 1u); ++mu_index) {
    const float mu = static_cast<float>(mu_index) * h;
    const float weight_0 = h * mu + h * h / 3.0f;
    const float weight_1 = h * mu + 2.0f * h * h / 3.0f;
    const float value_0 = directional[alpha_index * lut_size + mu_index].geometric_albedo;
    const float value_1 = directional[alpha_index * lut_size + mu_index + 1u].geometric_albedo;
    total += value_0 * weight_0 + value_1 * weight_1;
  }
  return saturate(total);
}

bool generate_conductor_interface(const GeneratedInterfacePaths& paths, const RefractiveIndexSample& ext_ior, const RefractiveIndexSample& int_ior, const ThinfilmEval& thinfilm,
  TaskScheduler& scheduler) {
  const SpectralQuery spect = {};
  const uint32_t entry_count = kEnergyCompensationConductorLutSize * kEnergyCompensationConductorLutSize;
  std::vector<SpectralDirectionalAlbedoResult> directional(entry_count);
  scheduler.execute(entry_count, [&](uint32_t begin, uint32_t end, uint32_t thread_id) {
    (void)thread_id;
    for (uint32_t index = begin; index < end; ++index) {
      const uint32_t alpha_index = index / kEnergyCompensationConductorLutSize;
      const uint32_t mu_index = index - alpha_index * kEnergyCompensationConductorLutSize;
      directional[index] = integrate_conductor_directional(spect, ext_ior, int_ior, thinfilm, mu_parameter(mu_index, kEnergyCompensationConductorLutSize),
        alpha_parameter(alpha_index, kEnergyCompensationConductorLutSize));
    }
  });

  std::vector<float4> image(entry_count, float4{0.0f, 0.0f, 0.0f, 0.0f});
  std::vector<float4> average(kEnergyCompensationConductorLutSize, float4{0.0f, 0.0f, 0.0f, 1.0f});
  std::vector<float4> geometric_image(entry_count, float4{0.0f, 0.0f, 0.0f, 1.0f});
  std::vector<float4> geometric_average(kEnergyCompensationConductorLutSize, float4{0.0f, 0.0f, 0.0f, 1.0f});
  std::vector<float4> conductor_fms(kEnergyCompensationConductorLutSize, float4{0.0f, 0.0f, 0.0f, 1.0f});
  for (uint32_t alpha_index = 0u; alpha_index < kEnergyCompensationConductorLutSize; ++alpha_index) {
    const float3 average_value = integrate_average(directional, alpha_index, kEnergyCompensationConductorLutSize, 0u);
    average[alpha_index] = float4(average_value.x, average_value.y, average_value.z, 1.0f);
    for (uint32_t mu_index = 0u; mu_index < kEnergyCompensationConductorLutSize; ++mu_index) {
      const uint32_t index = alpha_index * kEnergyCompensationConductorLutSize + mu_index;
      const SpectralDirectionalAlbedoResult& value = directional[index];
      image[index] = float4(value.albedo.x, value.albedo.y, value.albedo.z, value.visible_probability);
      geometric_image[index] = float4(value.geometric_albedo, value.visible_probability, 0.0f, 1.0f);
    }
    const float geometric_average_value = integrate_geometric_average(directional, alpha_index, kEnergyCompensationConductorLutSize);
    geometric_average[alpha_index] = float4(geometric_average_value, 0.0f, 0.0f, 1.0f);
    const ::SpectralResponse fms = bsdf_energy_compensated_conductor_fms(spect, ext_ior, int_ior, thinfilm, geometric_average_value);
    conductor_fms[alpha_index] = float4(fms.integrated.x, fms.integrated.y, fms.integrated.z, 1.0f);
  }

  return save_exr_rgba(paths.directional, image, kEnergyCompensationConductorLutSize, kEnergyCompensationConductorLutSize) &&
         save_exr_rgba(paths.average, average, kEnergyCompensationConductorLutSize, 1u) &&
         save_exr_rgba(paths.geometric, geometric_image, kEnergyCompensationConductorLutSize, kEnergyCompensationConductorLutSize) &&
         save_exr_rgba(paths.geometric_average, geometric_average, kEnergyCompensationConductorLutSize, 1u) &&
         save_exr_rgba(paths.conductor_fms, conductor_fms, kEnergyCompensationConductorLutSize, 1u);
}

bool generate_conductor_geometric_interface(const GeneratedInterfacePaths& paths, const RefractiveIndexSample& ext_ior, const RefractiveIndexSample& int_ior,
  const ThinfilmEval& thinfilm, TaskScheduler& scheduler) {
  const SpectralQuery spect = {};
  const uint32_t entry_count = kEnergyCompensationConductorLutSize * kEnergyCompensationConductorLutSize;
  std::vector<SpectralDirectionalAlbedoResult> directional(entry_count);
  scheduler.execute(entry_count, [&](uint32_t begin, uint32_t end, uint32_t thread_id) {
    (void)thread_id;
    for (uint32_t index = begin; index < end; ++index) {
      const uint32_t alpha_index = index / kEnergyCompensationConductorLutSize;
      const uint32_t mu_index = index - alpha_index * kEnergyCompensationConductorLutSize;
      directional[index] = integrate_conductor_directional(spect, ext_ior, int_ior, thinfilm, mu_parameter(mu_index, kEnergyCompensationConductorLutSize),
        alpha_parameter(alpha_index, kEnergyCompensationConductorLutSize));
    }
  });

  std::vector<float4> geometric_image(entry_count, float4{0.0f, 0.0f, 0.0f, 1.0f});
  std::vector<float4> geometric_average(kEnergyCompensationConductorLutSize, float4{0.0f, 0.0f, 0.0f, 1.0f});
  for (uint32_t alpha_index = 0u; alpha_index < kEnergyCompensationConductorLutSize; ++alpha_index) {
    for (uint32_t mu_index = 0u; mu_index < kEnergyCompensationConductorLutSize; ++mu_index) {
      const uint32_t index = alpha_index * kEnergyCompensationConductorLutSize + mu_index;
      const SpectralDirectionalAlbedoResult& value = directional[index];
      geometric_image[index] = float4(value.geometric_albedo, value.visible_probability, 0.0f, 1.0f);
    }
    const float geometric_average_value = integrate_geometric_average(directional, alpha_index, kEnergyCompensationConductorLutSize);
    geometric_average[alpha_index] = float4(geometric_average_value, 0.0f, 0.0f, 1.0f);
  }

  return save_exr_rgba(paths.geometric, geometric_image, kEnergyCompensationConductorLutSize, kEnergyCompensationConductorLutSize) &&
         save_exr_rgba(paths.geometric_average, geometric_average, kEnergyCompensationConductorLutSize, 1u);
}

bool generate_conductor_interface_spectral(const SceneData& data, const Material& material, const GeneratedInterfacePaths& paths, uint32_t thinfilm_slice_index,
  uint32_t thinfilm_slice_count, uint32_t wavelength_group_index, TaskScheduler& scheduler) {
  const uint32_t entry_count = kEnergyCompensationConductorLutSize * kEnergyCompensationConductorLutSize;
  std::vector<float4> image(entry_count, float4{0.0f, 0.0f, 0.0f, 0.0f});
  std::vector<float4> average(kEnergyCompensationConductorLutSize, float4{0.0f, 0.0f, 0.0f, 0.0f});
  std::vector<float4> conductor_fms(kEnergyCompensationConductorLutSize, float4{0.0f, 0.0f, 0.0f, 0.0f});

  for (uint32_t channel = 0u; channel < kEnergyCompensationSpectralWavelengthGroupSize; ++channel) {
    const uint32_t wavelength_index = wavelength_group_index * kEnergyCompensationSpectralWavelengthGroupSize + channel;
    const SpectralQuery spect = spectral_cache_query(wavelength_index);
    const RefractiveIndexSample ext_ior = sample_refractive_index(data, material.ext_ior, spect);
    const RefractiveIndexSample int_ior = sample_refractive_index(data, material.int_ior, spect);
    const ThinfilmEval thinfilm = sample_thinfilm_slice(data, material.thinfilm, spect, thinfilm_slice_index, thinfilm_slice_count);
    std::vector<SpectralDirectionalAlbedoResult> directional(entry_count);
    scheduler.execute(entry_count, [&](uint32_t begin, uint32_t end, uint32_t thread_id) {
      (void)thread_id;
      for (uint32_t index = begin; index < end; ++index) {
        const uint32_t alpha_index = index / kEnergyCompensationConductorLutSize;
        const uint32_t mu_index = index - alpha_index * kEnergyCompensationConductorLutSize;
        directional[index] = integrate_conductor_directional(spect, ext_ior, int_ior, thinfilm, mu_parameter(mu_index, kEnergyCompensationConductorLutSize),
          alpha_parameter(alpha_index, kEnergyCompensationConductorLutSize));
      }
    });

    for (uint32_t alpha_index = 0u; alpha_index < kEnergyCompensationConductorLutSize; ++alpha_index) {
      const float3 average_value = integrate_average(directional, alpha_index, kEnergyCompensationConductorLutSize, 0u);
      set_float4_channel(average[alpha_index], channel, average_value.x);
      for (uint32_t mu_index = 0u; mu_index < kEnergyCompensationConductorLutSize; ++mu_index) {
        const uint32_t index = alpha_index * kEnergyCompensationConductorLutSize + mu_index;
        set_float4_channel(image[index], channel, directional[index].albedo.x);
      }

      const float geometric_average_value = integrate_geometric_average(directional, alpha_index, kEnergyCompensationConductorLutSize);
      const ::SpectralResponse fms = bsdf_energy_compensated_conductor_fms(spect, ext_ior, int_ior, thinfilm, geometric_average_value);
      set_float4_channel(conductor_fms[alpha_index], channel, spectral_response_monochromatic(fms));
    }
  }

  return save_exr_rgba(paths.directional, image, kEnergyCompensationConductorLutSize, kEnergyCompensationConductorLutSize) &&
         save_exr_rgba(paths.average, average, kEnergyCompensationConductorLutSize, 1u) &&
         save_exr_rgba(paths.conductor_fms, conductor_fms, kEnergyCompensationConductorLutSize, 1u);
}

bool generate_dielectric_interface(const GeneratedInterfacePaths& paths, const RefractiveIndexSample& ext_ior, const RefractiveIndexSample& int_ior, const ThinfilmEval& thinfilm,
  TaskScheduler& scheduler) {
  const SpectralQuery spect = {};
  const uint32_t side_entry_count = kEnergyCompensationDielectricLutSize * kEnergyCompensationDielectricLutSize;
  std::vector<DielectricDirectionalAlbedoResult> directional(side_entry_count * 2u);
  std::vector<DielectricDirectionalAlbedoResult> total_directional(side_entry_count * 2u);
  scheduler.execute(side_entry_count * 2u, [&](uint32_t begin, uint32_t end, uint32_t thread_id) {
    (void)thread_id;
    for (uint32_t index = begin; index < end; ++index) {
      const uint32_t side = index / side_entry_count;
      const uint32_t side_index = index - side * side_entry_count;
      const uint32_t alpha_index = side_index / kEnergyCompensationDielectricLutSize;
      const uint32_t mu_index = side_index - alpha_index * kEnergyCompensationDielectricLutSize;
      const RefractiveIndexSample& source_ior = (side == 0u) ? ext_ior : int_ior;
      const RefractiveIndexSample& target_ior = (side == 0u) ? int_ior : ext_ior;
      directional[index] = integrate_dielectric_directional(spect, source_ior, target_ior, thinfilm, side == 0u, mu_parameter(mu_index, kEnergyCompensationDielectricLutSize),
        alpha_parameter(alpha_index, kEnergyCompensationDielectricLutSize));
    }
  });
  scheduler.execute(side_entry_count * 2u, [&](uint32_t begin, uint32_t end, uint32_t thread_id) {
    (void)thread_id;
    for (uint32_t index = begin; index < end; ++index) {
      const uint32_t side = index / side_entry_count;
      const uint32_t side_index = index - side * side_entry_count;
      const uint32_t alpha_index = side_index / kEnergyCompensationDielectricLutSize;
      const uint32_t mu_index = side_index - alpha_index * kEnergyCompensationDielectricLutSize;
      const RefractiveIndexSample& source_ior = (side == 0u) ? ext_ior : int_ior;
      const RefractiveIndexSample& target_ior = (side == 0u) ? int_ior : ext_ior;
      total_directional[index] = integrate_dielectric_total_directional(spect, source_ior, target_ior, thinfilm, side == 0u,
        mu_parameter(mu_index, kEnergyCompensationDielectricLutSize), alpha_parameter(alpha_index, kEnergyCompensationDielectricLutSize));
    }
  });

  const uint32_t width = kEnergyCompensationDielectricBranchCount * kEnergyCompensationDielectricLutSize;
  std::vector<float4> image(width * kEnergyCompensationDielectricLutSize, float4{0.0f, 0.0f, 0.0f, 0.0f});
  std::vector<float4> average(kEnergyCompensationDielectricAverageWidth * kEnergyCompensationDielectricLutSize, float4{0.0f, 0.0f, 0.0f, 1.0f});
  for (uint32_t alpha_index = 0u; alpha_index < kEnergyCompensationDielectricLutSize; ++alpha_index) {
    float3 single_average[kEnergyCompensationDielectricBranchCount] = {};
    float3 total_average[kEnergyCompensationDielectricBranchCount] = {};
    float3 residual_average[kEnergyCompensationDielectricBranchCount] = {};
    float3 residual_coefficient[kEnergyCompensationDielectricBranchCount] = {};

    for (uint32_t incident_side = 0u; incident_side < 2u; ++incident_side) {
      for (uint32_t outgoing_side = 0u; outgoing_side < 2u; ++outgoing_side) {
        const uint32_t branch = dielectric_branch_index(incident_side, outgoing_side);
        single_average[branch] = integrate_dielectric_branch_average(directional, alpha_index, incident_side, outgoing_side);
        total_average[branch] = integrate_dielectric_branch_average(total_directional, alpha_index, incident_side, outgoing_side);
      }
    }

    float3 side_residual[2] = {};
    for (uint32_t side = 0u; side < 2u; ++side) {
      const uint32_t branch_0 = dielectric_branch_index(side, 0u);
      const uint32_t branch_1 = dielectric_branch_index(side, 1u);
      const float3 row_single = saturate(single_average[branch_0] + single_average[branch_1]);
      side_residual[side] = max(float3(0.0f, 0.0f, 0.0f), float3(1.0f, 1.0f, 1.0f) - row_single);
    }

    for (uint32_t incident_side = 0u; incident_side < 2u; ++incident_side) {
      for (uint32_t outgoing_side = 0u; outgoing_side < 2u; ++outgoing_side) {
        const uint32_t branch = dielectric_branch_index(incident_side, outgoing_side);
        residual_average[branch] = max(float3(0.0f, 0.0f, 0.0f), total_average[branch] - single_average[branch]);
      }
    }

    clamp_dielectric_residual_row(residual_average, side_residual);

    for (uint32_t incident_side = 0u; incident_side < 2u; ++incident_side) {
      for (uint32_t outgoing_side = 0u; outgoing_side < 2u; ++outgoing_side) {
        const uint32_t branch = dielectric_branch_index(incident_side, outgoing_side);
        const float3 denominator = max(float3(kEpsilon, kEpsilon, kEpsilon), side_residual[incident_side] * side_residual[outgoing_side]);
        residual_coefficient[branch] = residual_average[branch] / denominator;
      }
    }

#if ETX_DEBUG
    const float validation_tolerance = 4.0f / sqrt(static_cast<float>(kEnergyCompensationSampleCount));
    for (uint32_t side = 0u; side < 2u; ++side) {
      const uint32_t branch_0 = dielectric_branch_index(side, 0u);
      const uint32_t branch_1 = dielectric_branch_index(side, 1u);
      const float3 single_row = single_average[branch_0] + single_average[branch_1];
      const float3 model_total_row = total_average[branch_0] + total_average[branch_1];
      const float3 total_row = single_row + residual_average[branch_0] + residual_average[branch_1];
      if ((isfinite(single_row.x) == false) || (isfinite(single_row.y) == false) || (isfinite(single_row.z) == false) || (isfinite(model_total_row.x) == false) ||
          (isfinite(model_total_row.y) == false) || (isfinite(model_total_row.z) == false) || (isfinite(total_row.x) == false) || (isfinite(total_row.y) == false) ||
          (isfinite(total_row.z) == false) || (single_row.x > (1.0f + validation_tolerance)) ||
          (single_row.y > (1.0f + validation_tolerance)) || (single_row.z > (1.0f + validation_tolerance)) || (total_row.x > (1.0f + validation_tolerance)) ||
          (total_row.y > (1.0f + validation_tolerance)) || (total_row.z > (1.0f + validation_tolerance)) ||
          (model_total_row.x > (1.0f + validation_tolerance)) || (model_total_row.y > (1.0f + validation_tolerance)) ||
          (model_total_row.z > (1.0f + validation_tolerance))) {
        log::error(
          "Invalid dielectric energy-compensation row alpha %u side %u single %.6f %.6f %.6f model %.6f %.6f %.6f total %.6f %.6f %.6f tolerance %.6f", alpha_index, side,
          single_row.x, single_row.y, single_row.z, model_total_row.x, model_total_row.y, model_total_row.z, total_row.x, total_row.y, total_row.z, validation_tolerance);
        return false;
      }
    }
    for (uint32_t branch = 0u; branch < kEnergyCompensationDielectricBranchCount; ++branch) {
      const float3 coefficient = residual_coefficient[branch];
      if ((isfinite(coefficient.x) == false) || (isfinite(coefficient.y) == false) || (isfinite(coefficient.z) == false) || (coefficient.x < 0.0f) || (coefficient.y < 0.0f) ||
          (coefficient.z < 0.0f)) {
        log::error("Invalid dielectric energy-compensation coefficient alpha %u branch %u coefficient %.6f %.6f %.6f", alpha_index, branch, coefficient.x, coefficient.y,
          coefficient.z);
        return false;
      }
    }
#endif

    for (uint32_t branch = 0u; branch < kEnergyCompensationDielectricBranchCount; ++branch) {
      const float3 average_value = saturate(single_average[branch]);
      average[alpha_index * kEnergyCompensationDielectricAverageWidth + branch] = float4(average_value.x, average_value.y, average_value.z, 1.0f);
      const float3 coefficient = max(float3(0.0f, 0.0f, 0.0f), residual_coefficient[branch]);
      average[alpha_index * kEnergyCompensationDielectricAverageWidth + 4u + branch] = float4(coefficient.x, coefficient.y, coefficient.z, 1.0f);
    }

    for (uint32_t side = 0u; side < 2u; ++side) {
      const uint32_t side_offset = side * side_entry_count;
      for (uint32_t mu_index = 0u; mu_index < kEnergyCompensationDielectricLutSize; ++mu_index) {
        const uint32_t source_index = side_offset + alpha_index * kEnergyCompensationDielectricLutSize + mu_index;
        const DielectricDirectionalAlbedoResult& value = directional[source_index];
        for (uint32_t outgoing_side = 0u; outgoing_side < 2u; ++outgoing_side) {
          const uint32_t branch = dielectric_branch_index(side, outgoing_side);
          const uint32_t x = branch * kEnergyCompensationDielectricLutSize + mu_index;
          const float3 branch_value = value.branch_albedo[outgoing_side];
          image[alpha_index * width + x] = float4(branch_value.x, branch_value.y, branch_value.z, value.branch_visible_probability[outgoing_side]);
        }
      }
    }
  }

  return save_exr_rgba(paths.directional, image, width, kEnergyCompensationDielectricLutSize) &&
         save_exr_rgba(paths.average, average, kEnergyCompensationDielectricAverageWidth, kEnergyCompensationDielectricLutSize);
}

bool generate_dielectric_interface_spectral(const SceneData& data, const Material& material, const GeneratedInterfacePaths& paths, uint32_t thinfilm_slice_index,
  uint32_t thinfilm_slice_count, uint32_t wavelength_group_index, TaskScheduler& scheduler) {
  const uint32_t side_entry_count = kEnergyCompensationDielectricLutSize * kEnergyCompensationDielectricLutSize;
  const uint32_t width = kEnergyCompensationDielectricBranchCount * kEnergyCompensationDielectricLutSize;
  std::vector<float4> image(width * kEnergyCompensationDielectricLutSize, float4{0.0f, 0.0f, 0.0f, 0.0f});
  std::vector<float4> probability(width * kEnergyCompensationDielectricLutSize, float4{0.0f, 0.0f, 0.0f, 0.0f});
  std::vector<float4> average(kEnergyCompensationDielectricAverageWidth * kEnergyCompensationDielectricLutSize, float4{0.0f, 0.0f, 0.0f, 0.0f});

  for (uint32_t channel = 0u; channel < kEnergyCompensationSpectralWavelengthGroupSize; ++channel) {
    const uint32_t wavelength_index = wavelength_group_index * kEnergyCompensationSpectralWavelengthGroupSize + channel;
    const SpectralQuery spect = spectral_cache_query(wavelength_index);
    const RefractiveIndexSample ext_ior = sample_refractive_index(data, material.ext_ior, spect);
    const RefractiveIndexSample int_ior = sample_refractive_index(data, material.int_ior, spect);
    const ThinfilmEval thinfilm = sample_thinfilm_slice(data, material.thinfilm, spect, thinfilm_slice_index, thinfilm_slice_count);
    std::vector<DielectricDirectionalAlbedoResult> directional(side_entry_count * 2u);
    std::vector<DielectricDirectionalAlbedoResult> total_directional(side_entry_count * 2u);
    scheduler.execute(side_entry_count * 2u, [&](uint32_t begin, uint32_t end, uint32_t thread_id) {
      (void)thread_id;
      for (uint32_t index = begin; index < end; ++index) {
        const uint32_t side = index / side_entry_count;
        const uint32_t side_index = index - side * side_entry_count;
        const uint32_t alpha_index = side_index / kEnergyCompensationDielectricLutSize;
        const uint32_t mu_index = side_index - alpha_index * kEnergyCompensationDielectricLutSize;
        const RefractiveIndexSample& source_ior = (side == 0u) ? ext_ior : int_ior;
        const RefractiveIndexSample& target_ior = (side == 0u) ? int_ior : ext_ior;
        directional[index] = integrate_dielectric_directional(spect, source_ior, target_ior, thinfilm, side == 0u,
          mu_parameter(mu_index, kEnergyCompensationDielectricLutSize), alpha_parameter(alpha_index, kEnergyCompensationDielectricLutSize));
      }
    });
    scheduler.execute(side_entry_count * 2u, [&](uint32_t begin, uint32_t end, uint32_t thread_id) {
      (void)thread_id;
      for (uint32_t index = begin; index < end; ++index) {
        const uint32_t side = index / side_entry_count;
        const uint32_t side_index = index - side * side_entry_count;
        const uint32_t alpha_index = side_index / kEnergyCompensationDielectricLutSize;
        const uint32_t mu_index = side_index - alpha_index * kEnergyCompensationDielectricLutSize;
        const RefractiveIndexSample& source_ior = (side == 0u) ? ext_ior : int_ior;
        const RefractiveIndexSample& target_ior = (side == 0u) ? int_ior : ext_ior;
        total_directional[index] = integrate_dielectric_total_directional(spect, source_ior, target_ior, thinfilm, side == 0u,
          mu_parameter(mu_index, kEnergyCompensationDielectricLutSize), alpha_parameter(alpha_index, kEnergyCompensationDielectricLutSize));
      }
    });

    for (uint32_t alpha_index = 0u; alpha_index < kEnergyCompensationDielectricLutSize; ++alpha_index) {
      float3 single_average[kEnergyCompensationDielectricBranchCount] = {};
      float3 total_average[kEnergyCompensationDielectricBranchCount] = {};
      float3 residual_average[kEnergyCompensationDielectricBranchCount] = {};
      float3 residual_coefficient[kEnergyCompensationDielectricBranchCount] = {};

      for (uint32_t incident_side = 0u; incident_side < 2u; ++incident_side) {
        for (uint32_t outgoing_side = 0u; outgoing_side < 2u; ++outgoing_side) {
          const uint32_t branch = dielectric_branch_index(incident_side, outgoing_side);
          single_average[branch] = integrate_dielectric_branch_average(directional, alpha_index, incident_side, outgoing_side);
          total_average[branch] = integrate_dielectric_branch_average(total_directional, alpha_index, incident_side, outgoing_side);
        }
      }

      float3 side_residual[2] = {};
      for (uint32_t side = 0u; side < 2u; ++side) {
        const uint32_t branch_0 = dielectric_branch_index(side, 0u);
        const uint32_t branch_1 = dielectric_branch_index(side, 1u);
        const float3 row_single = saturate(single_average[branch_0] + single_average[branch_1]);
        side_residual[side] = max(float3(0.0f, 0.0f, 0.0f), float3(1.0f, 1.0f, 1.0f) - row_single);
      }

      for (uint32_t incident_side = 0u; incident_side < 2u; ++incident_side) {
        for (uint32_t outgoing_side = 0u; outgoing_side < 2u; ++outgoing_side) {
          const uint32_t branch = dielectric_branch_index(incident_side, outgoing_side);
          residual_average[branch] = max(float3(0.0f, 0.0f, 0.0f), total_average[branch] - single_average[branch]);
        }
      }

      clamp_dielectric_residual_row(residual_average, side_residual);

      for (uint32_t incident_side = 0u; incident_side < 2u; ++incident_side) {
        for (uint32_t outgoing_side = 0u; outgoing_side < 2u; ++outgoing_side) {
          const uint32_t branch = dielectric_branch_index(incident_side, outgoing_side);
          const float3 denominator = max(float3(kEpsilon, kEpsilon, kEpsilon), side_residual[incident_side] * side_residual[outgoing_side]);
          residual_coefficient[branch] = residual_average[branch] / denominator;
        }
      }

      for (uint32_t branch = 0u; branch < kEnergyCompensationDielectricBranchCount; ++branch) {
        set_float4_channel(average[alpha_index * kEnergyCompensationDielectricAverageWidth + branch], channel, saturate(single_average[branch].x));
        set_float4_channel(average[alpha_index * kEnergyCompensationDielectricAverageWidth + 4u + branch], channel, max(0.0f, residual_coefficient[branch].x));
      }

      for (uint32_t side = 0u; side < 2u; ++side) {
        const uint32_t side_offset = side * side_entry_count;
        for (uint32_t mu_index = 0u; mu_index < kEnergyCompensationDielectricLutSize; ++mu_index) {
          const uint32_t source_index = side_offset + alpha_index * kEnergyCompensationDielectricLutSize + mu_index;
          const DielectricDirectionalAlbedoResult& value = directional[source_index];
          for (uint32_t outgoing_side = 0u; outgoing_side < 2u; ++outgoing_side) {
            const uint32_t branch = dielectric_branch_index(side, outgoing_side);
            const uint32_t x = branch * kEnergyCompensationDielectricLutSize + mu_index;
            const uint32_t image_index = alpha_index * width + x;
            set_float4_channel(image[image_index], channel, saturate(value.branch_albedo[outgoing_side].x));
            set_float4_channel(probability[image_index], channel, saturate(value.branch_visible_probability[outgoing_side]));
          }
        }
      }
    }
  }

  return save_exr_rgba(paths.directional, image, width, kEnergyCompensationDielectricLutSize) &&
         save_exr_rgba(paths.average, average, kEnergyCompensationDielectricAverageWidth, kEnergyCompensationDielectricLutSize) &&
         save_exr_rgba(paths.probability, probability, width, kEnergyCompensationDielectricLutSize);
}

bool ensure_cache_file(const SceneData& data, const Material& material, uint32_t material_class, uint32_t cache_mode, const GeneratedInterfacePaths& paths,
  TaskScheduler& scheduler) {
  const bool conductor = material_class == MaterialClass::Conductor;
  const SpectralQuery spect = {};
  const RefractiveIndexSample ext_ior = sample_refractive_index(data, material.ext_ior, spect);
  const RefractiveIndexSample int_ior = sample_refractive_index(data, material.int_ior, spect);
  const uint32_t slice_count = thinfilm_lut_slice_count(material.thinfilm);
  if (cache_mode == kBSDFEnergyCompensationCacheModeSpectralScalar) {
    bool spectral_result = true;
    for (uint32_t thinfilm_slice_index = 0u; thinfilm_slice_index < slice_count; ++thinfilm_slice_index) {
      if (conductor) {
        const GeneratedInterfacePaths geometric_paths = interface_slice_paths(paths, thinfilm_slice_index);
        const bool geometric_exists = std::filesystem::exists(geometric_paths.geometric);
        const bool geometric_average_exists = std::filesystem::exists(geometric_paths.geometric_average);
        if ((geometric_exists && geometric_average_exists) == false) {
          const ThinfilmEval thinfilm = sample_thinfilm_slice(data, material.thinfilm, spect, thinfilm_slice_index, slice_count);
          const bool generated_geometric = generate_conductor_geometric_interface(geometric_paths, ext_ior, int_ior, thinfilm, scheduler);
          spectral_result = generated_geometric && spectral_result;
        }
      }

      for (uint32_t wavelength_group = 0u; wavelength_group < kEnergyCompensationSpectralWavelengthGroupCount; ++wavelength_group) {
        const uint32_t packed_slice_index = thinfilm_slice_index * kEnergyCompensationSpectralWavelengthGroupCount + wavelength_group;
        const GeneratedInterfacePaths slice_paths_value = interface_slice_paths(paths, packed_slice_index);
        const bool directional_exists = std::filesystem::exists(slice_paths_value.directional);
        const bool average_exists = std::filesystem::exists(slice_paths_value.average);
        const bool conductor_fms_exists = conductor ? std::filesystem::exists(slice_paths_value.conductor_fms) : true;
        const bool probability_exists = conductor ? true : std::filesystem::exists(slice_paths_value.probability);
        if (((directional_exists && average_exists) && conductor_fms_exists) && probability_exists) {
          continue;
        }

        const auto time_begin = std::chrono::steady_clock::now();
        const bool generated = conductor ? generate_conductor_interface_spectral(data, material, slice_paths_value, thinfilm_slice_index, slice_count, wavelength_group, scheduler)
                                         : generate_dielectric_interface_spectral(data, material, slice_paths_value, thinfilm_slice_index, slice_count, wavelength_group, scheduler);
        const auto time_end = std::chrono::steady_clock::now();
        const double elapsed_seconds = std::chrono::duration<double>(time_end - time_begin).count();
        if (generated) {
          log::info("Generated spectral energy-compensation LUT cache %s in %.3f seconds", slice_paths_value.directional.generic_string().c_str(), elapsed_seconds);
        }
        spectral_result = generated && spectral_result;
      }
    }
    return spectral_result;
  }

  bool result = true;
  for (uint32_t slice_index = 0u; slice_index < slice_count; ++slice_index) {
    const GeneratedInterfacePaths slice_paths_value = interface_slice_paths(paths, slice_index);
    const bool directional_exists = std::filesystem::exists(slice_paths_value.directional);
    const bool average_exists = std::filesystem::exists(slice_paths_value.average);
    const bool geometric_exists = conductor ? std::filesystem::exists(slice_paths_value.geometric) : true;
    const bool geometric_average_exists = conductor ? std::filesystem::exists(slice_paths_value.geometric_average) : true;
    const bool conductor_fms_exists = conductor ? std::filesystem::exists(slice_paths_value.conductor_fms) : true;
    if ((((directional_exists && average_exists) && geometric_exists) && geometric_average_exists) && conductor_fms_exists) {
      continue;
    }

    const ThinfilmEval thinfilm = sample_thinfilm_slice(data, material.thinfilm, spect, slice_index, slice_count);
    const auto time_begin = std::chrono::steady_clock::now();
    const bool generated =
      conductor ? generate_conductor_interface(slice_paths_value, ext_ior, int_ior, thinfilm, scheduler) : generate_dielectric_interface(slice_paths_value, ext_ior, int_ior, thinfilm, scheduler);
    const auto time_end = std::chrono::steady_clock::now();
    const double elapsed_seconds = std::chrono::duration<double>(time_end - time_begin).count();
    if (generated) {
      log::info("Generated energy-compensation LUT cache %s in %.3f seconds", slice_paths_value.directional.generic_string().c_str(), elapsed_seconds);
    }
    result = generated && result;
  }
  return result;
}

bool bind_energy_compensation_interface(SceneData& data, const Material& material, uint32_t material_class, uint32_t cache_mode,
  std::unordered_map<uint64_t, uint32_t>& interface_cache, TaskScheduler& scheduler, uint32_t& out_interface_index) {
  const bool conductor = material_class == MaterialClass::Conductor;
  const uint64_t hash = hash_material_interface(data, material, material_class, cache_mode);
  const auto found = interface_cache.find(hash);
  if (found != interface_cache.end()) {
    out_interface_index = found->second;
    return true;
  }

  const GeneratedInterfacePaths paths = interface_paths(material_class, hash);
  if (ensure_cache_file(data, material, material_class, cache_mode, paths, scheduler) == false) {
    return false;
  }

  Scene::EnergyCompensationInterface interface_data = {};
  interface_data.cls = material_class;
  interface_data.cache_mode = cache_mode;
  const uint32_t slice_count = thinfilm_lut_slice_count(material.thinfilm);
  interface_data.thinfilm_slice_count = slice_count;
  if (cache_mode == kBSDFEnergyCompensationCacheModeSpectralScalar) {
    interface_data.spectral_wavelength_count = kEnergyCompensationSpectralWavelengthCount;
    interface_data.spectral_shortest_wavelength = kShortestWavelength;
    interface_data.spectral_longest_wavelength = kLongestWavelength;
  }
  if (cache_mode == kBSDFEnergyCompensationCacheModeSpectralScalar) {
    std::vector<float4> directional_pixels;
    std::vector<float4> average_pixels;
    std::vector<float4> geometric_pixels;
    std::vector<float4> geometric_average_pixels;
    std::vector<float4> conductor_fms_pixels;
    std::vector<float4> probability_pixels;
    const uint32_t directional_width = conductor ? kEnergyCompensationConductorLutSize : (kEnergyCompensationDielectricBranchCount * kEnergyCompensationDielectricLutSize);
    const uint32_t directional_height = conductor ? kEnergyCompensationConductorLutSize : kEnergyCompensationDielectricLutSize;
    const uint32_t average_width = conductor ? kEnergyCompensationConductorLutSize : kEnergyCompensationDielectricAverageWidth;
    const uint32_t average_height = conductor ? 1u : kEnergyCompensationDielectricLutSize;
    const uint32_t packed_slice_count = slice_count * kEnergyCompensationSpectralWavelengthGroupCount;
    if ((load_rgba32f_lut_slices(paths, directional_width, directional_height, packed_slice_count, directional_pixels, &GeneratedInterfacePaths::directional) == false) ||
        (load_rgba32f_lut_slices(paths, average_width, average_height, packed_slice_count, average_pixels, &GeneratedInterfacePaths::average) == false)) {
      return false;
    }

    interface_data.directional_lut = data.images.add_from_data_3d(directional_pixels.data(), uint3{directional_width, directional_height, packed_slice_count},
      Image::SkipSRGBConversion, {}, {1.0f, 1.0f, 1.0f});
    interface_data.average_lut =
      data.images.add_from_data_3d(average_pixels.data(), uint3{average_width, average_height, packed_slice_count}, Image::SkipSRGBConversion, {}, {1.0f, 1.0f, 1.0f});

    if (conductor) {
      if ((load_rgba32f_lut_slices(paths, kEnergyCompensationConductorLutSize, kEnergyCompensationConductorLutSize, slice_count, geometric_pixels,
             &GeneratedInterfacePaths::geometric) == false) ||
          (load_rgba32f_lut_slices(paths, kEnergyCompensationConductorLutSize, 1u, slice_count, geometric_average_pixels, &GeneratedInterfacePaths::geometric_average) == false) ||
          (load_rgba32f_lut_slices(paths, kEnergyCompensationConductorLutSize, 1u, packed_slice_count, conductor_fms_pixels, &GeneratedInterfacePaths::conductor_fms) == false)) {
        return false;
      }
      interface_data.geometric_lut = data.images.add_from_data_3d(geometric_pixels.data(), uint3{kEnergyCompensationConductorLutSize, kEnergyCompensationConductorLutSize, slice_count},
        Image::SkipSRGBConversion, {}, {1.0f, 1.0f, 1.0f});
      interface_data.geometric_average_lut = data.images.add_from_data_3d(geometric_average_pixels.data(), uint3{kEnergyCompensationConductorLutSize, 1u, slice_count},
        Image::SkipSRGBConversion, {}, {1.0f, 1.0f, 1.0f});
      interface_data.conductor_fms_lut = data.images.add_from_data_3d(conductor_fms_pixels.data(), uint3{kEnergyCompensationConductorLutSize, 1u, packed_slice_count},
        Image::SkipSRGBConversion, {}, {1.0f, 1.0f, 1.0f});
    } else {
      if (load_rgba32f_lut_slices(paths, directional_width, directional_height, packed_slice_count, probability_pixels, &GeneratedInterfacePaths::probability) == false) {
        return false;
      }
      interface_data.probability_lut = data.images.add_from_data_3d(probability_pixels.data(), uint3{directional_width, directional_height, packed_slice_count},
        Image::SkipSRGBConversion, {}, {1.0f, 1.0f, 1.0f});
    }
  } else if (slice_count == 1u) {
    interface_data.directional_lut = data.add_image(paths.directional.generic_string().c_str(), Image::SkipSRGBConversion);
    interface_data.average_lut = data.add_image(paths.average.generic_string().c_str(), Image::SkipSRGBConversion);
    interface_data.geometric_lut = conductor ? data.add_image(paths.geometric.generic_string().c_str(), Image::SkipSRGBConversion) : kInvalidIndex;
    interface_data.geometric_average_lut = conductor ? data.add_image(paths.geometric_average.generic_string().c_str(), Image::SkipSRGBConversion) : kInvalidIndex;
    interface_data.conductor_fms_lut = conductor ? data.add_image(paths.conductor_fms.generic_string().c_str(), Image::SkipSRGBConversion) : kInvalidIndex;
  } else {
    std::vector<float4> directional_pixels;
    std::vector<float4> average_pixels;
    std::vector<float4> geometric_pixels;
    std::vector<float4> geometric_average_pixels;
    std::vector<float4> conductor_fms_pixels;
    const uint32_t directional_width = conductor ? kEnergyCompensationConductorLutSize : (kEnergyCompensationDielectricBranchCount * kEnergyCompensationDielectricLutSize);
    const uint32_t directional_height = conductor ? kEnergyCompensationConductorLutSize : kEnergyCompensationDielectricLutSize;
    const uint32_t average_width = conductor ? kEnergyCompensationConductorLutSize : kEnergyCompensationDielectricAverageWidth;
    const uint32_t average_height = conductor ? 1u : kEnergyCompensationDielectricLutSize;
    if ((load_rgba32f_lut_slices(paths, directional_width, directional_height, slice_count, directional_pixels, &GeneratedInterfacePaths::directional) == false) ||
        (load_rgba32f_lut_slices(paths, average_width, average_height, slice_count, average_pixels, &GeneratedInterfacePaths::average) == false)) {
      return false;
    }

    interface_data.directional_lut =
      data.images.add_from_data_3d(directional_pixels.data(), uint3{directional_width, directional_height, slice_count}, Image::SkipSRGBConversion, {}, {1.0f, 1.0f, 1.0f});
    interface_data.average_lut =
      data.images.add_from_data_3d(average_pixels.data(), uint3{average_width, average_height, slice_count}, Image::SkipSRGBConversion, {}, {1.0f, 1.0f, 1.0f});

    if (conductor) {
      if ((load_rgba32f_lut_slices(paths, kEnergyCompensationConductorLutSize, kEnergyCompensationConductorLutSize, slice_count, geometric_pixels,
             &GeneratedInterfacePaths::geometric) == false) ||
          (load_rgba32f_lut_slices(paths, kEnergyCompensationConductorLutSize, 1u, slice_count, geometric_average_pixels, &GeneratedInterfacePaths::geometric_average) == false) ||
          (load_rgba32f_lut_slices(paths, kEnergyCompensationConductorLutSize, 1u, slice_count, conductor_fms_pixels, &GeneratedInterfacePaths::conductor_fms) == false)) {
        return false;
      }
      interface_data.geometric_lut = data.images.add_from_data_3d(geometric_pixels.data(), uint3{kEnergyCompensationConductorLutSize, kEnergyCompensationConductorLutSize, slice_count},
        Image::SkipSRGBConversion, {}, {1.0f, 1.0f, 1.0f});
      interface_data.geometric_average_lut = data.images.add_from_data_3d(geometric_average_pixels.data(), uint3{kEnergyCompensationConductorLutSize, 1u, slice_count},
        Image::SkipSRGBConversion, {}, {1.0f, 1.0f, 1.0f});
      interface_data.conductor_fms_lut = data.images.add_from_data_3d(conductor_fms_pixels.data(), uint3{kEnergyCompensationConductorLutSize, 1u, slice_count},
        Image::SkipSRGBConversion, {}, {1.0f, 1.0f, 1.0f});
    } else {
      interface_data.geometric_lut = kInvalidIndex;
      interface_data.geometric_average_lut = kInvalidIndex;
      interface_data.conductor_fms_lut = kInvalidIndex;
    }
  }
  const uint32_t interface_index = data.add_energy_compensation_interface(interface_data);
  interface_cache[hash] = interface_index;
  out_interface_index = interface_index;
  log::info("Bound material energy-compensation interface %u to %s", interface_index, paths.directional.generic_string().c_str());
  return true;
}

}  // namespace

bool ensure_energy_compensation_interfaces(SceneData& data, TaskScheduler& scheduler) {
  std::unordered_map<uint64_t, uint32_t> interface_cache;
  bool result = true;
  const uint32_t cache_mode = data.options.properties[Scene::Properties::Spectral] ? kBSDFEnergyCompensationCacheModeSpectralScalar : kBSDFEnergyCompensationCacheModeIntegratedRGB;

  data.energy_compensation_interfaces.clear();
  for (Material& material : data.materials) {
    material.energy_compensation_interface_index = kInvalidIndex;
    material.conductor_energy_compensation_interface_index = kInvalidIndex;
  }

  for (Material& material : data.materials) {
    if (material.cls == MaterialClass::OpenPBR) {
      Material dielectric_material = material;
      dielectric_material.cls = MaterialClass::Dielectric;
      const bool dielectric_bound = bind_energy_compensation_interface(data, dielectric_material, MaterialClass::Dielectric, cache_mode, interface_cache, scheduler,
        material.energy_compensation_interface_index);

      Material conductor_material = material;
      conductor_material.cls = MaterialClass::Conductor;
      conductor_material.int_ior.cls = SpectralDistribution::Conductor;
      if (data.defaults.conductor_eta == kInvalidIndex) {
        data.defaults.conductor_eta = data.add_spectrum(SpectralDistribution::constant(0.0f));
      }
      if (data.defaults.conductor_k == kInvalidIndex) {
        data.defaults.conductor_k = data.add_spectrum(SpectralDistribution::constant(1000000.0f));
      }
      conductor_material.int_ior.eta_index = data.defaults.conductor_eta;
      conductor_material.int_ior.k_index = data.defaults.conductor_k;
      const bool conductor_bound = bind_energy_compensation_interface(data, conductor_material, MaterialClass::Conductor, cache_mode, interface_cache, scheduler,
        material.conductor_energy_compensation_interface_index);
      result = (dielectric_bound && conductor_bound) && result;
      continue;
    }

    if (((material.cls != MaterialClass::Conductor) && (material.cls != MaterialClass::Dielectric)) && (material.cls != MaterialClass::Plastic)) {
      continue;
    }

    const uint32_t material_class = (material.cls == MaterialClass::Plastic) ? MaterialClass::Dielectric : material.cls;
    result = bind_energy_compensation_interface(data, material, material_class, cache_mode, interface_cache, scheduler, material.energy_compensation_interface_index) && result;
  }

  return result;
}

}  // namespace etx
