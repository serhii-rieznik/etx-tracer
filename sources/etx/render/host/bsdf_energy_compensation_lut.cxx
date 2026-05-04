#include <etx/render/host/bsdf_energy_compensation_lut.hxx>

#include <etx/core/core.hxx>
#include <etx/core/environment.hxx>
#include <etx/core/log.hxx>
#include <etx/render/interop/bsdf_energy_compensated_shared.hxx>
#include <etx/render/interop/bsdf_external_shared.hxx>

#include <tinyexr.hxx>

#include <chrono>
#include <filesystem>
#include <unordered_map>

namespace etx {

namespace {

constexpr uint32_t kEnergyCompensationGeneratorVersion = 19u;
constexpr uint32_t kEnergyCompensationConductorLutSize = 64u;
constexpr uint32_t kEnergyCompensationDielectricLutSize = 64u;
constexpr uint32_t kEnergyCompensationConductorSampleCount = 2048u;
constexpr uint32_t kEnergyCompensationSampleCount = 512u;
constexpr uint32_t kEnergyCompensationDielectricMultiScatterSampleCount = 2048u;
constexpr uint32_t kEnergyCompensationDielectricBranchCount = 4u;
constexpr uint32_t kEnergyCompensationDielectricAverageWidth = 8u;

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

uint64_t hash_material_interface(const SceneData& data, const Material& material, uint32_t material_class) {
  uint64_t result = 0u;
  result = etx_hash64_continue(&kEnergyCompensationGeneratorVersion, sizeof(kEnergyCompensationGeneratorVersion), result);
  result = etx_hash64_continue(&material_class, sizeof(material_class), result);
  result = etx_hash64_continue(&kEnergyCompensationConductorLutSize, sizeof(kEnergyCompensationConductorLutSize), result);
  result = etx_hash64_continue(&kEnergyCompensationDielectricLutSize, sizeof(kEnergyCompensationDielectricLutSize), result);
  result = etx_hash64_continue(&kEnergyCompensationConductorSampleCount, sizeof(kEnergyCompensationConductorSampleCount), result);
  result = etx_hash64_continue(&kEnergyCompensationSampleCount, sizeof(kEnergyCompensationSampleCount), result);
  result = etx_hash64_continue(&kEnergyCompensationDielectricMultiScatterSampleCount, sizeof(kEnergyCompensationDielectricMultiScatterSampleCount), result);
  result = etx_hash64_continue(&kEnergyCompensationDielectricBranchCount, sizeof(kEnergyCompensationDielectricBranchCount), result);
  result = etx_hash64_continue(&kEnergyCompensationDielectricAverageWidth, sizeof(kEnergyCompensationDielectricAverageWidth), result);
  result = hash_refractive_index(data, material.ext_ior, result);
  result = hash_refractive_index(data, material.int_ior, result);
  return result;
}

std::filesystem::path cache_directory() {
  return std::filesystem::path(env().file_in_data("bsdf/cache/energy_compensation"));
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

SpectralDirectionalAlbedoResult integrate_conductor_directional(const RefractiveIndexSample& ext_ior, const RefractiveIndexSample& int_ior, float mu_i, float alpha) {
  SpectralDirectionalAlbedoResult result = {};
  if (mu_i <= kEpsilon) {
    return result;
  }

  const SpectralQuery spect = {};
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
      const BSDFEnergyCompensatedLobe lobe = bsdf_energy_compensated_conductor_base_lobe(spect, w_i, w_o, alpha, ext_ior, int_ior, texture);
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

DielectricDirectionalAlbedoResult integrate_dielectric_directional(const RefractiveIndexSample& ext_ior, const RefractiveIndexSample& int_ior, bool incident_outside, float mu_i,
  float alpha) {
  DielectricDirectionalAlbedoResult result = {};
  if (mu_i <= kEpsilon) {
    return result;
  }

  const SpectralQuery spect = {};
  const ThinfilmEval thinfilm = empty_thinfilm();
  const float3 w_i = incident_direction_from_mu(mu_i);
  const float eta = spectral_response_monochromatic(spectral_response_div(int_ior.eta, ext_ior.eta));
  const auto texture = spectral_response_make(spect, 1.0f);
  const uint32_t incident_side = dielectric_side(incident_outside);
  const uint32_t opposite_side = 1u - incident_side;

  if (abs(eta - 1.0f) <= (16.0f * kEpsilon)) {
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
      const auto lobe = bsdf_energy_compensated_dielectric_base_lobe(spect, w_i, w_o_r, alpha, ext_ior, int_ior, texture);
      if ((fresnel_probability > kEpsilon) && (lobe.pdf > kEpsilon)) {
        result.branch_albedo[incident_side] += lobe.bsdf.integrated * (fresnel_probability / lobe.pdf);
        result.branch_visible_probability[incident_side] += fresnel_probability;
      }
    }

    if ((fresnel_probability < 1.0f) && (cos_theta_t2 > 0.0f)) {
      const float3 w_o_t = normalize(bsdf_external_refract(w_i, m, eta));
      if (w_o_t.z < 0.0f) {
        const auto lobe = bsdf_energy_compensated_dielectric_base_lobe(spect, w_i, w_o_t, alpha, ext_ior, int_ior, texture);
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

DielectricDirectionalAlbedoResult integrate_dielectric_total_directional(const RefractiveIndexSample& ext_ior, const RefractiveIndexSample& int_ior, bool incident_outside,
  float mu_i, float alpha) {
  DielectricDirectionalAlbedoResult result = {};
  if (mu_i <= kEpsilon) {
    return result;
  }

  const SpectralQuery spect = {};
  const ThinfilmEval thinfilm = empty_thinfilm();
  const float2 alpha2 = float2(alpha, alpha);
  const float3 w_i = incident_direction_from_mu(mu_i);
  const uint32_t incident_side = dielectric_side(incident_outside);
  const uint32_t opposite_side = 1u - incident_side;
  const float eta = spectral_response_monochromatic(spectral_response_div(int_ior.eta, ext_ior.eta));

  if (abs(eta - 1.0f) <= (16.0f * kEpsilon)) {
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

bool generate_conductor_interface(const GeneratedInterfacePaths& paths, const RefractiveIndexSample& ext_ior, const RefractiveIndexSample& int_ior, TaskScheduler& scheduler) {
  const uint32_t entry_count = kEnergyCompensationConductorLutSize * kEnergyCompensationConductorLutSize;
  std::vector<SpectralDirectionalAlbedoResult> directional(entry_count);
  scheduler.execute(entry_count, [&](uint32_t begin, uint32_t end, uint32_t thread_id) {
    (void)thread_id;
    for (uint32_t index = begin; index < end; ++index) {
      const uint32_t alpha_index = index / kEnergyCompensationConductorLutSize;
      const uint32_t mu_index = index - alpha_index * kEnergyCompensationConductorLutSize;
      directional[index] = integrate_conductor_directional(ext_ior, int_ior, mu_parameter(mu_index, kEnergyCompensationConductorLutSize),
        alpha_parameter(alpha_index, kEnergyCompensationConductorLutSize));
    }
  });

  std::vector<float4> image(entry_count, float4{0.0f, 0.0f, 0.0f, 0.0f});
  std::vector<float4> average(kEnergyCompensationConductorLutSize, float4{0.0f, 0.0f, 0.0f, 1.0f});
  std::vector<float4> geometric_image(entry_count, float4{0.0f, 0.0f, 0.0f, 1.0f});
  std::vector<float4> geometric_average(kEnergyCompensationConductorLutSize, float4{0.0f, 0.0f, 0.0f, 1.0f});
  std::vector<float4> conductor_fms(kEnergyCompensationConductorLutSize, float4{0.0f, 0.0f, 0.0f, 1.0f});
  const SpectralQuery spect = {};
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
    const ::SpectralResponse fms = bsdf_energy_compensated_conductor_fms(spect, ext_ior, int_ior, geometric_average_value);
    conductor_fms[alpha_index] = float4(fms.integrated.x, fms.integrated.y, fms.integrated.z, 1.0f);
  }

  return save_exr_rgba(paths.directional, image, kEnergyCompensationConductorLutSize, kEnergyCompensationConductorLutSize) &&
         save_exr_rgba(paths.average, average, kEnergyCompensationConductorLutSize, 1u) &&
         save_exr_rgba(paths.geometric, geometric_image, kEnergyCompensationConductorLutSize, kEnergyCompensationConductorLutSize) &&
         save_exr_rgba(paths.geometric_average, geometric_average, kEnergyCompensationConductorLutSize, 1u) &&
         save_exr_rgba(paths.conductor_fms, conductor_fms, kEnergyCompensationConductorLutSize, 1u);
}

bool generate_dielectric_interface(const GeneratedInterfacePaths& paths, const RefractiveIndexSample& ext_ior, const RefractiveIndexSample& int_ior, TaskScheduler& scheduler) {
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
      directional[index] = integrate_dielectric_directional(source_ior, target_ior, side == 0u, mu_parameter(mu_index, kEnergyCompensationDielectricLutSize),
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
      total_directional[index] = integrate_dielectric_total_directional(source_ior, target_ior, side == 0u,
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

bool ensure_cache_file(const SceneData& data, const Material& material, uint32_t material_class, const GeneratedInterfacePaths& paths, TaskScheduler& scheduler) {
  const bool conductor = material_class == MaterialClass::Conductor;
  const bool directional_exists = std::filesystem::exists(paths.directional);
  const bool average_exists = std::filesystem::exists(paths.average);
  const bool geometric_exists = conductor ? std::filesystem::exists(paths.geometric) : true;
  const bool geometric_average_exists = conductor ? std::filesystem::exists(paths.geometric_average) : true;
  const bool conductor_fms_exists = conductor ? std::filesystem::exists(paths.conductor_fms) : true;
  if ((((directional_exists && average_exists) && geometric_exists) && geometric_average_exists) && conductor_fms_exists) {
    return true;
  }

  const SpectralQuery spect = {};
  const RefractiveIndexSample ext_ior = sample_refractive_index(data, material.ext_ior, spect);
  const RefractiveIndexSample int_ior = sample_refractive_index(data, material.int_ior, spect);
  const auto time_begin = std::chrono::steady_clock::now();
  const bool generated = conductor ? generate_conductor_interface(paths, ext_ior, int_ior, scheduler) : generate_dielectric_interface(paths, ext_ior, int_ior, scheduler);
  const auto time_end = std::chrono::steady_clock::now();
  const double elapsed_seconds = std::chrono::duration<double>(time_end - time_begin).count();
  if (generated) {
    log::info("Generated energy-compensation LUT cache %s in %.3f seconds", paths.directional.generic_string().c_str(), elapsed_seconds);
  }
  return generated;
}

bool bind_energy_compensation_interface(SceneData& data, const Material& material, uint32_t material_class, std::unordered_map<uint64_t, uint32_t>& interface_cache,
  TaskScheduler& scheduler, uint32_t& out_interface_index) {
  const bool conductor = material_class == MaterialClass::Conductor;
  const uint64_t hash = hash_material_interface(data, material, material_class);
  const auto found = interface_cache.find(hash);
  if (found != interface_cache.end()) {
    out_interface_index = found->second;
    return true;
  }

  const GeneratedInterfacePaths paths = interface_paths(material_class, hash);
  if (ensure_cache_file(data, material, material_class, paths, scheduler) == false) {
    return false;
  }

  Scene::EnergyCompensationInterface interface_data = {};
  interface_data.cls = material_class;
  interface_data.directional_lut = data.add_image(paths.directional.generic_string().c_str(), Image::SkipSRGBConversion);
  interface_data.average_lut = data.add_image(paths.average.generic_string().c_str(), Image::SkipSRGBConversion);
  interface_data.geometric_lut = conductor ? data.add_image(paths.geometric.generic_string().c_str(), Image::SkipSRGBConversion) : kInvalidIndex;
  interface_data.geometric_average_lut = conductor ? data.add_image(paths.geometric_average.generic_string().c_str(), Image::SkipSRGBConversion) : kInvalidIndex;
  interface_data.conductor_fms_lut = conductor ? data.add_image(paths.conductor_fms.generic_string().c_str(), Image::SkipSRGBConversion) : kInvalidIndex;
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

  data.energy_compensation_interfaces.clear();
  for (Material& material : data.materials) {
    material.energy_compensation_interface_index = kInvalidIndex;
    material.conductor_energy_compensation_interface_index = kInvalidIndex;
  }

  for (Material& material : data.materials) {
    if (material.cls == MaterialClass::OpenPBR) {
      Material dielectric_material = material;
      dielectric_material.cls = MaterialClass::Dielectric;
      const bool dielectric_bound = bind_energy_compensation_interface(data, dielectric_material, MaterialClass::Dielectric, interface_cache, scheduler,
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
      const bool conductor_bound = bind_energy_compensation_interface(data, conductor_material, MaterialClass::Conductor, interface_cache, scheduler,
        material.conductor_energy_compensation_interface_index);
      result = (dielectric_bound && conductor_bound) && result;
      continue;
    }

    if (((material.cls != MaterialClass::Conductor) && (material.cls != MaterialClass::Dielectric)) && (material.cls != MaterialClass::Plastic)) {
      continue;
    }

    const uint32_t material_class = (material.cls == MaterialClass::Plastic) ? MaterialClass::Dielectric : material.cls;
    result = bind_energy_compensation_interface(data, material, material_class, interface_cache, scheduler, material.energy_compensation_interface_index) && result;
  }

  return result;
}

}  // namespace etx
