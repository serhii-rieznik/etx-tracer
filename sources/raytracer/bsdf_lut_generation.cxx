#include "bsdf_lut_generation.hxx"

#include <etx/core/environment.hxx>
#include <etx/core/log.hxx>
#include <etx/render/host/bsdf_energy_compensation_lut.hxx>
#include <etx/render/host/scene_data.hxx>
#include <etx/render/host/tasks.hxx>
#include <etx/render/interop/interop.hxx>
#include <etx/render/interop/bsdf_external_shared.hxx>
#include <etx/render/shared/spectrum.hxx>

#include <tinyexr.hxx>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <string>
#include <vector>

namespace etx {

namespace {

constexpr uint32_t kConductorLutSize = 32u;
constexpr uint32_t kDielectricLutSize = 64u;
constexpr uint32_t kDielectricLutWidth = kDielectricLutSize * kDielectricLutSize;
constexpr float kF0Max = 9.99000013e-1f;

struct DirectionalAlbedoResult {
  float albedo = 0.0f;
  float visible_probability = 0.0f;
};

struct NamedIor {
  std::string name = {};
  RefractiveIndex ior = {};
};

uint32_t conductor_index(uint32_t alpha_index, uint32_t mu_index) {
  return alpha_index * kConductorLutSize + mu_index;
}

uint32_t dielectric_index(uint32_t f0_index, uint32_t alpha_index, uint32_t mu_index) {
  return f0_index * kDielectricLutSize * kDielectricLutSize + alpha_index * kDielectricLutSize + mu_index;
}

uint32_t dielectric_average_index(uint32_t f0_index, uint32_t alpha_index) {
  return f0_index * kDielectricLutSize + alpha_index;
}

float lut_parameter(uint32_t index, uint32_t size) {
  return static_cast<float>(index) / static_cast<float>(size - 1u);
}

float alpha_parameter(uint32_t index, uint32_t size) {
  return max(kBSDFNormalDistributionMinAlpha, lut_parameter(index, size));
}

float dielectric_alpha_parameter(uint32_t index) {
  const float axis = lut_parameter(index, kDielectricLutSize);
  return kBSDFNormalDistributionMinAlpha + (1.0f - kBSDFNormalDistributionMinAlpha) * axis * axis;
}

float dielectric_f0_parameter(uint32_t index) {
  const float axis = lut_parameter(index, kDielectricLutSize);
  return kF0Max * axis * axis;
}

float lut_saturate(float value) {
  return min(1.0f, max(0.0f, value));
}

float lut_lerp(float a, float b, float t) {
  return a * (1.0f - t) + b * t;
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

float3 incident_direction_from_mu(float mu) {
  return float3{sqrt(max(0.0f, 1.0f - mu * mu)), 0.0f, mu};
}

float3 sample_vndf_local(ETX_IN(float3, w_i), float alpha, ETX_IN(float2, rnd)) {
  const float3 w_i_11 = normalize(float3(alpha * w_i.x, alpha * w_i.y, w_i.z));
  const float2 slope_11 = bsdf_external_sample_p22_11(acos(lut_saturate(w_i_11.z)), rnd, float2(alpha, alpha));

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

RefractiveIndexSample make_dielectric_ior(float eta) {
  RefractiveIndexSample result = {};
  result.eta = spectral_response_make(float3{eta, eta, eta});
  result.k = spectral_response_make(float3{0.0f, 0.0f, 0.0f});
  result.cls = SpectralDistribution::Dielectric;
  return result;
}

float dielectric_eta_from_f0(float f0) {
  const float sqrt_f0 = sqrt(lut_saturate(f0));
  return (1.0f + sqrt_f0) / max(kEpsilon, 1.0f - sqrt_f0);
}

float dielectric_fresnel_value(float cos_theta, ETX_IN(RefractiveIndexSample, ext_ior), ETX_IN(RefractiveIndexSample, int_ior)) {
  const SpectralQuery spect = {};
  const ThinfilmEval thinfilm = empty_thinfilm();
  const ::SpectralResponse fresnel = bsdf_fresnel_calculate(spect, cos_theta, ext_ior, int_ior, thinfilm);
  return lut_saturate(spectral_response_monochromatic(fresnel));
}

DirectionalAlbedoResult integrate_conductor_directional(float mu_i, float alpha, uint32_t sample_count) {
  DirectionalAlbedoResult result = {};
  if (mu_i <= kEpsilon) {
    return result;
  }

  const float2 alpha2 = float2{alpha, alpha};
  const float3 w_i = incident_direction_from_mu(mu_i);
  const float lambda_i = bsdf_external_ray_info_make(w_i, alpha2).Lambda;
  const float g1_i = 1.0f / (1.0f + lambda_i);

  for (uint32_t sample_index = 0u; sample_index < sample_count; ++sample_index) {
    const float3 m = sample_vndf_local(w_i, alpha, hammersley(sample_index, sample_count));
    const float i_dot_m = dot(w_i, m);
    const float3 w_o = -w_i + 2.0f * m * i_dot_m;
    if (w_o.z > 0.0f) {
      const float lambda_o = bsdf_external_ray_info_make(w_o, alpha2).Lambda;
      const float g2 = 1.0f / (1.0f + lambda_i + lambda_o);
      result.albedo += g2 / g1_i;
      result.visible_probability += 1.0f;
    }
  }

  const float inv_sample_count = 1.0f / static_cast<float>(sample_count);
  result.albedo = lut_saturate(result.albedo * inv_sample_count);
  result.visible_probability = lut_saturate(result.visible_probability * inv_sample_count);
  return result;
}

DirectionalAlbedoResult integrate_dielectric_directional(float mu_i, float alpha, float eta, uint32_t sample_count) {
  DirectionalAlbedoResult result = {};
  if (abs(eta - 1.0f) <= (16.0f * kEpsilon)) {
    result.albedo = 1.0f;
    result.visible_probability = 1.0f;
    return result;
  }

  if (mu_i <= kEpsilon) {
    return result;
  }

  const float2 alpha2 = float2{alpha, alpha};
  const float3 w_i = incident_direction_from_mu(mu_i);
  const float lambda_i = bsdf_external_ray_info_make(w_i, alpha2).Lambda;
  const float g1_i = 1.0f / (1.0f + lambda_i);
  const RefractiveIndexSample ext_ior = make_dielectric_ior(1.0f);
  const RefractiveIndexSample int_ior = make_dielectric_ior(eta);

  for (uint32_t sample_index = 0u; sample_index < sample_count; ++sample_index) {
    const float3 m = sample_vndf_local(w_i, alpha, hammersley(sample_index, sample_count));
    const float i_dot_m = dot(w_i, m);
    if (i_dot_m <= kEpsilon) {
      continue;
    }

    const float cos_theta_t2 = 1.0f - (1.0f - i_dot_m * i_dot_m) / (eta * eta);
    const float fresnel = (cos_theta_t2 <= 0.0f) ? 1.0f : dielectric_fresnel_value(i_dot_m, ext_ior, int_ior);
    const float3 w_o_r = -w_i + 2.0f * m * i_dot_m;
    if (w_o_r.z > 0.0f) {
      const float lambda_o = bsdf_external_ray_info_make(w_o_r, alpha2).Lambda;
      const float g2 = 1.0f / (1.0f + lambda_i + lambda_o);
      result.albedo += fresnel * g2 / g1_i;
      result.visible_probability += fresnel;
    }

    if ((fresnel < 1.0f) && (cos_theta_t2 > 0.0f)) {
      const float3 w_o_t = normalize(bsdf_external_refract(w_i, m, eta));
      if (w_o_t.z < 0.0f) {
        const float lambda_o = bsdf_external_ray_info_make(-w_o_t, alpha2).Lambda;
        const float g2 = bsdf_external_beta(1.0f + lambda_i, 1.0f + lambda_o);
        const float one_minus_fresnel = 1.0f - fresnel;
        result.albedo += one_minus_fresnel * g2 / g1_i;
        result.visible_probability += one_minus_fresnel;
      }
    }
  }

  const float inv_sample_count = 1.0f / static_cast<float>(sample_count);
  result.albedo = lut_saturate(result.albedo * inv_sample_count);
  result.visible_probability = lut_saturate(result.visible_probability * inv_sample_count);
  return result;
}

float integrate_conductor_average(const std::vector<DirectionalAlbedoResult>& conductor, uint32_t alpha_index) {
  float total = 0.0f;
  for (uint32_t mu_index = 0u; mu_index < kConductorLutSize; ++mu_index) {
    const float mu = lut_parameter(mu_index, kConductorLutSize);
    const float weight = ((mu_index == 0u) || (mu_index == (kConductorLutSize - 1u))) ? 0.5f : 1.0f;
    total += weight * conductor[conductor_index(alpha_index, mu_index)].albedo * mu;
  }
  return lut_saturate(2.0f * total / static_cast<float>(kConductorLutSize - 1u));
}

float integrate_dielectric_average(const std::vector<DirectionalAlbedoResult>& dielectric, uint32_t f0_index, uint32_t alpha_index) {
  float total = 0.0f;
  for (uint32_t mu_index = 0u; mu_index < kDielectricLutSize; ++mu_index) {
    const float mu = lut_parameter(mu_index, kDielectricLutSize);
    const float weight = ((mu_index == 0u) || (mu_index == (kDielectricLutSize - 1u))) ? 0.5f : 1.0f;
    total += weight * dielectric[dielectric_index(f0_index, alpha_index, mu_index)].albedo * mu;
  }
  return lut_saturate(2.0f * total / static_cast<float>(kDielectricLutSize - 1u));
}

float interpolate_row(const float row[kDielectricLutSize], float mu) {
  const float coord = lut_saturate(mu) * static_cast<float>(kDielectricLutSize - 1u);
  const uint32_t index_0 = static_cast<uint32_t>(floor(coord));
  const uint32_t index_1 = min(index_0 + 1u, kDielectricLutSize - 1u);
  const float t = coord - static_cast<float>(index_0);
  return lut_lerp(row[index_0], row[index_1], t);
}

void build_proposal_row(const float row[kDielectricLutSize], float cdf[kDielectricLutSize], float& total) {
  float weights[kDielectricLutSize] = {};
  total = 0.0f;
  const float inv_size = 1.0f / static_cast<float>(kDielectricLutSize);
  for (uint32_t mu_index = 0u; mu_index < kDielectricLutSize; ++mu_index) {
    const float mu_0 = static_cast<float>(mu_index) * inv_size;
    const float mu_1 = static_cast<float>(mu_index + 1u) * inv_size;
    const float mu_mid = 0.5f * (mu_0 + mu_1);
    weights[mu_index] = max(0.0f, 1.0f - interpolate_row(row, mu_mid)) * (mu_1 * mu_1 - mu_0 * mu_0);
    total += weights[mu_index];
  }

  float accumulated = 0.0f;
  for (uint32_t mu_index = 0u; mu_index < kDielectricLutSize; ++mu_index) {
    accumulated += weights[mu_index];
    if (total > kEpsilon) {
      cdf[mu_index] = lut_saturate(accumulated / total);
    } else {
      const float mu_1 = static_cast<float>(mu_index + 1u) * inv_size;
      cdf[mu_index] = mu_1 * mu_1;
    }
  }
  cdf[kDielectricLutSize - 1u] = 1.0f;
}

bool save_exr_rgba(const std::filesystem::path& path, const std::vector<float4>& pixels, uint32_t width, uint32_t height) {
  const uint64_t expected_pixel_count = static_cast<uint64_t>(width) * static_cast<uint64_t>(height);
  if (pixels.size() != expected_pixel_count) {
    log::error("Invalid LUT image dimensions for %s", path.generic_string().c_str());
    return false;
  }

  const std::filesystem::path parent_path = path.parent_path();
  if (parent_path.empty() == false) {
    std::error_code ec = {};
    std::filesystem::create_directories(parent_path, ec);
    if (ec.value() != 0) {
      log::error("Failed to create LUT directory %s", parent_path.generic_string().c_str());
      return false;
    }
  }

  const std::string file_name = path.generic_string();
  const char* error = nullptr;
  if (SaveEXR(reinterpret_cast<const float*>(pixels.data()), static_cast<int>(width), static_cast<int>(height), 4, false, file_name.c_str(), &error) !=
      TINYEXR_SUCCESS) {
    log::error("Failed to save EXR LUT %s: %s", file_name.c_str(), (error != nullptr) ? error : "unknown error");
    if (error != nullptr) {
      FreeEXRErrorMessage(error);
    }
    return false;
  }

  log::info("Saved BSDF LUT %s", file_name.c_str());
  return true;
}

bool write_lut_images(const std::filesystem::path& output_directory, const std::vector<DirectionalAlbedoResult>& conductor, const std::vector<float>& conductor_average,
  const std::vector<DirectionalAlbedoResult>& dielectric_outside, const std::vector<DirectionalAlbedoResult>& dielectric_inside,
  const std::vector<float>& dielectric_average_outside, const std::vector<float>& dielectric_average_inside) {
  std::vector<float4> conductor_image(kConductorLutSize * kConductorLutSize, float4{0.0f, 0.0f, 0.0f, 1.0f});
  for (uint32_t alpha_index = 0u; alpha_index < kConductorLutSize; ++alpha_index) {
    for (uint32_t mu_index = 0u; mu_index < kConductorLutSize; ++mu_index) {
      const uint32_t index = conductor_index(alpha_index, mu_index);
      float4& pixel = conductor_image[index];
      pixel.x = conductor[index].albedo;
      pixel.y = conductor[index].visible_probability;
    }
  }

  std::vector<float4> conductor_average_image(kConductorLutSize, float4{0.0f, 0.0f, 0.0f, 1.0f});
  for (uint32_t alpha_index = 0u; alpha_index < kConductorLutSize; ++alpha_index) {
    conductor_average_image[alpha_index].x = conductor_average[alpha_index];
  }

  std::vector<float4> dielectric_image(kDielectricLutWidth * kDielectricLutSize, float4{0.0f, 0.0f, 0.0f, 0.0f});
  for (uint32_t f0_index = 0u; f0_index < kDielectricLutSize; ++f0_index) {
    for (uint32_t alpha_index = 0u; alpha_index < kDielectricLutSize; ++alpha_index) {
      for (uint32_t mu_index = 0u; mu_index < kDielectricLutSize; ++mu_index) {
        const uint32_t source_index = dielectric_index(f0_index, alpha_index, mu_index);
        const uint32_t x = f0_index * kDielectricLutSize + mu_index;
        float4& pixel = dielectric_image[alpha_index * kDielectricLutWidth + x];
        pixel.x = dielectric_outside[source_index].albedo;
        pixel.y = dielectric_inside[source_index].albedo;
        pixel.z = dielectric_outside[source_index].visible_probability;
        pixel.w = dielectric_inside[source_index].visible_probability;
      }
    }
  }

  std::vector<float4> dielectric_average_image(kDielectricLutSize * kDielectricLutSize, float4{0.0f, 0.0f, 0.0f, 1.0f});
  std::vector<float4> dielectric_proposal_image(kDielectricLutWidth * kDielectricLutSize, float4{0.0f, 0.0f, 0.0f, 0.0f});
  for (uint32_t f0_index = 0u; f0_index < kDielectricLutSize; ++f0_index) {
    for (uint32_t alpha_index = 0u; alpha_index < kDielectricLutSize; ++alpha_index) {
      const uint32_t average_index = dielectric_average_index(f0_index, alpha_index);
      float4& average_pixel = dielectric_average_image[alpha_index * kDielectricLutSize + f0_index];
      average_pixel.x = dielectric_average_outside[average_index];
      average_pixel.y = dielectric_average_inside[average_index];

      float outside_row[kDielectricLutSize] = {};
      float inside_row[kDielectricLutSize] = {};
      for (uint32_t mu_index = 0u; mu_index < kDielectricLutSize; ++mu_index) {
        const uint32_t source_index = dielectric_index(f0_index, alpha_index, mu_index);
        outside_row[mu_index] = dielectric_outside[source_index].albedo;
        inside_row[mu_index] = dielectric_inside[source_index].albedo;
      }

      float outside_cdf[kDielectricLutSize] = {};
      float inside_cdf[kDielectricLutSize] = {};
      float outside_total = 0.0f;
      float inside_total = 0.0f;
      build_proposal_row(outside_row, outside_cdf, outside_total);
      build_proposal_row(inside_row, inside_cdf, inside_total);

      for (uint32_t mu_index = 0u; mu_index < kDielectricLutSize; ++mu_index) {
        const uint32_t x = f0_index * kDielectricLutSize + mu_index;
        float4& proposal_pixel = dielectric_proposal_image[alpha_index * kDielectricLutWidth + x];
        proposal_pixel.x = outside_cdf[mu_index];
        proposal_pixel.y = inside_cdf[mu_index];
        proposal_pixel.z = outside_total;
        proposal_pixel.w = inside_total;
      }
    }
  }

  if (save_exr_rgba(output_directory / "conductor_energy_compensation.exr", conductor_image, kConductorLutSize, kConductorLutSize) == false) {
    return false;
  }
  if (save_exr_rgba(output_directory / "conductor_energy_compensation_average.exr", conductor_average_image, kConductorLutSize, 1u) == false) {
    return false;
  }
  if (save_exr_rgba(output_directory / "dielectric_energy_compensation.exr", dielectric_image, kDielectricLutWidth, kDielectricLutSize) == false) {
    return false;
  }
  if (save_exr_rgba(output_directory / "dielectric_energy_compensation_average.exr", dielectric_average_image, kDielectricLutSize, kDielectricLutSize) == false) {
    return false;
  }
  if (save_exr_rgba(output_directory / "dielectric_energy_compensation_proposal.exr", dielectric_proposal_image, kDielectricLutWidth, kDielectricLutSize) == false) {
    return false;
  }

  return true;
}

std::string stem_name(const std::filesystem::path& path) {
  return path.stem().generic_string();
}

bool load_named_iors_from_directory(SceneData& data, const std::filesystem::path& directory, SpectralDistribution::Class expected_class, std::vector<NamedIor>& output) {
  output.clear();

  std::error_code ec = {};
  if (std::filesystem::exists(directory, ec) == false) {
    log::error("Named IOR directory does not exist: %s", directory.generic_string().c_str());
    return false;
  }

  std::vector<std::filesystem::path> paths = {};
  for (const auto& entry : std::filesystem::directory_iterator(directory, ec)) {
    if (ec.value() != 0) {
      log::error("Failed to iterate named IOR directory: %s", directory.generic_string().c_str());
      return false;
    }

    const bool regular_file = entry.is_regular_file(ec);
    if ((ec.value() == 0) && regular_file && (entry.path().extension() == ".spd")) {
      paths.emplace_back(entry.path());
    }
  }

  std::sort(paths.begin(), paths.end());

  for (const std::filesystem::path& path : paths) {
    SpectralDistribution eta = {};
    SpectralDistribution k = {};
    std::string title = {};
    const SpectralDistribution::Class loaded_class = SpectralDistribution::load_refractive_index(path.generic_string().c_str(), eta, k, title);
    if (loaded_class == SpectralDistribution::Invalid) {
      log::warning("Skipping invalid named IOR %s", path.generic_string().c_str());
      continue;
    }

    if (loaded_class != expected_class) {
      log::warning("Skipping named IOR %s because its class does not match the requested cache type", path.generic_string().c_str());
      continue;
    }

    NamedIor named = {};
    named.name = stem_name(path);
    named.ior.cls = loaded_class;
    named.ior.eta_index = data.add_spectrum((named.name + "_eta").c_str(), eta);
    named.ior.k_index = data.add_spectrum((named.name + "_k").c_str(), k);
    output.emplace_back(named);
  }

  if (output.empty()) {
    log::error("No named IORs loaded from %s", directory.generic_string().c_str());
    return false;
  }

  return true;
}

}  // namespace

bool generate_bsdf_energy_compensation_luts(const BSDFLutGenerationOptions& options) {
  if (options.sample_count == 0u) {
    log::error("BSDF LUT sample count must be greater than zero");
    return false;
  }

  const std::filesystem::path output_directory =
    options.output_directory.empty() ? std::filesystem::path(env().file_in_data("bsdf/energy_compensation")) : std::filesystem::path(options.output_directory);

  TaskScheduler scheduler = {};
  const auto time_begin = std::chrono::steady_clock::now();
  log::info("Generating BSDF energy compensation LUTs: conductor=%u dielectric=%u samples=%u threads=%u output=%s", kConductorLutSize, kDielectricLutSize, options.sample_count,
    scheduler.max_thread_count(), output_directory.generic_string().c_str());

  std::vector<DirectionalAlbedoResult> conductor(kConductorLutSize * kConductorLutSize);
  std::vector<float> conductor_average(kConductorLutSize, 0.0f);
  scheduler.execute(conductor.size(), [&](uint32_t begin, uint32_t end, uint32_t thread_id) {
    (void)thread_id;
    for (uint32_t index = begin; index < end; ++index) {
      const uint32_t alpha_index = index / kConductorLutSize;
      const uint32_t mu_index = index - alpha_index * kConductorLutSize;
      conductor[index] = integrate_conductor_directional(lut_parameter(mu_index, kConductorLutSize), alpha_parameter(alpha_index, kConductorLutSize), options.sample_count);
    }
  });
  for (uint32_t alpha_index = 0u; alpha_index < kConductorLutSize; ++alpha_index) {
    conductor_average[alpha_index] = integrate_conductor_average(conductor, alpha_index);
  }
  log::info("Generated conductor energy compensation LUTs");

  const uint32_t dielectric_entry_count = kDielectricLutSize * kDielectricLutSize * kDielectricLutSize;
  std::vector<DirectionalAlbedoResult> dielectric_outside(dielectric_entry_count);
  std::vector<DirectionalAlbedoResult> dielectric_inside(dielectric_entry_count);
  std::vector<float> dielectric_average_outside(kDielectricLutSize * kDielectricLutSize, 0.0f);
  std::vector<float> dielectric_average_inside(kDielectricLutSize * kDielectricLutSize, 0.0f);
  scheduler.execute(dielectric_entry_count, [&](uint32_t begin, uint32_t end, uint32_t thread_id) {
    (void)thread_id;
    for (uint32_t index = begin; index < end; ++index) {
      const uint32_t f0_index = index / (kDielectricLutSize * kDielectricLutSize);
      const uint32_t alpha_mu_index = index - f0_index * kDielectricLutSize * kDielectricLutSize;
      const uint32_t alpha_index = alpha_mu_index / kDielectricLutSize;
      const uint32_t mu_index = alpha_mu_index - alpha_index * kDielectricLutSize;
      const float f0 = dielectric_f0_parameter(f0_index);
      const float eta = dielectric_eta_from_f0(f0);
      const float alpha = dielectric_alpha_parameter(alpha_index);
      const float mu = lut_parameter(mu_index, kDielectricLutSize);
      dielectric_outside[index] = integrate_dielectric_directional(mu, alpha, eta, options.sample_count);
      dielectric_inside[index] = integrate_dielectric_directional(mu, alpha, 1.0f / eta, options.sample_count);
    }
  });
  for (uint32_t f0_index = 0u; f0_index < kDielectricLutSize; ++f0_index) {
    for (uint32_t alpha_index = 0u; alpha_index < kDielectricLutSize; ++alpha_index) {
      const uint32_t average_index = dielectric_average_index(f0_index, alpha_index);
      dielectric_average_outside[average_index] = integrate_dielectric_average(dielectric_outside, f0_index, alpha_index);
      dielectric_average_inside[average_index] = integrate_dielectric_average(dielectric_inside, f0_index, alpha_index);
    }
  }
  log::info("Generated dielectric energy compensation LUTs");

  if (write_lut_images(output_directory, conductor, conductor_average, dielectric_outside, dielectric_inside, dielectric_average_outside, dielectric_average_inside) == false) {
    return false;
  }

  const auto time_end = std::chrono::steady_clock::now();
  const double elapsed_seconds = std::chrono::duration<double>(time_end - time_begin).count();
  log::info("Finished BSDF energy compensation LUT generation in %.3f seconds", elapsed_seconds);
  return true;
}

bool pregenerate_named_bsdf_energy_compensation_lut_cache() {
  TaskScheduler scheduler = {};
  SceneData data(scheduler);
  data.clear(scheduler);

  std::vector<NamedIor> dielectrics = {};
  std::vector<NamedIor> conductors = {};
  const std::filesystem::path dielectric_directory = env().file_in_data("spectrum/dielectric");
  const std::filesystem::path conductor_directory = env().file_in_data("spectrum/conductor");
  if (load_named_iors_from_directory(data, dielectric_directory, SpectralDistribution::Dielectric, dielectrics) == false) {
    return false;
  }
  if (load_named_iors_from_directory(data, conductor_directory, SpectralDistribution::Conductor, conductors) == false) {
    return false;
  }

  const uint32_t dielectric_material_count = static_cast<uint32_t>(dielectrics.size() * dielectrics.size());
  const uint32_t conductor_material_count = static_cast<uint32_t>(dielectrics.size() * conductors.size());
  data.materials.reserve(static_cast<size_t>(dielectric_material_count) + static_cast<size_t>(conductor_material_count));

  for (const NamedIor& ext_ior : dielectrics) {
    for (const NamedIor& int_ior : dielectrics) {
      Material material = {};
      material.cls = MaterialClass::DielectricEnergyCompensated;
      material.ext_ior = ext_ior.ior;
      material.int_ior = int_ior.ior;
      data.materials.emplace_back(material);
    }
  }

  for (const NamedIor& ext_ior : dielectrics) {
    for (const NamedIor& int_ior : conductors) {
      Material material = {};
      material.cls = MaterialClass::ConductorEnergyCompensated;
      material.ext_ior = ext_ior.ior;
      material.int_ior = int_ior.ior;
      data.materials.emplace_back(material);
    }
  }

  log::info("Pregenerating named BSDF energy-compensation cache: dielectric SPDs=%zu conductor SPDs=%zu dielectric interfaces=%u conductor interfaces=%u",
    dielectrics.size(), conductors.size(), dielectric_material_count, conductor_material_count);

  const auto time_begin = std::chrono::steady_clock::now();
  const bool result = ensure_energy_compensation_interfaces(data, scheduler);
  const auto time_end = std::chrono::steady_clock::now();
  const double elapsed_seconds = std::chrono::duration<double>(time_end - time_begin).count();
  if (result) {
    log::info("Finished named BSDF energy-compensation cache pregeneration in %.3f seconds; cache entries=%zu", elapsed_seconds,
      data.energy_compensation_interfaces.size());
  }
  return result;
}

}  // namespace etx
