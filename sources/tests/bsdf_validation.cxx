#include <etx/core/environment.hxx>
#include <etx/render/host/bsdf_energy_compensation_lut.hxx>
#include <etx/render/host/scene_global.hxx>
#include <etx/render/host/scene_data.hxx>
#include <etx/render/shared/scene.hxx>
#include <etx/render/shared/scene_bsdf.hxx>

#include <cmath>
#include <cstdio>
#include <algorithm>

namespace {

constexpr uint32_t kIntegrationSamples = 32768u;
constexpr uint32_t kBsdfSamples = 2048u;

enum SpectrumSlot : uint32_t {
  SpectrumWhite,
  SpectrumBlack,
  SpectrumColored,
  SpectrumAirEta,
  SpectrumDielectricEta,
  SpectrumSapphireEta,
  SpectrumNamedPlasticEta,
  SpectrumNamedPlasticK,
  SpectrumNamedWaterEta,
  SpectrumNamedWaterK,
  SpectrumConductorEta,
  SpectrumConductorK,
  SpectrumMirrorEta,
  SpectrumMirrorK,
  SpectrumCount,
};

etx::SpectralDistribution make_spectrum(const float3& value) {
  etx::SpectralDistribution result = {};
  result.integrated_value = value;
  result.spectral_entry_count = 0u;
  return result;
}

etx::SpectralDistribution make_loaded_ior_constant(const float value) {
  etx::SpectralDistribution result = etx::SpectralDistribution::constant(value);
  result.integrated_value = etx::rgb_to_xyz(result.integrated_value);
  return result;
}

float3 sample_uniform_sphere(etx::Sampler& sampler) {
  const float2 sample = sampler.next_2d();
  const float z = 1.0f - 2.0f * sample.x;
  const float radius = sqrtf(max(0.0f, 1.0f - z * z));
  const float phi = kDoublePi * sample.y;
  return float3{radius * cosf(phi), radius * sinf(phi), z};
}

bool finite_response(const etx::SpectralResponse& value) {
  if (value.spectral()) {
    return std::isfinite(value.value);
  }

  return (std::isfinite(value.integrated.x)) && (std::isfinite(value.integrated.y)) && (std::isfinite(value.integrated.z));
}

bool non_negative_response(const etx::SpectralResponse& value) {
  if (value.spectral()) {
    return value.value >= -kEpsilon;
  }

  return (value.integrated.x >= -kEpsilon) && (value.integrated.y >= -kEpsilon) && (value.integrated.z >= -kEpsilon);
}

bool validate_sample(const etx::BSDFSample& sample) {
  if (std::isfinite(sample.pdf) == false) {
    return false;
  }

  if (sample.pdf < 0.0f) {
    return false;
  }

  if (finite_response(sample.weight) == false) {
    return false;
  }

  return non_negative_response(sample.weight);
}

float integrate_pdf(const etx::BSDFData& data, const etx::Material& material, const uint32_t seed) {
  etx::Sampler sampler(seed, seed ^ 0x9e3779b9u);
  float sum = 0.0f;

  for (uint32_t i = 0u; i < kIntegrationSamples; ++i) {
    const float3 w_o = sample_uniform_sphere(sampler);
    etx::Sampler pdf_sampler(seed + i, seed ^ i);
    const float pdf = etx::bsdf::pdf(data, w_o, material, pdf_sampler);
    if (std::isfinite(pdf) == false) {
      return -1.0f;
    }
    if (pdf < 0.0f) {
      return -1.0f;
    }
    sum += pdf;
  }

  return (sum / static_cast<float>(kIntegrationSamples)) * 4.0f * kPi;
}

float integrate_reverse_pdf(const etx::BSDFData& data, const etx::Material& material, const uint32_t seed) {
  etx::Sampler sampler(seed, seed ^ 0x63d8a1cbu);
  float sum = 0.0f;

  for (uint32_t i = 0u; i < kIntegrationSamples; ++i) {
    const float3 w_o = sample_uniform_sphere(sampler);
    etx::Sampler pdf_sampler(seed + i, seed ^ i);
    const float pdf = etx::bsdf::reverse_pdf(data, w_o, material, pdf_sampler);
    if (std::isfinite(pdf) == false) {
      return -1.0f;
    }
    if (pdf < 0.0f) {
      return -1.0f;
    }
    sum += pdf;
  }

  return (sum / static_cast<float>(kIntegrationSamples)) * 4.0f * kPi;
}

bool validate_sampling(const etx::BSDFData& data, const etx::Material& material, const uint32_t seed) {
  for (uint32_t i = 0u; i < kBsdfSamples; ++i) {
    etx::Sampler sampler(seed + i, seed ^ (i * 17u + 3u));
    const etx::BSDFSample sample = etx::bsdf::sample(data, material, sampler);
    if (validate_sample(sample) == false) {
      return false;
    }
  }

  return true;
}

bool validate_sample_pdf_match(const etx::BSDFData& data, const etx::Material& material, const uint32_t seed) {
  for (uint32_t i = 0u; i < kBsdfSamples; ++i) {
    etx::Sampler sampler(seed + i, seed ^ (i * 19u + 5u));
    const etx::BSDFSample sample = etx::bsdf::sample(data, material, sampler);
    if (validate_sample(sample) == false) {
      return false;
    }

    if (sample.valid() == false) {
      continue;
    }

    etx::Sampler pdf_sampler(seed + i + 100000u, seed ^ (i * 23u + 13u));
    const float pdf = etx::bsdf::pdf(data, sample.w_o, material, pdf_sampler);
    if (std::isfinite(pdf) == false) {
      return false;
    }

    const float tolerance = max(1.0e-3f, 0.05f * max(sample.pdf, pdf));
    if (fabsf(sample.pdf - pdf) > tolerance) {
      return false;
    }
  }

  return true;
}

bool validate_reverse_pdf_match(const etx::BSDFData& data, const etx::Material& material, const uint32_t seed) {
  etx::Sampler sampler(seed, seed ^ 0x8ed4f2a7u);

  for (uint32_t i = 0u; i < kBsdfSamples; ++i) {
    const float3 w_o = sample_uniform_sphere(sampler);
    etx::Sampler reverse_pdf_sampler(seed + i + 120000u, seed ^ (i * 29u + 17u));
    const float reverse_pdf = etx::bsdf::reverse_pdf(data, w_o, material, reverse_pdf_sampler);
    if (std::isfinite(reverse_pdf) == false) {
      return false;
    }

    etx::BSDFData reverse_data = data;
    reverse_data.w_i = -w_o;
    if (data.path_source == etx::PathSource::Camera) {
      reverse_data.path_source = etx::PathSource::Light;
    } else if (data.path_source == etx::PathSource::Light) {
      reverse_data.path_source = etx::PathSource::Camera;
    }
    etx::Sampler expected_pdf_sampler(seed + i + 140000u, seed ^ (i * 31u + 19u));
    const float expected_pdf = etx::bsdf::pdf(reverse_data, -data.w_i, material, expected_pdf_sampler);
    if (std::isfinite(expected_pdf) == false) {
      return false;
    }

    const float tolerance = max(1.0e-3f, 0.05f * max(reverse_pdf, expected_pdf));
    if (fabsf(reverse_pdf - expected_pdf) > tolerance) {
      return false;
    }
  }

  return true;
}

bool validate_equal_ior_dielectric_direction(const char* label, const etx::BSDFData& data, const etx::Material& material, const uint32_t seed) {
  etx::Material material_with_media = material;
  material_with_media.ext_medium = 7u;
  material_with_media.int_medium = 11u;

  etx::Sampler sampler(seed, seed ^ 0x9234a7bu);
  const etx::BSDFSample sample = etx::bsdf::sample(data, material_with_media, sampler);
  const bool reflection = (sample.properties & etx::BSDFSample::Reflection) != 0u;
  const bool transmission = (sample.properties & etx::BSDFSample::Transmission) != 0u;
  const bool medium_changed = (sample.properties & etx::BSDFSample::MediumChanged) != 0u;
  const bool entering = dot(data.nrm, data.w_i) < 0.0f;
  const uint32_t expected_medium = entering ? material_with_media.int_medium : material_with_media.ext_medium;
  const float3 direction_error = sample.w_o - data.w_i;

  if ((sample.valid() == false) || (sample.is_delta() == false) || reflection || (transmission == false) || (medium_changed == false) ||
      (sample.medium_index != expected_medium) || (fabsf(sample.pdf - 1.0f) > kEpsilon) || (fabsf(sample.eta - 1.0f) > kEpsilon) ||
      (sample.weight.valid() == false) || (fabsf(sample.weight.monochromatic() - 1.0f) > kEpsilon) || (dot(direction_error, direction_error) > 1.0e-8f)) {
    std::printf("%s equal-IOR sample failed pdf %.6f eta %.6f medium %u expected %u weight %.6f\n", label, sample.pdf, sample.eta, sample.medium_index, expected_medium,
      sample.weight.monochromatic());
    return false;
  }

  etx::Sampler eval_sampler(seed + 1u, seed ^ 0x34bac21u);
  const etx::BSDFEval eval = etx::bsdf::evaluate(data, data.w_i, material_with_media, eval_sampler);
  etx::Sampler pdf_sampler(seed + 2u, seed ^ 0x7129ea1u);
  const float pdf = etx::bsdf::pdf(data, data.w_i, material_with_media, pdf_sampler);
  if ((eval.bsdf.maximum() > kEpsilon) || (pdf > kEpsilon)) {
    std::printf("%s equal-IOR delta eval/pdf failed bsdf %.6f pdf %.6f\n", label, eval.bsdf.maximum(), pdf);
    return false;
  }

  return true;
}

struct SampleWeightStats {
  float3 average = {};
  float average_inverse_pdf = 0.0f;
  float p95_weight = 0.0f;
  float p99_weight = 0.0f;
  float max_weight = 0.0f;
  uint32_t valid_count = 0u;
  uint32_t nonzero_count = 0u;
  uint32_t invalid_count = 0u;
  uint32_t zero_pdf_count = 0u;
};

SampleWeightStats sample_weight_stats_rgb(const etx::BSDFData& data, const etx::Material& material, const uint32_t seed) {
  SampleWeightStats result = {};
  float weights[kBsdfSamples] = {};

  for (uint32_t i = 0u; i < kBsdfSamples; ++i) {
    etx::Sampler sampler(seed + i, seed ^ (i * 41u + 23u));
    const etx::BSDFSample sample = etx::bsdf::sample(data, material, sampler);
    if (validate_sample(sample) == false) {
      result.average = float3{-1.0f, -1.0f, -1.0f};
      result.invalid_count += 1u;
      return result;
    }

    weights[i] = 0.0f;
    if (sample.valid() == false) {
      result.zero_pdf_count += 1u;
      continue;
    }

    result.average += sample.weight.to_rgb();
    if (sample.pdf > kEpsilon) {
      result.average_inverse_pdf += 1.0f / sample.pdf;
    }
    const float weight = sample.weight.maximum();
    weights[result.valid_count] = weight;
    result.max_weight = max(result.max_weight, weight);
    result.valid_count += 1u;
    if (sample.weight.maximum() > kEpsilon) {
      result.nonzero_count += 1u;
    }
  }

  std::sort(weights, weights + kBsdfSamples);
  const uint32_t p95_index = min(kBsdfSamples - 1u, static_cast<uint32_t>(0.95f * static_cast<float>(kBsdfSamples - 1u)));
  const uint32_t p99_index = min(kBsdfSamples - 1u, static_cast<uint32_t>(0.99f * static_cast<float>(kBsdfSamples - 1u)));
  result.p95_weight = weights[p95_index];
  result.p99_weight = weights[p99_index];

  if (result.valid_count > 0u) {
    result.average *= (1.0f / static_cast<float>(kBsdfSamples));
    result.average_inverse_pdf *= (1.0f / static_cast<float>(result.valid_count));
  }
  return result;
}

float integrate_bsdf_energy(const etx::BSDFData& data, const etx::Material& material, const uint32_t seed) {
  etx::Sampler sampler(seed, seed ^ 0x517cc1b7u);
  float sum = 0.0f;

  for (uint32_t i = 0u; i < kIntegrationSamples; ++i) {
    const float3 w_o = sample_uniform_sphere(sampler);
    etx::Sampler eval_sampler(seed + i, seed ^ (i * 13u + 11u));
    const etx::BSDFEval eval = etx::bsdf::evaluate(data, w_o, material, eval_sampler);
    if (finite_response(eval.bsdf) == false) {
      return -1.0f;
    }
    if (non_negative_response(eval.bsdf) == false) {
      return -1.0f;
    }
    sum += eval.bsdf.monochromatic();
  }

  return (sum / static_cast<float>(kIntegrationSamples)) * 4.0f * kPi;
}

etx::Material make_conductor(const float roughness) {
  etx::Material result = {};
  result.cls = MaterialClass::Conductor;
  result.reflectance.spectrum_index = SpectrumWhite;
  result.roughness.value = float4{roughness, roughness, 0.0f, 0.0f};
  result.ext_ior.cls = etx::SpectralDistribution::Dielectric;
  result.int_ior.cls = etx::SpectralDistribution::Conductor;
  result.int_ior.eta_index = SpectrumConductorEta;
  result.int_ior.k_index = SpectrumConductorK;
  return result;
}

etx::Material make_mirror_conductor(const float roughness) {
  etx::Material result = make_conductor(roughness);
  result.int_ior.eta_index = SpectrumMirrorEta;
  result.int_ior.k_index = SpectrumMirrorK;
  return result;
}

etx::Material make_dielectric(const float roughness) {
  etx::Material result = {};
  result.cls = MaterialClass::Dielectric;
  result.reflectance.spectrum_index = SpectrumWhite;
  result.scattering.spectrum_index = SpectrumColored;
  result.roughness.value = float4{roughness, roughness, 0.0f, 0.0f};
  result.ext_ior.cls = etx::SpectralDistribution::Dielectric;
  result.int_ior.cls = etx::SpectralDistribution::Dielectric;
  result.int_ior.eta_index = SpectrumDielectricEta;
  return result;
}

etx::Material make_white_dielectric(const float roughness) {
  etx::Material result = make_dielectric(roughness);
  result.scattering.spectrum_index = SpectrumWhite;
  return result;
}

etx::Material make_white_sapphire_dielectric(const float roughness) {
  etx::Material result = make_white_dielectric(roughness);
  result.int_ior.eta_index = SpectrumSapphireEta;
  return result;
}

etx::Material make_white_equal_ior_dielectric(const float roughness) {
  etx::Material result = make_white_dielectric(roughness);
  result.ext_ior.eta_index = SpectrumAirEta;
  result.int_ior.eta_index = SpectrumAirEta;
  return result;
}

etx::Material make_named_plastic_water_dielectric(const float roughness) {
  etx::Material result = make_white_dielectric(roughness);
  result.ext_ior.eta_index = SpectrumNamedPlasticEta;
  result.ext_ior.k_index = SpectrumNamedPlasticK;
  result.int_ior.eta_index = SpectrumNamedWaterEta;
  result.int_ior.k_index = SpectrumNamedWaterK;
  return result;
}

etx::Material make_diffuse(const float roughness) {
  etx::Material result = {};
  result.cls = MaterialClass::Diffuse;
  result.scattering.spectrum_index = SpectrumWhite;
  result.roughness.value = float4{roughness, roughness, 0.0f, 0.0f};
  return result;
}

etx::Material make_plastic(const float roughness) {
  etx::Material result = {};
  result.cls = MaterialClass::Plastic;
  result.reflectance.spectrum_index = SpectrumWhite;
  result.scattering.spectrum_index = SpectrumWhite;
  result.roughness.value = float4{roughness, roughness, 0.0f, 0.0f};
  result.ext_ior.cls = etx::SpectralDistribution::Dielectric;
  result.ext_ior.eta_index = SpectrumAirEta;
  result.int_ior.cls = etx::SpectralDistribution::Dielectric;
  result.int_ior.eta_index = SpectrumDielectricEta;
  return result;
}

etx::Material make_black_substrate_plastic(const float roughness) {
  etx::Material result = make_plastic(roughness);
  result.scattering.spectrum_index = SpectrumBlack;
  return result;
}

etx::Material make_colored_substrate_black_coating_plastic(const float roughness) {
  etx::Material result = make_plastic(roughness);
  result.reflectance.spectrum_index = SpectrumBlack;
  result.scattering.spectrum_index = SpectrumColored;
  return result;
}

void set_basic_thinfilm(etx::Material& material, const float min_thickness, const float max_thickness) {
  material.thinfilm.min_thickness = min_thickness;
  material.thinfilm.max_thickness = max_thickness;
  material.thinfilm.ior.cls = etx::SpectralDistribution::Dielectric;
  material.thinfilm.ior.eta_index = SpectrumDielectricEta;
  material.thinfilm.ior.k_index = SpectrumBlack;
}

etx::Material make_standalone_thinfilm(const float min_thickness, const float max_thickness) {
  etx::Material result = {};
  result.cls = MaterialClass::Thinfilm;
  result.reflectance.spectrum_index = SpectrumWhite;
  result.scattering.spectrum_index = SpectrumWhite;
  result.ext_ior.cls = etx::SpectralDistribution::Dielectric;
  result.ext_ior.eta_index = SpectrumAirEta;
  result.int_ior.cls = etx::SpectralDistribution::Dielectric;
  result.int_ior.eta_index = SpectrumAirEta;
  set_basic_thinfilm(result, min_thickness, max_thickness);
  return result;
}

etx::Material make_thinfilm_delta_dielectric() {
  etx::Material result = make_white_dielectric(0.0f);
  set_basic_thinfilm(result, 500.0f, 500.0f);
  return result;
}

etx::Material make_thinfilm_delta_conductor() {
  etx::Material result = make_mirror_conductor(0.0f);
  set_basic_thinfilm(result, 500.0f, 500.0f);
  return result;
}

etx::Material make_thinfilm_delta_plastic() {
  etx::Material result = make_plastic(0.0f);
  set_basic_thinfilm(result, 500.0f, 500.0f);
  return result;
}

bool validate_energy_compensated_material(const char* label, const etx::BSDFData& data, const etx::Material& material, const float roughness, const uint32_t seed) {
  const float pdf_integral = integrate_pdf(data, material, seed);
  const float reverse_pdf_integral = integrate_reverse_pdf(data, material, seed + 50000u);
  const bool narrow_lobe = roughness <= 0.1f;

  if ((std::isfinite(pdf_integral) == false) || (pdf_integral < 0.0f) || (std::isfinite(reverse_pdf_integral) == false) || (reverse_pdf_integral < 0.0f)) {
    std::printf("%s roughness %.3f pdf %.6f reverse %.6f\n", label, roughness, pdf_integral, reverse_pdf_integral);
    return false;
  }

  if ((narrow_lobe == false) && (fabsf(pdf_integral - 1.0f) > 0.12f)) {
    std::printf("%s roughness %.3f pdf %.6f reverse %.6f\n", label, roughness, pdf_integral, reverse_pdf_integral);
    return false;
  }

  if (validate_sampling(data, material, seed + 10000u) == false) {
    std::printf("%s roughness %.3f produced invalid sample\n", label, roughness);
    return false;
  }

  if (validate_sample_pdf_match(data, material, seed + 15000u) == false) {
    std::printf("%s roughness %.3f sample pdf mismatch\n", label, roughness);
    return false;
  }

  if (validate_reverse_pdf_match(data, material, seed + 17000u) == false) {
    std::printf("%s roughness %.3f reverse pdf mismatch\n", label, roughness);
    return false;
  }

  const SampleWeightStats stats = sample_weight_stats_rgb(data, material, seed + 20000u);
  if ((stats.invalid_count > 0u) || (stats.valid_count == 0u) || (std::isfinite(stats.average.x) == false) || (std::isfinite(stats.max_weight) == false)) {
    std::printf("%s roughness %.3f invalid %u valid %u max_weight %.6f\n", label, roughness, stats.invalid_count, stats.valid_count, stats.max_weight);
    return false;
  }

  const float bsdf_energy = integrate_bsdf_energy(data, material, seed + 30000u);
  if ((std::isfinite(bsdf_energy) == false) || (bsdf_energy < 0.0f)) {
    std::printf("%s roughness %.3f bsdf energy %.6f\n", label, roughness, bsdf_energy);
    return false;
  }

  std::printf("%s roughness %.3f pdf %.6f reverse %.6f avg %.6f %.6f %.6f p95 %.6f p99 %.6f max %.6f valid %u zero_pdf %u bsdf_energy %.6f\n", label, roughness,
    pdf_integral, reverse_pdf_integral, stats.average.x, stats.average.y, stats.average.z, stats.p95_weight, stats.p99_weight, stats.max_weight, stats.valid_count,
    stats.zero_pdf_count, bsdf_energy);
  return true;
}

bool validate_energy_compensated_white_furnace_direction(const char* label, const float3& w_i_world, const etx::Material& material, const float roughness, const uint32_t seed) {
  const Vertex contract_vertex = {
    float3{0.0f, 0.0f, 0.0f},
    float3{0.0f, 0.0f, 1.0f},
    float3{1.0f, 0.0f, 0.0f},
    float3{0.0f, 1.0f, 0.0f},
    float2{0.5f, 0.5f},
  };
  const etx::BSDFData data = {etx::SpectralQuery{}, kInvalidIndex, etx::PathSource::Camera, contract_vertex, w_i_world};
  const SampleWeightStats stats = sample_weight_stats_rgb(data, material, seed);
  const float3 average_rgb = stats.average;
  const float max_rgb_error = max(fabsf(average_rgb.x - 1.0f), max(fabsf(average_rgb.y - 1.0f), fabsf(average_rgb.z - 1.0f)));

  if ((stats.invalid_count > 0u) || (stats.valid_count == 0u) || (std::isfinite(average_rgb.x) == false) || (std::isfinite(average_rgb.y) == false) ||
      (std::isfinite(average_rgb.z) == false) || (max_rgb_error > 0.12f)) {
    std::printf("%s furnace wi %.3f %.3f %.3f roughness %.3f rgb %.6f %.6f %.6f p95 %.6f p99 %.6f max %.6f valid %u zero_pdf %u invalid %u\n", label,
      w_i_world.x, w_i_world.y, w_i_world.z, roughness, average_rgb.x, average_rgb.y, average_rgb.z, stats.p95_weight, stats.p99_weight, stats.max_weight, stats.valid_count,
      stats.zero_pdf_count, stats.invalid_count);
    return false;
  }

  std::printf("%s furnace wi %.3f %.3f %.3f roughness %.3f rgb %.6f %.6f %.6f p95 %.6f p99 %.6f max %.6f valid %u zero_pdf %u invalid %u\n", label, w_i_world.x,
    w_i_world.y, w_i_world.z, roughness, average_rgb.x, average_rgb.y, average_rgb.z, stats.p95_weight, stats.p99_weight, stats.max_weight, stats.valid_count,
    stats.zero_pdf_count, stats.invalid_count);
  return true;
}

bool validate_eon_diffuse_material(const etx::BSDFData& data, const float roughness, const uint32_t seed) {
  const etx::Material material = make_diffuse(roughness);
  bool diagnostic_valid = validate_energy_compensated_material("eon diffuse", data, material, roughness, seed);
  diagnostic_valid = validate_energy_compensated_white_furnace_direction("eon diffuse", float3{0.0f, 0.0f, -1.0f}, material, roughness, seed + 100u) && diagnostic_valid;
  diagnostic_valid =
    validate_energy_compensated_white_furnace_direction("eon diffuse", normalize(float3{0.8660254f, 0.0f, -0.5f}), material, roughness, seed + 200u) && diagnostic_valid;
  return diagnostic_valid;
}

bool validate_plastic_sample_contract(const char* label, const etx::BSDFData& data, const etx::Material& material, const float roughness, const uint32_t seed) {
  etx::Material material_with_media = material;
  material_with_media.ext_medium = 7u;
  material_with_media.int_medium = 11u;

  bool saw_coating = false;
  bool saw_substrate = false;
  for (uint32_t i = 0u; i < kBsdfSamples; ++i) {
    etx::Sampler sampler(seed + i, seed ^ (i * 47u + 31u));
    const etx::BSDFSample sample = etx::bsdf::sample(data, material_with_media, sampler);
    if ((validate_sample(sample) == false) || (sample.valid() == false)) {
      std::printf("%s roughness %.3f invalid plastic sample\n", label, roughness);
      return false;
    }

    const bool reflection = (sample.properties & etx::BSDFSample::Reflection) != 0u;
    const bool diffuse = (sample.properties & etx::BSDFSample::Diffuse) != 0u;
    const bool transmission = (sample.properties & etx::BSDFSample::Transmission) != 0u;
    const bool medium_changed = (sample.properties & etx::BSDFSample::MediumChanged) != 0u;
    if ((((reflection == false) || transmission) || medium_changed) || ((sample.medium_index != data.current_medium) || (fabsf(sample.eta - 1.0f) > 1.0e-4f))) {
      std::printf("%s roughness %.3f invalid plastic metadata eta %.6f medium %u\n", label, roughness, sample.eta, sample.medium_index);
      return false;
    }

    if (diffuse) {
      saw_substrate = true;
    } else {
      saw_coating = true;
    }

    etx::Sampler eval_sampler(seed + i + 10000u, seed ^ (i * 53u + 37u));
    const etx::BSDFEval eval = etx::bsdf::evaluate(data, sample.w_o, material_with_media, eval_sampler);
    if ((eval.valid() == false) || (finite_response(eval.bsdf) == false)) {
      std::printf("%s roughness %.3f invalid plastic eval at sampled direction\n", label, roughness);
      return false;
    }

    etx::Sampler pdf_sampler(seed + i + 20000u, seed ^ (i * 59u + 41u));
    const float pdf = etx::bsdf::pdf(data, sample.w_o, material_with_media, pdf_sampler);
    if ((std::isfinite(pdf) == false) || (pdf <= kEpsilon)) {
      std::printf("%s roughness %.3f invalid plastic pdf %.6f\n", label, roughness, pdf);
      return false;
    }

    const float pdf_tolerance = max(1.0e-3f, 0.05f * max(sample.pdf, pdf));
    if (fabsf(sample.pdf - pdf) > pdf_tolerance) {
      std::printf("%s roughness %.3f plastic sample/pdf mismatch sample %.6f pdf %.6f\n", label, roughness, sample.pdf, pdf);
      return false;
    }

    const etx::SpectralResponse expected_weight = eval.bsdf / pdf;
    const float weight_error = fabsf(sample.weight.monochromatic() - expected_weight.monochromatic());
    const float weight_tolerance = max(1.0e-3f, 0.05f * max(sample.weight.monochromatic(), expected_weight.monochromatic()));
    if (weight_error > weight_tolerance) {
      std::printf("%s roughness %.3f plastic weight mismatch sample %.6f expected %.6f\n", label, roughness, sample.weight.monochromatic(),
        expected_weight.monochromatic());
      return false;
    }
  }

  if ((saw_coating == false) || (saw_substrate == false)) {
    std::printf("%s roughness %.3f missing plastic branch coating %u substrate %u\n", label, roughness, saw_coating ? 1u : 0u, saw_substrate ? 1u : 0u);
    return false;
  }

  std::printf("%s roughness %.3f plastic sample contract valid\n", label, roughness);
  return true;
}

bool validate_plastic_inner_matches_outer(const char* label, const etx::BSDFData& outside_data, const etx::BSDFData& inside_data, const etx::Material& material,
  const float roughness, const uint32_t seed) {
  const float3 outside_w_o = normalize(float3{0.25f, 0.1f, 0.963068f});
  const float3 inside_w_o = -outside_w_o;
  etx::Sampler outside_eval_sampler(seed, seed ^ 0x75e1a9cu);
  const etx::BSDFEval outside_eval = etx::bsdf::evaluate(outside_data, outside_w_o, material, outside_eval_sampler);
  etx::Sampler inside_eval_sampler(seed + 1u, seed ^ 0x9830cf1u);
  const etx::BSDFEval inside_eval = etx::bsdf::evaluate(inside_data, inside_w_o, material, inside_eval_sampler);
  if ((outside_eval.valid() == false) || (inside_eval.valid() == false)) {
    std::printf("%s roughness %.3f invalid inner/outer plastic eval\n", label, roughness);
    return false;
  }

  const float bsdf_error = fabsf(outside_eval.bsdf.monochromatic() - inside_eval.bsdf.monochromatic());
  const float bsdf_tolerance = max(1.0e-4f, 5.0e-3f * max(outside_eval.bsdf.monochromatic(), inside_eval.bsdf.monochromatic()));
  if (bsdf_error > bsdf_tolerance) {
    std::printf("%s roughness %.3f inner/outer bsdf mismatch outside %.6f inside %.6f\n", label, roughness, outside_eval.bsdf.monochromatic(),
      inside_eval.bsdf.monochromatic());
    return false;
  }

  etx::Sampler outside_pdf_sampler(seed + 2u, seed ^ 0x16bca45u);
  const float outside_pdf = etx::bsdf::pdf(outside_data, outside_w_o, material, outside_pdf_sampler);
  etx::Sampler inside_pdf_sampler(seed + 3u, seed ^ 0x85cf034u);
  const float inside_pdf = etx::bsdf::pdf(inside_data, inside_w_o, material, inside_pdf_sampler);
  const float pdf_tolerance = max(1.0e-4f, 5.0e-3f * max(outside_pdf, inside_pdf));
  if (fabsf(outside_pdf - inside_pdf) > pdf_tolerance) {
    std::printf("%s roughness %.3f inner/outer pdf mismatch outside %.6f inside %.6f\n", label, roughness, outside_pdf, inside_pdf);
    return false;
  }

  if (validate_plastic_sample_contract(label, inside_data, material, roughness, seed + 1000u) == false) {
    return false;
  }

  std::printf("%s roughness %.3f inner side matches outer side\n", label, roughness);
  return true;
}

bool validate_plastic_black_substrate_matches_dielectric_reflection(const char* label, const etx::BSDFData& data, const etx::Material& plastic,
  const etx::Material& dielectric, const float roughness, const uint32_t seed) {
  const float3 outgoing_direction = normalize(float3{0.0f, 0.0f, 1.0f});
  etx::Sampler plastic_eval_sampler(seed, seed ^ 0x7a53d21u);
  const etx::BSDFEval plastic_eval = etx::bsdf::evaluate(data, outgoing_direction, plastic, plastic_eval_sampler);
  etx::Sampler dielectric_eval_sampler(seed + 1u, seed ^ 0x3e19f5bu);
  const etx::BSDFEval dielectric_eval = etx::bsdf::evaluate(data, outgoing_direction, dielectric, dielectric_eval_sampler);
  if ((plastic_eval.valid() == false) || (dielectric_eval.valid() == false)) {
    std::printf("%s roughness %.3f invalid coating comparison eval\n", label, roughness);
    return false;
  }

  const float bsdf_error = fabsf(plastic_eval.bsdf.monochromatic() - dielectric_eval.bsdf.monochromatic());
  const float bsdf_tolerance = max(1.0e-4f, 5.0e-3f * max(plastic_eval.bsdf.monochromatic(), dielectric_eval.bsdf.monochromatic()));
  if (bsdf_error > bsdf_tolerance) {
    std::printf("%s roughness %.3f coating bsdf %.6f dielectric %.6f\n", label, roughness, plastic_eval.bsdf.monochromatic(),
      dielectric_eval.bsdf.monochromatic());
    return false;
  }

  const BSDFResourceContext context = etx::bsdf::detail::make_interop_context();
  const ::BSDFData interop_data = etx::bsdf::detail::make_interop_data(data);
  const LocalFrame frame = bsdf_plastic_coating_frame(interop_data, plastic);
  const float3 local_w_i = local_frame_to_local(frame, -interop_data.w_i);
  const float alpha = bsdf_energy_compensated_scalar_roughness(context, plastic, interop_data.tex);
  const BSDFPlasticCoatingReflectionProposal proposal =
    bsdf_plastic_coating_reflection_proposal(context, interop_data.spectrum_sample, plastic, local_w_i, alpha);
  if (proposal.probability <= kEpsilon) {
    std::printf("%s roughness %.3f invalid coating proposal probability %.6f\n", label, roughness, proposal.probability);
    return false;
  }

  etx::Sampler plastic_pdf_sampler(seed + 2u, seed ^ 0x6452ce7u);
  const float plastic_pdf = etx::bsdf::pdf(data, outgoing_direction, plastic, plastic_pdf_sampler);
  etx::Sampler dielectric_pdf_sampler(seed + 3u, seed ^ 0x150abe3u);
  const float dielectric_pdf = etx::bsdf::pdf(data, outgoing_direction, dielectric, dielectric_pdf_sampler);
  const float expected_dielectric_pdf = plastic_pdf * proposal.probability;
  const float pdf_tolerance = max(1.0e-4f, 5.0e-3f * max(expected_dielectric_pdf, dielectric_pdf));
  if (fabsf(expected_dielectric_pdf - dielectric_pdf) > pdf_tolerance) {
    std::printf("%s roughness %.3f coating pdf %.6f proposal %.6f dielectric %.6f\n", label, roughness, plastic_pdf, proposal.probability, dielectric_pdf);
    return false;
  }

  std::printf("%s roughness %.3f coating matches dielectric reflection\n", label, roughness);
  return true;
}

bool validate_plastic_low_roughness_substrate_exit(etx::Scene& original_scene, const etx::SpectralDistribution* spectra, const uint32_t spectrum_count) {
  const float roughness = 0.0f;
  etx::TaskScheduler scheduler = {};
  etx::SceneData scene_data(scheduler);
  scene_data.images.init(16u);
  scene_data.spectrum_values.assign(spectra, spectra + spectrum_count);
  scene_data.materials.emplace_back(make_colored_substrate_black_coating_plastic(roughness));

  if (etx::ensure_energy_compensation_interfaces(scene_data, scheduler) == false) {
    std::printf("plastic low roughness substrate exit failed to bind LUT\n");
    return false;
  }

  scene_data.images.load_images(scheduler);

  etx::Scene exact_scene = {};
  exact_scene.spectrums = etx::ArrayView<etx::SpectralDistribution>{scene_data.spectrum_values.data(), scene_data.spectrum_values.size()};
  exact_scene.images = etx::ArrayView<etx::Image>{scene_data.images.as_array(), scene_data.images.array_size()};
  exact_scene.materials = etx::ArrayView<etx::Material>{scene_data.materials.data(), scene_data.materials.size()};
  exact_scene.energy_compensation_interfaces =
    etx::ArrayView<etx::Scene::EnergyCompensationInterface>{scene_data.energy_compensation_interfaces.data(), scene_data.energy_compensation_interfaces.size()};

  etx::scene_global_clear(&original_scene);
  etx::scene_global_publish(&exact_scene, &exact_scene);

  const etx::Material material = scene_data.materials[0];
  const Vertex vertex = {
    float3{0.0f, 0.0f, 0.0f},
    float3{0.0f, 0.0f, 1.0f},
    float3{1.0f, 0.0f, 0.0f},
    float3{0.0f, 1.0f, 0.0f},
    float2{0.5f, 0.5f},
  };
  const etx::BSDFData data = {etx::SpectralQuery{}, kInvalidIndex, etx::PathSource::Camera, vertex, float3{0.0f, 0.0f, -1.0f}};
  const float external_mu_values[] = {0.8f, 0.5f, 0.25f, 0.05f};
  bool diagnostic_valid = true;
  for (uint32_t i = 0u; i < 4u; ++i) {
    const float mu = external_mu_values[i];
    const float sin_theta = sqrtf(max(0.0f, 1.0f - mu * mu));
    const float3 outgoing_direction = normalize(float3{sin_theta, 0.0f, mu});
    etx::Sampler eval_sampler(43000u + i, 0x9e3779b9u ^ i);
    const etx::BSDFEval eval = etx::bsdf::evaluate(data, outgoing_direction, material, eval_sampler);
    const float3 rgb = eval.bsdf.to_rgb();
    const bool substrate_visible = (eval.valid() && (std::isfinite(rgb.x)) && (rgb.x > 1.0e-3f));
    if (substrate_visible == false) {
      std::printf("plastic low roughness substrate exit mu %.3f failed rgb %.6f %.6f %.6f pdf %.6f\n", mu, rgb.x, rgb.y, rgb.z, eval.pdf);
      diagnostic_valid = false;
    } else {
      std::printf("plastic low roughness substrate exit mu %.3f rgb %.6f %.6f %.6f pdf %.6f\n", mu, rgb.x, rgb.y, rgb.z, eval.pdf);
    }
  }

  etx::scene_global_clear(&exact_scene);
  etx::scene_global_publish(&original_scene, &original_scene);
  return diagnostic_valid;
}

bool validate_standalone_thinfilm_contract(const char* label, const etx::BSDFData& data, const etx::Material& material, const uint32_t seed) {
  etx::Material material_with_media = material;
  material_with_media.ext_medium = 7u;
  material_with_media.int_medium = 11u;
  const uint32_t expected_transmission_medium = data.current_medium;

  bool saw_reflection = false;
  bool saw_transmission = false;
  for (uint32_t i = 0u; i < kBsdfSamples; ++i) {
    etx::Sampler sampler(seed + i, seed ^ (i * 67u + 43u));
    const etx::BSDFSample sample = etx::bsdf::sample(data, material_with_media, sampler);
    if ((validate_sample(sample) == false) || (sample.valid() == false) || (sample.is_delta() == false)) {
      std::printf("%s invalid standalone thinfilm sample\n", label);
      return false;
    }

    const bool reflection = (sample.properties & etx::BSDFSample::Reflection) != 0u;
    const bool transmission = (sample.properties & etx::BSDFSample::Transmission) != 0u;
    const bool medium_changed = (sample.properties & etx::BSDFSample::MediumChanged) != 0u;
    if (reflection) {
      saw_reflection = true;
      if ((transmission || medium_changed) || ((sample.medium_index != data.current_medium) || (fabsf(sample.eta - 1.0f) > kEpsilon))) {
        std::printf("%s invalid standalone thinfilm reflection metadata\n", label);
        return false;
      }
    } else if (transmission) {
      saw_transmission = true;
      if ((medium_changed || (sample.medium_index != expected_transmission_medium)) || (fabsf(sample.eta - 1.0f) > kEpsilon)) {
        std::printf("%s invalid standalone thinfilm transmission metadata medium %u expected %u\n", label, sample.medium_index, expected_transmission_medium);
        return false;
      }
    } else {
      std::printf("%s standalone thinfilm sample missing branch flags\n", label);
      return false;
    }

    etx::Sampler eval_sampler(seed + i + 10000u, seed ^ (i * 71u + 47u));
    const etx::BSDFEval eval = etx::bsdf::evaluate(data, sample.w_o, material_with_media, eval_sampler);
    etx::Sampler pdf_sampler(seed + i + 20000u, seed ^ (i * 73u + 53u));
    const float pdf = etx::bsdf::pdf(data, sample.w_o, material_with_media, pdf_sampler);
    if ((eval.valid() == false) || (std::isfinite(pdf) == false) || (pdf <= kEpsilon)) {
      std::printf("%s invalid standalone thinfilm eval/pdf %.6f\n", label, pdf);
      return false;
    }

    const float pdf_tolerance = max(1.0e-4f, 5.0e-3f * max(sample.pdf, pdf));
    if (fabsf(sample.pdf - pdf) > pdf_tolerance) {
      std::printf("%s standalone thinfilm pdf mismatch sample %.6f pdf %.6f\n", label, sample.pdf, pdf);
      return false;
    }

    const etx::SpectralResponse expected_weight = eval.bsdf / pdf;
    const float weight_error = fabsf(sample.weight.monochromatic() - expected_weight.monochromatic());
    const float weight_tolerance = max(1.0e-4f, 5.0e-3f * max(sample.weight.monochromatic(), expected_weight.monochromatic()));
    if (weight_error > weight_tolerance) {
      std::printf("%s standalone thinfilm weight mismatch sample %.6f expected %.6f\n", label, sample.weight.monochromatic(), expected_weight.monochromatic());
      return false;
    }
  }

  if ((saw_reflection == false) || (saw_transmission == false)) {
    std::printf("%s standalone thinfilm missing branch reflection %u transmission %u\n", label, saw_reflection ? 1u : 0u, saw_transmission ? 1u : 0u);
    return false;
  }

  std::printf("%s standalone thinfilm contract valid\n", label);
  return true;
}

bool validate_standalone_thinfilm_sheet_symmetry(const char* label, const etx::BSDFData& base_data, const etx::Material& material, const uint32_t seed) {
  etx::Material material_with_boundary_ior = material;
  material_with_boundary_ior.ext_medium = 7u;
  material_with_boundary_ior.int_medium = 11u;
  material_with_boundary_ior.int_ior.cls = etx::SpectralDistribution::Dielectric;
  material_with_boundary_ior.int_ior.eta_index = SpectrumDielectricEta;
  material_with_boundary_ior.int_ior.k_index = SpectrumBlack;

  const float mu_values[] = {1.0f, 0.5f, 0.2f, 0.05f};
  for (uint32_t i = 0u; i < 4u; ++i) {
    const float mu = mu_values[i];
    const float sin_theta = sqrtf(max(0.0f, 1.0f - mu * mu));
    etx::BSDFData outside_data = base_data;
    outside_data.w_i = normalize(float3{sin_theta, 0.0f, -mu});
    etx::BSDFData inside_data = base_data;
    inside_data.w_i = normalize(float3{sin_theta, 0.0f, mu});

    const float3 outside_reflection = normalize(reflect(outside_data.w_i, outside_data.nrm));
    const float3 inside_reflection = normalize(reflect(inside_data.w_i, -inside_data.nrm));

    etx::Sampler outside_sampler(seed + i, seed ^ (i * 101u + 73u));
    const etx::BSDFEval outside_eval = etx::bsdf::evaluate(outside_data, outside_reflection, material_with_boundary_ior, outside_sampler);
    etx::Sampler inside_sampler(seed + i + 100u, seed ^ (i * 103u + 79u));
    const etx::BSDFEval inside_eval = etx::bsdf::evaluate(inside_data, inside_reflection, material_with_boundary_ior, inside_sampler);
    if ((outside_eval.valid() == false) || (inside_eval.valid() == false)) {
      std::printf("%s invalid symmetric thinfilm reflection mu %.3f\n", label, mu);
      return false;
    }

    const float outside_value = outside_eval.bsdf.monochromatic();
    const float inside_value = inside_eval.bsdf.monochromatic();
    const float value_tolerance = max(1.0e-4f, 1.0e-3f * max(outside_value, inside_value));
    if (fabsf(outside_value - inside_value) > value_tolerance) {
      std::printf("%s asymmetric thinfilm reflection mu %.3f outside %.6f inside %.6f\n", label, mu, outside_value, inside_value);
      return false;
    }

    etx::Sampler outside_transmission_sampler(seed + i + 200u, seed ^ (i * 107u + 83u));
    const etx::BSDFEval outside_transmission = etx::bsdf::evaluate(outside_data, outside_data.w_i, material_with_boundary_ior, outside_transmission_sampler);
    etx::Sampler inside_transmission_sampler(seed + i + 300u, seed ^ (i * 109u + 89u));
    const etx::BSDFEval inside_transmission = etx::bsdf::evaluate(inside_data, inside_data.w_i, material_with_boundary_ior, inside_transmission_sampler);
    if ((outside_transmission.valid() == false) || (inside_transmission.valid() == false)) {
      std::printf("%s invalid symmetric thinfilm transmission mu %.3f\n", label, mu);
      return false;
    }

    const bool outside_medium_changed = (outside_transmission.properties & etx::BSDFSample::MediumChanged) != 0u;
    const bool inside_medium_changed = (inside_transmission.properties & etx::BSDFSample::MediumChanged) != 0u;
    if (((outside_medium_changed || inside_medium_changed) || (outside_transmission.medium_index != outside_data.current_medium)) ||
        (inside_transmission.medium_index != inside_data.current_medium)) {
      std::printf("%s standalone thinfilm transmission changed medium at mu %.3f\n", label, mu);
      return false;
    }

    const float outside_transmission_pdf = etx::bsdf::pdf(outside_data, outside_data.w_i, material_with_boundary_ior, outside_transmission_sampler);
    const float inside_transmission_pdf = etx::bsdf::pdf(inside_data, inside_data.w_i, material_with_boundary_ior, inside_transmission_sampler);
    if ((outside_transmission_pdf <= kEpsilon) || (inside_transmission_pdf <= kEpsilon)) {
      std::printf("%s standalone thinfilm artificial total reflection mu %.3f outside %.6f inside %.6f\n", label, mu, outside_transmission_pdf, inside_transmission_pdf);
      return false;
    }
  }

  std::printf("%s standalone thinfilm sheet symmetry valid\n", label);
  return true;
}

bool validate_delta_thinfilm_coating_sample(const char* label, const etx::BSDFData& data, const etx::Material& material, const uint32_t seed, bool require_transmission) {
  etx::Material material_with_media = material;
  material_with_media.ext_medium = 7u;
  material_with_media.int_medium = 11u;
  const bool entering = dot(data.nrm, data.w_i) < 0.0f;
  const uint32_t expected_transmission_medium = entering ? material_with_media.int_medium : material_with_media.ext_medium;

  bool saw_reflection = false;
  bool saw_transmission = false;
  for (uint32_t i = 0u; i < kBsdfSamples; ++i) {
    etx::Sampler sampler(seed + i, seed ^ (i * 79u + 59u));
    const etx::BSDFSample sample = etx::bsdf::sample(data, material_with_media, sampler);
    if ((validate_sample(sample) == false) || (sample.valid() == false) || (sample.is_delta() == false)) {
      std::printf("%s invalid delta thinfilm coating sample\n", label);
      return false;
    }

    const bool reflection = (sample.properties & etx::BSDFSample::Reflection) != 0u;
    const bool transmission = (sample.properties & etx::BSDFSample::Transmission) != 0u;
    const bool medium_changed = (sample.properties & etx::BSDFSample::MediumChanged) != 0u;
    if (reflection) {
      saw_reflection = true;
      if ((transmission || medium_changed) || (sample.medium_index != data.current_medium)) {
        std::printf("%s invalid delta thinfilm reflection metadata\n", label);
        return false;
      }
    } else if (transmission) {
      saw_transmission = true;
      if (((medium_changed == false) || (sample.medium_index != expected_transmission_medium)) || (sample.eta <= 0.0f)) {
        std::printf("%s invalid delta thinfilm transmission metadata eta %.6f medium %u expected %u\n", label, sample.eta, sample.medium_index,
          expected_transmission_medium);
        return false;
      }
    } else {
      std::printf("%s delta thinfilm coating sample missing branch flags\n", label);
      return false;
    }
  }

  if ((saw_reflection == false) || (require_transmission && (saw_transmission == false))) {
    std::printf("%s delta thinfilm coating missing branch reflection %u transmission %u\n", label, saw_reflection ? 1u : 0u, saw_transmission ? 1u : 0u);
    return false;
  }

  std::printf("%s delta thinfilm coating sample valid\n", label);
  return true;
}

bool validate_delta_plastic_thinfilm_contract(const char* label, const etx::BSDFData& data, const etx::Material& material, const uint32_t seed) {
  bool saw_coating = false;
  bool saw_substrate = false;
  for (uint32_t i = 0u; i < kBsdfSamples; ++i) {
    etx::Sampler sampler(seed + i, seed ^ (i * 83u + 61u));
    const etx::BSDFSample sample = etx::bsdf::sample(data, material, sampler);
    if ((validate_sample(sample) == false) || (sample.valid() == false)) {
      std::printf("%s invalid delta plastic thinfilm sample\n", label);
      return false;
    }

    const bool diffuse = (sample.properties & etx::BSDFSample::Diffuse) != 0u;
    const bool reflection = (sample.properties & etx::BSDFSample::Reflection) != 0u;
    const bool transmission = (sample.properties & etx::BSDFSample::Transmission) != 0u;
    const bool medium_changed = (sample.properties & etx::BSDFSample::MediumChanged) != 0u;
    if (((reflection == false) || transmission) || medium_changed) {
      std::printf("%s invalid delta plastic thinfilm metadata\n", label);
      return false;
    }
    if (diffuse) {
      saw_substrate = true;
    } else {
      saw_coating = true;
    }

    etx::Sampler eval_sampler(seed + i + 10000u, seed ^ (i * 89u + 67u));
    const etx::BSDFEval eval = etx::bsdf::evaluate(data, sample.w_o, material, eval_sampler);
    etx::Sampler pdf_sampler(seed + i + 20000u, seed ^ (i * 97u + 71u));
    const float pdf = etx::bsdf::pdf(data, sample.w_o, material, pdf_sampler);
    if ((eval.valid() == false) || (std::isfinite(pdf) == false) || (pdf <= kEpsilon)) {
      std::printf("%s invalid delta plastic thinfilm eval/pdf %.6f\n", label, pdf);
      return false;
    }

    const float pdf_tolerance = max(1.0e-4f, 5.0e-3f * max(sample.pdf, pdf));
    if (fabsf(sample.pdf - pdf) > pdf_tolerance) {
      std::printf("%s delta plastic thinfilm pdf mismatch sample %.6f pdf %.6f\n", label, sample.pdf, pdf);
      return false;
    }
  }

  if ((saw_coating == false) || (saw_substrate == false)) {
    std::printf("%s delta plastic thinfilm missing branch coating %u substrate %u\n", label, saw_coating ? 1u : 0u, saw_substrate ? 1u : 0u);
    return false;
  }

  std::printf("%s delta plastic thinfilm contract valid\n", label);
  return true;
}

bool validate_exact_plastic_interface(etx::Scene& original_scene, const etx::SpectralDistribution* spectra, const uint32_t spectrum_count, const float roughness,
  const uint32_t seed) {
  etx::TaskScheduler scheduler = {};
  etx::SceneData scene_data(scheduler);
  scene_data.images.init(16u);
  scene_data.spectrum_values.assign(spectra, spectra + spectrum_count);
  scene_data.materials.emplace_back(make_plastic(roughness));
  scene_data.materials.emplace_back(make_black_substrate_plastic(roughness));
  scene_data.materials.emplace_back(make_white_dielectric(roughness));

  if (etx::ensure_energy_compensation_interfaces(scene_data, scheduler) == false) {
    std::printf("plastic exact interface roughness %.3f failed to bind LUT\n", roughness);
    return false;
  }

  scene_data.images.load_images(scheduler);

  etx::Scene exact_scene = {};
  exact_scene.spectrums = etx::ArrayView<etx::SpectralDistribution>{scene_data.spectrum_values.data(), scene_data.spectrum_values.size()};
  exact_scene.images = etx::ArrayView<etx::Image>{scene_data.images.as_array(), scene_data.images.array_size()};
  exact_scene.materials = etx::ArrayView<etx::Material>{scene_data.materials.data(), scene_data.materials.size()};
  exact_scene.energy_compensation_interfaces =
    etx::ArrayView<etx::Scene::EnergyCompensationInterface>{scene_data.energy_compensation_interfaces.data(), scene_data.energy_compensation_interfaces.size()};

  etx::scene_global_clear(&original_scene);
  etx::scene_global_publish(&exact_scene, &exact_scene);

  const etx::Material material = scene_data.materials[0];
  const etx::Material black_substrate_material = scene_data.materials[1];
  const etx::Material dielectric_material = scene_data.materials[2];
  const Vertex vertex = {
    float3{0.0f, 0.0f, 0.0f},
    float3{0.0f, 0.0f, 1.0f},
    float3{1.0f, 0.0f, 0.0f},
    float3{0.0f, 1.0f, 0.0f},
    float2{0.5f, 0.5f},
  };
  const etx::BSDFData camera_data = {etx::SpectralQuery{}, kInvalidIndex, etx::PathSource::Camera, vertex, float3{0.0f, 0.0f, -1.0f}};
  const etx::BSDFData light_data = {etx::SpectralQuery{}, kInvalidIndex, etx::PathSource::Light, vertex, float3{0.0f, 0.0f, -1.0f}};
  const etx::BSDFData inside_data = {etx::SpectralQuery{}, kInvalidIndex, etx::PathSource::Camera, vertex, float3{0.0f, 0.0f, 1.0f}};

  bool diagnostic_valid = validate_energy_compensated_material("plastic coated diffuse", camera_data, material, roughness, seed);
  diagnostic_valid =
    validate_energy_compensated_material("plastic coated diffuse light", light_data, material, roughness, seed + 500u) && diagnostic_valid;
  diagnostic_valid =
    validate_energy_compensated_white_furnace_direction("plastic coated diffuse", float3{0.0f, 0.0f, -1.0f}, material, roughness, seed + 1000u) && diagnostic_valid;
  diagnostic_valid = validate_energy_compensated_white_furnace_direction("plastic coated diffuse", normalize(float3{0.8660254f, 0.0f, -0.5f}), material, roughness,
                       seed + 1100u) &&
                     diagnostic_valid;
  diagnostic_valid = validate_energy_compensated_white_furnace_direction("plastic coated diffuse grazing", normalize(float3{0.9848077f, 0.0f, -0.1736482f}), material,
                       roughness, seed + 1150u) &&
                     diagnostic_valid;
  diagnostic_valid = validate_plastic_sample_contract("plastic coated diffuse", camera_data, material, roughness, seed + 1250u) && diagnostic_valid;
  diagnostic_valid = validate_plastic_inner_matches_outer("plastic coated diffuse", camera_data, inside_data, material, roughness, seed + 1300u) && diagnostic_valid;
  diagnostic_valid = validate_plastic_black_substrate_matches_dielectric_reflection("plastic coated diffuse", camera_data, black_substrate_material, dielectric_material,
                       roughness, seed + 1350u) &&
                     diagnostic_valid;

  const float bsdf_energy = integrate_bsdf_energy(camera_data, material, seed + 1200u);
  if ((std::isfinite(bsdf_energy) == false) || (bsdf_energy < 0.0f) || (bsdf_energy > 1.05f)) {
    std::printf("plastic coated diffuse roughness %.3f overcompensated bsdf energy %.6f\n", roughness, bsdf_energy);
    diagnostic_valid = false;
  } else {
    std::printf("plastic coated diffuse roughness %.3f bsdf energy %.6f\n", roughness, bsdf_energy);
  }

  etx::scene_global_clear(&exact_scene);
  etx::scene_global_publish(&original_scene, &original_scene);
  return diagnostic_valid;
}

bool validate_energy_compensated_dielectric_transport_contract(const char* label, const etx::BSDFData& data, const etx::Material& material, const float roughness,
  const uint32_t seed) {
  if (validate_sampling(data, material, seed) == false) {
    std::printf("%s roughness %.3f produced invalid sample\n", label, roughness);
    return false;
  }

  if (validate_sample_pdf_match(data, material, seed + 1000u) == false) {
    std::printf("%s roughness %.3f sample pdf mismatch\n", label, roughness);
    return false;
  }

  if (validate_reverse_pdf_match(data, material, seed + 2000u) == false) {
    std::printf("%s roughness %.3f reverse pdf mismatch\n", label, roughness);
    return false;
  }

  etx::Material material_with_media = material;
  material_with_media.ext_medium = 7u;
  material_with_media.int_medium = 11u;
  const bool outside = (dot(data.nrm, data.w_i) < 0.0f);
  const uint32_t expected_transmission_medium = outside ? material_with_media.int_medium : material_with_media.ext_medium;
  bool saw_reflection = false;
  bool saw_transmission = false;
  for (uint32_t i = 0u; i < kBsdfSamples; ++i) {
    etx::Sampler sampler(seed + i + 3000u, seed ^ (i * 37u + 23u));
    const etx::BSDFSample sample = etx::bsdf::sample(data, material_with_media, sampler);
    if ((validate_sample(sample) == false) || (sample.valid() == false)) {
      std::printf("%s roughness %.3f invalid media sample\n", label, roughness);
      return false;
    }

    const bool reflection = (sample.properties & etx::BSDFSample::Reflection) != 0u;
    const bool transmission = ((sample.properties & etx::BSDFSample::Transmission) != 0u);
    const bool medium_changed = ((sample.properties & etx::BSDFSample::MediumChanged) != 0u);
    if (reflection) {
      saw_reflection = true;
      if ((transmission) || medium_changed || (sample.medium_index != data.current_medium) || (fabsf(sample.eta - 1.0f) > 1.0e-4f)) {
        std::printf("%s roughness %.3f invalid reflection metadata eta %.6f medium %u\n", label, roughness, sample.eta, sample.medium_index);
        return false;
      }
    } else if (transmission) {
      saw_transmission = true;
      if ((medium_changed == false) || (sample.medium_index != expected_transmission_medium) || (sample.eta <= 0.0f)) {
        std::printf("%s roughness %.3f invalid transmission metadata eta %.6f medium %u expected %u\n", label, roughness, sample.eta, sample.medium_index,
          expected_transmission_medium);
        return false;
      }
    } else {
      std::printf("%s roughness %.3f missing reflection/transmission flag\n", label, roughness);
      return false;
    }
  }

  if ((saw_reflection == false) || (saw_transmission == false)) {
    std::printf("%s roughness %.3f missing sampled branch reflection %u transmission %u\n", label, roughness, saw_reflection ? 1u : 0u, saw_transmission ? 1u : 0u);
    return false;
  }

  std::printf("%s roughness %.3f transport contract valid\n", label, roughness);
  return true;
}

bool validate_delta_dielectric_transmission_sample(const char* label, const etx::BSDFData& data, const etx::Material& material, const uint32_t seed) {
  etx::Material material_with_media = material;
  material_with_media.ext_medium = 7u;
  material_with_media.int_medium = 11u;

  const bool outside = (dot(data.nrm, data.w_i) < 0.0f);
  const uint32_t expected_medium = outside ? material_with_media.int_medium : material_with_media.ext_medium;

  for (uint32_t i = 0u; i < kBsdfSamples; ++i) {
    etx::Sampler sampler(seed + i, seed ^ (i * 43u + 29u));
    const etx::BSDFSample sample = etx::bsdf::sample(data, material_with_media, sampler);
    if ((validate_sample(sample) == false) || (sample.valid() == false)) {
      std::printf("%s invalid delta sample\n", label);
      return false;
    }

    const bool transmission = ((sample.properties & etx::BSDFSample::Transmission) != 0u);
    if (transmission == false) {
      continue;
    }

    const bool medium_changed = ((sample.properties & etx::BSDFSample::MediumChanged) != 0u);
    if ((medium_changed == false) || (sample.medium_index != expected_medium) || (sample.eta <= 0.0f)) {
      std::printf("%s invalid transmission metadata eta %.6f medium %u expected %u\n", label, sample.eta, sample.medium_index, expected_medium);
      return false;
    }

    const float expected_weight = sample.eta * sample.eta;
    const float weight = sample.weight.monochromatic();
    const float tolerance = max(1.0e-4f, 5.0e-4f * expected_weight);
    if (fabsf(weight - expected_weight) > tolerance) {
      std::printf("%s transmission weight %.6f expected eta^2 %.6f eta %.6f\n", label, weight, expected_weight, sample.eta);
      return false;
    }

    std::printf("%s delta transmission weight %.6f eta %.6f valid\n", label, weight, sample.eta);
    return true;
  }

  std::printf("%s did not sample transmission branch\n", label);
  return false;
}

bool validate_named_plastic_water_exact_interface(etx::Scene& original_scene, const etx::SpectralDistribution* spectra, const uint32_t spectrum_count) {
  etx::TaskScheduler scheduler = {};
  etx::SceneData scene_data(scheduler);
  scene_data.images.init(16u);
  scene_data.spectrum_values.assign(spectra, spectra + spectrum_count);
  scene_data.materials.emplace_back(make_named_plastic_water_dielectric(0.25f));

  if (etx::ensure_energy_compensation_interfaces(scene_data, scheduler) == false) {
    std::printf("energy compensated named plastic-water exact interface diagnostics failed to bind LUT\n");
    return true;
  }

  scene_data.images.load_images(scheduler);

  etx::Scene exact_scene = {};
  exact_scene.spectrums = etx::ArrayView<etx::SpectralDistribution>{scene_data.spectrum_values.data(), scene_data.spectrum_values.size()};
  exact_scene.images = etx::ArrayView<etx::Image>{scene_data.images.as_array(), scene_data.images.array_size()};
  exact_scene.materials = etx::ArrayView<etx::Material>{scene_data.materials.data(), scene_data.materials.size()};
  exact_scene.energy_compensation_interfaces =
    etx::ArrayView<etx::Scene::EnergyCompensationInterface>{scene_data.energy_compensation_interfaces.data(), scene_data.energy_compensation_interfaces.size()};

  etx::scene_global_clear(&original_scene);
  etx::scene_global_publish(&exact_scene, &exact_scene);

  const etx::Material material = scene_data.materials[0];
  const float roughness = 0.25f;
  bool diagnostic_valid = true;
  const float outside_mu_values[] = {1.0f, 0.5f, 0.1736482f, 0.05f, 0.01f};
  for (uint32_t i = 0u; i < 5u; ++i) {
    const float mu = outside_mu_values[i];
    const float sin_theta = sqrtf(max(0.0f, 1.0f - mu * mu));
    char outside_label[128] = {};
    snprintf(outside_label, sizeof(outside_label), "energy compensated named plastic-water exact outside mu %.3f", mu);
    diagnostic_valid = validate_energy_compensated_white_furnace_direction(outside_label, normalize(float3{sin_theta, 0.0f, -mu}), material, roughness, 34000u + i) &&
                       diagnostic_valid;

    char inside_label[128] = {};
    snprintf(inside_label, sizeof(inside_label), "energy compensated named plastic-water exact inside mu %.3f", mu);
    diagnostic_valid = validate_energy_compensated_white_furnace_direction(inside_label, normalize(float3{sin_theta, 0.0f, mu}), material, roughness, 34100u + i) &&
                       diagnostic_valid;
  }

  const Vertex contract_vertex = {
    float3{0.0f, 0.0f, 0.0f},
    float3{0.0f, 0.0f, 1.0f},
    float3{1.0f, 0.0f, 0.0f},
    float3{0.0f, 1.0f, 0.0f},
    float2{0.5f, 0.5f},
  };
  const etx::BSDFData camera_outside_data = {etx::SpectralQuery{}, kInvalidIndex, etx::PathSource::Camera, contract_vertex, normalize(float3{0.5f, 0.0f, -0.8660254f})};
  const etx::BSDFData light_outside_data = {etx::SpectralQuery{}, kInvalidIndex, etx::PathSource::Light, contract_vertex, normalize(float3{0.5f, 0.0f, -0.8660254f})};
  const etx::BSDFData camera_inside_data = {etx::SpectralQuery{}, kInvalidIndex, etx::PathSource::Camera, contract_vertex, normalize(float3{0.5f, 0.0f, 0.8660254f})};
  const etx::BSDFData light_inside_data = {etx::SpectralQuery{}, kInvalidIndex, etx::PathSource::Light, contract_vertex, normalize(float3{0.5f, 0.0f, 0.8660254f})};
  diagnostic_valid =
    validate_energy_compensated_dielectric_transport_contract("energy compensated named plastic-water exact camera outside", camera_outside_data, material, roughness, 34200u) &&
    diagnostic_valid;
  diagnostic_valid =
    validate_energy_compensated_dielectric_transport_contract("energy compensated named plastic-water exact light outside", light_outside_data, material, roughness, 34300u) &&
    diagnostic_valid;
  diagnostic_valid =
    validate_energy_compensated_dielectric_transport_contract("energy compensated named plastic-water exact camera inside", camera_inside_data, material, roughness, 34400u) &&
    diagnostic_valid;
  diagnostic_valid =
    validate_energy_compensated_dielectric_transport_contract("energy compensated named plastic-water exact light inside", light_inside_data, material, roughness, 34500u) &&
    diagnostic_valid;

  if (diagnostic_valid == false) {
    std::printf("energy compensated named plastic-water exact interface diagnostics detected non-unit furnace energy\n");
  }

  etx::scene_global_clear(&exact_scene);
  etx::scene_global_publish(&original_scene, &original_scene);
  return true;
}

bool validate_exact_energy_compensated_dielectric_interface(etx::Scene& original_scene, const etx::SpectralDistribution* spectra, const uint32_t spectrum_count,
  const char* label, const etx::Material& source_material, const float roughness, const uint32_t seed) {
  etx::TaskScheduler scheduler = {};
  etx::SceneData scene_data(scheduler);
  scene_data.images.init(16u);
  scene_data.spectrum_values.assign(spectra, spectra + spectrum_count);
  scene_data.materials.emplace_back(source_material);

  if (etx::ensure_energy_compensation_interfaces(scene_data, scheduler) == false) {
    std::printf("%s exact dielectric interface failed to bind LUT\n", label);
    return false;
  }

  scene_data.images.load_images(scheduler);

  etx::Scene exact_scene = {};
  exact_scene.spectrums = etx::ArrayView<etx::SpectralDistribution>{scene_data.spectrum_values.data(), scene_data.spectrum_values.size()};
  exact_scene.images = etx::ArrayView<etx::Image>{scene_data.images.as_array(), scene_data.images.array_size()};
  exact_scene.materials = etx::ArrayView<etx::Material>{scene_data.materials.data(), scene_data.materials.size()};
  exact_scene.energy_compensation_interfaces =
    etx::ArrayView<etx::Scene::EnergyCompensationInterface>{scene_data.energy_compensation_interfaces.data(), scene_data.energy_compensation_interfaces.size()};

  etx::scene_global_clear(&original_scene);
  etx::scene_global_publish(&exact_scene, &exact_scene);

  const etx::Material material = scene_data.materials[0];
  bool diagnostic_valid = true;
  diagnostic_valid = validate_energy_compensated_white_furnace_direction(label, float3{0.0f, 0.0f, -1.0f}, material, roughness, seed + 100u) && diagnostic_valid;
  diagnostic_valid = validate_energy_compensated_white_furnace_direction(label, float3{0.0f, 0.0f, 1.0f}, material, roughness, seed + 200u) && diagnostic_valid;
  diagnostic_valid =
    validate_energy_compensated_white_furnace_direction(label, normalize(float3{0.8660254f, 0.0f, -0.5f}), material, roughness, seed + 300u) && diagnostic_valid;
  diagnostic_valid =
    validate_energy_compensated_white_furnace_direction(label, normalize(float3{0.8660254f, 0.0f, 0.5f}), material, roughness, seed + 400u) && diagnostic_valid;

  const Vertex contract_vertex = {
    float3{0.0f, 0.0f, 0.0f},
    float3{0.0f, 0.0f, 1.0f},
    float3{1.0f, 0.0f, 0.0f},
    float3{0.0f, 1.0f, 0.0f},
    float2{0.5f, 0.5f},
  };
  const etx::BSDFData camera_outside_data = {etx::SpectralQuery{}, kInvalidIndex, etx::PathSource::Camera, contract_vertex, normalize(float3{0.5f, 0.0f, -0.8660254f})};
  const etx::BSDFData light_outside_data = {etx::SpectralQuery{}, kInvalidIndex, etx::PathSource::Light, contract_vertex, normalize(float3{0.5f, 0.0f, -0.8660254f})};
  const etx::BSDFData camera_inside_data = {etx::SpectralQuery{}, kInvalidIndex, etx::PathSource::Camera, contract_vertex, normalize(float3{0.5f, 0.0f, 0.8660254f})};
  const etx::BSDFData light_inside_data = {etx::SpectralQuery{}, kInvalidIndex, etx::PathSource::Light, contract_vertex, normalize(float3{0.5f, 0.0f, 0.8660254f})};
  const float camera_outside_energy = integrate_bsdf_energy(camera_outside_data, material, seed + 900u);
  const float light_outside_energy = integrate_bsdf_energy(light_outside_data, material, seed + 1000u);
  const float camera_inside_energy = integrate_bsdf_energy(camera_inside_data, material, seed + 1100u);
  const float light_inside_energy = integrate_bsdf_energy(light_inside_data, material, seed + 1200u);
  std::printf("%s roughness %.3f eval energy camera outside %.6f light outside %.6f camera inside %.6f light inside %.6f\n", label, roughness, camera_outside_energy,
    light_outside_energy, camera_inside_energy, light_inside_energy);
  diagnostic_valid = validate_energy_compensated_dielectric_transport_contract(label, camera_outside_data, material, roughness, seed + 500u) && diagnostic_valid;
  diagnostic_valid = validate_energy_compensated_dielectric_transport_contract(label, light_outside_data, material, roughness, seed + 600u) && diagnostic_valid;
  diagnostic_valid = validate_energy_compensated_dielectric_transport_contract(label, camera_inside_data, material, roughness, seed + 700u) && diagnostic_valid;
  diagnostic_valid = validate_energy_compensated_dielectric_transport_contract(label, light_inside_data, material, roughness, seed + 800u) && diagnostic_valid;

  etx::scene_global_clear(&exact_scene);
  etx::scene_global_publish(&original_scene, &original_scene);
  return diagnostic_valid;
}

bool validate_exact_energy_compensated_conductor_interface(etx::Scene& original_scene, const etx::SpectralDistribution* spectra, const uint32_t spectrum_count,
  const char* label, const etx::Material& source_material, const float roughness, const uint32_t seed) {
  etx::TaskScheduler scheduler = {};
  etx::SceneData scene_data(scheduler);
  scene_data.images.init(16u);
  scene_data.spectrum_values.assign(spectra, spectra + spectrum_count);
  scene_data.materials.emplace_back(source_material);

  if (etx::ensure_energy_compensation_interfaces(scene_data, scheduler) == false) {
    std::printf("%s exact interface failed to bind LUT\n", label);
    return false;
  }

  scene_data.images.load_images(scheduler);

  etx::Scene exact_scene = {};
  exact_scene.spectrums = etx::ArrayView<etx::SpectralDistribution>{scene_data.spectrum_values.data(), scene_data.spectrum_values.size()};
  exact_scene.images = etx::ArrayView<etx::Image>{scene_data.images.as_array(), scene_data.images.array_size()};
  exact_scene.materials = etx::ArrayView<etx::Material>{scene_data.materials.data(), scene_data.materials.size()};
  exact_scene.energy_compensation_interfaces =
    etx::ArrayView<etx::Scene::EnergyCompensationInterface>{scene_data.energy_compensation_interfaces.data(), scene_data.energy_compensation_interfaces.size()};

  etx::scene_global_clear(&original_scene);
  etx::scene_global_publish(&exact_scene, &exact_scene);

  const etx::Material material = scene_data.materials[0];
  const Vertex vertex = {
    float3{0.0f, 0.0f, 0.0f},
    float3{0.0f, 0.0f, 1.0f},
    float3{1.0f, 0.0f, 0.0f},
    float3{0.0f, 1.0f, 0.0f},
    float2{0.5f, 0.5f},
  };
  const etx::BSDFData data = {etx::SpectralQuery{}, kInvalidIndex, etx::PathSource::Camera, vertex, float3{0.0f, 0.0f, -1.0f}};
  const BSDFResourceContext context = etx::bsdf::detail::make_interop_context();
  const float alpha = bsdf_energy_compensated_scalar_roughness(context, material, data.tex);
  const RefractiveIndexSample ext_ior = bsdf_resource_evaluate_refractive_index(context, material.ext_ior, data.spectrum_sample);
  const RefractiveIndexSample int_ior = bsdf_resource_evaluate_refractive_index(context, material.int_ior, data.spectrum_sample);
  const SpectralResponse e_i_response = bsdf_energy_compensated_conductor_directional_albedo(context, data.spectrum_sample, material, 1.0f, alpha);
  const SpectralResponse e_average_response = bsdf_energy_compensated_conductor_average_albedo(context, data.spectrum_sample, material, alpha);
  const float e_i_scalar = bsdf_energy_compensated_conductor_geometric_directional_albedo(context, material, 1.0f, alpha);
  const float e_average_scalar = bsdf_energy_compensated_conductor_geometric_average_albedo(context, material, alpha);
  const float visible_probability = bsdf_energy_compensated_conductor_visible_probability(context, material, 1.0f, alpha);
  const SpectralResponse f_ms = bsdf_energy_compensated_conductor_fms(data.spectrum_sample, ext_ior, int_ior, e_average_scalar);
  std::printf("%s exact interface roughness %.3f e_i %.6f e_avg %.6f geom_i %.6f geom_avg %.6f visible %.6f f_ms %.6f\n", label, roughness,
    spectral_response_monochromatic(e_i_response), spectral_response_monochromatic(e_average_response), e_i_scalar, e_average_scalar, visible_probability,
    spectral_response_monochromatic(f_ms));

  bool diagnostic_valid = validate_energy_compensated_material(label, data, material, roughness, seed);
  diagnostic_valid = validate_energy_compensated_white_furnace_direction(label, float3{0.0f, 0.0f, -1.0f}, material, roughness, seed + 100u) && diagnostic_valid;
  diagnostic_valid =
    validate_energy_compensated_white_furnace_direction(label, normalize(float3{0.8660254f, 0.0f, -0.5f}), material, roughness, seed + 200u) && diagnostic_valid;

  const float bsdf_energy = integrate_bsdf_energy(data, material, seed + 300u);
  if ((std::isfinite(bsdf_energy) == false) || (bsdf_energy < 0.0f) || (bsdf_energy > 1.05f)) {
    std::printf("%s exact interface roughness %.3f overcompensated bsdf energy %.6f\n", label, roughness, bsdf_energy);
    diagnostic_valid = false;
  } else {
    std::printf("%s exact interface roughness %.3f bsdf energy %.6f\n", label, roughness, bsdf_energy);
  }

  etx::scene_global_clear(&exact_scene);
  etx::scene_global_publish(&original_scene, &original_scene);
  return diagnostic_valid;
}

}  // namespace

int main() {
  setvbuf(stdout, nullptr, _IONBF, 0);
  etx::env().setup("bin/bsdf_validation.exe");

  etx::SpectralDistribution spectra[SpectrumCount] = {};
  spectra[SpectrumWhite] = make_spectrum(float3{1.0f, 1.0f, 1.0f});
  spectra[SpectrumBlack] = make_spectrum(float3{0.0f, 0.0f, 0.0f});
  spectra[SpectrumColored] = make_spectrum(float3{0.8f, 0.35f, 0.15f});
  spectra[SpectrumAirEta] = make_spectrum(float3{1.0f, 1.0f, 1.0f});
  spectra[SpectrumDielectricEta] = make_spectrum(float3{1.5f, 1.5f, 1.5f});
  spectra[SpectrumSapphireEta] = make_spectrum(float3{1.77f, 1.77f, 1.77f});
  std::string named_ior_title = {};
  etx::SpectralDistribution::load_refractive_index(etx::env().file_in_data("spectrum/dielectric/plastic.spd"), spectra[SpectrumNamedPlasticEta],
    spectra[SpectrumNamedPlasticK], named_ior_title);
  etx::SpectralDistribution::load_refractive_index(etx::env().file_in_data("spectrum/dielectric/water.spd"), spectra[SpectrumNamedWaterEta], spectra[SpectrumNamedWaterK],
    named_ior_title);
  spectra[SpectrumConductorEta] = make_spectrum(float3{0.25f, 0.45f, 1.05f});
  spectra[SpectrumConductorK] = make_spectrum(float3{3.4f, 2.4f, 1.9f});
  spectra[SpectrumMirrorEta] = make_loaded_ior_constant(0.0f);
  spectra[SpectrumMirrorK] = make_loaded_ior_constant(1000000.0f);

  etx::Scene scene = {};
  scene.spectrums = etx::ArrayView<etx::SpectralDistribution>{spectra, SpectrumCount};

  etx::scene_global_init();
  etx::scene_global_publish(&scene, &scene);

  const Vertex vertex = {
    float3{0.0f, 0.0f, 0.0f},
    float3{0.0f, 0.0f, 1.0f},
    float3{1.0f, 0.0f, 0.0f},
    float3{0.0f, 1.0f, 0.0f},
    float2{0.5f, 0.5f},
  };
  const etx::BSDFData data = {etx::SpectralQuery{}, kInvalidIndex, etx::PathSource::Camera, vertex, float3{0.0f, 0.0f, -1.0f}};
  etx::BSDFData inside_data = data;
  inside_data.w_i = float3{0.0f, 0.0f, 1.0f};
  bool valid = true;
  const etx::Material standalone_thinfilm = make_standalone_thinfilm(0.0f, 500.0f);
  valid = validate_standalone_thinfilm_contract("standalone thinfilm outside", data, standalone_thinfilm, 22000u) && valid;
  valid = validate_standalone_thinfilm_contract("standalone thinfilm inside", inside_data, standalone_thinfilm, 22100u) && valid;
  valid = validate_standalone_thinfilm_sheet_symmetry("standalone thinfilm boundary ior ignored", data, standalone_thinfilm, 22150u) && valid;
  valid = validate_delta_thinfilm_coating_sample("delta dielectric thinfilm outside", data, make_thinfilm_delta_dielectric(), 22200u, true) && valid;
  valid = validate_delta_thinfilm_coating_sample("delta dielectric thinfilm inside", inside_data, make_thinfilm_delta_dielectric(), 22300u, true) && valid;
  valid = validate_delta_thinfilm_coating_sample("delta conductor thinfilm outside", data, make_thinfilm_delta_conductor(), 22400u, false) && valid;
  valid = validate_delta_plastic_thinfilm_contract("delta plastic thinfilm outside", data, make_thinfilm_delta_plastic(), 22500u) && valid;
  valid = validate_delta_plastic_thinfilm_contract("delta plastic thinfilm inside", inside_data, make_thinfilm_delta_plastic(), 22600u) && valid;
  valid = validate_named_plastic_water_exact_interface(scene, spectra, SpectrumCount) && valid;
  valid = validate_plastic_low_roughness_substrate_exit(scene, spectra, SpectrumCount) && valid;

  const float diffuse_roughness_values[] = {0.0f, 0.25f, 0.5f, 0.75f, 1.0f};
  for (uint32_t i = 0u; i < 5u; ++i) {
    const float roughness = diffuse_roughness_values[i];
    valid = validate_eon_diffuse_material(data, roughness, 24000u + i * 1000u) && valid;
  }

  const float plastic_roughness_values[] = {0.25f, 0.5f, 0.75f, 1.0f};
  for (uint32_t i = 0u; i < 4u; ++i) {
    const float plastic_roughness = plastic_roughness_values[i];
    valid = validate_exact_plastic_interface(scene, spectra, SpectrumCount, plastic_roughness, 26000u + i * 1000u) && valid;
  }

  const etx::Material equal_ior_dielectric = make_white_equal_ior_dielectric(1.0f);
  valid = validate_equal_ior_dielectric_direction("dielectric outside", data, equal_ior_dielectric, 25500u) && valid;
  valid = validate_equal_ior_dielectric_direction("dielectric inside", inside_data, equal_ior_dielectric, 25510u) && valid;

  const etx::Material delta_sapphire_dielectric = make_white_sapphire_dielectric(0.0f);
  valid = (validate_delta_dielectric_transmission_sample("sapphire delta dielectric outside", data, delta_sapphire_dielectric, 25520u) && valid);
  valid = (validate_delta_dielectric_transmission_sample("sapphire delta dielectric inside", inside_data, delta_sapphire_dielectric, 25530u) && valid);

  const float exact_conductor_roughness = 0.5f;
  valid = validate_exact_energy_compensated_conductor_interface(scene, spectra, SpectrumCount, "mirror conductor exact interface",
    make_mirror_conductor(exact_conductor_roughness), exact_conductor_roughness, 33400u) &&
          valid;

  const float exact_dielectric_roughness_values[] = {0.25f, 0.5f, 0.75f, 1.0f};
  for (uint32_t i = 0u; i < 4u; ++i) {
    const float rough_sapphire_roughness = exact_dielectric_roughness_values[i];
    valid = validate_exact_energy_compensated_dielectric_interface(scene, spectra, SpectrumCount, "sapphire dielectric exact interface",
              make_white_sapphire_dielectric(rough_sapphire_roughness), rough_sapphire_roughness, 34600u + i * 1000u) &&
            valid;
  }

  etx::scene_global_clear(&scene);
  etx::scene_global_deinit();

  return valid ? 0 : 1;
}
