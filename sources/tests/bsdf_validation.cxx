#include <etx/core/environment.hxx>
#include <etx/render/host/bsdf_energy_compensation_lut.hxx>
#include <etx/render/host/scene_global.hxx>
#include <etx/render/host/scene_data.hxx>
#include <etx/render/shared/scene.hxx>
#include <etx/render/shared/scene_bsdf.hxx>

#include <tinyexr.hxx>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <vector>

namespace etx {
namespace DielectricBSDF {
namespace detail {

constexpr bool kEnableRoughDielectricConnectibleReference = true;

struct Resources {
  float2 roughness = {};
  RefractiveIndexSample ext_ior = {};
  RefractiveIndexSample int_ior = {};
  ThinFilmEval thinfilm = {};
};

LocalFrame material_frame(const BSDFData& data) {
  return LocalFrame{data.tan, data.btn, data.nrm, LocalFrame::EnteringMaterial};
}

Resources load_resources(const BSDFData& data, const Material& material, Sampler& sampler) {
  Resources result = {};
  result.roughness = evaluate_roughness(material, data.tex);
  result.ext_ior = evaluate_refractive_index(material.ext_ior, data.spectrum_sample);
  result.int_ior = evaluate_refractive_index(material.int_ior, data.spectrum_sample);
  result.thinfilm = evaluate_thinfilm(data.spectrum_sample, material.thinfilm, data.tex, sampler);
  return result;
}

BSDFEval reference_eval_zero(const SpectralQuery& spectrum_sample) {
  BSDFEval result = {};
  result.func = SpectralResponse{spectrum_sample, 0.0f};
  result.bsdf = SpectralResponse{spectrum_sample, 0.0f};
  result.pdf = 0.0f;
  result.eta = 1.0f;
  return result;
}

float reference_pdf_with_resources(const BSDFData& data, const float3& w_o, const Resources& resources) {
  const LocalFrame frame = material_frame(data);
  const float3 w_i = local_frame_to_local(frame, -data.w_i);
  if (fabsf(w_i.z) <= kEpsilon) {
    return 0.0f;
  }

  const float3 local_w_o = local_frame_to_local(frame, w_o);
  if (fabsf(local_w_o.z) <= kEpsilon) {
    return 0.0f;
  }

  const bool outside = w_i.z > 0.0f;
  const bool reflection = (w_i.z * local_w_o.z) > 0.0f;

  float3 wh = {};
  float dwh_dwo = 0.0f;
  if (reflection) {
    wh = normalize(w_i + local_w_o);
    const float denominator = 4.0f * dot(local_w_o, wh);
    if (fabsf(denominator) <= kEpsilon) {
      return 0.0f;
    }
    dwh_dwo = 1.0f / denominator;
  } else {
    const float eta = outside ? (spectral_response_monochromatic(resources.int_ior.eta) / spectral_response_monochromatic(resources.ext_ior.eta))
                              : (spectral_response_monochromatic(resources.ext_ior.eta) / spectral_response_monochromatic(resources.int_ior.eta));
    wh = normalize(w_i + local_w_o * eta);
    const float sqrt_denom = dot(w_i, wh) + eta * dot(local_w_o, wh);
    if (fabsf(sqrt_denom) <= kEpsilon) {
      return 0.0f;
    }
    dwh_dwo = (eta * eta) * dot(local_w_o, wh) / (sqrt_denom * sqrt_denom);
  }

  wh *= (wh.z >= 0.0f) ? 1.0f : -1.0f;

  const float3 oriented_w_i = outside ? w_i : -w_i;
  const external::RayInfo ray = {oriented_w_i, resources.roughness};
  const float denominator = (1.0f + ray.Lambda) * ray.w.z;
  if (fabsf(denominator) <= kEpsilon) {
    return 0.0f;
  }

  const float d_ggx = external::D_ggx(wh, resources.roughness);
  float proposal = max(0.0f, dot(wh, ray.w) * d_ggx / denominator);
  const RefractiveIndexSample fresnel_ext_ior = outside ? resources.ext_ior : resources.int_ior;
  const RefractiveIndexSample fresnel_int_ior = outside ? resources.int_ior : resources.ext_ior;
  const float f = fresnel::calculate(data.spectrum_sample, dot(w_i, wh), fresnel_ext_ior, fresnel_int_ior, resources.thinfilm).monochromatic();

  proposal *= reflection ? f : (1.0f - f);
  return fabsf(proposal * dwh_dwo);
}

BSDFEval evaluate_reference_with_resources(const BSDFData& data, const float3& w_o, const Material& material, const Resources& resources, Sampler& sampler) {
  const LocalFrame frame = material_frame(data);
  const float3 w_i = local_frame_to_local(frame, -data.w_i);
  const float3 local_w_o = local_frame_to_local(frame, w_o);
  if ((fabsf(w_i.z) <= kEpsilon) || (fabsf(local_w_o.z) <= kEpsilon)) {
    return reference_eval_zero(data.spectrum_sample);
  }

  const bool forward_path = data.path_source == PathSource::Camera;
  const float backward_scale = fabsf(1.0f / w_i.z);

  SpectralResponse value = {data.spectrum_sample, 0.0f};
  if (w_i.z > 0.0f) {
    if (local_w_o.z >= 0.0f) {
      value = forward_path ? external::eval_dielectric(data.spectrum_sample, sampler, w_i, local_w_o, true, resources.roughness, resources.ext_ior, resources.int_ior, resources.thinfilm)
                           : external::eval_dielectric(data.spectrum_sample, sampler, local_w_o, w_i, true, resources.roughness, resources.ext_ior, resources.int_ior, resources.thinfilm) *
                               backward_scale;
    } else {
      value = forward_path ? external::eval_dielectric(data.spectrum_sample, sampler, w_i, local_w_o, false, resources.roughness, resources.ext_ior, resources.int_ior, resources.thinfilm)
                           : external::eval_dielectric(data.spectrum_sample, sampler, -local_w_o, -w_i, false, resources.roughness, resources.int_ior, resources.ext_ior, resources.thinfilm) *
                               backward_scale;
    }
  } else if (local_w_o.z <= 0.0f) {
    value = forward_path ? external::eval_dielectric(data.spectrum_sample, sampler, -w_i, -local_w_o, true, resources.roughness, resources.int_ior, resources.ext_ior, resources.thinfilm)
                         : external::eval_dielectric(data.spectrum_sample, sampler, -local_w_o, -w_i, true, resources.roughness, resources.int_ior, resources.ext_ior, resources.thinfilm) *
                             backward_scale;
  } else {
    value = forward_path ? external::eval_dielectric(data.spectrum_sample, sampler, -w_i, -local_w_o, false, resources.roughness, resources.int_ior, resources.ext_ior, resources.thinfilm)
                         : external::eval_dielectric(data.spectrum_sample, sampler, local_w_o, w_i, false, resources.roughness, resources.ext_ior, resources.int_ior, resources.thinfilm) *
                             backward_scale;
  }

  const bool reflection = (w_i.z * local_w_o.z) > 0.0f;
  const SpectralImage scattering_image = reflection ? material.reflectance : material.scattering;

  BSDFEval result = {};
  result.bsdf = value * apply_image(data.spectrum_sample, scattering_image, data.tex);
  result.func = result.bsdf / fabsf(local_w_o.z);
  result.pdf = reference_pdf_with_resources(data, w_o, resources);
  result.eta = 1.0f;
  return result;
}

}  // namespace detail
}  // namespace DielectricBSDF
}  // namespace etx

namespace {

constexpr uint32_t kIntegrationSamples = 8192u;
constexpr uint32_t kBsdfSamples = 2048u;
constexpr float kDielectricPdfIntegralTolerance = 0.35f;
constexpr float kWhiteDielectricSampleFurnaceTolerance = 0.12f;
constexpr float kWhiteDielectricEnergyTolerance = 0.12f;
constexpr float kRoughDielectricReferenceEnergyTolerance = 0.15f;
constexpr uint32_t kValidationConductorLutSize = 32u;
constexpr uint32_t kValidationDielectricLutSize = 64u;

enum SpectrumSlot : uint32_t {
  SpectrumWhite,
  SpectrumColored,
  SpectrumAirEta,
  SpectrumDielectricEta,
  SpectrumSapphireEta,
  SpectrumPlasticEta,
  SpectrumNamedPlasticEta,
  SpectrumNamedPlasticK,
  SpectrumWaterEta,
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

float integrate_rough_dielectric_reference_pdf(const etx::BSDFData& data, const etx::Material& material, const uint32_t seed) {
  etx::Sampler resource_sampler(seed, seed ^ 0xa24baed5u);
  const etx::DielectricBSDF::detail::Resources resources = etx::DielectricBSDF::detail::load_resources(data, material, resource_sampler);

  etx::Sampler sampler(seed + 1u, seed ^ 0x63d8a1cbu);
  float sum = 0.0f;

  for (uint32_t i = 0u; i < kIntegrationSamples; ++i) {
    const float3 w_o = sample_uniform_sphere(sampler);
    const float pdf = etx::DielectricBSDF::detail::reference_pdf_with_resources(data, w_o, resources);
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

float integrate_rough_dielectric_reference_energy(const etx::BSDFData& data, const etx::Material& material, const uint32_t seed) {
  etx::Sampler resource_sampler(seed, seed ^ 0x6c8e9cf5u);
  const etx::DielectricBSDF::detail::Resources resources = etx::DielectricBSDF::detail::load_resources(data, material, resource_sampler);

  etx::Sampler sampler(seed + 1u, seed ^ 0x9d735a2bu);
  float sum = 0.0f;

  for (uint32_t i = 0u; i < kIntegrationSamples; ++i) {
    const float3 w_o = sample_uniform_sphere(sampler);
    etx::Sampler eval_sampler(seed + i + 100u, seed ^ (i * 13u + 11u));
    const etx::BSDFEval eval = etx::DielectricBSDF::detail::evaluate_reference_with_resources(data, w_o, material, resources, eval_sampler);
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

float2 integrate_rough_dielectric_reference_lobe_energy(const etx::BSDFData& data, const etx::Material& material, const uint32_t seed) {
  etx::Sampler resource_sampler(seed, seed ^ 0x5a134f29u);
  const etx::DielectricBSDF::detail::Resources resources = etx::DielectricBSDF::detail::load_resources(data, material, resource_sampler);

  etx::Sampler sampler(seed + 1u, seed ^ 0xa934d735u);
  float reflection_sum = 0.0f;
  float transmission_sum = 0.0f;
  const LocalFrame frame = etx::DielectricBSDF::detail::material_frame(data);
  const float3 w_i = local_frame_to_local(frame, -data.w_i);

  for (uint32_t i = 0u; i < kIntegrationSamples; ++i) {
    const float3 w_o_world = sample_uniform_sphere(sampler);
    const float3 w_o = local_frame_to_local(frame, w_o_world);
    etx::Sampler eval_sampler(seed + i + 100u, seed ^ (i * 13u + 11u));
    const etx::BSDFEval eval = etx::DielectricBSDF::detail::evaluate_reference_with_resources(data, w_o_world, material, resources, eval_sampler);
    if ((finite_response(eval.bsdf) == false) || (non_negative_response(eval.bsdf) == false)) {
      return float2{-1.0f, -1.0f};
    }

    if ((w_i.z * w_o.z) > 0.0f) {
      reflection_sum += eval.bsdf.monochromatic();
    } else {
      transmission_sum += eval.bsdf.monochromatic();
    }
  }

  const float scale = 4.0f * kPi / static_cast<float>(kIntegrationSamples);
  return float2{reflection_sum * scale, transmission_sum * scale};
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

bool validate_rough_dielectric_sample_contract(const etx::BSDFData& data, const etx::Material& material, const uint32_t seed) {
  for (uint32_t i = 0u; i < kBsdfSamples; ++i) {
    etx::Sampler sampler(seed + i, seed ^ (i * 19u + 5u));
    const etx::BSDFSample sample = etx::bsdf::sample(data, material, sampler);
    if (validate_sample(sample) == false) {
      return false;
    }

    if (sample.valid() == false) {
      continue;
    }

    if constexpr (etx::DielectricBSDF::detail::kEnableRoughDielectricConnectibleReference == false) {
      if ((sample.is_delta() == false) || (fabsf(sample.pdf - 1.0f) > kEpsilon)) {
        return false;
      }
      continue;
    }

    const bool reflection = (sample.properties & etx::BSDFSample::Reflection) != 0u;
    const bool transmission = (sample.properties & etx::BSDFSample::Transmission) != 0u;
    const bool medium_changed = (sample.properties & etx::BSDFSample::MediumChanged) != 0u;

    if (sample.is_delta()) {
      return false;
    }

    if (((reflection) && (transmission)) || ((reflection == false) && (transmission == false))) {
      return false;
    }

    if ((transmission) && (medium_changed == false)) {
      return false;
    }

    if ((reflection) && (medium_changed)) {
      return false;
    }
  }

  return true;
}

float average_sample_weight(const etx::BSDFData& data, const etx::Material& material, const uint32_t seed) {
  float sum = 0.0f;
  uint32_t valid_sample_count = 0u;

  for (uint32_t i = 0u; i < kBsdfSamples; ++i) {
    etx::Sampler sampler(seed + i, seed ^ (i * 31u + 7u));
    const etx::BSDFSample sample = etx::bsdf::sample(data, material, sampler);
    if (validate_sample(sample) == false) {
      return -1.0f;
    }

    sum += sample.weight.monochromatic();
    valid_sample_count += 1u;
  }

  if (valid_sample_count == 0u) {
    return -1.0f;
  }

  return sum / static_cast<float>(valid_sample_count);
}

float3 average_sample_weight_rgb(const etx::BSDFData& data, const etx::Material& material, const uint32_t seed) {
  float3 sum = {};
  uint32_t valid_sample_count = 0u;

  for (uint32_t i = 0u; i < kBsdfSamples; ++i) {
    etx::Sampler sampler(seed + i, seed ^ (i * 37u + 19u));
    const etx::BSDFSample sample = etx::bsdf::sample(data, material, sampler);
    if (validate_sample(sample) == false) {
      return float3{-1.0f, -1.0f, -1.0f};
    }

    sum += sample.weight.to_rgb();
    valid_sample_count += 1u;
  }

  if (valid_sample_count == 0u) {
    return float3{-1.0f, -1.0f, -1.0f};
  }

  return sum * (1.0f / static_cast<float>(valid_sample_count));
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

struct EnergyCompensatedDielectricDecompositionStats {
  SampleWeightStats weight_stats = {};
  float connected_p95_weight = 0.0f;
  float connected_p99_weight = 0.0f;
  float connected_max_weight = 0.0f;
  float connected_max_pdf = 0.0f;
  float connected_max_bsdf = 0.0f;
  float3 connected_max_w_o = {};
  float max_weight = 0.0f;
  float max_base_weight = 0.0f;
  float max_compensation_weight = 0.0f;
  float max_pdf = 0.0f;
  float max_base_pdf = 0.0f;
  float max_compensation_pdf = 0.0f;
  float max_base_probability = 0.0f;
  float max_visible_probability = 0.0f;
  float max_e_i = 0.0f;
  float max_e_o = 0.0f;
  float max_e_average = 0.0f;
  float3 max_w_o = {};
  uint32_t reflection_count = 0u;
  uint32_t transmission_count = 0u;
  uint32_t base_pdf_dominant_count = 0u;
  uint32_t compensation_pdf_dominant_count = 0u;
  uint32_t connected_valid_count = 0u;
  uint32_t connected_zero_pdf_count = 0u;
};

EnergyCompensatedDielectricDecompositionStats energy_compensated_dielectric_decomposition_stats(const etx::BSDFData& data, const etx::Material& material,
  const uint32_t seed) {
  EnergyCompensatedDielectricDecompositionStats result = {};
  result.weight_stats = sample_weight_stats_rgb(data, material, seed);

  const BSDFResourceContext context = etx::bsdf::detail::make_interop_context();
  const LocalFrame frame = data.get_normal_frame(material);
  const float3 w_i = local_frame_to_local(frame, -data.w_i);
  if (abs(w_i.z) <= kEpsilon) {
    return result;
  }

  const float alpha = bsdf_energy_compensated_scalar_roughness(context, material, data.tex);
  const RefractiveIndexSample ext_ior = bsdf_resource_evaluate_refractive_index(context, material.ext_ior, data.spectrum_sample);
  const RefractiveIndexSample int_ior = bsdf_resource_evaluate_refractive_index(context, material.int_ior, data.spectrum_sample);
  const float f0 = bsdf_energy_compensated_dielectric_f0(ext_ior, int_ior);
  const bool outside_i = w_i.z > 0.0f;
  const bool i_low_to_high = bsdf_energy_compensated_dielectric_low_to_high(ext_ior, int_ior, outside_i);
  const SpectralResponse e_i_response =
    bsdf_energy_compensated_dielectric_directional_albedo(context, data.spectrum_sample, material, abs(w_i.z), alpha, f0, outside_i, i_low_to_high);
  const float e_i = spectral_response_monochromatic(e_i_response);
  const float visible_probability = bsdf_energy_compensated_dielectric_visible_probability(context, material, abs(w_i.z), alpha, f0, outside_i, i_low_to_high);
  const float base_probability = (visible_probability > kEpsilon) ? 1.0f : 0.0f;

  float connected_weights[kBsdfSamples] = {};
  etx::Sampler connected_sampler(seed + 60000u, seed ^ 0x4c2f19adu);
  for (uint32_t i = 0u; i < kBsdfSamples; ++i) {
    const float3 world_w_o = sample_uniform_sphere(connected_sampler);
    etx::Sampler eval_sampler(seed + i + 70000u, seed ^ (i * 53u + 37u));
    const etx::BSDFEval eval = etx::bsdf::evaluate(data, world_w_o, material, eval_sampler);
    if ((eval.valid() == false) || (finite_response(eval.bsdf) == false) || (eval.pdf <= kEpsilon)) {
      if ((eval.valid()) && (eval.pdf <= kEpsilon)) {
        result.connected_zero_pdf_count += 1u;
      }
      connected_weights[i] = 0.0f;
      continue;
    }

    const float weight = eval.bsdf.maximum() / eval.pdf;
    connected_weights[result.connected_valid_count] = weight;
    result.connected_valid_count += 1u;
    if (weight > result.connected_max_weight) {
      result.connected_max_weight = weight;
      result.connected_max_pdf = eval.pdf;
      result.connected_max_bsdf = eval.bsdf.maximum();
      result.connected_max_w_o = local_frame_to_local(frame, world_w_o);
    }
  }
  std::sort(connected_weights, connected_weights + kBsdfSamples);
  const uint32_t connected_p95_index = min(kBsdfSamples - 1u, static_cast<uint32_t>(0.95f * static_cast<float>(kBsdfSamples - 1u)));
  const uint32_t connected_p99_index = min(kBsdfSamples - 1u, static_cast<uint32_t>(0.99f * static_cast<float>(kBsdfSamples - 1u)));
  result.connected_p95_weight = connected_weights[connected_p95_index];
  result.connected_p99_weight = connected_weights[connected_p99_index];

  for (uint32_t i = 0u; i < kBsdfSamples; ++i) {
    etx::Sampler sampler(seed + i, seed ^ (i * 47u + 31u));
    const etx::BSDFSample sample = etx::bsdf::sample(data, material, sampler);
    if ((validate_sample(sample) == false) || (sample.valid() == false)) {
      continue;
    }

    const float3 w_o = local_frame_to_local(frame, sample.w_o);
    if (abs(w_o.z) <= kEpsilon) {
      continue;
    }

    const bool reflection = (w_i.z * w_o.z) > 0.0f;
    if (reflection) {
      result.reflection_count += 1u;
    } else {
      result.transmission_count += 1u;
    }

    const SpectralImage texture_image = reflection ? material.reflectance : material.scattering;
    const SpectralResponse texture = bsdf_resource_apply_image(context, data.spectrum_sample, texture_image, data.tex);
    const BSDFEnergyCompensatedLobe base_lobe = bsdf_energy_compensated_dielectric_base_lobe(data.spectrum_sample, w_i, w_o, alpha, ext_ior, int_ior, texture);

    const float normalized_base_pdf = base_lobe.pdf / max(kEpsilon, visible_probability);
    const float base_pdf = normalized_base_pdf;
    const float compensation_pdf = 0.0f;
    if (base_pdf >= compensation_pdf) {
      result.base_pdf_dominant_count += 1u;
    } else {
      result.compensation_pdf_dominant_count += 1u;
    }

    const float pdf = base_pdf + compensation_pdf;
    const float base_weight = spectral_response_monochromatic(base_lobe.bsdf) / max(kEpsilon, e_i * pdf);
    const float compensation_weight = 0.0f;
    const float weight = sample.weight.maximum();
    if (weight > result.max_weight) {
      result.max_weight = weight;
      result.max_base_weight = base_weight;
      result.max_compensation_weight = compensation_weight;
      result.max_pdf = pdf;
      result.max_base_pdf = base_pdf;
      result.max_compensation_pdf = compensation_pdf;
      result.max_base_probability = base_probability;
      result.max_visible_probability = visible_probability;
      result.max_e_i = e_i;
      result.max_e_o = 0.0f;
      result.max_e_average = 0.0f;
      result.max_w_o = w_o;
    }
  }

  return result;
}

struct DielectricSampleReasonStats {
  uint32_t valid_count = 0u;
  uint32_t grazing_input_count = 0u;
  uint32_t max_order_count = 0u;
  uint32_t invalid_state_count = 0u;
  uint32_t invalid_direction_count = 0u;
  uint32_t invalid_weight_count = 0u;
  uint32_t nan_count = 0u;
  uint32_t grazing_output_count = 0u;
};

DielectricSampleReasonStats dielectric_sample_reason_stats(const etx::BSDFData& data, const etx::Material& material, const uint32_t seed) {
  DielectricSampleReasonStats result = {};

  for (uint32_t i = 0u; i < kBsdfSamples; ++i) {
    etx::Sampler sampler(seed + i, seed ^ (i * 43u + 29u));
    const etx::DielectricBSDF::detail::Resources resources = etx::DielectricBSDF::detail::load_resources(data, material, sampler);
    const LocalFrame local_frame = etx::DielectricBSDF::detail::material_frame(data);
    const float3 w_i = local_frame_to_local(local_frame, -data.w_i);
    if (fabsf(w_i.z) <= kEpsilon) {
      result.grazing_input_count += 1u;
      continue;
    }

    const bool in_outside = w_i.z > 0.0f;
    const float direction_scale = in_outside ? 1.0f : -1.0f;
    const float3 oriented_w_i = direction_scale * w_i;
    const etx::RefractiveIndexSample outer_ior = in_outside ? resources.ext_ior : resources.int_ior;
    const etx::RefractiveIndexSample inner_ior = in_outside ? resources.int_ior : resources.ext_ior;

    etx::BSDFSample sample = {};
    sample.weight = etx::SpectralResponse{data.spectrum_sample, 1.0f};

    etx::external::RayInfo ray = {-oriented_w_i, resources.roughness};
    ray.updateHeight(1.0f);
    bool ray_outside = true;
    bool invalid_state = false;
    bool invalid_direction = false;
    bool invalid_weight = false;
    bool nan_state = false;
    bool max_order = false;

    uint32_t scattering_order = 0u;
    while (true) {
      const float sampled_height = etx::external::sampleHeight(ray, sampler.next());
      if (sampled_height == kMaxFloat) {
        break;
      }

      ray.updateHeight(sampled_height);
      const float2 rnd_slope = ((scattering_order == 0u) && sampler.has_fixed()) ? float2{sampler.fixed_u, sampler.fixed_v} : sampler.next_2d();
      const float rnd_reflection = ((scattering_order == 0u) && sampler.has_fixed()) ? sampler.fixed_w : sampler.next();
      const etx::RefractiveIndexSample phase_ext_ior = ray_outside ? outer_ior : inner_ior;
      const etx::RefractiveIndexSample phase_int_ior = ray_outside ? inner_ior : outer_ior;

      const etx::external::DielectricSample phase_sample =
        etx::external::samplePhaseFunction_dielectric(data.spectrum_sample, rnd_slope, rnd_reflection, -ray.w, resources.roughness, phase_ext_ior, phase_int_ior, resources.thinfilm);

      const float branch_probability = max(kEpsilon, phase_sample.weight.monochromatic());
      sample.weight *= phase_sample.weight / branch_probability;

      if (phase_sample.reflection) {
        ray.updateDirection(phase_sample.w_o, resources.roughness);
        ray.updateHeight(ray.h);
      } else {
        ray_outside = (ray_outside == false);
        ray.updateDirection(-phase_sample.w_o, resources.roughness);
        ray.updateHeight(-ray.h);
      }

      scattering_order += 1u;
      if (scattering_order > etx::external::kScatteringOrderMax) {
        max_order = true;
        break;
      }

      if ((ray.h != ray.h) || (ray.w.x != ray.w.x) || (ray.w.z != ray.w.z)) {
        nan_state = true;
        invalid_state = true;
        break;
      }
      if (sample.weight.valid() == false) {
        invalid_weight = true;
        invalid_state = true;
        break;
      }
    }

    if (max_order) {
      result.max_order_count += 1u;
      continue;
    }
    if (invalid_state) {
      result.invalid_state_count += 1u;
      if (invalid_direction) {
        result.invalid_direction_count += 1u;
      }
      if (invalid_weight) {
        result.invalid_weight_count += 1u;
      }
      if (nan_state) {
        result.nan_count += 1u;
      }
      continue;
    }

    const float3 local_w_o = direction_scale * (ray_outside ? ray.w : -ray.w);
    if (fabsf(local_w_o.z) <= kEpsilon) {
      result.grazing_output_count += 1u;
      continue;
    }

    result.valid_count += 1u;
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

float2 integrate_bsdf_lobe_energy(const etx::BSDFData& data, const etx::Material& material, const uint32_t seed) {
  etx::Sampler sampler(seed, seed ^ 0x814a2c9du);
  float reflection_sum = 0.0f;
  float transmission_sum = 0.0f;
  const LocalFrame frame = data.get_normal_frame(material);
  const float3 w_i = local_frame_to_local(frame, -data.w_i);

  for (uint32_t i = 0u; i < kIntegrationSamples; ++i) {
    const float3 w_o_world = sample_uniform_sphere(sampler);
    const float3 w_o = local_frame_to_local(frame, w_o_world);
    etx::Sampler eval_sampler(seed + i, seed ^ (i * 13u + 11u));
    const etx::BSDFEval eval = etx::bsdf::evaluate(data, w_o_world, material, eval_sampler);
    if ((finite_response(eval.bsdf) == false) || (non_negative_response(eval.bsdf) == false)) {
      return float2{-1.0f, -1.0f};
    }

    if ((w_i.z * w_o.z) > 0.0f) {
      reflection_sum += eval.bsdf.monochromatic();
    } else {
      transmission_sum += eval.bsdf.monochromatic();
    }
  }

  const float scale = 4.0f * kPi / static_cast<float>(kIntegrationSamples);
  return float2{reflection_sum * scale, transmission_sum * scale};
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

etx::Material make_energy_compensated_conductor(const float roughness) {
  etx::Material result = make_conductor(roughness);
  result.cls = MaterialClass::ConductorEnergyCompensated;
  return result;
}

etx::Material make_energy_compensated_mirror_conductor(const float roughness) {
  etx::Material result = make_mirror_conductor(roughness);
  result.cls = MaterialClass::ConductorEnergyCompensated;
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

etx::Material make_energy_compensated_dielectric(const float roughness) {
  etx::Material result = make_dielectric(roughness);
  result.cls = MaterialClass::DielectricEnergyCompensated;
  return result;
}

etx::Material make_white_energy_compensated_dielectric(const float roughness) {
  etx::Material result = make_white_dielectric(roughness);
  result.cls = MaterialClass::DielectricEnergyCompensated;
  return result;
}

etx::Material make_white_equal_ior_energy_compensated_dielectric(const float roughness) {
  etx::Material result = make_white_equal_ior_dielectric(roughness);
  result.cls = MaterialClass::DielectricEnergyCompensated;
  return result;
}

etx::Material make_white_reversed_energy_compensated_dielectric(const float roughness) {
  etx::Material result = make_white_energy_compensated_dielectric(roughness);
  result.ext_ior.eta_index = SpectrumDielectricEta;
  result.int_ior.eta_index = SpectrumAirEta;
  return result;
}

etx::Material make_white_plastic_water_energy_compensated_dielectric(const float roughness) {
  etx::Material result = make_white_energy_compensated_dielectric(roughness);
  result.ext_ior.eta_index = SpectrumPlasticEta;
  result.int_ior.eta_index = SpectrumWaterEta;
  return result;
}

etx::Material make_named_plastic_water_energy_compensated_dielectric(const float roughness) {
  etx::Material result = make_white_energy_compensated_dielectric(roughness);
  result.ext_ior.eta_index = SpectrumNamedPlasticEta;
  result.ext_ior.k_index = SpectrumNamedPlasticK;
  result.int_ior.eta_index = SpectrumNamedWaterEta;
  result.int_ior.k_index = SpectrumNamedWaterK;
  return result;
}

float expected_pdf_mass(const char* label, const float roughness) {
  if ((std::strcmp(label, "conductor") == 0) || (std::strcmp(label, "mirror conductor") == 0)) {
    const float cosine_mix = min(0.5f, sqrtf(max(0.0f, roughness * roughness)));
    const float microfacet_mix = 1.0f - cosine_mix;
    return cosine_mix + microfacet_mix / (1.0f + roughness * roughness);
  }

  return 1.0f;
}

bool is_dielectric_label(const char* label) {
  return ((std::strcmp(label, "dielectric") == 0) || (std::strcmp(label, "white dielectric") == 0));
}

bool validate_material(const char* label, const etx::BSDFData& data, const etx::Material& material, const float roughness, const uint32_t seed) {
  const bool dielectric = is_dielectric_label(label);
  const float pdf_integral = integrate_pdf(data, material, seed);
  if ((std::isfinite(pdf_integral) == false) || (pdf_integral < 0.0f)) {
    std::printf("%s roughness %.3f pdf integral %.6f\n", label, roughness, pdf_integral);
    return false;
  }

  if ((pdf_integral <= 0.0f) && (dielectric == false)) {
    std::printf("%s roughness %.3f pdf integral %.6f\n", label, roughness, pdf_integral);
    return false;
  }

  if ((dielectric) && (roughness >= 0.1f)) {
    if constexpr (etx::DielectricBSDF::detail::kEnableRoughDielectricConnectibleReference) {
      if (fabsf(pdf_integral - 1.0f) > kDielectricPdfIntegralTolerance) {
        std::printf("%s roughness %.3f pdf integral %.6f expected %.6f\n", label, roughness, pdf_integral, 1.0f);
        return false;
      }
    } else {
      if (pdf_integral > kEpsilon) {
        std::printf("%s roughness %.3f pdf integral %.6f expected %.6f\n", label, roughness, pdf_integral, 0.0f);
        return false;
      }
    }
  }

  if ((roughness >= 0.1f) && (dielectric == false)) {
    const float expected_mass = expected_pdf_mass(label, roughness);
    if (fabsf(pdf_integral - expected_mass) > 0.12f) {
      std::printf("%s roughness %.3f pdf integral %.6f expected %.6f\n", label, roughness, pdf_integral, expected_mass);
      return false;
    }
  }

  const float reverse_pdf_integral = integrate_reverse_pdf(data, material, seed + 50000u);
  if ((std::isfinite(reverse_pdf_integral) == false) || (reverse_pdf_integral < 0.0f)) {
    std::printf("%s roughness %.3f reverse pdf integral %.6f\n", label, roughness, reverse_pdf_integral);
    return false;
  }

  if (validate_sampling(data, material, seed + 10000u) == false) {
    std::printf("%s roughness %.3f produced invalid sample\n", label, roughness);
    return false;
  }

  if (((dielectric == false) || etx::DielectricBSDF::detail::kEnableRoughDielectricConnectibleReference) &&
      (validate_sample_pdf_match(data, material, seed + 15000u) == false)) {
    std::printf("%s roughness %.3f sample pdf mismatch\n", label, roughness);
    return false;
  }

  if ((dielectric) && (validate_rough_dielectric_sample_contract(data, material, seed + 16000u) == false)) {
    std::printf("%s roughness %.3f broke rough dielectric sample contract\n", label, roughness);
    return false;
  }

  const float average_weight = average_sample_weight(data, material, seed + 20000u);
  if ((std::isfinite(average_weight) == false) || (average_weight <= 0.02f)) {
    std::printf("%s roughness %.3f average sample weight %.6f\n", label, roughness, average_weight);
    return false;
  }

  const float bsdf_energy = integrate_bsdf_energy(data, material, seed + 30000u);
  if ((std::isfinite(bsdf_energy) == false) || (bsdf_energy < 0.0f)) {
    std::printf("%s roughness %.3f bsdf energy %.6f\n", label, roughness, bsdf_energy);
    return false;
  }

  std::printf("%s roughness %.3f pdf integral %.6f reverse %.6f average weight %.6f bsdf energy %.6f\n", label, roughness, pdf_integral, reverse_pdf_integral, average_weight,
    bsdf_energy);
  return true;
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
  const Vertex vertex = {
    float3{0.0f, 0.0f, 0.0f},
    float3{0.0f, 0.0f, 1.0f},
    float3{1.0f, 0.0f, 0.0f},
    float3{0.0f, 1.0f, 0.0f},
    float2{0.5f, 0.5f},
  };
  const etx::BSDFData data = {etx::SpectralQuery{}, kInvalidIndex, etx::PathSource::Camera, vertex, w_i_world};
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

bool validate_energy_compensated_dielectric_decomposition(const char* label, const float3& w_i_world, const etx::Material& material, const float roughness, const uint32_t seed) {
  const Vertex vertex = {
    float3{0.0f, 0.0f, 0.0f},
    float3{0.0f, 0.0f, 1.0f},
    float3{1.0f, 0.0f, 0.0f},
    float3{0.0f, 1.0f, 0.0f},
    float2{0.5f, 0.5f},
  };
  const etx::BSDFData data = {etx::SpectralQuery{}, kInvalidIndex, etx::PathSource::Camera, vertex, w_i_world};
  const EnergyCompensatedDielectricDecompositionStats stats = energy_compensated_dielectric_decomposition_stats(data, material, seed);
  const SampleWeightStats& weight_stats = stats.weight_stats;
  if ((weight_stats.invalid_count > 0u) || (weight_stats.valid_count == 0u) || (std::isfinite(weight_stats.average.x) == false) ||
      (std::isfinite(stats.max_weight) == false)) {
    std::printf("%s decomposition wi %.3f %.3f %.3f roughness %.3f invalid %u valid %u max %.6f\n", label, w_i_world.x, w_i_world.y, w_i_world.z, roughness,
      weight_stats.invalid_count, weight_stats.valid_count, stats.max_weight);
    return false;
  }

  std::printf(
    "%s decomposition wi %.3f %.3f %.3f roughness %.3f avg %.6f p95 %.6f p99 %.6f max %.6f connected_p95 %.6f connected_p99 %.6f connected_max %.6f connected_pdf %.6f connected_bsdf %.6f connected_wo %.3f %.3f %.3f base_w %.6f comp_w %.6f pdf %.6f base_pdf %.6f comp_pdf %.6f base_prob %.6f visible %.6f e_i %.6f e_o %.6f e_avg %.6f wo %.3f %.3f %.3f refl %u trans %u base_pdf_dom %u comp_pdf_dom %u connected_valid %u connected_zero_pdf %u\n",
    label, w_i_world.x, w_i_world.y, w_i_world.z, roughness, weight_stats.average.x, weight_stats.p95_weight, weight_stats.p99_weight, stats.max_weight,
    stats.connected_p95_weight, stats.connected_p99_weight, stats.connected_max_weight, stats.connected_max_pdf, stats.connected_max_bsdf, stats.connected_max_w_o.x,
    stats.connected_max_w_o.y, stats.connected_max_w_o.z, stats.max_base_weight, stats.max_compensation_weight, stats.max_pdf, stats.max_base_pdf,
    stats.max_compensation_pdf, stats.max_base_probability, stats.max_visible_probability, stats.max_e_i, stats.max_e_o, stats.max_e_average, stats.max_w_o.x,
    stats.max_w_o.y, stats.max_w_o.z, stats.reflection_count, stats.transmission_count, stats.base_pdf_dominant_count, stats.compensation_pdf_dominant_count,
    stats.connected_valid_count, stats.connected_zero_pdf_count);
  return true;
}

bool validate_mirror_furnace_direction(const float3& w_i_world, const float roughness, const uint32_t seed) {
  const Vertex vertex = {
    float3{0.0f, 0.0f, 0.0f},
    float3{0.0f, 0.0f, 1.0f},
    float3{1.0f, 0.0f, 0.0f},
    float3{0.0f, 1.0f, 0.0f},
    float2{0.5f, 0.5f},
  };
  const etx::BSDFData data = {etx::SpectralQuery{}, kInvalidIndex, etx::PathSource::Camera, vertex, w_i_world};
  const etx::Material material = make_mirror_conductor(roughness);
  const float average_weight = average_sample_weight(data, material, seed);
  const float3 average_rgb = average_sample_weight_rgb(data, material, seed + 1000u);
  const float max_rgb_error = max(fabsf(average_rgb.x - 1.0f), max(fabsf(average_rgb.y - 1.0f), fabsf(average_rgb.z - 1.0f)));

  if ((std::isfinite(average_weight) == false) || (std::isfinite(average_rgb.x) == false) || (std::isfinite(average_rgb.y) == false) ||
      (std::isfinite(average_rgb.z) == false) || (fabsf(average_weight - 1.0f) > 0.03f) || (max_rgb_error > 0.02f)) {
    std::printf("mirror furnace wi %.3f %.3f %.3f roughness %.3f average weight %.6f rgb %.6f %.6f %.6f\n", w_i_world.x, w_i_world.y, w_i_world.z, roughness,
      average_weight, average_rgb.x, average_rgb.y, average_rgb.z);
    return false;
  }

  std::printf("mirror furnace wi %.3f %.3f %.3f roughness %.3f average weight %.6f rgb %.6f %.6f %.6f\n", w_i_world.x, w_i_world.y, w_i_world.z, roughness,
    average_weight, average_rgb.x, average_rgb.y, average_rgb.z);
  return true;
}

bool validate_white_dielectric_direction(const char* label, const float3& w_i_world, const etx::Material& material, const float roughness, const uint32_t seed) {
  const Vertex vertex = {
    float3{0.0f, 0.0f, 0.0f},
    float3{0.0f, 0.0f, 1.0f},
    float3{1.0f, 0.0f, 0.0f},
    float3{0.0f, 1.0f, 0.0f},
    float2{0.5f, 0.5f},
  };
  const etx::BSDFData data = {etx::SpectralQuery{}, kInvalidIndex, etx::PathSource::Camera, vertex, w_i_world};
  const SampleWeightStats stats = sample_weight_stats_rgb(data, material, seed);
  const float3 average_rgb = stats.average;
  const float max_rgb_error = max(fabsf(average_rgb.x - 1.0f), max(fabsf(average_rgb.y - 1.0f), fabsf(average_rgb.z - 1.0f)));

  if ((std::isfinite(average_rgb.x) == false) || (std::isfinite(average_rgb.y) == false) || (std::isfinite(average_rgb.z) == false) ||
      (max_rgb_error > kWhiteDielectricSampleFurnaceTolerance)) {
    std::printf(
      "%s white dielectric wi %.3f %.3f %.3f roughness %.3f rgb %.6f %.6f %.6f valid %u nonzero %u inv_pdf %.6f max_weight %.6f\n", label, w_i_world.x, w_i_world.y,
      w_i_world.z, roughness, average_rgb.x, average_rgb.y, average_rgb.z, stats.valid_count, stats.nonzero_count, stats.average_inverse_pdf, stats.max_weight);
    return false;
  }

  std::printf("%s white dielectric wi %.3f %.3f %.3f roughness %.3f rgb %.6f %.6f %.6f valid %u nonzero %u inv_pdf %.6f max_weight %.6f\n", label, w_i_world.x,
    w_i_world.y, w_i_world.z, roughness, average_rgb.x, average_rgb.y, average_rgb.z, stats.valid_count, stats.nonzero_count, stats.average_inverse_pdf, stats.max_weight);
  return true;
}

bool validate_white_dielectric_direction(const char* label, const float3& w_i_world, const float roughness, const uint32_t seed) {
  const etx::Material material = make_white_dielectric(roughness);
  return validate_white_dielectric_direction(label, w_i_world, material, roughness, seed);
}

bool validate_white_dielectric_eval_direction(const char* label, const float3& w_i_world, const etx::Material& material, const float roughness, const uint32_t seed) {
  const Vertex vertex = {
    float3{0.0f, 0.0f, 0.0f},
    float3{0.0f, 0.0f, 1.0f},
    float3{1.0f, 0.0f, 0.0f},
    float3{0.0f, 1.0f, 0.0f},
    float2{0.5f, 0.5f},
  };
  const etx::BSDFData data = {etx::SpectralQuery{}, kInvalidIndex, etx::PathSource::Camera, vertex, w_i_world};
  const float bsdf_energy = integrate_bsdf_energy(data, material, seed);
  const float2 lobe_energy = integrate_bsdf_lobe_energy(data, material, seed + 1000u);

  if constexpr (etx::DielectricBSDF::detail::kEnableRoughDielectricConnectibleReference == false) {
    if ((std::isfinite(bsdf_energy) == false) || (fabsf(bsdf_energy) > kEpsilon)) {
      std::printf("%s white dielectric eval wi %.3f %.3f %.3f roughness %.3f bsdf energy %.6f reflection %.6f transmission %.6f\n", label, w_i_world.x,
        w_i_world.y, w_i_world.z, roughness, bsdf_energy, lobe_energy.x, lobe_energy.y);
      return false;
    }

    std::printf("%s white dielectric eval wi %.3f %.3f %.3f roughness %.3f bsdf energy %.6f reflection %.6f transmission %.6f\n", label, w_i_world.x, w_i_world.y,
      w_i_world.z, roughness, bsdf_energy, lobe_energy.x, lobe_energy.y);
    return true;
  }

  if ((std::isfinite(bsdf_energy) == false) || (fabsf(bsdf_energy - 1.0f) > kWhiteDielectricEnergyTolerance)) {
    std::printf("%s white dielectric eval wi %.3f %.3f %.3f roughness %.3f bsdf energy %.6f reflection %.6f transmission %.6f\n", label, w_i_world.x, w_i_world.y, w_i_world.z,
      roughness, bsdf_energy, lobe_energy.x, lobe_energy.y);
    return false;
  }

  std::printf("%s white dielectric eval wi %.3f %.3f %.3f roughness %.3f bsdf energy %.6f reflection %.6f transmission %.6f\n", label, w_i_world.x, w_i_world.y, w_i_world.z,
    roughness, bsdf_energy, lobe_energy.x, lobe_energy.y);
  return true;
}

bool validate_white_dielectric_eval_direction(const char* label, const float3& w_i_world, const float roughness, const uint32_t seed) {
  const etx::Material material = make_white_dielectric(roughness);
  return validate_white_dielectric_eval_direction(label, w_i_world, material, roughness, seed);
}

bool validate_rough_dielectric_reference_proposal(const char* label, const float3& w_i_world, const float roughness, const uint32_t seed) {
  if constexpr (etx::DielectricBSDF::detail::kEnableRoughDielectricConnectibleReference == false) {
    std::printf("%s rough dielectric reference proposal wi %.3f %.3f %.3f roughness %.3f skipped\n", label, w_i_world.x, w_i_world.y, w_i_world.z, roughness);
    return true;
  }

  const Vertex vertex = {
    float3{0.0f, 0.0f, 0.0f},
    float3{0.0f, 0.0f, 1.0f},
    float3{1.0f, 0.0f, 0.0f},
    float3{0.0f, 1.0f, 0.0f},
    float2{0.5f, 0.5f},
  };
  const etx::BSDFData data = {etx::SpectralQuery{}, kInvalidIndex, etx::PathSource::Camera, vertex, w_i_world};
  const etx::Material material = make_white_dielectric(roughness);
  const float pdf_integral = integrate_rough_dielectric_reference_pdf(data, material, seed);
  if ((std::isfinite(pdf_integral) == false) || (fabsf(pdf_integral - 1.0f) > 0.05f)) {
    std::printf("%s rough dielectric reference proposal wi %.3f %.3f %.3f roughness %.3f pdf integral %.6f\n", label, w_i_world.x, w_i_world.y, w_i_world.z, roughness,
      pdf_integral);
    return false;
  }

  std::printf("%s rough dielectric reference proposal wi %.3f %.3f %.3f roughness %.3f pdf integral %.6f\n", label, w_i_world.x, w_i_world.y, w_i_world.z, roughness,
    pdf_integral);
  return true;
}

bool validate_rough_dielectric_reference_evaluator(const char* label, const float3& w_i_world, const float roughness, const uint32_t seed) {
  if constexpr (etx::DielectricBSDF::detail::kEnableRoughDielectricConnectibleReference == false) {
    std::printf("%s rough dielectric reference evaluator wi %.3f %.3f %.3f roughness %.3f skipped\n", label, w_i_world.x, w_i_world.y, w_i_world.z, roughness);
    return true;
  }

  const Vertex vertex = {
    float3{0.0f, 0.0f, 0.0f},
    float3{0.0f, 0.0f, 1.0f},
    float3{1.0f, 0.0f, 0.0f},
    float3{0.0f, 1.0f, 0.0f},
    float2{0.5f, 0.5f},
  };
  const etx::BSDFData data = {etx::SpectralQuery{}, kInvalidIndex, etx::PathSource::Camera, vertex, w_i_world};
  const etx::Material material = make_white_dielectric(roughness);
  const float energy = integrate_rough_dielectric_reference_energy(data, material, seed);
  const float2 lobe_energy = integrate_rough_dielectric_reference_lobe_energy(data, material, seed + 1000u);
  std::printf("%s rough dielectric reference evaluator wi %.3f %.3f %.3f roughness %.3f energy %.6f reflection %.6f transmission %.6f\n", label, w_i_world.x, w_i_world.y,
    w_i_world.z, roughness, energy, lobe_energy.x, lobe_energy.y);
  return (std::isfinite(energy)) && (fabsf(energy - 1.0f) <= kRoughDielectricReferenceEnergyTolerance);
}

bool load_validation_lut_image(const char* relative_path, const uint2& expected_size, std::vector<float4>& pixels, etx::Image& image) {
  char path[2048] = {};
  etx::env().file_in_data(relative_path, path, sizeof(path));

  int width = 0;
  int height = 0;
  float* rgba_data = nullptr;
  const char* error = nullptr;
  if (LoadEXR(&rgba_data, &width, &height, path, &error) != TINYEXR_SUCCESS) {
    std::printf("failed to load LUT EXR %s: %s\n", path, (error != nullptr) ? error : "unknown error");
    if (error != nullptr) {
      FreeEXRErrorMessage(error);
    }
    return false;
  }

  if ((static_cast<uint32_t>(width) != expected_size.x) || (static_cast<uint32_t>(height) != expected_size.y)) {
    std::printf("LUT EXR %s has size %dx%d, expected %ux%u\n", path, width, height, expected_size.x, expected_size.y);
    free(rgba_data);
    return false;
  }

  const uint64_t pixel_count = static_cast<uint64_t>(expected_size.x) * static_cast<uint64_t>(expected_size.y);
  pixels.resize(pixel_count);
  memcpy(pixels.data(), rgba_data, pixel_count * sizeof(float4));
  free(rgba_data);

  for (float4& pixel : pixels) {
    if ((std::isfinite(pixel.x) == false) || (pixel.x < 0.0f)) {
      pixel.x = 0.0f;
    }
    if ((std::isfinite(pixel.y) == false) || (pixel.y < 0.0f)) {
      pixel.y = 0.0f;
    }
    if ((std::isfinite(pixel.z) == false) || (pixel.z < 0.0f)) {
      pixel.z = 0.0f;
    }
    if ((std::isfinite(pixel.w) == false) || (pixel.w < 0.0f)) {
      pixel.w = 0.0f;
    }
  }

  image = {};
  image.fsize = float2{static_cast<float>(expected_size.x), static_cast<float>(expected_size.y)};
  image.isize = expected_size;
  image.options = etx::Image::SkipSRGBConversion;
  image.format = etx::Image::Format::RGBA32F;
  image.data_size = static_cast<uint32_t>(pixel_count * sizeof(float4));
  image.pixels.f32 = etx::ArrayView<float4>{pixels.data(), pixels.size()};
  return true;
}

bool validate_named_plastic_water_exact_interface(etx::Scene& original_scene, const etx::SpectralDistribution* spectra, const uint32_t spectrum_count, const Vertex& vertex) {
  etx::TaskScheduler scheduler = {};
  etx::SceneData scene_data(scheduler);
  scene_data.images.init(16u);
  scene_data.spectrum_values.assign(spectra, spectra + spectrum_count);
  scene_data.materials.emplace_back(make_named_plastic_water_energy_compensated_dielectric(0.005f));

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
  const float roughness = 0.005f;
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

  if (diagnostic_valid == false) {
    std::printf("energy compensated named plastic-water exact interface diagnostics detected non-unit furnace energy\n");
  }

  etx::scene_global_clear(&exact_scene);
  etx::scene_global_publish(&original_scene, &original_scene);
  return true;
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
  spectra[SpectrumColored] = make_spectrum(float3{0.8f, 0.35f, 0.15f});
  spectra[SpectrumAirEta] = make_spectrum(float3{1.0f, 1.0f, 1.0f});
  spectra[SpectrumDielectricEta] = make_spectrum(float3{1.5f, 1.5f, 1.5f});
  spectra[SpectrumSapphireEta] = make_spectrum(float3{1.77f, 1.77f, 1.77f});
  spectra[SpectrumPlasticEta] = make_spectrum(float3{1.52f, 1.52f, 1.52f});
  spectra[SpectrumWaterEta] = make_spectrum(float3{1.333f, 1.333f, 1.333f});
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
  validate_named_plastic_water_exact_interface(scene, spectra, SpectrumCount, vertex);
  const float roughness_values[] = {0.25f, 0.5f, 0.75f, 1.0f};

  bool legacy_valid = true;
  bool valid = true;
  for (uint32_t i = 0u; i < 4u; ++i) {
    const float roughness = roughness_values[i];
    const etx::Material conductor = make_conductor(roughness);
    const etx::Material mirror_conductor = make_mirror_conductor(roughness);
    const etx::Material dielectric = make_dielectric(roughness);
    const etx::Material white_dielectric = make_white_dielectric(roughness);
    legacy_valid = validate_material("conductor", data, conductor, roughness, 1000u + i) && legacy_valid;
    legacy_valid = validate_material("mirror conductor", data, mirror_conductor, roughness, 3000u + i) && legacy_valid;
    legacy_valid = validate_material("dielectric", data, dielectric, roughness, 2000u + i) && legacy_valid;
    legacy_valid = validate_material("white dielectric", data, white_dielectric, roughness, 5000u + i) && legacy_valid;
  }

  const etx::Material equal_ior_dielectric = make_white_equal_ior_dielectric(1.0f);
  valid = validate_equal_ior_dielectric_direction("dielectric outside", data, equal_ior_dielectric, 25500u) && valid;
  valid = validate_equal_ior_dielectric_direction("dielectric inside", inside_data, equal_ior_dielectric, 25510u) && valid;

  const float exact_conductor_roughness = 0.5f;
  valid = validate_exact_energy_compensated_conductor_interface(scene, spectra, SpectrumCount, "energy compensated mirror conductor exact interface",
    make_energy_compensated_mirror_conductor(exact_conductor_roughness), exact_conductor_roughness, 33400u) &&
          valid;

  const float mirror_furnace_roughness = 0.75f;
  legacy_valid = validate_mirror_furnace_direction(float3{0.0f, 0.0f, -1.0f}, mirror_furnace_roughness, 4000u) && legacy_valid;
  legacy_valid = validate_mirror_furnace_direction(normalize(float3{0.5f, 0.0f, -0.8660254f}), mirror_furnace_roughness, 5000u) && legacy_valid;
  legacy_valid = validate_mirror_furnace_direction(normalize(float3{0.8660254f, 0.0f, -0.5f}), mirror_furnace_roughness, 6000u) && legacy_valid;
  legacy_valid = validate_mirror_furnace_direction(normalize(float3{0.9848077f, 0.0f, -0.1736482f}), mirror_furnace_roughness, 7000u) && legacy_valid;

  legacy_valid = validate_white_dielectric_direction("outside", float3{0.0f, 0.0f, -1.0f}, mirror_furnace_roughness, 8000u) && legacy_valid;
  legacy_valid = validate_white_dielectric_direction("inside", float3{0.0f, 0.0f, 1.0f}, mirror_furnace_roughness, 9000u) && legacy_valid;
  legacy_valid = validate_white_dielectric_direction("outside oblique", normalize(float3{0.8660254f, 0.0f, -0.5f}), 1.0f, 10000u) && legacy_valid;
  legacy_valid = validate_white_dielectric_direction("inside oblique", normalize(float3{0.8660254f, 0.0f, 0.5f}), 1.0f, 11000u) && legacy_valid;
  legacy_valid = validate_white_dielectric_eval_direction("outside", float3{0.0f, 0.0f, -1.0f}, 1.0f, 12000u) && legacy_valid;
  legacy_valid = validate_white_dielectric_eval_direction("inside", float3{0.0f, 0.0f, 1.0f}, 1.0f, 13000u) && legacy_valid;
  legacy_valid =
    validate_white_dielectric_eval_direction("outside oblique", normalize(float3{0.8660254f, 0.0f, -0.5f}), 1.0f, 14000u) && legacy_valid;
  legacy_valid = validate_white_dielectric_eval_direction("inside oblique", normalize(float3{0.8660254f, 0.0f, 0.5f}), 1.0f, 15000u) && legacy_valid;

  const etx::Material white_sapphire_dielectric = make_white_sapphire_dielectric(1.0f);
  legacy_valid =
    validate_white_dielectric_eval_direction("sapphire outside", float3{0.0f, 0.0f, -1.0f}, white_sapphire_dielectric, 1.0f, 16000u) && legacy_valid;
  legacy_valid = validate_white_dielectric_eval_direction("sapphire inside", float3{0.0f, 0.0f, 1.0f}, white_sapphire_dielectric, 1.0f, 17000u) && legacy_valid;
  legacy_valid =
    validate_white_dielectric_eval_direction("sapphire outside oblique", normalize(float3{0.8660254f, 0.0f, -0.5f}), white_sapphire_dielectric, 1.0f, 18000u) &&
    legacy_valid;
  legacy_valid =
    validate_white_dielectric_eval_direction("sapphire inside oblique", normalize(float3{0.8660254f, 0.0f, 0.5f}), white_sapphire_dielectric, 1.0f, 19000u) &&
    legacy_valid;

  legacy_valid = validate_rough_dielectric_reference_proposal("outside", float3{0.0f, 0.0f, -1.0f}, 1.0f, 20000u) && legacy_valid;
  legacy_valid = validate_rough_dielectric_reference_proposal("inside", float3{0.0f, 0.0f, 1.0f}, 1.0f, 21000u) && legacy_valid;
  legacy_valid = validate_rough_dielectric_reference_evaluator("outside", float3{0.0f, 0.0f, -1.0f}, 1.0f, 22000u) && legacy_valid;
  legacy_valid = validate_rough_dielectric_reference_evaluator("inside", float3{0.0f, 0.0f, 1.0f}, 1.0f, 23000u) && legacy_valid;
  legacy_valid = validate_rough_dielectric_reference_evaluator("outside oblique", normalize(float3{0.8660254f, 0.0f, -0.5f}), 1.0f, 24000u) && legacy_valid;
  legacy_valid = validate_rough_dielectric_reference_evaluator("inside oblique", normalize(float3{0.8660254f, 0.0f, 0.5f}), 1.0f, 25000u) && legacy_valid;

  if (legacy_valid == false) {
    std::printf("legacy baseline diagnostics reported existing rough BSDF contract mismatches\n");
  }

  etx::scene_global_clear(&scene);
  etx::scene_global_deinit();

  return valid ? 0 : 1;
}
