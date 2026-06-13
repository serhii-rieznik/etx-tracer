#pragma once

#include "bsdf_energy_compensated_shared.hxx"
#include "bsdf_plastic_shared.hxx"

struct OpenPBRComponents {
  Material conductor ETX_INIT({});
  Material dielectric ETX_INIT({});
  Material plastic ETX_INIT({});
  float metalness ETX_INIT(0.0f);
  float transmission ETX_INIT(0.0f);
  float conductor_weight ETX_INIT(0.0f);
  float dielectric_weight ETX_INIT(0.0f);
  float plastic_weight ETX_INIT(1.0f);
};

struct OpenPBRSampleResult {
  BSDFSample sample ETX_INIT({});
  float component_weight ETX_INIT(0.0f);
};

ETX_SHARED_INLINE void bsdf_openpbr_setup_conductor_material(ETX_IN(BSDFResourceContext, context), ETX_INOUT(Material, material)) {
  material.cls = MaterialClass::Conductor;
  material.int_ior.cls = SpectralDistribution::Conductor;
  material.int_ior.eta_index = bsdf_resource_default_conductor_eta(context);
  material.int_ior.k_index = bsdf_resource_default_conductor_k(context);
  material.reflectance = material.scattering;
  material.scattering.image_index = kInvalidIndex;
  material.energy_compensation_interface_index = material.conductor_energy_compensation_interface_index;
}

ETX_SHARED_INLINE void bsdf_openpbr_setup_dielectric_material(ETX_IN(BSDFResourceContext, context), ETX_INOUT(Material, material)) {
  material.cls = MaterialClass::Dielectric;
  material.int_ior.cls = SpectralDistribution::Dielectric;
  if (material.int_ior.eta_index == kInvalidIndex) {
    material.int_ior.eta_index = bsdf_resource_default_dielectric_eta(context);
  }
  material.int_ior.k_index = kInvalidIndex;
  material.reflectance.image_index = kInvalidIndex;
  const uint32_t white_spectrum = bsdf_resource_default_white_spectrum(context);
  if (white_spectrum != kInvalidIndex) {
    material.scattering.spectrum_index = white_spectrum;
    material.scattering.image_index = kInvalidIndex;
  } else {
    material.scattering = material.reflectance;
  }
}

ETX_SHARED_INLINE void bsdf_openpbr_setup_plastic_material(ETX_IN(BSDFResourceContext, context), ETX_INOUT(Material, material)) {
  material.cls = MaterialClass::Plastic;
  material.int_ior.cls = SpectralDistribution::Dielectric;
  if (material.int_ior.eta_index == kInvalidIndex) {
    material.int_ior.eta_index = bsdf_resource_default_dielectric_eta(context);
  }
  material.int_ior.k_index = kInvalidIndex;
}

ETX_SHARED_INLINE OpenPBRComponents bsdf_openpbr_make_components(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(Material, material)) {
  OpenPBRComponents result = ETX_ZERO(OpenPBRComponents);
  result.conductor = material;
  result.dielectric = material;
  result.plastic = material;
  bsdf_openpbr_setup_conductor_material(context, result.conductor);
  bsdf_openpbr_setup_dielectric_material(context, result.dielectric);
  bsdf_openpbr_setup_plastic_material(context, result.plastic);

  result.metalness = saturate(bsdf_resource_evaluate_metalness(context, material, data.tex));
  result.transmission = saturate(bsdf_resource_evaluate_transmission(context, material, data.tex));
  result.conductor_weight = result.metalness;
  result.dielectric_weight = (1.0f - result.metalness) * result.transmission;
  result.plastic_weight = (1.0f - result.metalness) * (1.0f - result.transmission);
  return result;
}

ETX_SHARED_NOINLINE BSDFEval bsdf_openpbr_mix_eval(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(float3, outgoing_direction),
  ETX_IN(OpenPBRComponents, components), ETX_INOUT(Sampler, sampler)) {
  BSDFEval result = ETX_ZERO(BSDFEval);
  result.func = spectral_response_zero(data.spectrum_sample);
  result.bsdf = spectral_response_zero(data.spectrum_sample);
  result.eta = 1.0f;
  result.medium_index = kInvalidIndex;

  if (components.conductor_weight > 0.0f) {
    const BSDFEval conductor = bsdf_conductor_energy_compensated_evaluate(context, data, outgoing_direction, components.conductor, sampler);
    result.func = spectral_response_add(result.func, spectral_response_mul(conductor.func, components.conductor_weight));
    result.bsdf = spectral_response_add(result.bsdf, spectral_response_mul(conductor.bsdf, components.conductor_weight));
    result.pdf += conductor.pdf * components.conductor_weight;
    result.properties |= conductor.properties;
    result.medium_index = conductor.medium_index;
  }

  if (components.dielectric_weight > 0.0f) {
    const BSDFEval dielectric = bsdf_dielectric_energy_compensated_evaluate(context, data, outgoing_direction, components.dielectric, sampler);
    result.func = spectral_response_add(result.func, spectral_response_mul(dielectric.func, components.dielectric_weight));
    result.bsdf = spectral_response_add(result.bsdf, spectral_response_mul(dielectric.bsdf, components.dielectric_weight));
    result.pdf += dielectric.pdf * components.dielectric_weight;
    result.properties |= dielectric.properties;
    result.medium_index = dielectric.medium_index;
    result.eta = dielectric.eta;
  }

  if (components.plastic_weight > 0.0f) {
    const BSDFEval plastic = bsdf_plastic_evaluate(context, data, outgoing_direction, components.plastic, sampler);
    result.func = spectral_response_add(result.func, spectral_response_mul(plastic.func, components.plastic_weight));
    result.bsdf = spectral_response_add(result.bsdf, spectral_response_mul(plastic.bsdf, components.plastic_weight));
    result.pdf += plastic.pdf * components.plastic_weight;
    result.properties |= plastic.properties;
    if (result.medium_index == kInvalidIndex) {
      result.medium_index = plastic.medium_index;
    }
  }

  return result;
}

ETX_SHARED_NOINLINE float bsdf_openpbr_mix_pdf(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(float3, outgoing_direction),
  ETX_IN(OpenPBRComponents, components), ETX_INOUT(Sampler, sampler)) {
  float result = 0.0f;
  if (components.conductor_weight > 0.0f) {
    result += bsdf_conductor_energy_compensated_pdf(context, data, outgoing_direction, components.conductor, sampler) * components.conductor_weight;
  }
  if (components.dielectric_weight > 0.0f) {
    result += bsdf_dielectric_energy_compensated_pdf(context, data, outgoing_direction, components.dielectric, sampler) * components.dielectric_weight;
  }
  if (components.plastic_weight > 0.0f) {
    result += bsdf_plastic_pdf(context, data, outgoing_direction, components.plastic, sampler) * components.plastic_weight;
  }
  return result;
}

ETX_SHARED_NOINLINE OpenPBRSampleResult bsdf_openpbr_sample_component(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(OpenPBRComponents, components),
  ETX_INOUT(Sampler, sampler)) {
  OpenPBRSampleResult result = ETX_ZERO(OpenPBRSampleResult);
  const float selector = bsdf_sampler_next(sampler);
  if (selector < components.conductor_weight) {
    result.sample = bsdf_conductor_energy_compensated_sample(context, data, components.conductor, sampler);
    result.component_weight = components.conductor_weight;
    return result;
  }

  if (selector < (components.conductor_weight + components.dielectric_weight)) {
    result.sample = bsdf_dielectric_energy_compensated_sample(context, data, components.dielectric, sampler);
    result.component_weight = components.dielectric_weight;
    return result;
  }

  result.sample = bsdf_plastic_sample(context, data, components.plastic, sampler);
  result.component_weight = components.plastic_weight;
  return result;
}

//
// TODO(OpenPBR GPU parity): DO NOT CLAIM OPENPBR GPU PARITY YET.
//
// This generic mixed-component sample path is still the active OpenPBR blocker.
// The runtime GPU BSDF harness can validate OpenPBR evaluate/pdf/reverse-pdf/
// is-delta/albedo, but compiling OpenPBR sample currently overflows SPIR-V IDs
// after real plastic sampling is enabled. Production wavefront shaders therefore
// compile OpenPBR only when an OpenPBR material exists in the scene, and OpenPBR
// scenes are expected to fail until this path is split into smaller shader pieces
// or otherwise made small enough for DXC/SPIR-V legalization.
//
ETX_SHARED_NOINLINE BSDFSample bsdf_openpbr_sample(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(Material, material), ETX_INOUT(Sampler, sampler)) {
  const OpenPBRComponents components = bsdf_openpbr_make_components(context, data, material);
  const OpenPBRSampleResult component_result = bsdf_openpbr_sample_component(context, data, components, sampler);
  BSDFSample result = component_result.sample;
  if (bsdf_sample_valid(result) == false) {
    return result;
  }

  if (bsdf_sample_is_delta(result)) {
    result.pdf *= component_result.component_weight;
    return result;
  }

  const BSDFEval eval = bsdf_openpbr_mix_eval(context, data, result.w_o, components, sampler);
  result.weight = spectral_response_zero(data.spectrum_sample);
  if (eval.pdf > 0.0f) {
    result.weight = spectral_response_div(eval.bsdf, eval.pdf);
  }
  result.pdf = eval.pdf;
  result.eta = eval.eta;
  result.properties = eval.properties;
  result.medium_index = eval.medium_index;
  return result;
}

ETX_SHARED_NOINLINE BSDFEval bsdf_openpbr_evaluate(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(float3, outgoing_direction),
  ETX_IN(Material, material), ETX_INOUT(Sampler, sampler)) {
  const OpenPBRComponents components = bsdf_openpbr_make_components(context, data, material);
  return bsdf_openpbr_mix_eval(context, data, outgoing_direction, components, sampler);
}

ETX_SHARED_NOINLINE float bsdf_openpbr_pdf(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(float3, outgoing_direction), ETX_IN(Material, material),
  ETX_INOUT(Sampler, sampler)) {
  const OpenPBRComponents components = bsdf_openpbr_make_components(context, data, material);
  return bsdf_openpbr_mix_pdf(context, data, outgoing_direction, components, sampler);
}

ETX_SHARED_INLINE bool bsdf_openpbr_is_delta(ETX_IN(Material, material), ETX_IN(float2, tex), ETX_INOUT(Sampler, sampler)) {
  (void)material;
  (void)tex;
  (void)sampler;
  return false;
}

ETX_SHARED_INLINE SpectralResponse bsdf_openpbr_albedo(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(Material, material), ETX_INOUT(Sampler, sampler)) {
  (void)sampler;
  return bsdf_resource_apply_image(context, data.spectrum_sample, material.scattering, data.tex);
}
