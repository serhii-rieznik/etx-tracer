#pragma once

#if (ETX_CPP)

#include "bsdf_conductor_shared.hxx"
#include "bsdf_dielectric_shared.hxx"

ETX_STATIC_CONST uint32_t kBSDFEnergyCompensationConductorLutSize = 64u;
ETX_STATIC_CONST uint32_t kBSDFEnergyCompensationDielectricLutSize = 64u;
ETX_STATIC_CONST float kBSDFEnergyCompensationF0Max = 9.99000013e-1f;

struct BSDFEnergyCompensatedLobe {
  SpectralResponse bsdf ETX_INIT({});
  float pdf ETX_INIT(0.0f);
};

struct BSDFEnergyCompensatedDielectricComponents {
  SpectralResponse base ETX_INIT({});
  SpectralResponse compensation ETX_INIT({});
  SpectralResponse incident_albedo ETX_INIT({});
  float incident_visible_probability ETX_INIT(0.0f);
  float base_pdf ETX_INIT(0.0f);
};

struct BSDFEnergyCompensatedDielectricBranchPair {
  float4 outside_value ETX_INIT({});
  float4 inside_value ETX_INIT({});
};

ETX_SHARED_INLINE float bsdf_energy_compensated_saturate(float value) {
  return min(1.0f, max(0.0f, value));
}

ETX_SHARED_INLINE float bsdf_energy_compensated_lerp(float a, float b, float t) {
  return a * (1.0f - t) + b * t;
}

ETX_SHARED_INLINE bool bsdf_energy_compensated_spectral_response_finite(ETX_IN(SpectralResponse, value)) {
  return isfinite(value.integrated.x) && isfinite(value.integrated.y) && isfinite(value.integrated.z) && isfinite(value.value);
}

ETX_SHARED_INLINE float bsdf_energy_compensated_lut_uv(float value, uint32_t size) {
  const float scale = static_cast<float>(size - 1u) / static_cast<float>(size);
  return bsdf_energy_compensated_saturate(value) * scale;
}

ETX_SHARED_INLINE float bsdf_energy_compensated_dielectric_f0_axis(float f0) {
  const float normalized_f0 = f0 / kBSDFEnergyCompensationF0Max;
  return sqrt(bsdf_energy_compensated_saturate(normalized_f0));
}

ETX_SHARED_INLINE float bsdf_energy_compensated_alpha_axis(float alpha) {
  const float clamped_alpha = max(kBSDFNormalDistributionMinAlpha, bsdf_energy_compensated_saturate(alpha));
  const float normalized_alpha = (clamped_alpha - kBSDFNormalDistributionMinAlpha) / (1.0f - kBSDFNormalDistributionMinAlpha);
  return sqrt(bsdf_energy_compensated_saturate(normalized_alpha));
}

ETX_SHARED_INLINE float bsdf_energy_compensated_dielectric_alpha_axis(float alpha) {
  return bsdf_energy_compensated_alpha_axis(alpha);
}

ETX_SHARED_INLINE bool bsdf_energy_compensated_image_has_size(ETX_IN(BSDFResourceContext, context), uint32_t image_index, uint32_t width, uint32_t height) {
  if ((context.scene == nullptr) || (image_index == kInvalidIndex) || (image_index >= context.scene->images.count)) {
    return false;
  }

  const auto& image = context.scene->images[image_index];
  return (image.isize.x == width) && (image.isize.y == height) && (image.format == Image::Format::RGBA32F);
}

ETX_SHARED_INLINE float4 bsdf_energy_compensated_sample_image(ETX_IN(BSDFResourceContext, context), uint32_t image_index, ETX_IN(float2, uv), uint32_t expected_width,
  uint32_t expected_height) {
  if (bsdf_energy_compensated_image_has_size(context, image_index, expected_width, expected_height) == false) {
    return float4(0.0f, 0.0f, 0.0f, 0.0f);
  }

  return context.scene->images[image_index].evaluate_rgba32f_fast(uv);
}

ETX_SHARED_INLINE bool bsdf_energy_compensated_material_interface_valid(ETX_IN(BSDFResourceContext, context), ETX_IN(Material, material), uint32_t material_class) {
  if ((context.scene == nullptr) || (material.energy_compensation_interface_index == kInvalidIndex) ||
      (material.energy_compensation_interface_index >= context.scene->energy_compensation_interfaces.count)) {
    return false;
  }

  const auto& interface_data = context.scene->energy_compensation_interfaces[material.energy_compensation_interface_index];
  return interface_data.cls == material_class;
}

ETX_SHARED_INLINE float bsdf_energy_compensated_conductor_alpha_axis(ETX_IN(BSDFResourceContext, context), ETX_IN(Material, material), float alpha) {
  (void)context;
  (void)material;
  return bsdf_energy_compensated_alpha_axis(alpha);
}

ETX_SHARED_INLINE uint32_t bsdf_energy_compensated_conductor_lut_index(ETX_IN(BSDFResourceContext, context), ETX_IN(Material, material)) {
  if (bsdf_energy_compensated_material_interface_valid(context, material, MaterialClass::Conductor)) {
    return context.scene->energy_compensation_interfaces[material.energy_compensation_interface_index].directional_lut;
  }

  return kInvalidIndex;
}

ETX_SHARED_INLINE uint32_t bsdf_energy_compensated_conductor_average_lut_index(ETX_IN(BSDFResourceContext, context), ETX_IN(Material, material)) {
  if (bsdf_energy_compensated_material_interface_valid(context, material, MaterialClass::Conductor)) {
    return context.scene->energy_compensation_interfaces[material.energy_compensation_interface_index].average_lut;
  }

  return kInvalidIndex;
}

ETX_SHARED_INLINE uint32_t bsdf_energy_compensated_dielectric_lut_index(ETX_IN(BSDFResourceContext, context), ETX_IN(Material, material)) {
  if (bsdf_energy_compensated_material_interface_valid(context, material, MaterialClass::Dielectric)) {
    return context.scene->energy_compensation_interfaces[material.energy_compensation_interface_index].directional_lut;
  }

  return kInvalidIndex;
}

ETX_SHARED_INLINE uint32_t bsdf_energy_compensated_dielectric_average_lut_index(ETX_IN(BSDFResourceContext, context), ETX_IN(Material, material)) {
  if (bsdf_energy_compensated_material_interface_valid(context, material, MaterialClass::Dielectric)) {
    return context.scene->energy_compensation_interfaces[material.energy_compensation_interface_index].average_lut;
  }

  return kInvalidIndex;
}

ETX_SHARED_INLINE float bsdf_energy_compensated_scalar_roughness(ETX_IN(BSDFResourceContext, context), ETX_IN(Material, material), ETX_IN(float2, uv)) {
  const float2 roughness = bsdf_resource_evaluate_roughness(context, material, uv);
  const float alpha = 0.5f * (roughness.x + roughness.y);
  return max(kBSDFNormalDistributionMinAlpha, bsdf_energy_compensated_saturate(alpha));
}

ETX_SHARED_INLINE bool bsdf_energy_compensated_material_supported(ETX_IN(BSDFResourceContext, context), ETX_IN(Material, material), ETX_IN(float2, uv)) {
  if (bsdf_resource_thinfilm_enabled(material.thinfilm)) {
    return false;
  }

  const float2 roughness = bsdf_resource_evaluate_roughness(context, material, uv);
  const float roughness_scale = max(1.0f, max(abs(roughness.x), abs(roughness.y)));
  const float tolerance = 16.0f * kEpsilon * roughness_scale;
  return abs(roughness.x - roughness.y) <= tolerance;
}

ETX_SHARED_INLINE SpectralResponse bsdf_energy_compensated_conductor_directional_albedo(ETX_IN(BSDFResourceContext, context), ETX_IN(SpectralQuery, spect),
  ETX_IN(Material, material), float mu, float alpha) {
  const uint32_t image_index = bsdf_energy_compensated_conductor_lut_index(context, material);
  const float2 uv = float2(bsdf_energy_compensated_lut_uv(mu, kBSDFEnergyCompensationConductorLutSize),
    bsdf_energy_compensated_lut_uv(bsdf_energy_compensated_conductor_alpha_axis(context, material, alpha), kBSDFEnergyCompensationConductorLutSize));
  const float4 value = bsdf_energy_compensated_sample_image(context, image_index, uv, kBSDFEnergyCompensationConductorLutSize, kBSDFEnergyCompensationConductorLutSize);
  return spectral_response_make(spect, saturate(float3(value.x, value.y, value.z)));
}

ETX_SHARED_INLINE float bsdf_energy_compensated_conductor_visible_probability(ETX_IN(BSDFResourceContext, context), ETX_IN(Material, material), float mu, float alpha) {
  if ((context.scene != nullptr) && (material.energy_compensation_interface_index != kInvalidIndex) &&
      (material.energy_compensation_interface_index < context.scene->energy_compensation_interfaces.count)) {
    const auto& interface_data = context.scene->energy_compensation_interfaces[material.energy_compensation_interface_index];
    if ((interface_data.cls == MaterialClass::Conductor) && (interface_data.geometric_lut != kInvalidIndex)) {
      const float2 uv = float2(bsdf_energy_compensated_lut_uv(mu, kBSDFEnergyCompensationConductorLutSize),
        bsdf_energy_compensated_lut_uv(bsdf_energy_compensated_conductor_alpha_axis(context, material, alpha), kBSDFEnergyCompensationConductorLutSize));
      const float4 value = bsdf_energy_compensated_sample_image(context, interface_data.geometric_lut, uv, kBSDFEnergyCompensationConductorLutSize,
        kBSDFEnergyCompensationConductorLutSize);
      return bsdf_energy_compensated_saturate(value.y);
    }
  }

  return 0.0f;
}

ETX_SHARED_INLINE SpectralResponse bsdf_energy_compensated_conductor_average_albedo(ETX_IN(BSDFResourceContext, context), ETX_IN(SpectralQuery, spect),
  ETX_IN(Material, material), float alpha) {
  const uint32_t image_index = bsdf_energy_compensated_conductor_average_lut_index(context, material);
  const float2 uv = float2(bsdf_energy_compensated_lut_uv(bsdf_energy_compensated_conductor_alpha_axis(context, material, alpha), kBSDFEnergyCompensationConductorLutSize), 0.0f);
  const float4 value = bsdf_energy_compensated_sample_image(context, image_index, uv, kBSDFEnergyCompensationConductorLutSize, 1u);
  return spectral_response_make(spect, saturate(float3(value.x, value.y, value.z)));
}

ETX_SHARED_INLINE float bsdf_energy_compensated_conductor_geometric_directional_albedo(ETX_IN(BSDFResourceContext, context), ETX_IN(Material, material), float mu, float alpha) {
  uint32_t image_index = kInvalidIndex;
  if ((context.scene != nullptr) && (material.energy_compensation_interface_index != kInvalidIndex) &&
      (material.energy_compensation_interface_index < context.scene->energy_compensation_interfaces.count)) {
    const auto& interface_data = context.scene->energy_compensation_interfaces[material.energy_compensation_interface_index];
    if ((interface_data.cls == MaterialClass::Conductor) && (interface_data.geometric_lut != kInvalidIndex)) {
      image_index = interface_data.geometric_lut;
    }
  }
  const float2 uv = float2(bsdf_energy_compensated_lut_uv(mu, kBSDFEnergyCompensationConductorLutSize),
    bsdf_energy_compensated_lut_uv(bsdf_energy_compensated_conductor_alpha_axis(context, material, alpha), kBSDFEnergyCompensationConductorLutSize));
  const float4 value = bsdf_energy_compensated_sample_image(context, image_index, uv, kBSDFEnergyCompensationConductorLutSize, kBSDFEnergyCompensationConductorLutSize);
  return bsdf_energy_compensated_saturate(value.x);
}

ETX_SHARED_INLINE float bsdf_energy_compensated_conductor_geometric_average_albedo(ETX_IN(BSDFResourceContext, context), ETX_IN(Material, material), float alpha) {
  uint32_t image_index = kInvalidIndex;
  if ((context.scene != nullptr) && (material.energy_compensation_interface_index != kInvalidIndex) &&
      (material.energy_compensation_interface_index < context.scene->energy_compensation_interfaces.count)) {
    const auto& interface_data = context.scene->energy_compensation_interfaces[material.energy_compensation_interface_index];
    if ((interface_data.cls == MaterialClass::Conductor) && (interface_data.geometric_average_lut != kInvalidIndex)) {
      image_index = interface_data.geometric_average_lut;
    }
  }
  const float2 uv = float2(bsdf_energy_compensated_lut_uv(bsdf_energy_compensated_conductor_alpha_axis(context, material, alpha), kBSDFEnergyCompensationConductorLutSize), 0.0f);
  const float4 value = bsdf_energy_compensated_sample_image(context, image_index, uv, kBSDFEnergyCompensationConductorLutSize, 1u);
  return bsdf_energy_compensated_saturate(value.x);
}

ETX_SHARED_INLINE SpectralResponse bsdf_energy_compensated_conductor_cached_fms(ETX_IN(BSDFResourceContext, context), ETX_IN(SpectralQuery, spect), ETX_IN(Material, material),
  float alpha) {
  uint32_t image_index = kInvalidIndex;
  if ((context.scene != nullptr) && (material.energy_compensation_interface_index != kInvalidIndex) &&
      (material.energy_compensation_interface_index < context.scene->energy_compensation_interfaces.count)) {
    const auto& interface_data = context.scene->energy_compensation_interfaces[material.energy_compensation_interface_index];
    if ((interface_data.cls == MaterialClass::Conductor) && (interface_data.conductor_fms_lut != kInvalidIndex)) {
      image_index = interface_data.conductor_fms_lut;
    }
  }

  const float2 uv = float2(bsdf_energy_compensated_lut_uv(bsdf_energy_compensated_conductor_alpha_axis(context, material, alpha), kBSDFEnergyCompensationConductorLutSize), 0.0f);
  const float4 value = bsdf_energy_compensated_sample_image(context, image_index, uv, kBSDFEnergyCompensationConductorLutSize, 1u);
  return spectral_response_make(spect, saturate(float3(value.x, value.y, value.z)));
}

ETX_SHARED_INLINE float bsdf_energy_compensated_dielectric_f0(ETX_IN(RefractiveIndexSample, ext_ior), ETX_IN(RefractiveIndexSample, int_ior)) {
  const float eta_ext = max(kEpsilon, spectral_response_monochromatic(ext_ior.eta));
  const float eta_int = max(kEpsilon, spectral_response_monochromatic(int_ior.eta));
  const float numerator = eta_int - eta_ext;
  const float denominator = max(kEpsilon, eta_int + eta_ext);
  const float f0 = (numerator * numerator) / (denominator * denominator);
  return min(kBSDFEnergyCompensationF0Max, bsdf_energy_compensated_saturate(f0));
}

ETX_SHARED_INLINE bool bsdf_energy_compensated_dielectric_low_to_high(ETX_IN(RefractiveIndexSample, ext_ior), ETX_IN(RefractiveIndexSample, int_ior), bool outside) {
  const float eta_ext = max(kEpsilon, spectral_response_monochromatic(ext_ior.eta));
  const float eta_int = max(kEpsilon, spectral_response_monochromatic(int_ior.eta));
  return outside ? (eta_ext <= eta_int) : (eta_int <= eta_ext);
}

ETX_SHARED_INLINE float bsdf_energy_compensated_dielectric_continuation_eta(ETX_IN(RefractiveIndexSample, ext_ior), ETX_IN(RefractiveIndexSample, int_ior), bool outside) {
  const float eta_ext = max(kEpsilon, spectral_response_monochromatic(ext_ior.eta));
  const float eta_int = max(kEpsilon, spectral_response_monochromatic(int_ior.eta));
  return outside ? (eta_int / eta_ext) : (eta_ext / eta_int);
}

ETX_SHARED_INLINE uint32_t bsdf_energy_compensated_dielectric_side(bool outside) {
  return outside ? 0u : 1u;
}

ETX_SHARED_INLINE uint32_t bsdf_energy_compensated_dielectric_branch_index(bool incident_outside, bool outgoing_outside) {
  return bsdf_energy_compensated_dielectric_side(incident_outside) * 2u + bsdf_energy_compensated_dielectric_side(outgoing_outside);
}

ETX_SHARED_INLINE float4 bsdf_energy_compensated_dielectric_branch_value(ETX_IN(BSDFResourceContext, context), ETX_IN(Material, material), float mu, float alpha,
  bool incident_outside, bool outgoing_outside) {
  const uint32_t image_index = bsdf_energy_compensated_dielectric_lut_index(context, material);
  const uint32_t expected_width = 4u * kBSDFEnergyCompensationDielectricLutSize;
  const uint32_t branch_offset = bsdf_energy_compensated_dielectric_branch_index(incident_outside, outgoing_outside) * kBSDFEnergyCompensationDielectricLutSize;
  const float x = (static_cast<float>(branch_offset) + bsdf_energy_compensated_saturate(mu) * static_cast<float>(kBSDFEnergyCompensationDielectricLutSize - 1u)) /
                  static_cast<float>(expected_width);
  const float y = bsdf_energy_compensated_lut_uv(bsdf_energy_compensated_dielectric_alpha_axis(alpha), kBSDFEnergyCompensationDielectricLutSize);
  return bsdf_energy_compensated_sample_image(context, image_index, float2(x, y), expected_width, kBSDFEnergyCompensationDielectricLutSize);
}

ETX_SHARED_INLINE BSDFEnergyCompensatedDielectricBranchPair bsdf_energy_compensated_dielectric_branch_pair_value(ETX_IN(BSDFResourceContext, context),
  ETX_IN(Material, material), float mu, float alpha, bool incident_outside) {
  BSDFEnergyCompensatedDielectricBranchPair result = ETX_ZERO(BSDFEnergyCompensatedDielectricBranchPair);
  const uint32_t image_index = bsdf_energy_compensated_dielectric_lut_index(context, material);
  const uint32_t expected_width = 4u * kBSDFEnergyCompensationDielectricLutSize;
  const uint32_t branch_0_offset = bsdf_energy_compensated_dielectric_branch_index(incident_outside, true) * kBSDFEnergyCompensationDielectricLutSize;
  const uint32_t branch_1_offset = bsdf_energy_compensated_dielectric_branch_index(incident_outside, false) * kBSDFEnergyCompensationDielectricLutSize;
  const float mu_axis = bsdf_energy_compensated_saturate(mu) * static_cast<float>(kBSDFEnergyCompensationDielectricLutSize - 1u);
  const float y = bsdf_energy_compensated_lut_uv(bsdf_energy_compensated_dielectric_alpha_axis(alpha), kBSDFEnergyCompensationDielectricLutSize);
  const float x_0 = (static_cast<float>(branch_0_offset) + mu_axis) / static_cast<float>(expected_width);
  const float x_1 = (static_cast<float>(branch_1_offset) + mu_axis) / static_cast<float>(expected_width);
  result.outside_value = bsdf_energy_compensated_sample_image(context, image_index, float2(x_0, y), expected_width, kBSDFEnergyCompensationDielectricLutSize);
  result.inside_value = bsdf_energy_compensated_sample_image(context, image_index, float2(x_1, y), expected_width, kBSDFEnergyCompensationDielectricLutSize);
  return result;
}

ETX_SHARED_INLINE SpectralResponse bsdf_energy_compensated_dielectric_branch_pair_albedo(ETX_IN(SpectralQuery, spect),
  ETX_IN(BSDFEnergyCompensatedDielectricBranchPair, pair)) {
  const float3 outside_albedo = saturate(float3(pair.outside_value.x, pair.outside_value.y, pair.outside_value.z));
  const float3 inside_albedo = saturate(float3(pair.inside_value.x, pair.inside_value.y, pair.inside_value.z));
  return spectral_response_make(spect, min(outside_albedo + inside_albedo, float3(1.0f, 1.0f, 1.0f)));
}

ETX_SHARED_INLINE float bsdf_energy_compensated_dielectric_branch_pair_visible_probability(ETX_IN(BSDFEnergyCompensatedDielectricBranchPair, pair)) {
  return bsdf_energy_compensated_saturate(pair.outside_value.w + pair.inside_value.w);
}

ETX_SHARED_INLINE SpectralResponse bsdf_energy_compensated_dielectric_branch_pair_selected_albedo(ETX_IN(SpectralQuery, spect),
  ETX_IN(BSDFEnergyCompensatedDielectricBranchPair, pair), bool outgoing_outside) {
  const float4 value = outgoing_outside ? pair.outside_value : pair.inside_value;
  return spectral_response_make(spect, saturate(float3(value.x, value.y, value.z)));
}

ETX_SHARED_INLINE float bsdf_energy_compensated_dielectric_branch_pair_selected_visible_probability(ETX_IN(BSDFEnergyCompensatedDielectricBranchPair, pair),
  bool outgoing_outside) {
  const float4 value = outgoing_outside ? pair.outside_value : pair.inside_value;
  return bsdf_energy_compensated_saturate(value.w);
}

ETX_SHARED_INLINE SpectralResponse bsdf_energy_compensated_dielectric_branch_albedo(ETX_IN(BSDFResourceContext, context), ETX_IN(SpectralQuery, spect),
  ETX_IN(Material, material), float mu, float alpha, bool incident_outside, bool outgoing_outside) {
  const float4 value = bsdf_energy_compensated_dielectric_branch_value(context, material, mu, alpha, incident_outside, outgoing_outside);
  return spectral_response_make(spect, saturate(float3(value.x, value.y, value.z)));
}

ETX_SHARED_INLINE float bsdf_energy_compensated_dielectric_branch_visible_probability(ETX_IN(BSDFResourceContext, context), ETX_IN(Material, material), float mu, float alpha,
  bool incident_outside, bool outgoing_outside) {
  const float4 value = bsdf_energy_compensated_dielectric_branch_value(context, material, mu, alpha, incident_outside, outgoing_outside);
  return bsdf_energy_compensated_saturate(value.w);
}

ETX_SHARED_INLINE SpectralResponse bsdf_energy_compensated_dielectric_directional_albedo(ETX_IN(BSDFResourceContext, context), ETX_IN(SpectralQuery, spect),
  ETX_IN(Material, material), float mu, float alpha, float f0, bool outside, bool low_to_high) {
  (void)f0;
  (void)low_to_high;
  const BSDFEnergyCompensatedDielectricBranchPair pair = bsdf_energy_compensated_dielectric_branch_pair_value(context, material, mu, alpha, outside);
  return bsdf_energy_compensated_dielectric_branch_pair_albedo(spect, pair);
}

ETX_SHARED_INLINE float bsdf_energy_compensated_dielectric_visible_probability(ETX_IN(BSDFResourceContext, context), ETX_IN(Material, material), float mu, float alpha, float f0,
  bool outside, bool low_to_high) {
  (void)f0;
  (void)low_to_high;
  const BSDFEnergyCompensatedDielectricBranchPair pair = bsdf_energy_compensated_dielectric_branch_pair_value(context, material, mu, alpha, outside);
  return bsdf_energy_compensated_dielectric_branch_pair_visible_probability(pair);
}

ETX_SHARED_INLINE float4 bsdf_energy_compensated_dielectric_average_value(ETX_IN(BSDFResourceContext, context), ETX_IN(Material, material), float alpha, uint32_t column) {
  const uint32_t image_index = bsdf_energy_compensated_dielectric_average_lut_index(context, material);
  const float x = static_cast<float>(column) / 8.0f;
  const float y = bsdf_energy_compensated_lut_uv(bsdf_energy_compensated_dielectric_alpha_axis(alpha), kBSDFEnergyCompensationDielectricLutSize);
  return bsdf_energy_compensated_sample_image(context, image_index, float2(x, y), 8u, kBSDFEnergyCompensationDielectricLutSize);
}

ETX_SHARED_INLINE SpectralResponse bsdf_energy_compensated_dielectric_branch_average_albedo(ETX_IN(BSDFResourceContext, context), ETX_IN(SpectralQuery, spect),
  ETX_IN(Material, material), float alpha, bool incident_outside, bool outgoing_outside) {
  const uint32_t column = bsdf_energy_compensated_dielectric_branch_index(incident_outside, outgoing_outside);
  const float4 value = bsdf_energy_compensated_dielectric_average_value(context, material, alpha, column);
  return spectral_response_make(spect, saturate(float3(value.x, value.y, value.z)));
}

ETX_SHARED_INLINE SpectralResponse bsdf_energy_compensated_dielectric_branch_coefficient(ETX_IN(BSDFResourceContext, context), ETX_IN(SpectralQuery, spect),
  ETX_IN(Material, material), float alpha, bool incident_outside, bool outgoing_outside) {
  const uint32_t column = 4u + bsdf_energy_compensated_dielectric_branch_index(incident_outside, outgoing_outside);
  const float4 value = bsdf_energy_compensated_dielectric_average_value(context, material, alpha, column);
  return spectral_response_make(spect, max(float3(value.x, value.y, value.z), float3(0.0f, 0.0f, 0.0f)));
}

ETX_SHARED_INLINE SpectralResponse bsdf_energy_compensated_dielectric_average_albedo(ETX_IN(BSDFResourceContext, context), ETX_IN(SpectralQuery, spect),
  ETX_IN(Material, material), float alpha, float f0, bool outside, bool low_to_high) {
  (void)f0;
  (void)low_to_high;
  const SpectralResponse outside_albedo = bsdf_energy_compensated_dielectric_branch_average_albedo(context, spect, material, alpha, outside, true);
  const SpectralResponse inside_albedo = bsdf_energy_compensated_dielectric_branch_average_albedo(context, spect, material, alpha, outside, false);
  return spectral_response_min(spectral_response_add(outside_albedo, inside_albedo), 1.0f);
}

ETX_SHARED_INLINE SpectralResponse bsdf_energy_compensated_dielectric_average_residual(ETX_IN(BSDFResourceContext, context), ETX_IN(SpectralQuery, spect),
  ETX_IN(Material, material), float alpha, bool outside) {
  const SpectralResponse average_albedo = bsdf_energy_compensated_dielectric_average_albedo(context, spect, material, alpha, 0.0f, outside, true);
  return spectral_response_max(spectral_response_sub(spectral_response_make(spect, 1.0f), average_albedo), 0.0f);
}

ETX_SHARED_INLINE float bsdf_energy_compensated_dielectric_compensation_branch_probability(ETX_IN(BSDFResourceContext, context), ETX_IN(SpectralQuery, spect),
  ETX_IN(Material, material), float alpha, bool incident_outside, bool outgoing_outside) {
  const SpectralResponse outside_residual = bsdf_energy_compensated_dielectric_average_residual(context, spect, material, alpha, true);
  const SpectralResponse inside_residual = bsdf_energy_compensated_dielectric_average_residual(context, spect, material, alpha, false);
  const SpectralResponse outside_coefficient = bsdf_energy_compensated_dielectric_branch_coefficient(context, spect, material, alpha, incident_outside, true);
  const SpectralResponse inside_coefficient = bsdf_energy_compensated_dielectric_branch_coefficient(context, spect, material, alpha, incident_outside, false);
  const float outside_probability = max(0.0f, spectral_response_monochromatic(spectral_response_mul(outside_coefficient, outside_residual)));
  const float inside_probability = max(0.0f, spectral_response_monochromatic(spectral_response_mul(inside_coefficient, inside_residual)));
  const float normalization = outside_probability + inside_probability;
  if (normalization <= kEpsilon) {
    return 0.0f;
  }
  const float probability = outgoing_outside ? outside_probability : inside_probability;
  return bsdf_energy_compensated_saturate(probability / normalization);
}

ETX_SHARED_INLINE ThinfilmEval bsdf_energy_compensated_empty_thinfilm() {
  ThinfilmEval result = ETX_ZERO(ThinfilmEval);
  result.ior.cls = SpectralDistribution::Invalid;
  result.rgb_wavelengths = kRGBWavelengths;
  result.thickness = 0.0f;
  return result;
}

ETX_SHARED_INLINE SpectralResponse bsdf_energy_compensated_average_fresnel(ETX_IN(SpectralQuery, spect), ETX_IN(RefractiveIndexSample, ext_ior),
  ETX_IN(RefractiveIndexSample, int_ior)) {
  const float nodes[8] = {
    1.98550718e-2f,
    1.01666761e-1f,
    2.37233795e-1f,
    4.08282679e-1f,
    5.91717321e-1f,
    7.62766205e-1f,
    8.98333239e-1f,
    9.80144928e-1f,
  };
  const float weights[8] = {
    5.06142681e-2f,
    1.11190517e-1f,
    1.56853323e-1f,
    1.81341892e-1f,
    1.81341892e-1f,
    1.56853323e-1f,
    1.11190517e-1f,
    5.06142681e-2f,
  };

  const ThinfilmEval thinfilm = bsdf_energy_compensated_empty_thinfilm();
  SpectralResponse result = spectral_response_make(spect, 0.0f);
  for (uint32_t i = 0u; i < 8u; ++i) {
    const float mu = nodes[i];
    const SpectralResponse fresnel = bsdf_fresnel_calculate(spect, mu, ext_ior, int_ior, thinfilm);
    result = spectral_response_add(result, spectral_response_mul(fresnel, 2.0f * weights[i] * mu));
  }
  return result;
}

ETX_SHARED_INLINE SpectralResponse bsdf_energy_compensated_conductor_fms(ETX_IN(SpectralQuery, spect), ETX_IN(RefractiveIndexSample, ext_ior),
  ETX_IN(RefractiveIndexSample, int_ior), float average_albedo) {
  const SpectralResponse fresnel_average = bsdf_energy_compensated_average_fresnel(spect, ext_ior, int_ior);
  if (bsdf_energy_compensated_spectral_response_finite(fresnel_average) == false) {
    return spectral_response_make(spect, 0.0f);
  }

  const SpectralResponse numerator = spectral_response_mul(spectral_response_mul(fresnel_average, fresnel_average), average_albedo);
  const SpectralResponse denominator =
    spectral_response_sub(spectral_response_make(spect, 1.0f), spectral_response_mul(fresnel_average, 1.0f - average_albedo));
  if ((bsdf_energy_compensated_spectral_response_finite(numerator) == false) || (bsdf_energy_compensated_spectral_response_finite(denominator) == false)) {
    return spectral_response_make(spect, 0.0f);
  }

  return spectral_response_div(numerator, spectral_response_max(denominator, kEpsilon));
}

ETX_SHARED_INLINE float3 bsdf_energy_compensated_sample_vndf_local(ETX_IN(float3, w_i), float alpha, ETX_IN(float2, rnd)) {
  const float3 w_i_11 = normalize(float3(alpha * w_i.x, alpha * w_i.y, w_i.z));
  const float2 slope_11 = bsdf_external_sample_p22_11(acos(bsdf_energy_compensated_saturate(w_i_11.z)), rnd, float2(alpha, alpha));

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

ETX_SHARED_INLINE float bsdf_energy_compensated_vndf_pdf(ETX_IN(float3, w_i), ETX_IN(float3, m), float alpha) {
  const BSDFExternalRayInfo ray = bsdf_external_ray_info_make(w_i, float2(alpha, alpha));
  const float denominator = (1.0f + ray.Lambda) * max(kEpsilon, w_i.z);
  return max(0.0f, dot(w_i, m)) * bsdf_external_d_ggx(m, float2(alpha, alpha)) / denominator;
}

ETX_SHARED_INLINE BSDFEnergyCompensatedLobe bsdf_energy_compensated_conductor_base_lobe(ETX_IN(SpectralQuery, spect), ETX_IN(float3, w_i), ETX_IN(float3, w_o), float alpha,
  ETX_IN(RefractiveIndexSample, ext_ior), ETX_IN(RefractiveIndexSample, int_ior), ETX_IN(SpectralResponse, reflectance)) {
  BSDFEnergyCompensatedLobe result = ETX_ZERO(BSDFEnergyCompensatedLobe);
  result.bsdf = spectral_response_make(spect, 0.0f);

  if ((w_i.z <= kEpsilon) || (w_o.z <= kEpsilon)) {
    return result;
  }

  const float3 half_vector_sum = w_i + w_o;
  const float half_vector_length_sq = dot(half_vector_sum, half_vector_sum);
  if (half_vector_length_sq <= kEpsilon) {
    return result;
  }

  const float3 m = normalize(half_vector_sum);
  if ((m.z <= kEpsilon) || (dot(w_i, m) <= kEpsilon) || (dot(w_o, m) <= kEpsilon)) {
    return result;
  }

  const ThinfilmEval thinfilm = bsdf_energy_compensated_empty_thinfilm();
  const SpectralResponse fresnel = bsdf_fresnel_calculate(spect, dot(w_i, m), ext_ior, int_ior, thinfilm);
  const float lambda_i = bsdf_external_ray_info_make(w_i, float2(alpha, alpha)).Lambda;
  const float lambda_o = bsdf_external_ray_info_make(w_o, float2(alpha, alpha)).Lambda;
  const float d = bsdf_external_d_ggx(m, float2(alpha, alpha));
  const float g2 = 1.0f / (1.0f + lambda_i + lambda_o);
  result.bsdf = spectral_response_mul(spectral_response_mul(spectral_response_mul(fresnel, reflectance), d * g2 / (4.0f * w_i.z)), 1.0f);

  const float vndf_pdf = bsdf_energy_compensated_vndf_pdf(w_i, m, alpha);
  result.pdf = vndf_pdf / max(kEpsilon, 4.0f * dot(w_o, m));
  return result;
}

ETX_SHARED_INLINE float bsdf_energy_compensated_conductor_pdf_local(ETX_IN(BSDFResourceContext, context), ETX_IN(SpectralQuery, spect), ETX_IN(Material, material),
  ETX_IN(float3, w_i), ETX_IN(float3, w_o), float alpha) {
  if ((w_i.z <= kEpsilon) || (w_o.z <= kEpsilon)) {
    return 0.0f;
  }

  const SpectralResponse e_i_response = bsdf_energy_compensated_conductor_directional_albedo(context, spect, material, w_i.z, alpha);
  const float e_i = spectral_response_monochromatic(e_i_response);
  const float visible_probability = bsdf_energy_compensated_conductor_visible_probability(context, material, w_i.z, alpha);
  const float specular_probability = (visible_probability > kEpsilon) ? bsdf_energy_compensated_saturate(e_i) : 0.0f;

  const float3 half_vector_sum = w_i + w_o;
  const float half_vector_length_sq = dot(half_vector_sum, half_vector_sum);
  float specular_pdf = 0.0f;
  if (half_vector_length_sq > kEpsilon) {
    const float3 m = normalize(half_vector_sum);
    if ((m.z > kEpsilon) && (dot(w_i, m) > kEpsilon) && (dot(w_o, m) > kEpsilon)) {
      const float raw_specular_pdf = bsdf_energy_compensated_vndf_pdf(w_i, m, alpha) / max(kEpsilon, 4.0f * dot(w_o, m));
      specular_pdf = raw_specular_pdf / max(kEpsilon, visible_probability);
    }
  }

  const float compensation_pdf = w_o.z * kInvPi;
  return specular_probability * specular_pdf + (1.0f - specular_probability) * compensation_pdf;
}

ETX_SHARED_INLINE float bsdf_energy_compensated_conductor_pdf_from_base_lobe(ETX_IN(BSDFResourceContext, context), ETX_IN(SpectralQuery, spect), ETX_IN(Material, material),
  ETX_IN(float3, w_i), ETX_IN(float3, w_o), float alpha, float base_lobe_pdf) {
  if ((w_i.z <= kEpsilon) || (w_o.z <= kEpsilon)) {
    return 0.0f;
  }

  const SpectralResponse e_i_response = bsdf_energy_compensated_conductor_directional_albedo(context, spect, material, w_i.z, alpha);
  const float e_i = spectral_response_monochromatic(e_i_response);
  const float visible_probability = bsdf_energy_compensated_conductor_visible_probability(context, material, w_i.z, alpha);
  const float specular_probability = (visible_probability > kEpsilon) ? bsdf_energy_compensated_saturate(e_i) : 0.0f;
  const float specular_pdf = (visible_probability > kEpsilon) ? (base_lobe_pdf / visible_probability) : 0.0f;
  const float compensation_pdf = w_o.z * kInvPi;
  return specular_probability * specular_pdf + (1.0f - specular_probability) * compensation_pdf;
}

ETX_SHARED_INLINE BSDFEval bsdf_conductor_energy_compensated_evaluate(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(float3, outgoing_direction),
  ETX_IN(Material, material), ETX_INOUT(Sampler, sampler)) {
  const LocalFrame frame = bsdf_data_get_normal_frame(data, material);
  const float3 w_i = local_frame_to_local(frame, -data.w_i);
  const float3 w_o = local_frame_to_local(frame, outgoing_direction);
  const float alpha = bsdf_energy_compensated_scalar_roughness(context, material, data.tex);
  const RefractiveIndexSample ext_ior = bsdf_resource_evaluate_refractive_index(context, material.ext_ior, data.spectrum_sample);
  const RefractiveIndexSample int_ior = bsdf_resource_evaluate_refractive_index(context, material.int_ior, data.spectrum_sample);

  if (bsdf_conductor_is_delta_with_context(context, material, data.tex)) {
    const ThinfilmEval thinfilm = bsdf_resource_evaluate_thinfilm(context, data.spectrum_sample, material.thinfilm, data.tex, sampler);
    const float3 ideal_w_o = bsdf_conductor_delta_reflect(data, material);
    const float3 actual_w_o = normalize(outgoing_direction);
    if (direction_matches(ideal_w_o, actual_w_o, 1.0f) == false) {
      return bsdf_eval_zero(data.spectrum_sample);
    }

    BSDFEval result = ETX_ZERO(BSDFEval);
    result.bsdf = bsdf_conductor_delta_weight(context, data, material, ext_ior, int_ior, thinfilm);
    result.func = result.bsdf;
    result.pdf = 1.0f;
    result.eta = 1.0f;
    result.properties = BSDFSample::Delta | BSDFSample::Reflection;
    result.medium_index = data.current_medium;
    return result;
  }

  if ((w_i.z <= kEpsilon) || (w_o.z <= kEpsilon)) {
    return bsdf_eval_zero(data.spectrum_sample);
  }
  if (bsdf_energy_compensated_material_supported(context, material, data.tex) == false) {
    return bsdf_eval_zero(data.spectrum_sample);
  }
  if (bsdf_energy_compensated_material_interface_valid(context, material, MaterialClass::Conductor) == false) {
    return bsdf_eval_zero(data.spectrum_sample);
  }

  const SpectralResponse reflectance = bsdf_resource_apply_image(context, data.spectrum_sample, material.reflectance, data.tex);
  const BSDFEnergyCompensatedLobe base_lobe =
    bsdf_energy_compensated_conductor_base_lobe(data.spectrum_sample, w_i, w_o, alpha, ext_ior, int_ior, reflectance);

  SpectralResponse compensation_bsdf = spectral_response_make(data.spectrum_sample, 0.0f);
  const float e_i_scalar = bsdf_energy_compensated_conductor_geometric_directional_albedo(context, material, w_i.z, alpha);
  const float e_o_scalar = bsdf_energy_compensated_conductor_geometric_directional_albedo(context, material, w_o.z, alpha);
  const float e_average_scalar = bsdf_energy_compensated_conductor_geometric_average_albedo(context, material, alpha);
  if ((1.0f - e_average_scalar) > kEpsilon) {
    const SpectralResponse f_ms = spectral_query_is_spectral(data.spectrum_sample) ? bsdf_energy_compensated_conductor_fms(data.spectrum_sample, ext_ior, int_ior, e_average_scalar) :
                                                                                     bsdf_energy_compensated_conductor_cached_fms(context, data.spectrum_sample, material, alpha);
    const float scalar = (1.0f - e_i_scalar) * (1.0f - e_o_scalar) * w_o.z / (kPi * (1.0f - e_average_scalar));
    compensation_bsdf = spectral_response_mul(spectral_response_mul(f_ms, reflectance), scalar);
  }

  BSDFEval result = ETX_ZERO(BSDFEval);
  result.bsdf = spectral_response_add(base_lobe.bsdf, compensation_bsdf);
  if (bsdf_energy_compensated_spectral_response_finite(result.bsdf) == false) {
    return bsdf_eval_zero(data.spectrum_sample);
  }

  result.func = spectral_response_div(result.bsdf, w_o.z);
  if (bsdf_energy_compensated_spectral_response_finite(result.func) == false) {
    return bsdf_eval_zero(data.spectrum_sample);
  }

  result.pdf = bsdf_energy_compensated_conductor_pdf_from_base_lobe(context, data.spectrum_sample, material, w_i, w_o, alpha, base_lobe.pdf);
  result.eta = 1.0f;
  result.properties = BSDFSample::Reflection;
  result.medium_index = data.current_medium;
  return result;
}

ETX_SHARED_INLINE BSDFSample bsdf_conductor_energy_compensated_sample(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(Material, material),
  ETX_INOUT(Sampler, sampler)) {
  const LocalFrame frame = bsdf_data_get_normal_frame(data, material);
  const float3 w_i = local_frame_to_local(frame, -data.w_i);
  if (w_i.z <= kEpsilon) {
    return bsdf_sample_zero(data.spectrum_sample);
  }

  if (bsdf_conductor_is_delta_with_context(context, material, data.tex)) {
    const RefractiveIndexSample ext_ior = bsdf_resource_evaluate_refractive_index(context, material.ext_ior, data.spectrum_sample);
    const RefractiveIndexSample int_ior = bsdf_resource_evaluate_refractive_index(context, material.int_ior, data.spectrum_sample);
    const ThinfilmEval thinfilm = bsdf_resource_evaluate_thinfilm(context, data.spectrum_sample, material.thinfilm, data.tex, sampler);
    BSDFSample result = ETX_ZERO(BSDFSample);
    result.w_o = bsdf_conductor_delta_reflect(data, material);
    result.weight = bsdf_conductor_delta_weight(context, data, material, ext_ior, int_ior, thinfilm);
    result.pdf = 1.0f;
    result.eta = 1.0f;
    result.properties = BSDFSample::Delta | BSDFSample::Reflection;
    result.medium_index = data.current_medium;
    return result;
  }
  if (bsdf_energy_compensated_material_supported(context, material, data.tex) == false) {
    return bsdf_sample_zero(data.spectrum_sample);
  }
  if (bsdf_energy_compensated_material_interface_valid(context, material, MaterialClass::Conductor) == false) {
    return bsdf_sample_zero(data.spectrum_sample);
  }

  const float alpha = bsdf_energy_compensated_scalar_roughness(context, material, data.tex);
  const SpectralResponse e_i_response = bsdf_energy_compensated_conductor_directional_albedo(context, data.spectrum_sample, material, w_i.z, alpha);
  const float e_i = spectral_response_monochromatic(e_i_response);
  const float visible_probability = bsdf_energy_compensated_conductor_visible_probability(context, material, w_i.z, alpha);
  const float specular_probability = (visible_probability > kEpsilon) ? bsdf_energy_compensated_saturate(e_i) : 0.0f;
  const bool has_fixed = bsdf_sampler_has_fixed(sampler);
  const float selector = has_fixed ? sampler.fixed_w : bsdf_sampler_next(sampler);
  const float2 rnd = has_fixed ? float2(sampler.fixed_u, sampler.fixed_v) : bsdf_sampler_next_2d(sampler);

  float3 local_w_o = float3(0.0f, 0.0f, 1.0f);
  if (selector < specular_probability) {
    local_w_o = float3(0.0f, 0.0f, -1.0f);
    bool first_attempt = true;
    while (local_w_o.z <= kEpsilon) {
      const float2 attempt_rnd = (first_attempt && has_fixed) ? rnd : bsdf_sampler_next_2d(sampler);
      const float3 m = bsdf_energy_compensated_sample_vndf_local(w_i, alpha, attempt_rnd);
      local_w_o = -w_i + 2.0f * m * dot(w_i, m);
      first_attempt = false;
    }
  } else {
    local_w_o = sample_cosine_distribution(rnd, 1.0f);
  }

  if (local_w_o.z <= kEpsilon) {
    return bsdf_sample_zero(data.spectrum_sample);
  }

  const float3 world_w_o = normalize(local_frame_from_local(frame, local_w_o));
  BSDFEval eval = bsdf_conductor_energy_compensated_evaluate(context, data, world_w_o, material, sampler);
  if (bsdf_eval_valid(eval) == false) {
    return bsdf_sample_zero(data.spectrum_sample);
  }
  if ((isfinite(eval.pdf) == false) || (eval.pdf <= kEpsilon)) {
    return bsdf_sample_zero(data.spectrum_sample);
  }

  BSDFSample result = ETX_ZERO(BSDFSample);
  result.w_o = world_w_o;
  result.pdf = eval.pdf;
  result.weight = spectral_response_div(eval.bsdf, eval.pdf);
  if (bsdf_energy_compensated_spectral_response_finite(result.weight) == false) {
    return bsdf_sample_zero(data.spectrum_sample);
  }
  result.eta = 1.0f;
  result.properties = BSDFSample::Reflection;
  result.medium_index = data.current_medium;
  return result;
}

ETX_SHARED_INLINE float bsdf_conductor_energy_compensated_pdf(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(float3, outgoing_direction),
  ETX_IN(Material, material), ETX_INOUT(Sampler, sampler)) {
  (void)sampler;
  const LocalFrame frame = bsdf_data_get_normal_frame(data, material);
  const float3 w_i = local_frame_to_local(frame, -data.w_i);
  const float3 w_o = local_frame_to_local(frame, outgoing_direction);
  if (bsdf_conductor_is_delta_with_context(context, material, data.tex)) {
    const float3 ideal_w_o = bsdf_conductor_delta_reflect(data, material);
    const float3 actual_w_o = normalize(outgoing_direction);
    return direction_matches(ideal_w_o, actual_w_o, 1.0f) ? 1.0f : 0.0f;
  }
  if (bsdf_energy_compensated_material_supported(context, material, data.tex) == false) {
    return 0.0f;
  }
  if (bsdf_energy_compensated_material_interface_valid(context, material, MaterialClass::Conductor) == false) {
    return 0.0f;
  }
  const float alpha = bsdf_energy_compensated_scalar_roughness(context, material, data.tex);
  return bsdf_energy_compensated_conductor_pdf_local(context, data.spectrum_sample, material, w_i, w_o, alpha);
}

ETX_SHARED_INLINE bool bsdf_conductor_energy_compensated_is_delta(ETX_IN(Material, material), ETX_IN(float2, tex), ETX_INOUT(Sampler, sampler)) {
  return bsdf_conductor_is_delta(material, tex, sampler);
}

ETX_SHARED_INLINE bool bsdf_conductor_energy_compensated_is_delta_with_context(ETX_IN(BSDFResourceContext, context), ETX_IN(Material, material), ETX_IN(float2, tex)) {
  return bsdf_conductor_is_delta_with_context(context, material, tex);
}

ETX_SHARED_INLINE SpectralResponse bsdf_conductor_energy_compensated_albedo(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(Material, material),
  ETX_INOUT(Sampler, sampler)) {
  return bsdf_conductor_albedo(context, data, material, sampler);
}

ETX_SHARED_INLINE BSDFEnergyCompensatedLobe bsdf_energy_compensated_dielectric_base_lobe(ETX_IN(SpectralQuery, spect), ETX_IN(float3, w_i_local),
  ETX_IN(float3, w_o_local), float alpha, ETX_IN(RefractiveIndexSample, ext_ior), ETX_IN(RefractiveIndexSample, int_ior), ETX_IN(SpectralResponse, texture)) {
  BSDFEnergyCompensatedLobe result = ETX_ZERO(BSDFEnergyCompensatedLobe);
  result.bsdf = spectral_response_make(spect, 0.0f);

  if ((abs(w_i_local.z) <= kEpsilon) || (abs(w_o_local.z) <= kEpsilon)) {
    return result;
  }

  const bool outside = w_i_local.z > 0.0f;
  const float direction_scale = outside ? 1.0f : -1.0f;
  const float3 w_i = direction_scale * w_i_local;
  const float3 w_o = direction_scale * w_o_local;
  const bool reflection = w_o.z > 0.0f;
  const RefractiveIndexSample phase_ext_ior = outside ? ext_ior : int_ior;
  const RefractiveIndexSample phase_int_ior = outside ? int_ior : ext_ior;
  const ThinfilmEval thinfilm = bsdf_energy_compensated_empty_thinfilm();
  const float eta = spectral_response_monochromatic(spectral_response_div(phase_int_ior.eta, phase_ext_ior.eta));
  const float lambda_i = bsdf_external_ray_info_make(w_i, float2(alpha, alpha)).Lambda;
  const float g1_i = 1.0f / (1.0f + lambda_i);

  if (reflection) {
    const float3 half_vector_sum = w_i + w_o;
    if (dot(half_vector_sum, half_vector_sum) <= kEpsilon) {
      return result;
    }

    const float3 m = normalize(half_vector_sum);
    if ((m.z <= kEpsilon) || (dot(w_i, m) <= kEpsilon) || (dot(w_o, m) <= kEpsilon)) {
      return result;
    }

    const SpectralResponse fresnel = bsdf_fresnel_calculate(spect, dot(w_i, m), phase_ext_ior, phase_int_ior, thinfilm);
    const float lambda_o = bsdf_external_ray_info_make(w_o, float2(alpha, alpha)).Lambda;
    const float d = bsdf_external_d_ggx(m, float2(alpha, alpha));
    const float g2 = 1.0f / (1.0f + lambda_i + lambda_o);
    result.bsdf = spectral_response_mul(spectral_response_mul(spectral_response_mul(fresnel, texture), d * g2 / (4.0f * w_i.z)), 1.0f);

    const float vndf_pdf = bsdf_energy_compensated_vndf_pdf(w_i, m, alpha);
    const float fresnel_probability = spectral_response_monochromatic(fresnel);
    result.pdf = fresnel_probability * vndf_pdf / max(kEpsilon, 4.0f * dot(w_o, m));
    return result;
  }

  float3 m = normalize(w_i + w_o * eta);
  m *= (m.z >= 0.0f) ? 1.0f : -1.0f;
  const float i_dot_m = dot(w_i, m);
  const float o_dot_m = dot(w_o, m);
  const float denominator = i_dot_m + eta * o_dot_m;
  if ((m.z <= kEpsilon) || (i_dot_m <= kEpsilon) || (o_dot_m >= -kEpsilon) || (abs(denominator) <= kEpsilon)) {
    return result;
  }

  const SpectralResponse fresnel = bsdf_fresnel_calculate(spect, i_dot_m, phase_ext_ior, phase_int_ior, thinfilm);
  const SpectralResponse one_minus_fresnel = spectral_response_sub(spectral_response_make(spect, 1.0f), fresnel);
  const float3 oriented_w_o = -w_o;
  const float lambda_o = bsdf_external_ray_info_make(oriented_w_o, float2(alpha, alpha)).Lambda;
  const float d = bsdf_external_d_ggx(m, float2(alpha, alpha));
  const float g2 = bsdf_external_beta(1.0f + lambda_i, 1.0f + lambda_o);
  if (isfinite(g2) == false) {
    return result;
  }
  const float scalar = i_dot_m * max(0.0f, -o_dot_m) * d * g2 / (w_i.z * denominator * denominator);
  if (isfinite(scalar) == false) {
    return result;
  }
  result.bsdf = spectral_response_mul(spectral_response_mul(one_minus_fresnel, texture), scalar * eta * eta);

  const float vndf_pdf = bsdf_energy_compensated_vndf_pdf(w_i, m, alpha);
  const float fresnel_probability = 1.0f - spectral_response_monochromatic(fresnel);
  const float dwh_dwo = (eta * eta) * abs(o_dot_m) / (denominator * denominator);
  result.pdf = fresnel_probability * vndf_pdf * dwh_dwo;
  return result;
}

ETX_SHARED_INLINE float bsdf_energy_compensated_dielectric_base_pdf_local(ETX_IN(SpectralQuery, spect), ETX_IN(float3, w_i_local), ETX_IN(float3, w_o_local), float alpha,
  ETX_IN(RefractiveIndexSample, ext_ior), ETX_IN(RefractiveIndexSample, int_ior)) {
  const SpectralResponse texture = spectral_response_make(spect, 1.0f);
  const BSDFEnergyCompensatedLobe lobe = bsdf_energy_compensated_dielectric_base_lobe(spect, w_i_local, w_o_local, alpha, ext_ior, int_ior, texture);
  return lobe.pdf;
}

ETX_SHARED_INLINE BSDFEnergyCompensatedDielectricComponents bsdf_energy_compensated_dielectric_components_local(ETX_IN(BSDFResourceContext, context), ETX_IN(SpectralQuery, spect),
  ETX_IN(Material, material), ETX_IN(float3, w_i), ETX_IN(float3, w_o), float alpha, ETX_IN(RefractiveIndexSample, ext_ior), ETX_IN(RefractiveIndexSample, int_ior),
  ETX_IN(SpectralResponse, texture)) {
  BSDFEnergyCompensatedDielectricComponents result = ETX_ZERO(BSDFEnergyCompensatedDielectricComponents);
  result.base = spectral_response_make(spect, 0.0f);
  result.compensation = spectral_response_make(spect, 0.0f);
  if ((abs(w_i.z) <= kEpsilon) || (abs(w_o.z) <= kEpsilon)) {
    return result;
  }

  const BSDFEnergyCompensatedLobe base_lobe = bsdf_energy_compensated_dielectric_base_lobe(spect, w_i, w_o, alpha, ext_ior, int_ior, texture);
  result.base = base_lobe.bsdf;
  result.base_pdf = base_lobe.pdf;
  const bool incident_outside = w_i.z > 0.0f;
  const bool outgoing_outside = w_o.z > 0.0f;
  const BSDFEnergyCompensatedDielectricBranchPair incident_pair = bsdf_energy_compensated_dielectric_branch_pair_value(context, material, abs(w_i.z), alpha, incident_outside);
  const BSDFEnergyCompensatedDielectricBranchPair outgoing_pair = bsdf_energy_compensated_dielectric_branch_pair_value(context, material, abs(w_o.z), alpha, outgoing_outside);
  const SpectralResponse e_i = bsdf_energy_compensated_dielectric_branch_pair_albedo(spect, incident_pair);
  const SpectralResponse e_o = bsdf_energy_compensated_dielectric_branch_pair_albedo(spect, outgoing_pair);
  result.incident_albedo = e_i;
  result.incident_visible_probability = bsdf_energy_compensated_dielectric_branch_pair_visible_probability(incident_pair);
  const SpectralResponse one = spectral_response_make(spect, 1.0f);
  const SpectralResponse d_i = spectral_response_max(spectral_response_sub(one, e_i), 0.0f);
  const SpectralResponse d_o = spectral_response_max(spectral_response_sub(one, e_o), 0.0f);
  const SpectralResponse coefficient = bsdf_energy_compensated_dielectric_branch_coefficient(context, spect, material, alpha, incident_outside, outgoing_outside);
  result.compensation =
    spectral_response_mul(spectral_response_mul(spectral_response_mul(texture, coefficient), spectral_response_mul(d_i, d_o)), abs(w_o.z) * kInvPi);
  if ((bsdf_energy_compensated_spectral_response_finite(result.base) == false) || (bsdf_energy_compensated_spectral_response_finite(result.compensation) == false)) {
    result.base = spectral_response_make(spect, 0.0f);
    result.compensation = spectral_response_make(spect, 0.0f);
  }

  return result;
}

ETX_SHARED_INLINE float bsdf_energy_compensated_dielectric_pdf_from_components(ETX_IN(BSDFResourceContext, context), ETX_IN(SpectralQuery, spect), ETX_IN(Material, material),
  ETX_IN(BSDFEnergyCompensatedDielectricComponents, components), ETX_IN(float3, w_o), float alpha, bool incident_outside) {
  if (components.incident_visible_probability <= kEpsilon) {
    return 0.0f;
  }

  const float normalized_base_pdf = components.base_pdf / components.incident_visible_probability;
  const float base_probability = bsdf_energy_compensated_saturate(spectral_response_monochromatic(components.incident_albedo));
  const float compensation_probability = max(0.0f, 1.0f - base_probability);
  const bool outgoing_outside = w_o.z > 0.0f;
  const float branch_probability =
    bsdf_energy_compensated_dielectric_compensation_branch_probability(context, spect, material, alpha, incident_outside, outgoing_outside);
  const float compensation_pdf = branch_probability * abs(w_o.z) * kInvPi;
  return base_probability * normalized_base_pdf + compensation_probability * compensation_pdf;
}

ETX_SHARED_INLINE SpectralResponse bsdf_energy_compensated_dielectric_bsdf_local(ETX_IN(BSDFResourceContext, context), ETX_IN(SpectralQuery, spect), ETX_IN(Material, material),
  ETX_IN(float3, w_i), ETX_IN(float3, w_o), float alpha, ETX_IN(RefractiveIndexSample, ext_ior), ETX_IN(RefractiveIndexSample, int_ior),
  ETX_IN(SpectralResponse, texture)) {
  const BSDFEnergyCompensatedDielectricComponents components =
    bsdf_energy_compensated_dielectric_components_local(context, spect, material, w_i, w_o, alpha, ext_ior, int_ior, texture);
  const SpectralResponse result = spectral_response_add(components.base, components.compensation);
  if (bsdf_energy_compensated_spectral_response_finite(result) == false) {
    return spectral_response_make(spect, 0.0f);
  }

  return result;
}

ETX_SHARED_INLINE float bsdf_energy_compensated_dielectric_pdf_local(ETX_IN(BSDFResourceContext, context), ETX_IN(SpectralQuery, spect), ETX_IN(Material, material),
  ETX_IN(float3, w_i), ETX_IN(float3, w_o), float alpha, ETX_IN(RefractiveIndexSample, ext_ior), ETX_IN(RefractiveIndexSample, int_ior)) {
  if ((abs(w_i.z) <= kEpsilon) || (abs(w_o.z) <= kEpsilon)) {
    return 0.0f;
  }

  const bool outside = w_i.z > 0.0f;
  const BSDFEnergyCompensatedDielectricBranchPair incident_pair = bsdf_energy_compensated_dielectric_branch_pair_value(context, material, abs(w_i.z), alpha, outside);
  const float visible_probability = bsdf_energy_compensated_dielectric_branch_pair_visible_probability(incident_pair);
  if (visible_probability <= kEpsilon) {
    return 0.0f;
  }

  const float base_pdf = bsdf_energy_compensated_dielectric_base_pdf_local(spect, w_i, w_o, alpha, ext_ior, int_ior);
  const float normalized_base_pdf = base_pdf / visible_probability;
  const SpectralResponse e_i = bsdf_energy_compensated_dielectric_branch_pair_albedo(spect, incident_pair);
  const float base_probability = bsdf_energy_compensated_saturate(spectral_response_monochromatic(e_i));
  const float compensation_probability = max(0.0f, 1.0f - base_probability);
  const bool outgoing_outside = w_o.z > 0.0f;
  const float branch_probability =
    bsdf_energy_compensated_dielectric_compensation_branch_probability(context, spect, material, alpha, outside, outgoing_outside);
  const float compensation_pdf = branch_probability * abs(w_o.z) * kInvPi;
  return base_probability * normalized_base_pdf + compensation_probability * compensation_pdf;
}

ETX_SHARED_INLINE BSDFEval bsdf_energy_compensated_dielectric_evaluate_physical_local(ETX_IN(BSDFResourceContext, context), ETX_IN(SpectralQuery, spect),
  ETX_IN(Material, material), ETX_IN(float3, w_i), ETX_IN(float3, w_o), float alpha, ETX_IN(RefractiveIndexSample, ext_ior), ETX_IN(RefractiveIndexSample, int_ior),
  ETX_IN(SpectralResponse, texture), uint32_t current_medium) {
  const BSDFEnergyCompensatedDielectricComponents components =
    bsdf_energy_compensated_dielectric_components_local(context, spect, material, w_i, w_o, alpha, ext_ior, int_ior, texture);
  const SpectralResponse value = spectral_response_add(components.base, components.compensation);
  if (spectral_response_is_zero(value)) {
    return bsdf_eval_zero(spect);
  }
  if (bsdf_energy_compensated_spectral_response_finite(value) == false) {
    return bsdf_eval_zero(spect);
  }

  const bool outside_i = w_i.z > 0.0f;
  const float pdf = bsdf_energy_compensated_dielectric_pdf_from_components(context, spect, material, components, w_o, alpha, outside_i);
  if ((isfinite(pdf) == false) || (pdf <= 0.0f)) {
    return bsdf_eval_zero(spect);
  }

  const bool reflection = (w_i.z * w_o.z) > 0.0f;
  BSDFEval result = ETX_ZERO(BSDFEval);
  result.bsdf = value;
  result.func = spectral_response_div(result.bsdf, abs(w_o.z));
  if (bsdf_energy_compensated_spectral_response_finite(result.func) == false) {
    return bsdf_eval_zero(spect);
  }
  result.pdf = pdf;
  result.eta = reflection ? 1.0f : bsdf_energy_compensated_dielectric_continuation_eta(ext_ior, int_ior, outside_i);
  result.properties = reflection ? BSDFSample::Reflection : (BSDFSample::Transmission | BSDFSample::MediumChanged);
  result.medium_index = reflection ? current_medium : (outside_i ? material.int_medium : material.ext_medium);
  return result;
}

ETX_SHARED_INLINE BSDFEval bsdf_dielectric_energy_compensated_evaluate(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(float3, outgoing_direction),
  ETX_IN(Material, material), ETX_INOUT(Sampler, sampler)) {
  (void)sampler;

  if (bsdf_energy_compensated_material_supported(context, material, data.tex) == false) {
    return bsdf_eval_zero(data.spectrum_sample);
  }
  if (bsdf_dielectric_is_delta_with_context(context, material, data.tex)) {
    return bsdf_eval_zero(data.spectrum_sample);
  }
  if (bsdf_energy_compensated_material_interface_valid(context, material, MaterialClass::Dielectric) == false) {
    return bsdf_eval_zero(data.spectrum_sample);
  }

  LocalFrame frame = ETX_ZERO(LocalFrame);
  frame.tan = data.tan;
  frame.btn = data.btn;
  frame.nrm = data.nrm;
  const float3 w_i = local_frame_to_local(frame, -data.w_i);
  const float3 w_o = local_frame_to_local(frame, outgoing_direction);
  if ((abs(w_i.z) <= kEpsilon) || (abs(w_o.z) <= kEpsilon)) {
    return bsdf_eval_zero(data.spectrum_sample);
  }

  const float alpha = bsdf_energy_compensated_scalar_roughness(context, material, data.tex);
  const RefractiveIndexSample ext_ior = bsdf_resource_evaluate_refractive_index(context, material.ext_ior, data.spectrum_sample);
  const RefractiveIndexSample int_ior = bsdf_resource_evaluate_refractive_index(context, material.int_ior, data.spectrum_sample);
  const bool reflection = (w_i.z * w_o.z) > 0.0f;
  const SpectralImage texture_image = reflection ? material.reflectance : material.scattering;
  const SpectralResponse texture = bsdf_resource_apply_image(context, data.spectrum_sample, texture_image, data.tex);
  BSDFEval physical_eval = bsdf_energy_compensated_dielectric_evaluate_physical_local(context, data.spectrum_sample, material, w_i, w_o, alpha, ext_ior, int_ior, texture,
    data.current_medium);
  if (bsdf_eval_valid(physical_eval) == false) {
    return physical_eval;
  }
  if ((reflection == false) && (data.path_source == PathSource::Light)) {
    const SpectralResponse reverse_bsdf = bsdf_energy_compensated_dielectric_bsdf_local(context, data.spectrum_sample, material, w_o, w_i, alpha, ext_ior, int_ior, texture);
    const float cosine_scale = abs(w_o.z) / max(kEpsilon, abs(w_i.z));
    physical_eval.bsdf = spectral_response_mul(reverse_bsdf, cosine_scale);
    if (bsdf_energy_compensated_spectral_response_finite(physical_eval.bsdf) == false) {
      return bsdf_eval_zero(data.spectrum_sample);
    }
    physical_eval.func = spectral_response_div(physical_eval.bsdf, abs(w_o.z));
    if (bsdf_energy_compensated_spectral_response_finite(physical_eval.func) == false) {
      return bsdf_eval_zero(data.spectrum_sample);
    }
  }
  return physical_eval;
}

ETX_SHARED_INLINE BSDFSample bsdf_dielectric_energy_compensated_sample(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(Material, material),
  ETX_INOUT(Sampler, sampler)) {
  if (bsdf_dielectric_is_delta_with_context(context, material, data.tex)) {
    return bsdf_dielectric_delta_sample(context, data, material, sampler);
  }
  if (bsdf_energy_compensated_material_supported(context, material, data.tex) == false) {
    return bsdf_sample_zero(data.spectrum_sample);
  }
  if (bsdf_energy_compensated_material_interface_valid(context, material, MaterialClass::Dielectric) == false) {
    return bsdf_sample_zero(data.spectrum_sample);
  }

  LocalFrame frame = ETX_ZERO(LocalFrame);
  frame.tan = data.tan;
  frame.btn = data.btn;
  frame.nrm = data.nrm;
  const float3 w_i_local = local_frame_to_local(frame, -data.w_i);
  if (abs(w_i_local.z) <= kEpsilon) {
    return bsdf_sample_zero(data.spectrum_sample);
  }

  const float alpha = bsdf_energy_compensated_scalar_roughness(context, material, data.tex);
  const RefractiveIndexSample ext_ior = bsdf_resource_evaluate_refractive_index(context, material.ext_ior, data.spectrum_sample);
  const RefractiveIndexSample int_ior = bsdf_resource_evaluate_refractive_index(context, material.int_ior, data.spectrum_sample);
  const bool outside = w_i_local.z > 0.0f;
  const BSDFEnergyCompensatedDielectricBranchPair incident_pair = bsdf_energy_compensated_dielectric_branch_pair_value(context, material, abs(w_i_local.z), alpha, outside);
  const float visible_probability = bsdf_energy_compensated_dielectric_branch_pair_visible_probability(incident_pair);
  if (visible_probability <= kEpsilon) {
    return bsdf_sample_zero(data.spectrum_sample);
  }
  const SpectralResponse e_i = bsdf_energy_compensated_dielectric_branch_pair_albedo(data.spectrum_sample, incident_pair);
  const float base_probability = bsdf_energy_compensated_saturate(spectral_response_monochromatic(e_i));

  const bool has_fixed = bsdf_sampler_has_fixed(sampler);
  const float2 rnd = has_fixed ? float2(sampler.fixed_u, sampler.fixed_v) : bsdf_sampler_next_2d(sampler);

  const float direction_scale = outside ? 1.0f : -1.0f;
  const float3 w_i = direction_scale * w_i_local;
  const RefractiveIndexSample phase_ext_ior = outside ? ext_ior : int_ior;
  const RefractiveIndexSample phase_int_ior = outside ? int_ior : ext_ior;
  const ThinfilmEval thinfilm = bsdf_energy_compensated_empty_thinfilm();
  bool candidate_valid = false;
  float3 local_w_o = float3(0.0f, 0.0f, 0.0f);
  const float proposal_selector = has_fixed ? sampler.fixed_w : bsdf_sampler_next(sampler);
  if (proposal_selector < base_probability) {
    bool first_attempt = true;
    while (candidate_valid == false) {
      const float2 attempt_rnd = (first_attempt && has_fixed) ? rnd : bsdf_sampler_next_2d(sampler);
      const float3 m = bsdf_energy_compensated_sample_vndf_local(w_i, alpha, attempt_rnd);
      const SpectralResponse fresnel = bsdf_fresnel_calculate(data.spectrum_sample, dot(w_i, m), phase_ext_ior, phase_int_ior, thinfilm);
      const float fresnel_probability = spectral_response_monochromatic(fresnel);
      const float branch_selector = bsdf_sampler_next(sampler);
      if (branch_selector < fresnel_probability) {
        local_w_o = direction_scale * (-w_i + 2.0f * m * dot(w_i, m));
        candidate_valid = (w_i_local.z * local_w_o.z) > kEpsilon;
      } else {
        const float eta = spectral_response_monochromatic(spectral_response_div(phase_int_ior.eta, phase_ext_ior.eta));
        local_w_o = direction_scale * normalize(bsdf_external_refract(w_i, m, eta));
        candidate_valid = (w_i_local.z * local_w_o.z) < -kEpsilon;
      }
      first_attempt = false;
    }
  } else {
    const float outside_branch_probability =
      bsdf_energy_compensated_dielectric_compensation_branch_probability(context, data.spectrum_sample, material, alpha, outside, true);
    const bool outgoing_outside = bsdf_sampler_next(sampler) < outside_branch_probability;
    const float2 compensation_rnd = has_fixed ? rnd : bsdf_sampler_next_2d(sampler);
    const float r = sqrt(max(0.0f, compensation_rnd.x));
    const float phi = kDoublePi * compensation_rnd.y;
    const float z = sqrt(max(0.0f, 1.0f - compensation_rnd.x));
    if (outgoing_outside) {
      local_w_o = float3(r * cos(phi), r * sin(phi), z);
    } else {
      local_w_o = float3(r * cos(phi), r * sin(phi), -z);
    }
    candidate_valid = abs(local_w_o.z) > kEpsilon;
  }

  if (abs(local_w_o.z) <= kEpsilon) {
    return bsdf_sample_zero(data.spectrum_sample);
  }

  const float3 world_w_o = normalize(local_frame_from_local(frame, local_w_o));
  const BSDFEval eval = bsdf_dielectric_energy_compensated_evaluate(context, data, world_w_o, material, sampler);
  if (bsdf_eval_valid(eval) == false) {
    return bsdf_sample_zero(data.spectrum_sample);
  }
  if ((isfinite(eval.pdf) == false) || (eval.pdf <= kEpsilon)) {
    return bsdf_sample_zero(data.spectrum_sample);
  }

  BSDFSample result = ETX_ZERO(BSDFSample);
  result.w_o = world_w_o;
  result.pdf = eval.pdf;
  result.weight = spectral_response_div(eval.bsdf, eval.pdf);
  if (bsdf_energy_compensated_spectral_response_finite(result.weight) == false) {
    return bsdf_sample_zero(data.spectrum_sample);
  }
  const bool reflection = (w_i_local.z * local_w_o.z) > 0.0f;
  result.eta = reflection ? 1.0f : bsdf_energy_compensated_dielectric_continuation_eta(ext_ior, int_ior, outside);
  result.properties = reflection ? BSDFSample::Reflection : (BSDFSample::Transmission | BSDFSample::MediumChanged);
  result.medium_index = reflection ? data.current_medium : (outside ? material.int_medium : material.ext_medium);
  return result;
}

ETX_SHARED_INLINE float bsdf_dielectric_energy_compensated_pdf(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(float3, outgoing_direction),
  ETX_IN(Material, material), ETX_INOUT(Sampler, sampler)) {
  (void)sampler;
  if (bsdf_energy_compensated_material_supported(context, material, data.tex) == false) {
    return 0.0f;
  }
  if (bsdf_dielectric_is_delta_with_context(context, material, data.tex)) {
    return 0.0f;
  }
  if (bsdf_energy_compensated_material_interface_valid(context, material, MaterialClass::Dielectric) == false) {
    return 0.0f;
  }

  LocalFrame frame = ETX_ZERO(LocalFrame);
  frame.tan = data.tan;
  frame.btn = data.btn;
  frame.nrm = data.nrm;
  const float3 w_i = local_frame_to_local(frame, -data.w_i);
  const float3 w_o = local_frame_to_local(frame, outgoing_direction);
  const float alpha = bsdf_energy_compensated_scalar_roughness(context, material, data.tex);
  const RefractiveIndexSample ext_ior = bsdf_resource_evaluate_refractive_index(context, material.ext_ior, data.spectrum_sample);
  const RefractiveIndexSample int_ior = bsdf_resource_evaluate_refractive_index(context, material.int_ior, data.spectrum_sample);
  return bsdf_energy_compensated_dielectric_pdf_local(context, data.spectrum_sample, material, w_i, w_o, alpha, ext_ior, int_ior);
}

ETX_SHARED_INLINE bool bsdf_dielectric_energy_compensated_is_delta(ETX_IN(Material, material), ETX_IN(float2, tex), ETX_INOUT(Sampler, sampler)) {
  return bsdf_dielectric_is_delta(material, tex, sampler);
}

ETX_SHARED_INLINE bool bsdf_dielectric_energy_compensated_is_delta_with_context(ETX_IN(BSDFResourceContext, context), ETX_IN(Material, material), ETX_IN(float2, tex)) {
  return bsdf_dielectric_is_delta_with_context(context, material, tex);
}

ETX_SHARED_INLINE SpectralResponse bsdf_dielectric_energy_compensated_albedo(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(Material, material),
  ETX_INOUT(Sampler, sampler)) {
  return bsdf_dielectric_albedo(context, data, material, sampler);
}

#endif
