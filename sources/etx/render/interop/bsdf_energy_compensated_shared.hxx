#pragma once

#if (ETX_CPP)

#include "bsdf_conductor_shared.hxx"
#include "bsdf_dielectric_shared.hxx"
#include "image_filter_shared.hxx"

ETX_STATIC_CONST uint32_t kBSDFEnergyCompensationConductorLutSize = 64u;
ETX_STATIC_CONST uint32_t kBSDFEnergyCompensationDielectricLutSize = 64u;
ETX_STATIC_CONST uint32_t kBSDFEnergyCompensationCacheModeIntegratedRGB = 0u;
ETX_STATIC_CONST uint32_t kBSDFEnergyCompensationCacheModeSpectralScalar = 1u;
ETX_STATIC_CONST uint32_t kBSDFEnergyCompensationSpectralWavelengthCount = 128u;
ETX_STATIC_CONST uint32_t kBSDFEnergyCompensationSpectralWavelengthGroupSize = 4u;
ETX_STATIC_CONST uint32_t kBSDFEnergyCompensationSpectralWavelengthGroupCount =
  kBSDFEnergyCompensationSpectralWavelengthCount / kBSDFEnergyCompensationSpectralWavelengthGroupSize;
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
  float thinfilm_lut_value ETX_INIT(0.0f);
};

struct BSDFEnergyCompensatedDielectricBranchPair {
  float4 outside_value ETX_INIT({});
  float4 inside_value ETX_INIT({});
};

struct BSDFEnergyCompensatedPreparedMaterial {
  float2 roughness ETX_INIT({});
  float alpha ETX_INIT(kBSDFNormalDistributionMinAlpha);
  RefractiveIndexSample ext_ior ETX_INIT({});
  RefractiveIndexSample int_ior ETX_INIT({});
  ThinfilmEval thinfilm ETX_INIT({});
  float thinfilm_lut_value ETX_INIT(0.0f);
  bool supported ETX_INIT(false);
  bool conductor_delta ETX_INIT(false);
  bool dielectric_delta ETX_INIT(false);
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
  if ((image_index == kInvalidIndex) || (image_index >= context.scene->images.count)) {
    return false;
  }

  const auto& image = context.scene->images[image_index];
  return (image.isize.x == width) && (image.isize.y == height) && (image.isize.z > 0u) && (image.format == Image::Format::RGBA32F);
}

ETX_SHARED_INLINE float bsdf_energy_compensated_float4_channel(ETX_IN(float4, value), uint32_t channel) {
  if (channel == 0u) {
    return value.x;
  }
  if (channel == 1u) {
    return value.y;
  }
  if (channel == 2u) {
    return value.z;
  }
  return value.w;
}

ETX_SHARED_INLINE float4 bsdf_energy_compensated_sample_image(ETX_IN(BSDFResourceContext, context), uint32_t image_index, ETX_IN(float2, uv), uint32_t expected_width,
  uint32_t expected_height, float thinfilm_lut_value) {
  if (bsdf_energy_compensated_image_has_size(context, image_index, expected_width, expected_height) == false) {
    return float4(0.0f, 0.0f, 0.0f, 0.0f);
  }

  const auto& image = context.scene->images[image_index];
  if (image.isize.z <= 1u) {
    return image.evaluate_rgba32f_fast(uv);
  }

  const float w = bsdf_energy_compensated_lut_uv(thinfilm_lut_value, image.isize.z);
  return image.evaluate_rgba32f_fast_3d(float3{uv.x, uv.y, w});
}

template <typename ImageType>
ETX_SHARED_INLINE float4 bsdf_energy_compensated_sample_image_layer(ETX_IN(ImageType, image), ETX_IN(float2, uv), uint32_t layer) {
  ETX_ASSERT(image.format == Image::Format::RGBA32F);
  ETX_ASSERT(image.pixels.f32.a != nullptr);
  ETX_ASSERT((image.isize.x > 0u) && (image.isize.y > 0u) && (image.isize.z > 0u));

  const float2 image_uv = uv * float2{image.fsize.x, image.fsize.y};
  const float x0 = image.tex_coord_u(image_uv.x, image.fsize.x);
  const float y0 = image.tex_coord_v(image_uv.y, image.fsize.y);
  const float dx = x0 - floorf(x0);
  const float dy = y0 - floorf(y0);

  const uint32_t slice = min(layer, image.isize.z - 1u);
  const uint32_t row_0 = clamp(static_cast<uint32_t>(y0), 0u, image.isize.y - 1u);
  const uint32_t row_1 = image.next_coord_v(row_0);
  const uint32_t col_0 = clamp(static_cast<uint32_t>(x0), 0u, image.isize.x - 1u);
  const uint32_t col_1 = image.next_coord_u(col_0);

  const uint32_t row_stride = image.isize.x;
  const uint32_t slice_offset = slice * image.isize.x * image.isize.y;
  const uint32_t row_offset_0 = row_0 * row_stride;
  const uint32_t row_offset_1 = row_1 * row_stride;
  const float4 p00 = image.pixels.f32.a[slice_offset + row_offset_0 + col_0];
  const float4 p01 = image.pixels.f32.a[slice_offset + row_offset_0 + col_1];
  const float4 p10 = image.pixels.f32.a[slice_offset + row_offset_1 + col_0];
  const float4 p11 = image.pixels.f32.a[slice_offset + row_offset_1 + col_1];
  return image_filter_shared_bilinear(p00, p01, p10, p11, dx, dy);
}

ETX_SHARED_INLINE float bsdf_energy_compensated_sample_spectral_scalar_image(ETX_IN(BSDFResourceContext, context), uint32_t image_index, ETX_IN(SpectralQuery, spect),
  ETX_IN(float2, uv), uint32_t expected_width, uint32_t expected_height, float thinfilm_lut_value,
  uint32_t cache_mode, uint32_t spectral_wavelength_count, uint32_t thinfilm_slice_count, float spectral_shortest_wavelength, float spectral_longest_wavelength) {
  if (bsdf_energy_compensated_image_has_size(context, image_index, expected_width, expected_height) == false) {
    return 0.0f;
  }

  if ((cache_mode != kBSDFEnergyCompensationCacheModeSpectralScalar) || (spectral_wavelength_count != kBSDFEnergyCompensationSpectralWavelengthCount) ||
      (thinfilm_slice_count == 0u)) {
    return 0.0f;
  }

  const auto& image = context.scene->images[image_index];
  const uint32_t expected_depth = thinfilm_slice_count * kBSDFEnergyCompensationSpectralWavelengthGroupCount;
  if (image.isize.z != expected_depth) {
    return 0.0f;
  }

  const float wavelength_range = max(kEpsilon, spectral_longest_wavelength - spectral_shortest_wavelength);
  const float wavelength_axis = bsdf_energy_compensated_saturate((spect.wavelength - spectral_shortest_wavelength) / wavelength_range) *
                                static_cast<float>(kBSDFEnergyCompensationSpectralWavelengthCount - 1u);
  const uint32_t wavelength_index_0 = min(static_cast<uint32_t>(wavelength_axis), kBSDFEnergyCompensationSpectralWavelengthCount - 1u);
  const uint32_t wavelength_index_1 = min(wavelength_index_0 + 1u, kBSDFEnergyCompensationSpectralWavelengthCount - 1u);
  const float wavelength_t = wavelength_axis - floorf(wavelength_axis);

  const float thinfilm_axis = bsdf_energy_compensated_saturate(thinfilm_lut_value) * static_cast<float>(thinfilm_slice_count - 1u);
  const uint32_t thinfilm_index_0 = min(static_cast<uint32_t>(thinfilm_axis), thinfilm_slice_count - 1u);
  const uint32_t thinfilm_index_1 = min(thinfilm_index_0 + 1u, thinfilm_slice_count - 1u);
  const float thinfilm_t = thinfilm_axis - floorf(thinfilm_axis);

  const uint32_t group_0 = wavelength_index_0 / kBSDFEnergyCompensationSpectralWavelengthGroupSize;
  const uint32_t group_1 = wavelength_index_1 / kBSDFEnergyCompensationSpectralWavelengthGroupSize;
  const uint32_t channel_0 = wavelength_index_0 - group_0 * kBSDFEnergyCompensationSpectralWavelengthGroupSize;
  const uint32_t channel_1 = wavelength_index_1 - group_1 * kBSDFEnergyCompensationSpectralWavelengthGroupSize;

  const uint32_t layer_00 = thinfilm_index_0 * kBSDFEnergyCompensationSpectralWavelengthGroupCount + group_0;
  const uint32_t layer_01 = thinfilm_index_0 * kBSDFEnergyCompensationSpectralWavelengthGroupCount + group_1;
  const uint32_t layer_10 = thinfilm_index_1 * kBSDFEnergyCompensationSpectralWavelengthGroupCount + group_0;
  const uint32_t layer_11 = thinfilm_index_1 * kBSDFEnergyCompensationSpectralWavelengthGroupCount + group_1;

  const float value_00 = bsdf_energy_compensated_float4_channel(bsdf_energy_compensated_sample_image_layer(image, uv, layer_00), channel_0);
  const float value_01 = bsdf_energy_compensated_float4_channel(bsdf_energy_compensated_sample_image_layer(image, uv, layer_01), channel_1);
  const float value_10 = bsdf_energy_compensated_float4_channel(bsdf_energy_compensated_sample_image_layer(image, uv, layer_10), channel_0);
  const float value_11 = bsdf_energy_compensated_float4_channel(bsdf_energy_compensated_sample_image_layer(image, uv, layer_11), channel_1);

  const float value_0 = bsdf_energy_compensated_lerp(value_00, value_01, wavelength_t);
  const float value_1 = bsdf_energy_compensated_lerp(value_10, value_11, wavelength_t);
  return bsdf_energy_compensated_lerp(value_0, value_1, thinfilm_t);
}

ETX_SHARED_INLINE SpectralResponse bsdf_energy_compensated_lut_response(ETX_IN(BSDFResourceContext, context), ETX_IN(SpectralQuery, spect), ETX_IN(Material, material),
  uint32_t image_index, ETX_IN(float2, uv), uint32_t expected_width, uint32_t expected_height, float thinfilm_lut_value) {
  if ((material.energy_compensation_interface_index == kInvalidIndex) ||
      (material.energy_compensation_interface_index >= context.scene->energy_compensation_interfaces.count)) {
    return spectral_response_make(spect, 0.0f);
  }

  const auto& interface_data = context.scene->energy_compensation_interfaces[material.energy_compensation_interface_index];
  if (spectral_query_is_spectral(spect) && (interface_data.cache_mode == kBSDFEnergyCompensationCacheModeSpectralScalar)) {
    const float value =
      bsdf_energy_compensated_sample_spectral_scalar_image(context, image_index, spect, uv, expected_width, expected_height, thinfilm_lut_value, interface_data.cache_mode,
        interface_data.spectral_wavelength_count, interface_data.thinfilm_slice_count, interface_data.spectral_shortest_wavelength, interface_data.spectral_longest_wavelength);
    return spectral_response_make(spect, bsdf_energy_compensated_saturate(value));
  }

  const float4 value = bsdf_energy_compensated_sample_image(context, image_index, uv, expected_width, expected_height, thinfilm_lut_value);
  return spectral_rgb_response(spect, saturate(float3(value.x, value.y, value.z)));
}

ETX_SHARED_INLINE float bsdf_energy_compensated_lut_scalar(ETX_IN(BSDFResourceContext, context), ETX_IN(SpectralQuery, spect), ETX_IN(Material, material),
  uint32_t image_index, ETX_IN(float2, uv), uint32_t expected_width, uint32_t expected_height, float thinfilm_lut_value, uint32_t integrated_channel) {
  if ((material.energy_compensation_interface_index == kInvalidIndex) ||
      (material.energy_compensation_interface_index >= context.scene->energy_compensation_interfaces.count)) {
    return 0.0f;
  }

  const auto& interface_data = context.scene->energy_compensation_interfaces[material.energy_compensation_interface_index];
  if (spectral_query_is_spectral(spect) && (interface_data.cache_mode == kBSDFEnergyCompensationCacheModeSpectralScalar)) {
    const float value =
      bsdf_energy_compensated_sample_spectral_scalar_image(context, image_index, spect, uv, expected_width, expected_height, thinfilm_lut_value, interface_data.cache_mode,
        interface_data.spectral_wavelength_count, interface_data.thinfilm_slice_count, interface_data.spectral_shortest_wavelength, interface_data.spectral_longest_wavelength);
    return value;
  }

  const float4 value = bsdf_energy_compensated_sample_image(context, image_index, uv, expected_width, expected_height, thinfilm_lut_value);
  return bsdf_energy_compensated_float4_channel(value, integrated_channel);
}

ETX_SHARED_INLINE float bsdf_energy_compensated_thinfilm_lut_value(ETX_IN(Material, material), ETX_IN(ThinfilmEval, thinfilm)) {
  const float thickness_delta = material.thinfilm.max_thickness - material.thinfilm.min_thickness;
  if ((bsdf_resource_thinfilm_enabled(material.thinfilm) == false) || (abs(thickness_delta) <= kEpsilon)) {
    return 0.0f;
  }

  return bsdf_energy_compensated_saturate((thinfilm.thickness - material.thinfilm.min_thickness) / thickness_delta);
}

ETX_SHARED_INLINE float bsdf_energy_compensated_scalar_roughness_from_value(ETX_IN(float2, roughness)) {
  const float alpha = 0.5f * (roughness.x + roughness.y);
  return max(kBSDFNormalDistributionMinAlpha, bsdf_energy_compensated_saturate(alpha));
}

ETX_SHARED_INLINE bool bsdf_energy_compensated_roughness_isotropic(ETX_IN(float2, roughness)) {
  const float roughness_scale = max(1.0f, max(abs(roughness.x), abs(roughness.y)));
  const float tolerance = 16.0f * kEpsilon * roughness_scale;
  return abs(roughness.x - roughness.y) <= tolerance;
}

ETX_SHARED_INLINE bool bsdf_energy_compensated_material_interface_valid(ETX_IN(BSDFResourceContext, context), ETX_IN(Material, material), uint32_t material_class) {
  if ((material.energy_compensation_interface_index == kInvalidIndex) ||
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

ETX_SHARED_INLINE uint32_t bsdf_energy_compensated_dielectric_probability_lut_index(ETX_IN(BSDFResourceContext, context), ETX_IN(Material, material)) {
  if (bsdf_energy_compensated_material_interface_valid(context, material, MaterialClass::Dielectric)) {
    return context.scene->energy_compensation_interfaces[material.energy_compensation_interface_index].probability_lut;
  }

  return kInvalidIndex;
}

ETX_SHARED_INLINE float bsdf_energy_compensated_scalar_roughness(ETX_IN(BSDFResourceContext, context), ETX_IN(Material, material), ETX_IN(float2, uv)) {
  const float2 roughness = bsdf_resource_evaluate_roughness(context, material, uv);
  return bsdf_energy_compensated_scalar_roughness_from_value(roughness);
}

ETX_SHARED_INLINE bool bsdf_energy_compensated_material_supported(ETX_IN(BSDFResourceContext, context), ETX_IN(Material, material), ETX_IN(float2, uv)) {
  const float2 roughness = bsdf_resource_evaluate_roughness(context, material, uv);
  return bsdf_energy_compensated_roughness_isotropic(roughness);
}

ETX_SHARED_INLINE BSDFEnergyCompensatedPreparedMaterial bsdf_energy_compensated_prepare_material(ETX_IN(BSDFResourceContext, context), ETX_IN(SpectralQuery, spect),
  ETX_IN(Material, material), ETX_IN(float2, uv), ETX_INOUT(Sampler, sampler)) {
  BSDFEnergyCompensatedPreparedMaterial result = ETX_ZERO(BSDFEnergyCompensatedPreparedMaterial);
  result.roughness = bsdf_resource_evaluate_roughness(context, material, uv);
  result.alpha = bsdf_energy_compensated_scalar_roughness_from_value(result.roughness);
  result.supported = bsdf_energy_compensated_roughness_isotropic(result.roughness);
  result.ext_ior = bsdf_resource_evaluate_refractive_index(context, material.ext_ior, spect);
  result.int_ior = bsdf_resource_evaluate_refractive_index(context, material.int_ior, spect);
  result.thinfilm = bsdf_resource_evaluate_thinfilm(context, spect, material.thinfilm, uv, sampler);
  result.thinfilm_lut_value = bsdf_energy_compensated_thinfilm_lut_value(material, result.thinfilm);
  const float max_roughness = max(result.roughness.x, result.roughness.y);
  result.conductor_delta = max_roughness <= kDeltaAlphaTreshold;
  result.dielectric_delta = (max_roughness <= kDeltaAlphaTreshold) ||
                            ((bsdf_resource_thinfilm_enabled(material.thinfilm) == false) && (bsdf_dielectric_equal_eta(result.ext_ior, result.int_ior)));
  return result;
}

ETX_SHARED_INLINE SpectralResponse bsdf_energy_compensated_conductor_directional_albedo(ETX_IN(BSDFResourceContext, context), ETX_IN(SpectralQuery, spect),
  ETX_IN(Material, material), float mu, float alpha, float thinfilm_lut_value) {
  const uint32_t image_index = bsdf_energy_compensated_conductor_lut_index(context, material);
  const float2 uv = float2(bsdf_energy_compensated_lut_uv(mu, kBSDFEnergyCompensationConductorLutSize),
    bsdf_energy_compensated_lut_uv(bsdf_energy_compensated_conductor_alpha_axis(context, material, alpha), kBSDFEnergyCompensationConductorLutSize));
  return bsdf_energy_compensated_lut_response(context, spect, material, image_index, uv, kBSDFEnergyCompensationConductorLutSize,
    kBSDFEnergyCompensationConductorLutSize, thinfilm_lut_value);
}

ETX_SHARED_INLINE SpectralResponse bsdf_energy_compensated_conductor_directional_albedo(ETX_IN(BSDFResourceContext, context), ETX_IN(SpectralQuery, spect),
  ETX_IN(Material, material), float mu, float alpha) {
  return bsdf_energy_compensated_conductor_directional_albedo(context, spect, material, mu, alpha, 0.0f);
}

ETX_SHARED_INLINE float bsdf_energy_compensated_conductor_visible_probability(ETX_IN(BSDFResourceContext, context), ETX_IN(Material, material), float mu, float alpha,
  float thinfilm_lut_value) {
  if ((material.energy_compensation_interface_index != kInvalidIndex) &&
      (material.energy_compensation_interface_index < context.scene->energy_compensation_interfaces.count)) {
    const auto& interface_data = context.scene->energy_compensation_interfaces[material.energy_compensation_interface_index];
    if ((interface_data.cls == MaterialClass::Conductor) && (interface_data.geometric_lut != kInvalidIndex)) {
      const float2 uv = float2(bsdf_energy_compensated_lut_uv(mu, kBSDFEnergyCompensationConductorLutSize),
        bsdf_energy_compensated_lut_uv(bsdf_energy_compensated_conductor_alpha_axis(context, material, alpha), kBSDFEnergyCompensationConductorLutSize));
      const float4 value = bsdf_energy_compensated_sample_image(context, interface_data.geometric_lut, uv, kBSDFEnergyCompensationConductorLutSize,
        kBSDFEnergyCompensationConductorLutSize, thinfilm_lut_value);
      return bsdf_energy_compensated_saturate(value.y);
    }
  }

  return 0.0f;
}

ETX_SHARED_INLINE float bsdf_energy_compensated_conductor_visible_probability(ETX_IN(BSDFResourceContext, context), ETX_IN(Material, material), float mu, float alpha) {
  return bsdf_energy_compensated_conductor_visible_probability(context, material, mu, alpha, 0.0f);
}

ETX_SHARED_INLINE SpectralResponse bsdf_energy_compensated_conductor_average_albedo(ETX_IN(BSDFResourceContext, context), ETX_IN(SpectralQuery, spect),
  ETX_IN(Material, material), float alpha, float thinfilm_lut_value) {
  const uint32_t image_index = bsdf_energy_compensated_conductor_average_lut_index(context, material);
  const float2 uv = float2(bsdf_energy_compensated_lut_uv(bsdf_energy_compensated_conductor_alpha_axis(context, material, alpha), kBSDFEnergyCompensationConductorLutSize), 0.0f);
  return bsdf_energy_compensated_lut_response(context, spect, material, image_index, uv, kBSDFEnergyCompensationConductorLutSize, 1u, thinfilm_lut_value);
}

ETX_SHARED_INLINE SpectralResponse bsdf_energy_compensated_conductor_average_albedo(ETX_IN(BSDFResourceContext, context), ETX_IN(SpectralQuery, spect),
  ETX_IN(Material, material), float alpha) {
  return bsdf_energy_compensated_conductor_average_albedo(context, spect, material, alpha, 0.0f);
}

ETX_SHARED_INLINE float bsdf_energy_compensated_conductor_geometric_directional_albedo(ETX_IN(BSDFResourceContext, context), ETX_IN(Material, material), float mu, float alpha,
  float thinfilm_lut_value) {
  uint32_t image_index = kInvalidIndex;
  if ((material.energy_compensation_interface_index != kInvalidIndex) &&
      (material.energy_compensation_interface_index < context.scene->energy_compensation_interfaces.count)) {
    const auto& interface_data = context.scene->energy_compensation_interfaces[material.energy_compensation_interface_index];
    if ((interface_data.cls == MaterialClass::Conductor) && (interface_data.geometric_lut != kInvalidIndex)) {
      image_index = interface_data.geometric_lut;
    }
  }
  const float2 uv = float2(bsdf_energy_compensated_lut_uv(mu, kBSDFEnergyCompensationConductorLutSize),
    bsdf_energy_compensated_lut_uv(bsdf_energy_compensated_conductor_alpha_axis(context, material, alpha), kBSDFEnergyCompensationConductorLutSize));
  const float4 value =
    bsdf_energy_compensated_sample_image(context, image_index, uv, kBSDFEnergyCompensationConductorLutSize, kBSDFEnergyCompensationConductorLutSize, thinfilm_lut_value);
  return bsdf_energy_compensated_saturate(value.x);
}

ETX_SHARED_INLINE float bsdf_energy_compensated_conductor_geometric_directional_albedo(ETX_IN(BSDFResourceContext, context), ETX_IN(Material, material), float mu, float alpha) {
  return bsdf_energy_compensated_conductor_geometric_directional_albedo(context, material, mu, alpha, 0.0f);
}

ETX_SHARED_INLINE float bsdf_energy_compensated_conductor_geometric_average_albedo(ETX_IN(BSDFResourceContext, context), ETX_IN(Material, material), float alpha,
  float thinfilm_lut_value) {
  uint32_t image_index = kInvalidIndex;
  if ((material.energy_compensation_interface_index != kInvalidIndex) &&
      (material.energy_compensation_interface_index < context.scene->energy_compensation_interfaces.count)) {
    const auto& interface_data = context.scene->energy_compensation_interfaces[material.energy_compensation_interface_index];
    if ((interface_data.cls == MaterialClass::Conductor) && (interface_data.geometric_average_lut != kInvalidIndex)) {
      image_index = interface_data.geometric_average_lut;
    }
  }
  const float2 uv = float2(bsdf_energy_compensated_lut_uv(bsdf_energy_compensated_conductor_alpha_axis(context, material, alpha), kBSDFEnergyCompensationConductorLutSize), 0.0f);
  const float4 value = bsdf_energy_compensated_sample_image(context, image_index, uv, kBSDFEnergyCompensationConductorLutSize, 1u, thinfilm_lut_value);
  return bsdf_energy_compensated_saturate(value.x);
}

ETX_SHARED_INLINE float bsdf_energy_compensated_conductor_geometric_average_albedo(ETX_IN(BSDFResourceContext, context), ETX_IN(Material, material), float alpha) {
  return bsdf_energy_compensated_conductor_geometric_average_albedo(context, material, alpha, 0.0f);
}

ETX_SHARED_INLINE SpectralResponse bsdf_energy_compensated_conductor_cached_fms(ETX_IN(BSDFResourceContext, context), ETX_IN(SpectralQuery, spect), ETX_IN(Material, material),
  float alpha, float thinfilm_lut_value) {
  uint32_t image_index = kInvalidIndex;
  if ((material.energy_compensation_interface_index != kInvalidIndex) &&
      (material.energy_compensation_interface_index < context.scene->energy_compensation_interfaces.count)) {
    const auto& interface_data = context.scene->energy_compensation_interfaces[material.energy_compensation_interface_index];
    if ((interface_data.cls == MaterialClass::Conductor) && (interface_data.conductor_fms_lut != kInvalidIndex)) {
      image_index = interface_data.conductor_fms_lut;
    }
  }

  const float2 uv = float2(bsdf_energy_compensated_lut_uv(bsdf_energy_compensated_conductor_alpha_axis(context, material, alpha), kBSDFEnergyCompensationConductorLutSize), 0.0f);
  return bsdf_energy_compensated_lut_response(context, spect, material, image_index, uv, kBSDFEnergyCompensationConductorLutSize, 1u, thinfilm_lut_value);
}

ETX_SHARED_INLINE SpectralResponse bsdf_energy_compensated_conductor_cached_fms(ETX_IN(BSDFResourceContext, context), ETX_IN(SpectralQuery, spect), ETX_IN(Material, material),
  float alpha) {
  return bsdf_energy_compensated_conductor_cached_fms(context, spect, material, alpha, 0.0f);
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

ETX_SHARED_INLINE float4 bsdf_energy_compensated_dielectric_branch_value(ETX_IN(BSDFResourceContext, context), ETX_IN(SpectralQuery, spect), ETX_IN(Material, material),
  float mu, float alpha, bool incident_outside, bool outgoing_outside, float thinfilm_lut_value) {
  const uint32_t image_index = bsdf_energy_compensated_dielectric_lut_index(context, material);
  const uint32_t expected_width = 4u * kBSDFEnergyCompensationDielectricLutSize;
  const uint32_t branch_offset = bsdf_energy_compensated_dielectric_branch_index(incident_outside, outgoing_outside) * kBSDFEnergyCompensationDielectricLutSize;
  const float x = (static_cast<float>(branch_offset) + bsdf_energy_compensated_saturate(mu) * static_cast<float>(kBSDFEnergyCompensationDielectricLutSize - 1u)) /
                  static_cast<float>(expected_width);
  const float y = bsdf_energy_compensated_lut_uv(bsdf_energy_compensated_dielectric_alpha_axis(alpha), kBSDFEnergyCompensationDielectricLutSize);
  const float2 uv = float2(x, y);
  if (spectral_query_is_spectral(spect)) {
    const uint32_t probability_image_index = bsdf_energy_compensated_dielectric_probability_lut_index(context, material);
    const float albedo = bsdf_energy_compensated_saturate(
      bsdf_energy_compensated_lut_scalar(context, spect, material, image_index, uv, expected_width, kBSDFEnergyCompensationDielectricLutSize, thinfilm_lut_value, 0u));
    const float visible_probability = bsdf_energy_compensated_saturate(bsdf_energy_compensated_lut_scalar(context, spect, material, probability_image_index, uv, expected_width,
      kBSDFEnergyCompensationDielectricLutSize, thinfilm_lut_value, 3u));
    return float4(albedo, albedo, albedo, visible_probability);
  }

  return bsdf_energy_compensated_sample_image(context, image_index, uv, expected_width, kBSDFEnergyCompensationDielectricLutSize, thinfilm_lut_value);
}

ETX_SHARED_INLINE BSDFEnergyCompensatedDielectricBranchPair bsdf_energy_compensated_dielectric_branch_pair_value(ETX_IN(BSDFResourceContext, context),
  ETX_IN(SpectralQuery, spect), ETX_IN(Material, material), float mu, float alpha, bool incident_outside, float thinfilm_lut_value) {
  BSDFEnergyCompensatedDielectricBranchPair result = ETX_ZERO(BSDFEnergyCompensatedDielectricBranchPair);
  const uint32_t image_index = bsdf_energy_compensated_dielectric_lut_index(context, material);
  const uint32_t expected_width = 4u * kBSDFEnergyCompensationDielectricLutSize;
  const uint32_t branch_0_offset = bsdf_energy_compensated_dielectric_branch_index(incident_outside, true) * kBSDFEnergyCompensationDielectricLutSize;
  const uint32_t branch_1_offset = bsdf_energy_compensated_dielectric_branch_index(incident_outside, false) * kBSDFEnergyCompensationDielectricLutSize;
  const float mu_axis = bsdf_energy_compensated_saturate(mu) * static_cast<float>(kBSDFEnergyCompensationDielectricLutSize - 1u);
  const float y = bsdf_energy_compensated_lut_uv(bsdf_energy_compensated_dielectric_alpha_axis(alpha), kBSDFEnergyCompensationDielectricLutSize);
  const float x_0 = (static_cast<float>(branch_0_offset) + mu_axis) / static_cast<float>(expected_width);
  const float x_1 = (static_cast<float>(branch_1_offset) + mu_axis) / static_cast<float>(expected_width);
  if (spectral_query_is_spectral(spect)) {
    const uint32_t probability_image_index = bsdf_energy_compensated_dielectric_probability_lut_index(context, material);
    const float outside_albedo = bsdf_energy_compensated_saturate(
      bsdf_energy_compensated_lut_scalar(context, spect, material, image_index, float2(x_0, y), expected_width, kBSDFEnergyCompensationDielectricLutSize, thinfilm_lut_value, 0u));
    const float inside_albedo = bsdf_energy_compensated_saturate(
      bsdf_energy_compensated_lut_scalar(context, spect, material, image_index, float2(x_1, y), expected_width, kBSDFEnergyCompensationDielectricLutSize, thinfilm_lut_value, 0u));
    const float outside_probability = bsdf_energy_compensated_saturate(bsdf_energy_compensated_lut_scalar(context, spect, material, probability_image_index, float2(x_0, y),
      expected_width, kBSDFEnergyCompensationDielectricLutSize, thinfilm_lut_value, 3u));
    const float inside_probability = bsdf_energy_compensated_saturate(bsdf_energy_compensated_lut_scalar(context, spect, material, probability_image_index, float2(x_1, y),
      expected_width, kBSDFEnergyCompensationDielectricLutSize, thinfilm_lut_value, 3u));
    result.outside_value = float4(outside_albedo, outside_albedo, outside_albedo, outside_probability);
    result.inside_value = float4(inside_albedo, inside_albedo, inside_albedo, inside_probability);
    return result;
  }

  result.outside_value = bsdf_energy_compensated_sample_image(context, image_index, float2(x_0, y), expected_width, kBSDFEnergyCompensationDielectricLutSize, thinfilm_lut_value);
  result.inside_value = bsdf_energy_compensated_sample_image(context, image_index, float2(x_1, y), expected_width, kBSDFEnergyCompensationDielectricLutSize, thinfilm_lut_value);
  return result;
}

ETX_SHARED_INLINE BSDFEnergyCompensatedDielectricBranchPair bsdf_energy_compensated_dielectric_branch_pair_value(ETX_IN(BSDFResourceContext, context),
  ETX_IN(SpectralQuery, spect), ETX_IN(Material, material), float mu, float alpha, bool incident_outside) {
  return bsdf_energy_compensated_dielectric_branch_pair_value(context, spect, material, mu, alpha, incident_outside, 0.0f);
}

ETX_SHARED_INLINE SpectralResponse bsdf_energy_compensated_dielectric_branch_pair_albedo(ETX_IN(SpectralQuery, spect),
  ETX_IN(BSDFEnergyCompensatedDielectricBranchPair, pair)) {
  const float3 outside_albedo = saturate(float3(pair.outside_value.x, pair.outside_value.y, pair.outside_value.z));
  const float3 inside_albedo = saturate(float3(pair.inside_value.x, pair.inside_value.y, pair.inside_value.z));
  if (spectral_query_is_spectral(spect)) {
    return spectral_response_make(spect, min(pair.outside_value.x + pair.inside_value.x, 1.0f));
  }
  return spectral_response_make(spect, min(outside_albedo + inside_albedo, float3(1.0f, 1.0f, 1.0f)));
}

ETX_SHARED_INLINE float bsdf_energy_compensated_dielectric_branch_pair_visible_probability(ETX_IN(BSDFEnergyCompensatedDielectricBranchPair, pair)) {
  return bsdf_energy_compensated_saturate(pair.outside_value.w + pair.inside_value.w);
}

ETX_SHARED_INLINE SpectralResponse bsdf_energy_compensated_dielectric_branch_pair_selected_albedo(ETX_IN(SpectralQuery, spect),
  ETX_IN(BSDFEnergyCompensatedDielectricBranchPair, pair), bool outgoing_outside) {
  const float4 value = outgoing_outside ? pair.outside_value : pair.inside_value;
  if (spectral_query_is_spectral(spect)) {
    return spectral_response_make(spect, bsdf_energy_compensated_saturate(value.x));
  }
  return spectral_response_make(spect, saturate(float3(value.x, value.y, value.z)));
}

ETX_SHARED_INLINE float bsdf_energy_compensated_dielectric_branch_pair_selected_visible_probability(ETX_IN(BSDFEnergyCompensatedDielectricBranchPair, pair),
  bool outgoing_outside) {
  const float4 value = outgoing_outside ? pair.outside_value : pair.inside_value;
  return bsdf_energy_compensated_saturate(value.w);
}

ETX_SHARED_INLINE SpectralResponse bsdf_energy_compensated_dielectric_branch_albedo(ETX_IN(BSDFResourceContext, context), ETX_IN(SpectralQuery, spect),
  ETX_IN(Material, material), float mu, float alpha, bool incident_outside, bool outgoing_outside, float thinfilm_lut_value) {
  const float4 value = bsdf_energy_compensated_dielectric_branch_value(context, spect, material, mu, alpha, incident_outside, outgoing_outside, thinfilm_lut_value);
  if (spectral_query_is_spectral(spect)) {
    return spectral_response_make(spect, bsdf_energy_compensated_saturate(value.x));
  }
  return spectral_response_make(spect, saturate(float3(value.x, value.y, value.z)));
}

ETX_SHARED_INLINE float bsdf_energy_compensated_dielectric_branch_visible_probability(ETX_IN(BSDFResourceContext, context), ETX_IN(SpectralQuery, spect),
  ETX_IN(Material, material), float mu, float alpha, bool incident_outside, bool outgoing_outside, float thinfilm_lut_value) {
  const float4 value = bsdf_energy_compensated_dielectric_branch_value(context, spect, material, mu, alpha, incident_outside, outgoing_outside, thinfilm_lut_value);
  return bsdf_energy_compensated_saturate(value.w);
}

ETX_SHARED_INLINE SpectralResponse bsdf_energy_compensated_dielectric_directional_albedo(ETX_IN(BSDFResourceContext, context), ETX_IN(SpectralQuery, spect),
  ETX_IN(Material, material), float mu, float alpha, float f0, bool outside, bool low_to_high, float thinfilm_lut_value) {
  (void)f0;
  (void)low_to_high;
  const BSDFEnergyCompensatedDielectricBranchPair pair = bsdf_energy_compensated_dielectric_branch_pair_value(context, spect, material, mu, alpha, outside, thinfilm_lut_value);
  return bsdf_energy_compensated_dielectric_branch_pair_albedo(spect, pair);
}

ETX_SHARED_INLINE float bsdf_energy_compensated_dielectric_visible_probability(ETX_IN(BSDFResourceContext, context), ETX_IN(SpectralQuery, spect), ETX_IN(Material, material),
  float mu, float alpha, float f0, bool outside, bool low_to_high, float thinfilm_lut_value) {
  (void)f0;
  (void)low_to_high;
  const BSDFEnergyCompensatedDielectricBranchPair pair = bsdf_energy_compensated_dielectric_branch_pair_value(context, spect, material, mu, alpha, outside, thinfilm_lut_value);
  return bsdf_energy_compensated_dielectric_branch_pair_visible_probability(pair);
}

ETX_SHARED_INLINE float4 bsdf_energy_compensated_dielectric_average_value(ETX_IN(BSDFResourceContext, context), ETX_IN(Material, material), float alpha, uint32_t column,
  float thinfilm_lut_value) {
  const uint32_t image_index = bsdf_energy_compensated_dielectric_average_lut_index(context, material);
  const float x = static_cast<float>(column) / 8.0f;
  const float y = bsdf_energy_compensated_lut_uv(bsdf_energy_compensated_dielectric_alpha_axis(alpha), kBSDFEnergyCompensationDielectricLutSize);
  return bsdf_energy_compensated_sample_image(context, image_index, float2(x, y), 8u, kBSDFEnergyCompensationDielectricLutSize, thinfilm_lut_value);
}

ETX_SHARED_INLINE SpectralResponse bsdf_energy_compensated_dielectric_branch_average_albedo(ETX_IN(BSDFResourceContext, context), ETX_IN(SpectralQuery, spect),
  ETX_IN(Material, material), float alpha, bool incident_outside, bool outgoing_outside, float thinfilm_lut_value) {
  const uint32_t column = bsdf_energy_compensated_dielectric_branch_index(incident_outside, outgoing_outside);
  const uint32_t image_index = bsdf_energy_compensated_dielectric_average_lut_index(context, material);
  const float x = static_cast<float>(column) / 8.0f;
  const float y = bsdf_energy_compensated_lut_uv(bsdf_energy_compensated_dielectric_alpha_axis(alpha), kBSDFEnergyCompensationDielectricLutSize);
  return bsdf_energy_compensated_lut_response(context, spect, material, image_index, float2(x, y), 8u, kBSDFEnergyCompensationDielectricLutSize, thinfilm_lut_value);
}

ETX_SHARED_INLINE SpectralResponse bsdf_energy_compensated_dielectric_branch_average_albedo(ETX_IN(BSDFResourceContext, context), ETX_IN(SpectralQuery, spect),
  ETX_IN(Material, material), float alpha, bool incident_outside, bool outgoing_outside) {
  return bsdf_energy_compensated_dielectric_branch_average_albedo(context, spect, material, alpha, incident_outside, outgoing_outside, 0.0f);
}

ETX_SHARED_INLINE SpectralResponse bsdf_energy_compensated_dielectric_branch_coefficient(ETX_IN(BSDFResourceContext, context), ETX_IN(SpectralQuery, spect),
  ETX_IN(Material, material), float alpha, bool incident_outside, bool outgoing_outside, float thinfilm_lut_value) {
  const uint32_t column = 4u + bsdf_energy_compensated_dielectric_branch_index(incident_outside, outgoing_outside);
  const uint32_t image_index = bsdf_energy_compensated_dielectric_average_lut_index(context, material);
  const float x = static_cast<float>(column) / 8.0f;
  const float y = bsdf_energy_compensated_lut_uv(bsdf_energy_compensated_dielectric_alpha_axis(alpha), kBSDFEnergyCompensationDielectricLutSize);
  if (spectral_query_is_spectral(spect)) {
    const float value = bsdf_energy_compensated_lut_scalar(context, spect, material, image_index, float2(x, y), 8u, kBSDFEnergyCompensationDielectricLutSize, thinfilm_lut_value, 0u);
    return spectral_response_make(spect, max(value, 0.0f));
  }

  const float4 value = bsdf_energy_compensated_sample_image(context, image_index, float2(x, y), 8u, kBSDFEnergyCompensationDielectricLutSize, thinfilm_lut_value);
  return spectral_response_make(spect, max(float3(value.x, value.y, value.z), float3(0.0f, 0.0f, 0.0f)));
}

ETX_SHARED_INLINE SpectralResponse bsdf_energy_compensated_dielectric_branch_coefficient(ETX_IN(BSDFResourceContext, context), ETX_IN(SpectralQuery, spect),
  ETX_IN(Material, material), float alpha, bool incident_outside, bool outgoing_outside) {
  return bsdf_energy_compensated_dielectric_branch_coefficient(context, spect, material, alpha, incident_outside, outgoing_outside, 0.0f);
}

ETX_SHARED_INLINE SpectralResponse bsdf_energy_compensated_dielectric_average_albedo(ETX_IN(BSDFResourceContext, context), ETX_IN(SpectralQuery, spect),
  ETX_IN(Material, material), float alpha, float f0, bool outside, bool low_to_high, float thinfilm_lut_value) {
  (void)f0;
  (void)low_to_high;
  const SpectralResponse outside_albedo = bsdf_energy_compensated_dielectric_branch_average_albedo(context, spect, material, alpha, outside, true, thinfilm_lut_value);
  const SpectralResponse inside_albedo = bsdf_energy_compensated_dielectric_branch_average_albedo(context, spect, material, alpha, outside, false, thinfilm_lut_value);
  return spectral_response_min(spectral_response_add(outside_albedo, inside_albedo), 1.0f);
}

ETX_SHARED_INLINE SpectralResponse bsdf_energy_compensated_dielectric_average_residual(ETX_IN(BSDFResourceContext, context), ETX_IN(SpectralQuery, spect),
  ETX_IN(Material, material), float alpha, bool outside, float thinfilm_lut_value) {
  const SpectralResponse average_albedo = bsdf_energy_compensated_dielectric_average_albedo(context, spect, material, alpha, 0.0f, outside, true, thinfilm_lut_value);
  return spectral_response_max(spectral_response_sub(spectral_response_make(spect, 1.0f), average_albedo), 0.0f);
}

ETX_SHARED_INLINE SpectralResponse bsdf_energy_compensated_dielectric_average_residual(ETX_IN(BSDFResourceContext, context), ETX_IN(SpectralQuery, spect),
  ETX_IN(Material, material), float alpha, bool outside) {
  return bsdf_energy_compensated_dielectric_average_residual(context, spect, material, alpha, outside, 0.0f);
}

ETX_SHARED_INLINE float bsdf_energy_compensated_dielectric_compensation_branch_probability(ETX_IN(BSDFResourceContext, context), ETX_IN(SpectralQuery, spect),
  ETX_IN(Material, material), float alpha, bool incident_outside, bool outgoing_outside, float thinfilm_lut_value) {
  const SpectralResponse outside_residual = bsdf_energy_compensated_dielectric_average_residual(context, spect, material, alpha, true, thinfilm_lut_value);
  const SpectralResponse inside_residual = bsdf_energy_compensated_dielectric_average_residual(context, spect, material, alpha, false, thinfilm_lut_value);
  const SpectralResponse outside_coefficient = bsdf_energy_compensated_dielectric_branch_coefficient(context, spect, material, alpha, incident_outside, true, thinfilm_lut_value);
  const SpectralResponse inside_coefficient = bsdf_energy_compensated_dielectric_branch_coefficient(context, spect, material, alpha, incident_outside, false, thinfilm_lut_value);
  const float outside_probability = max(0.0f, spectral_response_monochromatic(spectral_response_mul(outside_coefficient, outside_residual)));
  const float inside_probability = max(0.0f, spectral_response_monochromatic(spectral_response_mul(inside_coefficient, inside_residual)));
  const float normalization = outside_probability + inside_probability;
  if (normalization <= kEpsilon) {
    return 0.0f;
  }
  const float probability = outgoing_outside ? outside_probability : inside_probability;
  return bsdf_energy_compensated_saturate(probability / normalization);
}

ETX_SHARED_INLINE float bsdf_energy_compensated_dielectric_compensation_branch_probability(ETX_IN(BSDFResourceContext, context), ETX_IN(SpectralQuery, spect),
  ETX_IN(Material, material), float alpha, bool incident_outside, bool outgoing_outside) {
  return bsdf_energy_compensated_dielectric_compensation_branch_probability(context, spect, material, alpha, incident_outside, outgoing_outside, 0.0f);
}

ETX_SHARED_INLINE ThinfilmEval bsdf_energy_compensated_empty_thinfilm() {
  ThinfilmEval result = ETX_ZERO(ThinfilmEval);
  result.ior.cls = SpectralDistribution::Invalid;
  result.rgb_wavelengths = kRGBWavelengths;
  result.thickness = 0.0f;
  return result;
}

ETX_SHARED_INLINE SpectralResponse bsdf_energy_compensated_average_fresnel(ETX_IN(SpectralQuery, spect), ETX_IN(RefractiveIndexSample, ext_ior),
  ETX_IN(RefractiveIndexSample, int_ior), ETX_IN(ThinfilmEval, thinfilm)) {
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

  SpectralResponse result = spectral_response_make(spect, 0.0f);
  for (uint32_t i = 0u; i < 8u; ++i) {
    const float mu = nodes[i];
    const SpectralResponse fresnel = bsdf_fresnel_calculate(spect, mu, ext_ior, int_ior, thinfilm);
    result = spectral_response_add(result, spectral_response_mul(fresnel, 2.0f * weights[i] * mu));
  }
  return result;
}

ETX_SHARED_INLINE SpectralResponse bsdf_energy_compensated_conductor_fms(ETX_IN(SpectralQuery, spect), ETX_IN(RefractiveIndexSample, ext_ior),
  ETX_IN(RefractiveIndexSample, int_ior), ETX_IN(ThinfilmEval, thinfilm), float average_albedo) {
  const SpectralResponse fresnel_average = bsdf_energy_compensated_average_fresnel(spect, ext_ior, int_ior, thinfilm);
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
  ETX_IN(RefractiveIndexSample, ext_ior), ETX_IN(RefractiveIndexSample, int_ior), ETX_IN(ThinfilmEval, thinfilm), ETX_IN(SpectralResponse, reflectance)) {
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
  ETX_IN(float3, w_i), ETX_IN(float3, w_o), float alpha, float thinfilm_lut_value) {
  if ((w_i.z <= kEpsilon) || (w_o.z <= kEpsilon)) {
    return 0.0f;
  }

  const SpectralResponse e_i_response = bsdf_energy_compensated_conductor_directional_albedo(context, spect, material, w_i.z, alpha, thinfilm_lut_value);
  const float e_i = spectral_response_monochromatic(e_i_response);
  const float visible_probability = bsdf_energy_compensated_conductor_visible_probability(context, material, w_i.z, alpha, thinfilm_lut_value);
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

ETX_SHARED_INLINE float bsdf_energy_compensated_conductor_pdf_local(ETX_IN(BSDFResourceContext, context), ETX_IN(SpectralQuery, spect), ETX_IN(Material, material),
  ETX_IN(float3, w_i), ETX_IN(float3, w_o), float alpha) {
  return bsdf_energy_compensated_conductor_pdf_local(context, spect, material, w_i, w_o, alpha, 0.0f);
}

ETX_SHARED_INLINE float bsdf_energy_compensated_conductor_pdf_from_base_lobe(ETX_IN(BSDFResourceContext, context), ETX_IN(SpectralQuery, spect), ETX_IN(Material, material),
  ETX_IN(float3, w_i), ETX_IN(float3, w_o), float alpha, float base_lobe_pdf, float thinfilm_lut_value) {
  if ((w_i.z <= kEpsilon) || (w_o.z <= kEpsilon)) {
    return 0.0f;
  }

  const SpectralResponse e_i_response = bsdf_energy_compensated_conductor_directional_albedo(context, spect, material, w_i.z, alpha, thinfilm_lut_value);
  const float e_i = spectral_response_monochromatic(e_i_response);
  const float visible_probability = bsdf_energy_compensated_conductor_visible_probability(context, material, w_i.z, alpha, thinfilm_lut_value);
  const float specular_probability = (visible_probability > kEpsilon) ? bsdf_energy_compensated_saturate(e_i) : 0.0f;
  const float specular_pdf = (visible_probability > kEpsilon) ? (base_lobe_pdf / visible_probability) : 0.0f;
  const float compensation_pdf = w_o.z * kInvPi;
  return specular_probability * specular_pdf + (1.0f - specular_probability) * compensation_pdf;
}

ETX_SHARED_INLINE float bsdf_energy_compensated_conductor_pdf_from_base_lobe(ETX_IN(BSDFResourceContext, context), ETX_IN(SpectralQuery, spect), ETX_IN(Material, material),
  ETX_IN(float3, w_i), ETX_IN(float3, w_o), float alpha, float base_lobe_pdf) {
  return bsdf_energy_compensated_conductor_pdf_from_base_lobe(context, spect, material, w_i, w_o, alpha, base_lobe_pdf, 0.0f);
}

ETX_SHARED_INLINE BSDFEval bsdf_conductor_energy_compensated_evaluate_prepared_local(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data),
  ETX_IN(Material, material), ETX_IN(float3, w_i), ETX_IN(float3, w_o), ETX_IN(BSDFEnergyCompensatedPreparedMaterial, prepared),
  ETX_IN(SpectralResponse, reflectance)) {
  if ((w_i.z <= kEpsilon) || (w_o.z <= kEpsilon)) {
    return bsdf_eval_zero(data.spectrum_sample);
  }

  const BSDFEnergyCompensatedLobe base_lobe = bsdf_energy_compensated_conductor_base_lobe(data.spectrum_sample, w_i, w_o, prepared.alpha, prepared.ext_ior, prepared.int_ior,
    prepared.thinfilm, reflectance);

  SpectralResponse compensation_bsdf = spectral_response_make(data.spectrum_sample, 0.0f);
  const float e_i_scalar = bsdf_energy_compensated_conductor_geometric_directional_albedo(context, material, w_i.z, prepared.alpha, prepared.thinfilm_lut_value);
  const float e_o_scalar = bsdf_energy_compensated_conductor_geometric_directional_albedo(context, material, w_o.z, prepared.alpha, prepared.thinfilm_lut_value);
  const float e_average_scalar = bsdf_energy_compensated_conductor_geometric_average_albedo(context, material, prepared.alpha, prepared.thinfilm_lut_value);
  if ((1.0f - e_average_scalar) > kEpsilon) {
    const SpectralResponse f_ms = bsdf_energy_compensated_conductor_cached_fms(context, data.spectrum_sample, material, prepared.alpha, prepared.thinfilm_lut_value);
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

  result.pdf =
    bsdf_energy_compensated_conductor_pdf_from_base_lobe(context, data.spectrum_sample, material, w_i, w_o, prepared.alpha, base_lobe.pdf, prepared.thinfilm_lut_value);
  result.eta = 1.0f;
  result.properties = BSDFSample::Reflection;
  result.medium_index = data.current_medium;
  return result;
}

ETX_SHARED_INLINE BSDFEval bsdf_conductor_energy_compensated_evaluate(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(float3, outgoing_direction),
  ETX_IN(Material, material), ETX_INOUT(Sampler, sampler)) {
  const LocalFrame frame = bsdf_data_get_normal_frame(data, material);
  const float3 w_i = local_frame_to_local(frame, -data.w_i);
  const float3 w_o = local_frame_to_local(frame, outgoing_direction);

  if ((w_i.z <= kEpsilon) || (w_o.z <= kEpsilon)) {
    return bsdf_eval_zero(data.spectrum_sample);
  }

  const BSDFEnergyCompensatedPreparedMaterial prepared = bsdf_energy_compensated_prepare_material(context, data.spectrum_sample, material, data.tex, sampler);

  if (prepared.conductor_delta) {
    const float3 ideal_w_o = bsdf_conductor_delta_reflect(data, material);
    const float3 actual_w_o = normalize(outgoing_direction);
    if (direction_matches(ideal_w_o, actual_w_o, 1.0f) == false) {
      return bsdf_eval_zero(data.spectrum_sample);
    }

    BSDFEval result = ETX_ZERO(BSDFEval);
    result.bsdf = bsdf_conductor_delta_weight(context, data, material, prepared.ext_ior, prepared.int_ior, prepared.thinfilm);
    result.func = result.bsdf;
    result.pdf = 1.0f;
    result.eta = 1.0f;
    result.properties = BSDFSample::Delta | BSDFSample::Reflection;
    result.medium_index = data.current_medium;
    return result;
  }

  if (prepared.supported == false) {
    return bsdf_eval_zero(data.spectrum_sample);
  }
  if (bsdf_energy_compensated_material_interface_valid(context, material, MaterialClass::Conductor) == false) {
    return bsdf_eval_zero(data.spectrum_sample);
  }

  const SpectralResponse reflectance = bsdf_resource_apply_image(context, data.spectrum_sample, material.reflectance, data.tex);
  return bsdf_conductor_energy_compensated_evaluate_prepared_local(context, data, material, w_i, w_o, prepared, reflectance);
}

ETX_SHARED_INLINE BSDFSample bsdf_conductor_energy_compensated_sample(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(Material, material),
  ETX_INOUT(Sampler, sampler)) {
  const LocalFrame frame = bsdf_data_get_normal_frame(data, material);
  const float3 w_i = local_frame_to_local(frame, -data.w_i);
  if (w_i.z <= kEpsilon) {
    return bsdf_sample_zero(data.spectrum_sample);
  }

  const BSDFEnergyCompensatedPreparedMaterial prepared = bsdf_energy_compensated_prepare_material(context, data.spectrum_sample, material, data.tex, sampler);
  if (prepared.conductor_delta) {
    BSDFSample result = ETX_ZERO(BSDFSample);
    result.w_o = bsdf_conductor_delta_reflect(data, material);
    result.weight = bsdf_conductor_delta_weight(context, data, material, prepared.ext_ior, prepared.int_ior, prepared.thinfilm);
    result.pdf = 1.0f;
    result.eta = 1.0f;
    result.properties = BSDFSample::Delta | BSDFSample::Reflection;
    result.medium_index = data.current_medium;
    return result;
  }
  if (prepared.supported == false) {
    return bsdf_sample_zero(data.spectrum_sample);
  }
  if (bsdf_energy_compensated_material_interface_valid(context, material, MaterialClass::Conductor) == false) {
    return bsdf_sample_zero(data.spectrum_sample);
  }

  const SpectralResponse e_i_response = bsdf_energy_compensated_conductor_directional_albedo(context, data.spectrum_sample, material, w_i.z, prepared.alpha, prepared.thinfilm_lut_value);
  const float e_i = spectral_response_monochromatic(e_i_response);
  const float visible_probability = bsdf_energy_compensated_conductor_visible_probability(context, material, w_i.z, prepared.alpha, prepared.thinfilm_lut_value);
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
      const float3 m = bsdf_energy_compensated_sample_vndf_local(w_i, prepared.alpha, attempt_rnd);
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
  const SpectralResponse reflectance = bsdf_resource_apply_image(context, data.spectrum_sample, material.reflectance, data.tex);
  const BSDFEval eval = bsdf_conductor_energy_compensated_evaluate_prepared_local(context, data, material, w_i, local_w_o, prepared, reflectance);
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
  const LocalFrame frame = bsdf_data_get_normal_frame(data, material);
  const float3 w_i = local_frame_to_local(frame, -data.w_i);
  const float3 w_o = local_frame_to_local(frame, outgoing_direction);
  if ((w_i.z <= kEpsilon) || (w_o.z <= kEpsilon)) {
    return 0.0f;
  }

  const BSDFEnergyCompensatedPreparedMaterial prepared = bsdf_energy_compensated_prepare_material(context, data.spectrum_sample, material, data.tex, sampler);
  if (prepared.conductor_delta) {
    const float3 ideal_w_o = bsdf_conductor_delta_reflect(data, material);
    const float3 actual_w_o = normalize(outgoing_direction);
    return direction_matches(ideal_w_o, actual_w_o, 1.0f) ? 1.0f : 0.0f;
  }
  if (prepared.supported == false) {
    return 0.0f;
  }
  if (bsdf_energy_compensated_material_interface_valid(context, material, MaterialClass::Conductor) == false) {
    return 0.0f;
  }
  return bsdf_energy_compensated_conductor_pdf_local(context, data.spectrum_sample, material, w_i, w_o, prepared.alpha, prepared.thinfilm_lut_value);
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
  ETX_IN(float3, w_o_local), float alpha, ETX_IN(RefractiveIndexSample, ext_ior), ETX_IN(RefractiveIndexSample, int_ior), ETX_IN(ThinfilmEval, thinfilm),
  ETX_IN(SpectralResponse, texture)) {
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
  ETX_IN(RefractiveIndexSample, ext_ior), ETX_IN(RefractiveIndexSample, int_ior), ETX_IN(ThinfilmEval, thinfilm)) {
  const SpectralResponse texture = spectral_response_make(spect, 1.0f);
  const BSDFEnergyCompensatedLobe lobe = bsdf_energy_compensated_dielectric_base_lobe(spect, w_i_local, w_o_local, alpha, ext_ior, int_ior, thinfilm, texture);
  return lobe.pdf;
}

ETX_SHARED_INLINE BSDFEnergyCompensatedDielectricComponents bsdf_energy_compensated_dielectric_components_local(ETX_IN(BSDFResourceContext, context), ETX_IN(SpectralQuery, spect),
  ETX_IN(Material, material), ETX_IN(float3, w_i), ETX_IN(float3, w_o), float alpha, ETX_IN(RefractiveIndexSample, ext_ior), ETX_IN(RefractiveIndexSample, int_ior),
  ETX_IN(ThinfilmEval, thinfilm), ETX_IN(SpectralResponse, texture)) {
  BSDFEnergyCompensatedDielectricComponents result = ETX_ZERO(BSDFEnergyCompensatedDielectricComponents);
  result.base = spectral_response_make(spect, 0.0f);
  result.compensation = spectral_response_make(spect, 0.0f);
  if ((abs(w_i.z) <= kEpsilon) || (abs(w_o.z) <= kEpsilon)) {
    return result;
  }

  const BSDFEnergyCompensatedLobe base_lobe = bsdf_energy_compensated_dielectric_base_lobe(spect, w_i, w_o, alpha, ext_ior, int_ior, thinfilm, texture);
  result.base = base_lobe.bsdf;
  result.base_pdf = base_lobe.pdf;
  const float thinfilm_lut_value = bsdf_energy_compensated_thinfilm_lut_value(material, thinfilm);
  result.thinfilm_lut_value = thinfilm_lut_value;
  const bool incident_outside = w_i.z > 0.0f;
  const bool outgoing_outside = w_o.z > 0.0f;
  const BSDFEnergyCompensatedDielectricBranchPair incident_pair =
    bsdf_energy_compensated_dielectric_branch_pair_value(context, spect, material, abs(w_i.z), alpha, incident_outside, thinfilm_lut_value);
  const BSDFEnergyCompensatedDielectricBranchPair outgoing_pair =
    bsdf_energy_compensated_dielectric_branch_pair_value(context, spect, material, abs(w_o.z), alpha, outgoing_outside, thinfilm_lut_value);
  const SpectralResponse e_i = bsdf_energy_compensated_dielectric_branch_pair_albedo(spect, incident_pair);
  const SpectralResponse e_o = bsdf_energy_compensated_dielectric_branch_pair_albedo(spect, outgoing_pair);
  result.incident_albedo = e_i;
  result.incident_visible_probability = bsdf_energy_compensated_dielectric_branch_pair_visible_probability(incident_pair);
  const SpectralResponse one = spectral_response_make(spect, 1.0f);
  const SpectralResponse d_i = spectral_response_max(spectral_response_sub(one, e_i), 0.0f);
  const SpectralResponse d_o = spectral_response_max(spectral_response_sub(one, e_o), 0.0f);
  const SpectralResponse coefficient = bsdf_energy_compensated_dielectric_branch_coefficient(context, spect, material, alpha, incident_outside, outgoing_outside, thinfilm_lut_value);
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
    bsdf_energy_compensated_dielectric_compensation_branch_probability(context, spect, material, alpha, incident_outside, outgoing_outside, components.thinfilm_lut_value);
  const float compensation_pdf = branch_probability * abs(w_o.z) * kInvPi;
  return base_probability * normalized_base_pdf + compensation_probability * compensation_pdf;
}

ETX_SHARED_INLINE SpectralResponse bsdf_energy_compensated_dielectric_bsdf_local(ETX_IN(BSDFResourceContext, context), ETX_IN(SpectralQuery, spect), ETX_IN(Material, material),
  ETX_IN(float3, w_i), ETX_IN(float3, w_o), float alpha, ETX_IN(RefractiveIndexSample, ext_ior), ETX_IN(RefractiveIndexSample, int_ior),
  ETX_IN(ThinfilmEval, thinfilm), ETX_IN(SpectralResponse, texture)) {
  const BSDFEnergyCompensatedDielectricComponents components =
    bsdf_energy_compensated_dielectric_components_local(context, spect, material, w_i, w_o, alpha, ext_ior, int_ior, thinfilm, texture);
  const SpectralResponse result = spectral_response_add(components.base, components.compensation);
  if (bsdf_energy_compensated_spectral_response_finite(result) == false) {
    return spectral_response_make(spect, 0.0f);
  }

  return result;
}

ETX_SHARED_INLINE float bsdf_energy_compensated_dielectric_pdf_local(ETX_IN(BSDFResourceContext, context), ETX_IN(SpectralQuery, spect), ETX_IN(Material, material),
  ETX_IN(float3, w_i), ETX_IN(float3, w_o), float alpha, ETX_IN(RefractiveIndexSample, ext_ior), ETX_IN(RefractiveIndexSample, int_ior),
  ETX_IN(ThinfilmEval, thinfilm)) {
  if ((abs(w_i.z) <= kEpsilon) || (abs(w_o.z) <= kEpsilon)) {
    return 0.0f;
  }

  const bool outside = w_i.z > 0.0f;
  const float thinfilm_lut_value = bsdf_energy_compensated_thinfilm_lut_value(material, thinfilm);
  const BSDFEnergyCompensatedDielectricBranchPair incident_pair =
    bsdf_energy_compensated_dielectric_branch_pair_value(context, spect, material, abs(w_i.z), alpha, outside, thinfilm_lut_value);
  const float visible_probability = bsdf_energy_compensated_dielectric_branch_pair_visible_probability(incident_pair);
  if (visible_probability <= kEpsilon) {
    return 0.0f;
  }

  const float base_pdf = bsdf_energy_compensated_dielectric_base_pdf_local(spect, w_i, w_o, alpha, ext_ior, int_ior, thinfilm);
  const float normalized_base_pdf = base_pdf / visible_probability;
  const SpectralResponse e_i = bsdf_energy_compensated_dielectric_branch_pair_albedo(spect, incident_pair);
  const float base_probability = bsdf_energy_compensated_saturate(spectral_response_monochromatic(e_i));
  const float compensation_probability = max(0.0f, 1.0f - base_probability);
  const bool outgoing_outside = w_o.z > 0.0f;
  const float branch_probability =
    bsdf_energy_compensated_dielectric_compensation_branch_probability(context, spect, material, alpha, outside, outgoing_outside, thinfilm_lut_value);
  const float compensation_pdf = branch_probability * abs(w_o.z) * kInvPi;
  return base_probability * normalized_base_pdf + compensation_probability * compensation_pdf;
}

ETX_SHARED_INLINE BSDFEval bsdf_energy_compensated_dielectric_evaluate_physical_local(ETX_IN(BSDFResourceContext, context), ETX_IN(SpectralQuery, spect),
  ETX_IN(Material, material), ETX_IN(float3, w_i), ETX_IN(float3, w_o), float alpha, ETX_IN(RefractiveIndexSample, ext_ior), ETX_IN(RefractiveIndexSample, int_ior),
  ETX_IN(ThinfilmEval, thinfilm), ETX_IN(SpectralResponse, texture), uint32_t current_medium) {
  const BSDFEnergyCompensatedDielectricComponents components =
    bsdf_energy_compensated_dielectric_components_local(context, spect, material, w_i, w_o, alpha, ext_ior, int_ior, thinfilm, texture);
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

ETX_SHARED_INLINE BSDFEval bsdf_dielectric_energy_compensated_evaluate_prepared_local(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data),
  ETX_IN(Material, material), ETX_IN(float3, w_i), ETX_IN(float3, w_o), ETX_IN(BSDFEnergyCompensatedPreparedMaterial, prepared),
  ETX_IN(SpectralResponse, texture)) {
  BSDFEval physical_eval = bsdf_energy_compensated_dielectric_evaluate_physical_local(context, data.spectrum_sample, material, w_i, w_o, prepared.alpha, prepared.ext_ior,
    prepared.int_ior, prepared.thinfilm, texture, data.current_medium);
  if (bsdf_eval_valid(physical_eval) == false) {
    return physical_eval;
  }

  const bool reflection = (w_i.z * w_o.z) > 0.0f;
  if ((reflection == false) && (data.path_source == PathSource::Light)) {
    const SpectralResponse reverse_bsdf = bsdf_energy_compensated_dielectric_bsdf_local(context, data.spectrum_sample, material, w_o, w_i, prepared.alpha, prepared.ext_ior,
      prepared.int_ior, prepared.thinfilm, texture);
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

ETX_SHARED_INLINE BSDFEval bsdf_dielectric_energy_compensated_evaluate(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(float3, outgoing_direction),
  ETX_IN(Material, material), ETX_INOUT(Sampler, sampler)) {
  LocalFrame frame = ETX_ZERO(LocalFrame);
  frame.tan = data.tan;
  frame.btn = data.btn;
  frame.nrm = data.nrm;
  const float3 w_i = local_frame_to_local(frame, -data.w_i);
  const float3 w_o = local_frame_to_local(frame, outgoing_direction);
  if ((abs(w_i.z) <= kEpsilon) || (abs(w_o.z) <= kEpsilon)) {
    return bsdf_eval_zero(data.spectrum_sample);
  }

  const BSDFEnergyCompensatedPreparedMaterial prepared = bsdf_energy_compensated_prepare_material(context, data.spectrum_sample, material, data.tex, sampler);
  if (prepared.supported == false) {
    return bsdf_eval_zero(data.spectrum_sample);
  }
  if (prepared.dielectric_delta) {
    return bsdf_eval_zero(data.spectrum_sample);
  }
  if (bsdf_energy_compensated_material_interface_valid(context, material, MaterialClass::Dielectric) == false) {
    return bsdf_eval_zero(data.spectrum_sample);
  }

  const bool reflection = (w_i.z * w_o.z) > 0.0f;
  const SpectralImage texture_image = reflection ? material.reflectance : material.scattering;
  const SpectralResponse texture = bsdf_resource_apply_image(context, data.spectrum_sample, texture_image, data.tex);
  return bsdf_dielectric_energy_compensated_evaluate_prepared_local(context, data, material, w_i, w_o, prepared, texture);
}

ETX_SHARED_INLINE BSDFSample bsdf_dielectric_energy_compensated_sample(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(Material, material),
  ETX_INOUT(Sampler, sampler)) {
  LocalFrame frame = ETX_ZERO(LocalFrame);
  frame.tan = data.tan;
  frame.btn = data.btn;
  frame.nrm = data.nrm;
  const float3 w_i_local = local_frame_to_local(frame, -data.w_i);
  if (abs(w_i_local.z) <= kEpsilon) {
    return bsdf_sample_zero(data.spectrum_sample);
  }

  const BSDFEnergyCompensatedPreparedMaterial prepared = bsdf_energy_compensated_prepare_material(context, data.spectrum_sample, material, data.tex, sampler);
  if (prepared.dielectric_delta) {
    return bsdf_dielectric_delta_sample(context, data, material, sampler);
  }
  if (prepared.supported == false) {
    return bsdf_sample_zero(data.spectrum_sample);
  }
  if (bsdf_energy_compensated_material_interface_valid(context, material, MaterialClass::Dielectric) == false) {
    return bsdf_sample_zero(data.spectrum_sample);
  }

  const bool outside = w_i_local.z > 0.0f;
  const BSDFEnergyCompensatedDielectricBranchPair incident_pair =
    bsdf_energy_compensated_dielectric_branch_pair_value(context, data.spectrum_sample, material, abs(w_i_local.z), prepared.alpha, outside, prepared.thinfilm_lut_value);
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
  const RefractiveIndexSample phase_ext_ior = outside ? prepared.ext_ior : prepared.int_ior;
  const RefractiveIndexSample phase_int_ior = outside ? prepared.int_ior : prepared.ext_ior;
  bool candidate_valid = false;
  float3 local_w_o = float3(0.0f, 0.0f, 0.0f);
  const float proposal_selector = has_fixed ? sampler.fixed_w : bsdf_sampler_next(sampler);
  if (proposal_selector < base_probability) {
    bool first_attempt = true;
    while (candidate_valid == false) {
      const float2 attempt_rnd = (first_attempt && has_fixed) ? rnd : bsdf_sampler_next_2d(sampler);
      const float3 m = bsdf_energy_compensated_sample_vndf_local(w_i, prepared.alpha, attempt_rnd);
      const SpectralResponse fresnel = bsdf_fresnel_calculate(data.spectrum_sample, dot(w_i, m), phase_ext_ior, phase_int_ior, prepared.thinfilm);
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
      bsdf_energy_compensated_dielectric_compensation_branch_probability(context, data.spectrum_sample, material, prepared.alpha, outside, true, prepared.thinfilm_lut_value);
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
  const bool reflection = (w_i_local.z * local_w_o.z) > 0.0f;
  const SpectralImage texture_image = reflection ? material.reflectance : material.scattering;
  const SpectralResponse texture = bsdf_resource_apply_image(context, data.spectrum_sample, texture_image, data.tex);
  const BSDFEval eval = bsdf_dielectric_energy_compensated_evaluate_prepared_local(context, data, material, w_i_local, local_w_o, prepared, texture);
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
  result.eta = reflection ? 1.0f : bsdf_energy_compensated_dielectric_continuation_eta(prepared.ext_ior, prepared.int_ior, outside);
  result.properties = reflection ? BSDFSample::Reflection : (BSDFSample::Transmission | BSDFSample::MediumChanged);
  result.medium_index = reflection ? data.current_medium : (outside ? material.int_medium : material.ext_medium);
  return result;
}

ETX_SHARED_INLINE float bsdf_dielectric_energy_compensated_pdf(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(float3, outgoing_direction),
  ETX_IN(Material, material), ETX_INOUT(Sampler, sampler)) {
  LocalFrame frame = ETX_ZERO(LocalFrame);
  frame.tan = data.tan;
  frame.btn = data.btn;
  frame.nrm = data.nrm;
  const float3 w_i = local_frame_to_local(frame, -data.w_i);
  const float3 w_o = local_frame_to_local(frame, outgoing_direction);
  if ((abs(w_i.z) <= kEpsilon) || (abs(w_o.z) <= kEpsilon)) {
    return 0.0f;
  }

  const BSDFEnergyCompensatedPreparedMaterial prepared = bsdf_energy_compensated_prepare_material(context, data.spectrum_sample, material, data.tex, sampler);
  if (prepared.supported == false) {
    return 0.0f;
  }
  if (prepared.dielectric_delta) {
    return 0.0f;
  }
  if (bsdf_energy_compensated_material_interface_valid(context, material, MaterialClass::Dielectric) == false) {
    return 0.0f;
  }

  return bsdf_energy_compensated_dielectric_pdf_local(context, data.spectrum_sample, material, w_i, w_o, prepared.alpha, prepared.ext_ior, prepared.int_ior, prepared.thinfilm);
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
