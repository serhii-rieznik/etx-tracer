#pragma once

#include <etx/render/access/image_access_cpu.hxx>
#include <etx/render/access/image_evaluate_cpu.hxx>
#include <etx/render/interop/bsdf_energy_compensation_interface_shared.hxx>
#include <etx/render/access/spectrum_access_cpu.hxx>

namespace etx {
struct Scene;
}

struct BSDFResourceContext {
  const etx::Scene* scene = nullptr;
};

ETX_SHARED_INLINE BSDFResourceContext make_bsdf_resource_cpu_context(const etx::Scene& scene) {
  BSDFResourceContext result = {};
  result.scene = &scene;
  return result;
}

ETX_SHARED_INLINE SpectralResponse bsdf_resource_load_spectrum(ETX_IN(BSDFResourceContext, context), uint32_t spectrum_index, ETX_IN(SpectralQuery, spect)) {
  if ((spectrum_index == kInvalidIndex) || (spectrum_index >= context.scene->spectrums.count)) {
    return spectral_response_make(spect, 0.0f);
  }

  etx::SpectrumAccessCPUContext spectrum_context =
    etx::make_spectrum_access_cpu_context(reinterpret_cast<const ::SpectralDistribution*>(context.scene->spectrums.a), static_cast<uint32_t>(context.scene->spectrums.count));
  return etx::spectrum_access_evaluate(spectrum_context, spectrum_index, spect);
}

ETX_SHARED_INLINE bool bsdf_resource_image_has_alpha(ETX_IN(BSDFResourceContext, context), uint32_t image_index) {
  etx::ImageAccessCPUContext image_access = etx::make_image_access_cpu_context(*context.scene);
  return etx::image_access_has_alpha(image_access, image_index);
}

ETX_SHARED_INLINE bool bsdf_resource_image_try_evaluate_rgba(ETX_IN(BSDFResourceContext, context), uint32_t image_index, ETX_IN(float2, uv), ETX_OUT(float, image_pdf),
  ETX_OUT(float4, image_value)) {
  etx::ImageEvaluateCPUContext image_context = etx::make_image_evaluate_cpu_context(*context.scene);
  return etx::image_evaluate_try_rgba(image_context, image_index, uv, image_pdf, image_value);
}

ETX_SHARED_INLINE bool bsdf_resource_image_try_evaluate_rgba_no_pdf(ETX_IN(BSDFResourceContext, context), uint32_t image_index, ETX_IN(float2, uv),
  ETX_OUT(float4, image_value)) {
  etx::ImageEvaluateCPUContext image_context = etx::make_image_evaluate_cpu_context(*context.scene);
  return etx::image_evaluate_try_rgba_no_pdf(image_context, image_index, uv, image_value);
}

ETX_SHARED_INLINE float4 bsdf_resource_image_evaluate_rgba_no_pdf_or_zero(ETX_IN(BSDFResourceContext, context), uint32_t image_index, ETX_IN(float2, uv)) {
  float4 image_value = float4{0.0f, 0.0f, 0.0f, 0.0f};
  const bool value_loaded = bsdf_resource_image_try_evaluate_rgba_no_pdf(context, image_index, uv, image_value);
  return value_loaded ? image_value : float4{0.0f, 0.0f, 0.0f, 0.0f};
}

ETX_SHARED_INLINE bool bsdf_resource_image_try_evaluate_rgba_3d(ETX_IN(BSDFResourceContext, context), uint32_t image_index, ETX_IN(float3, uvw),
  ETX_OUT(float4, image_value)) {
  image_value = float4{0.0f, 0.0f, 0.0f, 0.0f};
  if ((image_index == kInvalidIndex) || (image_index >= context.scene->images.count)) {
    return false;
  }

  const auto& image = context.scene->images[image_index];
  if (image.format != Image::Format::RGBA32F) {
    return false;
  }

  image_value = image.evaluate_rgba32f_fast_3d(uvw);
  return true;
}

ETX_SHARED_INLINE float4 bsdf_resource_image_evaluate_rgba_3d_or_zero(ETX_IN(BSDFResourceContext, context), uint32_t image_index, ETX_IN(float3, uvw)) {
  float4 image_value = float4{0.0f, 0.0f, 0.0f, 0.0f};
  const bool value_loaded = bsdf_resource_image_try_evaluate_rgba_3d(context, image_index, uvw, image_value);
  return value_loaded ? image_value : float4{0.0f, 0.0f, 0.0f, 0.0f};
}

ETX_SHARED_INLINE bool bsdf_resource_image_try_evaluate_rgba_layer_no_pdf(ETX_IN(BSDFResourceContext, context), uint32_t image_index, ETX_IN(float2, uv), uint32_t layer,
  ETX_OUT(float4, image_value)) {
  image_value = float4{0.0f, 0.0f, 0.0f, 0.0f};
  if ((image_index == kInvalidIndex) || (image_index >= context.scene->images.count)) {
    return false;
  }

  const auto& image = context.scene->images[image_index];
  if ((image.format != Image::Format::RGBA32F) || (image.pixels.f32.a == nullptr) || (image.isize.z == 0u)) {
    return false;
  }

  const ImageFilterSharedAddress sample = image_filter_shared_address(uv, float2{image.fsize.x, image.fsize.y}, uint2{image.isize.x, image.isize.y}, image.options);
  const uint32_t slice = min(layer, image.isize.z - 1u);
  const uint32_t row_stride = image.isize.x;
  const uint32_t slice_offset = slice * image.isize.x * image.isize.y;
  const uint32_t row_offset_0 = sample.row_0 * row_stride;
  const uint32_t row_offset_1 = sample.row_1 * row_stride;
  const float4 p00 = image.pixels.f32.a[slice_offset + row_offset_0 + sample.col_0];
  const float4 p01 = image.pixels.f32.a[slice_offset + row_offset_0 + sample.col_1];
  const float4 p10 = image.pixels.f32.a[slice_offset + row_offset_1 + sample.col_0];
  const float4 p11 = image.pixels.f32.a[slice_offset + row_offset_1 + sample.col_1];
  image_value = image_filter_shared_bilinear(p00, p01, p10, p11, sample.dx, sample.dy);
  return true;
}

ETX_SHARED_INLINE float4 bsdf_resource_image_evaluate_rgba_layer_no_pdf_or_zero(ETX_IN(BSDFResourceContext, context), uint32_t image_index, ETX_IN(float2, uv),
  uint32_t layer) {
  float4 image_value = float4{0.0f, 0.0f, 0.0f, 0.0f};
  const bool value_loaded = bsdf_resource_image_try_evaluate_rgba_layer_no_pdf(context, image_index, uv, layer, image_value);
  return value_loaded ? image_value : float4{0.0f, 0.0f, 0.0f, 0.0f};
}

ETX_SHARED_INLINE bool bsdf_resource_image_has_size(ETX_IN(BSDFResourceContext, context), uint32_t image_index, uint32_t width, uint32_t height, ETX_OUT(uint32_t, depth)) {
  depth = 0u;
  if ((image_index == kInvalidIndex) || (image_index >= context.scene->images.count)) {
    return false;
  }

  const auto& image = context.scene->images[image_index];
  depth = image.isize.z;
  return (image.isize.x == width) && (image.isize.y == height) && (image.isize.z > 0u) && (image.format == Image::Format::RGBA32F);
}

ETX_SHARED_INLINE float bsdf_resource_image_sample_channel_or_default(ETX_IN(BSDFResourceContext, context), uint32_t image_index, uint32_t channel, ETX_IN(float2, uv),
  float default_value) {
  etx::ImageEvaluateCPUContext image_context = etx::make_image_evaluate_cpu_context(*context.scene);
  return etx::image_evaluate_sample_channel_or_default(image_context, image_index, channel, uv, default_value);
}

ETX_SHARED_INLINE bool bsdf_resource_energy_compensation_interface_try_load(ETX_IN(BSDFResourceContext, context), uint32_t interface_index,
  ETX_OUT(BSDFEnergyCompensationInterfaceData, interface_data)) {
  interface_data = ETX_ZERO(BSDFEnergyCompensationInterfaceData);
  if ((interface_index == kInvalidIndex) || (interface_index >= context.scene->energy_compensation_interfaces.count)) {
    return false;
  }

  const auto& source = context.scene->energy_compensation_interfaces[interface_index];
  interface_data.cls = source.cls;
  interface_data.cache_mode = source.cache_mode;
  interface_data.spectral_wavelength_count = source.spectral_wavelength_count;
  interface_data.thinfilm_slice_count = source.thinfilm_slice_count;
  interface_data.directional_lut = source.directional_lut;
  interface_data.average_lut = source.average_lut;
  interface_data.geometric_lut = source.geometric_lut;
  interface_data.geometric_average_lut = source.geometric_average_lut;
  interface_data.conductor_fms_lut = source.conductor_fms_lut;
  interface_data.probability_lut = source.probability_lut;
  interface_data.spectral_shortest_wavelength = source.spectral_shortest_wavelength;
  interface_data.spectral_longest_wavelength = source.spectral_longest_wavelength;
  return true;
}

ETX_SHARED_INLINE uint32_t bsdf_resource_default_white_spectrum(ETX_IN(BSDFResourceContext, context)) {
  return context.scene->defaults.white_spectrum;
}

ETX_SHARED_INLINE uint32_t bsdf_resource_default_dielectric_eta(ETX_IN(BSDFResourceContext, context)) {
  return context.scene->defaults.dielectric_eta;
}

ETX_SHARED_INLINE uint32_t bsdf_resource_default_conductor_eta(ETX_IN(BSDFResourceContext, context)) {
  return context.scene->defaults.conductor_eta;
}

ETX_SHARED_INLINE uint32_t bsdf_resource_default_conductor_k(ETX_IN(BSDFResourceContext, context)) {
  return context.scene->defaults.conductor_k;
}
