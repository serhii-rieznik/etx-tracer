#pragma once

#include <access/image_access_gpu.hxx>
#include <access/image_evaluate_gpu.hxx>
#include <access/spectrum_access_gpu.hxx>
#include <interop/bsdf_energy_compensation_interface_shared.hxx>
#include <interop/scene_gpu_access_shared.hxx>

struct BSDFResourceContext {
  uint images_descriptor_index;
  uint spectrums_descriptor_index;
  uint energy_compensation_interfaces_descriptor_index;
  uint scene_globals_descriptor_index;
};

BSDFResourceContext make_bsdf_resource_gpu_context(uint images_descriptor_index, uint spectrums_descriptor_index, uint energy_compensation_interfaces_descriptor_index,
  uint scene_globals_descriptor_index) {
  BSDFResourceContext result;
  result.images_descriptor_index = images_descriptor_index;
  result.spectrums_descriptor_index = spectrums_descriptor_index;
  result.energy_compensation_interfaces_descriptor_index = energy_compensation_interfaces_descriptor_index;
  result.scene_globals_descriptor_index = scene_globals_descriptor_index;
  return result;
}

SpectralResponse bsdf_resource_load_spectrum(BSDFResourceContext context, uint spectrum_index, SpectralQuery spect) {
  if ((scene_gpu_has_descriptor(context.spectrums_descriptor_index) == false) || (spectrum_index == kInvalidIndex)) {
    return spectral_response_make(spect, 0.0f);
  }

  ByteAddressBuffer spectrum_buffer = bindless_buffers[NonUniformResourceIndex(context.spectrums_descriptor_index)];
  SpectrumAccessGPUContext spectrum_context = make_spectrum_access_gpu_context(spectrum_buffer, context.spectrums_descriptor_index);
  return spectrum_access_evaluate(spectrum_context, spectrum_index, spect);
}

bool bsdf_resource_image_has_alpha(BSDFResourceContext context, uint image_index) {
  if (scene_gpu_has_descriptor(context.images_descriptor_index) == false) {
    return false;
  }

  ImageAccessGPUContext image_context = {context.images_descriptor_index};
  return image_access_has_alpha(image_context, image_index);
}

bool bsdf_resource_image_try_evaluate_rgba(BSDFResourceContext context, uint image_index, float2 uv, out float image_pdf, out float4 image_value) {
  if (scene_gpu_has_descriptor(context.images_descriptor_index) == false) {
    image_pdf = 0.0f;
    image_value = float4(1.0f, 1.0f, 1.0f, 1.0f);
    return false;
  }

  ImageEvaluateGPUContext image_context = make_image_evaluate_gpu_context(context.images_descriptor_index);
  float local_image_pdf = 0.0f;
  float4 local_image_value = float4(1.0f, 1.0f, 1.0f, 1.0f);
  const bool result = image_evaluate_try_rgba(image_context, image_index, uv, local_image_pdf, local_image_value);
  image_pdf = local_image_pdf;
  image_value = local_image_value;
  return result;
}

bool bsdf_resource_image_try_evaluate_rgba_no_pdf(BSDFResourceContext context, uint image_index, float2 uv, out float4 image_value) {
  if (scene_gpu_has_descriptor(context.images_descriptor_index) == false) {
    image_value = float4(1.0f, 1.0f, 1.0f, 1.0f);
    return false;
  }

  ImageEvaluateGPUContext image_context = make_image_evaluate_gpu_context(context.images_descriptor_index);
  float4 local_image_value = float4(1.0f, 1.0f, 1.0f, 1.0f);
  const bool result = image_evaluate_try_rgba_no_pdf(image_context, image_index, uv, local_image_value);
  image_value = local_image_value;
  return result;
}

[noinline] float4 bsdf_resource_image_evaluate_rgba_no_pdf_or_zero(BSDFResourceContext context, uint image_index, float2 uv) {
  float4 image_value = float4(0.0f, 0.0f, 0.0f, 0.0f);
  const bool value_loaded = bsdf_resource_image_try_evaluate_rgba_no_pdf(context, image_index, uv, image_value);
  return value_loaded ? image_value : float4(0.0f, 0.0f, 0.0f, 0.0f);
}

bool bsdf_resource_image_try_evaluate_rgba_3d(BSDFResourceContext context, uint image_index, float3 uvw, out float4 image_value) {
  if (scene_gpu_has_descriptor(context.images_descriptor_index) == false) {
    image_value = float4(0.0f, 0.0f, 0.0f, 0.0f);
    return false;
  }

  ImageEvaluateGPUContext image_context = make_image_evaluate_gpu_context(context.images_descriptor_index);
  float4 local_image_value = float4(0.0f, 0.0f, 0.0f, 0.0f);
  const bool result = image_evaluate_gpu_try_rgba_3d(image_context, image_index, uvw, local_image_value);
  image_value = local_image_value;
  return result;
}

[noinline] float4 bsdf_resource_image_evaluate_rgba_3d_or_zero(BSDFResourceContext context, uint image_index, float3 uvw) {
  float4 image_value = float4(0.0f, 0.0f, 0.0f, 0.0f);
  const bool value_loaded = bsdf_resource_image_try_evaluate_rgba_3d(context, image_index, uvw, image_value);
  return value_loaded ? image_value : float4(0.0f, 0.0f, 0.0f, 0.0f);
}

bool bsdf_resource_image_try_evaluate_rgba_layer_no_pdf(BSDFResourceContext context, uint image_index, float2 uv, uint layer, out float4 image_value) {
  if (scene_gpu_has_descriptor(context.images_descriptor_index) == false) {
    image_value = float4(0.0f, 0.0f, 0.0f, 0.0f);
    return false;
  }

  ImageEvaluateGPUContext image_context = make_image_evaluate_gpu_context(context.images_descriptor_index);
  float4 local_image_value = float4(0.0f, 0.0f, 0.0f, 0.0f);
  const bool result = image_evaluate_gpu_try_rgba_layer_no_pdf(image_context, image_index, uv, layer, local_image_value);
  image_value = local_image_value;
  return result;
}

[noinline] float4 bsdf_resource_image_evaluate_rgba_layer_no_pdf_or_zero(BSDFResourceContext context, uint image_index, float2 uv, uint layer) {
  float4 image_value = float4(0.0f, 0.0f, 0.0f, 0.0f);
  const bool value_loaded = bsdf_resource_image_try_evaluate_rgba_layer_no_pdf(context, image_index, uv, layer, image_value);
  return value_loaded ? image_value : float4(0.0f, 0.0f, 0.0f, 0.0f);
}

bool bsdf_resource_image_has_size(BSDFResourceContext context, uint image_index, uint width, uint height, out uint depth) {
  depth = 0u;
  if (scene_gpu_has_descriptor(context.images_descriptor_index) == false) {
    return false;
  }

  ImageAccessGPUContext image_context = {context.images_descriptor_index};
  ImageAccessGPUDesc image_access;
  if (image_access_try_load(image_context, image_index, image_access) == false) {
    return false;
  }

  depth = image_access.size.z;
  return (image_access.size.x == width) && (image_access.size.y == height) && (image_access.size.z > 0u) && (image_access.format == (uint)Image::Format::RGBA32F);
}

float bsdf_resource_image_sample_channel_or_default(BSDFResourceContext context, uint image_index, uint channel, float2 uv, float default_value) {
  if (scene_gpu_has_descriptor(context.images_descriptor_index) == false) {
    return default_value;
  }

  ImageEvaluateGPUContext image_context = make_image_evaluate_gpu_context(context.images_descriptor_index);
  return image_evaluate_sample_channel_or_default(image_context, image_index, channel, uv, default_value);
}

bool bsdf_resource_energy_compensation_interface_try_load(BSDFResourceContext context, uint interface_index, out BSDFEnergyCompensationInterfaceData interface_data) {
  interface_data = (BSDFEnergyCompensationInterfaceData)0;
  if ((scene_gpu_has_descriptor(context.energy_compensation_interfaces_descriptor_index) == false) || (interface_index == kInvalidIndex)) {
    return false;
  }

  ByteAddressBuffer interface_buffer = bindless_buffers[NonUniformResourceIndex(context.energy_compensation_interfaces_descriptor_index)];
  uint base_offset = interface_index * 48u;
  interface_data.cls = interface_buffer.Load(base_offset + 0u);
  interface_data.cache_mode = interface_buffer.Load(base_offset + 4u);
  interface_data.spectral_wavelength_count = interface_buffer.Load(base_offset + 8u);
  interface_data.thinfilm_slice_count = interface_buffer.Load(base_offset + 12u);
  interface_data.directional_lut = interface_buffer.Load(base_offset + 16u);
  interface_data.average_lut = interface_buffer.Load(base_offset + 20u);
  interface_data.geometric_lut = interface_buffer.Load(base_offset + 24u);
  interface_data.geometric_average_lut = interface_buffer.Load(base_offset + 28u);
  interface_data.conductor_fms_lut = interface_buffer.Load(base_offset + 32u);
  interface_data.probability_lut = interface_buffer.Load(base_offset + 36u);
  interface_data.spectral_shortest_wavelength = asfloat(interface_buffer.Load(base_offset + 40u));
  interface_data.spectral_longest_wavelength = asfloat(interface_buffer.Load(base_offset + 44u));
  return true;
}

uint bsdf_resource_load_scene_global_u32(BSDFResourceContext context, uint byte_offset) {
  if (scene_gpu_has_descriptor(context.scene_globals_descriptor_index) == false) {
    return kInvalidIndex;
  }

  ByteAddressBuffer scene_globals = bindless_buffers[NonUniformResourceIndex(context.scene_globals_descriptor_index)];
  return scene_globals.Load(byte_offset);
}

uint bsdf_resource_default_white_spectrum(BSDFResourceContext context) {
  return bsdf_resource_load_scene_global_u32(context, kSceneGlobalsDefaultWhiteSpectrumOffset);
}

uint bsdf_resource_default_dielectric_eta(BSDFResourceContext context) {
  return bsdf_resource_load_scene_global_u32(context, kSceneGlobalsDefaultDielectricEtaOffset);
}

uint bsdf_resource_default_conductor_eta(BSDFResourceContext context) {
  return bsdf_resource_load_scene_global_u32(context, kSceneGlobalsDefaultConductorEtaOffset);
}

uint bsdf_resource_default_conductor_k(BSDFResourceContext context) {
  return bsdf_resource_load_scene_global_u32(context, kSceneGlobalsDefaultConductorKOffset);
}
