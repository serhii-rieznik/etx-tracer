#pragma once

#include <access/material_access_shared.hxx>
#include <access/spectrum_access_gpu.hxx>
#include <access/image_access_gpu.hxx>
#include <access/image_evaluate_gpu.hxx>
#include <interop/gpu_abi_access_shared.hxx>
#include <interop/material_scattering_shared.hxx>

struct MaterialAccessGPUContext {
  uint materials_descriptor_index;
};

bool material_access_can_load(MaterialAccessGPUContext context, uint material_index) {
  return scene_resource_shared_is_available(context.materials_descriptor_index) && (material_index != kInvalidIndex);
}

bool material_access_try_load(MaterialAccessGPUContext context, uint material_index, out MaterialAccess access) {
  access = ETX_ZERO(MaterialAccess);
  if (material_access_can_load(context, material_index) == false) {
    return false;
  }

  ByteAddressBuffer material_buffer = bindless_buffers[NonUniformResourceIndex(context.materials_descriptor_index)];
  GPUABIAccessSharedContext access_context = make_gpu_abi_access_shared_context(material_buffer);
  access.material_class = gpu_abi_access_shared_material_class(access_context, material_index);
  access.int_medium_index = gpu_abi_access_shared_material_int_medium(access_context, material_index);
  access.ext_medium_index = gpu_abi_access_shared_material_ext_medium(access_context, material_index);
  access.scattering_spectrum_index = gpu_abi_access_shared_material_scattering_spectrum_index(access_context, material_index);
  access.scattering_image_index = gpu_abi_access_shared_material_scattering_image_index(access_context, material_index);
  access.opacity = gpu_abi_access_shared_material_opacity(access_context, material_index);
  return true;
}

SpectralResponse material_access_evaluate_scattering_spectral(MaterialAccessGPUContext material_context, SpectrumAccessGPUContext spectrum_context,
  ImageAccessGPUContext image_access_context, ImageEvaluateGPUContext image_evaluate_context, uint material_index, float2 uv, float ao, SpectralQuery spect,
  SpectralResponse fallback_value) {
  MaterialAccess access = ETX_ZERO(MaterialAccess);
  if (material_access_try_load(material_context, material_index, access) == false) {
    return fallback_value;
  }
  if (material_access_has_scattering(access) == false) {
    return fallback_value;
  }

  SpectralResponse scattering_value = spectrum_access_evaluate(spectrum_context, access.scattering_spectrum_index, spect);
  bool has_image = image_can_apply(image_access_context, access.scattering_image_index);
  float3 image_rgb = float3(1.0f, 1.0f, 1.0f);
  if (has_image) {
    float4 image_value = image_evaluate_sample_whole_or_default(image_evaluate_context, access.scattering_image_index, uv, float4(1.0f, 1.0f, 1.0f, 1.0f));
    image_rgb = image_value.xyz;
  }

  return material_scattering_shared_apply_spectral_clamped_ao(spect, scattering_value, image_rgb, has_image, ao);
}

float3 material_access_evaluate_scattering_integrated_or_fallback(MaterialAccessGPUContext material_context, SpectrumAccessGPUContext spectrum_context,
  ImageAccessGPUContext image_access_context, ImageEvaluateGPUContext image_evaluate_context, uint material_index, float2 uv, float ao, float3 fallback_color) {
  MaterialAccess access = ETX_ZERO(MaterialAccess);
  if (material_access_try_load(material_context, material_index, access) == false) {
    return fallback_color;
  }
  if (material_access_has_scattering(access) == false) {
    return fallback_color;
  }

  float3 scattering_integrated = spectrum_access_load_integrated(spectrum_context, access.scattering_spectrum_index);
  bool has_image = image_can_apply(image_access_context, access.scattering_image_index);
  float3 image_rgb = float3(1.0f, 1.0f, 1.0f);
  if (has_image) {
    float4 image_value = image_evaluate_sample_whole_or_default(image_evaluate_context, access.scattering_image_index, uv, float4(1.0f, 1.0f, 1.0f, 1.0f));
    image_rgb = image_value.xyz;
  }

  return material_scattering_shared_apply_integrated_clamped_ao(scattering_integrated, image_rgb, has_image, ao);
}
