#pragma once

#include <etx/render/access/material_access_shared.hxx>
#include <etx/render/access/spectrum_access_cpu.hxx>
#include <etx/render/access/image_access_cpu.hxx>
#include <etx/render/access/image_evaluate_cpu.hxx>
#include <etx/render/interop/material_scattering_shared.hxx>

struct Scene;

struct MaterialAccessCPUContext {
  const Scene* scene = nullptr;
};

ETX_SHARED_INLINE MaterialAccessCPUContext make_material_access_cpu_context(const Scene& scene) {
  MaterialAccessCPUContext result = {};
  result.scene = &scene;
  return result;
}

ETX_SHARED_INLINE bool material_access_can_load(ETX_IN(MaterialAccessCPUContext, context), uint32_t material_index) {
  return (context.scene != nullptr) && (material_index < context.scene->materials.count);
}

ETX_SHARED_INLINE MaterialAccess material_access_cpu_make(ETX_IN(Material, material)) {
  MaterialAccess result = {};
  result.material_class = material.cls;
  result.int_medium_index = material.int_medium;
  result.ext_medium_index = material.ext_medium;
  result.scattering_spectrum_index = material.scattering.spectrum_index;
  result.scattering_image_index = material.scattering.image_index;
  result.opacity = material.opacity;
  return result;
}

ETX_SHARED_INLINE bool material_access_try_load(
  ETX_IN(MaterialAccessCPUContext, context), uint32_t material_index, ETX_OUT(MaterialAccess, access)) {
  access = {};
  if (material_access_can_load(context, material_index) == false) {
    return false;
  }

  access = material_access_cpu_make(context.scene->materials[material_index]);
  return true;
}

ETX_SHARED_INLINE SpectralResponse material_access_evaluate_scattering_spectral(
  ETX_IN(MaterialAccessCPUContext, context), uint32_t material_index, ETX_IN(float2, uv), float ao, ETX_IN(SpectralQuery, spect), ETX_IN(SpectralResponse, fallback_value)) {
  MaterialAccess access = {};
  if (material_access_try_load(context, material_index, access) == false) {
    return fallback_value;
  }
  if (material_access_has_scattering(access) == false) {
    return fallback_value;
  }

  ETX_ASSERT(context.scene != nullptr);
  ETX_ASSERT(access.scattering_spectrum_index < static_cast<uint32_t>(context.scene->spectrums.count));
  SpectrumAccessCPUContext spectrum_context = make_spectrum_access_cpu_context(context.scene->spectrums.a, static_cast<uint32_t>(context.scene->spectrums.count));
  const ::SpectralResponse scattering_response = spectrum_access_evaluate(spectrum_context, access.scattering_spectrum_index, static_cast<const ::SpectralQuery&>(spect));
  SpectralQuery response_query = {scattering_response.wavelength, scattering_response.flags};
  SpectralResponse scattering_value = ::spectral_response_is_spectral(scattering_response) ? SpectralResponse{response_query, scattering_response.value}
                                                                                             : SpectralResponse{response_query, scattering_response.integrated};

  ImageAccessCPUContext image_access_context = make_image_access_cpu_context(*context.scene);
  const bool has_image = image_can_apply(image_access_context, access.scattering_image_index);
  float3 image_rgb = float3(1.0f, 1.0f, 1.0f);
  if (has_image) {
    ImageEvaluateCPUContext image_evaluate_context = make_image_evaluate_cpu_context(*context.scene);
    const float4 image_value = image_evaluate_sample_whole_or_default(image_evaluate_context, access.scattering_image_index, uv, float4(1.0f, 1.0f, 1.0f, 1.0f));
    image_rgb = float3(image_value.x, image_value.y, image_value.z);
  }

  const ::SpectralResponse response = material_scattering_shared_apply_spectral_clamped_ao(
    static_cast<const ::SpectralQuery&>(spect), static_cast<const ::SpectralResponse&>(scattering_value), image_rgb, has_image, ao);
  SpectralQuery result_query = {response.wavelength, response.flags};
  return ::spectral_response_is_spectral(response) ? SpectralResponse{result_query, response.value} : SpectralResponse{result_query, response.integrated};
}

ETX_SHARED_INLINE float3 material_access_evaluate_scattering_integrated_or_fallback(
  ETX_IN(MaterialAccessCPUContext, context), uint32_t material_index, ETX_IN(float2, uv), float ao, ETX_IN(float3, fallback_color)) {
  MaterialAccess access = {};
  if (material_access_try_load(context, material_index, access) == false) {
    return fallback_color;
  }
  if (material_access_has_scattering(access) == false) {
    return fallback_color;
  }

  ETX_ASSERT(context.scene != nullptr);
  ETX_ASSERT(access.scattering_spectrum_index < static_cast<uint32_t>(context.scene->spectrums.count));
  SpectrumAccessCPUContext spectrum_context = make_spectrum_access_cpu_context(context.scene->spectrums.a, static_cast<uint32_t>(context.scene->spectrums.count));
  float3 scattering_integrated = spectrum_access_load_integrated(spectrum_context, access.scattering_spectrum_index);

  ImageAccessCPUContext image_access_context = make_image_access_cpu_context(*context.scene);
  const bool has_image = image_can_apply(image_access_context, access.scattering_image_index);
  float3 image_rgb = float3(1.0f, 1.0f, 1.0f);
  if (has_image) {
    ImageEvaluateCPUContext image_evaluate_context = make_image_evaluate_cpu_context(*context.scene);
    const float4 image_value = image_evaluate_sample_whole_or_default(image_evaluate_context, access.scattering_image_index, uv, float4(1.0f, 1.0f, 1.0f, 1.0f));
    image_rgb = float3(image_value.x, image_value.y, image_value.z);
  }

  return material_scattering_shared_apply_integrated_clamped_ao(scattering_integrated, image_rgb, has_image, ao);
}
