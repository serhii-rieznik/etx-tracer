#pragma once

#include <access/emitter_access_shared.hxx>
#include <access/image_access_gpu.hxx>
#include <access/image_evaluate_gpu.hxx>
#include <access/spectrum_access_gpu.hxx>
#include <interop/gpu_abi_access_shared.hxx>
#include <interop/scene_globals_shared.hxx>
#include <interop/material_scattering_shared.hxx>

struct EmitterAccessGPUContext {
  uint emitter_instances_descriptor_index;
  uint emitter_profiles_descriptor_index;
  uint spectrums_descriptor_index;
  uint images_descriptor_index;
  uint scene_globals_descriptor_index;
};

EmitterAccessGPUContext make_emitter_access_gpu_context(
  uint emitter_instances_descriptor_index, uint emitter_profiles_descriptor_index, uint spectrums_descriptor_index, uint images_descriptor_index,
  uint scene_globals_descriptor_index) {
  EmitterAccessGPUContext context;
  context.emitter_instances_descriptor_index = emitter_instances_descriptor_index;
  context.emitter_profiles_descriptor_index = emitter_profiles_descriptor_index;
  context.spectrums_descriptor_index = spectrums_descriptor_index;
  context.images_descriptor_index = images_descriptor_index;
  context.scene_globals_descriptor_index = scene_globals_descriptor_index;
  return context;
}

bool emitter_access_try_load_scene_state(EmitterAccessGPUContext context, out uint emitter_instance_count, out uint emitter_profile_count) {
  emitter_instance_count = 0u;
  emitter_profile_count = 0u;
  if ((context.scene_globals_descriptor_index == kInvalidIndex) || (context.emitter_instances_descriptor_index == kInvalidIndex) ||
      (context.emitter_profiles_descriptor_index == kInvalidIndex)) {
    return false;
  }

  ByteAddressBuffer scene_globals = bindless_buffers[NonUniformResourceIndex(context.scene_globals_descriptor_index)];
  SceneGlobalsGPUSharedContext globals_context = make_scene_globals_gpu_shared_context(scene_globals);
  emitter_instance_count = scene_globals_shared_emitter_instance_count(globals_context);
  emitter_profile_count = scene_globals_shared_emitter_profile_count(globals_context);
  return (emitter_instance_count > 0u) && (emitter_profile_count > 0u);
}

bool emitter_access_try_load_profile(EmitterAccessGPUContext context, uint emitter_profile_count, inout EmitterAccess access) {
  if (access.emitter_profile_index >= emitter_profile_count) {
    return false;
  }

  ByteAddressBuffer emitter_profile_buffer = bindless_buffers[NonUniformResourceIndex(context.emitter_profiles_descriptor_index)];
  GPUABIAccessSharedContext access_context = make_gpu_abi_access_shared_context(emitter_profile_buffer);
  access.emission_spectrum_index = gpu_abi_access_shared_emitter_emission_spectrum_index(access_context, access.emitter_profile_index);
  access.emission_image_index = gpu_abi_access_shared_emitter_emission_image_index(access_context, access.emitter_profile_index);
  access.emitter_profile_class = gpu_abi_access_shared_emitter_profile_class(access_context, access.emitter_profile_index);
  access.emitter_profile_meta = gpu_abi_access_shared_emitter_profile_meta(access_context, access.emitter_profile_index);
  access.emitter_direction = gpu_abi_access_shared_emitter_profile_direction(access_context, access.emitter_profile_index);
  access.emitter_angular_size_cosine = gpu_abi_access_shared_emitter_profile_angular_size_cosine(access_context, access.emitter_profile_index);
  return access.emission_spectrum_index != kInvalidIndex;
}

bool emitter_access_try_load_from_instance(EmitterAccessGPUContext context, uint emitter_class, uint emitter_profile_index, out EmitterAccess access) {
  access = ETX_ZERO(EmitterAccess);

  uint emitter_instance_count = 0u;
  uint emitter_profile_count = 0u;
  if (emitter_access_try_load_scene_state(context, emitter_instance_count, emitter_profile_count) == false) {
    return false;
  }
  (void)emitter_instance_count;

  access.emitter_class = emitter_class;
  access.emitter_profile_index = emitter_profile_index;
  return emitter_access_try_load_profile(context, emitter_profile_count, access);
}

bool emitter_access_try_load(EmitterAccessGPUContext context, uint emitter_index, out EmitterAccess access) {
  access = ETX_ZERO(EmitterAccess);

  uint emitter_instance_count = 0u;
  uint emitter_profile_count = 0u;
  if (emitter_access_try_load_scene_state(context, emitter_instance_count, emitter_profile_count) == false) {
    return false;
  }
  if (emitter_index >= emitter_instance_count) {
    return false;
  }

  ByteAddressBuffer emitter_instance_buffer = bindless_buffers[NonUniformResourceIndex(context.emitter_instances_descriptor_index)];
  GPUABIAccessSharedContext access_context = make_gpu_abi_access_shared_context(emitter_instance_buffer);
  access.emitter_class = gpu_abi_access_shared_emitter_class(access_context, emitter_index);
  access.emitter_profile_index = gpu_abi_access_shared_emitter_profile_index(access_context, emitter_index);
  return emitter_access_try_load_profile(context, emitter_profile_count, access);
}

bool emitter_access_try_load_local(EmitterAccessGPUContext context, uint emitter_index, out EmitterAccess access) {
  if (emitter_access_try_load(context, emitter_index, access) == false) {
    return false;
  }

  return emitter_access_is_local_class(access.emitter_class);
}

bool emitter_access_try_load_distant(EmitterAccessGPUContext context, uint emitter_index, float3 direction, out EmitterAccess access) {
  if (emitter_access_try_load(context, emitter_index, access) == false) {
    return false;
  }

  return emitter_access_accepts_distant(
    access.emitter_class, access.emitter_profile_class, direction, access.emitter_direction, access.emitter_angular_size_cosine);
}

bool emitter_access_try_load_image_params(EmitterAccessGPUContext context, uint emission_image_index, out float2 image_offset, out float image_u_scale) {
  image_offset = float2(0.0f, 0.0f);
  image_u_scale = 1.0f;

  ImageAccessGPUContext image_context = {context.images_descriptor_index};
  ETX_ZERO_INIT(ImageAccessGPUDesc, image_access);
  if (image_access_try_load(image_context, emission_image_index, image_access) == false) {
    return false;
  }

  image_offset = image_access.uv_offset;
  image_u_scale = image_access.uv_scale.x;
  return true;
}

float2 emitter_access_environment_uv(EmitterAccessGPUContext context, EmitterAccess access, float3 direction) {
  float2 image_offset = float2(0.0f, 0.0f);
  float image_u_scale = 1.0f;
  emitter_access_try_load_image_params(context, access.emission_image_index, image_offset, image_u_scale);
  return emitter_access_shared_environment_uv(
    access.emitter_class, access.emitter_profile_meta, image_offset, image_u_scale, access.emitter_direction, access.emitter_angular_size_cosine, direction);
}

bool emitter_access_can_sample_spectrum(EmitterAccessGPUContext context, uint emission_spectrum_index) {
  return scene_resource_shared_can_sample_spectrum(context.spectrums_descriptor_index, emission_spectrum_index);
}

float3 emitter_access_load_spectrum_integrated(EmitterAccessGPUContext context, uint emission_spectrum_index) {
  ByteAddressBuffer spectrum_buffer = bindless_buffers[NonUniformResourceIndex(context.spectrums_descriptor_index)];
  SpectrumAccessGPUContext spectrum_context = make_spectrum_access_gpu_context(spectrum_buffer, context.spectrums_descriptor_index);
  return spectrum_access_load_integrated(spectrum_context, emission_spectrum_index);
}

SpectralResponse emitter_access_load_spectrum_spectral(EmitterAccessGPUContext context, uint emission_spectrum_index, SpectralQuery spect) {
  ByteAddressBuffer spectrum_buffer = bindless_buffers[NonUniformResourceIndex(context.spectrums_descriptor_index)];
  SpectrumAccessGPUContext spectrum_context = make_spectrum_access_gpu_context(spectrum_buffer, context.spectrums_descriptor_index);
  return spectrum_access_evaluate(spectrum_context, emission_spectrum_index, spect);
}

bool emitter_access_can_apply_image(EmitterAccessGPUContext context, uint emission_image_index) {
  ImageAccessGPUContext image_context = {context.images_descriptor_index};
  return image_can_apply(image_context, emission_image_index);
}

float3 emitter_access_evaluate_image_rgb(EmitterAccessGPUContext context, uint emission_image_index, float2 uv) {
  ImageEvaluateGPUContext image_context = make_image_evaluate_gpu_context(context.images_descriptor_index);
  float4 image_value = image_evaluate_sample_whole_or_default(image_context, emission_image_index, uv, float4(1.0f, 1.0f, 1.0f, 1.0f));
  return image_value.xyz;
}

float3 emitter_access_evaluate_integrated_source(EmitterAccessGPUContext context, uint emission_spectrum_index, uint emission_image_index, float2 uv) {
  if (emitter_access_can_sample_spectrum(context, emission_spectrum_index) == false) {
    return float3(0.0f, 0.0f, 0.0f);
  }

  float3 result = emitter_access_load_spectrum_integrated(context, emission_spectrum_index);
  bool apply_image = emitter_access_can_apply_image(context, emission_image_index);
  float3 image_rgb = float3(1.0f, 1.0f, 1.0f);
  if (apply_image) {
    image_rgb = emitter_access_evaluate_image_rgb(context, emission_image_index, uv);
  }

  return material_scattering_shared_apply_integrated_clamped_ao(result, image_rgb, apply_image, 1.0f);
}

SpectralResponse emitter_access_evaluate_spectral_source(
  EmitterAccessGPUContext context, uint emission_spectrum_index, uint emission_image_index, float2 uv, SpectralQuery spect) {
  SpectralResponse zero_value = spectral_response_zero(spect);
  if (emitter_access_can_sample_spectrum(context, emission_spectrum_index) == false) {
    return zero_value;
  }

  SpectralResponse result = emitter_access_load_spectrum_spectral(context, emission_spectrum_index, spect);
  bool apply_image = emitter_access_can_apply_image(context, emission_image_index);
  float3 image_rgb = float3(1.0f, 1.0f, 1.0f);
  if (apply_image) {
    image_rgb = emitter_access_evaluate_image_rgb(context, emission_image_index, uv);
  }

  return material_scattering_shared_apply_spectral_clamped_ao(spect, result, image_rgb, apply_image, 1.0f);
}
