#pragma once

#include "gpu_abi_constants.hxx"

struct SceneGPUSharedOptions {
  uint32_t min_path_length;
  uint32_t max_path_length;
  uint32_t samples;
  uint32_t random_path_termination;
  float radiance_clamp;
  uint32_t strategy_flags;
  uint32_t light_sampling;
  uint32_t properties_flags;
  uint32_t path_mode;
  uint32_t random_seed;
};

struct SceneGPUSharedGlobals {
  uint32_t vertex_count;
  uint32_t triangle_count;
  uint32_t emitter_profile_count;
  uint32_t emitter_instance_count;
  uint32_t environment_emitter_count;
  uint32_t active_emitter_count;
  float3 bounding_sphere_center;
  float bounding_sphere_radius;
  float3 emission_half_extent;
  uint32_t pixel_filter_image_index;
  float pixel_filter_radius;
};

ETX_SHARED_INLINE bool scene_gpu_has_descriptor(uint32_t descriptor_index) {
  return descriptor_index != kInvalidIndex;
}

ETX_SHARED_INLINE bool scene_gpu_has_material_spectrum_buffers(uint32_t materials_descriptor_index, uint32_t spectrums_descriptor_index) {
  return scene_gpu_has_descriptor(materials_descriptor_index) && scene_gpu_has_descriptor(spectrums_descriptor_index);
}

ETX_SHARED_INLINE bool scene_gpu_has_medium_spectrum_buffers(uint32_t mediums_descriptor_index, uint32_t spectrums_descriptor_index) {
  return scene_gpu_has_descriptor(mediums_descriptor_index) && scene_gpu_has_descriptor(spectrums_descriptor_index);
}

ETX_SHARED_INLINE bool scene_gpu_has_emitter_buffers(uint32_t emitter_instances_descriptor_index, uint32_t emitter_profiles_descriptor_index, uint32_t spectrums_descriptor_index,
  uint32_t scene_globals_descriptor_index) {
  return scene_gpu_has_descriptor(emitter_instances_descriptor_index) && scene_gpu_has_descriptor(emitter_profiles_descriptor_index) &&
         scene_gpu_has_descriptor(spectrums_descriptor_index) && scene_gpu_has_descriptor(scene_globals_descriptor_index);
}

ETX_SHARED_INLINE bool scene_gpu_can_sample_spectrum(uint32_t spectrums_descriptor_index, uint32_t spectrum_index) {
  return scene_gpu_has_descriptor(spectrums_descriptor_index) && scene_gpu_has_descriptor(spectrum_index);
}

ETX_SHARED_INLINE bool scene_gpu_can_apply_image(uint32_t images_descriptor_index, uint32_t image_index) {
  return scene_gpu_has_descriptor(images_descriptor_index) && scene_gpu_has_descriptor(image_index);
}

ETX_SHARED_INLINE uint32_t scene_gpu_load_u32(ByteAddressBuffer buffer, uint32_t byte_offset) {
  return buffer.Load(byte_offset);
}

ETX_SHARED_INLINE float scene_gpu_load_f32(ByteAddressBuffer buffer, uint32_t byte_offset) {
  return asfloat(buffer.Load(byte_offset));
}

ETX_SHARED_INLINE SceneGPUSharedOptions scene_gpu_load_options(uint32_t scene_options_descriptor_index) {
  SceneGPUSharedOptions result;
  result.min_path_length = 0u;
  result.max_path_length = 0u;
  result.samples = 1u;
  result.random_path_termination = 0u;
  result.radiance_clamp = 0.0f;
  result.strategy_flags = 0u;
  result.light_sampling = 0u;
  result.properties_flags = 0u;
  result.path_mode = 0u;
  result.random_seed = 0u;

  if (scene_gpu_has_descriptor(scene_options_descriptor_index) == false) {
    return result;
  }

  ByteAddressBuffer scene_options_buffer = bindless_buffers[NonUniformResourceIndex(scene_options_descriptor_index)];
  result.min_path_length = scene_gpu_load_u32(scene_options_buffer, kSceneOptionsMinPathLengthOffset);
  result.max_path_length = scene_gpu_load_u32(scene_options_buffer, kSceneOptionsMaxPathLengthOffset);
  result.samples = scene_gpu_load_u32(scene_options_buffer, kSceneOptionsSamplesOffset);
  result.random_path_termination = scene_gpu_load_u32(scene_options_buffer, kSceneOptionsRandomPathTerminationOffset);
  result.radiance_clamp = scene_gpu_load_f32(scene_options_buffer, kSceneOptionsRadianceClampOffset);
  result.strategy_flags = scene_gpu_load_u32(scene_options_buffer, kSceneOptionsStrategyFlagsOffset);
  result.light_sampling = scene_gpu_load_u32(scene_options_buffer, kSceneOptionsLightSamplingOffset);
  result.properties_flags = scene_gpu_load_u32(scene_options_buffer, kSceneOptionsPropertiesFlagsOffset);
  result.path_mode = scene_gpu_load_u32(scene_options_buffer, kSceneOptionsPathModeOffset);
  result.random_seed = scene_gpu_load_u32(scene_options_buffer, kSceneOptionsRandomSeedOffset);

  if (result.samples == 0u) {
    result.samples = 1u;
  }

  return result;
}

ETX_SHARED_INLINE bool scene_gpu_uses_spectral_mode(ETX_IN(SceneGPUSharedOptions, options)) {
  return (options.properties_flags & (1u << SceneProperty::Spectral)) != 0u;
}

ETX_SHARED_INLINE SceneGPUSharedGlobals scene_gpu_load_globals(ByteAddressBuffer scene_globals) {
  SceneGPUSharedGlobals result;
  result.vertex_count = scene_gpu_load_u32(scene_globals, kSceneGlobalsVertexCountOffset);
  result.triangle_count = scene_gpu_load_u32(scene_globals, kSceneGlobalsTriangleCountOffset);
  result.emitter_profile_count = scene_gpu_load_u32(scene_globals, kSceneGlobalsEmitterProfileCountOffset);
  result.emitter_instance_count = scene_gpu_load_u32(scene_globals, kSceneGlobalsEmitterInstanceCountOffset);
  result.environment_emitter_count = scene_gpu_load_u32(scene_globals, kSceneGlobalsEnvironmentEmitterCountOffset);
  result.active_emitter_count = scene_gpu_load_u32(scene_globals, kSceneGlobalsActiveEmitterCountOffset);
  result.bounding_sphere_center = asfloat(scene_globals.Load3(kSceneGlobalsBoundingSphereCenterOffset));
  result.bounding_sphere_radius = scene_gpu_load_f32(scene_globals, kSceneGlobalsBoundingSphereRadiusOffset);
  result.emission_half_extent = asfloat(scene_globals.Load3(kSceneGlobalsEmissionHalfExtentOffset));
  result.pixel_filter_image_index = scene_gpu_load_u32(scene_globals, kSceneGlobalsPixelFilterImageIndexOffset);
  result.pixel_filter_radius = scene_gpu_load_f32(scene_globals, kSceneGlobalsPixelFilterRadiusOffset);
  return result;
}

ETX_SHARED_INLINE uint32_t scene_gpu_environment_emitter(ByteAddressBuffer scene_globals, uint32_t index) {
  return scene_gpu_load_u32(scene_globals, kSceneGlobalsEnvironmentEmittersOffset + index * 4u);
}
