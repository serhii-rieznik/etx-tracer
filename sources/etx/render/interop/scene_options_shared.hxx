#pragma once

#include "gpu_abi_constants.hxx"

struct SceneOptionsGPUSharedContext {
  uint scene_options_descriptor_index;
};

ETX_SHARED_INLINE SceneOptionsGPUSharedContext make_scene_options_gpu_shared_context(uint scene_options_descriptor_index) {
  SceneOptionsGPUSharedContext context;
  context.scene_options_descriptor_index = scene_options_descriptor_index;
  return context;
}

ETX_SHARED_INLINE bool scene_options_shared_has_data(ETX_IN(SceneOptionsGPUSharedContext, context)) {
  return context.scene_options_descriptor_index != kInvalidIndex;
}

ETX_SHARED_INLINE uint32_t scene_options_shared_load_u32(ETX_IN(SceneOptionsGPUSharedContext, context), uint32_t byte_offset) {
  ByteAddressBuffer scene_options_buffer = bindless_buffers[NonUniformResourceIndex(context.scene_options_descriptor_index)];
  return scene_options_buffer.Load(byte_offset);
}

ETX_SHARED_INLINE uint32_t scene_options_shared_samples(ETX_IN(SceneOptionsGPUSharedContext, context)) {
  if (scene_options_shared_has_data(context) == false) {
    return 1u;
  }

  uint32_t result = scene_options_shared_load_u32(context, kSceneOptionsSamplesOffset);
  if (result == 0u) {
    return 1u;
  }
  return result;
}

ETX_SHARED_INLINE uint32_t scene_options_shared_properties_flags(ETX_IN(SceneOptionsGPUSharedContext, context)) {
  if (scene_options_shared_has_data(context) == false) {
    return 0u;
  }

  return scene_options_shared_load_u32(context, kSceneOptionsPropertiesFlagsOffset);
}

ETX_SHARED_INLINE bool scene_options_shared_uses_spectral_mode(ETX_IN(SceneOptionsGPUSharedContext, context)) {
  return (scene_options_shared_properties_flags(context) & (1u << SceneProperty::Spectral)) != 0u;
}
