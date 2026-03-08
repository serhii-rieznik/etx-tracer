#pragma once

#include "gpu_abi_constants.hxx"

struct SceneGlobalsGPUSharedContext {
  ByteAddressBuffer scene_globals;
};

ETX_SHARED_INLINE SceneGlobalsGPUSharedContext make_scene_globals_gpu_shared_context(ByteAddressBuffer scene_globals) {
  SceneGlobalsGPUSharedContext context;
  context.scene_globals = scene_globals;
  return context;
}

ETX_SHARED_INLINE uint32_t scene_globals_shared_load_u32(ETX_IN(SceneGlobalsGPUSharedContext, context), uint32_t byte_offset) {
  return context.scene_globals.Load(byte_offset);
}

ETX_SHARED_INLINE float scene_globals_shared_load_f32(ETX_IN(SceneGlobalsGPUSharedContext, context), uint32_t byte_offset) {
  return asfloat(context.scene_globals.Load(byte_offset));
}

ETX_SHARED_INLINE uint32_t scene_globals_shared_vertex_count(ETX_IN(SceneGlobalsGPUSharedContext, context)) {
  return scene_globals_shared_load_u32(context, kSceneGlobalsVertexCountOffset);
}

ETX_SHARED_INLINE uint32_t scene_globals_shared_triangle_count(ETX_IN(SceneGlobalsGPUSharedContext, context)) {
  return scene_globals_shared_load_u32(context, kSceneGlobalsTriangleCountOffset);
}

ETX_SHARED_INLINE float scene_globals_shared_bounding_sphere_radius(ETX_IN(SceneGlobalsGPUSharedContext, context)) {
  return scene_globals_shared_load_f32(context, kSceneGlobalsBoundingSphereRadiusOffset);
}

ETX_SHARED_INLINE uint32_t scene_globals_shared_environment_emitter_count(ETX_IN(SceneGlobalsGPUSharedContext, context)) {
  return scene_globals_shared_load_u32(context, kSceneGlobalsEnvironmentEmitterCountOffset);
}

ETX_SHARED_INLINE uint32_t scene_globals_shared_environment_emitter(ETX_IN(SceneGlobalsGPUSharedContext, context), uint32_t index) {
  return scene_globals_shared_load_u32(context, kSceneGlobalsEnvironmentEmittersOffset + index * 4u);
}

ETX_SHARED_INLINE uint32_t scene_globals_shared_emitter_profile_count(ETX_IN(SceneGlobalsGPUSharedContext, context)) {
  return scene_globals_shared_load_u32(context, kSceneGlobalsEmitterProfileCountOffset);
}

ETX_SHARED_INLINE uint32_t scene_globals_shared_emitter_instance_count(ETX_IN(SceneGlobalsGPUSharedContext, context)) {
  return scene_globals_shared_load_u32(context, kSceneGlobalsEmitterInstanceCountOffset);
}
