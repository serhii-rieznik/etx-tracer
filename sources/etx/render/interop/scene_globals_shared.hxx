#pragma once

#include "gpu_abi_constants.hxx"

#ifndef ETX_SCENE_GLOBALS_SHARED_CONTEXT_TYPE
# error "ETX_SCENE_GLOBALS_SHARED_CONTEXT_TYPE must be defined before including scene_globals_shared.hxx"
#endif

#ifndef ETX_SCENE_GLOBALS_SHARED_LOAD_U32
# error "ETX_SCENE_GLOBALS_SHARED_LOAD_U32 must be defined before including scene_globals_shared.hxx"
#endif

#ifndef ETX_SCENE_GLOBALS_SHARED_LOAD_F32
# error "ETX_SCENE_GLOBALS_SHARED_LOAD_F32 must be defined before including scene_globals_shared.hxx"
#endif

ETX_SHARED_INLINE uint32_t scene_globals_shared_vertex_count(ETX_IN(ETX_SCENE_GLOBALS_SHARED_CONTEXT_TYPE, context)) {
  return ETX_SCENE_GLOBALS_SHARED_LOAD_U32(context, kSceneGlobalsVertexCountOffset);
}

ETX_SHARED_INLINE uint32_t scene_globals_shared_triangle_count(ETX_IN(ETX_SCENE_GLOBALS_SHARED_CONTEXT_TYPE, context)) {
  return ETX_SCENE_GLOBALS_SHARED_LOAD_U32(context, kSceneGlobalsTriangleCountOffset);
}

ETX_SHARED_INLINE float scene_globals_shared_bounding_sphere_radius(ETX_IN(ETX_SCENE_GLOBALS_SHARED_CONTEXT_TYPE, context)) {
  return ETX_SCENE_GLOBALS_SHARED_LOAD_F32(context, kSceneGlobalsBoundingSphereRadiusOffset);
}

ETX_SHARED_INLINE uint32_t scene_globals_shared_environment_emitter_count(ETX_IN(ETX_SCENE_GLOBALS_SHARED_CONTEXT_TYPE, context)) {
  return ETX_SCENE_GLOBALS_SHARED_LOAD_U32(context, kSceneGlobalsEnvironmentEmitterCountOffset);
}

ETX_SHARED_INLINE uint32_t scene_globals_shared_environment_emitter(ETX_IN(ETX_SCENE_GLOBALS_SHARED_CONTEXT_TYPE, context), uint32_t index) {
  return ETX_SCENE_GLOBALS_SHARED_LOAD_U32(context, kSceneGlobalsEnvironmentEmittersOffset + index * 4u);
}

ETX_SHARED_INLINE uint32_t scene_globals_shared_emitter_profile_count(ETX_IN(ETX_SCENE_GLOBALS_SHARED_CONTEXT_TYPE, context)) {
  return ETX_SCENE_GLOBALS_SHARED_LOAD_U32(context, kSceneGlobalsEmitterProfileCountOffset);
}

ETX_SHARED_INLINE uint32_t scene_globals_shared_emitter_instance_count(ETX_IN(ETX_SCENE_GLOBALS_SHARED_CONTEXT_TYPE, context)) {
  return ETX_SCENE_GLOBALS_SHARED_LOAD_U32(context, kSceneGlobalsEmitterInstanceCountOffset);
}
