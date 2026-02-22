#pragma once

#include "gpu_abi_constants.hxx"

#ifndef ETX_EMITTER_MEDIUM_SHARED_CONTEXT_TYPE
# error "ETX_EMITTER_MEDIUM_SHARED_CONTEXT_TYPE must be defined before including emitter_medium_shared.hxx"
#endif

#ifndef ETX_EMITTER_MEDIUM_SHARED_HAS_REQUIRED_SCENE_BUFFERS
# error "ETX_EMITTER_MEDIUM_SHARED_HAS_REQUIRED_SCENE_BUFFERS must be defined before including emitter_medium_shared.hxx"
#endif

#ifndef ETX_EMITTER_MEDIUM_SHARED_LOAD_EMITTER_PROFILE_COUNT
# error "ETX_EMITTER_MEDIUM_SHARED_LOAD_EMITTER_PROFILE_COUNT must be defined before including emitter_medium_shared.hxx"
#endif

#ifndef ETX_EMITTER_MEDIUM_SHARED_LOAD_TRIANGLE_COUNT
# error "ETX_EMITTER_MEDIUM_SHARED_LOAD_TRIANGLE_COUNT must be defined before including emitter_medium_shared.hxx"
#endif

#ifndef ETX_EMITTER_MEDIUM_SHARED_LOAD_MATERIAL_COUNT
# error "ETX_EMITTER_MEDIUM_SHARED_LOAD_MATERIAL_COUNT must be defined before including emitter_medium_shared.hxx"
#endif

#ifndef ETX_EMITTER_MEDIUM_SHARED_LOAD_PROFILE_MEDIUM_INDEX
# error "ETX_EMITTER_MEDIUM_SHARED_LOAD_PROFILE_MEDIUM_INDEX must be defined before including emitter_medium_shared.hxx"
#endif

#ifndef ETX_EMITTER_MEDIUM_SHARED_LOAD_TRIANGLE_MATERIAL_INDEX
# error "ETX_EMITTER_MEDIUM_SHARED_LOAD_TRIANGLE_MATERIAL_INDEX must be defined before including emitter_medium_shared.hxx"
#endif

#ifndef ETX_EMITTER_MEDIUM_SHARED_LOAD_MATERIAL_EXT_MEDIUM_INDEX
# error "ETX_EMITTER_MEDIUM_SHARED_LOAD_MATERIAL_EXT_MEDIUM_INDEX must be defined before including emitter_medium_shared.hxx"
#endif

ETX_SHARED_INLINE uint32_t emitter_medium_shared_external_index(
  ETX_IN(ETX_EMITTER_MEDIUM_SHARED_CONTEXT_TYPE, context), uint32_t emitter_class, uint32_t emitter_profile_index, uint32_t emitter_triangle_index) {
  if (ETX_EMITTER_MEDIUM_SHARED_HAS_REQUIRED_SCENE_BUFFERS(context) == false) {
    return kInvalidIndex;
  }

  if (emitter_class == EmitterClass::Area) {
    uint32_t triangle_count = ETX_EMITTER_MEDIUM_SHARED_LOAD_TRIANGLE_COUNT(context);
    if (emitter_triangle_index >= triangle_count) {
      return kInvalidIndex;
    }

    uint32_t material_index = ETX_EMITTER_MEDIUM_SHARED_LOAD_TRIANGLE_MATERIAL_INDEX(context, emitter_triangle_index);
    uint32_t material_count = ETX_EMITTER_MEDIUM_SHARED_LOAD_MATERIAL_COUNT(context);
    if (material_index >= material_count) {
      return kInvalidIndex;
    }

    return ETX_EMITTER_MEDIUM_SHARED_LOAD_MATERIAL_EXT_MEDIUM_INDEX(context, material_index);
  }

  uint32_t emitter_profile_count = ETX_EMITTER_MEDIUM_SHARED_LOAD_EMITTER_PROFILE_COUNT(context);
  if (emitter_profile_index >= emitter_profile_count) {
    return kInvalidIndex;
  }

  return ETX_EMITTER_MEDIUM_SHARED_LOAD_PROFILE_MEDIUM_INDEX(context, emitter_profile_index);
}
