#pragma once

#include "gpu_abi_constants.hxx"

#ifndef ETX_SCENE_OPTIONS_SHARED_CONTEXT_TYPE
# error "ETX_SCENE_OPTIONS_SHARED_CONTEXT_TYPE must be defined before including scene_options_shared.hxx"
#endif

#ifndef ETX_SCENE_OPTIONS_SHARED_HAS_DATA
# error "ETX_SCENE_OPTIONS_SHARED_HAS_DATA must be defined before including scene_options_shared.hxx"
#endif

#ifndef ETX_SCENE_OPTIONS_SHARED_LOAD_U32
# error "ETX_SCENE_OPTIONS_SHARED_LOAD_U32 must be defined before including scene_options_shared.hxx"
#endif

ETX_SHARED_INLINE uint32_t scene_options_shared_samples(ETX_IN(ETX_SCENE_OPTIONS_SHARED_CONTEXT_TYPE, context)) {
  if (ETX_SCENE_OPTIONS_SHARED_HAS_DATA(context) == false) {
    return 1u;
  }

  uint32_t result = ETX_SCENE_OPTIONS_SHARED_LOAD_U32(context, kSceneOptionsSamplesOffset);
  if (result == 0u) {
    return 1u;
  }
  return result;
}

ETX_SHARED_INLINE uint32_t scene_options_shared_properties_flags(ETX_IN(ETX_SCENE_OPTIONS_SHARED_CONTEXT_TYPE, context)) {
  if (ETX_SCENE_OPTIONS_SHARED_HAS_DATA(context) == false) {
    return 0u;
  }

  return ETX_SCENE_OPTIONS_SHARED_LOAD_U32(context, kSceneOptionsPropertiesFlagsOffset);
}

ETX_SHARED_INLINE bool scene_options_shared_uses_spectral_mode(ETX_IN(ETX_SCENE_OPTIONS_SHARED_CONTEXT_TYPE, context)) {
  return (scene_options_shared_properties_flags(context) & (1u << SceneProperty::Spectral)) != 0u;
}
