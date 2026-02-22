#pragma once

#include "interop.hxx"

#ifndef ETX_ENVIRONMENT_EMITTER_SELECT_SHARED_CONTEXT_TYPE
# error "ETX_ENVIRONMENT_EMITTER_SELECT_SHARED_CONTEXT_TYPE must be defined before including environment_emitter_select_shared.hxx"
#endif

#ifndef ETX_ENVIRONMENT_EMITTER_SELECT_SHARED_HAS_SCENE_GLOBALS
# error "ETX_ENVIRONMENT_EMITTER_SELECT_SHARED_HAS_SCENE_GLOBALS must be defined before including environment_emitter_select_shared.hxx"
#endif

#ifndef ETX_ENVIRONMENT_EMITTER_SELECT_SHARED_LOAD_EMITTER_INSTANCE_COUNT
# error "ETX_ENVIRONMENT_EMITTER_SELECT_SHARED_LOAD_EMITTER_INSTANCE_COUNT must be defined before including environment_emitter_select_shared.hxx"
#endif

#ifndef ETX_ENVIRONMENT_EMITTER_SELECT_SHARED_LOAD_ENVIRONMENT_EMITTER_COUNT
# error "ETX_ENVIRONMENT_EMITTER_SELECT_SHARED_LOAD_ENVIRONMENT_EMITTER_COUNT must be defined before including environment_emitter_select_shared.hxx"
#endif

#ifndef ETX_ENVIRONMENT_EMITTER_SELECT_SHARED_LOAD_ENVIRONMENT_EMITTER
# error "ETX_ENVIRONMENT_EMITTER_SELECT_SHARED_LOAD_ENVIRONMENT_EMITTER must be defined before including environment_emitter_select_shared.hxx"
#endif

#ifndef ETX_ENVIRONMENT_EMITTER_SELECT_SHARED_MAX_COUNT
# error "ETX_ENVIRONMENT_EMITTER_SELECT_SHARED_MAX_COUNT must be defined before including environment_emitter_select_shared.hxx"
#endif

#ifndef ETX_ENVIRONMENT_EMITTER_SELECT_SHARED_RND
# error "ETX_ENVIRONMENT_EMITTER_SELECT_SHARED_RND must be defined before including environment_emitter_select_shared.hxx"
#endif

ETX_SHARED_INLINE bool environment_emitter_select_shared_try_select_random(
  ETX_INOUT(ETX_ENVIRONMENT_EMITTER_SELECT_SHARED_CONTEXT_TYPE, context), ETX_OUT(uint32_t, emitter_index), ETX_OUT(uint32_t, emitter_count)) {
  emitter_index = kInvalidIndex;
  emitter_count = 0u;

  if (ETX_ENVIRONMENT_EMITTER_SELECT_SHARED_HAS_SCENE_GLOBALS(context) == false) {
    return false;
  }

  uint32_t emitter_instance_count = ETX_ENVIRONMENT_EMITTER_SELECT_SHARED_LOAD_EMITTER_INSTANCE_COUNT(context);
  emitter_count = ETX_ENVIRONMENT_EMITTER_SELECT_SHARED_LOAD_ENVIRONMENT_EMITTER_COUNT(context);
  uint32_t max_count = ETX_ENVIRONMENT_EMITTER_SELECT_SHARED_MAX_COUNT(context);
  if (emitter_count > max_count) {
    emitter_count = max_count;
  }
  if (emitter_count == 0u) {
    return false;
  }

  uint32_t selected = uint32_t(ETX_ENVIRONMENT_EMITTER_SELECT_SHARED_RND(context) * float(emitter_count));
  if (selected >= emitter_count) {
    selected = emitter_count - 1u;
  }

  emitter_index = ETX_ENVIRONMENT_EMITTER_SELECT_SHARED_LOAD_ENVIRONMENT_EMITTER(context, selected);
  if (emitter_index >= emitter_instance_count) {
    emitter_index = kInvalidIndex;
    return false;
  }

  return true;
}
