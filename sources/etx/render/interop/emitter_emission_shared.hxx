#pragma once

#include "math_shared.hxx"
#include "gpu_abi_constants.hxx"

ETX_SHARED_INLINE bool emitter_emission_shared_is_atmosphere(uint32_t emitter_profile_meta) {
  return (emitter_profile_meta & EmitterProfileMeta::Atmosphere) != 0u;
}

ETX_SHARED_INLINE bool emitter_emission_shared_accepts_direction(ETX_IN(float3, direction), ETX_IN(float3, emitter_direction), float angular_size_cosine) {
  return dot(normalize(direction), normalize(emitter_direction)) >= angular_size_cosine;
}

ETX_SHARED_INLINE bool emitter_emission_shared_is_local_class(uint32_t emitter_class) {
  return emitter_class == EmitterClass::Area;
}

ETX_SHARED_INLINE bool emitter_emission_shared_accepts_distant_access(
  uint32_t emitter_class, uint32_t emitter_profile_class, ETX_IN(float3, direction), ETX_IN(float3, emitter_direction), float emitter_angular_size_cosine) {
  if (emitter_emission_shared_is_local_class(emitter_class)) {
    return false;
  }

  if ((emitter_class == EmitterClass::Directional) && (emitter_profile_class == EmitterClass::Directional)) {
    bool accepts_direction = emitter_emission_shared_accepts_direction(direction, emitter_direction, emitter_angular_size_cosine);
    if (accepts_direction == false) {
      return false;
    }
  }

  return true;
}

#if defined(ETX_EMITTER_EMISSION_SHARED_CONTEXT_TYPE)

# if !defined(ETX_EMITTER_EMISSION_SHARED_ACCESS_TYPE)
#  error "ETX_EMITTER_EMISSION_SHARED_ACCESS_TYPE must be defined before including emitter_emission_shared.hxx"
# endif

# if !defined(ETX_EMITTER_EMISSION_SHARED_HAS_REQUIRED_SCENE_BUFFERS)
#  error "ETX_EMITTER_EMISSION_SHARED_HAS_REQUIRED_SCENE_BUFFERS must be defined before including emitter_emission_shared.hxx"
# endif

# if !defined(ETX_EMITTER_EMISSION_SHARED_LOAD_EMITTER_INSTANCE_COUNT)
#  error "ETX_EMITTER_EMISSION_SHARED_LOAD_EMITTER_INSTANCE_COUNT must be defined before including emitter_emission_shared.hxx"
# endif

# if !defined(ETX_EMITTER_EMISSION_SHARED_LOAD_EMITTER_PROFILE_COUNT)
#  error "ETX_EMITTER_EMISSION_SHARED_LOAD_EMITTER_PROFILE_COUNT must be defined before including emitter_emission_shared.hxx"
# endif

# if !defined(ETX_EMITTER_EMISSION_SHARED_LOAD_INSTANCE_CLASS)
#  error "ETX_EMITTER_EMISSION_SHARED_LOAD_INSTANCE_CLASS must be defined before including emitter_emission_shared.hxx"
# endif

# if !defined(ETX_EMITTER_EMISSION_SHARED_LOAD_INSTANCE_PROFILE_INDEX)
#  error "ETX_EMITTER_EMISSION_SHARED_LOAD_INSTANCE_PROFILE_INDEX must be defined before including emitter_emission_shared.hxx"
# endif

# if !defined(ETX_EMITTER_EMISSION_SHARED_LOAD_PROFILE_EMISSION_SPECTRUM_INDEX)
#  error "ETX_EMITTER_EMISSION_SHARED_LOAD_PROFILE_EMISSION_SPECTRUM_INDEX must be defined before including emitter_emission_shared.hxx"
# endif

# if !defined(ETX_EMITTER_EMISSION_SHARED_LOAD_PROFILE_EMISSION_IMAGE_INDEX)
#  error "ETX_EMITTER_EMISSION_SHARED_LOAD_PROFILE_EMISSION_IMAGE_INDEX must be defined before including emitter_emission_shared.hxx"
# endif

# if !defined(ETX_EMITTER_EMISSION_SHARED_LOAD_PROFILE_CLASS)
#  error "ETX_EMITTER_EMISSION_SHARED_LOAD_PROFILE_CLASS must be defined before including emitter_emission_shared.hxx"
# endif

# if !defined(ETX_EMITTER_EMISSION_SHARED_LOAD_PROFILE_META)
#  error "ETX_EMITTER_EMISSION_SHARED_LOAD_PROFILE_META must be defined before including emitter_emission_shared.hxx"
# endif

# if !defined(ETX_EMITTER_EMISSION_SHARED_LOAD_PROFILE_DIRECTION)
#  error "ETX_EMITTER_EMISSION_SHARED_LOAD_PROFILE_DIRECTION must be defined before including emitter_emission_shared.hxx"
# endif

# if !defined(ETX_EMITTER_EMISSION_SHARED_LOAD_PROFILE_ANGULAR_SIZE_COSINE)
#  error "ETX_EMITTER_EMISSION_SHARED_LOAD_PROFILE_ANGULAR_SIZE_COSINE must be defined before including emitter_emission_shared.hxx"
# endif

ETX_SHARED_INLINE bool emitter_emission_shared_try_load_scene_state(
  ETX_IN(ETX_EMITTER_EMISSION_SHARED_CONTEXT_TYPE, context), ETX_OUT(uint32_t, emitter_instance_count), ETX_OUT(uint32_t, emitter_profile_count)) {
  emitter_instance_count = 0u;
  emitter_profile_count = 0u;

  if (ETX_EMITTER_EMISSION_SHARED_HAS_REQUIRED_SCENE_BUFFERS(context) == false) {
    return false;
  }

  emitter_instance_count = ETX_EMITTER_EMISSION_SHARED_LOAD_EMITTER_INSTANCE_COUNT(context);
  emitter_profile_count = ETX_EMITTER_EMISSION_SHARED_LOAD_EMITTER_PROFILE_COUNT(context);
  return true;
}

ETX_SHARED_INLINE void emitter_emission_shared_set_default_access(ETX_OUT(ETX_EMITTER_EMISSION_SHARED_ACCESS_TYPE, access)) {
  access.emitter_class = EmitterClass::Area;
  access.emitter_profile_index = kInvalidIndex;
  access.emitter_profile_class = EmitterClass::Area;
  access.emitter_profile_meta = 0u;
  access.emission_spectrum_index = kInvalidIndex;
  access.emission_image_index = kInvalidIndex;
  access.emitter_direction = float3(0.0f, 0.0f, 1.0f);
  access.emitter_angular_size_cosine = -1.0f;
}

ETX_SHARED_INLINE bool emitter_emission_shared_try_load_profile_access(
  ETX_IN(ETX_EMITTER_EMISSION_SHARED_CONTEXT_TYPE, context), uint32_t emitter_profile_count, ETX_INOUT(ETX_EMITTER_EMISSION_SHARED_ACCESS_TYPE, access)) {
  if (access.emitter_profile_index >= emitter_profile_count) {
    return false;
  }

  access.emission_spectrum_index = ETX_EMITTER_EMISSION_SHARED_LOAD_PROFILE_EMISSION_SPECTRUM_INDEX(context, access.emitter_profile_index);
  access.emission_image_index = ETX_EMITTER_EMISSION_SHARED_LOAD_PROFILE_EMISSION_IMAGE_INDEX(context, access.emitter_profile_index);
  if (access.emission_spectrum_index == kInvalidIndex) {
    return false;
  }

  access.emitter_profile_class = ETX_EMITTER_EMISSION_SHARED_LOAD_PROFILE_CLASS(context, access.emitter_profile_index);
  access.emitter_profile_meta = ETX_EMITTER_EMISSION_SHARED_LOAD_PROFILE_META(context, access.emitter_profile_index);
  access.emitter_direction = ETX_EMITTER_EMISSION_SHARED_LOAD_PROFILE_DIRECTION(context, access.emitter_profile_index);
  access.emitter_angular_size_cosine = ETX_EMITTER_EMISSION_SHARED_LOAD_PROFILE_ANGULAR_SIZE_COSINE(context, access.emitter_profile_index);
  return true;
}

ETX_SHARED_INLINE bool emitter_emission_shared_try_load_access_from_instance(
  ETX_IN(ETX_EMITTER_EMISSION_SHARED_CONTEXT_TYPE, context), uint32_t emitter_class, uint32_t emitter_profile_index,
  ETX_OUT(ETX_EMITTER_EMISSION_SHARED_ACCESS_TYPE, access)) {
  emitter_emission_shared_set_default_access(access);

  uint32_t emitter_instance_count = 0u;
  uint32_t emitter_profile_count = 0u;
  if (emitter_emission_shared_try_load_scene_state(context, emitter_instance_count, emitter_profile_count) == false) {
    return false;
  }
  (void)emitter_instance_count;

  access.emitter_class = emitter_class;
  access.emitter_profile_index = emitter_profile_index;
  return emitter_emission_shared_try_load_profile_access(context, emitter_profile_count, access);
}

ETX_SHARED_INLINE bool emitter_emission_shared_try_load_access(
  ETX_IN(ETX_EMITTER_EMISSION_SHARED_CONTEXT_TYPE, context), uint32_t emitter_index, ETX_OUT(ETX_EMITTER_EMISSION_SHARED_ACCESS_TYPE, access)) {
  emitter_emission_shared_set_default_access(access);

  uint32_t emitter_instance_count = 0u;
  uint32_t emitter_profile_count = 0u;
  if (emitter_emission_shared_try_load_scene_state(context, emitter_instance_count, emitter_profile_count) == false) {
    return false;
  }

  if (emitter_index >= emitter_instance_count) {
    return false;
  }

  access.emitter_class = ETX_EMITTER_EMISSION_SHARED_LOAD_INSTANCE_CLASS(context, emitter_index);
  access.emitter_profile_index = ETX_EMITTER_EMISSION_SHARED_LOAD_INSTANCE_PROFILE_INDEX(context, emitter_index);
  return emitter_emission_shared_try_load_profile_access(context, emitter_profile_count, access);
}

ETX_SHARED_INLINE bool emitter_emission_shared_try_load_local_access(
  ETX_IN(ETX_EMITTER_EMISSION_SHARED_CONTEXT_TYPE, context), uint32_t emitter_index, ETX_OUT(ETX_EMITTER_EMISSION_SHARED_ACCESS_TYPE, access)) {
  if (emitter_emission_shared_try_load_access(context, emitter_index, access) == false) {
    return false;
  }

  return emitter_emission_shared_is_local_class(access.emitter_class);
}

ETX_SHARED_INLINE bool emitter_emission_shared_try_load_distant_access(
  ETX_IN(ETX_EMITTER_EMISSION_SHARED_CONTEXT_TYPE, context), uint32_t emitter_index, ETX_IN(float3, direction), ETX_OUT(ETX_EMITTER_EMISSION_SHARED_ACCESS_TYPE, access)) {
  if (emitter_emission_shared_try_load_access(context, emitter_index, access) == false) {
    return false;
  }
  return emitter_emission_shared_accepts_distant_access(
    access.emitter_class, access.emitter_profile_class, direction, access.emitter_direction, access.emitter_angular_size_cosine);
}

#endif
