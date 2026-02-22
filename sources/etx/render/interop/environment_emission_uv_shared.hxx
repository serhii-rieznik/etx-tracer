#pragma once

#include "projection.hxx"
#include "emitter_emission_shared.hxx"

#ifndef ETX_ENVIRONMENT_EMISSION_UV_SHARED_CONTEXT_TYPE
# error "ETX_ENVIRONMENT_EMISSION_UV_SHARED_CONTEXT_TYPE must be defined before including environment_emission_uv_shared.hxx"
#endif

#ifndef ETX_ENVIRONMENT_EMISSION_UV_SHARED_IS_ENVIRONMENT_CLASS
# error "ETX_ENVIRONMENT_EMISSION_UV_SHARED_IS_ENVIRONMENT_CLASS must be defined before including environment_emission_uv_shared.hxx"
#endif

#ifndef ETX_ENVIRONMENT_EMISSION_UV_SHARED_IS_DIRECTIONAL_CLASS
# error "ETX_ENVIRONMENT_EMISSION_UV_SHARED_IS_DIRECTIONAL_CLASS must be defined before including environment_emission_uv_shared.hxx"
#endif

#ifndef ETX_ENVIRONMENT_EMISSION_UV_SHARED_TRY_LOAD_IMAGE_PARAMS
# error "ETX_ENVIRONMENT_EMISSION_UV_SHARED_TRY_LOAD_IMAGE_PARAMS must be defined before including environment_emission_uv_shared.hxx"
#endif

ETX_SHARED_INLINE float2 environment_emission_uv_shared(
  ETX_IN(ETX_ENVIRONMENT_EMISSION_UV_SHARED_CONTEXT_TYPE, context), uint32_t emitter_class, uint32_t emitter_profile_meta, uint32_t emission_image_index,
  ETX_IN(float3, emitter_direction), float emitter_angular_size_cosine, ETX_IN(float3, direction)) {
  if (ETX_ENVIRONMENT_EMISSION_UV_SHARED_IS_DIRECTIONAL_CLASS(emitter_class)) {
    float equivalent_disk_size = 0.0f;
    if (emitter_angular_size_cosine > kEpsilon) {
      float sin_half_angle = sqrt(max(0.0f, 1.0f - emitter_angular_size_cosine * emitter_angular_size_cosine));
      equivalent_disk_size = 2.0f * (sin_half_angle / emitter_angular_size_cosine);
    }

    return disk_uv(normalize(emitter_direction), normalize(direction), equivalent_disk_size, emitter_angular_size_cosine);
  }

  if (ETX_ENVIRONMENT_EMISSION_UV_SHARED_IS_ENVIRONMENT_CLASS(emitter_class)) {
    float2 image_offset = float2(0.0f, 0.0f);
    float image_u_scale = 1.0f;
    ETX_ENVIRONMENT_EMISSION_UV_SHARED_TRY_LOAD_IMAGE_PARAMS(context, emission_image_index, image_offset, image_u_scale);

    bool is_atmosphere = emitter_emission_shared_is_atmosphere(emitter_profile_meta);
    return projection_environment_direction_to_uv(normalize(direction), image_offset, image_u_scale, is_atmosphere);
  }

  return float2(0.5f, 0.5f);
}
