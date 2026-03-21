#pragma once

struct ETX_ALIGNED EmitterAccess {
  uint32_t emitter_class ETX_INIT(EmitterClass::Area);
  uint32_t emitter_profile_index ETX_INIT(kInvalidIndex);
  uint32_t emitter_profile_class ETX_INIT(EmitterClass::Area);
  uint32_t emitter_profile_meta ETX_INIT(0u);
  uint32_t medium_index ETX_INIT(kInvalidIndex);
  uint32_t emission_spectrum_index ETX_INIT(kInvalidIndex);
  uint32_t emission_image_index ETX_INIT(kInvalidIndex);
  float3 emitter_direction ETX_INIT(float3(0.0f, 0.0f, 1.0f));
  float emitter_angular_size_cosine ETX_INIT(-1.0f);
};

ETX_SHARED_INLINE bool emitter_access_is_atmosphere(uint32_t emitter_profile_meta) {
  return (emitter_profile_meta & EmitterProfileMeta::Atmosphere) != 0u;
}

ETX_SHARED_INLINE bool emitter_access_accepts_direction(ETX_IN(float3, direction), ETX_IN(float3, emitter_direction), float angular_size_cosine) {
  return direction_matches(direction, emitter_direction, angular_size_cosine);
}

ETX_SHARED_INLINE bool emitter_access_is_local_class(uint32_t emitter_class) {
  return emitter_class == EmitterClass::Area;
}

ETX_SHARED_INLINE bool emitter_access_accepts_distant(
  uint32_t emitter_class, uint32_t emitter_profile_class, ETX_IN(float3, direction), ETX_IN(float3, emitter_direction), float emitter_angular_size_cosine) {
  if (emitter_access_is_local_class(emitter_class)) {
    return false;
  }

  if ((emitter_class == EmitterClass::Directional) && (emitter_profile_class == EmitterClass::Directional)) {
    bool accepts_direction = emitter_access_accepts_direction(direction, emitter_direction, emitter_angular_size_cosine);
    if (accepts_direction == false) {
      return false;
    }
  }

  return true;
}

ETX_SHARED_INLINE float2 emitter_access_shared_environment_uv(
  uint32_t emitter_class, uint32_t emitter_profile_meta, ETX_IN(float2, image_offset), float image_u_scale, ETX_IN(float3, emitter_direction),
  float emitter_angular_size_cosine, ETX_IN(float3, direction)) {
  if (emitter_class == EmitterClass::Directional) {
    float equivalent_disk_size = 0.0f;
    if (emitter_angular_size_cosine > kEpsilon) {
      float sin_half_angle = sqrt(max(0.0f, 1.0f - emitter_angular_size_cosine * emitter_angular_size_cosine));
      equivalent_disk_size = 2.0f * (sin_half_angle / emitter_angular_size_cosine);
    }

    return disk_uv(normalize(emitter_direction), normalize(direction), equivalent_disk_size, emitter_angular_size_cosine);
  }

  if (emitter_class == EmitterClass::Environment) {
    bool is_atmosphere = emitter_access_is_atmosphere(emitter_profile_meta);
    return projection_environment_direction_to_uv(normalize(direction), image_offset, image_u_scale, is_atmosphere);
  }

  return float2(0.5f, 0.5f);
}
