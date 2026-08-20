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
  float emitter_angular_size ETX_INIT(0.0f);
  float emitter_angular_size_cosine ETX_INIT(-1.0f);
};

ETX_SHARED_INLINE float4 emitter_access_environment_rotation(ETX_IN(float3, stored_xyz), float stored_w) {
  float4 rotation = float4(stored_xyz.x, stored_xyz.y, stored_xyz.z, stored_w);
  float length_squared = dot(rotation, rotation);
  return (length_squared > kEpsilon) ? (rotation / sqrt(length_squared)) : float4(0.0f, 0.0f, 0.0f, 1.0f);
}

ETX_SHARED_INLINE float3 emitter_access_rotate_direction(ETX_IN(float4, rotation), ETX_IN(float3, direction)) {
  float3 q = float3(rotation.x, rotation.y, rotation.z);
  float3 t = 2.0f * cross(q, direction);
  return direction + rotation.w * t + cross(q, t);
}

ETX_SHARED_INLINE float3 emitter_access_environment_local_to_world(ETX_IN(float3, stored_xyz), float stored_w, ETX_IN(float3, local_direction)) {
  return normalize(emitter_access_rotate_direction(emitter_access_environment_rotation(stored_xyz, stored_w), local_direction));
}

ETX_SHARED_INLINE float3 emitter_access_environment_world_to_local(ETX_IN(float3, stored_xyz), float stored_w, ETX_IN(float3, world_direction)) {
  float4 rotation = emitter_access_environment_rotation(stored_xyz, stored_w);
  float4 inverse_rotation = float4(-rotation.x, -rotation.y, -rotation.z, rotation.w);
  return normalize(emitter_access_rotate_direction(inverse_rotation, world_direction));
}

ETX_SHARED_INLINE bool emitter_access_is_atmosphere(uint32_t emitter_profile_meta) {
  return (emitter_profile_meta & EmitterProfileMeta::Atmosphere) != 0u;
}

ETX_SHARED_INLINE bool emitter_access_accepts_direction(ETX_IN(float3, direction), ETX_IN(float3, emitter_direction), float angular_size_cosine) {
  return direction_matches(direction, emitter_direction, angular_size_cosine);
}

ETX_SHARED_INLINE bool emitter_access_is_local_class(uint32_t emitter_class) {
  return emitter_class == EmitterClass::Area;
}

ETX_SHARED_INLINE bool emitter_access_accepts_distant(uint32_t emitter_class, uint32_t emitter_profile_class, ETX_IN(float3, direction), ETX_IN(float3, emitter_direction),
  float emitter_angular_size_cosine) {
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

ETX_SHARED_INLINE float2 emitter_access_shared_environment_uv(uint32_t emitter_class, uint32_t emitter_profile_meta, ETX_IN(float2, image_offset), float image_u_scale,
  ETX_IN(float3, emitter_direction), float emitter_angular_size, float emitter_angular_size_cosine, ETX_IN(float3, direction)) {
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
    float3 local_direction = emitter_access_environment_world_to_local(emitter_direction, emitter_angular_size, direction);
    return projection_environment_direction_to_uv(local_direction, image_offset, image_u_scale, is_atmosphere);
  }

  return float2(0.5f, 0.5f);
}
