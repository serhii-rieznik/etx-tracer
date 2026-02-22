#pragma once

#include "camera.hxx"

ETX_SHARED_INLINE float2 camera_shared_center_uv(ETX_IN(uint2, pixel), ETX_IN(uint2, dim)) {
  return float2(
    (float(pixel.x) + 0.5f) / float(dim.x) * 2.0f - 1.0f,
    (float(pixel.y) + 0.5f) / float(dim.y) * 2.0f - 1.0f);
}

ETX_SHARED_INLINE float2 camera_shared_jittered_uv(ETX_IN(uint2, pixel), ETX_IN(uint2, dim), ETX_IN(float2, rnd)) {
  float sample_radius = 0.5f;
  return float2(
    (float(pixel.x) + 0.5f + sample_radius * (rnd.x * 2.0f - 1.0f)) / float(dim.x) * 2.0f - 1.0f,
    (float(pixel.y) + 0.5f + sample_radius * (rnd.y * 2.0f - 1.0f)) / float(dim.y) * 2.0f - 1.0f);
}

ETX_SHARED_INLINE float2 camera_shared_flip_y(ETX_IN(float2, uv)) {
  return float2(uv.x, -uv.y);
}

ETX_SHARED_INLINE float camera_shared_film_pdf_out(ETX_IN(Camera, camera), ETX_IN(float3, to_point)) {
  float3 camera_to_point = to_point - camera.position;
  float distance_squared = dot(camera_to_point, camera_to_point);
  if (distance_squared <= kEpsilon) {
    return 0.0f;
  }

  float3 w_i = camera_to_point / sqrt(distance_squared);
  if (camera.cls == Camera::Class::Equirectangular) {
    float2 uv = direction_to_uv(w_i, float2(0.0f, 0.0f), 1.0f, Projection::Equirectangular);
    return projection_environment_image_pdf_to_solid_angle(1.0f, uv, Projection::Equirectangular);
  }

  float cos_t = dot(w_i, camera.direction);
  return 1.0f / abs(camera.area * cos_t * cos_t * cos_t);
}

ETX_SHARED_INLINE float camera_shared_clip_direction_scale(ETX_IN(Camera, camera), ETX_IN(float3, direction_to_camera)) {
  if (camera.cls == Camera::Class::Equirectangular) {
    return 1.0f;
  }

  return max(kEpsilon, abs(dot(direction_to_camera, camera.direction)));
}

ETX_SHARED_INLINE float3 camera_shared_clamp_view_direction_away_from_up(
  ETX_IN(float3, view_direction), ETX_IN(float3, up_vector), ETX_IN(float3, fallback_right_vector), float min_cosine_threshold) {
  float angle_offset_degrees = 0.01f;
  float angle_offset_radians = angle_offset_degrees * kPi / 180.0f;

  float3 view_dir = normalize(view_direction);
  float up_dot = abs(dot(view_dir, up_vector));

  if (up_dot > min_cosine_threshold) {
    float3 right = cross(view_dir, up_vector);
    if (length(right) < kEpsilon) {
      right = cross(view_dir, fallback_right_vector);
    }
    right = normalize(right);
    float3 offset = right * angle_offset_radians;
    view_dir = normalize(view_dir + offset);
  }

  return view_dir;
}
