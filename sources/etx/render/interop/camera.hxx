#pragma once

#include "interop.hxx"
#include "projection.hxx"
#include "ray.hxx"

struct ETX_ALIGNED Camera {
  struct Class {
    enum : uint32_t {
      Perspective,
      Equirectangular,
    };
  };

  float3 position ETX_INIT({});
  uint32_t cls ETX_INIT(Class::Perspective);
  float3 direction ETX_INIT({});
  float aspect ETX_INIT({});
  float3 side ETX_INIT({});
  float tan_half_fov ETX_INIT({});
  float3 up ETX_INIT({});
  float image_plane ETX_INIT({});
  uint2 film_size ETX_INIT({});
  float lens_radius ETX_INIT({});
  float focal_distance ETX_INIT({});
  float clip_near ETX_INIT({});
  float clip_far ETX_INIT({});
  uint32_t lens_image ETX_INIT(kInvalidIndex);
  uint32_t medium_index ETX_INIT(kInvalidIndex);
  float4x4 view_proj ETX_INIT({});
  float area ETX_INIT({});
  float pad[3] ETX_INIT({});
};

ETX_SHARED_INLINE float3 camera_target(ETX_IN(Camera, camera)) {
  return camera.position + camera.direction;
}

ETX_SHARED_INLINE Ray camera_ray_make(ETX_IN(float3, origin), ETX_IN(float3, direction), float min_t, float max_t) {
  Ray result;
  result.o = origin;
  result.min_t = min_t;
  result.d = direction;
  result.max_t = max_t;
  return result;
}

ETX_SHARED_INLINE Ray camera_generate_ray(ETX_IN(Camera, camera), ETX_IN(float2, uv), ETX_IN(float2, sensor_sample)) {
  if (camera.cls == Camera::Class::Equirectangular) {
    return camera_ray_make(camera.position, from_spherical(uv.x * kPi, uv.y * kHalfPi), kRayEpsilon, kMaxFloat);
  }

  float3 origin = camera.position;
  float3 direction = camera.direction;
  float3 s = uv.x * camera.side;
  float3 u = uv.y * camera.up / camera.aspect;
  float3 out_direction = normalize(camera.tan_half_fov * (s + u) + direction);

  if ((camera.lens_radius > kEpsilon) && (camera.focal_distance > kEpsilon)) {
    float2 lens_sample = sensor_sample * camera.lens_radius;
    origin = origin + camera.side * lens_sample.x + camera.up * lens_sample.y;
    float focal_plane_distance = camera.focal_distance / dot(out_direction, direction);
    float3 focal_point = camera.position + focal_plane_distance * out_direction;
    out_direction = normalize(focal_point - origin);
  }

  float cos_t = dot(out_direction, direction);
  float min_t = (camera.clip_near > 0.0f) ? (camera.clip_near / cos_t) : kRayEpsilon;
  float max_t = (camera.clip_far > 0.0f) ? (camera.clip_far / cos_t) : kMaxFloat;
  return camera_ray_make(origin, out_direction, max(min_t, kRayEpsilon), max_t);
}

