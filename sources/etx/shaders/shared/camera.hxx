#pragma once

#include "types.hxx"

struct ETX_ALIGNED Camera {
  float3 position;
  uint32_t cls;

  float3 direction;
  float aspect;

  float3 side;
  float tan_half_fov;

  float3 up;
  float image_plane;

  uint2 film_size;
  float lens_radius;
  float focal_distance;

  float clip_near;
  float clip_far;
  uint32_t lens_image;
  uint32_t medium_index;

  float4x4 view_proj;
  float area;
  float pad[3];

  // ETX_GPU_CODE float3 target() { return {position.x + direction.x, position.y + direction.y, position.z + direction.z}; }
};

#if defined(__cplusplus)

# include <etx/render/shared/camera.hxx>

namespace etx {
inline ::Camera to_shader_camera(const etx::Camera& source) {
  ::Camera result = {};
  result.position = source.position;
  result.cls = static_cast<uint32_t>(source.cls);
  result.direction = source.direction;
  result.aspect = source.aspect;
  result.side = source.side;
  result.tan_half_fov = source.tan_half_fov;
  result.up = source.up;
  result.image_plane = source.image_plane;
  result.film_size = source.film_size;
  result.lens_radius = source.lens_radius;
  result.focal_distance = source.focal_distance;
  result.clip_near = source.clip_near;
  result.clip_far = source.clip_far;
  result.lens_image = source.lens_image;
  result.medium_index = source.medium_index;
  result.view_proj = source.view_proj;
  result.area = source.area;
  result.pad[0] = source.pad[0];
  result.pad[1] = source.pad[1];
  result.pad[2] = source.pad[2];
  return result;
}
}  // namespace etx
#endif
