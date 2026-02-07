#pragma once

#include "interop.hxx"

struct ETX_ALIGNED Camera {
  struct Class {
    enum : uint32_t {
      Perspective,
      Equirectangular,
    };
  };

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
};

ETX_GPU_CODE float3 camera_target(ETX_IN(Camera, camera)) {
  return camera.position + camera.direction;
}
