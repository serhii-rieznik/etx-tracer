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

#if defined(__cplusplus)
static_assert(std::is_standard_layout_v<Camera>, "Camera must stay standard layout for C++/HLSL interop");
static_assert(alignof(Camera) == 16, "Camera alignment must match HLSL packing");
static_assert(sizeof(Camera) == 176, "Camera size changed; update shared ABI or padding");
static_assert(offsetof(Camera, film_size) == 64, "Camera::film_size offset changed");
static_assert(offsetof(Camera, view_proj) == 96, "Camera::view_proj offset changed");
#endif
