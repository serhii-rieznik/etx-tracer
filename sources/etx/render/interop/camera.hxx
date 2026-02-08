#pragma once

#include "interop.hxx"

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
