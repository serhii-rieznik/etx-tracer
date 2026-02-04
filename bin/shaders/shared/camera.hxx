#ifndef ETX_RENDER_SHARED_CAMERA_HXX
#define ETX_RENDER_SHARED_CAMERA_HXX

#if !defined(ETX_ALIGNED)
# define ETX_ALIGNED
#endif

#if !defined(ETX_EMPTY_INIT)
# define ETX_EMPTY_INIT
#endif

#if !defined(ETX_INIT_WITH)
# define ETX_INIT_WITH(x)
#endif

#if !defined(ETX_GPU_CODE)
# define ETX_GPU_CODE
#endif

// Forward declare or assume types if not included.
// In HLSL context, these are built-ins.
// In C++ context, they come from math libs, but this file is meant for Shader use mainly here.
// However, C++ logic uses `shared/camera.hxx`, so we are fixing the SHADER side include.

struct ETX_ALIGNED Camera {
  float3 position;
  uint cls;

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
  uint lens_image;
  uint medium_index;

  float4x4 view_proj;
  float area;
  float pad[3];

  ETX_GPU_CODE float3 target() {
    return position + direction;
  }
};

#endif
