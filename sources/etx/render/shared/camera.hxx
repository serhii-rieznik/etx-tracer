#pragma once

#include <etx/core/core.hxx>
#include <etx/render/shared/base.hxx>

namespace etx {

struct ETX_ALIGNED Camera {
  enum class Class : uint32_t {
    Perspective,
    Equirectangular,
  };

  float4x4 view_proj ETX_EMPTY_INIT;

  float3 position ETX_EMPTY_INIT;
  Class cls ETX_INIT_WITH(Class::Perspective);

  float3 target ETX_EMPTY_INIT;
  float tan_half_fov ETX_EMPTY_INIT;

  float3 side ETX_EMPTY_INIT;
  float aspect ETX_EMPTY_INIT;

  float3 up ETX_EMPTY_INIT;
  float area ETX_EMPTY_INIT;

  float3 direction ETX_EMPTY_INIT;
  float image_plane ETX_EMPTY_INIT;

  uint2 film_size ETX_EMPTY_INIT;
  float lens_radius ETX_EMPTY_INIT;
  float focal_distance ETX_EMPTY_INIT;

  float clip_near ETX_INIT_WITH(1.0f / 256.0f);
  float clip_far ETX_INIT_WITH(1024.0f);
  uint32_t lens_image ETX_INIT_WITH(kInvalidIndex);
  uint32_t medium_index ETX_INIT_WITH(kInvalidIndex);
};

inline constexpr float3 kWorldRight = {1.0f, 0.0f, 0.0f};
inline constexpr float3 kWorldUp = {0.0f, 1.0f, 0.0f};
inline constexpr float3 kWorldForward = {0.0f, 0.0f, -1.0f};

struct ETX_ALIGNED CameraSample {
  float3 position ETX_EMPTY_INIT;
  float3 normal ETX_EMPTY_INIT;
  float3 direction ETX_EMPTY_INIT;
  float2 uv ETX_EMPTY_INIT;
  float weight ETX_EMPTY_INIT;
  float pdf_dir ETX_EMPTY_INIT;
  float pdf_area ETX_EMPTY_INIT;
  float pdf_dir_out ETX_EMPTY_INIT;

  ETX_GPU_CODE bool valid() const {
    return (pdf_dir > 0.0f) && (weight > 0.0f);
  }
};

struct ETX_ALIGNED CameraEval {
  float3 normal ETX_EMPTY_INIT;
  float pdf_dir ETX_EMPTY_INIT;
};

struct PixelFilter {
  uint32_t image_index = kInvalidIndex;
  float radius = 1.0f;

  static PixelFilter empty() {
    return {kInvalidIndex, 0.0f};
  }
};

}  // namespace etx
