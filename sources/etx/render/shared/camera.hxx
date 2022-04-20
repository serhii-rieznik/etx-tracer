#pragma once

#include <etx/render/shared/base.hxx>

namespace etx {

struct ETX_ALIGNED Camera {
  enum class Class : uint32_t {
    Perspective,
    Equirectangular,
  };

  float4x4 view_proj ETX_EMPTY_INIT;

  Class cls = Class::Perspective;

  float3 position ETX_EMPTY_INIT;
  float tan_half_fov ETX_EMPTY_INIT;

  float3 side ETX_EMPTY_INIT;
  float aspect ETX_EMPTY_INIT;

  float3 up ETX_EMPTY_INIT;
  float area ETX_EMPTY_INIT;

  float3 direction ETX_EMPTY_INIT;
  float lens_radius ETX_EMPTY_INIT;

  uint2 image_size ETX_EMPTY_INIT;
  float image_plane ETX_EMPTY_INIT;
  float focal_distance ETX_EMPTY_INIT;
};

struct CameraSample {
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

struct CameraEval {
  float3 normal ETX_EMPTY_INIT;
  float pdf_dir ETX_EMPTY_INIT;
};

}  // namespace etx
