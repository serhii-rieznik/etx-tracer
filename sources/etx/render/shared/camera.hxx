#pragma once

#include <etx/render/interop/camera.hxx>

namespace etx {

using Camera = ::Camera;

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

  ETX_SHARED_INLINE bool valid() const {
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
