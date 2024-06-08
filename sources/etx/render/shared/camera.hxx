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

  uint2 image_size ETX_EMPTY_INIT;
  uint2 pad ETX_EMPTY_INIT;
};

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

struct ETX_ALIGNED FilmData {
  ArrayView<float4> data ETX_EMPTY_INIT;
  uint2 dimensions ETX_EMPTY_INIT;

  ETX_GPU_CODE void accumulate(const float4& value, uint32_t x, uint32_t y, float t) {
    if ((x >= dimensions.x) || (y >= dimensions.y)) {
      return;
    }
    ETX_VALIDATE(value);
    uint32_t i = x + (dimensions.y - 1 - y) * dimensions.x;
    data[i] = (t <= 0.0f) ? value : lerp(value, data[i], t);
  }

  ETX_GPU_CODE void accumulate(const float4& value, const float2& ndc_coord, float t) {
    float2 uv = ndc_coord * 0.5f + 0.5f;
    uint32_t ax = static_cast<uint32_t>(uv.x * float(dimensions.x));
    uint32_t ay = static_cast<uint32_t>(uv.y * float(dimensions.y));
    accumulate(value, ax, ay, t);
  }

  ETX_GPU_CODE void atomic_add(const float4& value, uint32_t x, uint32_t y) {
    if ((x >= dimensions.x) || (y >= dimensions.y)) {
      return;
    }

    ETX_VALIDATE(value);
    auto& ptr = data[x + 1llu * y * dimensions.x];
    atomic_add_float(&ptr.x, value.x);
    atomic_add_float(&ptr.y, value.y);
    atomic_add_float(&ptr.z, value.z);
  }
};

struct PixelSampler {
  uint32_t image_index = kInvalidIndex;
  float radius = 1.0f;

  static PixelSampler empty() {
    return {kInvalidIndex, 0.0f};
  }
};

struct Lens : PixelSampler {
  float focal_distance = 0.0f;
};

}  // namespace etx
