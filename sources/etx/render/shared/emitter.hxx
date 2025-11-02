#pragma once

#include <etx/render/shared/base.hxx>

namespace etx {

struct ETX_ALIGNED EmitterProfile {
  enum class Class : uint32_t {
    Area,
    Environment,
    Directional,

    Undefined = kInvalidIndex,
  };

  enum class Direction : uint32_t {
    Single,
    TwoSided,
    Omni,
  };

  SpectralImage emission = {};
  float3 direction = {};
  Class cls = Class::Undefined;
  Direction emission_direction = Direction::Single;
  uint32_t medium_index = kInvalidIndex;
  float collimation = 1.0f;
  float angular_size = 0.0f;
  float equivalent_disk_size = 0.0f;
  float angular_size_cosine = 1.0f;
  float pad0 = 0.0f;
  float pad1 = {};

  EmitterProfile() = default;

  EmitterProfile(Class c)
    : cls(c) {
  }

  ETX_GPU_CODE bool is_distant() const {
    return !is_local();
  }

  ETX_GPU_CODE bool is_local() const {
    return (cls == Class::Area);
  }

  ETX_GPU_CODE bool is_delta() const {
    return (cls == Class::Directional);
  }
};

struct Emitter {
  Emitter() = default;

  Emitter(EmitterProfile::Class c)
    : cls(c) {
  }

  EmitterProfile::Class cls = EmitterProfile::Class::Undefined;
  uint32_t profile = kInvalidIndex;
  uint32_t triangle_index = kInvalidIndex;
  float spectrum_weight = 0.0f;
  float additional_weight = 0.0f;
  float triangle_area = 0.0f;
  float pad0 = 0.0f;
  float pad1 = 0.0f;

  ETX_GPU_CODE bool is_distant() const {
    return is_local() == false;
  }

  ETX_GPU_CODE bool is_local() const {
    return (cls == EmitterProfile::Class::Area);
  }

  ETX_GPU_CODE bool is_delta() const {
    return (cls == EmitterProfile::Class::Directional);
  }
};

struct ETX_ALIGNED EmitterSample {
  SpectralResponse value = {};

  float3 barycentric = {};
  float pdf_sample = 0.0f;

  float3 origin = {};
  float pdf_area = 0.0f;

  float3 normal = {};
  float pdf_dir = 0.0f;

  float3 direction = {};
  float pdf_dir_out = 0.0f;

  float2 image_uv = {};
  uint32_t emitter_index = kInvalidIndex;
  uint32_t triangle_index = kInvalidIndex;
  uint32_t medium_index = kInvalidIndex;

  bool is_delta = false;
  bool is_distant = false;
};

struct EmitterRadianceQuery {
  float3 source_position = {};
  float3 target_position = {};
  float3 direction = {};
  float2 uv = {};
  bool directly_visible = false;
};

}  // namespace etx
