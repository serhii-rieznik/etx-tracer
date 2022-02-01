#pragma once

#include <etx/render/shared/base.hxx>

namespace etx {

struct alignas(16) Emitter {
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

  SpectralDistribution emission = {};
  Class cls = Class::Undefined;
  Direction emission_direction = Direction::Single;
  uint32_t triangle_index = kInvalidIndex;
  uint32_t medium_index = kInvalidIndex;
  uint32_t image_index = kInvalidIndex;
  float3 direction = {};
  float collimation = 1.0f;
  float angular_size = 0.0f;
  float equivalent_disk_size = 0.0f;
  float angular_size_cosine = 1.0f;
  float weight = 0.0f;

  Emitter() = default;

  Emitter(Class c)
    : cls(c) {
  }

  ETX_GPU_CODE bool is_distant() const {
    return (cls == Class::Environment) || (cls == Class::Directional);
  }

  ETX_GPU_CODE bool is_local() const {
    return (cls == Class::Area);
  }

  ETX_GPU_CODE bool is_delta() const {
    return (cls == Class::Directional);
  }
};

}  // namespace etx
