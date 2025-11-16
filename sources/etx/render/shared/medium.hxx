#pragma once

#include <etx/render/shared/spectrum.hxx>
#include <etx/render/shared/sampler.hxx>

namespace etx {

struct ETX_ALIGNED Medium {
  enum class Class : uint16_t {
    Homogeneous,
    Heterogeneous,
  };

  struct ETX_ALIGNED Instance {
    SpectralResponse extinction;
    float anisotropy = 0.0f;
    uint32_t index = kInvalidIndex;

    bool valid() const {
      return (index != kInvalidIndex) || (extinction.maximum() > 0.0f);
    }
  };

  struct ETX_ALIGNED Sample {
    SpectralResponse weight = {};
    float3 pos = {};
    float sampled_medium_t = {};

    ETX_GPU_CODE bool sampled_medium() const {
      return sampled_medium_t > 0.0f;
    }

    ETX_GPU_CODE bool valid() const {
      return weight.valid();
    }
  };

  ArrayView<float> density = {};
  BoundingBox bounds = {};
  Class cls = Class::Homogeneous;
  uint16_t enable_explicit_connections = true;
  uint32_t absorption_index = kInvalidIndex;
  uint32_t scattering_index = kInvalidIndex;
  float phase_function_g = 0.0f;
  float max_sigma = 0.0f;
  uint3 dimensions = {};
};

}  // namespace etx
