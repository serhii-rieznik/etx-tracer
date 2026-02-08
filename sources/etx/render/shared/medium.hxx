#pragma once

#include <etx/render/shared/spectrum.hxx>
#include <etx/render/shared/sampler.hxx>
#include <etx/render/shared/density_grid.hxx>

namespace etx {

struct MediumStorage {
  // Density grid storage - raw data for heterogeneous mediums
  std::vector<float> density_data;

  // Clear all storage
  void clear() {
    density_data.clear();
  }
};

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

    ETX_SHARED_INLINE bool sampled_medium() const {
      return sampled_medium_t > 0.0f;
    }

    ETX_SHARED_INLINE bool valid() const {
      return weight.valid();
    }
  };

  // View to density data (points to external storage)
  ArrayView<float> density_view;

  // Other medium properties (small data, kept in view)
  DensityGrid grid = {};
  BoundingBox bounds = {};
  Class cls = Class::Homogeneous;
  uint16_t enable_explicit_connections = true;
  uint32_t absorption_index = kInvalidIndex;
  uint32_t scattering_index = kInvalidIndex;
  float phase_function_g = 0.0f;
};

}  // namespace etx
