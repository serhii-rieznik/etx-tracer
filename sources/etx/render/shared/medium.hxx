#pragma once

#include <etx/render/interop/medium.hxx>
#include <etx/render/shared/spectrum.hxx>
#include <etx/render/shared/buffer_view.hxx>
#include <etx/render/shared/sampler.hxx>
#include <etx/render/shared/density_grid.hxx>

namespace etx {

using MediumInstance = ::MediumInstance;
using MediumSample = ::MediumSample;

struct ETX_ALIGNED Medium : public ::Medium {
  ArrayView<float> density_view;
  BufferHandle density_buffer = {};
  BufferView density_data = {};

  ETX_SHARED_INLINE DensityGrid::Type grid_type_enum() const {
    const auto type = static_cast<DensityGrid::Type>(grid.type);
    if ((type != DensityGrid::Type::Texture3D) && (type != DensityGrid::Type::NoiseFunction)) {
      return DensityGrid::Type::Texture3D;
    }
    return type;
  }

  ETX_SHARED_INLINE void set_grid_type(DensityGrid::Type type) {
    grid.type = static_cast<uint32_t>(type);
  }

  ETX_SHARED_INLINE NoiseFunction noise_type_enum() const {
    if (grid.noise_type >= static_cast<uint32_t>(NoiseFunction::Count)) {
      return NoiseFunction::Perlin;
    }
    return static_cast<NoiseFunction>(grid.noise_type);
  }

  ETX_SHARED_INLINE void set_noise_type(NoiseFunction type) {
    grid.noise_type = static_cast<uint32_t>(type);
  }

  ETX_SHARED_INLINE bool has_grid_data() const {
    DensityGrid density_grid = {};
    density_grid.density = density_view;
    return density_grid.has_data(grid);
  }

  ETX_SHARED_INLINE float sample_density(const float3& local_coord, const BoundingBox& bounds) const {
    DensityGrid density_grid = {};
    density_grid.density = density_view;
    return density_grid.sample(local_coord, bounds, grid);
  }
};

}  // namespace etx
