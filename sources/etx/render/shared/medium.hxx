#pragma once

#include <etx/render/interop/medium.hxx>
#include <etx/render/shared/spectrum.hxx>
#include <etx/render/shared/buffer_view.hxx>
#include <etx/render/shared/sampler.hxx>
#include <etx/render/shared/density_grid.hxx>

namespace etx {

using MediumInstance = ::MediumInstance;
using MediumSample = ::MediumSample;

struct ETX_ALIGNED Medium {
  using Class = ::Medium::Class;
  enum : uint16_t {
    Homogeneous = ::Medium::Homogeneous,
    Heterogeneous = ::Medium::Heterogeneous,
  };

  MediumGrid grid = {};
  BoundingBox bounds = {};
  uint32_t absorption_index = kInvalidIndex;
  uint32_t scattering_index = kInvalidIndex;
  float phase_function_g = 0.0f;
  uint16_t enable_explicit_connections = 1u;
  Class cls = Homogeneous;
  AffineTransform world_to_object = {};
  BoundingBox local_bounds = {};

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

  ETX_SHARED_INLINE float sample_density_world(const float3& world_position) const {
    const float3 local_coord = medium_world_to_local(world_to_object, local_bounds, world_position);
    if (medium_local_coordinate_valid(local_coord) == false) {
      return 0.0f;
    }
    return sample_density(local_coord, local_bounds);
  }
};

}  // namespace etx
