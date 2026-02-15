#pragma once

#include <etx/render/interop/interop.hxx>
#include <etx/render/interop/medium.hxx>
#include <etx/render/interop/medium_density_shared.hxx>

namespace etx {

enum class NoiseFunction : uint32_t {
  Perlin,
  Worley,
  Billow,
  Voronoi,
  Lattice,
  Uniform,

  Count,
};

struct ETX_ALIGNED DensityGrid {
  enum class Type : uint16_t {
    Texture3D,
    NoiseFunction,
  };

  ArrayView<float> density = {};

  ETX_SHARED_INLINE MediumDensitySharedGrid to_shared_grid(const MediumGrid& grid, uint32_t density_count_override) const {
    MediumDensitySharedGrid result = {};
    result.dimensions = grid.dimensions;
    result.type = grid.type;
    result.noise_type = grid.noise_type;
    result.density_data_offset = grid.density_data_offset;
    result.density_count = density_count_override;
    result.noise_seed = grid.noise_seed;
    result.noise_offset = grid.noise_offset;
    result.noise_enable_border_fade = grid.noise_enable_border_fade;
    result.noise_octaves = grid.noise_octaves;
    result.noise_scale = grid.noise_scale;
    result.noise_lacunarity = grid.noise_lacunarity;
    result.noise_persistence = grid.noise_persistence;
    result.noise_power = grid.noise_power;
    result.noise_sharpness = grid.noise_sharpness;
    result.noise_border_fade_distance = grid.noise_border_fade_distance;
    result.density_data_chunk_index = grid.density_data_chunk_index;
    return result;
  }

  ETX_SHARED_INLINE float sample_texture_3d(const float3& local_coord, const uint3& dimensions) const {
    MediumDensitySharedTextureSample3D sample = {};
    if (medium_density_shared_prepare_texture_sample_3d(local_coord, dimensions, sample) == false) {
      return 0.0f;
    }

    float d000 = density[sample.ix + sample.iy * dimensions.x + sample.iz * dimensions.x * dimensions.y];
    float d001 = density[sample.nx + sample.iy * dimensions.x + sample.iz * dimensions.x * dimensions.y];
    float d010 = density[sample.ix + sample.ny * dimensions.x + sample.iz * dimensions.x * dimensions.y];
    float d011 = density[sample.nx + sample.ny * dimensions.x + sample.iz * dimensions.x * dimensions.y];
    float d100 = density[sample.ix + sample.iy * dimensions.x + sample.nz * dimensions.x * dimensions.y];
    float d101 = density[sample.nx + sample.iy * dimensions.x + sample.nz * dimensions.x * dimensions.y];
    float d110 = density[sample.ix + sample.ny * dimensions.x + sample.nz * dimensions.x * dimensions.y];
    float d111 = density[sample.nx + sample.ny * dimensions.x + sample.nz * dimensions.x * dimensions.y];
    return medium_density_shared_trilerp(d000, d001, d010, d011, d100, d101, d110, d111, sample.dx, sample.dy, sample.dz);
  }

  ETX_SHARED_INLINE float sample_noise(const float3& local_coord, const BoundingBox& bounds, const MediumGrid& grid) const {
    MediumDensitySharedGrid shared_grid = to_shared_grid(grid, grid.density_count);
    return medium_density_shared_sample_noise(local_coord, bounds.p_min, bounds.p_max, shared_grid.noise_type, shared_grid.noise_scale, shared_grid.noise_octaves,
      shared_grid.noise_lacunarity, shared_grid.noise_persistence, shared_grid.noise_seed, shared_grid.noise_offset, shared_grid.noise_enable_border_fade,
      shared_grid.noise_border_fade_distance);
  }

  ETX_SHARED_INLINE float sample(const float3& local_coord, const BoundingBox& bounds, const MediumGrid& grid) const {
    float value = 0.0f;
    const Type type = static_cast<Type>(grid.type);
    if (type == Type::NoiseFunction) {
      value = sample_noise(local_coord, bounds, grid);
    } else if (type == Type::Texture3D) {
      value = sample_texture_3d(local_coord, grid.dimensions);
    }
    MediumDensitySharedGrid shared_grid = to_shared_grid(grid, grid.density_count);
    return medium_density_shared_apply_shape(value, shared_grid.noise_power, shared_grid.noise_sharpness);
  }

  ETX_SHARED_INLINE bool has_data(const MediumGrid& grid) const {
    uint32_t density_count = (density.count > 0ull) ? 1u : 0u;
    MediumDensitySharedGrid shared_grid = to_shared_grid(grid, density_count);
    return medium_density_shared_has_grid_data(shared_grid.type, shared_grid.dimensions, shared_grid.density_count);
  }
};

}  // namespace etx
