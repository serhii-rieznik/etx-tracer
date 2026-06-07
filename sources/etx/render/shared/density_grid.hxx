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

struct ETX_ALIGNED MediumGrid {
  uint3 dimensions = {};
  uint32_t type = MediumGridType::Texture3D;

  uint32_t noise_type = MediumNoiseType::Perlin;
  uint32_t noise_seed = 0u;

  float3 noise_offset = {};
  uint32_t noise_enable_border_fade = 0u;

  uint32_t noise_octaves = 1u;
  float noise_scale = 1.0f;
  float noise_lacunarity = 2.0f;
  float noise_persistence = 0.5f;

  float noise_power = 1.0f;
  float noise_sharpness = 1.0f;
  float noise_border_fade_distance = 0.1f;
  uint32_t density_image_index = kInvalidIndex;
};

struct MediumTextureSampleContext {
  ArrayView<float> density = {};
};

ETX_SHARED_INLINE float medium_texture_sample_density(ETX_INOUT(MediumTextureSampleContext, context), ETX_IN(uint3, dimensions), uint32_t x, uint32_t y, uint32_t z) {
  uint32_t index = x + y * dimensions.x + z * dimensions.x * dimensions.y;
  return context.density[index];
}

#include <etx/render/interop/medium_texture_sample_shared.hxx>

ETX_SHARED_INLINE float density_grid_sample_direct(ETX_IN(ArrayView<float>, density), ETX_IN(MediumDensitySharedGrid, grid), ETX_IN(float3, bounds_min), ETX_IN(float3, bounds_max),
  ETX_IN(float3, local_coord)) {
  float value = 0.0f;
  if (grid.type == MediumGridType::NoiseFunction) {
    value = medium_density_shared_sample_noise(local_coord, bounds_min, bounds_max, grid.noise_type, grid.noise_scale, grid.noise_octaves, grid.noise_lacunarity,
      grid.noise_persistence, grid.noise_seed, grid.noise_offset, grid.noise_enable_border_fade, grid.noise_border_fade_distance);
  } else if (grid.type == MediumGridType::Texture3D) {
    MediumTextureSampleContext texture_context = {density};
    value = medium_texture_sample_shared_3d(texture_context, local_coord, grid.dimensions);
  }

  return medium_density_shared_apply_shape(value, grid.noise_power, grid.noise_sharpness);
}

ETX_SHARED_INLINE bool density_grid_has_data_direct(ETX_IN(MediumDensitySharedGrid, grid), bool texture_ready) {
  if (medium_density_shared_has_grid_data(grid.type, grid.dimensions, grid.density_count) == false) {
    return false;
  }

  if (grid.type == MediumGridType::NoiseFunction) {
    return true;
  }

  return texture_ready;
}

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
    result.density_data_offset = kInvalidIndex;
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
    result.density_data_chunk_index = kInvalidIndex;
    result.density_image_index = grid.density_image_index;
    return result;
  }

  ETX_SHARED_INLINE float sample_texture_3d(const float3& local_coord, const uint3& dimensions) const {
    MediumTextureSampleContext context = {density};
    return medium_texture_sample_shared_3d(context, local_coord, dimensions);
  }

  ETX_SHARED_INLINE float sample_noise(const float3& local_coord, const BoundingBox& bounds, const MediumGrid& grid) const {
    MediumDensitySharedGrid shared_grid = to_shared_grid(grid, static_cast<uint32_t>(density.count));
    return medium_density_shared_sample_noise(local_coord, bounds.p_min, bounds.p_max, shared_grid.noise_type, shared_grid.noise_scale, shared_grid.noise_octaves,
      shared_grid.noise_lacunarity, shared_grid.noise_persistence, shared_grid.noise_seed, shared_grid.noise_offset, shared_grid.noise_enable_border_fade,
      shared_grid.noise_border_fade_distance);
  }

  ETX_SHARED_INLINE float sample(const float3& local_coord, const BoundingBox& bounds, const MediumGrid& grid) const {
    MediumDensitySharedGrid shared_grid = to_shared_grid(grid, static_cast<uint32_t>(density.count));
    return density_grid_sample_direct(density, shared_grid, bounds.p_min, bounds.p_max, local_coord);
  }

  ETX_SHARED_INLINE bool has_data(const MediumGrid& grid) const {
    uint32_t density_count = (density.count > 0ull) ? 1u : 0u;
    MediumDensitySharedGrid shared_grid = to_shared_grid(grid, density_count);
    return medium_density_shared_has_grid_data(shared_grid.type, shared_grid.dimensions, shared_grid.density_count, shared_grid.density_image_index);
  }
};

}  // namespace etx
