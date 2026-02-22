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

struct DensityTextureSampleSharedContext {
  ArrayView<float> density = {};
};

ETX_SHARED_INLINE float density_texture_sample_shared_density(
  ETX_INOUT(DensityTextureSampleSharedContext, context), ETX_IN(uint3, dimensions), uint32_t x, uint32_t y, uint32_t z) {
  uint32_t index = x + y * dimensions.x + z * dimensions.x * dimensions.y;
  return context.density[index];
}

#define ETX_MEDIUM_TEXTURE_SAMPLE_SHARED_CONTEXT_TYPE DensityTextureSampleSharedContext
#define ETX_MEDIUM_TEXTURE_SAMPLE_SHARED_DENSITY(context, dimensions, x, y, z) density_texture_sample_shared_density(context, dimensions, x, y, z)
#include <etx/render/interop/medium_texture_sample_shared.hxx>
#undef ETX_MEDIUM_TEXTURE_SAMPLE_SHARED_DENSITY
#undef ETX_MEDIUM_TEXTURE_SAMPLE_SHARED_CONTEXT_TYPE

struct DensityGridPolicySharedContext {
  ArrayView<float> density = {};
  MediumDensitySharedGrid grid = {};
  float3 bounds_min = {};
  float3 bounds_max = {};
};

ETX_SHARED_INLINE MediumDensitySharedGrid density_grid_policy_shared_grid(ETX_INOUT(DensityGridPolicySharedContext, context)) {
  return context.grid;
}

ETX_SHARED_INLINE float density_grid_policy_shared_sample_noise(ETX_INOUT(DensityGridPolicySharedContext, context), ETX_IN(float3, local_coord)) {
  return medium_density_shared_sample_noise(local_coord, context.bounds_min, context.bounds_max, context.grid.noise_type, context.grid.noise_scale,
    context.grid.noise_octaves, context.grid.noise_lacunarity, context.grid.noise_persistence, context.grid.noise_seed, context.grid.noise_offset,
    context.grid.noise_enable_border_fade, context.grid.noise_border_fade_distance);
}

ETX_SHARED_INLINE float density_grid_policy_shared_sample_texture(ETX_INOUT(DensityGridPolicySharedContext, context), ETX_IN(float3, local_coord)) {
  DensityTextureSampleSharedContext texture_context = {context.density};
  return medium_texture_sample_shared_3d(texture_context, local_coord, context.grid.dimensions);
}

ETX_SHARED_INLINE bool density_grid_policy_shared_texture_ready(ETX_INOUT(DensityGridPolicySharedContext, context)) {
  (void)context;
  return true;
}

#define ETX_MEDIUM_GRID_POLICY_SHARED_CONTEXT_TYPE DensityGridPolicySharedContext
#define ETX_MEDIUM_GRID_POLICY_SHARED_GRID(context) density_grid_policy_shared_grid(context)
#define ETX_MEDIUM_GRID_POLICY_SHARED_SAMPLE_NOISE(context, local_coord) density_grid_policy_shared_sample_noise(context, local_coord)
#define ETX_MEDIUM_GRID_POLICY_SHARED_SAMPLE_TEXTURE(context, local_coord) density_grid_policy_shared_sample_texture(context, local_coord)
#define ETX_MEDIUM_GRID_POLICY_SHARED_TEXTURE_READY(context) density_grid_policy_shared_texture_ready(context)
#include <etx/render/interop/medium_grid_policy_shared.hxx>
#undef ETX_MEDIUM_GRID_POLICY_SHARED_TEXTURE_READY
#undef ETX_MEDIUM_GRID_POLICY_SHARED_SAMPLE_TEXTURE
#undef ETX_MEDIUM_GRID_POLICY_SHARED_SAMPLE_NOISE
#undef ETX_MEDIUM_GRID_POLICY_SHARED_GRID
#undef ETX_MEDIUM_GRID_POLICY_SHARED_CONTEXT_TYPE

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
    DensityTextureSampleSharedContext context = {density};
    return medium_texture_sample_shared_3d(context, local_coord, dimensions);
  }

  ETX_SHARED_INLINE float sample_noise(const float3& local_coord, const BoundingBox& bounds, const MediumGrid& grid) const {
    MediumDensitySharedGrid shared_grid = to_shared_grid(grid, grid.density_count);
    return medium_density_shared_sample_noise(local_coord, bounds.p_min, bounds.p_max, shared_grid.noise_type, shared_grid.noise_scale, shared_grid.noise_octaves,
      shared_grid.noise_lacunarity, shared_grid.noise_persistence, shared_grid.noise_seed, shared_grid.noise_offset, shared_grid.noise_enable_border_fade,
      shared_grid.noise_border_fade_distance);
  }

  ETX_SHARED_INLINE float sample(const float3& local_coord, const BoundingBox& bounds, const MediumGrid& grid) const {
    MediumDensitySharedGrid shared_grid = to_shared_grid(grid, grid.density_count);
    DensityGridPolicySharedContext context = {density, shared_grid, bounds.p_min, bounds.p_max};
    return medium_grid_policy_shared_sample_density(context, local_coord);
  }

  ETX_SHARED_INLINE bool has_data(const MediumGrid& grid) const {
    uint32_t density_count = (density.count > 0ull) ? 1u : 0u;
    MediumDensitySharedGrid shared_grid = to_shared_grid(grid, density_count);
    DensityGridPolicySharedContext context = {density, shared_grid};
    return medium_grid_policy_shared_has_grid_data(context);
  }
};

}  // namespace etx
