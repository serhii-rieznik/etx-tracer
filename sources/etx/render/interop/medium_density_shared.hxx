#pragma once

#include "interop.hxx"
#include "medium.hxx"

ETX_SHARED_INLINE float medium_density_shared_floor(float value) {
#if defined(__cplusplus)
  return floorf(value);
#else
  return floor(value);
#endif
}

ETX_SHARED_INLINE float medium_density_shared_sqrt(float value) {
#if defined(__cplusplus)
  return sqrtf(value);
#else
  return sqrt(value);
#endif
}

ETX_SHARED_INLINE float medium_density_shared_abs(float value) {
#if defined(__cplusplus)
  return fabsf(value);
#else
  return abs(value);
#endif
}

ETX_SHARED_INLINE float medium_density_shared_pow(float value, float power) {
#if defined(__cplusplus)
  return powf(value, power);
#else
  return pow(value, power);
#endif
}

ETX_SHARED_INLINE float medium_density_shared_lerp(float a, float b, float t) {
  return a + (b - a) * t;
}

ETX_SHARED_INLINE uint32_t medium_density_shared_to_u32(float value) {
#if defined(__cplusplus)
  return static_cast<uint32_t>(value);
#else
  return uint(value);
#endif
}

ETX_SHARED_INLINE int medium_density_shared_floor_to_int(float value) {
#if defined(__cplusplus)
  return static_cast<int>(medium_density_shared_floor(value));
#else
  return int(medium_density_shared_floor(value));
#endif
}

ETX_SHARED_INLINE uint32_t medium_density_shared_wrap_255(int value) {
#if defined(__cplusplus)
  return static_cast<uint32_t>(value) & 255u;
#else
  return uint(value) & 255u;
#endif
}

struct MediumDensitySharedTextureSample3D {
  uint32_t ix ETX_INIT(0u);
  uint32_t nx ETX_INIT(0u);
  uint32_t iy ETX_INIT(0u);
  uint32_t ny ETX_INIT(0u);
  uint32_t iz ETX_INIT(0u);
  uint32_t nz ETX_INIT(0u);
  float dx ETX_INIT(0.0f);
  float dy ETX_INIT(0.0f);
  float dz ETX_INIT(0.0f);
};

ETX_SHARED_INLINE MediumDensitySharedTextureSample3D medium_density_shared_zero_texture_sample_3d() {
  MediumDensitySharedTextureSample3D result;
  result.ix = 0u;
  result.nx = 0u;
  result.iy = 0u;
  result.ny = 0u;
  result.iz = 0u;
  result.nz = 0u;
  result.dx = 0.0f;
  result.dy = 0.0f;
  result.dz = 0.0f;
  return result;
}

struct MediumDensitySharedGrid {
  uint3 dimensions ETX_INIT({});
  uint32_t type ETX_INIT(MediumGridType::Texture3D);
  uint32_t noise_type ETX_INIT(MediumNoiseType::Perlin);
  uint32_t density_data_offset ETX_INIT(kInvalidIndex);
  uint32_t density_count ETX_INIT(0u);
  uint32_t noise_seed ETX_INIT(0u);
  float3 noise_offset ETX_INIT({});
  uint32_t noise_enable_border_fade ETX_INIT(0u);
  uint32_t noise_octaves ETX_INIT(1u);
  float noise_scale ETX_INIT(1.0f);
  float noise_lacunarity ETX_INIT(2.0f);
  float noise_persistence ETX_INIT(0.5f);
  float noise_power ETX_INIT(1.0f);
  float noise_sharpness ETX_INIT(1.0f);
  float noise_border_fade_distance ETX_INIT(0.1f);
  uint32_t density_data_chunk_index ETX_INIT(kInvalidIndex);
};

ETX_SHARED_INLINE bool medium_density_shared_prepare_texture_sample_3d(ETX_IN(float3, local_coord), ETX_IN(uint3, dimensions),
  ETX_OUT(MediumDensitySharedTextureSample3D, sample)) {
  sample = medium_density_shared_zero_texture_sample_3d();
  if ((local_coord.x < 0.0f) || (local_coord.y < 0.0f) || (local_coord.z < 0.0f) || (local_coord.x >= 1.0f) || (local_coord.y >= 1.0f) || (local_coord.z >= 1.0f)) {
    return false;
  }

  if ((dimensions.x == 0u) || (dimensions.y == 0u) || (dimensions.z == 0u)) {
    return false;
  }

  float px = clamp(local_coord.x * float(dimensions.x) - 0.5f, 0.0f, float(dimensions.x) - 1.0f);
  float py = clamp(local_coord.y * float(dimensions.y) - 0.5f, 0.0f, float(dimensions.y) - 1.0f);
  float pz = clamp(local_coord.z * float(dimensions.z) - 0.5f, 0.0f, float(dimensions.z) - 1.0f);

  sample.ix = min(dimensions.x - 1u, medium_density_shared_to_u32(px));
  sample.nx = min(dimensions.x - 1u, sample.ix + 1u);
  sample.iy = min(dimensions.y - 1u, medium_density_shared_to_u32(py));
  sample.ny = min(dimensions.y - 1u, sample.iy + 1u);
  sample.iz = min(dimensions.z - 1u, medium_density_shared_to_u32(pz));
  sample.nz = min(dimensions.z - 1u, sample.iz + 1u);

  sample.dx = px - medium_density_shared_floor(px);
  sample.dy = py - medium_density_shared_floor(py);
  sample.dz = pz - medium_density_shared_floor(pz);
  return true;
}

ETX_SHARED_INLINE float medium_density_shared_trilerp(float d000, float d001, float d010, float d011, float d100, float d101, float d110, float d111, float dx, float dy,
  float dz) {
  float d_bottom = medium_density_shared_lerp(medium_density_shared_lerp(d000, d001, dx), medium_density_shared_lerp(d010, d011, dx), dy);
  float d_top = medium_density_shared_lerp(medium_density_shared_lerp(d100, d101, dx), medium_density_shared_lerp(d110, d111, dx), dy);
  return medium_density_shared_lerp(d_bottom, d_top, dz);
}

ETX_SHARED_INLINE bool medium_density_shared_has_texture_data(ETX_IN(uint3, dimensions), uint32_t density_count) {
  return (dimensions.x > 0u) && (dimensions.y > 0u) && (dimensions.z > 0u) && (density_count > 0u);
}

ETX_SHARED_INLINE bool medium_density_shared_has_grid_data(uint32_t grid_type, ETX_IN(uint3, dimensions), uint32_t density_count) {
  if (grid_type == MediumGridType::NoiseFunction) {
    return true;
  }

  if (grid_type != MediumGridType::Texture3D) {
    return false;
  }

  return medium_density_shared_has_texture_data(dimensions, density_count);
}

ETX_SHARED_INLINE float3 medium_density_shared_bounds_from_local(ETX_IN(float3, p), ETX_IN(float3, bounds_min), ETX_IN(float3, bounds_max)) {
  float3 size = bounds_max - bounds_min;
  return float3(p.x * size.x + bounds_min.x, p.y * size.y + bounds_min.y, p.z * size.z + bounds_min.z);
}

ETX_SHARED_INLINE float medium_density_shared_fade(float t) {
  return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
}

ETX_SHARED_INLINE uint32_t medium_density_shared_hash3d(uint32_t x, uint32_t y, uint32_t z, uint32_t seed) {
  uint32_t h = seed;
  h ^= x;
  h ^= y << 8u;
  h ^= z << 16u;
  h *= 0x9e3779b9u;
  h ^= h >> 16u;
  h *= 0x85ebca6bu;
  h ^= h >> 13u;
  h *= 0xc2b2ae35u;
  h ^= h >> 16u;
  return h;
}

ETX_SHARED_INLINE float medium_density_shared_gradient3d(uint32_t hash, float x, float y, float z) {
  uint32_t h = hash & 15u;
  float u = (h < 8u) ? x : y;
  float v = (h < 4u) ? y : (((h == 12u) || (h == 14u)) ? x : z);
  return (((h & 1u) == 0u) ? u : -u) + (((h & 2u) == 0u) ? v : -v);
}

ETX_SHARED_INLINE float medium_density_shared_perlin_noise_3d(ETX_IN(float3, pos), uint32_t seed) {
  int i = medium_density_shared_floor_to_int(pos.x);
  int j = medium_density_shared_floor_to_int(pos.y);
  int k = medium_density_shared_floor_to_int(pos.z);

  float x = pos.x - float(i);
  float y = pos.y - float(j);
  float z = pos.z - float(k);

  uint32_t ii = medium_density_shared_wrap_255(i);
  uint32_t jj = medium_density_shared_wrap_255(j);
  uint32_t kk = medium_density_shared_wrap_255(k);

  float u = medium_density_shared_fade(x);
  float v = medium_density_shared_fade(y);
  float w = medium_density_shared_fade(z);

  uint32_t h000 = medium_density_shared_hash3d(ii, jj, kk, seed);
  uint32_t h100 = medium_density_shared_hash3d(ii + 1u, jj, kk, seed);
  uint32_t h010 = medium_density_shared_hash3d(ii, jj + 1u, kk, seed);
  uint32_t h110 = medium_density_shared_hash3d(ii + 1u, jj + 1u, kk, seed);
  uint32_t h001 = medium_density_shared_hash3d(ii, jj, kk + 1u, seed);
  uint32_t h101 = medium_density_shared_hash3d(ii + 1u, jj, kk + 1u, seed);
  uint32_t h011 = medium_density_shared_hash3d(ii, jj + 1u, kk + 1u, seed);
  uint32_t h111 = medium_density_shared_hash3d(ii + 1u, jj + 1u, kk + 1u, seed);

  float g000 = medium_density_shared_gradient3d(h000, x, y, z);
  float g100 = medium_density_shared_gradient3d(h100, x - 1.0f, y, z);
  float g010 = medium_density_shared_gradient3d(h010, x, y - 1.0f, z);
  float g110 = medium_density_shared_gradient3d(h110, x - 1.0f, y - 1.0f, z);
  float g001 = medium_density_shared_gradient3d(h001, x, y, z - 1.0f);
  float g101 = medium_density_shared_gradient3d(h101, x - 1.0f, y, z - 1.0f);
  float g011 = medium_density_shared_gradient3d(h011, x, y - 1.0f, z - 1.0f);
  float g111 = medium_density_shared_gradient3d(h111, x - 1.0f, y - 1.0f, z - 1.0f);

  float c000 = medium_density_shared_lerp(medium_density_shared_lerp(g000, g100, u), medium_density_shared_lerp(g010, g110, u), v);
  float c001 = medium_density_shared_lerp(medium_density_shared_lerp(g001, g101, u), medium_density_shared_lerp(g011, g111, u), v);
  return medium_density_shared_lerp(c000, c001, w);
}

ETX_SHARED_INLINE float3 medium_density_shared_random_point_in_cell(uint32_t cell_x, uint32_t cell_y, uint32_t cell_z, uint32_t point_index, uint32_t seed) {
  uint32_t h = medium_density_shared_hash3d(cell_x, cell_y, cell_z, seed + point_index);
  float x = float(h & 0xFFFFu) * (1.0f / 65536.0f);
  h *= 0x9e3779b9u;
  float y = float(h & 0xFFFFu) * (1.0f / 65536.0f);
  h *= 0x9e3779b9u;
  float z = float(h & 0xFFFFu) * (1.0f / 65536.0f);
  return float3(x, y, z);
}

ETX_SHARED_INLINE float medium_density_shared_worley_noise_3d(ETX_IN(float3, pos), uint32_t seed) {
  int cell_x = medium_density_shared_floor_to_int(pos.x);
  int cell_y = medium_density_shared_floor_to_int(pos.y);
  int cell_z = medium_density_shared_floor_to_int(pos.z);

  float local_x = pos.x - float(cell_x);
  float local_y = pos.y - float(cell_y);
  float local_z = pos.z - float(cell_z);

  float min_dist1_sq = kMaxFloat;
  float min_dist2_sq = kMaxFloat;

  for (int dz = -1; dz <= 1; ++dz) {
    for (int dy = -1; dy <= 1; ++dy) {
      for (int dx = -1; dx <= 1; ++dx) {
        uint32_t cx = medium_density_shared_wrap_255(cell_x + dx);
        uint32_t cy = medium_density_shared_wrap_255(cell_y + dy);
        uint32_t cz = medium_density_shared_wrap_255(cell_z + dz);

        float3 feature_point = medium_density_shared_random_point_in_cell(cx, cy, cz, 0u, seed);
        float3 point_pos = float3(float(dx), float(dy), float(dz)) + feature_point;
        float3 delta = point_pos - float3(local_x, local_y, local_z);
        float dist_sq = dot(delta, delta);

        if (dist_sq < min_dist1_sq) {
          min_dist2_sq = min_dist1_sq;
          min_dist1_sq = dist_sq;
        } else if (dist_sq < min_dist2_sq) {
          min_dist2_sq = dist_sq;
        }
      }
    }
  }

  if (min_dist1_sq >= kMaxFloat) {
    return 0.0f;
  }
  if (min_dist2_sq >= kMaxFloat) {
    return medium_density_shared_sqrt(min_dist1_sq);
  }
  return medium_density_shared_sqrt(min_dist2_sq) - medium_density_shared_sqrt(min_dist1_sq);
}

ETX_SHARED_INLINE float medium_density_shared_voronoi_noise_3d(ETX_IN(float3, pos), uint32_t seed) {
  int cell_x = medium_density_shared_floor_to_int(pos.x);
  int cell_y = medium_density_shared_floor_to_int(pos.y);
  int cell_z = medium_density_shared_floor_to_int(pos.z);

  float local_x = pos.x - float(cell_x);
  float local_y = pos.y - float(cell_y);
  float local_z = pos.z - float(cell_z);

  float min_dist_sq = kMaxFloat;
  for (int dz = -1; dz <= 1; ++dz) {
    for (int dy = -1; dy <= 1; ++dy) {
      for (int dx = -1; dx <= 1; ++dx) {
        uint32_t cx = medium_density_shared_wrap_255(cell_x + dx);
        uint32_t cy = medium_density_shared_wrap_255(cell_y + dy);
        uint32_t cz = medium_density_shared_wrap_255(cell_z + dz);

        float3 feature_point = medium_density_shared_random_point_in_cell(cx, cy, cz, 0u, seed);
        float3 point_pos = float3(float(dx), float(dy), float(dz)) + feature_point;
        float3 delta = point_pos - float3(local_x, local_y, local_z);
        float dist_sq = dot(delta, delta);
        min_dist_sq = min(min_dist_sq, dist_sq);
      }
    }
  }

  return medium_density_shared_sqrt(min_dist_sq);
}

ETX_SHARED_INLINE float medium_density_shared_lattice_noise_3d(ETX_IN(float3, pos), uint32_t seed) {
  int cell_x = medium_density_shared_floor_to_int(pos.x);
  int cell_y = medium_density_shared_floor_to_int(pos.y);
  int cell_z = medium_density_shared_floor_to_int(pos.z);

  float min_dist_sq = kMaxFloat;
  for (int dz = -1; dz <= 1; ++dz) {
    for (int dy = -1; dy <= 1; ++dy) {
      for (int dx = -1; dx <= 1; ++dx) {
        float3 grid_point = float3(float(cell_x + dx), float(cell_y + dy), float(cell_z + dz));
        float3 delta = grid_point - pos;
        float dist_sq = dot(delta, delta);
        min_dist_sq = min(min_dist_sq, dist_sq);
      }
    }
  }

  return medium_density_shared_sqrt(min_dist_sq);
}

ETX_SHARED_INLINE float medium_density_shared_fbm_noise_3d(ETX_IN(float3, pos), uint32_t noise_type, uint32_t seed, float scale, uint32_t octaves, float lacunarity,
  float persistence) {
  float value = 0.0f;
  float amplitude = 1.0f;
  float frequency = scale;
  float max_amplitude_sum = 0.0f;
  const float amplitude_threshold = 0.001f;

#if defined(__cplusplus)
  for (uint32_t i = 0u; i < octaves; ++i) {
#else
  [loop] for (uint32_t i = 0u; i < octaves; ++i) {
#endif
    if (amplitude < amplitude_threshold) {
      break;
    }

    float3 sample_pos = pos * frequency;
    float n = 0.0f;
    if (noise_type == MediumNoiseType::Perlin) {
      n = medium_density_shared_perlin_noise_3d(sample_pos, seed + i);
    } else if (noise_type == MediumNoiseType::Worley) {
      n = medium_density_shared_worley_noise_3d(sample_pos, seed + i);
    } else if (noise_type == MediumNoiseType::Billow) {
      n = medium_density_shared_abs(medium_density_shared_perlin_noise_3d(sample_pos, seed + i));
    } else if (noise_type == MediumNoiseType::Voronoi) {
      n = medium_density_shared_voronoi_noise_3d(sample_pos, seed + i);
    } else if (noise_type == MediumNoiseType::Lattice) {
      n = medium_density_shared_lattice_noise_3d(sample_pos, seed + i);
    } else if (noise_type == MediumNoiseType::Uniform) {
      n = 1.0f;
    }

    value += n * amplitude;
    max_amplitude_sum += amplitude;
    amplitude *= persistence;
    frequency *= lacunarity;
  }

  float normalized = (max_amplitude_sum > 0.0f) ? (value / max_amplitude_sum) : 0.0f;
  return saturate(normalized);
}

ETX_SHARED_INLINE float3 medium_density_shared_normalized_world_pos(ETX_IN(float3, local_coord), ETX_IN(float3, bounds_min), ETX_IN(float3, bounds_max),
  ETX_IN(float3, noise_offset)) {
  float3 world_pos = medium_density_shared_bounds_from_local(local_coord, bounds_min, bounds_max) + noise_offset;
  float3 bbox_size = bounds_max - bounds_min;
  float max_dimension = max(bbox_size.x, max(bbox_size.y, bbox_size.z));
  if (max_dimension > kEpsilon) {
    return world_pos / max_dimension;
  }
  return world_pos;
}

ETX_SHARED_INLINE float medium_density_shared_apply_border_fade(float value, ETX_IN(float3, local_coord), uint32_t noise_enable_border_fade, float noise_border_fade_distance) {
  if (noise_enable_border_fade == 0u) {
    return value;
  }

  float dist_to_min_x = local_coord.x;
  float dist_to_max_x = 1.0f - local_coord.x;
  float dist_to_min_y = local_coord.y;
  float dist_to_max_y = 1.0f - local_coord.y;
  float dist_to_min_z = local_coord.z;
  float dist_to_max_z = 1.0f - local_coord.z;
  float min_dist = min(dist_to_min_x, min(dist_to_max_x, min(dist_to_min_y, min(dist_to_max_y, min(dist_to_min_z, dist_to_max_z)))));
  float fade_distance = max(noise_border_fade_distance, kEpsilon);
  float fade_factor = clamp(min_dist / fade_distance, 0.0f, 1.0f);
  return value * fade_factor;
}

ETX_SHARED_INLINE float medium_density_shared_sample_noise(ETX_IN(float3, local_coord), ETX_IN(float3, bounds_min), ETX_IN(float3, bounds_max), uint32_t noise_type,
  float noise_scale, uint32_t noise_octaves, float noise_lacunarity, float noise_persistence, uint32_t noise_seed, ETX_IN(float3, noise_offset), uint32_t noise_enable_border_fade,
  float noise_border_fade_distance) {
  float3 normalized_world_pos = medium_density_shared_normalized_world_pos(local_coord, bounds_min, bounds_max, noise_offset);
  float value = medium_density_shared_fbm_noise_3d(normalized_world_pos, noise_type, noise_seed, noise_scale, noise_octaves, noise_lacunarity, noise_persistence);
  value = medium_density_shared_apply_border_fade(value, local_coord, noise_enable_border_fade, noise_border_fade_distance);
  return saturate(value);
}

ETX_SHARED_INLINE float medium_density_shared_apply_shape(float value, float noise_power, float noise_sharpness) {
  float safe_power = max(noise_power, 0.0f);
  float shaped_value = medium_density_shared_pow(max(value, 0.0f), safe_power);
  return saturate((shaped_value - 0.5f) * noise_sharpness + 0.5f);
}
