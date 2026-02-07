#pragma once

#include <etx/render/interop/interop.hxx>

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

namespace {

ETX_GPU_CODE float fade(float t) {
  return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
}

ETX_GPU_CODE uint32_t hash3d(uint32_t x, uint32_t y, uint32_t z, uint32_t seed) {
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

ETX_GPU_CODE float gradient3d(uint32_t hash, float x, float y, float z) {
  uint32_t h = hash & 15u;
  float u = h < 8u ? x : y;
  float v = h < 4u ? y : (h == 12u || h == 14u ? x : z);
  return ((h & 1u) == 0u ? u : -u) + ((h & 2u) == 0u ? v : -v);
}

ETX_GPU_CODE float perlin_noise_3d(const float3& pos, uint32_t seed) {
  int32_t i = static_cast<int32_t>(floorf(pos.x));
  int32_t j = static_cast<int32_t>(floorf(pos.y));
  int32_t k = static_cast<int32_t>(floorf(pos.z));

  float x = pos.x - float(i);
  float y = pos.y - float(j);
  float z = pos.z - float(k);

  i &= 255u;
  j &= 255u;
  k &= 255u;

  float u = fade(x);
  float v = fade(y);
  float w = fade(z);

  uint32_t h000 = hash3d(i, j, k, seed);
  uint32_t h100 = hash3d(i + 1u, j, k, seed);
  uint32_t h010 = hash3d(i, j + 1u, k, seed);
  uint32_t h110 = hash3d(i + 1u, j + 1u, k, seed);
  uint32_t h001 = hash3d(i, j, k + 1u, seed);
  uint32_t h101 = hash3d(i + 1u, j, k + 1u, seed);
  uint32_t h011 = hash3d(i, j + 1u, k + 1u, seed);
  uint32_t h111 = hash3d(i + 1u, j + 1u, k + 1u, seed);

  float g000 = gradient3d(h000, x, y, z);
  float g100 = gradient3d(h100, x - 1.0f, y, z);
  float g010 = gradient3d(h010, x, y - 1.0f, z);
  float g110 = gradient3d(h110, x - 1.0f, y - 1.0f, z);
  float g001 = gradient3d(h001, x, y, z - 1.0f);
  float g101 = gradient3d(h101, x - 1.0f, y, z - 1.0f);
  float g011 = gradient3d(h011, x, y - 1.0f, z - 1.0f);
  float g111 = gradient3d(h111, x - 1.0f, y - 1.0f, z - 1.0f);

  float c000 = lerp(lerp(g000, g100, u), lerp(g010, g110, u), v);
  float c001 = lerp(lerp(g001, g101, u), lerp(g011, g111, u), v);
  return lerp(c000, c001, w);
}

ETX_GPU_CODE float3 random_point_in_cell(uint32_t cell_x, uint32_t cell_y, uint32_t cell_z, uint32_t point_index, uint32_t seed) {
  uint32_t h = hash3d(cell_x, cell_y, cell_z, seed + point_index);
  float x = (h & 0xFFFFu) / 65536.0f;
  h = h * 0x9e3779b9u;
  float y = (h & 0xFFFFu) / 65536.0f;
  h = h * 0x9e3779b9u;
  float z = (h & 0xFFFFu) / 65536.0f;
  return float3{x, y, z};
}

ETX_GPU_CODE float worley_noise_3d(const float3& pos, uint32_t seed) {
  int32_t cell_x = static_cast<int32_t>(floorf(pos.x));
  int32_t cell_y = static_cast<int32_t>(floorf(pos.y));
  int32_t cell_z = static_cast<int32_t>(floorf(pos.z));

  float local_x = pos.x - float(cell_x);
  float local_y = pos.y - float(cell_y);
  float local_z = pos.z - float(cell_z);

  float min_dist1_sq = kMaxFloat;
  float min_dist2_sq = kMaxFloat;

  for (int32_t dz = -1; dz <= 1; ++dz) {
    for (int32_t dy = -1; dy <= 1; ++dy) {
      for (int32_t dx = -1; dx <= 1; ++dx) {
        int32_t check_x = cell_x + dx;
        int32_t check_y = cell_y + dy;
        int32_t check_z = cell_z + dz;

        uint32_t cx = static_cast<uint32_t>(check_x) & 255u;
        uint32_t cy = static_cast<uint32_t>(check_y) & 255u;
        uint32_t cz = static_cast<uint32_t>(check_z) & 255u;

        float3 feature_point = random_point_in_cell(cx, cy, cz, 0u, seed);
        float3 cell_offset = float3{float(dx), float(dy), float(dz)};
        float3 point_pos = cell_offset + feature_point;
        float3 delta = point_pos - float3{local_x, local_y, local_z};
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
    return sqrtf(min_dist1_sq);
  }
  return sqrtf(min_dist2_sq) - sqrtf(min_dist1_sq);
}

ETX_GPU_CODE float voronoi_noise_3d(const float3& pos, uint32_t seed) {
  int32_t cell_x = static_cast<int32_t>(floorf(pos.x));
  int32_t cell_y = static_cast<int32_t>(floorf(pos.y));
  int32_t cell_z = static_cast<int32_t>(floorf(pos.z));

  float local_x = pos.x - float(cell_x);
  float local_y = pos.y - float(cell_y);
  float local_z = pos.z - float(cell_z);

  float min_dist_sq = kMaxFloat;

  for (int32_t dz = -1; dz <= 1; ++dz) {
    for (int32_t dy = -1; dy <= 1; ++dy) {
      for (int32_t dx = -1; dx <= 1; ++dx) {
        int32_t check_x = cell_x + dx;
        int32_t check_y = cell_y + dy;
        int32_t check_z = cell_z + dz;

        uint32_t cx = static_cast<uint32_t>(check_x) & 255u;
        uint32_t cy = static_cast<uint32_t>(check_y) & 255u;
        uint32_t cz = static_cast<uint32_t>(check_z) & 255u;

        float3 feature_point = random_point_in_cell(cx, cy, cz, 0u, seed);
        float3 cell_offset = float3{float(dx), float(dy), float(dz)};
        float3 point_pos = cell_offset + feature_point;
        float3 delta = point_pos - float3{local_x, local_y, local_z};
        float dist_sq = dot(delta, delta);

        if (dist_sq < min_dist_sq) {
          min_dist_sq = dist_sq;
        }
      }
    }
  }

  return sqrtf(min_dist_sq);
}

ETX_GPU_CODE float lattice_noise_3d(const float3& pos, uint32_t seed) {
  int32_t cell_x = static_cast<int32_t>(floorf(pos.x));
  int32_t cell_y = static_cast<int32_t>(floorf(pos.y));
  int32_t cell_z = static_cast<int32_t>(floorf(pos.z));

  float local_x = pos.x - float(cell_x);
  float local_y = pos.y - float(cell_y);
  float local_z = pos.z - float(cell_z);

  float min_dist_sq = kMaxFloat;

  for (int32_t dz = -1; dz <= 1; ++dz) {
    for (int32_t dy = -1; dy <= 1; ++dy) {
      for (int32_t dx = -1; dx <= 1; ++dx) {
        float3 grid_point = float3{float(cell_x + dx), float(cell_y + dy), float(cell_z + dz)};
        float3 delta = grid_point - pos;
        float dist_sq = dot(delta, delta);

        if (dist_sq < min_dist_sq) {
          min_dist_sq = dist_sq;
        }
      }
    }
  }

  return sqrtf(min_dist_sq);
}

ETX_GPU_CODE float fbm_noise_3d(const float3& pos, NoiseFunction noise_type, uint32_t seed, float scale, uint32_t octaves, float lacunarity, float persistence, float power) {
  float value = 0.0f;
  float amplitude = 1.0f;
  float frequency = scale;
  float max_amplitude_sum = 0.0f;
  const float amplitude_threshold = 0.001f;

  for (uint32_t i = 0; i < octaves; ++i) {
    if (amplitude < amplitude_threshold) {
      break;
    }

    float3 sample_pos = pos * frequency;
    float n = 0.0f;
    switch (noise_type) {
      case NoiseFunction::Perlin:
        n = perlin_noise_3d(sample_pos, seed + i);
        break;
      case NoiseFunction::Worley:
        n = worley_noise_3d(sample_pos, seed + i);
        break;
      case NoiseFunction::Billow:
        n = fabsf(perlin_noise_3d(sample_pos, seed + i));
        break;
      case NoiseFunction::Voronoi:
        n = voronoi_noise_3d(sample_pos, seed + i);
        break;
      case NoiseFunction::Lattice:
        n = lattice_noise_3d(sample_pos, seed + i);
        break;
      case NoiseFunction::Uniform:
        n = 1.0f;
        break;
      case NoiseFunction::Count:
        break;
    }
    value += n * amplitude;
    max_amplitude_sum += amplitude;
    amplitude *= persistence;
    frequency *= lacunarity;
  }

  float normalized = (max_amplitude_sum > 0.0f) ? (value / max_amplitude_sum) : 0.0f;
  return saturate(normalized);
}

}  // namespace

struct ETX_ALIGNED NoiseParameters {
  float scale = 1.0f;
  uint32_t octaves = 1u;
  float lacunarity = 2.0f;
  float persistence = 0.5f;
  uint32_t seed = 0u;
  float power = 1.0f;
  float sharpness = 1.0f;
  float3 offset = {};
  uint32_t enable_border_fade = 0u;
  float border_fade_distance = 0.1f;
};

struct ETX_ALIGNED DensityGrid {
  enum class Type : uint16_t {
    Texture3D,
    NoiseFunction,
  };

  Type type = Type::Texture3D;
  NoiseFunction noise_type = NoiseFunction::Perlin;
  ArrayView<float> density = {};
  uint3 dimensions = {};
  NoiseParameters noise = {};

  ETX_GPU_CODE float sample_texture_3d(const float3& local_coord) const {
    if ((local_coord.x < 0.0f) || (local_coord.y < 0.0f) || (local_coord.z < 0.0f) || (local_coord.x >= 1.0f) || (local_coord.y >= 1.0f) || (local_coord.z >= 1.0f)) {
      return 0.0f;
    }

    float px = clamp(local_coord.x * float(dimensions.x) - 0.5f, 0.0f, float(dimensions.x) - 1.0f);
    float py = clamp(local_coord.y * float(dimensions.y) - 0.5f, 0.0f, float(dimensions.y) - 1.0f);
    float pz = clamp(local_coord.z * float(dimensions.z) - 0.5f, 0.0f, float(dimensions.z) - 1.0f);

    uint32_t ix = min(dimensions.x - 1u, static_cast<uint32_t>(px));
    uint32_t nx = min(dimensions.x - 1u, ix + 1u);

    uint32_t iy = min(dimensions.y - 1u, static_cast<uint32_t>(py));
    uint32_t ny = min(dimensions.y - 1u, iy + 1u);

    uint32_t iz = min(dimensions.z - 1u, static_cast<uint32_t>(pz));
    uint32_t nz = min(dimensions.z - 1u, iz + 1u);

    float d000 = density[ix + iy * dimensions.x + iz * dimensions.x * dimensions.y];
    float d001 = density[nx + iy * dimensions.x + iz * dimensions.x * dimensions.y];
    float d010 = density[ix + ny * dimensions.x + iz * dimensions.x * dimensions.y];
    float d011 = density[nx + ny * dimensions.x + iz * dimensions.x * dimensions.y];
    float d100 = density[ix + iy * dimensions.x + nz * dimensions.x * dimensions.y];
    float d101 = density[nx + iy * dimensions.x + nz * dimensions.x * dimensions.y];
    float d110 = density[ix + ny * dimensions.x + nz * dimensions.x * dimensions.y];
    float d111 = density[nx + ny * dimensions.x + nz * dimensions.x * dimensions.y];

    float dx = px - floorf(px);
    float dy = py - floorf(py);
    float dz = pz - floorf(pz);

    float d_bottom = lerp(lerp(d000, d001, dx), lerp(d010, d011, dx), dy);
    float d_top = lerp(lerp(d100, d101, dx), lerp(d110, d111, dx), dy);
    return lerp(d_bottom, d_top, dz);
  }

  ETX_GPU_CODE float sample_noise(const float3& local_coord, const BoundingBox& bounds) const {
    float3 world_pos = bounds.from_local(local_coord) + noise.offset;
    float3 bbox_size = bounds.p_max - bounds.p_min;
    float max_dimension = max(bbox_size.x, max(bbox_size.y, bbox_size.z));
    float3 normalized_world_pos = {
      (max_dimension > kEpsilon) ? (world_pos.x / max_dimension) : world_pos.x,
      (max_dimension > kEpsilon) ? (world_pos.y / max_dimension) : world_pos.y,
      (max_dimension > kEpsilon) ? (world_pos.z / max_dimension) : world_pos.z,
    };
    float n = fbm_noise_3d(normalized_world_pos, noise_type, noise.seed, noise.scale, noise.octaves, noise.lacunarity, noise.persistence, 1.0f);

    if (noise.enable_border_fade != 0u) {
      float dist_to_min_x = local_coord.x;
      float dist_to_max_x = 1.0f - local_coord.x;
      float dist_to_min_y = local_coord.y;
      float dist_to_max_y = 1.0f - local_coord.y;
      float dist_to_min_z = local_coord.z;
      float dist_to_max_z = 1.0f - local_coord.z;
      float min_dist = min(dist_to_min_x, min(dist_to_max_x, min(dist_to_min_y, min(dist_to_max_y, min(dist_to_min_z, dist_to_max_z)))));
      float fade_factor = clamp(min_dist / noise.border_fade_distance, 0.0f, 1.0f);
      n *= fade_factor;
    }

    return saturate(n);
  }

  ETX_GPU_CODE float sample(const float3& local_coord, const BoundingBox& bounds) const {
    float value = 0.0f;
    if (type == Type::NoiseFunction) {
      value = sample_noise(local_coord, bounds);
    } else if (type == Type::Texture3D) {
      value = sample_texture_3d(local_coord);
    }
    value = powf(value, noise.power);
    value = saturate((value - 0.5f) * noise.sharpness + 0.5f);
    return value;
  }

  ETX_GPU_CODE bool has_data() const {
    return (type == Type::NoiseFunction) || (density.count > 0);
  }
};

}  // namespace etx
