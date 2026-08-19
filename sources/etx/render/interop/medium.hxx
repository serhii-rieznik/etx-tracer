#pragma once

#include "interop.hxx"
#include "bounding_box.hxx"
#include "math_shared.hxx"
#include "spectrum.hxx"

struct MediumGridType {
  enum : uint32_t {
    Texture3D,
    NoiseFunction,
  };
};

struct MediumNoiseType {
  enum : uint32_t {
    Perlin,
    Worley,
    Billow,
    Voronoi,
    Lattice,
    Uniform,

    Count,
  };
};

struct ETX_ALIGNED MediumGrid {
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
  uint32_t density_image_index ETX_INIT(kInvalidIndex);
};

struct ETX_ALIGNED Medium {
  using Class = uint16_t;
  enum : uint16_t {
    Homogeneous,
    Heterogeneous,
  };

  MediumGrid grid ETX_INIT({});
  BoundingBox bounds ETX_INIT({});
  uint32_t absorption_index ETX_INIT(kInvalidIndex);
  uint32_t scattering_index ETX_INIT(kInvalidIndex);
  float phase_function_g ETX_INIT(0.0f);
  uint16_t enable_explicit_connections ETX_INIT(1u);
  Class cls ETX_INIT(Homogeneous);
  AffineTransform world_to_object ETX_INIT({});
  BoundingBox local_bounds ETX_INIT({});
};

ETX_SHARED_INLINE float3 medium_transform_point(ETX_IN(AffineTransform, transform), ETX_IN(float3, position)) {
  return float3(transform.rows[0].x * position.x + transform.rows[0].y * position.y + transform.rows[0].z * position.z + transform.rows[0].w,
    transform.rows[1].x * position.x + transform.rows[1].y * position.y + transform.rows[1].z * position.z + transform.rows[1].w,
    transform.rows[2].x * position.x + transform.rows[2].y * position.y + transform.rows[2].z * position.z + transform.rows[2].w);
}

ETX_SHARED_INLINE float3 medium_world_to_local(ETX_IN(AffineTransform, world_to_object), ETX_IN(BoundingBox, local_bounds), ETX_IN(float3, world_position)) {
  const float3 object_position = medium_transform_point(world_to_object, world_position);
  const float3 size = local_bounds.p_max - local_bounds.p_min;
  float3 result = float3(0.0f, 0.0f, 0.0f);
  result.x = (size.x > kEpsilon) ? ((object_position.x - local_bounds.p_min.x) / size.x) : 0.0f;
  result.y = (size.y > kEpsilon) ? ((object_position.y - local_bounds.p_min.y) / size.y) : 0.0f;
  result.z = (size.z > kEpsilon) ? ((object_position.z - local_bounds.p_min.z) / size.z) : 0.0f;
  return result;
}

ETX_SHARED_INLINE bool medium_local_coordinate_valid(ETX_IN(float3, local_coord)) {
  return (local_coord.x >= 0.0f) && (local_coord.y >= 0.0f) && (local_coord.z >= 0.0f) && (local_coord.x < 1.0f) && (local_coord.y < 1.0f) && (local_coord.z < 1.0f);
}

struct ETX_ALIGNED MediumInstance {
  SpectralResponse extinction;
  float anisotropy ETX_INIT(0.0f);
  uint32_t index ETX_INIT(kInvalidIndex);
};

struct ETX_ALIGNED MediumSample {
  SpectralResponse weight ETX_INIT({});
  float3 pos ETX_INIT({});
  float sampled_medium_t ETX_INIT(0.0f);
};

ETX_SHARED_INLINE bool medium_sample_valid(ETX_IN(MediumSample, sample)) {
  return (sample.sampled_medium_t > 0.0f) && (spectral_response_maximum(sample.weight) > 0.0f);
}

ETX_SHARED_INLINE bool medium_sample_sampled_medium(ETX_IN(MediumSample, sample)) {
  return sample.sampled_medium_t > 0.0f;
}

ETX_SHARED_INLINE bool medium_instance_valid(ETX_IN(MediumInstance, instance)) {
  return (instance.index != kInvalidIndex) || (spectral_response_maximum(instance.extinction) > 0.0f);
}

ETX_SHARED_INLINE SpectralResponse medium_transmittance(ETX_IN(MediumInstance, instance), float distance) {
  return spectral_response_exp(spectral_response_mul(instance.extinction, -distance));
}
