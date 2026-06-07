#pragma once

#include "interop.hxx"
#include "bounding_box.hxx"
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
};

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
