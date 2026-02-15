#pragma once

#include "interop.hxx"

ETX_STATIC_CONST uint32_t kSamplerPathSourceUndefined = 0u;
ETX_STATIC_CONST uint32_t kSamplerPathSourceCamera = 1u;
ETX_STATIC_CONST uint32_t kSamplerPathSourceLight = 2u;

ETX_STATIC_CONST uint32_t kSamplerStreamBSDF = 0u;
ETX_STATIC_CONST uint32_t kSamplerStreamConnection = 1u;
ETX_STATIC_CONST uint32_t kSamplerStreamSupport = 2u;
ETX_STATIC_CONST uint32_t kSamplerStreamOther = 3u;

ETX_STATIC_CONST uint32_t kSamplerBlueNoiseTileSize = 128u;
ETX_STATIC_CONST uint32_t kSamplerBlueNoiseSampleCount = 256u;
ETX_STATIC_CONST uint32_t kSamplerBlueNoiseDimensionCount = 8u;

struct ETX_ALIGNED SamplerPolicy {
  uint32_t enable_blue_noise ETX_INIT(0u);
  uint32_t blue_noise_sample_limit ETX_INIT(kSamplerBlueNoiseSampleCount);
  uint32_t blue_noise_dimension_limit ETX_INIT(kSamplerBlueNoiseDimensionCount);
  uint32_t direct_camera_only ETX_INIT(1u);
};

ETX_SHARED_INLINE uint32_t sampler_stream_dimension_base(uint32_t stream) {
  if (stream == kSamplerStreamBSDF) {
    return 0u;
  }

  if (stream == kSamplerStreamConnection) {
    return 2u;
  }

  if (stream == kSamplerStreamSupport) {
    return 4u;
  }

  return 6u;
}

ETX_SHARED_INLINE bool sampler_use_blue_noise_for_interaction(ETX_IN(SamplerPolicy, policy), uint32_t path_source, uint32_t interaction_index, uint32_t current_sample) {
  if (policy.enable_blue_noise == 0u) {
    return false;
  }

  if ((policy.direct_camera_only != 0u) && (path_source != kSamplerPathSourceCamera)) {
    return false;
  }

  if (interaction_index != 1u) {
    return false;
  }

  if (current_sample >= policy.blue_noise_sample_limit) {
    return false;
  }

  return true;
}

ETX_SHARED_INLINE bool sampler_stream_supports_blue_noise(ETX_IN(SamplerPolicy, policy), uint32_t stream) {
  if ((stream != kSamplerStreamBSDF) && (stream != kSamplerStreamConnection) && (stream != kSamplerStreamSupport)) {
    return false;
  }

  uint32_t dimension_base = sampler_stream_dimension_base(stream);
  if ((dimension_base + 1u) >= policy.blue_noise_dimension_limit) {
    return false;
  }

  return true;
}

ETX_SHARED_INLINE bool sampler_use_blue_noise(ETX_IN(SamplerPolicy, policy), uint32_t path_source, uint32_t interaction_index, uint32_t current_sample, uint32_t stream) {
  if (sampler_use_blue_noise_for_interaction(policy, path_source, interaction_index, current_sample) == false) {
    return false;
  }

  return sampler_stream_supports_blue_noise(policy, stream);
}
