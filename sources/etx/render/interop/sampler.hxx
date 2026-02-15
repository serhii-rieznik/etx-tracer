#pragma once

#include "interop.hxx"

#if defined(__cplusplus)
#include <cstring>
#endif

ETX_SHARED_INLINE uint32_t sampler_random_seed(uint32_t val0, uint32_t val1) {
  uint32_t v0 = val0;
  uint32_t v1 = val1;
  uint32_t s0 = 0u;
  for (uint32_t n = 0u; n < 16u; ++n) {
    s0 += 0x9e3779b9u;
    v0 += ((v1 << 4u) + 0xa341316cu) ^ (v1 + s0) ^ ((v1 >> 5u) + 0xc8013ea4u);
    v1 += ((v0 << 4u) + 0xad90777du) ^ (v0 + s0) ^ ((v0 >> 5u) + 0x7e95761eu);
  }
  return v0;
}

ETX_SHARED_INLINE float sampler_as_float(uint32_t bits) {
#if defined(__cplusplus)
  float result = 0.0f;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
#else
  return asfloat(bits);
#endif
}

ETX_SHARED_INLINE float sampler_next_random(ETX_INOUT(uint32_t, seed)) {
  seed = (seed ^ 61u) ^ (seed >> 16u);
  seed *= 9u;
  seed = seed ^ (seed >> 4u);
  seed *= 0x27d4eb2du;
  seed = seed ^ (seed >> 15u);
  uint32_t wrapped_bits = (seed >> 9u) | 0x3f800000u;
  return sampler_as_float(wrapped_bits) - 1.0f;
}

ETX_SHARED_INLINE float2 sampler_next_2d(ETX_INOUT(uint32_t, seed)) {
  return float2(sampler_next_random(seed), sampler_next_random(seed));
}
