#pragma once

#include "interop.hxx"
#include "sampler.hxx"

struct Sampler {
  uint32_t seed ETX_INIT(0u);
  float fixed_u ETX_INIT(0.0f);
  float fixed_v ETX_INIT(0.0f);
  float fixed_w ETX_INIT(0.0f);
};

ETX_SHARED_INLINE float bsdf_sampler_next(ETX_INOUT(Sampler, sampler)) {
  return sampler_next_random(sampler.seed);
}

ETX_SHARED_INLINE float2 bsdf_sampler_next_2d(ETX_INOUT(Sampler, sampler)) {
  return float2(bsdf_sampler_next(sampler), bsdf_sampler_next(sampler));
}

ETX_SHARED_INLINE void bsdf_sampler_init(ETX_INOUT(Sampler, sampler), uint32_t value_0, uint32_t value_1) {
  sampler.seed = sampler_random_seed(value_0, value_1);
}

ETX_SHARED_INLINE void bsdf_sampler_push_fixed(ETX_INOUT(Sampler, sampler), float u, float v, float w) {
  sampler.fixed_u = u;
  sampler.fixed_v = v;
  sampler.fixed_w = w;
}

ETX_SHARED_INLINE void bsdf_sampler_pop_fixed(ETX_INOUT(Sampler, sampler)) {
  sampler.fixed_u = 0.0f;
  sampler.fixed_v = 0.0f;
  sampler.fixed_w = 0.0f;
}

ETX_SHARED_INLINE bool bsdf_sampler_has_fixed(ETX_IN(Sampler, sampler)) {
  return ((sampler.fixed_u * sampler.fixed_u) + (sampler.fixed_v * sampler.fixed_v) + (sampler.fixed_w * sampler.fixed_w)) > kEpsilon;
}
