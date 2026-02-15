#pragma once

#include <etx/render/interop/interop.hxx>
#include <etx/render/interop/sampler.hxx>

namespace etx {

struct Sampler {
  uint32_t seed = 0;
  float fixed_u = 0.0f;
  float fixed_v = 0.0f;
  float fixed_w = 0.0f;

  Sampler() {
  }

  Sampler(uint32_t state)
    : seed(state) {
  }

  Sampler(uint32_t a, uint32_t b)
    : seed(random_seed(a, b)) {
  }

  void init(uint32_t a, uint32_t b) {
    seed = random_seed(a, b);
  }

  float next() {
    return next_random(seed);
  }

  float2 next_2d() {
    float a = next();
    float b = next();
    return {a, b};
  }

  void push_fixed(float u, float v, float w) {
    fixed_u = u;
    fixed_v = v;
    fixed_w = w;
  }

  void pop_fixed() {
    fixed_u = 0.0f;
    fixed_v = 0.0f;
    fixed_w = 0.0f;
  }

  bool has_fixed() const {
    return (sqr(fixed_u) + sqr(fixed_v) + sqr(fixed_w)) > kEpsilon;
  }

  static uint32_t random_seed(const uint32_t val0, const uint32_t val1) {
    return ::sampler_random_seed(val0, val1);
  }

  static float next_random(uint32_t& seed) {
    return ::sampler_next_random(seed);
  }
};

}  // namespace etx
