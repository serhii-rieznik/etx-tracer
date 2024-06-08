#pragma once

#include <etx/render/shared/base.hxx>

namespace etx {

struct Sampler {
  uint32_t seed = 0;

  ETX_SHARED_CODE Sampler() {
  }

  ETX_SHARED_CODE Sampler(uint32_t state)
    : seed(state) {
  }

  ETX_SHARED_CODE Sampler(uint32_t a, uint32_t b)
    : seed(random_seed(a, b)) {
  }

  ETX_SHARED_CODE void init(uint32_t a, uint32_t b) {
    seed = random_seed(a, b);
  }

  ETX_SHARED_CODE float next() {
    return next_random(seed);
  }

  ETX_SHARED_CODE float2 next_2d() {
    float a = next();
    float b = next();
    return {a, b};
  }

  static ETX_SHARED_CODE uint32_t random_seed(const uint32_t val0, const uint32_t val1) {
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

  static ETX_SHARED_CODE float next_random(uint32_t& seed) {
    seed = (seed ^ 61u) ^ (seed >> 16u);
    seed *= 9u;
    seed = seed ^ (seed >> 4u);
    seed *= 0x27d4eb2du;
    seed = seed ^ (seed >> 15u);
    union {
      uint32_t i;
      float f;
    } wrap = {(seed >> 9) | 0x3f800000u};
    return wrap.f - 1.0f;
  }
};

}  // namespace etx
