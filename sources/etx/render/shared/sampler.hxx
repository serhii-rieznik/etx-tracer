#pragma once

#include <etx/render/shared/base.hxx>

namespace etx {

struct Sampler {
  uint32_t seed = 0;

  Sampler() = default;

  ETX_GPU_CODE Sampler(uint32_t state)
    : seed(state) {
  }

  ETX_GPU_CODE Sampler(uint32_t a, uint32_t b)
    : seed(random_seed(a, b)) {
  }

  ETX_GPU_CODE void init(uint32_t a, uint32_t b) {
    seed = random_seed(a, b);
  }

  ETX_GPU_CODE float next() {
    return next_random(seed);
  }

  ETX_GPU_CODE void start_pixel(const int2&) {
  }

  ETX_GPU_CODE void next_sample() {
  }

  static ETX_GPU_CODE uint32_t random_seed(const uint32_t val0, const uint32_t val1) {
    uint32_t v0 = val0;
    uint32_t v1 = val1;
    uint32_t s0 = 0;
    for (uint32_t n = 0; n < 16; ++n) {
      s0 += 0x9e3779b9;
      v0 += ((v1 << 4) + 0xa341316c) ^ (v1 + s0) ^ ((v1 >> 5) + 0xc8013ea4);
      v1 += ((v0 << 4) + 0xad90777d) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7e95761e);
    }
    return v0;
  }

  static ETX_GPU_CODE float next_random(uint32_t& previous) {
    previous = previous * 1664525u + 1013904223u;
    union {
      uint32_t i;
      float f;
    } wrap = {(previous >> 9) | 0x3f800000};
    return wrap.f - 1.0f;
  }
};

}  // namespace etx
