#pragma once

#include <cstdint>

struct BNSampler {
  BNSampler() = default;
  BNSampler(uint32_t pixel_x, uint32_t pixel_y, uint32_t target_samples, uint32_t current_sample);

  void init(uint32_t pixel_x, uint32_t pixel_y, uint32_t target_samples, uint32_t current_sample);
  float next();
  float get(uint32_t dimension) const;

 private:
  struct Impl;
  uint8_t _impl_data[32] = {};
};
