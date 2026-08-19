#pragma once

#include <cstdint>
#include <vector>

bool build_blue_noise_gpu_data(uint32_t target_samples, std::vector<uint8_t>& data);

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
