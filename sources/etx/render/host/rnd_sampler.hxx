#pragma once

#include <etx/render/shared/sampler.hxx>

#include <random>

namespace etx {

struct RNDSampler : public Sampler {
  RNDSampler() {
    init(_dis(_gen), _dis(_gen));
  }

 private:
  std::random_device _rd;
  std::mt19937 _gen = std::mt19937(_rd());
  std::uniform_int_distribution<uint32_t> _dis;
};

}  // namespace etx
