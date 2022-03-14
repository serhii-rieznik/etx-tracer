#pragma once

#include <etx/render/shared/distribution.hxx>

namespace etx {

struct DistributionBuilder {
  DistributionBuilder(Distribution& dist, uint32_t capacity)
    : _dist(dist)
    , _capacity(capacity + 1) {
    ETX_ASSERT(capacity > 0);
    ETX_ASSERT(_dist.size == 0);
    ETX_ASSERT(_dist.values == nullptr);
    _dist.size = capacity;
    _dist.values = reinterpret_cast<Distribution::Entry*>(calloc(_capacity, sizeof(Distribution::Entry)));
  }

  void add(float value) {
    ETX_ASSERT(_size + 1 <= _capacity);
    _dist.values[_size++] = {value, 0.0f, 0.0f};
  }

  void set(uint32_t loc, float value) {
    ETX_ASSERT(loc <= _capacity);
    _dist.values[loc] = {value, 0.0f, 0.0f};
  }

  void set_size(uint32_t size) {
    _size = size;
  }

  void finalize() {
    ETX_ASSERT(_size + 1 == _capacity);

    _dist.total_weight = 0.0f;
    for (uint32_t i = 0; i < _size; ++i) {
      _dist.values[i].cdf = _dist.total_weight;
      _dist.total_weight += _dist.values[i].value;
    }

    if (_dist.total_weight == 0.0f) {
      for (uint64_t i = 0; i < _dist.size; ++i) {
        _dist.values[i].value = 1.0f;
        _dist.values[i].pdf = 1.0f / float(_dist.size);
        _dist.values[i].cdf = float(i) / float(_dist.size);
      }
    } else {
      for (uint32_t i = 0; i < _size; ++i) {
        _dist.values[i].pdf = _dist.values[i].value / _dist.total_weight;
        _dist.values[i].cdf /= _dist.total_weight;
      }
    }

    _dist.values[_size++] = {0.0f, 0.0f, 1.0f};
  }

 private:
  Distribution& _dist;
  uint32_t _capacity = 0;
  uint32_t _size = 0;
};

}  // namespace etx
