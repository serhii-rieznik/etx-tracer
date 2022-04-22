#pragma once

#include <etx/render/shared/distribution.hxx>

namespace etx {

struct DistributionBuilder {
  DistributionBuilder(Distribution& dist, uint32_t size)
    : _dist(dist)
    , _capacity(size + 1) {
    ETX_ASSERT(size > 0);
    ETX_ASSERT(_dist.values.count == 0);
    ETX_ASSERT(_dist.values.a == nullptr);
    _dist.values.count = size;
    _dist.values.a = reinterpret_cast<Distribution::Entry*>(calloc(_capacity, sizeof(Distribution::Entry)));
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
      for (uint64_t i = 0; i < _dist.values.count; ++i) {
        _dist.values[i].value = 1.0f;
        _dist.values[i].pdf = 1.0f / float(_dist.values.count);
        _dist.values[i].cdf = float(i) / float(_dist.values.count);
      }
    } else {
      for (uint32_t i = 0; i < _size; ++i) {
        _dist.values[i].pdf = _dist.values[i].value / _dist.total_weight;
        _dist.values[i].cdf /= _dist.total_weight;
      }
    }

    _dist.values.a[_size++] = {0.0f, 0.0f, 1.0f};
  }

 private:
  Distribution& _dist;
  uint32_t _capacity = 0;
  uint32_t _size = 0;
};

}  // namespace etx
