#pragma once

#include <etx/render/shared/distribution.hxx>

namespace etx {

struct DistributionBuilder {
  DistributionBuilder(Distribution& dist, uint32_t size)
    : _dist(dist)
    , _capacity(size + 1) {
    ETX_ASSERT(size > 0);
    _values.count = size;
    _values.a = reinterpret_cast<Distribution::Entry*>(calloc(_capacity, sizeof(Distribution::Entry)));
  }

  void add(float value) {
    ETX_ASSERT(_size + 1 <= _capacity);
    _values[_size++] = {value, 0.0f, 0.0f};
  }

  void set(uint32_t loc, float value) {
    ETX_ASSERT(loc <= _capacity);
    _values[loc] = {value, 0.0f, 0.0f};
  }

  void set_size(uint32_t size) {
    _size = size;
  }

  void finalize() {
    ETX_ASSERT(_size + 1 == _capacity);

    float total_weight = 0.0f;
    for (uint32_t i = 0; i < _size; ++i) {
      _values[i].cdf = total_weight;
      total_weight += _values[i].value;
    }

    if (total_weight == 0.0f) {
      for (uint64_t i = 0; i < _values.count; ++i) {
        _values[i].value = 1.0f;
        _values[i].pdf = 1.0f / float(_values.count);
        _values[i].cdf = float(i) / float(_values.count);
      }
    } else {
      for (uint32_t i = 0; i < _size; ++i) {
        _values[i].pdf = _values[i].value / total_weight;
        _values[i].cdf /= total_weight;
      }
    }

    _values.a[_size++] = {0.0f, 0.0f, 1.0f};

    if (_dist.values.a != nullptr) {
      free(_dist.values.a);
    }
    _dist.total_weight = total_weight;
    _dist.values = _values;
  }

 private:
  Distribution& _dist;
  ArrayView<Distribution::Entry> _values;
  uint32_t _capacity = 0;
  uint32_t _size = 0;
};

}  // namespace etx
