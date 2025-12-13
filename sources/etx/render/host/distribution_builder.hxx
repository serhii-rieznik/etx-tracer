#pragma once

#include <etx/render/shared/distribution.hxx>

namespace etx {

struct DistributionBuilder {
  DistributionBuilder(Distribution& dist, Distribution::Entry* external_buffer, uint32_t capacity, uint32_t initial_size = 0)
    : _dist(dist)
    , _values{external_buffer, capacity}
    , _size(initial_size) {
  }

  void add(float value) {
    ETX_ASSERT(_size + 1 <= _values.count);
    _values[_size++] = {value, 0.0f, 0.0f};
  }

  void set(uint32_t loc, float value) {
    ETX_ASSERT(loc < _values.count);
    _values[loc] = {value, 0.0f, 0.0f};
  }

  static float finalize_entries(Distribution::Entry* entries, uint32_t count) {
    float total_weight = 0.0f;
    for (uint32_t i = 0; i < count; ++i) {
      entries[i].cdf = total_weight;
      total_weight += entries[i].value;
    }

    if (total_weight == 0.0f) {
      for (uint32_t i = 0; i < count; ++i) {
        entries[i].value = 1.0f;
        entries[i].pdf = 1.0f / float(count);
        entries[i].cdf = float(i) / float(count);
      }
    } else {
      for (uint32_t i = 0; i < count; ++i) {
        entries[i].pdf = entries[i].value / total_weight;
        entries[i].cdf /= total_weight;
      }
    }

    // Add sentinel entry
    entries[count] = {0.0f, 0.0f, 1.0f};

    return total_weight;
  }

  void finalize() {
    float total_weight = finalize_entries(_values.a, _size);
    _dist.total_weight = total_weight;
    _dist.values = {_values.a, _size};
  }

 private:
  Distribution& _dist;
  ArrayView<Distribution::Entry> _values;
  uint32_t _size = 0;
};

}  // namespace etx
