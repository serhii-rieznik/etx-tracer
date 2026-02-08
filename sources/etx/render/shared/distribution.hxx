#pragma once

#include <etx/render/interop/interop.hxx>

namespace etx {

struct ETX_ALIGNED Distribution {
  struct Entry {
    float value = 0.0f;
    float pdf = 0.0f;
    float cdf = 0.0f;
    uint32_t reference = kInvalidIndex;
  };
  ArrayView<Entry> values ETX_EMPTY_INIT;
  float total_weight ETX_EMPTY_INIT;

  ETX_SHARED_INLINE uint32_t sample(float rnd, float& pdf) const {
    if ((values.count == 0) || (values.a == nullptr)) {
      pdf = 0.0f;
      return kInvalidIndex;
    }
    auto index = sample(rnd);
    pdf = values[index].pdf;
    return index;
  }

  ETX_SHARED_INLINE uint32_t sample(float rnd) const {
    if ((values.count == 0) || (values.a == nullptr)) {
      return kInvalidIndex;
    }
    uint32_t b = 0;
    uint32_t e = static_cast<uint32_t>(values.count);
    do {
      uint32_t m = b + (e - b) / 2;
      if (values[m].cdf >= rnd) {
        e = m;
      } else {
        b = m;
      }
    } while ((e - b) > 1);
    return b;
  }

  static ETX_SHARED_INLINE Distribution build(Distribution::Entry* entries, uint32_t count) {
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
    entries[count] = {0.0f, 0.0f, 1.0f};

    Distribution result;
    result.values = {entries, count};
    result.total_weight = total_weight;
    return result;
  }
};

}  // namespace etx