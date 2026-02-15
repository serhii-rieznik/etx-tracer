#pragma once

#include "interop.hxx"

struct ETX_ALIGNED DistributionEntry {
  float value ETX_INIT(0.0f);
  float pdf ETX_INIT(0.0f);
  float cdf ETX_INIT(0.0f);
  uint32_t reference ETX_INIT(kInvalidIndex);
};

struct ETX_ALIGNED DistributionSearchRange {
  uint32_t begin ETX_INIT(0u);
  uint32_t end ETX_INIT(0u);
};

ETX_SHARED_INLINE DistributionSearchRange distribution_search_begin(uint32_t count) {
  DistributionSearchRange result;
  result.begin = 0u;
  result.end = count;
  return result;
}

ETX_SHARED_INLINE bool distribution_search_active(ETX_IN(DistributionSearchRange, range)) {
  return (range.end - range.begin) > 1u;
}

ETX_SHARED_INLINE uint32_t distribution_search_middle(ETX_IN(DistributionSearchRange, range)) {
  return range.begin + (range.end - range.begin) / 2u;
}

ETX_SHARED_INLINE void distribution_search_update(ETX_INOUT(DistributionSearchRange, range), uint32_t middle, float middle_cdf, float rnd) {
  if (middle_cdf >= rnd) {
    range.end = middle;
  } else {
    range.begin = middle;
  }
}
