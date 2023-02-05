#pragma once

#include <etx/rt/shared/vcm_shared.hxx>
#include <vector>

namespace etx {

struct VCMSpatialGrid {
  VCMSpatialGridData data;

  void construct(const Scene& scene, const VCMLightVertex* samples, uint64_t sample_count, float radius);

 private:
  std::vector<uint32_t> _indices;
  std::vector<uint32_t> _cell_ends;
};

}  // namespace etx
