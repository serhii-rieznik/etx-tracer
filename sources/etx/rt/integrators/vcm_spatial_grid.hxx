#pragma once

#include <etx/rt/shared/vcm_shared.hxx>
#include <vector>

namespace etx {

struct VCMSpatialGrid {
  VCMSpatialGridData data;

  void construct(const Scene& scene, const VCMLightVertex* samples, uint64_t sample_count, float radius, TaskScheduler& scheduler);

 private:
  std::vector<uint32_t> _cell_ends;
  std::vector<float3> _positions;
  std::vector<float3> _normals;
  std::vector<float3> _w_in;
  std::vector<float> _d_vcm;
  std::vector<float> _d_vm;
  std::vector<uint32_t> _path_lengths;
  std::vector<float3> _throughput_rgb_div_pdf;
};

}  // namespace etx
