#include <etx/core/core.hxx>

#include <etx/rt/shared/path_tracing_shared.hxx>

namespace etx {

float2 sample_blue_noise(const uint2& pixel, const uint32_t total_samples, const uint32_t current_sample, uint32_t dimension) {
  auto smp = BNSampler(pixel.x, pixel.y, total_samples, current_sample);
  float u = smp.get(dimension + 0u);
  float v = smp.get(dimension + 1u);
  return {u, v};
}

}  // namespace etx
