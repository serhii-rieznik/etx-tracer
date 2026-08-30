#include <etx/core/core.hxx>

#include <etx/rt/shared/path_tracing_shared.hxx>

namespace etx {

float2 sample_blue_noise_at_translated_pixel(const uint2& sample_pixel, const uint32_t total_samples, const uint32_t current_sample, const uint32_t dimension) {
  const BNSampler smp(sample_pixel.x, sample_pixel.y, total_samples, current_sample);
  const float u = smp.get(dimension + 0u);
  const float v = smp.get(dimension + 1u);
  return {u, v};
}

}  // namespace etx
