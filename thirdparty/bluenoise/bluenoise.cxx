#include <bluenoise.hxx>

#include <assert.h>
#include <stdio.h>
#include <bluenoise_shared.hpp>

#include <algorithm>
#include <cstddef>

#if defined(_MSC_VER)
# include <intrin.h>
#endif

namespace spp_1 {
#include <samplerBlueNoiseErrorDistribution_128x128_OptimizedFor_2d2d2d2d_1spp.hpp>
}
namespace spp_2 {
#include <samplerBlueNoiseErrorDistribution_128x128_OptimizedFor_2d2d2d2d_2spp.hpp>
}
namespace spp_4 {
#include <samplerBlueNoiseErrorDistribution_128x128_OptimizedFor_2d2d2d2d_4spp.hpp>
}
namespace spp_8 {
#include <samplerBlueNoiseErrorDistribution_128x128_OptimizedFor_2d2d2d2d_8spp.hpp>
}
namespace spp_16 {
#include <samplerBlueNoiseErrorDistribution_128x128_OptimizedFor_2d2d2d2d_16spp.hpp>
}
namespace spp_32 {
#include <samplerBlueNoiseErrorDistribution_128x128_OptimizedFor_2d2d2d2d_32spp.hpp>
}
namespace spp_64 {
#include <samplerBlueNoiseErrorDistribution_128x128_OptimizedFor_2d2d2d2d_64spp.hpp>
}
namespace spp_128 {
#include <samplerBlueNoiseErrorDistribution_128x128_OptimizedFor_2d2d2d2d_128spp.hpp>
}
namespace spp_256 {
#include <samplerBlueNoiseErrorDistribution_128x128_OptimizedFor_2d2d2d2d_256spp.hpp>
}

struct BNSampler::Impl {
  // (int pixel_i, int pixel_j, int sampleIndex, int sampleDimension)
  using sampling_function = float (*)(int, int, int, int);

  static constexpr sampling_function sampling_functions[] = {
    spp_1::samplerBlueNoiseErrorDistribution_128x128_OptimizedFor_2d2d2d2d_1spp,
    spp_2::samplerBlueNoiseErrorDistribution_128x128_OptimizedFor_2d2d2d2d_2spp,
    spp_4::samplerBlueNoiseErrorDistribution_128x128_OptimizedFor_2d2d2d2d_4spp,
    spp_8::samplerBlueNoiseErrorDistribution_128x128_OptimizedFor_2d2d2d2d_8spp,
    spp_16::samplerBlueNoiseErrorDistribution_128x128_OptimizedFor_2d2d2d2d_16spp,
    spp_32::samplerBlueNoiseErrorDistribution_128x128_OptimizedFor_2d2d2d2d_32spp,
    spp_64::samplerBlueNoiseErrorDistribution_128x128_OptimizedFor_2d2d2d2d_64spp,
    spp_128::samplerBlueNoiseErrorDistribution_128x128_OptimizedFor_2d2d2d2d_128spp,
    spp_256::samplerBlueNoiseErrorDistribution_128x128_OptimizedFor_2d2d2d2d_256spp,
  };

  sampling_function func;
  uint32_t dim = 0;
  uint32_t px = 0;
  uint32_t py = 0;
  uint32_t sample = 0;
};

uint32_t next_power(uint32_t v) {
  v--;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  v++;
  return v;
}

namespace {

constexpr size_t kBlueNoiseTileValueCount = 128u * 128u * 8u;
constexpr size_t kBlueNoiseSobolValueCount = 256u * 8u;
constexpr size_t kBlueNoiseGPUDataSize = 2u * kBlueNoiseTileValueCount + kBlueNoiseSobolValueCount;

template <size_t RankingCount, size_t ScramblingCount, size_t SobolCount>
bool build_blue_noise_gpu_data_from_tables(const int (&ranking)[RankingCount], const int (&scrambling)[ScramblingCount], const int (&sobol)[SobolCount],
  std::vector<uint8_t>& data) {
  static_assert(RankingCount == kBlueNoiseTileValueCount);
  static_assert(ScramblingCount == kBlueNoiseTileValueCount);
  static_assert(SobolCount == 256u * 256u);

  data.resize(kBlueNoiseGPUDataSize);
  std::transform(ranking, ranking + RankingCount, data.begin(), [](int value) {
    return static_cast<uint8_t>(value);
  });
  std::transform(scrambling, scrambling + ScramblingCount, data.begin() + kBlueNoiseTileValueCount, [](int value) {
    return static_cast<uint8_t>(value);
  });

  uint8_t* compact_sobol = data.data() + 2u * kBlueNoiseTileValueCount;
  for (size_t sample_index = 0u; sample_index < 256u; ++sample_index) {
    for (size_t dimension = 0u; dimension < 8u; ++dimension) {
      compact_sobol[sample_index * 8u + dimension] = static_cast<uint8_t>(sobol[sample_index * 256u + dimension]);
    }
  }
  return true;
}

}  // namespace

bool build_blue_noise_gpu_data(uint32_t target_samples, std::vector<uint8_t>& data) {
  const uint32_t clamped_samples = std::max(1u, std::min(target_samples, 256u));
  switch (next_power(clamped_samples)) {
    case 1u:
      return build_blue_noise_gpu_data_from_tables(spp_1::rankingTile, spp_1::scramblingTile, sobol_256spp_256d, data);
    case 2u:
      return build_blue_noise_gpu_data_from_tables(spp_2::rankingTile, spp_2::scramblingTile, sobol_256spp_256d, data);
    case 4u:
      return build_blue_noise_gpu_data_from_tables(spp_4::rankingTile, spp_4::scramblingTile, sobol_256spp_256d, data);
    case 8u:
      return build_blue_noise_gpu_data_from_tables(spp_8::rankingTile, spp_8::scramblingTile, sobol_256spp_256d, data);
    case 16u:
      return build_blue_noise_gpu_data_from_tables(spp_16::rankingTile, spp_16::scramblingTile, sobol_256spp_256d, data);
    case 32u:
      return build_blue_noise_gpu_data_from_tables(spp_32::rankingTile, spp_32::scramblingTile, sobol_256spp_256d, data);
    case 64u:
      return build_blue_noise_gpu_data_from_tables(spp_64::rankingTile, spp_64::scramblingTile, sobol_256spp_256d, data);
    case 128u:
      return build_blue_noise_gpu_data_from_tables(spp_128::rankingTile, spp_128::scramblingTile, sobol_256spp_256d, data);
    case 256u:
      return build_blue_noise_gpu_data_from_tables(spp_256::rankingTile, spp_256::scramblingTile, sobol_256spp_256d, data);
    default:
      data.clear();
      return false;
  }
}

BNSampler::BNSampler(uint32_t pixel_x, uint32_t pixel_y, uint32_t target_samples, uint32_t current_sample) {
  static_assert(sizeof(_impl_data) >= sizeof(Impl), "Not enought storage");

  target_samples = next_power((target_samples == 0u) ? 1u : (target_samples > 256u ? 256u : target_samples));

#if defined(_MSC_VER)
  unsigned long p = 1;
  _BitScanForward(&p, target_samples);
#else
  uint32_t p = 31u - __builtin_clz(target_samples);
#endif

  constexpr auto sampling_functions_count = sizeof(BNSampler::Impl::sampling_functions) / sizeof(BNSampler::Impl::sampling_functions[0]);
  assert(p < sampling_functions_count);

  auto impl = reinterpret_cast<Impl*>(_impl_data);
  impl->dim = 0;
  impl->px = pixel_x;
  impl->py = pixel_y;
  impl->sample = current_sample;
  impl->func = BNSampler::Impl::sampling_functions[p];
}

float BNSampler::next() {
  auto impl = reinterpret_cast<Impl*>(_impl_data);
  return get(impl->dim++);
  // impl->func(impl->px, impl->py, impl->sample, impl->dim++);
}

float BNSampler::get(uint32_t dimension) const {
  auto impl = reinterpret_cast<const Impl*>(_impl_data);
  return impl->func(impl->px, impl->py, impl->sample, dimension);
}
