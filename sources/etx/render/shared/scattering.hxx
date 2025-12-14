#pragma once

#include <etx/render/host/tasks.hxx>
#include <etx/render/shared/spectrum.hxx>
#include <etx/render/shared/image.hxx>

namespace etx {

namespace scattering {

struct ETX_ALIGNED ScatteringSpectrums {
  SpectralDistribution rayleigh = {};
  SpectralDistribution mie = {};
  SpectralDistribution ozone = {};
  SpectralDistribution black = {};
};

struct OpticalDepthData {
  static constexpr uint32_t kWidth = ETX_DEBUG ? 128u : 1024u;
  static constexpr uint32_t kHeight = ETX_DEBUG ? 128u : 1024u;
  static constexpr uint32_t kSize = kWidth * kHeight;

  std::vector<float4> data;

  OpticalDepthData() {
    data.resize(kSize);
  }

  OpticalDepthData(const OpticalDepthData&) = delete;
  OpticalDepthData& operator=(const OpticalDepthData&) = delete;
  OpticalDepthData(OpticalDepthData&&) = delete;
  OpticalDepthData& operator=(OpticalDepthData&&) = delete;

  float4 evaluate(const float2& uv) const {
    float x = uv.x * (kWidth - 1.0f);
    float y = uv.y * (kHeight - 1.0f);

    uint32_t x0 = static_cast<uint32_t>(x);
    uint32_t y0 = static_cast<uint32_t>(y);
    float dx = x - x0;
    float dy = y - y0;

    x0 = min(x0, kWidth - 1);
    y0 = min(y0, kHeight - 1);
    uint32_t x1 = min(x0 + 1, kWidth - 1);
    uint32_t y1 = min(y0 + 1, kHeight - 1);

    const float4& p00 = data[y0 * kWidth + x0];
    const float4& p01 = data[y0 * kWidth + x1];
    const float4& p10 = data[y1 * kWidth + x0];
    const float4& p11 = data[y1 * kWidth + x1];

    float4 result = p00 * (1.0f - dx) * (1.0f - dy) + p01 * dx * (1.0f - dy) + p10 * (1.0f - dx) * dy + p11 * dx * dy;

    return result;
  }
};

void init(TaskScheduler& scheduler, ScatteringSpectrums& spectrums, OpticalDepthData& extinction);

void generate_sky_image(const Parameters& parameters, const uint2& dimensions, const float3& light_direction, const OpticalDepthData& extinction, float4* buffer,
  const ScatteringSpectrums& spectrums, TaskScheduler& scheduler);

void generate_sun_image(const Parameters& parameters, const uint2& dimensions, const float3& light_direction, const float angular_size, float4* buffer,
  const ScatteringSpectrums& spectrums, TaskScheduler& scheduler);

}  // namespace scattering
}  // namespace etx
