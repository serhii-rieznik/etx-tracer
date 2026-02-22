#pragma once

#include <etx/render/host/tasks.hxx>
#include <etx/render/shared/spectrum.hxx>
#include <etx/render/shared/image.hxx>

namespace etx {
struct RHIContext;

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
  OpticalDepthData(OpticalDepthData&&) = default;
  OpticalDepthData& operator=(OpticalDepthData&&) = default;

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

struct LightSource {
  SpectralDistribution emission_spectrum = {};
  float3 direction = {};
  float angular_size = {};
  float intensity_scale = {};
};

struct GpuContext {
  bool initialized = false;
};

struct GpuOpticalDepthRequest {
  Parameters atmosphere = {};
  uint2 dimensions = {OpticalDepthData::kWidth, OpticalDepthData::kHeight};
};

struct GpuSkyRequest {
  Parameters atmosphere = {};
  uint2 dimensions = {};
  const std::vector<LightSource>* light_sources = nullptr;
};

struct GpuSunRequest {
  Parameters atmosphere = {};
  uint2 dimensions = {};
  float3 light_direction = {};
  float angular_size = 0.0f;
};

void init(TaskScheduler& scheduler, ScatteringSpectrums& spectrums, OpticalDepthData& extinction);

OpticalDepthData precompute_optical_depth(TaskScheduler& scheduler);

void generate_sky_image(const Parameters& parameters, const uint2& dimensions, const std::vector<LightSource>& light_sources, const OpticalDepthData& extinction, float4* buffer,
  TaskScheduler& scheduler);

void generate_sun_image(const Parameters& parameters, const uint2& dimensions, const float3& light_direction, const float angular_size, float4* buffer, TaskScheduler& scheduler);

bool gpu_init(RHIContext& rhi, GpuContext& context);
bool gpu_reload_shaders(RHIContext& rhi, GpuContext& context);
void gpu_cleanup(RHIContext& rhi, GpuContext& context);

bool gpu_precompute_optical_depth_texture(RHIContext& rhi, const GpuOpticalDepthRequest& request, GpuContext& context);
bool gpu_generate_sky_image(RHIContext& rhi, const GpuSkyRequest& request, GpuContext& context);
bool gpu_generate_sun_image(RHIContext& rhi, const GpuSunRequest& request, GpuContext& context);
bool gpu_download_sky_image(RHIContext& rhi, std::vector<float4>& out_pixels, uint2& out_dimensions, GpuContext& context);

}  // namespace scattering
}  // namespace etx
