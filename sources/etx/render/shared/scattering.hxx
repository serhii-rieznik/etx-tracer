#pragma once

#include <etx/render/shared/spectrum.hxx>
#include <etx/render/shared/image.hxx>

#include <etx/rhi/rhi.hxx>

namespace etx {

struct RHIContext;

namespace scattering {

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
  RHIPipeline optical_depth_pipeline = {};
  RHIPipeline sky_pipeline = {};
  RHIPipeline sky_finalize_pipeline = {};
  RHIPipeline sun_pipeline = {};
  RHITexture optical_depth_texture = {};
  RHIResourceState optical_depth_texture_state = RHIResourceState::Undefined;
  RHIBuffer sky_light_input_buffer = {};
  RHIBuffer sky_spectrum_input_buffer = {};
  uint32_t sky_input_buffer_capacity = 0u;
  bool initialized = false;
};

SpectralDistribution rayleigh_spectrum();
SpectralDistribution mie_spectrum();
SpectralDistribution ozone_spectrum();

bool gpu_init(RHIContext& rhi, GpuContext& context);

void gpu_cleanup(RHIContext& rhi, GpuContext& context);

bool gpu_precompute_optical_depth(RHIContext& rhi, GpuContext& context);

bool gpu_record_generate_sky_raw(RHIContext& rhi, RHICommandBuffer cmd, GpuContext& context, const Parameters& parameters, const uint2& dimensions,
  const std::vector<LightSource>& light_sources, RHITexture output_texture, RHIResourceState& output_texture_state);

bool gpu_generate_sky(RHIContext& rhi, GpuContext& context, const Parameters& parameters, const uint2& dimensions, const std::vector<LightSource>& light_sources,
  RHITexture output_texture, RHIResourceState& output_texture_state);

bool gpu_create_sky_texture(RHIContext& rhi, GpuContext& context, const Parameters& parameters, const uint2& dimensions, const std::vector<LightSource>& light_sources,
  RHITexture& out_texture, RHIResourceState& out_texture_state);

bool generate_sky_image(RHIContext& rhi, GpuContext& context, const Parameters& parameters, const uint2& dimensions, const std::vector<LightSource>& light_sources, float4* buffer);

bool generate_sun_image(RHIContext& rhi, GpuContext& context, const Parameters& parameters, const uint2& dimensions, const float3& light_direction, const float angular_size,
  float4* buffer);

bool gpu_record_generate_sun(RHIContext& rhi, RHICommandBuffer cmd, GpuContext& context, const Parameters& parameters, const uint2& dimensions, const float3& light_direction,
  float angular_size, RHITexture output_texture, RHIResourceState& output_texture_state);

bool gpu_generate_sun(RHIContext& rhi, GpuContext& context, const Parameters& parameters, const uint2& dimensions, const float3& light_direction, float angular_size,
  RHITexture output_texture, RHIResourceState& output_texture_state);

bool gpu_create_sun_texture(RHIContext& rhi, GpuContext& context, const Parameters& parameters, const uint2& dimensions, const float3& light_direction, float angular_size,
  RHITexture& out_texture, RHIResourceState& out_texture_state);

bool gpu_generate_sun_image(RHIContext& rhi, GpuContext& context, const Parameters& parameters, const uint2& dimensions, const float3& light_direction, float angular_size,
  std::vector<float4>& out_pixels);

}  // namespace scattering
}  // namespace etx
