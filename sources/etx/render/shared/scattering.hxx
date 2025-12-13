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

void init(TaskScheduler& scheduler, ScatteringSpectrums& spectrums, Image& extinction);

void generate_sky_image(const Parameters& parameters, const uint2& dimensions, const float3& light_direction, Image& extinction, float4* buffer,
  const ScatteringSpectrums& spectrums, TaskScheduler& scheduler);

void generate_sun_image(const Parameters& parameters, const uint2& dimensions, const float3& light_direction, const float angular_size, float4* buffer,
  const ScatteringSpectrums& spectrums, TaskScheduler& scheduler);

}  // namespace scattering
}  // namespace etx
