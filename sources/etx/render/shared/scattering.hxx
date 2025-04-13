#pragma once

#include <etx/render/host/tasks.hxx>
#include <etx/render/shared/spectrum.hxx>
#include <etx/render/shared/image.hxx>

namespace etx {

namespace scattering {

struct Parameters {
  float altitude = 1000.0f;
  float anisotropy = 0.825f;
  float rayleigh_scale = 1.0f;
  float mie_scale = 1.0f;
  float ozone_scale = 1.0f;
};

void init(TaskScheduler& scheduler, Pointer<Spectrums> spectrums, Image& extinction);
void generate_sky_image(const Parameters& parameters, const uint2& dimensions, const float3& light_direction, Image& extinction, float4* buffer, TaskScheduler& scheduler);
void generate_sun_image(const Parameters& parameters, const uint2& dimensions, const float3& light_direction, const float angular_size, float4* buffer, TaskScheduler& scheduler);

}  // namespace scattering
}  // namespace etx
