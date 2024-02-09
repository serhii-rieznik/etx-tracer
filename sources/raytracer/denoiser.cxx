#include "denoiser.hxx"

#include <etx/core/log.hxx>
#include <etx/render/shared/spectrum.hxx>

#if (ETX_PLATFORM_WINDOWS)
# include <OpenImageDenoise/oidn.hpp>
#endif

namespace etx {

struct DenoiserImpl {
#if (ETX_PLATFORM_WINDOWS)
  OIDNDevice device = {};

  bool check_errors() {
    const char* error_str = nullptr;
    auto err = oidnGetDeviceError(device, &error_str);
    if (err == OIDN_ERROR_NONE)
      return true;

    log::error(error_str);
    return false;
  }
#endif
};

Denoiser::Denoiser() {
  ETX_PIMPL_INIT(Denoiser);
}

Denoiser::~Denoiser() {
  ETX_PIMPL_CLEANUP(Denoiser);
}

void Denoiser::init() {
#if (ETX_PLATFORM_WINDOWS)
  int32_t device_count = oidnGetNumPhysicalDevices();
  _private->device = oidnNewDevice(OIDN_DEVICE_TYPE_DEFAULT);
  oidnCommitDevice(_private->device);
#endif
}

void Denoiser::shutdown() {
#if (ETX_PLATFORM_WINDOWS)
  oidnReleaseDevice(_private->device);
#endif
}

void Denoiser::denoise(const float4* image, const float4* albedo, const float4* normal, float4* output, const uint2 size) {
#if (ETX_PLATFORM_WINDOWS)
  auto color_buffer = oidnNewBuffer(_private->device, sizeof(float3) * size.x * size.y);
  auto normal_buffer = oidnNewBuffer(_private->device, sizeof(float3) * size.x * size.y);
  auto filter = oidnNewFilter(_private->device, "RT");

  oidnSetFilterImage(filter, "color", color_buffer, OIDN_FORMAT_FLOAT3, size.x, size.y, 0, 0, 0);
  oidnSetFilterImage(filter, "output", color_buffer, OIDN_FORMAT_FLOAT3, size.x, size.y, 0, 0, 0);
  oidnSetFilterBool(filter, "hdr", true);
  oidnCommitFilter(filter);

  if (_private->check_errors()) {
    auto ptr = reinterpret_cast<float3*>(oidnGetBufferData(color_buffer));
    for (uint32_t i = 0, e = size.x * size.y; i < e; ++i) {
      ptr[i] = max(float3{}, spectrum::xyz_to_rgb({image[i].x, image[i].y, image[i].z}));
    }
  }

  if (_private->check_errors()) {
    oidnExecuteFilter(filter);
  }

  if (_private->check_errors()) {
    auto ptr = reinterpret_cast<float3*>(oidnGetBufferData(color_buffer));
    for (uint32_t i = 0, e = size.x * size.y; i < e; ++i) {
      float3 xyz = max(float3{}, spectrum::rgb_to_xyz({ptr[i].x, ptr[i].y, ptr[i].z}));
      output[i] = {xyz.x, xyz.y, xyz.z, image[i].w};
    }
  }

  oidnReleaseBuffer(color_buffer);
  oidnReleaseBuffer(normal_buffer);
#else
  for (uint32_t i = 0, e = size.x * size.y; i < e; ++i) {
    output[i] = image[i];
  }
#endif
}

}  // namespace etx
