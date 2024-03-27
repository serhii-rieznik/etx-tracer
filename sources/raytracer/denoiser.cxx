#include "denoiser.hxx"

#include <etx/core/log.hxx>
#include <etx/render/shared/spectrum.hxx>

#include <OpenImageDenoise/oidn.hpp>

namespace etx {

struct DenoiserImpl {
  OIDNDevice device = {};

  bool check_errors() {
    const char* error_str = nullptr;
    auto err = oidnGetDeviceError(device, &error_str);
    if (err == OIDN_ERROR_NONE)
      return true;

    log::error(error_str);
    return false;
  }
};

Denoiser::Denoiser() {
  ETX_PIMPL_INIT(Denoiser);
}

Denoiser::~Denoiser() {
  ETX_PIMPL_CLEANUP(Denoiser);
}

void Denoiser::init() {
  int32_t device_count = oidnGetNumPhysicalDevices();
  _private->device = oidnNewDevice(OIDN_DEVICE_TYPE_DEFAULT);
  oidnCommitDevice(_private->device);
}

void Denoiser::shutdown() {
  oidnReleaseDevice(_private->device);
}

void Denoiser::denoise(const float4* image, const float4* albedo, const float4* normal, float4* output, const uint2 size) {
  auto color_buffer = oidnNewBuffer(_private->device, sizeof(float3) * size.x * size.y);
  auto albedo_buffer = oidnNewBuffer(_private->device, sizeof(float3) * size.x * size.y);
  auto normal_buffer = oidnNewBuffer(_private->device, sizeof(float3) * size.x * size.y);

  auto filter = oidnNewFilter(_private->device, "RT");
  oidnSetFilterImage(filter, "color", color_buffer, OIDN_FORMAT_FLOAT3, size.x, size.y, 0, 0, 0);
  oidnSetFilterImage(filter, "output", color_buffer, OIDN_FORMAT_FLOAT3, size.x, size.y, 0, 0, 0);
  oidnSetFilterImage(filter, "normal", normal_buffer, OIDN_FORMAT_FLOAT3, size.x, size.y, 0, 0, 0);
  oidnSetFilterImage(filter, "albedo", albedo_buffer, OIDN_FORMAT_FLOAT3, size.x, size.y, 0, 0, 0);
  oidnSetFilterBool(filter, "cleanAux", true);
  oidnSetFilterBool(filter, "hdr", true);
  oidnCommitFilter(filter);

  auto albedo_prefilter = oidnNewFilter(_private->device, "RT");  // same filter type as for beauty
  oidnSetFilterImage(albedo_prefilter, "albedo", albedo_buffer, OIDN_FORMAT_FLOAT3, size.x, size.y, 0, 0, 0);
  oidnSetFilterImage(albedo_prefilter, "output", albedo_buffer, OIDN_FORMAT_FLOAT3, size.x, size.y, 0, 0, 0);
  oidnCommitFilter(albedo_prefilter);

  auto normal_prefilter = oidnNewFilter(_private->device, "RT");  // same filter type as for beauty
  oidnSetFilterImage(normal_prefilter, "normal", normal_buffer, OIDN_FORMAT_FLOAT3, size.x, size.y, 0, 0, 0);
  oidnSetFilterImage(normal_prefilter, "output", normal_buffer, OIDN_FORMAT_FLOAT3, size.x, size.y, 0, 0, 0);
  oidnCommitFilter(normal_prefilter);

  if (_private->check_errors()) {
    auto ptr = reinterpret_cast<float3*>(oidnGetBufferData(color_buffer));
    auto alb = reinterpret_cast<float3*>(oidnGetBufferData(albedo_buffer));
    auto nrm = reinterpret_cast<float3*>(oidnGetBufferData(normal_buffer));
    for (uint32_t i = 0, e = size.x * size.y; i < e; ++i) {
      ptr[i] = max(float3{}, spectrum::xyz_to_rgb({image[i].x, image[i].y, image[i].z}));
      alb[i] = max(float3{}, spectrum::xyz_to_rgb({albedo[i].x, albedo[i].y, albedo[i].z}));
      nrm[i] = {normal[i].x, normal[i].y, normal[i].z};
    }
  }

  if (_private->check_errors()) {
    oidnExecuteFilter(albedo_prefilter);
  }

  if (_private->check_errors()) {
    oidnExecuteFilter(normal_prefilter);
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
  oidnReleaseBuffer(albedo_buffer);
  oidnReleaseFilter(normal_prefilter);
  oidnReleaseFilter(albedo_prefilter);
  oidnReleaseFilter(filter);
}

}  // namespace etx
