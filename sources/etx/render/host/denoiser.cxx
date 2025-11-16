#include <etx/render/host/denoiser.hxx>

#include <etx/core/core.hxx>
#include <etx/render/shared/spectrum.hxx>

#include <OpenImageDenoise/oidn.hpp>

namespace etx {

struct DenoiserImpl {
  OIDNDevice device = {};
  OIDNFilter normal_prefilter = {};
  OIDNFilter albedo_prefilter = {};
  OIDNFilter filter = {};
  OIDNBuffer color_buffer_in = {};
  OIDNBuffer color_buffer_out = {};
  OIDNBuffer normal_buffer = {};
  OIDNBuffer albedo_buffer = {};
  uint2 size = {};
  uint64_t data_size = 0;

  bool check_errors() {
    const char* error_str = nullptr;
    auto err = oidnGetDeviceError(device, &error_str);
    if (err == OIDN_ERROR_NONE)
      return true;

    log::error(error_str);
    return false;
  }

  void release_buffers() {
    if (color_buffer_in) {
      oidnReleaseBuffer(color_buffer_in);
      color_buffer_in = {};
    }
    if (color_buffer_out) {
      oidnReleaseBuffer(color_buffer_out);
      color_buffer_out = {};
    }
    if (normal_buffer) {
      oidnReleaseBuffer(normal_buffer);
      normal_buffer = {};
    }
    if (albedo_buffer) {
      oidnReleaseBuffer(albedo_buffer);
      albedo_buffer = {};
    }
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

  _private->albedo_prefilter = oidnNewFilter(_private->device, "RT");  // same filter type as for beauty
  oidnSetFilterInt(_private->albedo_prefilter, "quality", OIDN_QUALITY_HIGH);

  _private->normal_prefilter = oidnNewFilter(_private->device, "RT");  // same filter type as for beauty
  oidnSetFilterInt(_private->normal_prefilter, "quality", OIDN_QUALITY_HIGH);

  _private->filter = oidnNewFilter(_private->device, "RT");
  oidnSetFilterBool(_private->filter, "cleanAux", true);
  oidnSetFilterBool(_private->filter, "hdr", true);
  oidnSetFilterInt(_private->filter, "quality", OIDN_QUALITY_HIGH);

  _private->check_errors();
}

void Denoiser::shutdown() {
  _private->release_buffers();
  oidnReleaseFilter(_private->normal_prefilter);
  oidnReleaseFilter(_private->albedo_prefilter);
  oidnReleaseFilter(_private->filter);
  oidnReleaseDevice(_private->device);
}

void Denoiser::allocate_buffers(float3* albedo, float3* normal, const uint2& size) {
  TimeMeasure tm;
  uint64_t data_size = sizeof(float3) * size.x * size.y;

  if (_private->data_size != data_size) {
    _private->release_buffers();
  }

  _private->size = size;
  _private->data_size = data_size;

  _private->albedo_buffer = oidnNewSharedBuffer(_private->device, albedo, data_size);
  oidnSetFilterImage(_private->albedo_prefilter, "albedo", _private->albedo_buffer, OIDN_FORMAT_FLOAT3, size.x, size.y, 0, sizeof(float3), sizeof(float3) * size.x);
  oidnSetFilterImage(_private->albedo_prefilter, "output", _private->albedo_buffer, OIDN_FORMAT_FLOAT3, size.x, size.y, 0, sizeof(float3), sizeof(float3) * size.x);
  oidnCommitFilter(_private->albedo_prefilter);

  _private->normal_buffer = oidnNewSharedBuffer(_private->device, normal, data_size);
  oidnSetFilterImage(_private->normal_prefilter, "normal", _private->normal_buffer, OIDN_FORMAT_FLOAT3, size.x, size.y, 0, sizeof(float3), sizeof(float3) * size.x);
  oidnSetFilterImage(_private->normal_prefilter, "output", _private->normal_buffer, OIDN_FORMAT_FLOAT3, size.x, size.y, 0, sizeof(float3), sizeof(float3) * size.x);
  oidnCommitFilter(_private->normal_prefilter);

  log::info("Denoiser: allocate_buffers - %.3fms", tm.measure_ms());
}

void Denoiser::denoise(float4* input, float3* output) {
  if (_private->color_buffer_in) {
    oidnReleaseBuffer(_private->color_buffer_in);
    _private->color_buffer_in = {};
  }

  if (_private->color_buffer_out) {
    oidnReleaseBuffer(_private->color_buffer_out);
    _private->color_buffer_out = {};
  }

  _private->color_buffer_in = oidnNewSharedBuffer(_private->device, input, sizeof(float4) * _private->size.x * _private->size.y);
  _private->color_buffer_out = oidnNewSharedBuffer(_private->device, output, _private->data_size);
  oidnSetFilterImage(_private->filter, "color", _private->color_buffer_in, OIDN_FORMAT_FLOAT3, _private->size.x, _private->size.y, 0, sizeof(float4),
    sizeof(float4) * _private->size.x);
  oidnSetFilterImage(_private->filter, "normal", _private->normal_buffer, OIDN_FORMAT_FLOAT3, _private->size.x, _private->size.y, 0, sizeof(float3),
    sizeof(float3) * _private->size.x);
  oidnSetFilterImage(_private->filter, "albedo", _private->albedo_buffer, OIDN_FORMAT_FLOAT3, _private->size.x, _private->size.y, 0, sizeof(float3),
    sizeof(float3) * _private->size.x);
  oidnSetFilterImage(_private->filter, "output", _private->color_buffer_out, OIDN_FORMAT_FLOAT3, _private->size.x, _private->size.y, 0, sizeof(float3),
    sizeof(float3) * _private->size.x);
  oidnCommitFilter(_private->filter);

  if (_private->check_errors()) {
    TimeMeasure tm;
    oidnExecuteFilter(_private->albedo_prefilter);
    oidnExecuteFilter(_private->normal_prefilter);
    oidnExecuteFilter(_private->filter);
    log::info("Denoiser: denoise - %.3fms", tm.measure_ms());
  }
}

}  // namespace etx
