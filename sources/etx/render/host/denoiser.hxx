#pragma once

#include <etx/core/pimpl.hxx>
#include <etx/render/interop/interop.hxx>

namespace etx {

struct Denoiser {
  Denoiser();
  ~Denoiser();

  void init();
  void shutdown();
  void release_buffers();

  void allocate_buffers(float3* albedo, float3* normal, const uint2& size);
  void denoise(float4* input, float3* output);

  ETX_DECLARE_PIMPL(Denoiser, 128);
};

}  // namespace etx
