#pragma once

#include <etx/core/pimpl.hxx>
#include <etx/render/shared/base.hxx>

namespace etx {

struct Denoiser {
  Denoiser();
  ~Denoiser();

  void init();
  void shutdown();

  void allocate_buffers(float4* input, float4* albedo, float4* normal, float4* output, const uint2 size);
  void denoise();

  ETX_DECLARE_PIMPL(Denoiser, 128);
};

}  // namespace etx
