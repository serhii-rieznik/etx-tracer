#pragma once

#include <etx/core/pimpl.hxx>
#include <etx/render/shared/base.hxx>

namespace etx {

struct Denoiser {
  Denoiser();
  ~Denoiser();

  void init();
  void shutdown();

  void denoise(const float4* image, const float4* albedo, const float4* normal, float4* output, const uint2 size);

  ETX_DECLARE_PIMPL(Denoiser, 512);
};

}  // namespace etx
