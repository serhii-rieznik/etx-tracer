#pragma once

#include "interop.hxx"

struct SpectralImage {
  uint32_t spectrum_index ETX_INIT(kInvalidIndex);
  uint32_t image_index ETX_INIT(kInvalidIndex);
};

struct SampledImage {
  float4 value ETX_INIT({});
  uint32_t image_index ETX_INIT(kInvalidIndex);
  uint32_t channel ETX_INIT(kInvalidIndex);
};
