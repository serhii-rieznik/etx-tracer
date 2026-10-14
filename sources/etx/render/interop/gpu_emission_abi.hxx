#pragma once

#include "interop.hxx"

ETX_STATIC_CONST uint32_t kGPUEmissionRequestHeaderSize = 16u;
ETX_STATIC_CONST uint32_t kGPUEmissionRequestPendingOffset = 0u;
ETX_STATIC_CONST uint32_t kGPUEmissionRequestCapacityOffset = 4u;
ETX_STATIC_CONST uint32_t kGPUEmissionRequestStride = 768u;
ETX_STATIC_CONST uint32_t kGPUEmissionRequestCompactStride = 416u;
ETX_STATIC_CONST uint32_t kGPUEmissionRequestResultOffset = 0u;
ETX_STATIC_CONST uint32_t kGPUEmissionRequestSamplerOffset = 24u;
ETX_STATIC_CONST uint32_t kGPUEmissionRequestMaterialOffset = 40u;
ETX_STATIC_CONST uint32_t kGPUEmissionRequestSpectrumOffset = 44u;
ETX_STATIC_CONST uint32_t kGPUEmissionRequestUVOffset = 52u;
ETX_STATIC_CONST uint32_t kGPUEmissionRequestGeometricCosineOffset = 60u;
ETX_STATIC_CONST uint32_t kGPUEmissionRequestOutgoingOffset = 64u;
ETX_STATIC_CONST uint32_t kGPUEmissionRequestActiveOffset = 76u;
ETX_STATIC_CONST uint32_t kGPUEmissionRequestOperationOffset = 80u;
ETX_STATIC_CONST uint32_t kGPUEmissionRequestCoatingOffset = 96u;
ETX_STATIC_CONST uint32_t kGPUEmissionRequestChildOffset = 248u;
ETX_STATIC_CONST uint32_t kGPUEmissionCoatingOperationOffset = 144u;
ETX_STATIC_CONST uint32_t kGPUEmissionVelvetOperationOffset = 152u;

#if ETX_CPP
static_assert((kGPUEmissionRequestCoatingOffset + 152u) <= kGPUEmissionRequestChildOffset);
static_assert((kGPUEmissionRequestChildOffset + 156u) <= kGPUEmissionRequestCompactStride);
static_assert((kGPUEmissionRequestChildOffset + 396u) <= kGPUEmissionRequestStride);
#endif

struct GPUEmissionRequestOperation {
  enum : uint32_t { Classify, Simple, FiniteConductor, Begin, Coating, Velvet };
};
