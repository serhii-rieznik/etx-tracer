#pragma once

#include "interop.hxx"

ETX_STATIC_CONST uint32_t kGPUCoatingRequestHeaderSize = 16u;
ETX_STATIC_CONST uint32_t kGPUCoatingRequestPendingOffset = 0u;
ETX_STATIC_CONST uint32_t kGPUCoatingRequestCapacityOffset = 4u;
ETX_STATIC_CONST uint32_t kGPUCoatingRequestStride = 944u;
ETX_STATIC_CONST uint32_t kGPUCoatingRequestResultOffset = 0u;
ETX_STATIC_CONST uint32_t kGPUCoatingRequestSamplerOffset = 64u;
ETX_STATIC_CONST uint32_t kGPUCoatingRequestMaterialOffset = 80u;
ETX_STATIC_CONST uint32_t kGPUCoatingRequestDataOffset = 84u;
ETX_STATIC_CONST uint32_t kGPUCoatingRequestOutgoingOffset = 168u;
ETX_STATIC_CONST uint32_t kGPUCoatingRequestActiveOffset = 180u;
ETX_STATIC_CONST uint32_t kGPUCoatingRequestKindOffset = 184u;
ETX_STATIC_CONST uint32_t kGPUCoatingRequestConsumerSeedOffset = 188u;
ETX_STATIC_CONST uint32_t kGPUCoatingRequestWalkOffset = 192u;
ETX_STATIC_CONST uint32_t kGPUCoatingWalkStride = 752u;
ETX_STATIC_CONST uint32_t kGPUCoatingWalkReverseExitOffset = 396u;
ETX_STATIC_CONST uint32_t kGPUCoatingWalkResultOffset = 448u;
ETX_STATIC_CONST uint32_t kGPUCoatingWalkThroughputOffset = 472u;
ETX_STATIC_CONST uint32_t kGPUCoatingWalkExitFunctionOffset = 496u;
ETX_STATIC_CONST uint32_t kGPUCoatingWalkOutgoingOffset = 520u;
ETX_STATIC_CONST uint32_t kGPUCoatingWalkDirectionOffset = 532u;
ETX_STATIC_CONST uint32_t kGPUCoatingWalkHeightOffset = 544u;
ETX_STATIC_CONST uint32_t kGPUCoatingWalkContinuationPdfOffset = 548u;
ETX_STATIC_CONST uint32_t kGPUCoatingWalkDepthOffset = 552u;
ETX_STATIC_CONST uint32_t kGPUCoatingWalkOperationOffset = 556u;
ETX_STATIC_CONST uint32_t kGPUCoatingWalkSampleTargetOffset = 560u;
ETX_STATIC_CONST uint32_t kGPUCoatingWalkExitFromPhaseOffset = 564u;

ETX_STATIC_CONST uint32_t kGPUCoatingWalkVelvetOffset = 568u;
ETX_STATIC_CONST uint32_t kGPUCoatingWalkVelvetOutgoingOffset = 724u;
ETX_STATIC_CONST uint32_t kGPUCoatingWalkContinuationDeltaOffset = 736u;

#if ETX_CPP == 1
static_assert((kGPUCoatingRequestConsumerSeedOffset + sizeof(uint32_t)) == kGPUCoatingRequestWalkOffset, "Coating consumer seed must fit before its walk");
static_assert((kGPUCoatingRequestWalkOffset + kGPUCoatingWalkStride) == kGPUCoatingRequestStride, "Coating request walk must fit its stride");
static_assert((kGPUCoatingWalkVelvetOutgoingOffset + 12u) == kGPUCoatingWalkContinuationDeltaOffset, "Coating continuation flag must follow the existing child direction");
static_assert((kGPUCoatingWalkContinuationDeltaOffset + sizeof(uint32_t)) <= kGPUCoatingWalkStride, "Coating continuation flag must fit its walk stride");
#endif

struct GPUCoatingRequestKind {
  enum : uint32_t {
    Sample,
    Evaluation,
  };
};
