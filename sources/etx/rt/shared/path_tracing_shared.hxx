#pragma once

#include <etx/render/shared/base.hxx>
#include <etx/render/shared/scene.hxx>

namespace etx {

struct ETX_ALIGNED PathTracingGPUData {
  float4* output ETX_EMPTY_INIT;
  uint64_t acceleration_structure ETX_EMPTY_INIT;
};

}  // namespace etx
