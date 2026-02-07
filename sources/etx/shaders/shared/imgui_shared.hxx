#pragma once

#include "types.hxx"

struct ETX_ALIGNED ImGuiPushConstants {
  float2 scale;
  float2 translate;
  uint32_t vertex_buffer_index;
  uint32_t texture_index;
  uint32_t sampler_index;
  uint32_t padding;
};
