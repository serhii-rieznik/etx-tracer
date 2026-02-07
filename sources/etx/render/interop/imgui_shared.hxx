#pragma once

#include "interop.hxx"

struct ETX_ALIGNED ImGuiPushConstants {
  float2 scale;
  float2 translate;
  uint32_t vertex_buffer_index;
  uint32_t texture_index;
  uint32_t sampler_index;
  uint32_t padding;
};

#if defined(__cplusplus)
static_assert(std::is_standard_layout_v<ImGuiPushConstants>, "ImGuiPushConstants must stay standard layout for C++/HLSL interop");
static_assert(alignof(ImGuiPushConstants) == 16, "ImGuiPushConstants alignment must match HLSL packing");
static_assert(sizeof(ImGuiPushConstants) == 32, "ImGuiPushConstants size changed; update shared ABI or padding");
static_assert(offsetof(ImGuiPushConstants, vertex_buffer_index) == 16, "ImGuiPushConstants::vertex_buffer_index offset changed");
#endif
