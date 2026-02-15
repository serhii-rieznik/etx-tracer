#pragma once

#include "gpu_scene_shared.hxx"

struct ETX_ALIGNED GPURTConstants {
  uint32_t camera_buffer_index;
  uint32_t as_index;
  uint32_t output_image_index;
  uint32_t frame_index;
  uint32_t sample_index;
  uint32_t blue_noise_buffer_index;
  uint32_t pad1;
  uint32_t pad2;
  GPUScene scene;
};

#if defined(__cplusplus)
static_assert(std::is_standard_layout_v<GPURTConstants>, "GPURTConstants must stay standard layout for C++/HLSL interop");
static_assert(alignof(GPURTConstants) == 16, "GPURTConstants alignment must match HLSL packing");
static_assert(sizeof(GPURTConstants) == 96, "GPURTConstants size changed; update shared ABI or padding");
static_assert(offsetof(GPURTConstants, as_index) == 4, "GPURTConstants::as_index offset changed");
static_assert(offsetof(GPURTConstants, blue_noise_buffer_index) == 20, "GPURTConstants::blue_noise_buffer_index offset changed");
static_assert(offsetof(GPURTConstants, scene) == 32, "GPURTConstants::scene offset changed");
#endif
