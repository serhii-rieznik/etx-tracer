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

