#pragma once

#include "camera.hxx"

struct ETX_ALIGNED GPURTConstants {
  Camera camera;
  uint32_t as_index;
  uint32_t output_image_index;
  uint32_t frame_index;
  uint32_t sample_index;
};
