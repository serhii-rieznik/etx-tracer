#pragma once

#include "gpu_scene_shared.hxx"
#include "gpu_wavefront_shared.hxx"

struct ETX_ALIGNED GPURTConstants {
  uint32_t camera_buffer_index;
  uint32_t as_index;
  uint32_t output_image_index;
  uint32_t frame_index;
  uint32_t sample_index;
  uint32_t blue_noise_buffer_index;
  uint32_t wavefront_buffer_index;
  uint32_t path_iteration;
  uint32_t connect_light_vertex_length;
  uint32_t render_window_origin_x;
  uint32_t render_window_origin_y;
  uint32_t render_window_width;
  uint32_t render_window_height;
  uint32_t dispatch_item_offset;
  uint32_t dispatch_item_count;
  uint32_t work_queue_index;
  float vcm_radius;
  float vcm_vm_weight;
  float vcm_vc_weight;
  float vcm_vm_normalization;
  uint32_t vcm_grid_mask;
  uint32_t vcm_kernel;
  uint32_t vcm_light_vertex_count;
  uint32_t pad2;
  GPUScene scene;
};
