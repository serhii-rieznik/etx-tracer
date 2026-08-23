#include "gpu_rt_wavefront_common.hlsl"

[numthreads(64, 1, 1)] void wavefront_camera_connect_light_clear_main(uint3 dtid : SV_DispatchThreadID) {
  const uint dispatch_index = dtid.x;
  const uint batch_index = dtid.y;
  if (batch_index != 0u) {
    return;
  }

  GPUWavefrontResources resources = wavefront_load_resources();
  if ((dispatch_index == 0u) && (batch_index == 0u)) {
    wavefront_work_queue_reset(resources.shadow_queue_buffer, wavefront_shadow_queue_base_offset(resources, kGPUWavefrontShadowQueueConnectLight));
  }
  if (resources.connect_light_task_buffer == kInvalidIndex) {
    return;
  }

  const uint queue_descriptor = wavefront_queue_current_descriptor(true);
  const uint queue_count = wavefront_queue_count(queue_descriptor);
  if (dispatch_index >= queue_count) {
    return;
  }

  const uint path_index = wavefront_queue_load(queue_descriptor, dispatch_index);
  GPUWavefrontPathMeta meta = wavefront_load_path_meta(resources.path_meta_buffer, path_index);
  uint vertex_index = (constants.dispatch_item_offset != 0u) ? meta.reserved0 : meta.reserved1;
  uint vertex_path_length = min(meta.light_path_length, constants.connect_light_vertex_length);

  for (uint item_index = 0u; item_index < constants.dispatch_item_count; ++item_index) {
    const uint light_vertex_length = constants.connect_light_vertex_length - item_index;
    const uint task_index = item_index * resources.path_capacity + dispatch_index;
    uint light_vertex_index = kInvalidIndex;
    uint previous_light_vertex_index = kInvalidIndex;

    if (resources.light_vertex_counter_buffer == kInvalidIndex) {
      light_vertex_index = wavefront_light_vertex_slot(path_index, light_vertex_length);
      previous_light_vertex_index = wavefront_light_vertex_slot(path_index, light_vertex_length - 1u);
    } else {
      while ((vertex_index != kInvalidIndex) && (vertex_path_length > light_vertex_length)) {
        vertex_index = wavefront_light_previous_vertex_index(resources, vertex_index);
        vertex_path_length -= 1u;
      }
      if ((vertex_index != kInvalidIndex) && (vertex_path_length == light_vertex_length)) {
        const uint previous_vertex_index = wavefront_light_previous_vertex_index(resources, vertex_index);
        light_vertex_index = vertex_index;
        previous_light_vertex_index = previous_vertex_index;
        vertex_index = previous_vertex_index;
        vertex_path_length -= 1u;
      }
    }
    wavefront_initialize_connect_light_candidate(resources.connect_light_task_buffer, task_index, light_vertex_index, previous_light_vertex_index);
  }

  if (resources.light_vertex_counter_buffer != kInvalidIndex) {
    meta.reserved1 = vertex_index;
    wavefront_store_path_meta(resources.path_meta_buffer, path_index, meta);
  }
}
