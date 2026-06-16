#include "gpu_rt_wavefront_common.hlsl"

[numthreads(64, 1, 1)] void wavefront_camera_connect_light_clear_main(uint3 dtid : SV_DispatchThreadID) {
  const uint dispatch_index = dtid.x;
  const uint batch_index = dtid.y;
  if ((constants.dispatch_item_count != 0u) && (batch_index >= constants.dispatch_item_count)) {
    return;
  }

  GPUWavefrontResources resources = wavefront_load_resources();
  if (resources.connect_light_task_buffer == kInvalidIndex) {
    return;
  }

  if (dispatch_index >= resources.path_capacity) {
    return;
  }

  const uint task_index = batch_index * resources.path_capacity + dispatch_index;
  GPUWavefrontConnectLightTask empty_task = (GPUWavefrontConnectLightTask)0;
  wavefront_store_connect_light_task(resources.connect_light_task_buffer, task_index, empty_task);
}
