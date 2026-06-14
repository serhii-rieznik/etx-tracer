#include "gpu_rt_wavefront_common.hlsl"

[numthreads(64, 1, 1)] void wavefront_camera_connect_light_clear_main(uint3 dtid : SV_DispatchThreadID) {
  const uint dispatch_index = dtid.x;
  GPUWavefrontResources resources = wavefront_load_resources();
  if (resources.connect_light_task_buffer == kInvalidIndex) {
    return;
  }

  const uint vertex_stride = wavefront_light_fixed_max_bounces(resources) + 1u;
  const uint task_count = resources.path_capacity * vertex_stride;
  if (dispatch_index >= task_count) {
    return;
  }

  GPUWavefrontConnectLightTask empty_task = (GPUWavefrontConnectLightTask)0;
  wavefront_store_connect_light_task(resources.connect_light_task_buffer, dispatch_index, empty_task);
}
