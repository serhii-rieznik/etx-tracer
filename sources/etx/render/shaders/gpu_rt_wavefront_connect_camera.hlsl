#include "gpu_rt_wavefront_common.hlsl"

[numthreads(64, 1, 1)] void wavefront_light_connect_camera_prepare_main(uint3 dtid : SV_DispatchThreadID) {
  const uint dispatch_index = dtid.x;
  GPUWavefrontResources resources = wavefront_load_resources();
  if (resources.connect_camera_task_buffer == kInvalidIndex) {
    return;
  }

  uint queue_descriptor = wavefront_queue_current_descriptor(false);
  uint queue_count = wavefront_queue_count(queue_descriptor);
  if (dispatch_index >= queue_count) {
    return;
  }

  GPUWavefrontConnectCameraTask empty_task = (GPUWavefrontConnectCameraTask)0;
  empty_task.medium_index = kInvalidIndex;
  wavefront_store_connect_camera_task(resources.connect_camera_task_buffer, dispatch_index, empty_task);
}

  [numthreads(64, 1, 1)] void wavefront_light_connect_camera_accumulate_main(uint3 dtid : SV_DispatchThreadID) {
  const uint queue_index = dtid.x;

  GPUWavefrontResources resources = wavefront_load_resources();
  if ((resources.connect_camera_task_buffer == kInvalidIndex) || (resources.connect_camera_result_buffer == kInvalidIndex)) {
    return;
  }

  const uint queue_count = wavefront_shadow_queue_count(resources, kGPUWavefrontShadowQueueConnectCamera);
  if (queue_index >= queue_count) {
    return;
  }
  const uint dispatch_index = wavefront_shadow_queue_load(resources, kGPUWavefrontShadowQueueConnectCamera, queue_index);

  GPUWavefrontConnectCameraTask task = wavefront_load_connect_camera_task(resources.connect_camera_task_buffer, dispatch_index);
  GPUWavefrontConnectCameraResult result_value = wavefront_load_connect_camera_result(resources.connect_camera_result_buffer, dispatch_index);
  if (task.flags == 0u) {
    return;
  }

  if (result_value.visible == 0u) {
    return;
  }

  SpectralResponse value = spectral_response_mul(task.contribution, result_value.transmittance);
  SpectralQuery spect = (SpectralQuery)0;
  spect.wavelength = value.wavelength;
  spect.flags = value.flags;
  wavefront_film_add(task.pixel_index, spectral_response_to_rgb(value) * wavefront_spectral_weight(spect));
}
