#include "gpu_rt_wavefront_common.hlsl"

[numthreads(64, 1, 1)] void wavefront_camera_connect_light_accumulate_main(uint3 dtid : SV_DispatchThreadID) {
  const uint queue_index = dtid.x;

  GPUWavefrontResources resources = wavefront_load_resources();
  if ((resources.connect_light_task_buffer == kInvalidIndex) || (resources.connect_light_result_buffer == kInvalidIndex)) {
    return;
  }

  const uint queue_count = wavefront_shadow_queue_count(resources, kGPUWavefrontShadowQueueConnectLight);
  if (queue_index >= queue_count) {
    return;
  }
  const uint task_index = wavefront_shadow_queue_load(resources, kGPUWavefrontShadowQueueConnectLight, queue_index);
  GPUWavefrontConnectLightTask task = wavefront_load_connect_light_task(resources.connect_light_task_buffer, task_index);
  GPUWavefrontConnectLightResult result_value = wavefront_load_connect_light_result(resources.connect_light_result_buffer, task_index);
  if ((task.flags != GPUWavefrontConnectLightTaskFlags::Ready) || (result_value.visible == 0u)) {
    return;
  }

  SpectralResponse value = spectral_response_mul(task.contribution, result_value.transmittance);
  SpectralQuery spect = (SpectralQuery)0;
  spect.wavelength = value.wavelength;
  spect.flags = value.flags;
  wavefront_film_add(task.pixel_index, wavefront_spectral_estimate(value, spect));
}
