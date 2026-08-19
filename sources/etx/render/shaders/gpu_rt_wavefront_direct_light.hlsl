#include "gpu_rt_wavefront_common.hlsl"

[numthreads(64, 1, 1)] void wavefront_camera_direct_light_accumulate_main(uint3 dtid : SV_DispatchThreadID) {
  const uint queue_index = dtid.x;

  GPUWavefrontResources resources = wavefront_load_resources();
  if ((resources.direct_light_task_buffer == kInvalidIndex) || (resources.direct_light_result_buffer == kInvalidIndex)) {
    return;
  }

  const uint queue_count = wavefront_shadow_queue_count(resources, kGPUWavefrontShadowQueueDirectLight);
  if (queue_index >= queue_count) {
    return;
  }
  const uint dispatch_index = wavefront_shadow_queue_load(resources, kGPUWavefrontShadowQueueDirectLight, queue_index);

  GPUWavefrontDirectLightTask task = wavefront_load_direct_light_task(resources.direct_light_task_buffer, dispatch_index);
  GPUWavefrontDirectLightResult result_value = wavefront_load_direct_light_result(resources.direct_light_result_buffer, dispatch_index);
  if ((task.flags == 0u) || (result_value.visible == 0u)) {
    return;
  }

  SpectralResponse value = spectral_response_mul(task.contribution, result_value.transmittance);
  SpectralQuery spect = (SpectralQuery)0;
  spect.wavelength = value.wavelength;
  spect.flags = value.flags;
  wavefront_film_add(task.pixel_index, spectral_response_to_rgb(value) * wavefront_spectral_weight(spect));
}
