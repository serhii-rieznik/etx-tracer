#include "gpu_rt_wavefront_common.hlsl"

[numthreads(64, 1, 1)] void wavefront_camera_connect_light_accumulate_main(uint3 dtid : SV_DispatchThreadID) {
  const uint dispatch_index = dtid.x;

  GPUWavefrontResources resources = wavefront_load_resources();
  if ((resources.connect_light_task_buffer == kInvalidIndex) || (resources.connect_light_result_buffer == kInvalidIndex)) {
    return;
  }

  GPUWavefrontConnectLightTask task = wavefront_load_connect_light_task(resources.connect_light_task_buffer, dispatch_index);
  GPUWavefrontConnectLightResult result_value = wavefront_load_connect_light_result(resources.connect_light_result_buffer, dispatch_index);
  if ((task.flags != GPUWavefrontConnectLightTaskFlags::Ready) || (result_value.visible == 0u)) {
    return;
  }

  SpectralResponse value = spectral_response_mul(task.contribution, result_value.transmittance);
  SpectralQuery spect = (SpectralQuery)0;
  spect.wavelength = value.wavelength;
  spect.flags = value.flags;
  wavefront_film_add(task.pixel_index, spectral_response_to_rgb(value) * wavefront_spectral_weight(spect));
}
