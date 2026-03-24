#include "gpu_rt_wavefront_trace_common.hlsl"

[numthreads(64, 1, 1)] void wavefront_camera_direct_light_shadow_main(uint3 dtid : SV_DispatchThreadID) {
  const uint dispatch_index = dtid.x;

  GPUWavefrontResources resources = wavefront_load_resources();
  if ((resources.direct_light_task_buffer == kInvalidIndex) || (resources.direct_light_result_buffer == kInvalidIndex)) {
    return;
  }

  GPUWavefrontDirectLightTask task = wavefront_load_direct_light_task(resources.direct_light_task_buffer, dispatch_index);
  GPUWavefrontDirectLightResult result_value = (GPUWavefrontDirectLightResult)0;
  result_value.transmittance = spectral_response_make(spectral_query_sample(), 0.0f);
  if (task.flags == 0u) {
    wavefront_store_direct_light_result(resources.direct_light_result_buffer, dispatch_index, result_value);
    return;
  }

  SpectralQuery spect = (SpectralQuery)0;
  spect.wavelength = task.contribution.wavelength;
  spect.flags = task.contribution.flags;
  result_value.transmittance = spectral_response_make(spect, 1.0f);

  uint seed = task.sampler_seed;
  result_value.visible = wavefront_trace_transmittance_to_point(task.shadow_ray.o, task.shadow_target, spect, task.medium_index, seed, result_value.transmittance) ? 1u : 0u;
  GPUWavefrontPathState state = wavefront_load_path_state(resources.camera_state_buffer, task.path_index);
  if (wavefront_path_state_valid(state)) {
    state.sampler_seed = seed;
    wavefront_store_path_state(resources.camera_state_buffer, task.path_index, state);
  }
  wavefront_store_direct_light_result(resources.direct_light_result_buffer, dispatch_index, result_value);
}

  [numthreads(64, 1, 1)] void wavefront_camera_connect_light_shadow_main(uint3 dtid : SV_DispatchThreadID) {
  const uint dispatch_index = dtid.x;

  GPUWavefrontResources resources = wavefront_load_resources();
  if ((resources.connect_light_task_buffer == kInvalidIndex) || (resources.connect_light_result_buffer == kInvalidIndex)) {
    return;
  }

  GPUWavefrontConnectLightTask task = wavefront_load_connect_light_task(resources.connect_light_task_buffer, dispatch_index);
  GPUWavefrontConnectLightResult result_value = (GPUWavefrontConnectLightResult)0;
  result_value.transmittance = spectral_response_make(spectral_query_sample(), 0.0f);
  if (task.flags == 0u) {
    wavefront_store_connect_light_result(resources.connect_light_result_buffer, dispatch_index, result_value);
    return;
  }

  SpectralQuery spect = (SpectralQuery)0;
  spect.wavelength = task.contribution.wavelength;
  spect.flags = task.contribution.flags;
  result_value.transmittance = spectral_response_make(spect, 1.0f);

  uint seed = task.sampler_seed;
  result_value.visible = wavefront_trace_transmittance_to_point(task.shadow_ray.o, task.shadow_target, spect, task.medium_index, seed, result_value.transmittance) ? 1u : 0u;
  GPUWavefrontPathState state = wavefront_load_path_state(resources.camera_state_buffer, task.path_index);
  if (wavefront_path_state_valid(state)) {
    state.sampler_seed = seed;
    wavefront_store_path_state(resources.camera_state_buffer, task.path_index, state);
  }
  wavefront_store_connect_light_result(resources.connect_light_result_buffer, dispatch_index, result_value);
}

[numthreads(64, 1, 1)] void wavefront_light_connect_camera_shadow_main(uint3 dtid : SV_DispatchThreadID) {
  const uint dispatch_index = dtid.x;

  GPUWavefrontResources resources = wavefront_load_resources();
  if ((resources.connect_camera_task_buffer == kInvalidIndex) || (resources.connect_camera_result_buffer == kInvalidIndex)) {
    return;
  }

  GPUWavefrontConnectCameraTask task = wavefront_load_connect_camera_task(resources.connect_camera_task_buffer, dispatch_index);
  GPUWavefrontConnectCameraResult result_value = (GPUWavefrontConnectCameraResult)0;
  result_value.transmittance = spectral_response_make(spectral_query_sample(), 0.0f);
  if (task.flags == 0u) {
    wavefront_store_connect_camera_result(resources.connect_camera_result_buffer, dispatch_index, result_value);
    return;
  }

  SpectralQuery spect = (SpectralQuery)0;
  spect.wavelength = task.contribution.wavelength;
  spect.flags = task.contribution.flags;
  result_value.transmittance = spectral_response_make(spect, 1.0f);

  uint seed = task.sampler_seed;
  result_value.visible = wavefront_trace_transmittance_to_point(task.shadow_ray.o, task.shadow_target, spect, task.medium_index, seed, result_value.transmittance) ? 1u : 0u;
  GPUWavefrontPathState state = wavefront_load_path_state(resources.light_state_buffer, task.path_index);
  if (wavefront_path_state_valid(state)) {
    state.sampler_seed = seed;
    wavefront_store_path_state(resources.light_state_buffer, task.path_index, state);
  }
  wavefront_store_connect_camera_result(resources.connect_camera_result_buffer, dispatch_index, result_value);
}
