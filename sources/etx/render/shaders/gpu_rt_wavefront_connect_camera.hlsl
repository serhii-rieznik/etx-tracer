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

  float mis_weight = 1.0f;
#if ETX_UPBP
  if (scene_path_mode_is_upbp()) {
    if ((result_value.upbp_tracking_valid == 0u) || (result_value.visible == 0u)) {
      return;
    }
    GPUUPBPResources upbp_resources = upbp_load_resources(resources);
    GPUUPBPPathState path_state = upbp_load_path_state(upbp_resources.path_state_buffer, upbp_path_state_index(upbp_resources, false, task.path_index));
    if (((path_state.flags & GPUUPBPPathStateFlags::Valid) == 0u) || (path_state.last_vertex_index == kInvalidIndex)) {
      return;
    }
    GPUUPBPVertex light_vertex = upbp_load_vertex(upbp_resources.vertex_buffer, path_state.last_vertex_index);
    mis_weight = upbp_bpt_light_tracing_cross_technique_weight(upbp_resources.iteration, light_vertex, task.shadow_ray.d, asfloat(task.upbp_camera_area_density_bits),
      asfloat(task.upbp_scattering_pdf_reverse_bits), asfloat(result_value.upbp_log_transport_pdf_reverse_bits));
    if (mis_weight <= 0.0f) {
      return;
    }
  } else
#endif
    if (result_value.visible == 0u) {
    return;
  }

  SpectralResponse value = spectral_response_mul(spectral_response_mul(task.contribution, result_value.transmittance), mis_weight);
  SpectralQuery spect = (SpectralQuery)0;
  spect.wavelength = value.wavelength;
  spect.flags = value.flags;
  wavefront_film_add(task.pixel_index, wavefront_spectral_estimate(value, spect));
}
