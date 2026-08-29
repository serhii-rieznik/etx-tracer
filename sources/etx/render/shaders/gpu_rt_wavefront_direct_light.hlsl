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
    GPUUPBPPathState path_state = upbp_load_path_state(upbp_resources.path_state_buffer, upbp_path_state_index(upbp_resources, true, task.path_index));
    if (((path_state.flags & GPUUPBPPathStateFlags::Valid) == 0u) || (path_state.last_vertex_index == kInvalidIndex)) {
      return;
    }
    GPUUPBPVertex camera_vertex = upbp_load_vertex(upbp_resources.vertex_buffer, path_state.last_vertex_index);
    mis_weight = upbp_bpt_nee_cross_technique_weight(upbp_resources.iteration, camera_vertex, task.shadow_ray.d, task.mis_weight, asfloat(task.upbp_auxiliary1_bits),
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
