#include "gpu_rt_wavefront_trace_common.hlsl"

[numthreads(64, 1, 1)] void wavefront_camera_direct_light_shadow_main(uint3 dtid : SV_DispatchThreadID) {
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
#if ETX_UPBP
  if (scene_path_mode_is_upbp()) {
    uint medium_seed = task.upbp_auxiliary0_bits;
    bool visible = false;
    GPUUPBPConnectionInterval connection = (GPUUPBPConnectionInterval)0;
    uint tracking_failure = GPUUPBPConnectionTrackingFailure::None;
    const bool source_is_medium = (task.flags & GPUWavefrontPointConnectionTaskFlags::SourceMedium) != 0u;
    const bool tracking_valid = wavefront_upbp_trace_connection_to_point(task.shadow_ray.o, task.shadow_target, spect, task.medium_index, task.inline_medium_extinction,
      task.inline_medium_flags, source_is_medium, seed, medium_seed, visible, connection, tracking_failure);
    if (tracking_valid == false) {
      const GPUUPBPResources upbp_resources = upbp_load_resources(resources);
      const GPUUPBPPathState path_state = upbp_load_path_state(upbp_resources.path_state_buffer, upbp_path_state_index(upbp_resources, true, task.path_index));
      upbp_mark_failed_connection(upbp_resources, path_state.global_path_index, 1u, path_state.path_length + 1u, 1u, tracking_failure);
    }
    result_value.transmittance = connection.weight;
    result_value.visible = tracking_valid && visible ? 1u : 0u;
    result_value.upbp_log_transport_pdf_forward_bits = asuint(connection.log_transport_pdf_forward);
    result_value.upbp_log_transport_pdf_reverse_bits = asuint(connection.log_transport_pdf_reverse);
    result_value.upbp_tracking_valid = tracking_valid ? 1u : 0u;
  } else
#endif
  {
    result_value.visible = wavefront_trace_transmittance_to_point(task.shadow_ray.o, task.shadow_target, spect, task.medium_index, seed, result_value.transmittance) ? 1u : 0u;
  }
  if (scene_path_mode_is_upbp() == false) {
    GPUWavefrontPathState state = wavefront_load_path_state(resources.camera_state_buffer, task.path_index);
    if (wavefront_path_state_valid(state)) {
      state.sampler_seed = seed;
      wavefront_store_path_state(resources.camera_state_buffer, task.path_index, state);
    }
  }
  wavefront_store_direct_light_result(resources.direct_light_result_buffer, dispatch_index, result_value);
}

  [numthreads(64, 1, 1)] void wavefront_camera_connect_light_shadow_main(uint3 dtid : SV_DispatchThreadID) {
  const uint queue_index = dtid.x;

  GPUWavefrontResources resources = wavefront_load_resources();
  if (resources.connect_light_task_buffer == kInvalidIndex) {
    return;
  }

  const uint queue_count = wavefront_shadow_queue_dispatch_count(resources, kGPUWavefrontShadowQueueConnectLight);
  if (queue_index >= queue_count) {
    return;
  }
  const uint task_index = wavefront_shadow_queue_load(resources, kGPUWavefrontShadowQueueConnectLight, queue_index);
#if ETX_UPBP
  if (scene_path_mode_is_upbp()) {
    if (wavefront_claim_connect_light_task(resources.connect_light_task_buffer, task_index) == false) {
      return;
    }
    DeviceMemoryBarrier();
  }
#endif
  GPUWavefrontConnectLightTask task = wavefront_load_connect_light_task(resources.connect_light_task_buffer, task_index);

  SpectralQuery spect = (SpectralQuery)0;
  spect.wavelength = task.contribution.wavelength;
  spect.flags = task.contribution.flags;
  SpectralResponse transmittance = spectral_response_make(spect, 1.0f);

  uint seed = task.sampler_seed;
  bool visible = false;
#if ETX_UPBP
  if (scene_path_mode_is_upbp()) {
    GPUUPBPResources upbp_resources = upbp_load_resources(resources);
    const GPUUPBPVertex camera_vertex = upbp_load_vertex(upbp_resources.vertex_buffer, task.upbp_camera_vertex_index);
    const GPUUPBPVertex light_vertex = upbp_load_bpt_light_vertex(upbp_resources, task.upbp_light_vertex_index);
    GPUUPBPConnectionInterval connection = (GPUUPBPConnectionInterval)0;
    uint intersection_seed = task.upbp_intersection_seed;
    uint medium_seed = task.upbp_medium_seed;
    uint tracking_failure = GPUUPBPConnectionTrackingFailure::None;
    const bool source_is_medium = upbp_vertex_is_medium(light_vertex);
    const bool tracking_valid = wavefront_upbp_trace_connection_to_point(task.shadow_origin, task.shadow_target, spect, task.medium_index, task.inline_medium_extinction,
      task.inline_medium_flags, source_is_medium, intersection_seed, medium_seed, visible, connection, tracking_failure);
    if (tracking_valid == false) {
      upbp_mark_failed_connection(upbp_resources, camera_vertex.global_path_index, 3u, camera_vertex.path_length + 1u, light_vertex.path_length + 1u, tracking_failure);
    }
    if (tracking_valid && visible) {
      const float mis_weight = upbp_bpt_connection_cross_technique_weight(upbp_resources.iteration, light_vertex, camera_vertex, asfloat(task.upbp_light_pdf_forward_bits),
        asfloat(task.upbp_light_pdf_reverse_bits), asfloat(task.upbp_camera_pdf_forward_bits), asfloat(task.upbp_camera_pdf_reverse_bits), connection.log_transport_pdf_forward,
        connection.log_transport_pdf_reverse);
      if (mis_weight > 0.0f) {
        const SpectralResponse value = spectral_response_mul(spectral_response_mul(task.contribution, connection.weight), mis_weight);
        wavefront_film_add(task.pixel_index, wavefront_spectral_estimate(value, spect));
      }
    }
    return;
  }
#endif
  if ((task.inline_medium_flags & GPUWavefrontSubsurfaceFlags::InlineMedium) != 0u) {
    visible = wavefront_trace_transmittance_to_point_inline_medium(task.shadow_origin, task.shadow_target, spect, task.medium_index, task.inline_medium_extinction,
      task.inline_medium_flags, seed, transmittance);
  } else {
    visible = wavefront_trace_transmittance_to_point(task.shadow_origin, task.shadow_target, spect, task.medium_index, seed, transmittance);
  }
  if (visible) {
    SpectralResponse value = spectral_response_mul(task.contribution, transmittance);
    wavefront_film_add(task.pixel_index, wavefront_spectral_estimate(value, spect));
  }
}

[numthreads(64, 1, 1)] void wavefront_light_connect_camera_shadow_main(uint3 dtid : SV_DispatchThreadID) {
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
#if ETX_UPBP
  if (scene_path_mode_is_upbp()) {
    uint medium_seed = task.upbp_auxiliary1_bits;
    bool visible = false;
    GPUUPBPConnectionInterval connection = (GPUUPBPConnectionInterval)0;
    uint tracking_failure = GPUUPBPConnectionTrackingFailure::None;
    const bool source_is_medium = (task.flags & GPUWavefrontPointConnectionTaskFlags::SourceMedium) != 0u;
    const bool tracking_valid = wavefront_upbp_trace_connection_to_point(task.shadow_ray.o, task.shadow_target, spect, task.medium_index, task.inline_medium_extinction,
      task.inline_medium_flags, source_is_medium, seed, medium_seed, visible, connection, tracking_failure);
    if (tracking_valid == false) {
      const GPUUPBPResources upbp_resources = upbp_load_resources(resources);
      const GPUUPBPPathState path_state = upbp_load_path_state(upbp_resources.path_state_buffer, upbp_path_state_index(upbp_resources, false, task.path_index));
      upbp_mark_failed_connection(upbp_resources, path_state.global_path_index, 2u, 1u, path_state.path_length + 1u, tracking_failure);
    }
    result_value.transmittance = connection.weight;
    result_value.visible = tracking_valid && visible ? 1u : 0u;
    result_value.upbp_log_transport_pdf_forward_bits = asuint(connection.log_transport_pdf_forward);
    result_value.upbp_log_transport_pdf_reverse_bits = asuint(connection.log_transport_pdf_reverse);
    result_value.upbp_tracking_valid = tracking_valid ? 1u : 0u;
  } else
#endif
  {
    result_value.visible = wavefront_trace_transmittance_to_point(task.shadow_ray.o, task.shadow_target, spect, task.medium_index, seed, result_value.transmittance) ? 1u : 0u;
  }
  if (scene_path_mode_is_upbp() == false) {
    GPUWavefrontPathState state = wavefront_load_path_state(resources.light_state_buffer, task.path_index);
    if (wavefront_path_state_valid(state)) {
      state.sampler_seed = seed;
      wavefront_store_path_state(resources.light_state_buffer, task.path_index, state);
    }
  }
  wavefront_store_connect_camera_result(resources.connect_camera_result_buffer, dispatch_index, result_value);
}
