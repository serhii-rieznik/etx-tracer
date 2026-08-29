#include "gpu_rt_wavefront_common.hlsl"
#include "gpu_rt_wavefront_emitter_sample.hlsl"

float wavefront_direct_light_ris_candidate_weight(WavefrontEmitterSample sample_value, float3 source_position, bool source_is_surface, float3 source_normal) {
  float radiance_weight = spectral_response_to_xyz(sample_value.value).y;
  if (radiance_weight <= 0.0f) {
    return 0.0f;
  }

  float3 to_emitter = sample_value.origin - source_position;
  float len_sq = dot(to_emitter, to_emitter);
  float source_alignment = 1.0f;
  if (source_is_surface && (len_sq > kEpsilon)) {
    source_alignment = abs(dot(source_normal, to_emitter) / sqrt(len_sq));
  }

  if (sample_value.is_distant != 0u) {
    return radiance_weight * source_alignment;
  }

  float emitter_orientation = dot(sample_value.normal, -to_emitter);
  if ((emitter_orientation <= 0.0f) || (len_sq <= kEpsilon)) {
    return 0.0f;
  }

  float distance_weight = 1.0f / max(1.0f, len_sq);
  return radiance_weight * distance_weight * (emitter_orientation / sqrt(len_sq)) * source_alignment;
}

bool wavefront_sample_direct_light_ris(uint light_sampling_mode, SpectralQuery spect, float3 source_position, bool source_is_surface, float3 source_normal, inout uint seed,
  out WavefrontEmitterSample sample_value) {
  sample_value = (WavefrontEmitterSample)0;

  ByteAddressBuffer scene_globals = bindless_buffers[NonUniformResourceIndex(constants.scene.scene_globals)];
  SceneGPUSharedGlobals globals_data = scene_gpu_load_globals(scene_globals);
  if (globals_data.emitter_instance_count == 0u) {
    return false;
  }

  uint candidate_count = min(4u * globals_data.emitter_instance_count, 16u);
  if (candidate_count == 0u) {
    return false;
  }

  float weight_sum = 0.0f;
  float selected_weight = 0.0f;
  WavefrontEmitterSample selected_sample = (WavefrontEmitterSample)0;
  for (uint i = 0u; i < candidate_count; ++i) {
    uint emitter_index = kInvalidIndex;
    float pdf_sample = 0.0f;
    if (wavefront_sample_emitter_index(light_sampling_mode, seed, emitter_index, pdf_sample) == false) {
      continue;
    }

    WavefrontEmitterSample candidate = (WavefrontEmitterSample)0;
    if (wavefront_sample_emitter_to_point_from_index(emitter_index, pdf_sample, spect, source_position, seed, candidate) == false) {
      continue;
    }

    float candidate_weight = wavefront_direct_light_ris_candidate_weight(candidate, source_position, source_is_surface, source_normal);
    float weight = (pdf_sample > 0.0f) ? (candidate_weight / pdf_sample) : 0.0f;
    weight_sum += weight;
    float reservoir_rnd = rnd01(seed) * weight_sum;
    if ((weight > 0.0f) && (reservoir_rnd < weight)) {
      selected_sample = candidate;
      selected_weight = weight;
    }
  }

  if (selected_weight <= 0.0f) {
    return false;
  }

  float reservoir_scale = weight_sum / (float(candidate_count) * selected_weight);
  selected_sample.value = spectral_response_mul(selected_sample.value, reservoir_scale);
  sample_value = selected_sample;
  return true;
}

float wavefront_medium_direct_light_weight(GPUWavefrontPathMeta path_meta, GPUWavefrontPathVertex current_vertex, WavefrontEmitterSample emitter_sample, MediumAccess medium_access,
  float phase_value) {
  if (scene_multiple_importance_sampling_enabled() == false) {
    return 1.0f;
  }

  float sampling_pdf = emitter_sample.pdf_dir * emitter_sample.pdf_sample;
  if (sampling_pdf <= 0.0f) {
    return 0.0f;
  }

  if (scene_path_mode_is_path_tracing()) {
    float direct_pdf = (emitter_sample.is_delta != 0u) ? 0.0f : phase_value;
    return power_heuristic(sampling_pdf, direct_pdf);
  }

  float reverse_phase_pdf = gpu_medium_phase_function(medium_access, -emitter_sample.direction, current_vertex.w_i);
  float w_light = (emitter_sample.is_delta != 0u) ? 0.0f : wavefront_safe_div(phase_value, sampling_pdf);
  float emitter_cosine = abs(dot(emitter_sample.direction, emitter_sample.normal));
  float density_ratio = wavefront_safe_div(emitter_sample.pdf_dir * emitter_cosine, emitter_sample.pdf_dir_out);
  float adjacent_connection = current_vertex.forward_pdf;
  if (scene_path_mode_uses_bdpt_fast() && (path_meta.camera_path_length != 1u)) {
    adjacent_connection = 0.0f;
  }
  float w_camera = wavefront_safe_div(adjacent_connection + current_vertex.reverse_pdf * reverse_phase_pdf, density_ratio);
  return 1.0f / (1.0f + w_light + w_camera);
}

[numthreads(64, 1, 1)] void wavefront_camera_direct_light_sample_main(uint3 dtid : SV_DispatchThreadID) {
  const uint dispatch_index = dtid.x;
  const bool enable_camera_direct_light = true;
  if (scene_path_mode_is_light_tracing()) {
    return;
  }

  GPUWavefrontResources resources = wavefront_load_resources();
  if (resources.direct_light_sample_buffer == kInvalidIndex) {
    return;
  }
  if (dispatch_index >= resources.path_capacity) {
    return;
  }

  GPUWavefrontDirectLightSample empty_sample = (GPUWavefrontDirectLightSample)0;
  wavefront_store_direct_light_sample(resources.direct_light_sample_buffer, dispatch_index, empty_sample);
  if (resources.direct_light_task_buffer != kInvalidIndex) {
    GPUWavefrontDirectLightTask empty_task = (GPUWavefrontDirectLightTask)0;
    empty_task.medium_index = kInvalidIndex;
    wavefront_store_direct_light_task(resources.direct_light_task_buffer, dispatch_index, empty_task);
  }
  if (resources.direct_light_result_buffer != kInvalidIndex) {
    GPUWavefrontDirectLightResult empty_result = (GPUWavefrontDirectLightResult)0;
    wavefront_store_direct_light_result(resources.direct_light_result_buffer, dispatch_index, empty_result);
  }
  if (enable_camera_direct_light == false) {
    return;
  }

  uint queue_descriptor = wavefront_queue_current_descriptor(true);
  uint queue_count = wavefront_queue_count(queue_descriptor);
  if (dispatch_index >= queue_count) {
    return;
  }

  uint path_index = wavefront_queue_load(queue_descriptor, dispatch_index);
  GPUWavefrontPathState state = wavefront_load_path_state(resources.camera_state_buffer, path_index);
  GPUWavefrontHit hit = wavefront_load_hit(resources.camera_hit_buffer, path_index);
  GPUWavefrontPathMeta meta = wavefront_load_path_meta(resources.path_meta_buffer, path_index);
  if ((wavefront_hit_valid(hit) == false) || wavefront_hit_is_miss(hit) || (meta.camera_path_length == 0u)) {
    return;
  }

  GPUWavefrontPathVertex current_vertex = wavefront_load_path_vertex(resources.camera_vertex_buffer, wavefront_camera_vertex_slot(path_index, meta.camera_path_length));
  const bool subsurface_medium_vertex = (wavefront_path_vertex_is_subsurface(current_vertex)) && (wavefront_path_vertex_is_medium(current_vertex));
  const bool medium_direct_connection_disabled =
    wavefront_path_vertex_is_medium(current_vertex) && (wavefront_medium_explicit_connections_enabled(current_vertex.medium_index) == false);
  if ((wavefront_path_vertex_valid(current_vertex) == false) || (wavefront_path_vertex_connectible(current_vertex) == false) || subsurface_medium_vertex ||
      medium_direct_connection_disabled) {
    return;
  }
  uint connection_length = meta.camera_path_length + 1u;
  if ((scene_strategy_enabled(kSceneStrategyConnectToLight) == false) || (connection_length < load_scene_options_min_path_length()) ||
      (connection_length > load_scene_options_max_path_length())) {
    return;
  }

  uint seed = state.sampler_seed;
#if ETX_UPBP
  GPUUPBPResources upbp_resources = (GPUUPBPResources)0;
  GPUUPBPPathState upbp_path_state = (GPUUPBPPathState)0;
  const bool upbp = scene_path_mode_is_upbp();
  if (upbp) {
    upbp_resources = upbp_load_resources(resources);
    upbp_path_state = upbp_load_path_state(upbp_resources.path_state_buffer, upbp_path_state_index(upbp_resources, true, path_index));
    if (((upbp_path_state.flags & GPUUPBPPathStateFlags::Valid) == 0u) || (upbp_path_state.path_length == 0u)) {
      return;
    }
    seed = upbp_deterministic_seed(upbp_path_state.global_path_index, upbp_path_state.path_length + 1u, 1u, kUPBPRandomDomainEmitterConnection);
  }
#else
  const bool upbp = false;
#endif
  WavefrontEmitterSample emitter_sample = (WavefrontEmitterSample)0;
  uint light_sampling_mode = load_scene_options_light_sampling();
  bool source_is_surface = wavefront_path_vertex_is_surface(current_vertex);
  float3 source_normal = wavefront_path_vertex_is_surface(current_vertex) ? current_vertex.normal : float3(0.0f, 0.0f, 0.0f);
  bool sampled = false;
  if ((light_sampling_mode == kSceneLightSamplingRISFromDistribution) || (light_sampling_mode == kSceneLightSamplingRISUniform)) {
    sampled = wavefront_sample_direct_light_ris(light_sampling_mode, state.spect, current_vertex.position, source_is_surface, source_normal, seed, emitter_sample);
  } else {
    sampled = wavefront_sample_emitter_to_point(light_sampling_mode, state.spect, current_vertex.position, seed, emitter_sample);
  }
  if (upbp == false) {
    state.sampler_seed = seed;
    wavefront_store_path_state(resources.camera_state_buffer, path_index, state);
  }
  if (sampled == false) {
    return;
  }

  GPUWavefrontDirectLightSample sample_value = (GPUWavefrontDirectLightSample)0;
  sample_value.value = emitter_sample.value;
  sample_value.origin = emitter_sample.origin;
  sample_value.pdf_sample = emitter_sample.pdf_sample;
  sample_value.direction = emitter_sample.direction;
  sample_value.pdf_area = emitter_sample.pdf_area;
  sample_value.normal = emitter_sample.normal;
  sample_value.pdf_dir = emitter_sample.pdf_dir;
  sample_value.pdf_dir_out = emitter_sample.pdf_dir_out;
  sample_value.texcoord = emitter_sample.image_uv;
  sample_value.emitter_index = emitter_sample.emitter_index;
  sample_value.triangle_index = emitter_sample.triangle_index;
  sample_value.flags = GPUWavefrontDirectLightSampleFlags::Valid;
  if (emitter_sample.is_delta != 0u) {
    sample_value.flags |= GPUWavefrontDirectLightSampleFlags::Delta;
  }
  if (emitter_sample.is_distant != 0u) {
    sample_value.flags |= GPUWavefrontDirectLightSampleFlags::Distant;
  }
  wavefront_store_direct_light_sample(resources.direct_light_sample_buffer, dispatch_index, sample_value);

  if (wavefront_path_vertex_is_medium(current_vertex)) {
    MediumAccess medium_access = (MediumAccess)0;
    if (wavefront_try_load_medium(current_vertex.medium_index, medium_access) == false) {
      return;
    }

    float sampling_pdf = emitter_sample.pdf_dir * emitter_sample.pdf_sample;
    float phase_value = gpu_medium_phase_function(medium_access, current_vertex.w_i, emitter_sample.direction);
    if ((sampling_pdf <= 0.0f) || (phase_value <= 0.0f)) {
      return;
    }

    float mis_weight = wavefront_medium_direct_light_weight(meta, current_vertex, emitter_sample, medium_access, phase_value);
    float upbp_w_light = 0.0f;
    float upbp_emission_to_direct_ratio = 0.0f;
    float upbp_reverse_phase_pdf = 0.0f;
#if ETX_UPBP
    if (upbp) {
      const float light_cosine = emitter_sample.is_distant != 0u ? 1.0f : abs(dot(emitter_sample.normal, -emitter_sample.direction));
      if (upbp_bpt_nee_competitor_terms(emitter_sample.pdf_sample, emitter_sample.pdf_dir, emitter_sample.pdf_dir_out, emitter_sample.is_delta != 0u, phase_value, 1.0f,
            light_cosine, upbp_w_light, upbp_emission_to_direct_ratio) == false) {
        return;
      }
      upbp_reverse_phase_pdf = gpu_medium_phase_function(medium_access, -emitter_sample.direction, current_vertex.w_i);
      mis_weight = 1.0f;
    }
#endif
    if (mis_weight <= 0.0f) {
      return;
    }
    SpectralResponse contribution = spectral_response_mul(current_vertex.throughput, spectral_response_mul(emitter_sample.value, phase_value * (mis_weight / sampling_pdf)));
    if (gpu_valid_spectral_response(contribution) == false) {
      return;
    }

    float3 shadow_delta = emitter_sample.origin - current_vertex.position;
    float shadow_distance = length(shadow_delta);
    if (shadow_distance <= kRayEpsilon) {
      return;
    }

    GPUWavefrontDirectLightTask task = (GPUWavefrontDirectLightTask)0;
    task.shadow_ray.o = current_vertex.position;
    task.shadow_ray.d = shadow_delta / shadow_distance;
    task.shadow_ray.min_t = kRayEpsilon;
    task.shadow_ray.max_t = shadow_distance;
    task.shadow_target = emitter_sample.origin;
    task.contribution = contribution;
    task.mis_weight = upbp ? upbp_w_light : mis_weight;
    if (upbp) {
      task.upbp_scattering_pdf_reverse_bits = asuint(upbp_reverse_phase_pdf);
      task.upbp_auxiliary1_bits = asuint(upbp_emission_to_direct_ratio);
    }
    task.pixel_index = current_vertex.pixel_index;
    task.medium_index = current_vertex.medium_index;
    task.flags = GPUWavefrontPointConnectionTaskFlags::Ready | GPUWavefrontPointConnectionTaskFlags::SourceMedium;
    task.path_index = path_index;
#if ETX_UPBP
    if (upbp) {
      task.sampler_seed = upbp_deterministic_seed(upbp_path_state.global_path_index, upbp_path_state.path_length + 1u, 1u, kUPBPRandomDomainIntersectionTraversal);
      task.upbp_auxiliary0_bits = upbp_deterministic_seed(upbp_path_state.global_path_index, upbp_path_state.path_length + 1u, 1u, kUPBPRandomDomainConnectionTransmittance);
    } else
#endif
    {
      task.sampler_seed = state.sampler_seed;
    }
    wavefront_store_direct_light_task(resources.direct_light_task_buffer, dispatch_index, task);
    wavefront_shadow_queue_append(resources, kGPUWavefrontShadowQueueDirectLight, dispatch_index);
    return;
  }
}
