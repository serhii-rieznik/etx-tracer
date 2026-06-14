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

float wavefront_direct_light_vertex_to_vertex_area_pdf(float pdf_dir, GPUWavefrontPathVertex from_vertex, GPUWavefrontPathVertex to_vertex) {
  if (wavefront_path_vertex_is_infinite_emitter(to_vertex)) {
    return pdf_dir;
  }
  return wavefront_convert_solid_angle_pdf_to_area(pdf_dir, from_vertex.position, to_vertex.position, wavefront_path_vertex_is_surface(to_vertex), to_vertex.normal);
}

float wavefront_direct_light_emitter_sample_pdf(WavefrontEmitterSample sample_value) {
  if (sample_value.is_distant != 0u) {
    float directional_pdf = (sample_value.is_delta != 0u) ? 1.0f : sample_value.pdf_dir;
    return sample_value.pdf_sample * directional_pdf;
  }

  return sample_value.pdf_sample * sample_value.pdf_area;
}

float wavefront_emitter_sample_from_emitter_pdf(WavefrontEmitterSample sample_value, GPUWavefrontPathVertex target_vertex) {
  if (sample_value.is_distant != 0u) {
    float3 w_o = normalize(sample_value.origin - target_vertex.position);
    float cosine_term = wavefront_path_vertex_is_surface(target_vertex) ? abs(dot(target_vertex.geo_normal, w_o)) : 1.0f;
    return sample_value.pdf_area * cosine_term;
  }

  float3 w_o = target_vertex.position - sample_value.origin;
  float distance_squared = dot(w_o, w_o);
  if (distance_squared <= kEpsilon) {
    return 0.0f;
  }

  w_o *= rsqrt(distance_squared);
  float3 emitter_normal = normalize(sample_value.normal);
  float exponent = 1.0f;
  if (sample_value.triangle_index != kInvalidIndex) {
    TriangleData tri = load_triangle(bindless_buffers[NonUniformResourceIndex(constants.scene.triangles)], sample_value.triangle_index);
    Material emitter_material = (Material)0;
    MaterialAccessGPUContext material_context = {constants.scene.materials};
    if (material_access_try_load_full(material_context, tri.material_index, emitter_material) == false) {
      return 0.0f;
    }
    exponent = scene_math_shared_collimation_to_exponent(emitter_material.emission_collimation);
    emitter_normal = tri.geo_n;
  }

  float pdf_dir = pow(max(0.0f, dot(emitter_normal, w_o)), exponent) * kInvPi;
  return wavefront_convert_solid_angle_pdf_to_area(pdf_dir, sample_value.origin, target_vertex.position, wavefront_path_vertex_is_surface(target_vertex), target_vertex.normal);
}

float wavefront_emitter_sample_to_vertex_area_pdf(WavefrontEmitterSample sample_value, GPUWavefrontPathVertex source_vertex, float pdf_dir) {
  if (sample_value.is_distant != 0u) {
    GPUEmitterInstanceABIData emitter_instance = (GPUEmitterInstanceABIData)0;
    if (try_load_emitter_instance(sample_value.emitter_index, emitter_instance) == false) {
      return 0.0f;
    }

    if (emitter_instance.emitter_class == EmitterClass::Environment) {
      return pdf_dir;
    }

    return wavefront_convert_solid_angle_pdf_to_area(pdf_dir, source_vertex.position, sample_value.origin, false, float3(0.0f, 0.0f, 0.0f));
  }

  return wavefront_convert_solid_angle_pdf_to_area(pdf_dir, source_vertex.position, sample_value.origin, true, sample_value.normal);
}

float wavefront_medium_direct_light_weight(
  GPUWavefrontResources resources, uint path_index, GPUWavefrontPathMeta path_meta, GPUWavefrontPathVertex current_vertex, WavefrontEmitterSample emitter_sample,
  MediumAccess medium_access, float phase_value) {
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

  if (path_meta.camera_path_length == 0u) {
    return 1.0f;
  }

  GPUWavefrontPathVertex previous_vertex =
    wavefront_load_path_vertex(resources.camera_vertex_buffer, wavefront_camera_vertex_slot(path_index, path_meta.camera_path_length - 1u));
  if (wavefront_path_vertex_valid(previous_vertex) == false) {
    return 0.0f;
  }

  float p_sample = wavefront_direct_light_emitter_sample_pdf(emitter_sample);
  float p_fwd = previous_vertex.pdf_from_prev * current_vertex.pdf_from_prev;
  float p_connection = p_fwd * p_sample;
  float p_direct = 0.0f;
  if (emitter_sample.is_delta == 0u) {
    float p_bsdf_sample = wavefront_emitter_sample_to_vertex_area_pdf(emitter_sample, current_vertex, phase_value);
    p_direct = p_fwd * p_bsdf_sample;
  }

  float reverse_phase_pdf = gpu_medium_phase_function(medium_access, -emitter_sample.direction, current_vertex.w_i);
  float z_prev_backward_pdf = wavefront_direct_light_vertex_to_vertex_area_pdf(reverse_phase_pdf, current_vertex, previous_vertex);
  float p_bck = previous_vertex.pdf_history;
  if (path_meta.camera_path_length > 1u) {
    p_bck *= z_prev_backward_pdf;
  }

  float from_emitter = wavefront_emitter_sample_from_emitter_pdf(emitter_sample, current_vertex);
  float p_light_path = p_sample * from_emitter * p_bck;
  return balance_heuristic(p_connection, p_direct, p_light_path);
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
  if ((wavefront_path_vertex_valid(current_vertex) == false) || (wavefront_path_vertex_connectible(current_vertex) == false) || subsurface_medium_vertex) {
    return;
  }
  uint connection_length = meta.camera_path_length + 1u;
  if ((scene_strategy_enabled(kSceneStrategyConnectToLight) == false) || (connection_length < load_scene_options_min_path_length()) ||
      (connection_length > load_scene_options_max_path_length())) {
    return;
  }

  uint seed = state.sampler_seed;
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
  state.sampler_seed = seed;
  wavefront_store_path_state(resources.camera_state_buffer, path_index, state);
  if (sampled == false) {
    return;
  }

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

    float mis_weight = wavefront_medium_direct_light_weight(resources, path_index, meta, current_vertex, emitter_sample, medium_access, phase_value);
    if (mis_weight <= 0.0f) {
      return;
    }
    SpectralResponse contribution =
      spectral_response_mul(current_vertex.throughput, spectral_response_mul(emitter_sample.value, phase_value * (mis_weight / sampling_pdf)));
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
    task.mis_weight = mis_weight;
    task.pixel_index = current_vertex.pixel_index;
    task.medium_index = current_vertex.medium_index;
    task.flags = 1u;
    task.path_index = path_index;
    task.sampler_seed = state.sampler_seed;
    wavefront_store_direct_light_task(resources.direct_light_task_buffer, dispatch_index, task);
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
}
