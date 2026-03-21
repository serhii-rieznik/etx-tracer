#include "gpu_rt_wavefront_common.hlsl"
#include "gpu_rt_wavefront_emitter_sample.hlsl"

float wavefront_direct_light_ris_candidate_weight(WavefrontEmitterSample sample_value, float3 source_position, float3 source_normal) {
  float radiance_weight = luminance(spectral_response_to_rgb(sample_value.value));
  if (radiance_weight <= 0.0f) {
    return 0.0f;
  }

  float3 to_emitter = sample_value.origin - source_position;
  float len_sq = dot(to_emitter, to_emitter);
  float source_alignment = 1.0f;
  if (len_sq > kEpsilon) {
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

bool wavefront_sample_direct_light_ris(uint light_sampling_mode, SpectralQuery spect, float3 source_position, float3 source_normal, inout uint seed,
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

    float candidate_weight = wavefront_direct_light_ris_candidate_weight(candidate, source_position, source_normal);
    float weight = (pdf_sample > 0.0f) ? (candidate_weight / pdf_sample) : 0.0f;
    weight_sum += weight;
    float reservoir_rnd = rnd01(seed);
    if ((weight > 0.0f) && ((reservoir_rnd * weight_sum) < weight)) {
      selected_sample = candidate;
      selected_weight = weight;
    }
  }

  if (selected_weight <= 0.0f) {
    return false;
  }

  selected_sample.value = spectral_response_mul(selected_sample.value, weight_sum / (float(candidate_count) * selected_weight));
  sample_value = selected_sample;
  return true;
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

  GPUWavefrontPathVertex current_vertex = wavefront_load_path_vertex(resources.camera_vertex_buffer, wavefront_vertex_slot(path_index, meta.camera_path_length));
  if ((wavefront_path_vertex_valid(current_vertex) == false) || (wavefront_path_vertex_connectible(current_vertex) == false)) {
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
  bool sampled = false;
  if ((light_sampling_mode == kSceneLightSamplingRISFromDistribution) || (light_sampling_mode == kSceneLightSamplingRISUniform)) {
    sampled = wavefront_sample_direct_light_ris(light_sampling_mode, state.spect, current_vertex.position, current_vertex.normal, seed, emitter_sample);
  } else {
    sampled = wavefront_sample_emitter_to_point(light_sampling_mode, state.spect, current_vertex.position, seed, emitter_sample);
  }
  state.sampler_seed = seed;
  wavefront_store_path_state(resources.camera_state_buffer, path_index, state);
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
