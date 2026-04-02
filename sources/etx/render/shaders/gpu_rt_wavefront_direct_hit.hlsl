#include "gpu_rt_wavefront_surface_common.hlsl"

[numthreads(64, 1, 1)] void wavefront_camera_direct_hit_accumulate_main(uint3 dtid : SV_DispatchThreadID) {
  const uint dispatch_index = dtid.x;
  if (scene_path_mode_is_light_tracing()) {
    return;
  }
  if (scene_strategy_enabled(kSceneStrategyDirectHit) == false) {
    return;
  }

  GPUWavefrontResources resources = wavefront_load_resources();
  uint queue_descriptor = wavefront_queue_current_descriptor(true);
  uint queue_count = wavefront_queue_count(queue_descriptor);
  if (dispatch_index >= queue_count) {
    return;
  }

  uint path_index = wavefront_queue_load(queue_descriptor, dispatch_index);
  GPUWavefrontPathState state = wavefront_load_path_state(resources.camera_state_buffer, path_index);
  GPUWavefrontHit hit = wavefront_load_hit(resources.camera_hit_buffer, path_index);
  GPUWavefrontPathMeta meta = wavefront_load_path_meta(resources.path_meta_buffer, path_index);
  if ((wavefront_hit_valid(hit) == false) || wavefront_hit_is_miss(hit)) {
    return;
  }
  if (hit.emitter_index == kInvalidIndex) {
    return;
  }
  if ((meta.camera_path_length < load_scene_options_min_path_length()) || (meta.camera_path_length > load_scene_options_max_path_length())) {
    return;
  }

  GPUEmitterInstanceABIData emitter_instance = (GPUEmitterInstanceABIData)0;
  if (try_load_emitter_instance(hit.emitter_index, emitter_instance) == false) {
    return;
  }
  if (emitter_instance.emitter_class != EmitterClass::Area) {
    return;
  }

  uint current_vertex_index = wavefront_camera_vertex_slot(path_index, meta.camera_path_length);
  uint previous_vertex_index = wavefront_camera_vertex_slot(path_index, meta.camera_path_length - 1u);
  GPUWavefrontPathVertex current_vertex = wavefront_load_path_vertex(resources.camera_vertex_buffer, current_vertex_index);
  GPUWavefrontPathVertex previous_vertex = wavefront_load_path_vertex(resources.camera_vertex_buffer, previous_vertex_index);
  if ((wavefront_path_vertex_valid(current_vertex) == false) || (wavefront_path_vertex_valid(previous_vertex) == false)) {
    return;
  }

  SpectralResponse contribution = wavefront_compute_local_direct_hit_contribution(state.spect, meta.camera_path_length, current_vertex, previous_vertex, hit.emitter_index);
  if (spectral_response_is_zero(contribution)) {
    return;
  }

  wavefront_film_add(state.pixel_index, spectral_response_to_rgb(contribution) * wavefront_spectral_weight(state.spect));
}
