#pragma once

#include "gpu_rt_wavefront_surface_common.hlsl"

#if ETX_UPBP
bool wavefront_upbp_finalize_vertex(bool from_camera, uint path_index, GPUWavefrontPathState state, bool has_departure, out bool terminal) {
  terminal = false;
  if (scene_path_mode_is_upbp() == false) {
    return true;
  }
  GPUUPBPResources resources = upbp_load_resources(wavefront_load_resources());
  SpectralResponse outgoing_throughput = spectral_response_zero(state.spect);
  if (has_departure) {
    outgoing_throughput = state.throughput;
  }
  if (upbp_finalize_physical_vertex(resources, from_camera, path_index, outgoing_throughput, has_departure, terminal)) {
    return true;
  }
  upbp_mark_failed_path(resources, from_camera, path_index, GPUUPBPPathFailure::FinalizeVertex);
  return false;
}

bool wavefront_upbp_random_continue(bool from_camera, uint path_index, float eta, inout SpectralResponse throughput) {
  const GPUUPBPResources resources = upbp_load_resources(wavefront_load_resources());
  const GPUUPBPPathState path_state = upbp_load_path_state(resources.path_state_buffer, upbp_path_state_index(resources, from_camera, path_index));
  if (((path_state.flags & GPUUPBPPathStateFlags::Valid) == 0u) || (path_state.path_length == 0u) || (upbp_path_segment_count(path_state) == 0u)) {
    return false;
  }
  const uint domain = from_camera ? kUPBPRandomDomainCameraRussianRoulette : kUPBPRandomDomainLightRussianRoulette;
  uint seed = upbp_deterministic_seed(path_state.global_path_index, path_state.path_length - 1u, upbp_path_segment_count(path_state) - 1u, domain);
  return gpu_random_continue(path_state.path_length - 1u, load_scene_options_random_path_termination(), eta, seed, throughput);
}
#endif

void wavefront_surface_continue_finalize(bool from_camera, uint dispatch_index) {
  uint queue_descriptor = wavefront_queue_current_descriptor(from_camera);
  uint queue_count = wavefront_queue_count(queue_descriptor);
  if (dispatch_index >= queue_count) {
    return;
  }

  GPUWavefrontResources resources = wavefront_load_resources();
  uint path_index = wavefront_queue_load(queue_descriptor, dispatch_index);
  uint state_descriptor = from_camera ? resources.camera_state_buffer : resources.light_state_buffer;
  uint hit_descriptor = from_camera ? resources.camera_hit_buffer : resources.light_hit_buffer;
  GPUWavefrontPathState state = wavefront_load_path_state(state_descriptor, path_index);
  GPUWavefrontHit hit = wavefront_load_hit(hit_descriptor, path_index);
  if ((wavefront_path_state_valid(state) == false) || (wavefront_hit_valid(hit) == false) || wavefront_hit_is_miss(hit)) {
    return;
  }

  if ((state.reserved0 & GPUWavefrontPendingContinuationFlags::Prepared) == 0u) {
    state.flags = 0u;
    wavefront_store_path_state(state_descriptor, path_index, state);
    return;
  }

  uint pending_flags = state.reserved0;
  state.reserved0 = 0u;
  if ((pending_flags & GPUWavefrontPendingContinuationFlags::Continue) == 0u) {
#if ETX_UPBP
    bool upbp_terminal = false;
    wavefront_upbp_finalize_vertex(from_camera, path_index, state, false, upbp_terminal);
#endif
    state.flags = 0u;
    wavefront_store_path_state(state_descriptor, path_index, state);
    return;
  }

  uint vertex_descriptor = from_camera ? resources.camera_vertex_buffer : resources.light_vertex_buffer;
  GPUWavefrontPathVertex current_vertex = wavefront_load_path_vertex(vertex_descriptor, wavefront_path_vertex_slot(from_camera, path_index, state.path_length));
  if (wavefront_path_vertex_valid(current_vertex) == false) {
    state.flags = 0u;
    wavefront_store_path_state(state_descriptor, path_index, state);
    return;
  }

  uint next_path_length = state.path_length + 1u;
  bool within_depth_policy = next_path_length <= resources.max_path_length;
  if (within_depth_policy && scene_path_mode_is_path_tracing() && (next_path_length > load_scene_options_max_path_length())) {
    within_depth_policy = (gpu_path_tracing_neutral_eta(state.eta) == false) || wavefront_path_state_refractive_depth_limit_reached(state);
  }
  if (within_depth_policy == false) {
#if ETX_UPBP
    bool upbp_terminal = false;
    wavefront_upbp_finalize_vertex(from_camera, path_index, state, false, upbp_terminal);
#endif
    state.flags = 0u;
    wavefront_store_path_state(state_descriptor, path_index, state);
    return;
  }
  wavefront_path_tracing_update_refractive_depth(state, next_path_length);
  if (wavefront_path_tracing_should_terminate_neutral_depth(state, next_path_length)) {
#if ETX_UPBP
    bool upbp_terminal = false;
    wavefront_upbp_finalize_vertex(from_camera, path_index, state, false, upbp_terminal);
#endif
    state.flags = 0u;
    wavefront_store_path_state(state_descriptor, path_index, state);
    return;
  }

  const uint continuation_path_length = next_path_length;
  bool random_continue = spectral_response_is_zero(state.throughput) == false;
  if (random_continue) {
#if ETX_UPBP
    if (scene_path_mode_is_upbp()) {
      random_continue = wavefront_upbp_random_continue(from_camera, path_index, state.eta, state.throughput);
    } else
#endif
    {
      random_continue = gpu_random_continue(continuation_path_length, load_scene_options_random_path_termination(), state.eta, state.sampler_seed, state.throughput);
    }
  }
  if (random_continue == false) {
#if ETX_UPBP
    bool upbp_terminal = false;
    wavefront_upbp_finalize_vertex(from_camera, path_index, state, false, upbp_terminal);
#endif
    state.flags = 0u;
    wavefront_store_path_state(state_descriptor, path_index, state);
    return;
  }

  state.path_length = next_path_length;
  const uint persistent_flags = wavefront_path_state_persistent_flags(state);
  state.flags = persistent_flags | GPUWavefrontPathFlags::Valid | (from_camera ? GPUWavefrontPathFlags::From_camera : GPUWavefrontPathFlags::From_light);
  if (wavefront_path_vertex_connectible(current_vertex)) {
    state.flags |= GPUWavefrontPathFlags::Connectible;
  }
#if ETX_UPBP
  bool upbp_terminal = false;
  if ((wavefront_upbp_finalize_vertex(from_camera, path_index, state, true, upbp_terminal) == false) || upbp_terminal) {
    state.flags = 0u;
    wavefront_store_path_state(state_descriptor, path_index, state);
    return;
  }
#endif
  wavefront_enqueue_next_state(from_camera, path_index, state);
}
