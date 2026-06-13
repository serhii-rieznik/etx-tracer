#pragma once

#include "gpu_rt_wavefront_surface_common.hlsl"

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

  if (wavefront_path_vertex_is_medium(current_vertex)) {
    uint next_path_length = state.path_length + 1u;
    bool within_depth_policy = next_path_length <= resources.max_path_length;
    if (within_depth_policy && scene_path_mode_is_path_tracing() && (next_path_length > load_scene_options_max_path_length())) {
      within_depth_policy = (gpu_path_tracing_neutral_eta(state.eta) == false) || wavefront_path_state_refractive_depth_limit_reached(state);
    }
    if (within_depth_policy == false) {
      state.flags = 0u;
      wavefront_store_path_state(state_descriptor, path_index, state);
      return;
    }
    wavefront_path_tracing_update_refractive_depth(state, next_path_length);
    if (wavefront_path_tracing_should_terminate_neutral_depth(state, next_path_length)) {
      state.flags = 0u;
      wavefront_store_path_state(state_descriptor, path_index, state);
      return;
    }

    const uint continuation_path_length = next_path_length;
    if (spectral_response_is_zero(state.throughput) ||
        (gpu_random_continue(continuation_path_length, load_scene_options_random_path_termination(), state.eta, state.sampler_seed, state.throughput) == false)) {
      state.flags = 0u;
      wavefront_store_path_state(state_descriptor, path_index, state);
      return;
    }

    state.path_length = next_path_length;
    const uint depth_flag = wavefront_path_state_depth_flag(state);
    state.flags = depth_flag | GPUWavefrontPathFlags::Valid | (from_camera ? GPUWavefrontPathFlags::From_camera : GPUWavefrontPathFlags::From_light);
    if (wavefront_path_vertex_connectible(current_vertex)) {
      state.flags |= GPUWavefrontPathFlags::Connectible;
    }
    wavefront_enqueue_next_state(from_camera, path_index, state);
    return;
  }

  uint next_path_length = state.path_length + 1u;
  bool within_depth_policy = next_path_length <= resources.max_path_length;
  if (within_depth_policy && scene_path_mode_is_path_tracing() && (next_path_length > load_scene_options_max_path_length())) {
    within_depth_policy = (gpu_path_tracing_neutral_eta(state.eta) == false) || wavefront_path_state_refractive_depth_limit_reached(state);
  }
  if (within_depth_policy == false) {
    state.flags = 0u;
    wavefront_store_path_state(state_descriptor, path_index, state);
    return;
  }
  wavefront_path_tracing_update_refractive_depth(state, next_path_length);
  if (wavefront_path_tracing_should_terminate_neutral_depth(state, next_path_length)) {
    state.flags = 0u;
    wavefront_store_path_state(state_descriptor, path_index, state);
    return;
  }

  const uint continuation_path_length = next_path_length;
  if (spectral_response_is_zero(state.throughput) ||
      (gpu_random_continue(continuation_path_length, load_scene_options_random_path_termination(), state.eta, state.sampler_seed, state.throughput) == false)) {
    state.flags = 0u;
    wavefront_store_path_state(state_descriptor, path_index, state);
    return;
  }

  state.path_length = next_path_length;
  const uint depth_flag = wavefront_path_state_depth_flag(state);
  state.flags = depth_flag | GPUWavefrontPathFlags::Valid | (from_camera ? GPUWavefrontPathFlags::From_camera : GPUWavefrontPathFlags::From_light);
  if (wavefront_path_vertex_connectible(current_vertex)) {
    state.flags |= GPUWavefrontPathFlags::Connectible;
  }
  wavefront_enqueue_next_state(from_camera, path_index, state);
}
