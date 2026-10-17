#include "gpu_rt_wavefront_surface_common.hlsl"

void wavefront_accumulate_camera_direct_hit(WavefrontEmissionConsumer consumer, SpectralResponse radiance) {
  const SpectralResponse contribution = spectral_response_mul(spectral_response_mul(radiance, consumer.factor), consumer.selected_weight);
  if (spectral_response_is_zero(contribution) == false) {
    wavefront_film_add(consumer.output_pixel_index, wavefront_spectral_estimate(contribution, consumer.spect));
  }
}

void wavefront_finish_camera_direct_hit(GPUWavefrontResources resources, uint slot, WavefrontEmissionConsumer consumer, SpectralResponse radiance, uint seed) {
  GPUWavefrontPathState state = wavefront_load_path_state(resources.camera_state_buffer, consumer.owner_index);
  state.sampler_seed = seed;
  wavefront_store_path_state(resources.camera_state_buffer, consumer.owner_index, state);
  wavefront_emission_consumer_complete(resources, slot);
  wavefront_accumulate_camera_direct_hit(consumer, radiance);
}

[numthreads(64, 1, 1)] void wavefront_camera_direct_hit_accumulate_main(uint3 dtid : SV_DispatchThreadID) {
  const uint dispatch_index = dtid.x;
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
  if (scene_path_mode_is_light_tracing() && ((state.flags & GPUWavefrontPathFlags::Camera_source_prefix) == 0u)) {
    return;
  }
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
  if (emitter_access_is_local_class(emitter_instance.emitter_class) == false) {
    return;
  }

  uint current_vertex_index = wavefront_camera_vertex_slot(path_index, meta.camera_path_length);
  uint previous_vertex_index = wavefront_camera_vertex_slot(path_index, meta.camera_path_length - 1u);
  GPUWavefrontPathVertex current_vertex = wavefront_load_path_vertex(resources.camera_vertex_buffer, current_vertex_index);
  GPUWavefrontPathVertex previous_vertex = wavefront_load_path_vertex(resources.camera_vertex_buffer, previous_vertex_index);
  if ((wavefront_path_vertex_valid(current_vertex) == false) || (wavefront_path_vertex_valid(previous_vertex) == false)) {
    return;
  }

  float pdf_area = 0.0f;
  float pdf_dir = 0.0f;
  float pdf_dir_out = 0.0f;
  WavefrontEmissionInput emission_input = wavefront_empty_emission_input();
  if (wavefront_propose_local_direct_hit(hit.emitter_index, previous_vertex.position, current_vertex.position, current_vertex.texcoord, current_vertex.normal,
        current_vertex.material_index, current_vertex.barycentric, emission_input, pdf_area, pdf_dir, pdf_dir_out) == false) {
    return;
  }

  WavefrontEmissionConsumer consumer = (WavefrontEmissionConsumer)0;
  consumer.owner_index = path_index;
  consumer.sampler_seed = state.sampler_seed;
  consumer.output_pixel_index = state.pixel_index;
  consumer.spect = state.spect;
  consumer.factor = current_vertex.throughput;
  consumer.selected_weight = wavefront_local_direct_hit_weight(meta.camera_path_length, current_vertex, previous_vertex, hit.emitter_index, pdf_area, pdf_dir, pdf_dir_out);
  Material material = (Material)0;
  if (try_load_material_full(emission_input.material_index, material) == false) {
    return;
  }
  if (thermal_surface_uses_substrate_escape(material) == false) {
    const BSDFResourceContext context = make_bsdf_resource_gpu_context(constants.scene.images, constants.scene.spectrums, constants.scene.spectral_values,
      constants.scene.energy_compensation_interfaces, constants.scene.scene_globals);
    const SpectralResponse radiance =
      thermal_surface_combined_radiance(context, state.spect, material, emission_input.uv, emission_input.geometric_cosine, emission_input.local_outgoing.z);
    wavefront_accumulate_camera_direct_hit(consumer, radiance);
    return;
  }
  wavefront_queue_emission_input(resources, dispatch_index, emission_input, state.spect, state.sampler_seed);
  wavefront_emission_consumer_suspend(resources, dispatch_index, consumer);
}

  [numthreads(64, 1, 1)] void wavefront_camera_direct_hit_accumulate_resume_main(uint3 dtid : SV_DispatchThreadID) {
  if (dtid.x >= constants.dispatch_item_count) {
    return;
  }
  const uint slot = constants.dispatch_item_offset + dtid.x;
  const GPUWavefrontResources resources = wavefront_load_resources();
  WavefrontEmissionConsumer consumer = (WavefrontEmissionConsumer)0;
  if (wavefront_emission_consumer_load(resources, slot, consumer) == false) {
    return;
  }
  uint seed = 0u;
  const SpectralResponse radiance = wavefront_completed_emission(resources, slot, seed);
  wavefront_finish_camera_direct_hit(resources, slot, consumer, radiance, seed);
}
