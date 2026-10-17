#include "gpu_rt_wavefront_common.hlsl"
#include "gpu_rt_wavefront_emitter_sample.hlsl"
#include "gpu_rt_wavefront_init_camera.hlsl"

void wavefront_finish_light_emission(GPUWavefrontResources resources, uint path_index, uint2 output_pixel, uint output_pixel_index, SpectralQuery spect, uint seed,
  WavefrontEmitterSample emitter_sample, uint sample_index) {
#if ETX_UPBP
  const GPUUPBPResources upbp_resources = upbp_load_resources(resources);
#endif
  if ((emitter_sample.pdf_area <= 0.0f) || (emitter_sample.pdf_dir <= 0.0f) || spectral_response_is_zero(emitter_sample.value)) {
    return;
  }

  float emission_pdf = emitter_sample.pdf_dir * emitter_sample.pdf_area * emitter_sample.pdf_sample;
  if (emission_pdf <= 0.0f) {
    return;
  }

  float cosine_term = dot(emitter_sample.direction, emitter_sample.normal);
  GPUWavefrontPathState state = (GPUWavefrontPathState)0;
  state.ray.o = emitter_sample.origin;
  if (emitter_sample.triangle_index != kInvalidIndex) {
    GPUWavefrontHit emitter_hit = (GPUWavefrontHit)0;
    emitter_hit.vertex.pos = emitter_sample.origin;
    emitter_hit.vertex.nrm = emitter_sample.normal;
    emitter_hit.triangle_index = emitter_sample.triangle_index;
    emitter_hit.instance_index = emitter_sample.instance_index;
    emitter_hit.barycentric = emitter_sample.barycentric.yz;
    state.ray.o = wavefront_surface_shading_position(emitter_hit, emitter_sample.direction);
  }
  state.ray.d = emitter_sample.direction;
  state.ray.min_t = kRayEpsilon;
  state.ray.max_t = kMaxFloat;
  state.throughput = spectral_response_mul(emitter_sample.value, cosine_term / emission_pdf);
  state.eta = 1.0f;
  state.eta_scale = 1.0f;
  state.sampled_bsdf_pdf = emitter_sample.pdf_dir;
  state.forward_pdf = emitter_sample.is_distant != 0u ? wavefront_safe_div(1.0f, emitter_sample.pdf_area) : wavefront_safe_div(1.0f, emitter_sample.pdf_dir);
  state.reverse_pdf = 0.0f;
  if ((emitter_sample.is_delta == 0u) && (emitter_sample.is_sample_only == 0u)) {
    float reverse_numerator = (emitter_sample.is_distant != 0u) ? 1.0f : cosine_term;
    state.reverse_pdf = wavefront_safe_div(reverse_numerator, emission_pdf);
  }
  state.d_vm = scene_path_mode_is_vcm() ? state.reverse_pdf : 0.0f;
  state.medium_index = emitter_sample.medium_index;
  state.path_length = 1u;
  state.pixel_index = output_pixel_index;
  state.flags = GPUWavefrontPathFlags::Valid | GPUWavefrontPathFlags::From_light | GPUWavefrontPathFlags::Connectible;
  state.sample_index = sample_index;
  state.sampler_seed = seed;
  state.pixel = output_pixel;
  state.spect = spect;
  state.last_vertex_index = (resources.light_vertex_counter_buffer != kInvalidIndex) ? path_index : wavefront_light_vertex_slot(path_index, 0u);
  wavefront_store_path_state(resources.light_state_buffer, path_index, state);
#if ETX_UPBP
  if (scene_path_mode_is_upbp() && (upbp_initialize_light_path(resources, path_index, upbp_resources.iteration.light_batch_offset + path_index, emitter_sample, state) == false)) {
    return;
  }
#endif
  wavefront_write_root_light_vertex(path_index, emitter_sample, spect, output_pixel_index);
  if (resources.path_meta_buffer != kInvalidIndex) {
    GPUWavefrontPathMeta meta = wavefront_load_path_meta(resources.path_meta_buffer, path_index);
    meta.light_path_length = 0u;
    meta.reserved0 = state.last_vertex_index;
    meta.flags |= GPUWavefrontPathMetaFlags::Light_active;
    meta.light_mis_history = scene_path_mode_uses_bdpt_fast() ? 1.0f : 0.0f;
    meta.from_delta = emitter_sample.is_delta;
    wavefront_store_path_meta(resources.path_meta_buffer, path_index, meta);
  }
  wavefront_work_queue_append_wave(wavefront_queue_current_descriptor(false), 0u, path_index, true);
}

void wavefront_initialize_light_path(uint path_index, uint2 output_pixel, uint sample_index) {
  Camera camera = load_camera(bindless_buffers[NonUniformResourceIndex(constants.camera_buffer_index)]);
  uint output_pixel_index = output_pixel.x + output_pixel.y * camera.film_size.x;
  uint2 camera_space_pixel = uint2(output_pixel.x, camera.film_size.y - 1u - output_pixel.y);
  uint seed_pixel_index = camera_space_pixel.x + camera_space_pixel.y * camera.film_size.x;
#if ETX_UPBP
  GPUWavefrontResources upbp_wavefront_resources = wavefront_load_resources();
  GPUUPBPResources upbp_resources = upbp_load_resources(upbp_wavefront_resources);
  const bool upbp_linear_dispatch = scene_path_mode_is_upbp() && (constants.dispatch_item_count > 0u);
  if (upbp_linear_dispatch) {
    seed_pixel_index = upbp_resources.iteration.light_batch_offset + path_index;
    output_pixel_index = seed_pixel_index;
    output_pixel = uint2(seed_pixel_index % camera.film_size.x, (seed_pixel_index / camera.film_size.x) % camera.film_size.y);
  }
#endif
  uint seed = scene_random_domain_seed(seed_pixel_index, sample_index, kSamplerRandomDomainLightPathRoot);
  const SpectralQuery spect = wavefront_vcm_iteration_spectral_query(sample_index);
  if (scene_uses_spectral_mode()) {
    rnd01(seed);
  }
#if ETX_UPBP
  if (upbp_linear_dispatch) {
    seed = upbp_deterministic_seed(upbp_resources.iteration.light_batch_offset + path_index, 0u, 0u, kUPBPRandomDomainLightPath);
  }
#endif
  GPUWavefrontResources resources = wavefront_load_resources();
  GPUWavefrontPathState cleared_state = (GPUWavefrontPathState)0;
  cleared_state.sample_index = sample_index;
  wavefront_store_path_state(resources.light_state_buffer, path_index, cleared_state);
#if ETX_UPBP
  if (scene_path_mode_is_upbp()) {
    // A zero-contribution emission must not retain the previous batch's path history.
    GPUUPBPPathState empty_path = (GPUUPBPPathState)0;
    empty_path.first_vertex_index = kInvalidIndex;
    empty_path.last_vertex_index = kInvalidIndex;
    empty_path.current_segment_index = kInvalidIndex;
    empty_path.current_interval_index = kInvalidIndex;
    empty_path.global_path_index = upbp_resources.iteration.light_batch_offset + path_index;
    empty_path.flags = GPUUPBPPathStateFlags::Light;
    upbp_store_path_state(upbp_resources.path_state_buffer, upbp_path_state_index(upbp_resources, false, path_index), empty_path);
  }
#endif
  if (resources.light_subsurface_state_buffer != kInvalidIndex) {
    GPUWavefrontSubsurfaceState subsurface_state = (GPUWavefrontSubsurfaceState)0;
    subsurface_state.material_index = kInvalidIndex;
    subsurface_state.owner_instance_index = kInvalidIndex;
    subsurface_state.medium_index = kInvalidIndex;
    subsurface_state.scatter_material_index = kInvalidIndex;
    wavefront_store_subsurface_state(resources.light_subsurface_state_buffer, path_index, subsurface_state);
  }
  if (resources.path_meta_buffer != kInvalidIndex) {
    GPUWavefrontPathMeta cleared_meta = wavefront_load_path_meta(resources.path_meta_buffer, path_index);
    cleared_meta.light_path_length = 0u;
    cleared_meta.light_mis_history = 0.0f;
    cleared_meta.from_delta = 0u;
    cleared_meta.flags &= ~GPUWavefrontPathMetaFlags::Light_active;
    wavefront_store_path_meta(resources.path_meta_buffer, path_index, cleared_meta);
  }

  WavefrontEmitterSample emitter_sample = (WavefrontEmitterSample)0;
  WavefrontEmissionInput emission_input = wavefront_empty_emission_input();
  if (wavefront_propose_light_emission(spect, seed, emitter_sample, emission_input) == false) {
    return;
  }
  if (emission_input.material_index != kInvalidIndex) {
    WavefrontEmissionConsumer consumer = (WavefrontEmissionConsumer)0;
    consumer.owner_index = path_index;
    consumer.output_pixel = output_pixel;
    consumer.output_pixel_index = output_pixel_index;
    consumer.spect = spect;
    consumer.candidate = emitter_sample;
    wavefront_queue_emission_input(resources, path_index, emission_input, spect, seed);
    wavefront_emission_consumer_suspend(resources, path_index, consumer);
    return;
  }
  wavefront_finish_light_emission(resources, path_index, output_pixel, output_pixel_index, spect, seed, emitter_sample, sample_index);
}

[numthreads(8, 8, 1)] void wavefront_init_light_path_0_main(uint3 dtid : SV_DispatchThreadID) {
  if (scene_path_mode_is_path_tracing() || (constants.camera_buffer_index == kInvalidIndex)) {
    return;
  }
  uint path_index;
  uint2 output_pixel = (uint2)0;
#if ETX_UPBP
  if (scene_path_mode_is_upbp() && (constants.dispatch_item_count > 0u)) {
    path_index = dtid.x + dtid.y * max(1u, constants.render_window_width);
    if (path_index >= constants.dispatch_item_count) {
      return;
    }
  } else
#endif
  {
    if (wavefront_render_window_contains(dtid.xy) == false) {
      return;
    }
    const uint2 size = wavefront_render_window_size();
    path_index = wavefront_render_window_local_index(dtid.xy) + dtid.z * size.x * size.y;
    if (path_index >= wavefront_load_resources().path_capacity) {
      return;
    }
    output_pixel = wavefront_output_pixel(dtid.xy);
  }
  wavefront_initialize_light_path(path_index, output_pixel, constants.sample_index + dtid.z);
}

  [numthreads(64, 1, 1)] void wavefront_init_light_path_0_resume_main(uint3 dtid : SV_DispatchThreadID) {
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
  consumer.candidate.value = wavefront_completed_sampled_emission(resources, slot, consumer.candidate, seed);
  wavefront_emission_consumer_complete(resources, slot);
  wavefront_finish_light_emission(resources, consumer.owner_index, consumer.output_pixel, consumer.output_pixel_index, consumer.spect, seed, consumer.candidate,
    WAVEFRONT_RO_BUFFER(resources.light_state_buffer).Load(consumer.owner_index * kGPUWavefrontPathStateStride + kGPUWavefrontPathStateSampleIndexOffset));
}

bool wavefront_claim_scheduled_path(GPUWavefrontResources resources, out uint work_index) {
  RWByteAddressBuffer scheduler = WAVEFRONT_RW_BUFFER(resources.scheduler_buffer);
  const uint wave_count = WaveActiveCountBits(true);
  const uint wave_offset = WavePrefixCountBits(true);
  uint wave_base = 0u;
  uint reserved_count = 0u;
  if (WaveIsFirstLane()) {
    const uint total_work = scheduler.Load(kGPUWavefrontSchedulerTotalWorkOffset);
    // Each refill visits one resident pool; the host reserves that much counter headroom.
    scheduler.InterlockedAdd(kGPUWavefrontSchedulerNextWorkOffset, wave_count, wave_base);
    if (wave_base < total_work) {
      reserved_count = min(wave_count, total_work - wave_base);
      const uint remaining = total_work - wave_base - reserved_count;
      const uint camera_queue = wavefront_queue_current_descriptor(true);
      const uint light_queue = wavefront_queue_current_descriptor(false);
      uint ignored;
      if (camera_queue != kInvalidIndex) {
        WAVEFRONT_RW_BUFFER(camera_queue).InterlockedMin(kGPUWavefrontQueuePad2Offset, remaining, ignored);
      }
      if (light_queue != kInvalidIndex) {
        WAVEFRONT_RW_BUFFER(light_queue).InterlockedMin(kGPUWavefrontQueuePad2Offset, remaining, ignored);
      }
    }
  }
  wave_base = WaveReadLaneFirst(wave_base);
  reserved_count = WaveReadLaneFirst(reserved_count);
  work_index = wave_base + wave_offset;
  return wave_offset < reserved_count;
}

[numthreads(64, 1, 1)] void wavefront_refill_paths_main(uint3 dtid : SV_DispatchThreadID) {
  const GPUWavefrontResources resources = wavefront_load_resources();
  const uint path_index = dtid.x;
  if ((path_index >= resources.path_capacity) || (resources.scheduler_buffer == kInvalidIndex)) {
    return;
  }
  if (WAVEFRONT_RO_BUFFER(resources.scheduler_buffer).Load(kGPUWavefrontSchedulerRefillOffset) == 0u) {
    return;
  }
  const uint state_flags_offset = path_index * kGPUWavefrontPathStateStride + kGPUWavefrontPathStateFlagsOffset;
  if ((resources.camera_state_buffer != kInvalidIndex) && ((WAVEFRONT_RO_BUFFER(resources.camera_state_buffer).Load(state_flags_offset) & GPUWavefrontPathFlags::Valid) != 0u)) {
    return;
  }
  if ((resources.light_state_buffer != kInvalidIndex) && ((WAVEFRONT_RO_BUFFER(resources.light_state_buffer).Load(state_flags_offset) & GPUWavefrontPathFlags::Valid) != 0u)) {
    return;
  }
  uint work_index;
  if (wavefront_claim_scheduled_path(resources, work_index) == false) {
    return;
  }
  const uint2 size = wavefront_render_window_size();
  const uint pixel_count = size.x * size.y;
  const uint pixel_index = work_index % pixel_count;
  const uint2 output_pixel = wavefront_output_pixel(uint2(pixel_index % size.x, pixel_index / size.x));
  const uint sample_index = constants.sample_index + work_index / pixel_count;
  if (resources.camera_state_buffer != kInvalidIndex) {
    wavefront_initialize_camera_path(path_index, output_pixel, sample_index, true);
  }
#if ETX_WAVEFRONT_PATH_TRACING_ONLY == 0
  if (resources.light_state_buffer != kInvalidIndex) {
    wavefront_initialize_light_path(path_index, output_pixel, sample_index);
  }
#endif
}
