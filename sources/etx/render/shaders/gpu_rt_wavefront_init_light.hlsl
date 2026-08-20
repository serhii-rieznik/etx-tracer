#include "gpu_rt_wavefront_common.hlsl"
#include "gpu_rt_wavefront_emitter_sample.hlsl"

[numthreads(8, 8, 1)] void wavefront_init_light_path_0_main(uint3 dtid : SV_DispatchThreadID) {
  if (scene_path_mode_is_path_tracing()) {
    return;
  }
  if (constants.camera_buffer_index == kInvalidIndex) {
    return;
  }
  Camera camera = load_camera(bindless_buffers[NonUniformResourceIndex(constants.camera_buffer_index)]);
  if (wavefront_render_window_contains(dtid.xy) == false) {
    return;
  }

  uint2 output_pixel = wavefront_output_pixel(dtid.xy);
  uint output_pixel_index = output_pixel.x + output_pixel.y * camera.film_size.x;
  uint path_index = wavefront_render_window_local_index(dtid.xy);
  uint2 camera_space_pixel = uint2(output_pixel.x, camera.film_size.y - 1u - output_pixel.y);
  uint seed_pixel_index = camera_space_pixel.x + camera_space_pixel.y * camera.film_size.x;
  uint seed = scene_random_seed(seed_pixel_index, constants.sample_index);
  SpectralQuery spect = spectral_query_sample();
  if (scene_path_mode_is_vcm()) {
    spect = wavefront_vcm_iteration_spectral_query();
    if (scene_uses_spectral_mode()) {
      rnd01(seed);
    } else if (scene_has_diffraction_grating()) {
      rnd01(seed);
      rnd01(seed);
    }
  } else if (scene_uses_spectral_mode()) {
    spect = spectral_query_spectral_sample(rnd01(seed));
  } else if (scene_has_diffraction_grating()) {
    spect = diffraction_transport_sample_query(false, true, rnd01(seed), rnd01(seed));
  }
  GPUWavefrontResources resources = wavefront_load_resources();
  GPUWavefrontPathState cleared_state = (GPUWavefrontPathState)0;
  wavefront_store_path_state(resources.light_state_buffer, path_index, cleared_state);
  if (resources.light_subsurface_state_buffer != kInvalidIndex) {
    GPUWavefrontSubsurfaceState subsurface_state = (GPUWavefrontSubsurfaceState)0;
    subsurface_state.material_index = kInvalidIndex;
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
  if (wavefront_sample_light_emission(spect, seed, emitter_sample) == false) {
    return;
  }
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
  if (emitter_sample.is_delta == 0u) {
    float reverse_numerator = (emitter_sample.is_distant != 0u) ? 1.0f : cosine_term;
    state.reverse_pdf = wavefront_safe_div(reverse_numerator, emission_pdf);
  }
  state.d_vm = scene_path_mode_is_vcm() ? (state.reverse_pdf * constants.vcm_vc_weight) : 0.0f;
  state.medium_index = emitter_sample.medium_index;
  state.path_length = 1u;
  state.pixel_index = output_pixel_index;
  state.flags = GPUWavefrontPathFlags::Valid | GPUWavefrontPathFlags::From_light | GPUWavefrontPathFlags::Connectible;
  state.path_source = PathSource::Light;
  state.sampler_seed = seed;
  state.pixel = output_pixel;
  state.spect = spect;
  state.last_vertex_index = (resources.light_vertex_counter_buffer != kInvalidIndex) ? path_index : wavefront_light_vertex_slot(path_index, 0u);
  wavefront_store_path_state(resources.light_state_buffer, path_index, state);
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
  wavefront_queue_append(wavefront_queue_current_descriptor(false), path_index);
}
