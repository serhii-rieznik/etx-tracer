#include "gpu_rt_wavefront_common.hlsl"

[numthreads(8, 8, 1)] void wavefront_init_camera_path_0_main(uint3 dtid : SV_DispatchThreadID) {
  if (scene_path_mode_is_light_tracing()) {
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
  uint seed = scene_random_domain_seed(seed_pixel_index, constants.sample_index, kSamplerRandomDomainCameraPathRoot);
  SpectralQuery spect = spectral_query_sample();
  if (scene_path_mode_is_vcm() || scene_path_mode_is_upbp()) {
    spect = wavefront_vcm_iteration_spectral_query();
    if (scene_uses_spectral_mode()) {
      rnd01(seed);
    }
  } else if (scene_uses_spectral_mode()) {
    spect = spectral_query_spectral_sample(rnd01(seed));
  }
  uint film_seed = seed;
#if ETX_UPBP
  if (scene_path_mode_is_upbp()) {
    const GPUUPBPResources upbp_resources = upbp_load_resources(wavefront_load_resources());
    const uint global_path_index = upbp_resources.iteration.camera_batch_offset + path_index;
    film_seed = upbp_deterministic_seed(global_path_index, 0u, 0u, kUPBPRandomDomainFilmSample);
    seed = upbp_deterministic_seed(global_path_index, 0u, 0u, kUPBPRandomDomainCameraPath);
  }
#endif
  float2 uv_sample = float2(rnd01(film_seed), rnd01(film_seed));
  if (scene_path_mode_is_upbp() == false) {
    seed = film_seed;
  }
  if (sample_use_blue_noise_primary(constants.sample_index, kSamplerStreamOther)) {
    uint film_dimension = sampler_stream_dimension_base(kSamplerStreamOther);
    uint2 sample_pixel = sample_blue_noise_translated_pixel(camera_space_pixel);
    uv_sample = float2(sample_blue_noise_value_at_translated_pixel(sample_pixel, constants.sample_index, film_dimension + 0u),
      sample_blue_noise_value_at_translated_pixel(sample_pixel, constants.sample_index, film_dimension + 1u));
  }
  float2 uv = camera_sample_film_uv(output_pixel, camera.film_size, uv_sample);
  float2 lens_rnd = float2(rnd01(seed), rnd01(seed));
  GPUWavefrontPathState state = (GPUWavefrontPathState)0;
  state.ray = camera_generate_primary_ray(camera, uv, lens_rnd);
  state.throughput = spectral_response_make(spect, 1.0f);
  state.eta = 1.0f;
  state.eta_scale = 1.0f;
  state.sampled_bsdf_pdf = camera_film_shared_evaluate_out(camera, state.ray).pdf_dir;
  state.forward_pdf = wavefront_safe_div(1.0f, state.sampled_bsdf_pdf);
  state.reverse_pdf = 0.0f;
  state.d_vm = 0.0f;
  state.medium_index = camera.medium_index;
  state.path_length = 1u;
  state.pixel_index = output_pixel_index;
  state.flags = GPUWavefrontPathFlags::Valid | GPUWavefrontPathFlags::Connectible | GPUWavefrontPathFlags::From_camera;
  state.path_source = PathSource::Camera;
  state.sampler_seed = seed;
  state.pixel = camera_space_pixel;
  state.spect = spect;
  state.film_uv = uv;
  state.last_vertex_index = wavefront_camera_vertex_slot(path_index, 0u);
  GPUWavefrontResources resources = wavefront_load_resources();
#if ETX_UPBP
  if (scene_path_mode_is_upbp()) {
    GPUUPBPResources upbp_resources = upbp_load_resources(resources);
    if (upbp_initialize_camera_path(resources, path_index, upbp_resources.iteration.camera_batch_offset + path_index, camera, state) == false) {
      return;
    }
  }
#endif
  wavefront_store_path_state(resources.camera_state_buffer, path_index, state);
  if (resources.camera_subsurface_state_buffer != kInvalidIndex) {
    GPUWavefrontSubsurfaceState subsurface_state = (GPUWavefrontSubsurfaceState)0;
    subsurface_state.material_index = kInvalidIndex;
    subsurface_state.medium_index = kInvalidIndex;
    subsurface_state.scatter_material_index = kInvalidIndex;
    wavefront_store_subsurface_state(resources.camera_subsurface_state_buffer, path_index, subsurface_state);
  }
  wavefront_write_root_camera_vertex(path_index, camera, state.ray, spect, output_pixel_index);
  if (resources.path_meta_buffer != kInvalidIndex) {
    GPUWavefrontPathMeta meta = (GPUWavefrontPathMeta)0;
    if (scene_path_mode_is_bdpt_full()) {
      meta = wavefront_load_path_meta(resources.path_meta_buffer, path_index);
    }
    meta.camera_path_length = 1u;
    meta.camera_mis_history = scene_path_mode_uses_bdpt_fast() ? 1.0f : 0.0f;
    if (scene_path_mode_is_bdpt_full()) {
      meta.flags |= GPUWavefrontPathMetaFlags::Camera_active;
    } else {
      meta.flags = GPUWavefrontPathMetaFlags::Camera_active;
    }
    wavefront_store_path_meta(resources.path_meta_buffer, path_index, meta);
  }
  wavefront_queue_store(wavefront_queue_current_descriptor(true), path_index, path_index);
  if ((dtid.x == 0u) && (dtid.y == 0u)) {
    uint2 render_window_size = wavefront_render_window_size();
    WAVEFRONT_RW_BUFFER(wavefront_queue_current_descriptor(true)).Store(0u, render_window_size.x * render_window_size.y);
  }
}
