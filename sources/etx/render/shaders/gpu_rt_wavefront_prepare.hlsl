#include "gpu_rt_wavefront_common.hlsl"

[numthreads(64, 1, 1)] void wavefront_prepare_spectral_values_main(uint3 dtid : SV_DispatchThreadID) {
#if ETX_SPECTRAL_MODE == ETX_SPECTRAL_MODE_SPECTRAL
  if (constants.scene.spectral_values == kInvalidIndex) {
    return;
  }

  const SpectralQuery spect = wavefront_vcm_iteration_spectral_query();
  RWByteAddressBuffer spectral_values = bindless_rw_buffers[NonUniformResourceIndex(constants.scene.spectral_values)];
  if (dtid.x == 0u) {
    const float3 rgb_estimate_scale = spectral_response_to_rgb_estimate(spectral_response_make(spect, 1.0f));
    spectral_values.Store3(kGPUSpectralValuesRGBEstimateScaleOffset, asuint(rgb_estimate_scale));
  }

  if (constants.scene.spectrums == kInvalidIndex) {
    return;
  }

  if (dtid.x < constants.scene.spectrum_count) {
    ByteAddressBuffer spectrum_buffer = bindless_buffers[NonUniformResourceIndex(constants.scene.spectrums)];
    const SpectrumAccessGPUContext spectrum_context = make_spectrum_access_gpu_context(spectrum_buffer, constants.scene.spectrums);
    const float value = spectrum_access_evaluate_wavelength(spectrum_context, dtid.x, spect.wavelength);
    spectral_values.Store(kGPUSpectralValuesDataOffset + dtid.x * 4u, asuint(value));
  }
#else
  (void)dtid;
#endif
}

  [numthreads(8, 8, 1)] void wavefront_prepare_sample_main(uint3 dtid : SV_DispatchThreadID) {
  if (constants.dispatch_item_offset != 0u) {
    if ((dtid.x == 0u) && (dtid.y == 0u)) {
      wavefront_queue_reset(wavefront_queue_current_descriptor(false));
      wavefront_queue_reset(wavefront_queue_next_descriptor(false));
      wavefront_reset_work_queues(wavefront_load_resources());
    }
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
  uint pixel_index = output_pixel.x + output_pixel.y * camera.film_size.x;
  if (constants.sample_index == 0u) {
    wavefront_film_store(pixel_index, float4(0.0f, 0.0f, 0.0f, 0.0f));
  }
  if ((dtid.x == 0u) && (dtid.y == 0u)) {
    wavefront_queue_reset(wavefront_queue_current_descriptor(true));
    wavefront_queue_reset(wavefront_queue_next_descriptor(true));
    wavefront_queue_reset(wavefront_queue_current_descriptor(false));
    wavefront_queue_reset(wavefront_queue_next_descriptor(false));
    GPUWavefrontResources resources = wavefront_load_resources();
    wavefront_reset_work_queues(resources);
    if (resources.light_vertex_counter_buffer != kInvalidIndex) {
      RWByteAddressBuffer counter_buffer = bindless_rw_buffers[NonUniformResourceIndex(resources.light_vertex_counter_buffer)];
      counter_buffer.Store(0u, resources.path_capacity);
    }
  }
}

[numthreads(1, 1, 1)] void wavefront_swap_queues_main(uint3 dtid : SV_DispatchThreadID) {
  if ((dtid.x != 0u) || (dtid.y != 0u) || (dtid.z != 0u)) {
    return;
  }

  wavefront_queue_reset(wavefront_queue_current_descriptor(true));
  wavefront_queue_reset(wavefront_queue_current_descriptor(false));
  wavefront_reset_work_queues(wavefront_load_resources());
}

  [numthreads(8, 8, 1)] void wavefront_finalize_sample_main(uint3 dtid : SV_DispatchThreadID) {
  if (constants.camera_buffer_index == kInvalidIndex) {
    return;
  }
  Camera camera = load_camera(bindless_buffers[NonUniformResourceIndex(constants.camera_buffer_index)]);
  if (wavefront_render_window_contains(dtid.xy) == false) {
    return;
  }

  uint2 output_pixel = wavefront_output_pixel(dtid.xy);
  uint pixel_index = output_pixel.x + output_pixel.y * camera.film_size.x;
  float4 value = wavefront_film_load(pixel_index);
  float sample_count = float(max(1u, constants.sample_index + 1u));
  bindless_storage_textures[NonUniformResourceIndex(constants.output_image_index)][output_pixel] = float4(max(value.xyz / sample_count, float3(0.0f, 0.0f, 0.0f)), 1.0f);
}
