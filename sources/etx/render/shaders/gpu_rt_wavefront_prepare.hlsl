#include "gpu_rt_wavefront_common.hlsl"

[numthreads(8, 8, 1)] void wavefront_prepare_sample_main(uint3 dtid : SV_DispatchThreadID) {
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
  }
}

  [numthreads(1, 1, 1)] void wavefront_swap_queues_main(uint3 dtid : SV_DispatchThreadID) {
  if ((dtid.x != 0u) || (dtid.y != 0u) || (dtid.z != 0u)) {
    return;
  }

  wavefront_queue_reset(wavefront_queue_current_descriptor(true));
  wavefront_queue_reset(wavefront_queue_current_descriptor(false));
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
