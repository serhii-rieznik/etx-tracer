#include "gpu_rt_wavefront_common.hlsl"

void wavefront_store_dispatch_args(RWByteAddressBuffer buffer, uint offset, uint queue_count, uint group_count_y) {
  buffer.Store(offset + 0u, (queue_count + 63u) / 64u);
  buffer.Store(offset + 4u, group_count_y);
  buffer.Store(offset + 8u, 1u);
  buffer.Store(offset + 12u, 0u);
}

[numthreads(64, 1, 1)] void wavefront_build_dispatch_args_main(uint3 dtid : SV_DispatchThreadID) {
  GPUWavefrontResources resources = wavefront_load_resources();
  if (resources.dispatch_args_buffer == kInvalidIndex) {
    return;
  }

  RWByteAddressBuffer dispatch_args = WAVEFRONT_RW_BUFFER(resources.dispatch_args_buffer);
  const uint camera_count = wavefront_queue_count(wavefront_queue_current_descriptor(true));
  const uint light_count = wavefront_queue_count(wavefront_queue_current_descriptor(false));
  if (dtid.x == 0u) {
    wavefront_store_dispatch_args(dispatch_args, kGPUWavefrontCameraDispatchArgsOffset, camera_count, 1u);
    wavefront_store_dispatch_args(dispatch_args, kGPUWavefrontLightDispatchArgsOffset, light_count, 1u);
    for (uint batch_size = 1u; batch_size <= kGPUWavefrontConnectDispatchArgsCount; ++batch_size) {
      const uint offset = kGPUWavefrontConnectDispatchArgsOffset + (batch_size - 1u) * kGPUWavefrontDispatchArgsStride;
      wavefront_store_dispatch_args(dispatch_args, offset, camera_count, batch_size);
    }
    for (uint queue_index = 0u; queue_index < kGPUWavefrontMaterialQueueCount; ++queue_index) {
      const bool from_camera = queue_index < kGPUWavefrontMaterialQueueCountPerPathType;
      const uint material_queue_index = queue_index % kGPUWavefrontMaterialQueueCountPerPathType;
      const uint material_count = wavefront_material_queue_count(resources, from_camera, material_queue_index);
      const uint offset = kGPUWavefrontMaterialDispatchArgsOffset + queue_index * kGPUWavefrontDispatchArgsStride;
      wavefront_store_dispatch_args(dispatch_args, offset, material_count, 1u);
    }
    for (uint queue_index = 0u; queue_index < kGPUWavefrontShadowQueueCount; ++queue_index) {
      const uint shadow_count = wavefront_shadow_queue_count(resources, queue_index);
      const uint offset = kGPUWavefrontShadowDispatchArgsOffset + queue_index * kGPUWavefrontDispatchArgsStride;
      wavefront_store_dispatch_args(dispatch_args, offset, shadow_count, 1u);
    }
  }

  const uint camera_dielectric_count = wavefront_material_queue_count(resources, true, kGPUWavefrontMaterialQueueDielectric);
  const uint light_dielectric_count = wavefront_material_queue_count(resources, false, kGPUWavefrontMaterialQueueDielectric);
  const uint chunk_count = 1u + ((resources.path_capacity - 1u) / kGPUWavefrontHeavyContinuationChunkSize);
  if (dtid.x >= chunk_count) {
    return;
  }

  const uint item_offset = dtid.x * kGPUWavefrontHeavyContinuationChunkSize;
  const uint camera_chunk_count = (camera_dielectric_count > item_offset) ? min(camera_dielectric_count - item_offset, kGPUWavefrontHeavyContinuationChunkSize) : 0u;
  const uint light_chunk_count = (light_dielectric_count > item_offset) ? min(light_dielectric_count - item_offset, kGPUWavefrontHeavyContinuationChunkSize) : 0u;
  const uint camera_offset = kGPUWavefrontFixedDispatchArgsBufferSize + dtid.x * kGPUWavefrontDispatchArgsStride;
  const uint light_offset = kGPUWavefrontFixedDispatchArgsBufferSize + (chunk_count + dtid.x) * kGPUWavefrontDispatchArgsStride;
  wavefront_store_dispatch_args(dispatch_args, camera_offset, camera_chunk_count, 1u);
  wavefront_store_dispatch_args(dispatch_args, light_offset, light_chunk_count, 1u);
}
