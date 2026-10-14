#pragma once

#include <interop/gpu_rt_shared.hxx>
#include <interop/gpu_wavefront_abi.hxx>

[[vk::push_constant]] GPURTConstants constants;
#include "bindless.hlsl"
#include "gpu_spectral_tables.hlsl"
#include <interop/gpu_abi_access_shared.hxx>
#include <interop/scene_gpu_access_shared.hxx>

#define WAVEFRONT_RW_BUFFER(descriptor_index) bindless_rw_buffers[NonUniformResourceIndex(descriptor_index)]
#define WAVEFRONT_RO_BUFFER(descriptor_index) bindless_buffers[NonUniformResourceIndex(descriptor_index)]

float2 wavefront_load_float2(ByteAddressBuffer buffer, uint byte_offset) {
  return asfloat(buffer.Load2(byte_offset));
}

float3 wavefront_load_float3(ByteAddressBuffer buffer, uint byte_offset) {
  return asfloat(buffer.Load3(byte_offset));
}

float4 wavefront_load_float4(ByteAddressBuffer buffer, uint byte_offset) {
  return asfloat(buffer.Load4(byte_offset));
}

void wavefront_store_float2(RWByteAddressBuffer buffer, uint byte_offset, float2 value) {
  buffer.Store2(byte_offset, asuint(value));
}

void wavefront_store_float3(RWByteAddressBuffer buffer, uint byte_offset, float3 value) {
  buffer.Store3(byte_offset, asuint(value));
}

void wavefront_store_float4(RWByteAddressBuffer buffer, uint byte_offset, float4 value) {
  buffer.Store4(byte_offset, asuint(value));
}

GPUWavefrontResources wavefront_load_resources() {
  ByteAddressBuffer buffer = WAVEFRONT_RO_BUFFER(constants.wavefront_buffer_index);
  GPUWavefrontResources result = (GPUWavefrontResources)0;
  result.camera_state_buffer = buffer.Load(kGPUWavefrontResourcesCameraStateBufferOffset);
  result.light_state_buffer = buffer.Load(kGPUWavefrontResourcesLightStateBufferOffset);
  result.camera_hit_buffer = buffer.Load(kGPUWavefrontResourcesCameraHitBufferOffset);
  result.light_hit_buffer = buffer.Load(kGPUWavefrontResourcesLightHitBufferOffset);
  result.camera_queue_a_buffer = buffer.Load(kGPUWavefrontResourcesCameraQueueABufferOffset);
  result.camera_queue_b_buffer = buffer.Load(kGPUWavefrontResourcesCameraQueueBBufferOffset);
  result.light_queue_a_buffer = buffer.Load(kGPUWavefrontResourcesLightQueueABufferOffset);
  result.light_queue_b_buffer = buffer.Load(kGPUWavefrontResourcesLightQueueBBufferOffset);
  result.camera_vertex_buffer = buffer.Load(kGPUWavefrontResourcesCameraVertexBufferOffset);
  result.light_vertex_buffer = buffer.Load(kGPUWavefrontResourcesLightVertexBufferOffset);
  result.film_buffer = buffer.Load(kGPUWavefrontResourcesFilmBufferOffset);
  result.path_meta_buffer = buffer.Load(kGPUWavefrontResourcesPathMetaBufferOffset);
  result.direct_light_sample_buffer = buffer.Load(kGPUWavefrontResourcesDirectLightSampleBufferOffset);
  result.direct_light_task_buffer = buffer.Load(kGPUWavefrontResourcesDirectLightTaskBufferOffset);
  result.direct_light_result_buffer = buffer.Load(kGPUWavefrontResourcesDirectLightResultBufferOffset);
  result.connect_light_task_buffer = buffer.Load(kGPUWavefrontResourcesConnectLightTaskBufferOffset);
  result.connect_light_result_buffer = buffer.Load(kGPUWavefrontResourcesConnectLightResultBufferOffset);
  result.connect_camera_task_buffer = buffer.Load(kGPUWavefrontResourcesConnectCameraTaskBufferOffset);
  result.connect_camera_result_buffer = buffer.Load(kGPUWavefrontResourcesConnectCameraResultBufferOffset);
  result.camera_subsurface_state_buffer = buffer.Load(kGPUWavefrontResourcesCameraSubsurfaceStateBufferOffset);
  result.light_subsurface_state_buffer = buffer.Load(kGPUWavefrontResourcesLightSubsurfaceStateBufferOffset);
  result.path_capacity = buffer.Load(kGPUWavefrontResourcesPathCapacityOffset);
  result.max_path_length = buffer.Load(kGPUWavefrontResourcesMaxPathLengthOffset);
  result.camera_vertex_capacity = buffer.Load(kGPUWavefrontResourcesCameraVertexCapacityOffset);
  result.light_vertex_capacity = buffer.Load(kGPUWavefrontResourcesLightVertexCapacityOffset);
  result.camera_fixed_max_bounces = buffer.Load(kGPUWavefrontResourcesCameraFixedMaxBouncesOffset);
  result.light_fixed_max_bounces = buffer.Load(kGPUWavefrontResourcesLightFixedMaxBouncesOffset);
  result.dispatch_args_buffer = buffer.Load(kGPUWavefrontResourcesDispatchArgsBufferOffset);
  result.material_queue_buffer = buffer.Load(kGPUWavefrontResourcesMaterialQueueBufferOffset);
  result.shadow_queue_buffer = buffer.Load(kGPUWavefrontResourcesShadowQueueBufferOffset);
  result.light_vertex_counter_buffer = buffer.Load(kGPUWavefrontResourcesLightVertexCounterBufferOffset);
  result.fast_light_endpoint_buffer = buffer.Load(kGPUWavefrontResourcesFastLightEndpointBufferOffset);
  result.vcm_grid_heads_buffer = buffer.Load(kGPUWavefrontResourcesVCMGridHeadsBufferOffset);
  result.vcm_grid_next_buffer = buffer.Load(kGPUWavefrontResourcesVCMGridNextBufferOffset);
  result.upbp_resources_buffer = buffer.Load(kGPUWavefrontResourcesUPBPResourcesBufferOffset);
  result.coating_request_buffer = buffer.Load(kGPUWavefrontResourcesCoatingRequestBufferOffset);
  result.emission_request_buffer = buffer.Load(kGPUWavefrontResourcesEmissionRequestBufferOffset);
  result.emission_consumer_buffer = buffer.Load(kGPUWavefrontResourcesEmissionConsumerBufferOffset);
  result.emission_request_stride = buffer.Load(kGPUWavefrontResourcesEmissionRequestStrideOffset);
  result.transport_work_buffer = buffer.Load(kGPUWavefrontResourcesTransportWorkBufferOffset);
  result.vcm_coating_buffer = buffer.Load(kGPUWavefrontResourcesVCMCoatingBufferOffset);
  return result;
}

uint wavefront_queue_current_descriptor(bool from_camera) {
  GPUWavefrontResources resources = wavefront_load_resources();
  bool even_iteration = (constants.path_iteration & 1u) == 0u;
  if (from_camera) {
    return even_iteration ? resources.camera_queue_a_buffer : resources.camera_queue_b_buffer;
  }
  return even_iteration ? resources.light_queue_a_buffer : resources.light_queue_b_buffer;
}

uint wavefront_queue_next_descriptor(bool from_camera) {
  GPUWavefrontResources resources = wavefront_load_resources();
  bool even_iteration = (constants.path_iteration & 1u) == 0u;
  if (from_camera) {
    return even_iteration ? resources.camera_queue_b_buffer : resources.camera_queue_a_buffer;
  }
  return even_iteration ? resources.light_queue_b_buffer : resources.light_queue_a_buffer;
}

uint wavefront_queue_count(uint descriptor_index) {
  if (descriptor_index == kInvalidIndex) {
    return 0u;
  }
  return WAVEFRONT_RO_BUFFER(descriptor_index).Load(kGPUWavefrontQueueCountOffset);
}

void wavefront_queue_reset(uint descriptor_index) {
  if (descriptor_index == kInvalidIndex) {
    return;
  }
  RWByteAddressBuffer buffer = WAVEFRONT_RW_BUFFER(descriptor_index);
  buffer.Store(kGPUWavefrontQueueCountOffset, 0u);
  buffer.Store(kGPUWavefrontQueueMaxPathLengthOffset, 0u);
  buffer.Store(kGPUWavefrontQueuePad1Offset, 0u);
  buffer.Store(kGPUWavefrontQueuePad2Offset, 0u);
}

void wavefront_queue_store(uint descriptor_index, uint slot, uint value) {
  WAVEFRONT_RW_BUFFER(descriptor_index).Store(kGPUWavefrontQueueIndicesOffset + slot * 4u, value);
}

uint wavefront_queue_load(uint descriptor_index, uint slot) {
  return WAVEFRONT_RO_BUFFER(descriptor_index).Load(kGPUWavefrontQueueIndicesOffset + slot * 4u);
}

uint wavefront_queue_append(uint descriptor_index, uint value) {
  RWByteAddressBuffer buffer = WAVEFRONT_RW_BUFFER(descriptor_index);
  uint slot = 0u;
  buffer.InterlockedAdd(kGPUWavefrontQueueCountOffset, 1u, slot);
  buffer.Store(kGPUWavefrontQueueIndicesOffset + slot * 4u, value);
  return slot;
}

void wavefront_queue_update_max_path_length(uint descriptor_index, uint path_length) {
  if (descriptor_index == kInvalidIndex) {
    return;
  }
  uint original = 0u;
  WAVEFRONT_RW_BUFFER(descriptor_index).InterlockedMax(kGPUWavefrontQueueMaxPathLengthOffset, path_length, original);
}

uint2 wavefront_render_window_origin() {
  return uint2(constants.render_window_origin_x, constants.render_window_origin_y);
}

uint2 wavefront_render_window_size() {
  return uint2(constants.render_window_width, constants.render_window_height);
}

bool wavefront_render_window_contains(uint2 local_pixel) {
  return all(local_pixel < wavefront_render_window_size());
}

uint2 wavefront_output_pixel(uint2 local_pixel) {
  return wavefront_render_window_origin() + local_pixel;
}

uint wavefront_render_window_local_index(uint2 local_pixel) {
  uint2 render_window_size = wavefront_render_window_size();
  return local_pixel.x + local_pixel.y * render_window_size.x;
}

float4 wavefront_film_load(uint pixel_index) {
  ByteAddressBuffer buffer = WAVEFRONT_RO_BUFFER(wavefront_load_resources().film_buffer);
  return wavefront_load_float4(buffer, pixel_index * 16u);
}

void wavefront_film_store(uint pixel_index, float4 value) {
  RWByteAddressBuffer buffer = WAVEFRONT_RW_BUFFER(wavefront_load_resources().film_buffer);
  wavefront_store_float4(buffer, pixel_index * 16u, value);
}

#include "gpu_rt_wavefront_work_queue.hlsl"
