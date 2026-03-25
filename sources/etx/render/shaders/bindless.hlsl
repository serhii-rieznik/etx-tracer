#pragma once

enum SamplerType : uint {
  LinearRepeat = 0,
  LinearClamp = 1,
  NearestRepeat = 2,
  NearestClamp = 3,
  LinearRepeatUClampV = 4,
};

[[vk::binding(0, 0)]] ByteAddressBuffer bindless_buffers[];
[[vk::binding(1, 0)]] Texture2D bindless_textures[];
[[vk::binding(2, 0)]] SamplerState bindless_samplers[];
[[vk::binding(3, 0)]] RWTexture2D<float4> bindless_storage_textures[];
[[vk::binding(4, 0)]] RaytracingAccelerationStructure bindless_accel_structs[];
[[vk::binding(5, 0)]] RWByteAddressBuffer bindless_rw_buffers[];

float4 SampleTexture(uint texture_index, uint sampler_index, float2 uv) {
  return bindless_textures[texture_index].Sample(bindless_samplers[sampler_index], uv);
}
