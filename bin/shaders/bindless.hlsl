#pragma once

enum SamplerType : uint {
  LinearRepeat = 0,
  LinearClamp = 1,
  NearestRepeat = 2,
  NearestClamp = 3,
};

[[vk::binding(0, 0)]] ByteAddressBuffer bindless_buffers[];
[[vk::binding(1, 0)]] Texture2D bindless_textures[];
[[vk::binding(2, 0)]] SamplerState bindless_samplers[];
[[vk::binding(3, 0)]] RWTexture2D<float4> bindless_storage_textures[];

float4 SampleTexture(uint texture_index, uint sampler_index, float2 uv) {
  return bindless_textures[texture_index].Sample(bindless_samplers[sampler_index], uv);
}
