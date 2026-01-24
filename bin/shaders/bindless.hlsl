// Common bindless declarations for all shaders
// This file defines the bindless descriptor set layout used across all pipelines

#ifndef BINDLESS_HLSL
#define BINDLESS_HLSL

// Bindless descriptor set (set = 0, always bound at slot 0)
#define BINDLESS_SET 0

// All bindless resources use the same space (descriptor set 0)
#define BINDLESS_SPACE space0

// Pre-defined sampler types (must match RHISamplerType enum)
enum SamplerType : uint {
  LinearRepeat = 0,
  LinearClamp = 1,
  NearestRepeat = 2,
  NearestClamp = 3,
};

// Buffer resources (raw byte access for maximum flexibility)
[[vk::binding(0, 0)]] ByteAddressBuffer bindless_buffers[];

// Texture resources
[[vk::binding(1, 0)]] Texture2D bindless_textures[];

// Sampler resources
[[vk::binding(2, 0)]] SamplerState bindless_samplers[];

// Storage textures
[[vk::binding(3, 0)]] RWTexture2D<float4> bindless_storage_textures[];

// Helper functions to access bindless resources
float4 SampleTexture(uint texture_index, uint sampler_index, float2 uv) {
  return bindless_textures[texture_index].Sample(bindless_samplers[sampler_index], uv);
}

#endif  // BINDLESS_HLSL