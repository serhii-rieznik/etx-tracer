#include <shaders/bindless.hlsl>

struct PushConstants {
  uint outputTextureIndex;
  uint width;
  uint height;
  float time;
};

[[vk::push_constant]]
PushConstants pushConstants;

[numthreads(16, 16, 1)]
void CSMain(uint3 gid : SV_DispatchThreadID) {
  if (gid.x >= pushConstants.width || gid.y >= pushConstants.height) {
    return;
  }
  
  RWTexture2D<float4> outputTexture = bindless_storage_textures[NonUniformResourceIndex(pushConstants.outputTextureIndex)];
  
  float2 uv = float2(gid.xy) / float2(pushConstants.width, pushConstants.height);
  float2 centered_uv = uv * 2.0f - 1.0f;
  
  // Combine multiple waves for complexity
  float pattern = sin(uv.x * 10.0f + pushConstants.time) * cos(uv.y * 10.0f - pushConstants.time * 0.5f);
  pattern += 0.5f * sin(uv.x * 20.0f - pushConstants.time * 1.5f);
  pattern += 0.3f * sin(length(centered_uv) * 30.0f - pushConstants.time * 2.0f);
  
  float3 color = 0.5f + 0.5f * cos(pushConstants.time + float3(uv.x, uv.y, uv.x) + float3(0, 2, 4));
  color *= (0.5f + 0.5f * pattern);
  
  // Add a moving grid to verify pixel-perfect writes
  float2 grid = frac(uv * 10.0f + pushConstants.time * 0.1f);
  float grid_line = step(0.95f, grid.x) + step(0.95f, grid.y);
  color += grid_line * 0.2f;

  outputTexture[gid.xy] = float4(color, 1.0f);
}
