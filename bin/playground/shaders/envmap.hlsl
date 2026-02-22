#include <shaders/bindless.hlsl>

struct PushConstants {
  uint envmapIndex;
  uint samplerIndex;
  uint2 pad;
  column_major float4x4 invViewProj;
};

[[vk::push_constant]]
PushConstants pushConstants;

struct VSOutput {
  float4 position : SV_Position;
  float2 ndc : TEXCOORD;
};

VSOutput VSMain(uint vertexId : SV_VertexID) {
  VSOutput output;
  // Fullscreen triangle from vertex ID — no vertex buffer needed
  // vertex 0: (-1,-1)  vertex 1: (3,-1)  vertex 2: (-1, 3)
  float2 uv = float2((vertexId << 1u) & 2u, vertexId & 2u);
  float2 pos = uv * 2.0f - 1.0f;
  output.position = float4(pos, 0.0f, 1.0f);
  output.ndc = pos;
  return output;
}

static const float kPi = 3.14159265358979f;

float4 PSMain(VSOutput input) : SV_Target0 {
  // Reconstruct world-space direction from NDC via inverse view-projection
  float4 world = mul(pushConstants.invViewProj, float4(input.ndc, 1.0f, 1.0f));
  float3 dir = normalize(world.xyz / world.w);

  // Equirectangular mapping
  float u = atan2(dir.z, dir.x) / (2.0f * kPi) + 0.5f;
  float v = acos(clamp(dir.y, -1.0f, 1.0f)) / kPi;

  Texture2D envmap = bindless_textures[NonUniformResourceIndex(pushConstants.envmapIndex)];
  SamplerState s = bindless_samplers[NonUniformResourceIndex(pushConstants.samplerIndex)];
  return envmap.SampleLevel(s, float2(u, v), 0.0f);
}
