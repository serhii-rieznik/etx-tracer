#include <shaders/bindless.hlsl>

struct PushConstants {
  uint hdrTextureIndex;
  uint samplerIndex;
  float exposure;
  uint outputGammaEncode;
};

[[vk::push_constant]]
PushConstants pushConstants;

struct VSOutput {
  float4 position : SV_Position;
  float2 uv : TEXCOORD;
};

VSOutput VSMain(uint vertexId : SV_VertexID) {
  VSOutput output;
  float2 uv = float2((vertexId << 1u) & 2u, vertexId & 2u);
  float2 pos = uv * 2.0f - 1.0f;
  output.position = float4(pos, 0.0f, 1.0f);
  output.uv = float2(uv.x, 1.0f - uv.y);
  return output;
}

float3 aces_fitted(float3 color) {
  const float a = 2.51f;
  const float b = 0.03f;
  const float c = 2.43f;
  const float d = 0.59f;
  const float e = 0.14f;
  float3 mapped = (color * (a * color + b)) / (color * (c * color + d) + e);
  return saturate(mapped);
}

float4 PSMain(VSOutput input) : SV_Target0 {
  Texture2D hdr_texture = bindless_textures[NonUniformResourceIndex(pushConstants.hdrTextureIndex)];
  SamplerState s = bindless_samplers[NonUniformResourceIndex(pushConstants.samplerIndex)];

  float3 hdr_color = hdr_texture.SampleLevel(s, input.uv, 0.0f).rgb;
  float exposure = max(pushConstants.exposure, 0.0f);
  float3 linear_color = aces_fitted(hdr_color * exposure);

  if (pushConstants.outputGammaEncode != 0u) {
    linear_color = pow(max(linear_color, 0.0f), 1.0f / 2.2f);
  }

  return float4(linear_color, 1.0f);
}
