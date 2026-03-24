#include <shaders/bindless.hlsl>

struct PushConstants {
  uint texture_index;
  uint sampler_index;
  float2 center_ndc;
  float2 half_size_ndc;
  float2 pad0;
  float4 tint;
};

[[vk::push_constant]]
PushConstants push_constants;

struct VSOutput {
  float4 position : SV_Position;
  float2 uv       : TEXCOORD0;
};

VSOutput VSMain(uint vertex_id : SV_VertexID) {
  static const uint k_indices[6] = {0u, 1u, 2u, 2u, 1u, 3u};
  static const float2 k_corners[4] = {
    float2(-1.0f, -1.0f),
    float2(1.0f, -1.0f),
    float2(-1.0f, 1.0f),
    float2(1.0f, 1.0f),
  };

  uint corner_index = k_indices[min(vertex_id, 5u)];
  float2 corner = k_corners[corner_index];

  VSOutput output;
  output.position = float4(push_constants.center_ndc + (corner * push_constants.half_size_ndc), 0.0f, 1.0f);
  output.uv = corner * 0.5f + 0.5f;
  return output;
}

float4 PSMain(VSOutput input) : SV_Target0 {
  float2 p = input.uv * 2.0f - 1.0f;
  float r2 = dot(p, p);
  if (r2 > 1.0f) {
    return float4(0.0f, 0.0f, 0.0f, 0.0f);
  }

  float edge_mask = 1.0f - smoothstep(0.94f, 1.0f, sqrt(r2));
  Texture2D sun_texture = bindless_textures[NonUniformResourceIndex(push_constants.texture_index)];
  SamplerState sun_sampler = bindless_samplers[NonUniformResourceIndex(push_constants.sampler_index)];
  float3 sun_rgb = sun_texture.SampleLevel(sun_sampler, input.uv, 0.0f).rgb * push_constants.tint.rgb * edge_mask;
  return float4(sun_rgb, edge_mask);
}
