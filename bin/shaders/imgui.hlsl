#include "bindless.hlsl"

struct VertexToPixel {
  float4 position : SV_Position;
  float2 texcoord : TEXCOORD0;
  float4 color : COLOR0;
};

struct ImGuiPushConstants {
  float2 scale;
  float2 translate;
  uint vertex_buffer_index;
  uint texture_index;
  uint sampler_index;
  uint padding;
};

[[vk::push_constant]] ImGuiPushConstants pushConstants;

struct ImDrawVert {
  float2 pos;
  float2 uv;
  uint col;
};

VertexToPixel vs_main(uint vertex_id : SV_VertexID) {
  ByteAddressBuffer vertex_buffer = bindless_buffers[pushConstants.vertex_buffer_index];
  uint offset = vertex_id * 20;
  uint4 data0 = vertex_buffer.Load4(offset);
  uint data1 = vertex_buffer.Load(offset + 16);

  VertexToPixel output;
  output.position = float4(asfloat(data0.xy) * pushConstants.scale + pushConstants.translate, 0.0, 1.0);
  output.texcoord = asfloat(data0.zw);
  output.color = float4(
    (data1 & 0xFF) / 255.0,
    ((data1 >> 8) & 0xFF) / 255.0,
    ((data1 >> 16) & 0xFF) / 255.0,
    ((data1 >> 24) & 0xFF) / 255.0
  );
  return output;
}

float4 ps_main(VertexToPixel input) : SV_Target {
  float4 tex_color = SampleTexture(pushConstants.texture_index, pushConstants.sampler_index, input.texcoord);
  return input.color * tex_color;
}