#include "bindless.hlsl"
#include <interop/imgui_shared.hxx>

[[vk::push_constant]] ImGuiPushConstants pushConstants;

struct VertexToPixel {
  float4 position : SV_Position;
  float2 texcoord : TEXCOORD0;
  float4 color    : COLOR0;
};

float srgb_to_linear_channel(float value) {
  return value <= 0.04045 ? value / 12.92 : pow((value + 0.055) / 1.055, 2.4);
}

float3 srgb_to_linear(float3 value) {
  return float3(srgb_to_linear_channel(value.x), srgb_to_linear_channel(value.y), srgb_to_linear_channel(value.z));
}

VertexToPixel vs_main(uint vertex_id : SV_VertexID) {
  ByteAddressBuffer vertex_buffer = bindless_buffers[pushConstants.vertex_buffer_index];
  uint offset = vertex_id * 20;
  uint4 data0 = vertex_buffer.Load4(offset);
  uint data1 = vertex_buffer.Load(offset + 16);

  VertexToPixel output;
  output.position = float4(asfloat(data0.xy) * pushConstants.scale + pushConstants.translate, 0.0, 1.0);
  output.texcoord = asfloat(data0.zw);
  output.color = float4(((data1 >> 0u) & 0xFF) / 255.0, ((data1 >> 8u) & 0xFF) / 255.0, ((data1 >> 16) & 0xFF) / 255.0, ((data1 >> 24) & 0xFF) / 255.0);
#if defined(ETX_IMGUI_SRGB_TARGET)
  // ImGui stores style and packed vertex colors in display-referred sRGB.
  // Decode them before linear blending into an sRGB render attachment.
  output.color.rgb = srgb_to_linear(output.color.rgb);
#endif
  return output;
}

float4 ps_main(VertexToPixel input) : SV_Target {
  float4 tex_color = SampleTexture(pushConstants.texture_index, pushConstants.sampler_index, input.texcoord);
  return input.color * tex_color;
}
