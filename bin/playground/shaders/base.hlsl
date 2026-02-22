#include <interop/geometry.hxx>
#include <shaders/bindless.hlsl>

// Bindless resource passing through push constants
struct PushConstants {
  uint vertexBufferIndex;
  uint testTextureIndex;
  uint testSamplerIndex;
  uint padding;
  column_major float4x4 mvp;
};

[[vk::push_constant]]
PushConstants pushConstants;

struct VSOutput {
  float4 position : SV_Position;
  float2 uv : TEXCOORD;
};

VSOutput VSMain(uint vertexId : SV_VertexID) {
  VSOutput output;
  
  // Access data through bindless buffer array
  ByteAddressBuffer vb = bindless_buffers[NonUniformResourceIndex(pushConstants.vertexBufferIndex)];
  
  // Load using shared ABI 
  Vertex v = vb.Load<Vertex>(vertexId * 56);
  
  output.position = mul(pushConstants.mvp, float4(v.pos, 1.0f));
  output.uv = v.tex;
  
  return output;
}

float4 PSMain(VSOutput input) : SV_Target0 {
  Texture2D testTex = bindless_textures[NonUniformResourceIndex(pushConstants.testTextureIndex)];
  SamplerState testSampler = bindless_samplers[NonUniformResourceIndex(pushConstants.testSamplerIndex)];
  return testTex.Sample(testSampler, input.uv);
}
