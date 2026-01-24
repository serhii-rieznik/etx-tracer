#include "common.hlsl"

VertexOutput main(VertexInput input) {
  VertexOutput output;
  output.position = float4(input.position, 1.0);
  output.color = input.color;
  return output;
}