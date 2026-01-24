#include "common.hlsl"

float4 main(VertexOutput input)
  : SV_Target {
  return input.color;
}