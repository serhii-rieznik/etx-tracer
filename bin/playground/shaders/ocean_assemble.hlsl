// ocean_assemble.hlsl
#include <shaders/bindless.hlsl>

struct AssemblePushConstants {
  uint inHtIndex;
  uint inDxDzIndex;
  uint inDerivSpec0Index;
  uint inDerivSpec1Index;
  uint inDerivSpec2Index;
  uint outDispIndex;
  uint outDerivUIndex;
  uint outDerivVIndex;
  uint outSlopeMetricIndex;
  uint N;
  float lambda;  // choppiness
};
[[vk::push_constant]] AssemblePushConstants assemblePc;

[numthreads(8, 8, 1)] void AssembleMain(uint3 id : SV_DispatchThreadID) {
  if (id.x >= assemblePc.N || id.y >= assemblePc.N)
    return;

  Texture2D<float4> inHt = bindless_textures[NonUniformResourceIndex(assemblePc.inHtIndex)];
  Texture2D<float4> inDxDz = bindless_textures[NonUniformResourceIndex(assemblePc.inDxDzIndex)];
  Texture2D<float4> inDerivSpec0 = bindless_textures[NonUniformResourceIndex(assemblePc.inDerivSpec0Index)];
  Texture2D<float4> inDerivSpec1 = bindless_textures[NonUniformResourceIndex(assemblePc.inDerivSpec1Index)];
  Texture2D<float4> inDerivSpec2 = bindless_textures[NonUniformResourceIndex(assemblePc.inDerivSpec2Index)];
  RWTexture2D<float4> outDisp = bindless_storage_textures[NonUniformResourceIndex(assemblePc.outDispIndex)];
  RWTexture2D<float4> outDerivU = bindless_storage_textures[NonUniformResourceIndex(assemblePc.outDerivUIndex)];
  RWTexture2D<float4> outDerivV = bindless_storage_textures[NonUniformResourceIndex(assemblePc.outDerivVIndex)];
  RWTexture2D<float4> outSlopeMetric = bindless_storage_textures[NonUniformResourceIndex(assemblePc.outSlopeMetricIndex)];

  // IFFT output is unnormalized in this implementation, scale by 1/(N*N) here.
  float inv_n2 = 1.0 / float(assemblePc.N * assemblePc.N);
  float sign = ((id.x + id.y) % 2 == 1) ? -1.0 : 1.0;

  float4 in_ht = inHt.Load(int3(id.xy, 0));
  float4 in_dxdz = inDxDz.Load(int3(id.xy, 0));
  float y = in_ht.x * sign * inv_n2;
  float x = in_dxdz.x * sign * inv_n2;
  float z = in_dxdz.z * sign * inv_n2;

  float3 disp = float3(x * assemblePc.lambda, y, z * assemblePc.lambda);
  outDisp[id.xy] = float4(disp, 1.0);

  float4 deriv0 = inDerivSpec0.Load(int3(id.xy, 0));
  float4 deriv1 = inDerivSpec1.Load(int3(id.xy, 0));
  float4 deriv2 = inDerivSpec2.Load(int3(id.xy, 0));
  float du_x = deriv0.x * sign * inv_n2 * assemblePc.lambda;
  float du_y = deriv0.z * sign * inv_n2;
  float du_z = deriv1.x * sign * inv_n2 * assemblePc.lambda;
  float dv_x = deriv1.z * sign * inv_n2 * assemblePc.lambda;
  float dv_y = deriv2.x * sign * inv_n2;
  float dv_z = deriv2.z * sign * inv_n2 * assemblePc.lambda;

  float3 dPdu = float3(1.0 + du_x, du_y, du_z);
  float3 dPdv = float3(dv_x, dv_y, 1.0 + dv_z);
  outDerivU[id.xy] = float4(dPdu, 0.0);
  outDerivV[id.xy] = float4(dPdv, 0.0);

  float3 area_vec = cross(dPdv, dPdu);
  float area_len2 = dot(area_vec, area_vec);
  float3 n = float3(0.0, 1.0, 0.0);
  if (area_len2 > 1.0e-12) {
    n = area_vec * rsqrt(area_len2);
  }
  float ny = max(abs(n.y), 1.0e-3);
  float slope_energy = ((n.x * n.x) + (n.z * n.z)) / (ny * ny);
  slope_energy = min(slope_energy, 64.0);
  float area_mag = sqrt(max(area_len2, 0.0));
  outSlopeMetric[id.xy] = float4(slope_energy, area_mag, n.y, 0.0);
}
