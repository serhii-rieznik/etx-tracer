// ocean_assemble.hlsl
#include <shaders/bindless.hlsl>

struct AssemblePushConstants {
    uint inHtIndex;
    uint inDxDzIndex;
    uint outDispIndex;
    uint outDerivUIndex;
    uint outDerivVIndex;
    uint outSlopeMetricIndex;
    uint N;
    float lambda; // choppiness
    float L;
};
[[vk::push_constant]] AssemblePushConstants assemblePc;

[numthreads(8, 8, 1)]
void AssembleMain(uint3 id : SV_DispatchThreadID) {
    if (id.x >= assemblePc.N || id.y >= assemblePc.N) return;
    
    RWTexture2D<float4> inHt = bindless_storage_textures[NonUniformResourceIndex(assemblePc.inHtIndex)];
    RWTexture2D<float4> inDxDz = bindless_storage_textures[NonUniformResourceIndex(assemblePc.inDxDzIndex)];
    RWTexture2D<float4> outDisp = bindless_storage_textures[NonUniformResourceIndex(assemblePc.outDispIndex)];
    RWTexture2D<float4> outDerivU = bindless_storage_textures[NonUniformResourceIndex(assemblePc.outDerivUIndex)];
    RWTexture2D<float4> outDerivV = bindless_storage_textures[NonUniformResourceIndex(assemblePc.outDerivVIndex)];
    RWTexture2D<float4> outSlopeMetric = bindless_storage_textures[NonUniformResourceIndex(assemblePc.outSlopeMetricIndex)];
    
    // IFFT output is unnormalized in this implementation, scale by 1/(N*N) here.
    float inv_n2 = 1.0 / float(assemblePc.N * assemblePc.N);
    float sign = ((id.x + id.y) % 2 == 1) ? -1.0 : 1.0;
    
    float y = inHt[id.xy].x * sign * inv_n2;
    float x = inDxDz[id.xy].x * sign * inv_n2;
    float z = inDxDz[id.xy].z * sign * inv_n2;
    
    float3 disp = float3(x * assemblePc.lambda, y, z * assemblePc.lambda);
    outDisp[id.xy] = float4(disp, 1.0);
    
    int2 left = int2((id.x - 1 + assemblePc.N) % assemblePc.N, id.y);
    int2 right = int2((id.x + 1) % assemblePc.N, id.y);
    int2 up = int2(id.x, (id.y - 1 + assemblePc.N) % assemblePc.N);
    int2 down = int2(id.x, (id.y + 1) % assemblePc.N);

    float sign_left = ((left.x + left.y) % 2 == 1) ? -1.0 : 1.0;
    float sign_right = ((right.x + right.y) % 2 == 1) ? -1.0 : 1.0;
    float sign_up = ((up.x + up.y) % 2 == 1) ? -1.0 : 1.0;
    float sign_down = ((down.x + down.y) % 2 == 1) ? -1.0 : 1.0;

    float3 disp_left = float3(
      inDxDz[left].x * sign_left * inv_n2 * assemblePc.lambda,
      inHt[left].x * sign_left * inv_n2,
      inDxDz[left].z * sign_left * inv_n2 * assemblePc.lambda);
    float3 disp_right = float3(
      inDxDz[right].x * sign_right * inv_n2 * assemblePc.lambda,
      inHt[right].x * sign_right * inv_n2,
      inDxDz[right].z * sign_right * inv_n2 * assemblePc.lambda);
    float3 disp_up = float3(
      inDxDz[up].x * sign_up * inv_n2 * assemblePc.lambda,
      inHt[up].x * sign_up * inv_n2,
      inDxDz[up].z * sign_up * inv_n2 * assemblePc.lambda);
    float3 disp_down = float3(
      inDxDz[down].x * sign_down * inv_n2 * assemblePc.lambda,
      inHt[down].x * sign_down * inv_n2,
      inDxDz[down].z * sign_down * inv_n2 * assemblePc.lambda);

    float quad_size = assemblePc.L / float(assemblePc.N);
    float inv_double_quad = 1.0 / max(2.0 * quad_size, 1.0e-6);
    float3 d_disp_du = (disp_right - disp_left) * inv_double_quad;
    float3 d_disp_dv = (disp_down - disp_up) * inv_double_quad;

    float3 dPdu = float3(1.0, 0.0, 0.0) + d_disp_du;
    float3 dPdv = float3(0.0, 0.0, 1.0) + d_disp_dv;
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
