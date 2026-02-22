// ocean_assemble.hlsl
#include <shaders/bindless.hlsl>

struct AssemblePushConstants {
    uint inHtIndex;
    uint inDxDzIndex;
    uint outDispIndex;
    uint outNormalIndex;
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
    RWTexture2D<float4> outNorm = bindless_storage_textures[NonUniformResourceIndex(assemblePc.outNormalIndex)];
    
    // IFFT output is unnormalized in this implementation, scale by 1/(N*N) here.
    float inv_n2 = 1.0 / float(assemblePc.N * assemblePc.N);
    float sign = ((id.x + id.y) % 2 == 1) ? -1.0 : 1.0;
    
    float y = inHt[id.xy].x * sign * inv_n2;
    float x = inDxDz[id.xy].x * sign * inv_n2;
    float z = inDxDz[id.xy].z * sign * inv_n2;
    
    float3 disp = float3(x * assemblePc.lambda, y, z * assemblePc.lambda);
    outDisp[id.xy] = float4(disp, 1.0);
    
    // To calculate normal, we can use finite differences on the displacement map.
    // However, wait for the next frame for neighboring pixels? We can use group shared memory,
    // or simply read the current frame's neighbors from the input textures.
    int2 left = int2((id.x - 1 + assemblePc.N) % assemblePc.N, id.y);
    int2 right = int2((id.x + 1) % assemblePc.N, id.y);
    int2 up = int2(id.x, (id.y - 1 + assemblePc.N) % assemblePc.N);
    int2 down = int2(id.x, (id.y + 1) % assemblePc.N);
    
    float2 s_left = ((left.x + left.y) % 2 == 1 ? -1.0 : 1.0) * float2(inDxDz[left].x * assemblePc.lambda * inv_n2, inHt[left].x * inv_n2);
    float2 s_right = ((right.x + right.y) % 2 == 1 ? -1.0 : 1.0) * float2(inDxDz[right].x * assemblePc.lambda * inv_n2, inHt[right].x * inv_n2);
    float3 d_left = float3(s_left.x, s_left.y, inDxDz[left].z * ((left.x + left.y) % 2 == 1 ? -1.0 : 1.0) * assemblePc.lambda * inv_n2);
    float3 d_right = float3(s_right.x, s_right.y, inDxDz[right].z * ((right.x + right.y) % 2 == 1 ? -1.0 : 1.0) * assemblePc.lambda * inv_n2);
    
    float2 s_up = ((up.x + up.y) % 2 == 1 ? -1.0 : 1.0) * float2(inDxDz[up].z * assemblePc.lambda * inv_n2, inHt[up].x * inv_n2);
    float2 s_down = ((down.x + down.y) % 2 == 1 ? -1.0 : 1.0) * float2(inDxDz[down].z * assemblePc.lambda * inv_n2, inHt[down].x * inv_n2);
    float3 d_up = float3(inDxDz[up].x * ((up.x + up.y) % 2 == 1 ? -1.0 : 1.0) * assemblePc.lambda * inv_n2, s_up.y, s_up.x);
    float3 d_down = float3(inDxDz[down].x * ((down.x + down.y) % 2 == 1 ? -1.0 : 1.0) * assemblePc.lambda * inv_n2, s_down.y, s_down.x);
    
    // Scale size by world-space quad size for this cascade.
    float quad_size = assemblePc.L / float(assemblePc.N);
    
    float3 p_left = float3(-quad_size, 0.0, 0.0) + d_left;
    float3 p_right = float3(quad_size, 0.0, 0.0) + d_right;
    float3 p_up = float3(0.0, 0.0, -quad_size) + d_up;
    float3 p_down = float3(0.0, 0.0, quad_size) + d_down;
    
    float3 tangentX = p_right - p_left;
    float3 tangentZ = p_down - p_up;
    
    float3 normal = float3(0.0, 1.0, 0.0);
    float3 n = cross(tangentZ, tangentX);
    float n2 = dot(n, n);
    if (n2 > 1e-20) {
        normal = normalize(n);
    }
    outNorm[id.xy] = float4(normal, 0.0);
}
