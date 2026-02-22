// ocean_fft.hlsl
#include <shaders/bindless.hlsl>

struct FFTPushConstants {
    uint inTexIndex;
    uint outTexIndex;
    uint N;
    uint pass;
    uint direction; // 0 for horizontal, 1 for vertical
    uint log2N;
};
[[vk::push_constant]] FFTPushConstants fftPc;

uint reverseBits(uint x, uint bits) {
    uint res = 0;
    for (uint i = 0; i < bits; i++) {
        res = (res << 1) | (x & 1);
        x >>= 1;
    }
    return res;
}

float2 complexMultiply(float2 a, float2 b) {
    return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

float2 complexExp(float theta) {
    return float2(cos(theta), sin(theta));
}

[numthreads(8, 8, 1)]
void FFTMain(uint3 id : SV_DispatchThreadID) {
    RWTexture2D<float4> inTex = bindless_storage_textures[NonUniformResourceIndex(fftPc.inTexIndex)];
    RWTexture2D<float4> outTex = bindless_storage_textures[NonUniformResourceIndex(fftPc.outTexIndex)];
    
    uint n = fftPc.N;
    if (id.x >= n || id.y >= n) return;
    
    uint k = (fftPc.direction == 0) ? id.x : id.y;
    uint other = (fftPc.direction == 0) ? id.y : id.x;
    
    uint p = fftPc.pass;
    uint half_len = 1 << p;
    uint len = half_len << 1;
    uint group = k / len;
    uint t = k % half_len;
    
    uint in_idx1, in_idx2;
    if (p == 0) {
        in_idx1 = reverseBits(group * len + t, fftPc.log2N);
        in_idx2 = reverseBits(group * len + t + half_len, fftPc.log2N);
    } else {
        in_idx1 = group * len + t;
        in_idx2 = group * len + t + half_len;
    }
    
    uint2 uv1 = (fftPc.direction == 0) ? uint2(in_idx1, other) : uint2(other, in_idx1);
    uint2 uv2 = (fftPc.direction == 0) ? uint2(in_idx2, other) : uint2(other, in_idx2);
    
    float4 p1 = inTex[uv1];
    float4 p2 = inTex[uv2];
    
    float angle = 2.0 * 3.1415926535 * float(t) / float(len); 
    float2 twiddle = complexExp(angle);
    
    float2 p2_xy_twiddled = complexMultiply(twiddle, p2.xy);
    float2 p2_zw_twiddled = complexMultiply(twiddle, p2.zw);
    
    float4 res;
    if (k % len < half_len) {
        res = float4(p1.xy + p2_xy_twiddled, p1.zw + p2_zw_twiddled);
    } else {
        res = float4(p1.xy - p2_xy_twiddled, p1.zw - p2_zw_twiddled);
    }
    
    outTex[id.xy] = res;
}
