#include <shaders/bindless.hlsl>
// Hash functions for random numbers
uint hash(uint state) {
  state ^= 2747636419u;
  state *= 2654435769u;
  state ^= state >> 16;
  state *= 2654435769u;
  state ^= state >> 16;
  state *= 2654435769u;
  return state;
}

float random(inout uint state) {
  state = hash(state + 1u);
  return float(state) / 4294967295.0;  // 2^32 - 1
}

// Box-Muller transform for N(0, 1) normally distributed random numbers
float2 gaussianRandom(inout uint state) {
  float u1 = random(state);
  float u2 = random(state);

  // Avoid log(0)
  u1 = max(u1, 1e-6);

  float r = sqrt(-2.0 * log(u1));
  float theta = 2.0 * 3.1415926535 * u2;

  return float2(r * cos(theta), r * sin(theta));
}

struct OceanComputePushConstants {
  uint outH0Index;
  uint inH0Index;
  uint outHtIndex;    // ht_Dx_Dz
  uint outDxDzIndex;  // to store Choppy X and Z
  uint outDerivSpec0Index;
  uint outDerivSpec1Index;
  uint outDerivSpec2Index;
  uint N;
  uint seed;
  float L;
  float A;
  float windDirX;
  float windDirY;
  float windSpeed;
  float waterDepth;
  float jonswapGamma;
  float directionalSpread;
  float time;
  float kMin;
  float kMax;
  float kMinSoft;
  float kMaxSoft;
  float choppiness;
};
[[vk::push_constant]] OceanComputePushConstants pushConstants;

static const float k_surface_tension_over_density = 7.28e-5;

float dispersionOmega(float k_len, float water_depth) {
  if (k_len <= 1e-6) {
    return 0.0;
  }
  float safe_depth = max(water_depth, 1e-3);
  float kh = k_len * safe_depth;
  float restoring = (9.81 * k_len) + (k_surface_tension_over_density * k_len * k_len * k_len);
  return sqrt(max(restoring * tanh(kh), 0.0));
}

float dispersionDOmegaDK(float k_len, float water_depth, float omega) {
  if ((k_len <= 1e-6) || (omega <= 1e-6)) {
    return 0.0;
  }
  float safe_depth = max(water_depth, 1e-3);
  float kh = k_len * safe_depth;
  float tanh_kh = tanh(kh);
  float sech_kh = 1.0 / cosh(kh);
  float sech2_kh = sech_kh * sech_kh;
  float k2 = k_len * k_len;
  float restoring = (9.81 * k_len) + (k_surface_tension_over_density * k_len * k2);
  float restoring_dk = 9.81 + (3.0 * k_surface_tension_over_density * k2);
  float omega_sq_dk = (restoring_dk * tanh_kh) + (restoring * safe_depth * sech2_kh);
  return 0.5 * omega_sq_dk / omega;
}

float jonswapFrequencySpectrum(float omega, float wind_speed, float jonswap_gamma) {
  if (omega <= 1e-6) {
    return 0.0;
  }
  float safe_wind_speed = max(wind_speed, 0.1);
  float omega_p = (0.877 * 9.81) / safe_wind_speed;
  float sigma = (omega <= omega_p) ? 0.07 : 0.09;
  float alpha = 0.0081;
  float omega_ratio = omega_p / omega;
  float omega_ratio4 = omega_ratio * omega_ratio * omega_ratio * omega_ratio;
  float omega5 = omega * omega * omega * omega * omega;
  float pm = alpha * 9.81 * 9.81 * exp(-1.25 * omega_ratio4) / max(omega5, 1e-9);

  float delta = omega - omega_p;
  float sigma_omega_p = max(sigma * omega_p, 1e-6);
  float exponent = -(delta * delta) / (2.0 * sigma_omega_p * sigma_omega_p);
  float gamma_peak = pow(max(jonswap_gamma, 1.0), exp(exponent));
  return pm * gamma_peak;
}

float directionalSpreadingWeight(float2 k, float2 wind_dir, float directional_spread) {
  float k_len = length(k);
  if (k_len <= 1e-6) {
    return 0.0;
  }
  float spread = max(directional_spread, 0.0);
  float2 k_normalized = k / k_len;
  float k_dot_w = dot(k_normalized, wind_dir);
  float forward = pow(max(0.0, k_dot_w), spread);
  float backward = 0.0 * pow(max(0.0, -k_dot_w), spread);  // Remove trailing waves for sharper choppiness
  return forward + backward;
}

float oceanWaveSpectrum(float2 k, float2 wind_dir, float wind_speed, float water_depth, float jonswap_gamma, float directional_spread, float amplitude) {
  float k_len = length(k);
  if (k_len <= 1e-6) {
    return 0.0;
  }

  float omega = dispersionOmega(k_len, water_depth);
  if (omega <= 1e-6) {
    return 0.0;
  }
  float d_omega_d_k = dispersionDOmegaDK(k_len, water_depth, omega);
  if (d_omega_d_k <= 0.0) {
    return 0.0;
  }

  float s_omega = jonswapFrequencySpectrum(omega, wind_speed, jonswap_gamma);
  if (s_omega <= 0.0) {
    return 0.0;
  }
  float directional = directionalSpreadingWeight(k, wind_dir, directional_spread);
  if (directional <= 0.0) {
    return 0.0;
  }

  float kh = k_len * max(water_depth, 1e-3);
  float depth_attenuation = tanh(kh);
  depth_attenuation *= depth_attenuation;
  float spectral_density_k = s_omega * (d_omega_d_k / max(k_len, 1e-6)) * directional * depth_attenuation;
  float gravity_restoring = 9.81 * k_len;
  float capillary_restoring = k_surface_tension_over_density * k_len * k_len * k_len;
  float omega_p = (0.877 * 9.81) / max(wind_speed, 0.1);
  float omega_ratio = omega / max(omega_p, 1e-6);
  float capillary_gate = smoothstep(2.0, 5.0, omega_ratio);
  float capillary_boost = sqrt((gravity_restoring + capillary_restoring) / max(gravity_restoring, 1e-6));
  capillary_boost = min(capillary_boost, 2.0);
  float micro_scale_m = 0.004;
  float micro_damp = 1.0 / (1.0 + pow(k_len * micro_scale_m, 4.0));
  float short_wave_boost = 1.0 + ((capillary_boost - 1.0) * capillary_gate * micro_damp);
  spectral_density_k *= short_wave_boost;
  return amplitude * spectral_density_k;
}

[numthreads(8, 8, 1)] void GenerateH0(uint3 id : SV_DispatchThreadID) {
  uint N = pushConstants.N;
  if ((id.x >= N) || (id.y >= N)) {
    return;
  }

  RWTexture2D<float4> outH0 = bindless_storage_textures[NonUniformResourceIndex(pushConstants.outH0Index)];
  float L = pushConstants.L;

  float n_x = float(id.x) - float(N) / 2.0;
  float n_y = float(id.y) - float(N) / 2.0;

  float2 k = float2(2.0 * 3.1415926535 * n_x / L, 2.0 * 3.1415926535 * n_y / L);
  float k_len = length(k);

  float band_weight = 1.0;
  if (pushConstants.kMinSoft > 0.0) {
    float k_min_start = pushConstants.kMin - pushConstants.kMinSoft;
    float k_min_end = pushConstants.kMin + pushConstants.kMinSoft;
    band_weight *= smoothstep(k_min_start, k_min_end, k_len);
  } else if (k_len < pushConstants.kMin) {
    band_weight = 0.0;
  }
  if (pushConstants.kMaxSoft > 0.0) {
    float k_max_start = pushConstants.kMax - pushConstants.kMaxSoft;
    float k_max_end = pushConstants.kMax + pushConstants.kMaxSoft;
    band_weight *= (1.0 - smoothstep(k_max_start, k_max_end, k_len));
  } else if (k_len > pushConstants.kMax) {
    band_weight = 0.0;
  }
  if (band_weight <= 0.0) {
    outH0[id.xy] = float4(0.0, 0.0, 0.0, 0.0);
    return;
  }

  float2 wind_dir = float2(pushConstants.windDirX, pushConstants.windDirY);
  float wind_dir_len2 = dot(wind_dir, wind_dir);
  if (wind_dir_len2 < 1e-8) {
    wind_dir = float2(1.0, 0.0);
  } else {
    wind_dir *= rsqrt(wind_dir_len2);
  }
  float p_k = oceanWaveSpectrum(k, wind_dir, pushConstants.windSpeed, pushConstants.waterDepth, pushConstants.jonswapGamma, pushConstants.directionalSpread, pushConstants.A);

  uint state1 = (id.x + (id.y * N)) ^ pushConstants.seed;

  float2 rand1 = gaussianRandom(state1);

  float2 h0_k = rand1 * sqrt((p_k * band_weight) / 2.0);

  outH0[id.xy] = float4(h0_k.x, h0_k.y, 0.0, 0.0);
}

// Complex arithmetic
float2 complexMultiply(float2 a, float2 b) {
  return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

float2 complexExp(float theta) {
  return float2(cos(theta), sin(theta));
}

[numthreads(8, 8, 1)] void UpdateSpectrum(uint3 id : SV_DispatchThreadID) {
  uint N = pushConstants.N;
  if ((id.x >= N) || (id.y >= N)) {
    return;
  }

  Texture2D<float4> inH0 = bindless_textures[NonUniformResourceIndex(pushConstants.inH0Index)];
  RWTexture2D<float4> outHt = bindless_storage_textures[NonUniformResourceIndex(pushConstants.outHtIndex)];
  RWTexture2D<float4> outDxDz = bindless_storage_textures[NonUniformResourceIndex(pushConstants.outDxDzIndex)];
  RWTexture2D<float4> outDerivSpec0 = bindless_storage_textures[NonUniformResourceIndex(pushConstants.outDerivSpec0Index)];
  RWTexture2D<float4> outDerivSpec1 = bindless_storage_textures[NonUniformResourceIndex(pushConstants.outDerivSpec1Index)];
  RWTexture2D<float4> outDerivSpec2 = bindless_storage_textures[NonUniformResourceIndex(pushConstants.outDerivSpec2Index)];
  float L = pushConstants.L;
  float time = pushConstants.time;

  float n_x = float(id.x) - float(N) / 2.0;
  float n_y = float(id.y) - float(N) / 2.0;

  float2 k = float2(2.0 * 3.1415926535 * n_x / L, 2.0 * 3.1415926535 * n_y / L);
  float k_len = length(k);

  float w = dispersionOmega(k_len, pushConstants.waterDepth);

  uint2 mirror_id = uint2((N - id.x) % N, (N - id.y) % N);
  float4 h0_data = inH0.Load(int3(id.xy, 0));
  float4 h0_mirror_data = inH0.Load(int3(mirror_id, 0));
  float2 h0_k = h0_data.xy;
  float2 h0_minus_k_conj = float2(h0_mirror_data.x, -h0_mirror_data.y);

  float2 exp_iwt = complexExp(w * time);
  float2 exp_minus_iwt = complexExp(-w * time);

  float2 h_k_t = complexMultiply(h0_k, exp_iwt) + complexMultiply(h0_minus_k_conj, exp_minus_iwt);

  float2 dx = float2(0.0, 0.0);
  float2 dz = float2(0.0, 0.0);

  if (k_len > 0.001) {
    float2 i_k_x = float2(0.0, -k.x / k_len);
    float2 i_k_y = float2(0.0, -k.y / k_len);

    dx = complexMultiply(i_k_x, h_k_t);
    dz = complexMultiply(i_k_y, h_k_t);
  }

  outHt[id.xy] = float4(h_k_t.x, h_k_t.y, 0.0, 0.0);
  outDxDz[id.xy] = float4(dx.x, dx.y, dz.x, dz.y);

  float2 d_op_x = float2(0.0, k.x);
  float2 d_op_z = float2(0.0, k.y);
  float2 du_x = complexMultiply(d_op_x, dx);
  float2 du_y = complexMultiply(d_op_x, h_k_t);
  float2 du_z = complexMultiply(d_op_x, dz);
  float2 dv_x = complexMultiply(d_op_z, dx);
  float2 dv_y = complexMultiply(d_op_z, h_k_t);
  float2 dv_z = complexMultiply(d_op_z, dz);
  outDerivSpec0[id.xy] = float4(du_x.x, du_x.y, du_y.x, du_y.y);
  outDerivSpec1[id.xy] = float4(du_z.x, du_z.y, dv_x.x, dv_x.y);
  outDerivSpec2[id.xy] = float4(dv_y.x, dv_y.y, dv_z.x, dv_z.y);
}
