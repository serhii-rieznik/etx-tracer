#pragma once

#include "math_shared.hxx"

ETX_SHARED_INLINE float medium_phase_shared_henyey_greenstein(ETX_IN(float3, w_i), ETX_IN(float3, w_o), float g) {
  float cos_t = dot(w_i, w_o);
  float d = 1.0f + g * g - 2.0f * g * cos_t;
#if defined(__cplusplus)
  return (1.0f / (4.0f * kPi)) * (1.0f - g * g) / (d * sqrtf(d));
#else
  return (1.0f / (4.0f * kPi)) * (1.0f - g * g) / (d * sqrt(d));
#endif
}

ETX_SHARED_INLINE float3 medium_phase_shared_sample_henyey_greenstein(ETX_IN(float3, w_i), float g, ETX_IN(float2, sample_random)) {
  float cos_theta = 0.0f;
  if (abs(g) < 1e-3f) {
    cos_theta = 1.0f - 2.0f * sample_random.x;
  } else {
    float sqr_term = (1.0f - g * g) / (1.0f + g * (2.0f * sample_random.x - 1.0f));
    cos_theta = (1.0f + g * g - sqr_term * sqr_term) / (2.0f * g);
  }

  float sin_theta = sqrt(max(0.0f, 1.0f - cos_theta * cos_theta));
  float phi = kDoublePi * sample_random.y;

  OrthonormalBasis basis = orthonormal_basis(w_i);
#if defined(__cplusplus)
  return (basis.u * cosf(phi) + basis.v * sinf(phi)) * sin_theta - w_i * cos_theta;
#else
  return (basis.u * cos(phi) + basis.v * sin(phi)) * sin_theta - w_i * cos_theta;
#endif
}
