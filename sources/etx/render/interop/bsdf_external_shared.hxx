#pragma once

#include "bsdf_fresnel_shared.hxx"

ETX_STATIC_CONST uint32_t kBSDFExternalScatteringOrderMax = 16u;
ETX_STATIC_CONST float kBSDFExternalLambdaMax = 1.0e10f;

struct BSDFExternalRayInfo {
  float3 w ETX_INIT({});
  float Lambda ETX_INIT(0.0f);
  float h ETX_INIT(0.0f);
  float C1 ETX_INIT(0.0f);
  float G1 ETX_INIT(0.0f);
  float pad ETX_INIT(0.0f);
};

ETX_SHARED_INLINE BSDFExternalRayInfo bsdf_external_ray_info_make(ETX_IN(float3, w), ETX_IN(float2, alpha)) {
  BSDFExternalRayInfo result = ETX_ZERO(BSDFExternalRayInfo);
  result.w = w;
  result.w.z = clamp(result.w.z, -1.0f, 1.0f);

  if (result.w.z > 0.9999f) {
    result.Lambda = 0.0f;
    return result;
  }

  if (result.w.z < -0.9999f) {
    result.Lambda = -1.0f;
    return result;
  }

  float cos_theta = result.w.z;
  if (abs(cos_theta) <= kEpsilon) {
    result.Lambda = kBSDFExternalLambdaMax;
    return result;
  }

  float theta = acos(cos_theta);
  float sin_theta = sin(theta);
  float tan_theta = sin_theta / cos_theta;
  float sin_theta_sq = max(kEpsilon, 1.0f - result.w.z * result.w.z);
  float inv_sin_theta_2 = 1.0f / sin_theta_sq;
  float cos_phi_2 = result.w.x * result.w.x * inv_sin_theta_2;
  float sin_phi_2 = result.w.y * result.w.y * inv_sin_theta_2;
  float alpha_value = sqrt(max(kEpsilon, cos_phi_2 * alpha.x * alpha.x + sin_phi_2 * alpha.y * alpha.y));
  float a = 1.0f / (tan_theta * alpha_value);
  if (abs(a) <= kEpsilon) {
    result.Lambda = kBSDFExternalLambdaMax;
    return result;
  }

  result.Lambda = min(kBSDFExternalLambdaMax, 0.5f * (-1.0f + ((a > 0.0f) ? 1.0f : -1.0f) * sqrt(1.0f + 1.0f / (a * a))));
  return result;
}

ETX_SHARED_INLINE BSDFExternalRayInfo bsdf_external_ray_info_update_direction(ETX_IN(BSDFExternalRayInfo, ray), ETX_IN(float3, in_w), ETX_IN(float2, alpha)) {
  BSDFExternalRayInfo result = ray;
  result = bsdf_external_ray_info_make(in_w, alpha);
  result.h = ray.h;
  result.C1 = ray.C1;
  result.G1 = ray.G1;
  return result;
}

ETX_SHARED_INLINE BSDFExternalRayInfo bsdf_external_ray_info_update_height(ETX_IN(BSDFExternalRayInfo, ray), float in_h) {
  BSDFExternalRayInfo result = ray;
  result.h = in_h;
  result.C1 = min(1.0f, max(0.0f, 0.5f * (result.h + 1.0f)));
  if (result.w.z > 0.9999f) {
    result.G1 = 1.0f;
  } else if (result.w.z <= 0.0f) {
    result.G1 = 0.0f;
  } else {
    result.G1 = pow(result.C1, result.Lambda);
  }

  return result;
}

ETX_SHARED_INLINE float bsdf_external_inverse_c1(float u) {
  return max(-1.0f, min(1.0f, 2.0f * u - 1.0f));
}

ETX_SHARED_INLINE float bsdf_external_sample_height(ETX_IN(BSDFExternalRayInfo, ray), float u) {
  if (ray.w.z > 0.9999f) {
    return kMaxFloat;
  }

  if (ray.w.z < -0.9999f) {
    return bsdf_external_inverse_c1(u * ray.C1);
  }

  if (abs(ray.w.z) < 0.0001f) {
    return ray.h;
  }

  if (u > (1.0f - ray.G1)) {
    return kMaxFloat;
  }

  float p1 = pow((1.0f - u), 1.0f / ray.Lambda);
  if (p1 <= 0.0f) {
    return kMaxFloat;
  }

  float u1 = ray.C1 / p1;
  float result = bsdf_external_inverse_c1(u1);
  return result;
}

ETX_SHARED_INLINE float bsdf_external_d_ggx(ETX_IN(float3, wm), ETX_IN(float2, alpha)) {
  if (wm.z <= kEpsilon) {
    return 0.0f;
  }

  float slope_x = -wm.x / wm.z;
  float slope_y = -wm.y / wm.z;

  float ax = max(kEpsilon, alpha.x * alpha.x);
  float ay = max(kEpsilon, alpha.y * alpha.y);
  float axy = max(kEpsilon, alpha.x * alpha.y);

  float tmp = 1.0f + slope_x * slope_x / ax + slope_y * slope_y / ay;
  float p22 = 1.0f / (kPi * axy * tmp * tmp);
  return p22 / (wm.z * wm.z * wm.z * wm.z);
}

ETX_SHARED_INLINE float2 bsdf_external_sample_p22_11(float theta_i, ETX_IN(float2, rnd), ETX_IN(float2, alpha)) {
  (void)alpha;

  float2 slope = float2(0.0f, 0.0f);
  if (theta_i < 0.0001f) {
    float r = sqrt(rnd.x / (1.0f - rnd.x));
    float phi = kDoublePi * rnd.y;
    slope.x = r * cos(phi);
    slope.y = r * sin(phi);
    return slope;
  }

  float sin_theta_i = sin(theta_i);
  float cos_theta_i = cos(theta_i);
  float tan_theta_i = sin_theta_i / cos_theta_i;

  float projected_area = 0.5f * (cos_theta_i + 1.0f);
  if (projected_area < 0.0001f) {
    return slope;
  }

  float c = 1.0f / projected_area;
  float a = 2.0f * rnd.x / cos_theta_i / c - 1.0f;
  float b = tan_theta_i;
  float tmp = 1.0f / (a * a - 1.0f);
  float d = sqrt(max(0.0f, b * b * tmp * tmp - (a * a - b * b) * tmp));
  float slope_x_1 = b * tmp - d;
  float slope_x_2 = b * tmp + d;
  slope.x = ((a < 0.0f) || (slope_x_2 > (1.0f / tan_theta_i))) ? slope_x_1 : slope_x_2;

  float u2 = 0.0f;
  float sign = 0.0f;
  if (rnd.y > 0.5f) {
    sign = 1.0f;
    u2 = 2.0f * (rnd.y - 0.5f);
  } else {
    sign = -1.0f;
    u2 = 2.0f * (0.5f - rnd.y);
  }

  float z = (u2 * (u2 * (u2 * 0.27385f - 0.73369f) + 0.46341f)) / (u2 * (u2 * (u2 * 0.093073f + 0.309420f) - 1.0f) + 0.597999f);
  slope.y = sign * z * sqrt(1.0f + slope.x * slope.x);
  return slope;
}

ETX_SHARED_INLINE float bsdf_external_abgam(float x) {
  const float gam[7] = {
    1.0f / 12.0f,
    1.0f / 30.0f,
    53.0f / 210.0f,
    195.0f / 371.0f,
    22999.0f / 22737.0f,
    29944523.0f / 19733142.0f,
    109535241009.0f / 48264275462.0f,
  };
  const float kHalfLogDoublePi = 0.918938518f;
  return kHalfLogDoublePi - x + (x - 0.5f) * log(x) + gam[0] / (x + gam[1] / (x + gam[2] / (x + gam[3] / (x + gam[4] / (x + gam[5] / (x + gam[6] / x))))));
}

ETX_SHARED_INLINE float bsdf_external_gamma(float x) {
  return exp(bsdf_external_abgam(x + 5.0f)) / (x * (x + 1.0f) * (x + 2.0f) * (x + 3.0f) * (x + 4.0f));
}

ETX_SHARED_INLINE float bsdf_external_log_gamma_approx(float x) {
  return bsdf_external_abgam(x + 5.0f) - log(x) - log(x + 1.0f) - log(x + 2.0f) - log(x + 3.0f) - log(x + 4.0f);
}

ETX_SHARED_INLINE float bsdf_external_beta(float m, float n) {
  if ((isfinite(m) == false) || (isfinite(n) == false) || (m <= 0.0f) || (n <= 0.0f) || (m > kBSDFExternalLambdaMax) || (n > kBSDFExternalLambdaMax)) {
    return 0.0f;
  }
#if (ETX_CPP)
  return exp(lgamma(m) + lgamma(n) - lgamma(m + n));
#else
  return exp(bsdf_external_log_gamma_approx(m) + bsdf_external_log_gamma_approx(n) - bsdf_external_log_gamma_approx(m + n));
#endif
}

ETX_SHARED_INLINE float3 bsdf_external_refract(ETX_IN(float3, wi), ETX_IN(float3, wm), float eta) {
  if (abs(eta) <= kEpsilon) {
    return float3(0.0f, 0.0f, 0.0f);
  }

  float cos_theta_i = dot(wi, wm);
  float cos_theta_t2 = 1.0f - (1.0f - cos_theta_i * cos_theta_i) / (eta * eta);
  float cos_theta_t = -sqrt(max(0.0f, cos_theta_t2));
  return wm * (dot(wi, wm) / eta + cos_theta_t) - wi / eta;
}

struct BSDFExternalDielectricSample {
  float3 w_o ETX_INIT({});
  bool reflection ETX_INIT(false);
};

ETX_SHARED_INLINE float3 bsdf_external_sample_vndf_local(ETX_IN(float3, w_i), float alpha, ETX_IN(float2, rnd)) {
  const float3 w_i_11 = normalize(float3(alpha * w_i.x, alpha * w_i.y, w_i.z));
  const float2 slope_11 = bsdf_external_sample_p22_11(acos(saturate(w_i_11.z)), rnd, float2(alpha, alpha));

  float2 slope = slope_11;
  const float wi_xy_length_sq = (w_i_11.x * w_i_11.x) + (w_i_11.y * w_i_11.y);
  if (wi_xy_length_sq > (kEpsilon * kEpsilon)) {
    const float phi = atan2(w_i_11.y, w_i_11.x);
    slope = float2(cos(phi) * slope_11.x - sin(phi) * slope_11.y, sin(phi) * slope_11.x + cos(phi) * slope_11.y);
  }
  slope.x *= alpha;
  slope.y *= alpha;

  if ((slope.x != slope.x) || isinf(slope.x)) {
    return (w_i.z > 0.0f) ? float3(0.0f, 0.0f, 1.0f) : normalize(float3(w_i.x, w_i.y, 0.0f));
  }
  return normalize(float3(-slope.x, -slope.y, 1.0f));
}

ETX_SHARED_INLINE float bsdf_external_vndf_pdf(ETX_IN(float3, w_i), ETX_IN(float3, m), float alpha) {
  const BSDFExternalRayInfo ray = bsdf_external_ray_info_make(w_i, float2(alpha, alpha));
  const float denominator = (1.0f + ray.Lambda) * max(kEpsilon, w_i.z);
  return max(0.0f, dot(w_i, m)) * bsdf_external_d_ggx(m, float2(alpha, alpha)) / denominator;
}

ETX_SHARED_INLINE BSDFExternalDielectricSample bsdf_external_sample_phase_function_dielectric(ETX_IN(SpectralQuery, spect), ETX_IN(float2, rnd_slope), float rnd_reflection,
  ETX_IN(float3, wi), ETX_IN(float2, alpha), ETX_IN(RefractiveIndexSample, ext_ior), ETX_IN(RefractiveIndexSample, int_ior), ETX_IN(ThinfilmEval, thinfilm)) {
  float3 wi_11 = normalize(float3(alpha.x * wi.x, alpha.y * wi.y, wi.z));
  float2 slope_11 = bsdf_external_sample_p22_11(acos(wi_11.z), rnd_slope, alpha);

  float2 slope = slope_11;
  const float wi_xy_length_sq = (wi_11.x * wi_11.x) + (wi_11.y * wi_11.y);
  if (wi_xy_length_sq > (kEpsilon * kEpsilon)) {
    const float phi = atan2(wi_11.y, wi_11.x);
    slope = float2(cos(phi) * slope_11.x - sin(phi) * slope_11.y, sin(phi) * slope_11.x + cos(phi) * slope_11.y);
  }
  slope.x *= alpha.x;
  slope.y *= alpha.y;

  float3 wm = float3(0.0f, 0.0f, 0.0f);
  if (isnan(slope.x) || isinf(slope.x)) {
    wm = (wi.z > 0.0f) ? float3(0.0f, 0.0f, 1.0f) : normalize(float3(wi.x, wi.y, 0.0f));
  } else {
    wm = normalize(float3(-slope.x, -slope.y, 1.0f));
  }

  float i_dot_m = dot(wi, wm);
  SpectralResponse f = bsdf_fresnel_calculate(spect, i_dot_m, ext_ior, int_ior, thinfilm);
  float eta = spectral_response_monochromatic(spectral_response_div(int_ior.eta, ext_ior.eta));

  BSDFExternalDielectricSample result = ETX_ZERO(BSDFExternalDielectricSample);
  result.reflection = rnd_reflection < spectral_response_monochromatic(f);
  if (result.reflection) {
    result.w_o = -wi + 2.0f * wm * i_dot_m;
  } else {
    const float3 refracted_w_o = bsdf_external_refract(wi, wm, eta);
    result.w_o = (dot(refracted_w_o, refracted_w_o) > kEpsilon) ? normalize(refracted_w_o) : float3(0.0f, 0.0f, 0.0f);
  }
  return result;
}
