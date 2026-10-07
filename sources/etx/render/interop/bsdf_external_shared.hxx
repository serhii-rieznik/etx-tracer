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

  float sin_theta = sqrt(max(0.0f, 1.0f - cos_theta * cos_theta));
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

ETX_SHARED_INLINE float bsdf_external_log_one_plus(float x) {
  if (x > 0.5f) {
    return log(1.0f + x);
  }
  const float z = x / (2.0f + x);
  const float z2 = z * z;
  // log(1+x) = 2*atanh(x/(2+x)); the omitted tail is below float precision for x <= 0.5.
  return 2.0f * z * (1.0f + z2 * (1.0f / 3.0f + z2 * (1.0f / 5.0f + z2 * (1.0f / 7.0f + z2 * (1.0f / 9.0f + z2 * (1.0f / 11.0f + z2 / 13.0f))))));
}

ETX_SHARED_INLINE float bsdf_external_beta(float m, float n) {
  if ((isfinite(m) == false) || (isfinite(n) == false) || (m <= 0.0f) || (n <= 0.0f) || (m > kBSDFExternalLambdaMax) || (n > kBSDFExternalLambdaMax)) {
    return 0.0f;
  }
  float recurrence = 0.0f;
  while (m < 8.0f) {
    recurrence += bsdf_external_log_one_plus(n / m);
    m += 1.0f;
  }
  while (n < 8.0f) {
    recurrence += bsdf_external_log_one_plus(m / n);
    n += 1.0f;
  }
  const float sum = m + n;
  const float m_log = bsdf_external_log_one_plus(n / m);
  const float n_log = bsdf_external_log_one_plus(m / n);
  const float m_inv = 1.0f / m;
  const float n_inv = 1.0f / n;
  const float sum_inv = 1.0f / sum;
  const float correction = (m_inv + n_inv - sum_inv) / 12.0f - (m_inv * m_inv * m_inv + n_inv * n_inv * n_inv - sum_inv * sum_inv * sum_inv) / 360.0f +
                           (pow(m_inv, 5.0f) + pow(n_inv, 5.0f) - pow(sum_inv, 5.0f)) / 1260.0f;
  // The ratio form avoids subtracting large log-gamma values at grazing incidence.
  const float log_beta = recurrence - (m - 0.5f) * m_log - (n - 0.5f) * n_log - 0.5f * log(sum) + 0.9189385332f + correction;
  return exp(log_beta);
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

ETX_SHARED_INLINE float3 bsdf_external_sample_vndf_local(ETX_IN(float3, w_i), ETX_IN(float2, alpha), ETX_IN(float2, rnd)) {
  const float3 view = normalize(float3(alpha.x * w_i.x, alpha.y * w_i.y, w_i.z));
  const float tangent_length_squared = view.x * view.x + view.y * view.y;
  const float3 tangent = (tangent_length_squared > 0.0f) ? float3(-view.y, view.x, 0.0f) / sqrt(tangent_length_squared) : float3(1.0f, 0.0f, 0.0f);
  const float3 bitangent = cross(view, tangent);
  const float radius = sqrt(rnd.x);
  const float azimuth = kDoublePi * rnd.y;
  const float disk_x = radius * cos(azimuth);
  const float hemisphere_fraction = 0.5f * (1.0f + view.z);
  const float disk_y = (1.0f - hemisphere_fraction) * sqrt(max(0.0f, 1.0f - disk_x * disk_x)) + hemisphere_fraction * radius * sin(azimuth);
  const float3 normal = disk_x * tangent + disk_y * bitangent + sqrt(max(0.0f, 1.0f - disk_x * disk_x - disk_y * disk_y)) * view;
  return normalize(float3(alpha.x * normal.x, alpha.y * normal.y, max(0.0f, normal.z)));
}

ETX_SHARED_INLINE float3 bsdf_external_sample_vndf_local(ETX_IN(float3, w_i), float alpha, ETX_IN(float2, rnd)) {
  return bsdf_external_sample_vndf_local(w_i, float2(alpha, alpha), rnd);
}

ETX_SHARED_INLINE float bsdf_external_vndf_pdf(ETX_IN(float3, w_i), ETX_IN(float3, m), float alpha) {
  const BSDFExternalRayInfo ray = bsdf_external_ray_info_make(w_i, float2(alpha, alpha));
  const float denominator = (1.0f + ray.Lambda) * max(kEpsilon, w_i.z);
  return max(0.0f, dot(w_i, m)) * bsdf_external_d_ggx(m, float2(alpha, alpha)) / denominator;
}

ETX_SHARED_INLINE BSDFExternalDielectricSample bsdf_external_sample_phase_function_dielectric(ETX_IN(SpectralQuery, spect), ETX_IN(float2, rnd_slope), float rnd_reflection,
  ETX_IN(float3, wi), ETX_IN(float2, alpha), ETX_IN(RefractiveIndexSample, ext_ior), ETX_IN(RefractiveIndexSample, int_ior), ETX_IN(ThinfilmEval, thinfilm)) {
  const float3 wm = bsdf_external_sample_vndf_local(wi, alpha, rnd_slope);

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
