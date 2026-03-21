#pragma once

#include "bsdf_fresnel_shared.hxx"

ETX_SHARED_INLINE SpectralResponse bsdf_external_eval_diffuse(
  ETX_INOUT(Sampler, sampler), ETX_IN(float3, incoming_direction), ETX_IN(float3, outgoing_direction), ETX_IN(float2, alpha), ETX_IN(SpectralResponse, albedo)) {
  (void)sampler;
  (void)incoming_direction;
  (void)alpha;

  float cosine = max(0.0f, outgoing_direction.z);
  return spectral_response_mul(albedo, cosine * kInvPi);
}

ETX_SHARED_INLINE float3 bsdf_external_sample_diffuse(ETX_INOUT(Sampler, sampler), ETX_IN(float3, incoming_direction), ETX_IN(float2, alpha)) {
  (void)incoming_direction;
  (void)alpha;
  return sample_cosine_distribution(bsdf_sampler_next_2d(sampler), 1.0f);
}

ETX_SHARED_INLINE float3 bsdf_external_sample_diffuse(
  ETX_INOUT(Sampler, sampler), ETX_IN(float3, incoming_direction), ETX_IN(float2, alpha), ETX_IN(SpectralResponse, albedo), ETX_OUT(SpectralResponse, energy)) {
  (void)incoming_direction;
  (void)alpha;
  energy = albedo;
  return sample_cosine_distribution(bsdf_sampler_next_2d(sampler), 1.0f);
}

ETX_SHARED_INLINE SpectralResponse bsdf_external_vmf_diffuse_brdf(
  ETX_IN(float3, incoming_direction), ETX_IN(float3, outgoing_direction), ETX_IN(float2, roughness), ETX_IN(SpectralResponse, albedo)) {
  (void)incoming_direction;
  (void)outgoing_direction;
  (void)roughness;
  return spectral_response_mul(albedo, kInvPi);
}

ETX_STATIC_CONST uint32_t kBSDFExternalScatteringOrderMax = 16u;

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

  if (result.w.z > 0.9999f) {
    result.Lambda = 0.0f;
    return result;
  }

  if (result.w.z < -0.9999f) {
    result.Lambda = -1.0f;
    return result;
  }

  float theta = acos(result.w.z);
  float cos_theta = result.w.z;
  float sin_theta = sin(theta);
  float tan_theta = sin_theta / cos_theta;
  float inv_sin_theta_2 = 1.0f / (1.0f - result.w.z * result.w.z);
  float cos_phi_2 = result.w.x * result.w.x * inv_sin_theta_2;
  float sin_phi_2 = result.w.y * result.w.y * inv_sin_theta_2;
  float alpha_value = sqrt(cos_phi_2 * alpha.x * alpha.x + sin_phi_2 * alpha.y * alpha.y);
  float a = 1.0f / tan_theta / alpha_value;
  result.Lambda = 0.5f * (-1.0f + ((a > 0.0f) ? 1.0f : -1.0f) * sqrt(1.0f + 1.0f / (a * a)));
  return result;
}

ETX_SHARED_INLINE BSDFExternalRayInfo bsdf_external_ray_info_update_direction(
  ETX_IN(BSDFExternalRayInfo, ray), ETX_IN(float3, in_w), ETX_IN(float2, alpha)) {
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

ETX_SHARED_INLINE float3 bsdf_external_sample_vndf(ETX_INOUT(Sampler, sampler), ETX_IN(float3, wi), ETX_IN(float2, alpha)) {
  float3 wi_11 = normalize(float3(alpha.x * wi.x, alpha.y * wi.y, wi.z));
  float2 slope_11 = bsdf_external_sample_p22_11(acos(wi_11.z), bsdf_sampler_next_2d(sampler), alpha);

  float phi = atan2(wi_11.y, wi_11.x);
  float2 slope = float2(cos(phi) * slope_11.x - sin(phi) * slope_11.y, sin(phi) * slope_11.x + cos(phi) * slope_11.y);

  slope.x *= alpha.x;
  slope.y *= alpha.y;

  if (((slope.x != slope.x) == true) || isinf(slope.x)) {
    if (wi.z > 0.0f) {
      return float3(0.0f, 0.0f, 1.0f);
    }
    return normalize(float3(wi.x, wi.y, 0.0f));
  }

  return normalize(float3(-slope.x, -slope.y, 1.0f));
}

ETX_SHARED_INLINE SpectralResponse bsdf_external_phase_function_reflection(ETX_IN(SpectralQuery, spect), ETX_IN(BSDFExternalRayInfo, ray), ETX_IN(float3, wo),
  ETX_IN(float2, alpha), ETX_IN(RefractiveIndexSample, ext_ior), ETX_IN(RefractiveIndexSample, int_ior), ETX_IN(ThinfilmEval, thinfilm)) {
  if (ray.w.z > 0.9999f) {
    return spectral_response_make(spect, 0.0f);
  }

  float projected_area = (ray.w.z < -0.9999f) ? 1.0f : ray.Lambda * ray.w.z;
  if (projected_area < kEpsilon) {
    return spectral_response_make(spect, 0.0f);
  }

  float3 wh = normalize(-ray.w + wo);
  if (wh.z < 0.0f) {
    return spectral_response_make(spect, 0.0f);
  }

  float w_dot_h = dot(-ray.w, wh);
  if (w_dot_h < kEpsilon) {
    return spectral_response_make(spect, 0.0f);
  }

  SpectralResponse f = bsdf_fresnel_calculate(spect, w_dot_h, ext_ior, int_ior, thinfilm);
  float d_ggx = bsdf_external_d_ggx(wh, alpha);
  float d = d_ggx / (4.0f * projected_area);
  return spectral_response_mul(f, d);
}

ETX_SHARED_INLINE float3 bsdf_external_sample_phase_function_conductor(ETX_IN(SpectralQuery, spect), ETX_IN(float2, slope_rnd), ETX_IN(float3, wi), ETX_IN(float2, alpha),
  ETX_IN(RefractiveIndexSample, ext_ior), ETX_IN(RefractiveIndexSample, int_ior), ETX_IN(ThinfilmEval, thinfilm), ETX_OUT(SpectralResponse, weight)) {
  float3 wi_11 = normalize(float3(alpha.x * wi.x, alpha.y * wi.y, wi.z));
  float2 slope_11 = bsdf_external_sample_p22_11(acos(wi_11.z), slope_rnd, alpha);

  float phi = atan2(wi_11.y, wi_11.x);
  float2 slope = float2(cos(phi) * slope_11.x - sin(phi) * slope_11.y, sin(phi) * slope_11.x + cos(phi) * slope_11.y);
  slope.x *= alpha.x;
  slope.y *= alpha.y;

  float3 wm = float3(0.0f, 0.0f, 0.0f);
  if (((slope.x != slope.x) == true) || isinf(slope.x)) {
    wm = (wi.z > 0.0f) ? float3(0.0f, 0.0f, 1.0f) : normalize(float3(wi.x, wi.y, 0.0f));
  } else {
    wm = normalize(float3(-slope.x, -slope.y, 1.0f));
  }

  float i_dot_m = dot(wi, wm);
  weight = bsdf_fresnel_calculate(spect, i_dot_m, ext_ior, int_ior, thinfilm);
  return -wi + 2.0f * wm * i_dot_m;
}

ETX_SHARED_INLINE float bsdf_external_mis_weight_conductor(ETX_IN(float3, wi), ETX_IN(float3, wo), ETX_IN(float2, alpha)) {
  if ((wi.x == -wo.x) && (wi.y == -wo.y) && (wi.z == -wo.z)) {
    return 1.0f;
  }

  float3 wh = normalize(wi + wo);
  return bsdf_external_d_ggx((wh.z > 0.0f) ? wh : -wh, alpha);
}

ETX_SHARED_INLINE SpectralResponse bsdf_external_eval_conductor(ETX_IN(SpectralQuery, spect), ETX_INOUT(Sampler, sampler), ETX_IN(float3, wi), ETX_IN(float3, wo),
  ETX_IN(float2, alpha), ETX_IN(RefractiveIndexSample, ext_ior), ETX_IN(RefractiveIndexSample, int_ior), ETX_IN(ThinfilmEval, thinfilm)) {
  if ((wi.z <= 0.0f) || (wo.z <= 0.0f)) {
    return spectral_response_make(spect, 0.0f);
  }

  BSDFExternalRayInfo ray = bsdf_external_ray_info_make(-wi, alpha);
  ray = bsdf_external_ray_info_update_height(ray, 1.0f);
  SpectralResponse energy = spectral_response_make(spect, 1.0f);
  BSDFExternalRayInfo ray_shadowing = bsdf_external_ray_info_make(wo, alpha);

  float3 wh = normalize(wi + wo);
  float d = bsdf_external_d_ggx(wh, alpha);
  float g2 = 1.0f / (1.0f + (-ray.Lambda - 1.0f) + ray_shadowing.Lambda);
  SpectralResponse single_scattering = spectral_response_mul(bsdf_fresnel_calculate(spect, dot(ray.w, wh), ext_ior, int_ior, thinfilm), d * g2 / (4.0f * wi.z));

  float wi_mis_weight = 0.0f;
  SpectralResponse multiple_scattering = spectral_response_make(spect, 0.0f);
  uint32_t current_scattering_order = 0u;
  while (current_scattering_order < kBSDFExternalScatteringOrderMax) {
    ray = bsdf_external_ray_info_update_height(ray, bsdf_external_sample_height(ray, bsdf_sampler_next(sampler)));
    if (ray.h == kMaxFloat) {
      break;
    }

    current_scattering_order += 1u;
    if (current_scattering_order > 1u) {
      SpectralResponse phase_function = bsdf_external_phase_function_reflection(spect, ray, wo, alpha, ext_ior, int_ior, thinfilm);
      ray_shadowing = bsdf_external_ray_info_update_height(ray_shadowing, ray.h);
      SpectralResponse intensity = spectral_response_mul(spectral_response_mul(energy, phase_function), ray_shadowing.G1);
      float mis = wi_mis_weight / (wi_mis_weight + bsdf_external_mis_weight_conductor(-ray.w, wo, alpha));
      multiple_scattering = spectral_response_add(multiple_scattering, spectral_response_mul(intensity, mis));
    }

    float2 slope_rnd = ((current_scattering_order == 1u) && bsdf_sampler_has_fixed(sampler)) ? float2(sampler.fixed_u, sampler.fixed_v) : bsdf_sampler_next_2d(sampler);
    SpectralResponse weight = spectral_response_make(spect, 0.0f);
    ray = bsdf_external_ray_info_update_direction(
      ray, bsdf_external_sample_phase_function_conductor(spect, slope_rnd, -ray.w, alpha, ext_ior, int_ior, thinfilm, weight), alpha);
    energy = spectral_response_mul(energy, weight);
    ray = bsdf_external_ray_info_update_height(ray, ray.h);

    if (current_scattering_order == 1u) {
      wi_mis_weight = bsdf_external_mis_weight_conductor(wi, ray.w, alpha);
    }

    if (((ray.h != ray.h) == true) || ((ray.w.x != ray.w.x) == true)) {
      return spectral_response_make(spect, 0.0f);
    }
  }

  return spectral_response_add(spectral_response_mul(single_scattering, 0.5f), multiple_scattering);
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

ETX_SHARED_INLINE float bsdf_external_beta(float m, float n) {
  return bsdf_external_gamma(m) * bsdf_external_gamma(n) / bsdf_external_gamma(m + n);
}

ETX_SHARED_INLINE float3 bsdf_external_refract(ETX_IN(float3, wi), ETX_IN(float3, wm), float eta) {
  float cos_theta_i = dot(wi, wm);
  float cos_theta_t2 = 1.0f - (1.0f - cos_theta_i * cos_theta_i) / (eta * eta);
  float cos_theta_t = -sqrt(max(0.0f, cos_theta_t2));
  return wm * (dot(wi, wm) / eta + cos_theta_t) - wi / eta;
}

ETX_SHARED_INLINE SpectralResponse bsdf_external_eval_phase_function_dielectric(ETX_IN(SpectralQuery, spect), ETX_IN(BSDFExternalRayInfo, ray), ETX_IN(float3, wo),
  bool reflection, ETX_IN(RefractiveIndexSample, ext_ior), ETX_IN(RefractiveIndexSample, int_ior), ETX_IN(ThinfilmEval, thinfilm), ETX_IN(float2, alpha)) {
  if (ray.w.z > 0.9999f) {
    return spectral_response_make(spect, 0.0f);
  }

  if (reflection) {
    return bsdf_external_phase_function_reflection(spect, ray, wo, alpha, ext_ior, int_ior, thinfilm);
  }

  float projected_area = (ray.w.z < -0.9999f) ? 1.0f : ray.Lambda * ray.w.z;
  if (projected_area < kEpsilon) {
    return spectral_response_make(spect, 0.0f);
  }

  float eta = spectral_response_monochromatic(spectral_response_div(int_ior.eta, ext_ior.eta));
  float3 wh = normalize(-ray.w + wo * eta);
  wh *= (wh.z > 0.0f) ? 1.0f : -1.0f;

  float i_dot_m = -dot(wh, ray.w);
  if (i_dot_m < 0.0f) {
    return spectral_response_make(spect, 0.0f);
  }

  float o_dot_m = dot(wo, wh);
  float denom = i_dot_m + eta * o_dot_m;
  float scalar = i_dot_m * max(0.0f, -o_dot_m) * bsdf_external_d_ggx(wh, alpha) / (projected_area * denom * denom);
  SpectralResponse f = bsdf_fresnel_calculate(spect, i_dot_m, ext_ior, int_ior, thinfilm);
  SpectralResponse one_minus_f = spectral_response_sub(spectral_response_make(spect, 1.0f), f);
  return spectral_response_mul(one_minus_f, scalar);
}

struct BSDFExternalDielectricSample {
  float3 w_o ETX_INIT({});
  SpectralResponse weight ETX_INIT({});
  bool reflection ETX_INIT(false);
};

ETX_SHARED_INLINE BSDFExternalDielectricSample bsdf_external_sample_phase_function_dielectric(ETX_IN(SpectralQuery, spect), ETX_IN(float2, rnd_slope), float rnd_reflection,
  ETX_IN(float3, wi), ETX_IN(float2, alpha), ETX_IN(RefractiveIndexSample, ext_ior), ETX_IN(RefractiveIndexSample, int_ior), ETX_IN(ThinfilmEval, thinfilm)) {
  float3 wi_11 = normalize(float3(alpha.x * wi.x, alpha.y * wi.y, wi.z));
  float2 slope_11 = bsdf_external_sample_p22_11(acos(wi_11.z), rnd_slope, alpha);

  float phi = atan2(wi_11.y, wi_11.x);
  float2 slope = float2(cos(phi) * slope_11.x - sin(phi) * slope_11.y, sin(phi) * slope_11.x + cos(phi) * slope_11.y);
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
    result.weight = f;
  } else {
    result.weight = spectral_response_sub(spectral_response_make(spect, 1.0f), f);
  }
  result.w_o = result.reflection ? (-wi + 2.0f * wm * i_dot_m) : normalize(bsdf_external_refract(wi, wm, eta));
  return result;
}

ETX_SHARED_INLINE float bsdf_external_mis_weight_dielectric(ETX_IN(float3, wi), ETX_IN(float3, wo), bool reflection, float eta, ETX_IN(float2, alpha)) {
  if (reflection) {
    if ((wi.x == -wo.x) && (wi.y == -wo.y) && (wi.z == -wo.z)) {
      return 1.0f;
    }

    float3 wh = normalize(wi + wo);
    return bsdf_external_d_ggx((wh.z > 0.0f) ? wh : -wh, alpha);
  }

  float3 wh = normalize(wi + wo * eta);
  return bsdf_external_d_ggx((wh.z > 0.0f) ? wh : -wh, alpha);
}

ETX_SHARED_INLINE SpectralResponse bsdf_external_eval_dielectric(ETX_IN(SpectralQuery, spect), ETX_INOUT(Sampler, sampler), ETX_IN(float3, wi), ETX_IN(float3, wo),
  bool wo_outside, ETX_IN(float2, alpha), ETX_IN(RefractiveIndexSample, ext_ior), ETX_IN(RefractiveIndexSample, int_ior), ETX_IN(ThinfilmEval, thinfilm)) {
  if ((wi.z <= 0.0f) || ((wo.z <= 0.0f) && wo_outside) || ((wo.z >= 0.0f) && (wo_outside == false))) {
    return spectral_response_make(spect, 0.0f);
  }

  BSDFExternalRayInfo ray = bsdf_external_ray_info_make(-wi, alpha);
  ray = bsdf_external_ray_info_update_height(ray, 1.0f);
  bool outside = true;
  BSDFExternalRayInfo ray_shadowing = bsdf_external_ray_info_make(wo_outside ? wo : -wo, alpha);

  SpectralResponse single_scattering = spectral_response_make(spect, 0.0f);
  SpectralResponse multiple_scattering = spectral_response_make(spect, 0.0f);
  float wi_mis_weight = 0.0f;
  float eta = spectral_response_monochromatic(spectral_response_div(int_ior.eta, ext_ior.eta));

  int current_scattering_order = 0;
  while (current_scattering_order < int(kBSDFExternalScatteringOrderMax)) {
    ray = bsdf_external_ray_info_update_height(ray, bsdf_external_sample_height(ray, bsdf_sampler_next(sampler)));
    if (ray.h == kMaxFloat) {
      break;
    }

    current_scattering_order += 1;
    if (current_scattering_order == 1) {
      SpectralResponse phase_function = bsdf_external_eval_phase_function_dielectric(spect, ray, wo, wo_outside, ext_ior, int_ior, thinfilm, alpha);
      float g2_g1 = 0.0f;
      if (wo_outside) {
        g2_g1 = (1.0f + (-ray.Lambda - 1.0f)) / (1.0f + (-ray.Lambda - 1.0f) + ray_shadowing.Lambda);
      } else {
        g2_g1 = (1.0f + (-ray.Lambda - 1.0f)) * bsdf_external_beta(1.0f + (-ray.Lambda - 1.0f), 1.0f + ray_shadowing.Lambda);
      }

      if (isfinite(g2_g1)) {
        single_scattering = spectral_response_mul(phase_function, g2_g1);
      }
    }

    if (current_scattering_order > 1) {
      SpectralResponse phase_function = spectral_response_make(spect, 0.0f);
      float mis = 0.0f;
      if (outside) {
        phase_function = bsdf_external_eval_phase_function_dielectric(spect, ray, wo, wo_outside, ext_ior, int_ior, thinfilm, alpha);
        mis = wi_mis_weight / (wi_mis_weight + bsdf_external_mis_weight_dielectric(-ray.w, wo, wo_outside, eta, alpha));
      } else {
        phase_function = bsdf_external_eval_phase_function_dielectric(spect, ray, -wo, (wo_outside == false), int_ior, ext_ior, thinfilm, alpha);
        mis = wi_mis_weight / (wi_mis_weight + bsdf_external_mis_weight_dielectric(-ray.w, -wo, (wo_outside == false), 1.0f / eta, alpha));
      }

      ray_shadowing = bsdf_external_ray_info_update_height(ray_shadowing, (outside == wo_outside) ? ray.h : -ray.h);
      multiple_scattering = spectral_response_add(multiple_scattering, spectral_response_mul(spectral_response_mul(phase_function, ray_shadowing.G1), mis));
    }

    float2 rnd_slope = ((current_scattering_order == 1) && bsdf_sampler_has_fixed(sampler)) ? float2(sampler.fixed_u, sampler.fixed_v) : bsdf_sampler_next_2d(sampler);
    float rnd_reflection = ((current_scattering_order == 1) && bsdf_sampler_has_fixed(sampler)) ? sampler.fixed_w : bsdf_sampler_next(sampler);
    RefractiveIndexSample phase_ext_ior = ext_ior;
    RefractiveIndexSample phase_int_ior = int_ior;
    if (outside == false) {
      phase_ext_ior = int_ior;
      phase_int_ior = ext_ior;
    }
    BSDFExternalDielectricSample next_sample =
      bsdf_external_sample_phase_function_dielectric(spect, rnd_slope, rnd_reflection, -ray.w, alpha, phase_ext_ior, phase_int_ior, thinfilm);
    if (next_sample.reflection) {
      ray = bsdf_external_ray_info_update_direction(ray, next_sample.w_o, alpha);
      ray = bsdf_external_ray_info_update_height(ray, ray.h);
    } else {
      outside = (outside == false);
      ray = bsdf_external_ray_info_update_direction(ray, -next_sample.w_o, alpha);
      ray = bsdf_external_ray_info_update_height(ray, -ray.h);
    }

    if (current_scattering_order == 1) {
      wi_mis_weight = bsdf_external_mis_weight_dielectric(wi, ray.w, outside, eta, alpha);
    }

    if (((ray.h != ray.h) == true) || ((ray.w.x != ray.w.x) == true) || (ray.w.z <= kEpsilon)) {
      return spectral_response_make(spect, 0.0f);
    }
  }

  return spectral_response_add(spectral_response_mul(single_scattering, 0.5f), multiple_scattering);
}
