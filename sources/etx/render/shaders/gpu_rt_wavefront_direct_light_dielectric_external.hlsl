#pragma once

#include <interop/bsdf_fresnel_shared.hxx>

static const uint32_t kWavefrontDirectLightDielectricScatteringOrderMax = 16u;

struct WavefrontDirectLightDielectricRayInfo {
  float3 w;
  float lambda_value;
  float h;
  float c1;
  float g1;
  float pad;
};

ETX_SHARED_INLINE WavefrontDirectLightDielectricRayInfo wavefront_direct_light_dielectric_ray_info_make(ETX_IN(float3, w), ETX_IN(float2, alpha)) {
  WavefrontDirectLightDielectricRayInfo result = ETX_ZERO(WavefrontDirectLightDielectricRayInfo);
  result.w = w;

  if (result.w.z > 0.9999f) {
    result.lambda_value = 0.0f;
    return result;
  }

  if (result.w.z < -0.9999f) {
    result.lambda_value = -1.0f;
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
  result.lambda_value = 0.5f * (-1.0f + ((a > 0.0f) ? 1.0f : -1.0f) * sqrt(1.0f + 1.0f / (a * a)));
  return result;
}

ETX_SHARED_INLINE WavefrontDirectLightDielectricRayInfo wavefront_direct_light_dielectric_ray_info_update_direction(
  ETX_IN(WavefrontDirectLightDielectricRayInfo, ray), ETX_IN(float3, in_w), ETX_IN(float2, alpha)) {
  WavefrontDirectLightDielectricRayInfo result = wavefront_direct_light_dielectric_ray_info_make(in_w, alpha);
  result.h = ray.h;
  result.c1 = ray.c1;
  result.g1 = ray.g1;
  return result;
}

ETX_SHARED_INLINE WavefrontDirectLightDielectricRayInfo wavefront_direct_light_dielectric_ray_info_update_height(
  ETX_IN(WavefrontDirectLightDielectricRayInfo, ray), float in_h) {
  WavefrontDirectLightDielectricRayInfo result = ray;
  result.h = in_h;
  result.c1 = min(1.0f, max(0.0f, 0.5f * (result.h + 1.0f)));
  if (result.w.z > 0.9999f) {
    result.g1 = 1.0f;
  } else if (result.w.z <= 0.0f) {
    result.g1 = 0.0f;
  } else {
    result.g1 = pow(result.c1, result.lambda_value);
  }

  return result;
}

ETX_SHARED_INLINE float wavefront_direct_light_dielectric_inverse_c1(float u) {
  return max(-1.0f, min(1.0f, 2.0f * u - 1.0f));
}

ETX_SHARED_INLINE float wavefront_direct_light_dielectric_sample_height(ETX_IN(WavefrontDirectLightDielectricRayInfo, ray), float u) {
  if (ray.w.z > 0.9999f) {
    return kMaxFloat;
  }

  if (ray.w.z < -0.9999f) {
    return wavefront_direct_light_dielectric_inverse_c1(u * ray.c1);
  }

  if (abs(ray.w.z) < 0.0001f) {
    return ray.h;
  }

  if (u > (1.0f - ray.g1)) {
    return kMaxFloat;
  }

  float p1 = pow((1.0f - u), 1.0f / ray.lambda_value);
  if (p1 <= 0.0f) {
    return kMaxFloat;
  }

  float u1 = ray.c1 / p1;
  return wavefront_direct_light_dielectric_inverse_c1(u1);
}

ETX_SHARED_INLINE float wavefront_direct_light_dielectric_d_ggx(ETX_IN(float3, wm), ETX_IN(float2, alpha)) {
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

ETX_SHARED_INLINE float2 wavefront_direct_light_dielectric_sample_p22_11(float theta_i, ETX_IN(float2, rnd), ETX_IN(float2, alpha)) {
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
  float sign_value = 0.0f;
  if (rnd.y > 0.5f) {
    sign_value = 1.0f;
    u2 = 2.0f * (rnd.y - 0.5f);
  } else {
    sign_value = -1.0f;
    u2 = 2.0f * (0.5f - rnd.y);
  }

  float z = (u2 * (u2 * (u2 * 0.27385f - 0.73369f) + 0.46341f)) / (u2 * (u2 * (u2 * 0.093073f + 0.309420f) - 1.0f) + 0.597999f);
  slope.y = sign_value * z * sqrt(1.0f + slope.x * slope.x);
  return slope;
}

ETX_SHARED_INLINE float wavefront_direct_light_dielectric_abgam(float x) {
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

ETX_SHARED_INLINE float wavefront_direct_light_dielectric_gamma(float x) {
  return exp(wavefront_direct_light_dielectric_abgam(x + 5.0f)) / (x * (x + 1.0f) * (x + 2.0f) * (x + 3.0f) * (x + 4.0f));
}

ETX_SHARED_INLINE float wavefront_direct_light_dielectric_beta(float m, float n) {
  return wavefront_direct_light_dielectric_gamma(m) * wavefront_direct_light_dielectric_gamma(n) / wavefront_direct_light_dielectric_gamma(m + n);
}

ETX_SHARED_INLINE float3 wavefront_direct_light_dielectric_refract(ETX_IN(float3, wi), ETX_IN(float3, wm), float eta) {
  float cos_theta_i = dot(wi, wm);
  float cos_theta_t2 = 1.0f - (1.0f - cos_theta_i * cos_theta_i) / (eta * eta);
  float cos_theta_t = -sqrt(max(0.0f, cos_theta_t2));
  return wm * (dot(wi, wm) / eta + cos_theta_t) - wi / eta;
}

ETX_SHARED_INLINE SpectralResponse wavefront_direct_light_dielectric_phase_function_reflection(ETX_IN(SpectralQuery, spect),
  ETX_IN(WavefrontDirectLightDielectricRayInfo, ray), ETX_IN(float3, wo), ETX_IN(float2, alpha), ETX_IN(RefractiveIndexSample, ext_ior),
  ETX_IN(RefractiveIndexSample, int_ior), ETX_IN(ThinfilmEval, thinfilm)) {
  if (ray.w.z > 0.9999f) {
    return spectral_response_make(spect, 0.0f);
  }

  float projected_area = (ray.w.z < -0.9999f) ? 1.0f : ray.lambda_value * ray.w.z;
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
  float d_ggx = wavefront_direct_light_dielectric_d_ggx(wh, alpha);
  float d = d_ggx / (4.0f * projected_area);
  return spectral_response_mul(f, d);
}

ETX_SHARED_INLINE SpectralResponse wavefront_direct_light_dielectric_eval_phase_function(ETX_IN(SpectralQuery, spect),
  ETX_IN(WavefrontDirectLightDielectricRayInfo, ray), ETX_IN(float3, wo), bool reflection, ETX_IN(RefractiveIndexSample, ext_ior),
  ETX_IN(RefractiveIndexSample, int_ior), ETX_IN(ThinfilmEval, thinfilm), ETX_IN(float2, alpha)) {
  if (ray.w.z > 0.9999f) {
    return spectral_response_make(spect, 0.0f);
  }

  if (reflection) {
    return wavefront_direct_light_dielectric_phase_function_reflection(spect, ray, wo, alpha, ext_ior, int_ior, thinfilm);
  }

  float projected_area = (ray.w.z < -0.9999f) ? 1.0f : ray.lambda_value * ray.w.z;
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
  float scalar = i_dot_m * max(0.0f, -o_dot_m) * wavefront_direct_light_dielectric_d_ggx(wh, alpha) / (projected_area * denom * denom);
  SpectralResponse f = bsdf_fresnel_calculate(spect, i_dot_m, ext_ior, int_ior, thinfilm);
  SpectralResponse one_minus_f = spectral_response_sub(spectral_response_make(spect, 1.0f), f);
  return spectral_response_mul(one_minus_f, scalar);
}

struct WavefrontDirectLightDielectricSample {
  float3 w_o;
  SpectralResponse weight;
  bool reflection;
};

ETX_SHARED_INLINE WavefrontDirectLightDielectricSample wavefront_direct_light_dielectric_sample_phase_function(ETX_IN(SpectralQuery, spect), ETX_IN(float2, rnd_slope),
  float rnd_reflection, ETX_IN(float3, wi), ETX_IN(float2, alpha), ETX_IN(RefractiveIndexSample, ext_ior), ETX_IN(RefractiveIndexSample, int_ior),
  ETX_IN(ThinfilmEval, thinfilm)) {
  float3 wi_11 = normalize(float3(alpha.x * wi.x, alpha.y * wi.y, wi.z));
  float2 slope_11 = wavefront_direct_light_dielectric_sample_p22_11(acos(wi_11.z), rnd_slope, alpha);

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

  WavefrontDirectLightDielectricSample result = ETX_ZERO(WavefrontDirectLightDielectricSample);
  result.reflection = rnd_reflection < spectral_response_monochromatic(f);
  if (result.reflection) {
    result.weight = f;
  } else {
    result.weight = spectral_response_sub(spectral_response_make(spect, 1.0f), f);
  }
  result.w_o = result.reflection ? (-wi + 2.0f * wm * i_dot_m) : normalize(wavefront_direct_light_dielectric_refract(wi, wm, eta));
  return result;
}

ETX_SHARED_INLINE float wavefront_direct_light_dielectric_mis_weight(ETX_IN(float3, wi), ETX_IN(float3, wo), bool reflection, float eta, ETX_IN(float2, alpha)) {
  if (reflection) {
    if ((wi.x == -wo.x) && (wi.y == -wo.y) && (wi.z == -wo.z)) {
      return 1.0f;
    }

    float3 wh = normalize(wi + wo);
    return wavefront_direct_light_dielectric_d_ggx((wh.z > 0.0f) ? wh : -wh, alpha);
  }

  float3 wh = normalize(wi + wo * eta);
  return wavefront_direct_light_dielectric_d_ggx((wh.z > 0.0f) ? wh : -wh, alpha);
}

ETX_SHARED_INLINE SpectralResponse wavefront_direct_light_dielectric_eval(ETX_IN(SpectralQuery, spect), ETX_INOUT(Sampler, sampler), ETX_IN(float3, wi),
  ETX_IN(float3, wo), bool wo_outside, ETX_IN(float2, alpha), ETX_IN(RefractiveIndexSample, ext_ior), ETX_IN(RefractiveIndexSample, int_ior),
  ETX_IN(ThinfilmEval, thinfilm)) {
  if ((wi.z <= 0.0f) || ((wo.z <= 0.0f) && wo_outside) || ((wo.z >= 0.0f) && (wo_outside == false))) {
    return spectral_response_make(spect, 0.0f);
  }

  WavefrontDirectLightDielectricRayInfo ray = wavefront_direct_light_dielectric_ray_info_make(-wi, alpha);
  ray = wavefront_direct_light_dielectric_ray_info_update_height(ray, 1.0f);
  bool outside = true;
  WavefrontDirectLightDielectricRayInfo ray_shadowing = wavefront_direct_light_dielectric_ray_info_make(wo_outside ? wo : -wo, alpha);

  SpectralResponse single_scattering = spectral_response_make(spect, 0.0f);
  SpectralResponse multiple_scattering = spectral_response_make(spect, 0.0f);
  float wi_mis_weight = 0.0f;
  float eta = spectral_response_monochromatic(spectral_response_div(int_ior.eta, ext_ior.eta));

  int current_scattering_order = 0;
  while (current_scattering_order < int(kWavefrontDirectLightDielectricScatteringOrderMax)) {
    ray = wavefront_direct_light_dielectric_ray_info_update_height(ray, wavefront_direct_light_dielectric_sample_height(ray, bsdf_sampler_next(sampler)));
    if (ray.h == kMaxFloat) {
      break;
    }

    current_scattering_order += 1;
    if (current_scattering_order == 1) {
      SpectralResponse phase_function =
        wavefront_direct_light_dielectric_eval_phase_function(spect, ray, wo, wo_outside, ext_ior, int_ior, thinfilm, alpha);
      float g2_g1 = 0.0f;
      if (wo_outside) {
        g2_g1 = (1.0f + (-ray.lambda_value - 1.0f)) / (1.0f + (-ray.lambda_value - 1.0f) + ray_shadowing.lambda_value);
      } else {
        g2_g1 = (1.0f + (-ray.lambda_value - 1.0f)) *
                wavefront_direct_light_dielectric_beta(1.0f + (-ray.lambda_value - 1.0f), 1.0f + ray_shadowing.lambda_value);
      }

      if (isfinite(g2_g1)) {
        single_scattering = spectral_response_mul(phase_function, g2_g1);
      }
    }

    if (current_scattering_order > 1) {
      SpectralResponse phase_function = spectral_response_make(spect, 0.0f);
      float mis = 0.0f;
      if (outside) {
        phase_function = wavefront_direct_light_dielectric_eval_phase_function(spect, ray, wo, wo_outside, ext_ior, int_ior, thinfilm, alpha);
        mis = wi_mis_weight / (wi_mis_weight + wavefront_direct_light_dielectric_mis_weight(-ray.w, wo, wo_outside, eta, alpha));
      } else {
        phase_function = wavefront_direct_light_dielectric_eval_phase_function(spect, ray, -wo, (wo_outside == false), int_ior, ext_ior, thinfilm, alpha);
        mis = wi_mis_weight / (wi_mis_weight + wavefront_direct_light_dielectric_mis_weight(-ray.w, -wo, (wo_outside == false), 1.0f / eta, alpha));
      }

      ray_shadowing = wavefront_direct_light_dielectric_ray_info_update_height(ray_shadowing, (outside == wo_outside) ? ray.h : -ray.h);
      multiple_scattering = spectral_response_add(multiple_scattering, spectral_response_mul(spectral_response_mul(phase_function, ray_shadowing.g1), mis));
    }

    float2 rnd_slope = ((current_scattering_order == 1) && bsdf_sampler_has_fixed(sampler)) ? float2(sampler.fixed_u, sampler.fixed_v) : bsdf_sampler_next_2d(sampler);
    float rnd_reflection = ((current_scattering_order == 1) && bsdf_sampler_has_fixed(sampler)) ? sampler.fixed_w : bsdf_sampler_next(sampler);
    RefractiveIndexSample phase_ext_ior = ext_ior;
    RefractiveIndexSample phase_int_ior = int_ior;
    if (outside == false) {
      phase_ext_ior = int_ior;
      phase_int_ior = ext_ior;
    }
    WavefrontDirectLightDielectricSample next_sample =
      wavefront_direct_light_dielectric_sample_phase_function(spect, rnd_slope, rnd_reflection, -ray.w, alpha, phase_ext_ior, phase_int_ior, thinfilm);
    if (next_sample.reflection) {
      ray = wavefront_direct_light_dielectric_ray_info_update_direction(ray, next_sample.w_o, alpha);
      ray = wavefront_direct_light_dielectric_ray_info_update_height(ray, ray.h);
    } else {
      outside = (outside == false);
      ray = wavefront_direct_light_dielectric_ray_info_update_direction(ray, -next_sample.w_o, alpha);
      ray = wavefront_direct_light_dielectric_ray_info_update_height(ray, -ray.h);
    }

    if (current_scattering_order == 1) {
      wi_mis_weight = wavefront_direct_light_dielectric_mis_weight(wi, ray.w, outside, eta, alpha);
    }

    if (((ray.h != ray.h) == true) || ((ray.w.x != ray.w.x) == true) || (ray.w.z <= kEpsilon)) {
      return spectral_response_make(spect, 0.0f);
    }
  }

  return spectral_response_add(spectral_response_mul(single_scattering, 0.5f), multiple_scattering);
}
