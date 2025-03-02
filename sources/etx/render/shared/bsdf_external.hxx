#pragma once

//
// Multiple-Scattering Microfacet BSDFs with the Smith Model
// Eric Heitz, Johannes Hanika, Eugene d’Eon and Carsten Dachsbacher
//
// https://eheitzresearch.wordpress.com/240-2/
//
// (adapted)

namespace etx {
namespace external {

constexpr uint32_t kScatteringOrderMax = 16u;

struct ETX_ALIGNED RayInfo {
  float3 w = {};
  float Lambda = 0.0f;
  float h = 0.0f;
  float C1 = 0.0f;
  float G1 = 0.0f;
  float pad = 0.0f;

  ETX_GPU_CODE RayInfo(const float3& w, const float2& alpha) {
    updateDirection(w, alpha);
  }

  ETX_GPU_CODE void updateDirection(const float3& in_w, const float2& alpha) {
    w = in_w;

    if (w.z > 0.9999f) {
      Lambda = 0.0f;
      return;
    }

    if (w.z < -0.9999f) {
      Lambda = -1.0f;
      return;
    }

    const float theta = acosf(w.z);
    const float cosTheta = w.z;
    const float sinTheta = sinf(theta);
    const float tanTheta = sinTheta / cosTheta;
    const float invSinTheta2 = 1.0f / (1.0f - w.z * w.z);
    const float cosPhi2 = w.x * w.x * invSinTheta2;
    const float sinPhi2 = w.y * w.y * invSinTheta2;
    const float alpha_value = sqrtf(cosPhi2 * alpha.x * alpha.x + sinPhi2 * alpha.y * alpha.y);
    const float a = 1.0f / tanTheta / alpha_value;
    Lambda = 0.5f * (-1.0f + ((a > 0) ? 1.0f : -1.0f) * sqrtf(1.0f + 1.0f / (a * a)));
  }

  ETX_GPU_CODE void updateHeight(const float& in_h) {
    h = in_h;
    ETX_ASSERT(isfinite(h));

    C1 = min(1.0f, max(0.0f, 0.5f * (h + 1.0f)));
    ETX_CHECK_FINITE(C1);

    if (w.z > 0.9999f)
      G1 = 1.0f;
    else if (w.z <= 0.0f)
      G1 = 0.0f;
    else
      G1 = powf(C1, Lambda);

    ETX_CHECK_FINITE(G1);
  }
};

ETX_GPU_CODE float invC1(const float U) {
  return max(-1.0f, min(1.0f, 2.0f * U - 1.0f));
}

ETX_GPU_CODE float sampleHeight(const RayInfo& ray, const float U) {
  if (ray.w.z > 0.9999f)
    return kMaxFloat;

  if (ray.w.z < -0.9999f) {
    return invC1(U * ray.C1);
  }

  if (fabsf(ray.w.z) < 0.0001f)
    return ray.h;

  if (U > 1.0f - ray.G1)  // leave the microsurface
    return kMaxFloat;

  // probability of intersection
  float P1 = powf((1.0f - U), 1.0f / ray.Lambda);
  ETX_VALIDATE(P1);

  if (P1 <= 0.0f)  // leave the microsurface
    return kMaxFloat;

  float U1 = ray.C1 / P1;
  ETX_VALIDATE(U1);

  float result = invC1(U1);
  ETX_ASSERT(isfinite(result));

  return result;
}

ETX_GPU_CODE float D_ggx(const float3& wm, const float2& alpha) {
  if (wm.z <= kEpsilon)
    return 0.0f;

  // slope of wm
  const float slope_x = -wm.x / wm.z;
  const float slope_y = -wm.y / wm.z;

  const float ax = fmaxf(kEpsilon, alpha.x * alpha.x);
  const float ay = fmaxf(kEpsilon, alpha.y * alpha.y);
  const float axy = fmaxf(kEpsilon, alpha.x * alpha.y);

  // P22
  const float tmp = 1.0f + slope_x * slope_x / ax + slope_y * slope_y / ay;
  ETX_VALIDATE(tmp);

  const float P22 = 1.0f / (kPi * axy * tmp * tmp);
  ETX_VALIDATE(P22);

  // value
  return P22 / (wm.z * wm.z * wm.z * wm.z);
}

ETX_GPU_CODE float2 sampleP22_11(const float theta_i, const float2& rnd, const float2& alpha) {
  float2 slope = {};

  if (theta_i < 0.0001f) {
    const float r = sqrtf(rnd.x / (1.0f - rnd.x));
    const float phi = kDoublePi * rnd.y;
    slope.x = r * cosf(phi);
    slope.y = r * sinf(phi);
    return slope;
  }

  // constant
  const float sin_theta_i = sinf(theta_i);
  const float cos_theta_i = cosf(theta_i);
  const float tan_theta_i = sin_theta_i / cos_theta_i;

  // projected area
  const float projectedarea = 0.5f * (cos_theta_i + 1.0f);
  if (projectedarea < 0.0001f)
    return {};

  // normalization coefficient
  const float c = 1.0f / projectedarea;

  const float A = 2.0f * rnd.x / cos_theta_i / c - 1.0f;
  const float B = tan_theta_i;
  const float tmp = 1.0f / (A * A - 1.0f);

  const float D = sqrtf(max(0.0f, B * B * tmp * tmp - (A * A - B * B) * tmp));
  const float slope_x_1 = B * tmp - D;
  const float slope_x_2 = B * tmp + D;
  slope.x = (A < 0.0f || slope_x_2 > 1.0f / tan_theta_i) ? slope_x_1 : slope_x_2;

  float U2;
  float S;
  if (rnd.y > 0.5f) {
    S = 1.0f;
    U2 = 2.0f * (rnd.y - 0.5f);
  } else {
    S = -1.0f;
    U2 = 2.0f * (0.5f - rnd.y);
  }

  const float z = (U2 * (U2 * (U2 * 0.27385f - 0.73369f) + 0.46341f)) / (U2 * (U2 * (U2 * 0.093073f + 0.309420f) - 1.000000f) + 0.597999f);
  slope.y = S * z * sqrtf(1.0f + slope.x * slope.x);

  return slope;
}

ETX_GPU_CODE float3 sampleVNDF(Sampler& smp, const float3& wi, const float2& alpha) {
  // sample D_wi
  // stretch to match configuration with alpha=1.0
  const float3 wi_11 = normalize(float3{alpha.x * wi.x, alpha.y * wi.y, wi.z});

  // sample visible slope with alpha=1.0
  float2 slope_11 = sampleP22_11(acosf(wi_11.z), smp.next_2d(), alpha);

  // align with view direction
  const float phi = atan2f(wi_11.y, wi_11.x);
  float2 slope = {
    cosf(phi) * slope_11.x - sinf(phi) * slope_11.y,
    sinf(phi) * slope_11.x + cosf(phi) * slope_11.y,
  };

  // stretch back
  slope.x *= alpha.x;
  slope.y *= alpha.y;

  // if numerical instability
  if ((slope.x != slope.x) || isinf(slope.x)) {
    if (wi.z > 0)
      return float3{0.0f, 0.0f, 1.0f};
    else
      return normalize(float3{wi.x, wi.y, 0.0f});
  }

  return normalize(float3{-slope.x, -slope.y, 1.0f});
}

ETX_GPU_CODE SpectralResponse phase_function_reflection(SpectralQuery spect, const RayInfo& ray, const float3& wo, const float2& alpha, const RefractiveIndex::Sample& ext_ior,
  const RefractiveIndex::Sample& int_ior, const Thinfilm::Eval& thinfilm) {
  if (ray.w.z > 0.9999f)
    return {spect, 0.0f};

  // projected area
  float projectedArea = (ray.w.z < -0.9999f) ? 1.0f : ray.Lambda * ray.w.z;
  ETX_CHECK_FINITE(projectedArea);

  if (projectedArea < kEpsilon)
    return {spect, 0.0f};

  // half float3
  const float3 wh = normalize(-ray.w + wo);
  if (wh.z < 0.0f)
    return {spect, 0.0f};

  float w_dot_h = dot(-ray.w, wh);
  if (w_dot_h < kEpsilon)
    return {spect, 0.0f};

  // value
  const auto f = fresnel::calculate(spect, w_dot_h, ext_ior, int_ior, thinfilm);
  ETX_VALIDATE(f);

  const float d_ggx = D_ggx(wh, alpha);
  ETX_VALIDATE(d_ggx);

  const float d = d_ggx / (4.0f * projectedArea);
  ETX_VALIDATE(d);

  return f * d;
}

ETX_GPU_CODE float3 samplePhaseFunction_conductor(SpectralQuery spect, Sampler& smp, const float3& wi, const float2& alpha, const RefractiveIndex::Sample& ext_ior,
  const RefractiveIndex::Sample& int_ior, const Thinfilm::Eval& thinfilm, SpectralResponse& weight) {
  // sample D_wi
  // stretch to match configuration with alpha=1.0
  const float3 wi_11 = normalize(float3{alpha.x * wi.x, alpha.y * wi.y, wi.z});

  // sample visible slope with alpha=1.0
  float2 slope_11 = sampleP22_11(acosf(wi_11.z), smp.next_2d(), alpha);

  // align with view direction
  const float phi = atan2f(wi_11.y, wi_11.x);
  float2 slope = {cosf(phi) * slope_11.x - sinf(phi) * slope_11.y, sinf(phi) * slope_11.x + cosf(phi) * slope_11.y};

  // stretch back
  slope.x *= alpha.x;
  slope.y *= alpha.y;

  // compute normal
  float3 wm = {};
  // if numerical instability
  if ((slope.x != slope.x) || isinf(slope.x)) {
    wm = (wi.z > 0) ? float3{0.0f, 0.0f, 1.0f} : normalize(float3{wi.x, wi.y, 0.0f});
  } else {
    wm = normalize(float3{-slope.x, -slope.y, 1.0f});
  }

  // reflect
  float i_dot_m = dot(wi, wm);
  weight = fresnel::calculate(spect, i_dot_m, ext_ior, int_ior, thinfilm);
  return -wi + 2.0f * wm * i_dot_m;
}

// MIS weights for bidirectional path tracing on the microsurface
ETX_GPU_CODE float MISweight_conductor(const float3& wi, const float3& wo, const float2& alpha) {
  if (wi.x == -wo.x && wi.y == -wo.y && wi.z == -wo.z)
    return 1.0f;
  const float3 wh = normalize(wi + wo);
  return D_ggx((wh.z > 0) ? wh : -wh, alpha);
}

ETX_GPU_CODE SpectralResponse eval_conductor(SpectralQuery spect, Sampler& smp, const float3& wi, const float3& wo, const float2& alpha, const RefractiveIndex::Sample& ext_ior,
  const RefractiveIndex::Sample& int_ior, const Thinfilm::Eval& thinfilm) {
  if (wi.z <= 0 || wo.z <= 0)
    return {spect, 0.0f};

  // init
  RayInfo ray = {-wi, alpha};
  ray.updateHeight(1.0f);
  SpectralResponse energy = {spect, 1.0f};

  RayInfo ray_shadowing = {wo, alpha};

  // eval single scattering
  const float3 wh = normalize(wi + wo);
  const float D = D_ggx(wh, alpha);
  const float G2 = 1.0f / (1.0f + (-ray.Lambda - 1.0f) + ray_shadowing.Lambda);
  SpectralResponse singleScattering = fresnel::calculate(spect, dot(ray.w, wh), ext_ior, int_ior, thinfilm) * D * G2 / (4.0f * wi.z);
  ETX_VALIDATE(singleScattering);

  // MIS weight
  float wi_MISweight = 0.0f;

  // multiple scattering
  SpectralResponse multipleScattering = {spect, 0.0f};

  // random walk
  uint32_t current_scatteringOrder = 0;
  while (current_scatteringOrder < kScatteringOrderMax) {
    // next height
    ray.updateHeight(sampleHeight(ray, smp.next()));

    // leave the microsurface?
    if (ray.h == kMaxFloat)
      break;

    current_scatteringOrder++;

    // next event estimation
    if (current_scatteringOrder > 1)  // single scattering is already computed
    {
      SpectralResponse phasefunction = phase_function_reflection(spect, ray, wo, alpha, ext_ior, int_ior, thinfilm);
      ETX_VALIDATE(phasefunction);

      ray_shadowing.updateHeight(ray.h);
      float shadowing = ray_shadowing.G1;
      SpectralResponse I = energy * phasefunction * shadowing;
      ETX_VALIDATE(I);

      // MIS
      const float MIS = wi_MISweight / (wi_MISweight + MISweight_conductor(-ray.w, wo, alpha));
      ETX_VALIDATE(MIS);
      multipleScattering += I * MIS;
      ETX_VALIDATE(multipleScattering);
    }

    // next direction
    SpectralResponse weight;
    ray.updateDirection(samplePhaseFunction_conductor(spect, smp, -ray.w, alpha, ext_ior, int_ior, thinfilm, weight), alpha);
    energy = energy * weight;
    ray.updateHeight(ray.h);

    if (current_scatteringOrder == 1)
      wi_MISweight = MISweight_conductor(wi, ray.w, alpha);

    // if NaN (should not happen, just in case)
    if ((ray.h != ray.h) || (ray.w.x != ray.w.x))
      return {spect, 0.0f};
  }

  // 0.5f = MIS weight of singleScattering
  // multipleScattering already weighted by MIS
  return 0.5f * singleScattering + multipleScattering;
}

ETX_GPU_CODE float abgam(float x) {
  constexpr float gam[] = {1.0f / 12.0f, 1.0f / 30.0f, 53.0f / 210.0f, 195.0f / 371.0f, 22999.0f / 22737.0f, 29944523.0f / 19733142.0f, 109535241009.0f / 48264275462.0f};
  constexpr float kHalfLogDoublePi = 0.918938518f;  // 0.5f * logf(kDoublePi)
  return kHalfLogDoublePi - x + (x - 0.5f) * logf(x) + gam[0] / (x + gam[1] / (x + gam[2] / (x + gam[3] / (x + gam[4] / (x + gam[5] / (x + gam[6] / x))))));
}

ETX_GPU_CODE float gamma(float x) {
  return expf(abgam(x + 5.0f)) / (x * (x + 1.0f) * (x + 2.0f) * (x + 3.0f) * (x + 4.0f));
}

ETX_GPU_CODE float beta(float m, float n) {
  return gamma(m) * gamma(n) / gamma(m + n);
}

ETX_GPU_CODE float3 refract(const float3& wi, const float3& wm, const float eta) {
  const float cos_theta_i = dot(wi, wm);
  const float cos_theta_t2 = 1.0f - (1.0f - cos_theta_i * cos_theta_i) / (eta * eta);
  const float cos_theta_t = -sqrtf(max(0.0f, cos_theta_t2));
  return wm * (dot(wi, wm) / eta + cos_theta_t) - wi / eta;
}

// by convention, ray is always outside
ETX_GPU_CODE SpectralResponse evalPhaseFunction_dielectric(const SpectralQuery spect, const RayInfo& ray, const float3& wo, const bool reflection,
  const RefractiveIndex::Sample& ext_ior, const RefractiveIndex::Sample& int_ior, const Thinfilm::Eval& thinfilm, const float2& alpha) {
  if (ray.w.z > 0.9999f)
    return {spect, 0.0f};

  if (reflection) {
    return phase_function_reflection(spect, ray, wo, alpha, ext_ior, int_ior, thinfilm);
  }

  float projectedArea = (ray.w.z < -0.9999f) ? 1.0f : ray.Lambda * ray.w.z;
  if (projectedArea < kEpsilon)
    return {spect, 0.0f};

  float eta = (int_ior.eta / ext_ior.eta).monochromatic();
  float3 wh = normalize(-ray.w + wo * eta);
  wh *= (wh.z > 0) ? 1.0f : -1.0f;

  float i_dot_m = -dot(wh, ray.w);
  if (i_dot_m < 0)
    return {spect, 0.0f};

  float o_dot_m = dot(wo, wh);
  float scalar = eta * eta * i_dot_m * max(0.0f, -o_dot_m) * D_ggx(wh, alpha)  //
                 / (projectedArea * sqr(i_dot_m + eta * o_dot_m));

  SpectralResponse f = fresnel::calculate(spect, i_dot_m, ext_ior, int_ior, thinfilm);
  return (1.0f - f) * scalar;
}

// by convention, wi is always outside
struct DielectricSample {
  float3 w_o = {};
  SpectralResponse weight = {};
  bool reflection = {};
};

ETX_GPU_CODE DielectricSample samplePhaseFunction_dielectric(const SpectralQuery spect, Sampler& smp, const float3& wi, const float2& alpha, const RefractiveIndex::Sample& ext_ior,
  const RefractiveIndex::Sample& int_ior, const Thinfilm::Eval& thinfilm) {
  // stretch to match configuration with alpha=1.0
  const float3 wi_11 = normalize(float3{alpha.x * wi.x, alpha.y * wi.y, wi.z});

  // sample visible slope with alpha=1.0
  float2 slope_11 = sampleP22_11(acosf(wi_11.z), smp.next_2d(), alpha);

  // align with view direction
  const float phi = atan2f(wi_11.y, wi_11.x);
  float2 slope = {
    cosf(phi) * slope_11.x - sinf(phi) * slope_11.y,
    sinf(phi) * slope_11.x + cosf(phi) * slope_11.y,
  };

  // stretch back
  slope.x *= alpha.x;
  slope.y *= alpha.y;

  // compute normal
  float3 wm = {};

  // if numerical instability
  if (isnan(slope.x) || isinf(slope.x)) {
    wm = (wi.z > 0) ? float3{0.0f, 0.0f, 1.0f} : normalize(float3{wi.x, wi.y, 0.0f});
  } else {
    wm = normalize(float3{-slope.x, -slope.y, 1.0f});
  }

  float i_dot_m = dot(wi, wm);
  auto f = fresnel::calculate(spect, i_dot_m, ext_ior, int_ior, thinfilm);
  float eta = (int_ior.eta / ext_ior.eta).monochromatic();

  DielectricSample result = {};
  result.reflection = smp.next() < f.monochromatic();
  result.weight = result.reflection ? f : 1.0f - f;
  result.w_o = result.reflection ? (-wi + 2.0f * wm * i_dot_m) : normalize(refract(wi, wm, eta));
  return result;
}

// MIS weights for bidirectional path tracing on the microsurface
ETX_GPU_CODE float MISweight_dielectric(const float3& wi, const float3& wo, const bool reflection, const float eta, const float2& alpha) {
  if (reflection) {
    if (wi.x == -wo.x && wi.y == -wo.y && wi.z == -wo.z)
      return 1.0f;
    const float3 wh = normalize(wi + wo);
    return D_ggx((wh.z > 0) ? wh : -wh, alpha);
  } else {
    const float3 wh = normalize(wi + wo * eta);
    return D_ggx((wh.z > 0) ? wh : -wh, alpha);
  }
}

ETX_GPU_CODE SpectralResponse eval_dielectric(const SpectralQuery spect, Sampler& smp, const float3& wi, const float3& wo, const bool wo_outside, const float2& alpha,
  const RefractiveIndex::Sample& ext_ior, const RefractiveIndex::Sample& int_ior, const Thinfilm::Eval& thinfilm) {
  if ((wi.z <= 0) || (wo.z <= 0 && wo_outside) || (wo.z >= 0 && !wo_outside))
    return {spect, 0.0f};

  // init
  RayInfo ray = {-wi, alpha};
  ray.updateHeight(1.0f);
  bool outside = true;

  RayInfo ray_shadowing = {wo_outside ? wo : -wo, alpha};

  SpectralResponse singleScattering = {spect, 0.0f};
  SpectralResponse multipleScattering = {spect, 0.0f};
  float wi_MISweight = 0.0f;

  float eta = (int_ior.eta / ext_ior.eta).monochromatic();

  // random walk
  int current_scatteringOrder = 0;
  while (current_scatteringOrder < kScatteringOrderMax) {
    // next height
    ray.updateHeight(sampleHeight(ray, smp.next()));

    // leave the microsurface?
    if (ray.h == kMaxFloat)
      break;

    current_scatteringOrder++;

    // next event estimation

    // single scattering
    if (current_scatteringOrder == 1) {
      auto phasefunction = evalPhaseFunction_dielectric(spect, ray, wo, wo_outside, ext_ior, int_ior, thinfilm, alpha);

      // closed masking and shadowing (we compute G2 / G1 because G1 is already in the phase function)
      float G2_G1;
      if (wo_outside)
        G2_G1 = (1.0f + (-ray.Lambda - 1.0f)) / (1.0f + (-ray.Lambda - 1.0f) + ray_shadowing.Lambda);
      else
        G2_G1 = (1.0f + (-ray.Lambda - 1.0f)) * beta(1.0f + (-ray.Lambda - 1.0f), 1.0f + ray_shadowing.Lambda);

      if (isfinite(G2_G1)) {
        singleScattering = phasefunction * G2_G1;
      }
      ETX_VALIDATE(singleScattering);
    }

    // multiple scattering
    if (current_scatteringOrder > 1) {
      SpectralResponse phasefunction = {};
      float MIS = {};
      if (outside) {
        phasefunction = evalPhaseFunction_dielectric(spect, ray, wo, wo_outside, ext_ior, int_ior, thinfilm, alpha);
        MIS = wi_MISweight / (wi_MISweight + MISweight_dielectric(-ray.w, wo, wo_outside, eta, alpha));
      } else {
        phasefunction = evalPhaseFunction_dielectric(spect, ray, -wo, !wo_outside, int_ior, ext_ior, thinfilm, alpha);
        MIS = wi_MISweight / (wi_MISweight + MISweight_dielectric(-ray.w, -wo, !wo_outside, 1.0f / eta, alpha));
      }

      ray_shadowing.updateHeight((outside == wo_outside) ? ray.h : -ray.h);

      multipleScattering += phasefunction * ray_shadowing.G1 * MIS;
      ETX_VALIDATE(multipleScattering);
    }

    // next direction
    auto next_sample = samplePhaseFunction_dielectric(spect, smp, -ray.w, alpha, (outside ? ext_ior : int_ior), (outside ? int_ior : ext_ior), thinfilm);
    if (next_sample.reflection) {
      ray.updateDirection(next_sample.w_o, alpha);
      ray.updateHeight(ray.h);
    } else {
      outside = !outside;
      ray.updateDirection(-next_sample.w_o, alpha);
      ray.updateHeight(-ray.h);
    }

    if (current_scatteringOrder == 1)
      wi_MISweight = MISweight_dielectric(wi, ray.w, outside, eta, alpha);

    if ((ray.h != ray.h) || (ray.w.x != ray.w.x) || (ray.w.z <= kEpsilon))
      return {spect, 0.0f};
  }

  // 0.5f = MIS weight of singleScattering
  // multipleScattering already weighted by MIS
  return 0.5f * singleScattering + multipleScattering;
}

ETX_GPU_CODE float3 samplePhaseFunction_diffuse(Sampler& smp, const float3& wm) {
  float r1 = 2.0f * smp.next() - 1.0f;
  float r2 = 2.0f * smp.next() - 1.0f;

  // concentric map code from
  // http://psgraphics.blogspot.ch/2011/01/improved-code-for-concentric-map.html
  float phi = 0.0f;
  float r = (r1 * r1 > r2 * r2) ? r1 : r2;

  if (r1 * r1 > r2 * r2) {
    phi = (kPi / 4.0f) * (r2 / r1);
  } else if ((r1 != 0.0f) && (r2 != 0.0f)) {
    phi = (kPi / 2.0f) - (r1 / r2) * (kPi / 4.0f);
  }

  float x = r * cosf(phi);
  float y = r * sinf(phi);
  float z = sqrtf(max(0.0f, 1.0f - x * x - y * y));

  auto basis = orthonormal_basis(wm);
  return x * basis.u + y * basis.v + z * wm;
}

ETX_GPU_CODE SpectralResponse eval_diffuse(Sampler& smp, const float3& wi, const float3& wo, const float2& alpha, const SpectralResponse& albedo) {
  RayInfo ray_shadowing = {wo, alpha};

  RayInfo ray = {-wi, alpha};
  ray.updateHeight(1.0f);

  SpectralResponse res = {albedo, 0.0f};
  SpectralResponse energy = {albedo, 1.0f};

  // random walk
  int scattering_order = 0;
  while (true) {
    // next height
    ray.updateHeight(sampleHeight(ray, smp.next()));
    // leave the microsurface?
    if (ray.h == kMaxFloat)
      break;

    // sample VNDF
    float3 wm = sampleVNDF(smp, -ray.w, alpha);

    // next event estimation
    SpectralResponse phasefunction = energy * albedo * max(0.0f, dot(wm, wo) * kInvPi);
    ETX_VALIDATE(phasefunction);

    if (scattering_order == 0) {  // closed masking and shadowing (we compute G2 / G1 because G1 is already in the phase function)
      float G2_G1 = -ray.Lambda / (ray_shadowing.Lambda - ray.Lambda);
      if (G2_G1 > 0) {
        res += phasefunction * G2_G1;
        ETX_VALIDATE(res);
      }
    } else {
      ray_shadowing.updateHeight(ray.h);
      res += phasefunction * ray_shadowing.G1;
      ETX_VALIDATE(res);
    }

    // next direction
    ray.updateDirection(samplePhaseFunction_diffuse(smp, wm), alpha);
    ray.updateHeight(ray.h);

    energy = energy * albedo;

    // if NaN (should not happen, just in case)
    if ((scattering_order++ > kScatteringOrderMax) || (ray.h != ray.h) || (ray.w.x != ray.w.x))
      return {albedo, 0.0f};
  }

  return res;
}

ETX_GPU_CODE float3 sample_diffuse(Sampler& smp, const float3& wi, const float2& alpha) {
  // init
  RayInfo ray = {-wi, alpha};
  ray.updateHeight(1.0f);

  // random walk
  int current_scatteringOrder = 0;
  while (true) {
    ray.updateHeight(sampleHeight(ray, smp.next()));
    // leave the microsurface?
    if (ray.h == kMaxFloat)
      break;

    current_scatteringOrder++;

    // sample VNDF
    float3 wm = sampleVNDF(smp, -ray.w, alpha);

    // next direction
    ray.updateDirection(samplePhaseFunction_diffuse(smp, wm), alpha);
    ray.updateHeight(ray.h);
    if (current_scatteringOrder > kScatteringOrderMax) {
      return float3{0, 0, 1};
    }
  }

  return ray.w;
}

ETX_GPU_CODE float3 sample_diffuse(Sampler& smp, const float3& wi, const float2& alpha, const SpectralResponse& albedo, SpectralResponse& energy) {
  energy = {albedo, 1.0f};

  // init
  RayInfo ray = {-wi, alpha};
  ray.updateHeight(1.0f);

  // random walk
  int current_scatteringOrder = 0;
  while (true) {
    ray.updateHeight(sampleHeight(ray, smp.next()));
    // leave the microsurface?
    if (ray.h == kMaxFloat)
      break;

    current_scatteringOrder++;

    // sample VNDF
    float3 wm = sampleVNDF(smp, -ray.w, alpha);

    // next direction
    ray.updateDirection(samplePhaseFunction_diffuse(smp, wm), alpha);
    ray.updateHeight(ray.h);

    energy = energy * albedo;

    if (current_scatteringOrder > kScatteringOrderMax) {
      energy = {albedo, 0.0f};
      return float3{0, 0, 1};
    }
  }

  return ray.w;
}

//
// VMF Diffuse: A Unified Rough Diffuse BRDF
// Eugene d’Eon and Andrea Weidlich
// Reference: https://www.shadertoy.com/view/MXKSWW
//

inline float erf(float x) {
  // https://en.wikipedia.org/wiki/Error_function (Bürmann series)
  float e = expf(-x * x);
  return (x >= 0.0f ? 1.0f : -1.0f) * 2.0f / kSqrtPI * sqrtf(1.0f - e) * (kSqrtPI / 2.0f + 31.0f / 200.0f * e - 341.0f / 8000.0f * e * e);
}

inline SpectralResponse erf(const SpectralResponse& x) {
  // https://en.wikipedia.org/wiki/Error_function (Bürmann series)
  if (x.spectral()) {
    return {x.query(), erf(x.value)};
  }

  SpectralResponse e = exp(-x * x);
  return sign(x) * 2.0f / kSqrtPI * sqrt(1.0f - e) * (kSqrtPI / 2.0f + 31.0f / 200.0f * e - 341.0f / 8000.0f * e * e);
}

inline SpectralResponse fm(float ui, float uo, float r, SpectralResponse c) {
  SpectralResponse C = sqrt(1.0f - c);
  SpectralResponse Ck = (1.0f - 0.5441615108674713f * C - 0.45302863761693374f * (1.0f - c)) / (1.0f + 1.4293127703064865f * C);
  SpectralResponse Ca = c / pow(1.0075f + 1.16942f * C, atan((0.0225272f + (-0.264641f + r) * r) * erf(c)));
  return max(0.0f, 0.384016f * (-0.341969f + Ca) * Ca * Ck * (-0.0578978f / (0.287663f + ui * uo) + fabsf(-0.0898863f + tanhf(r))));
}

inline float sigmaBeckmannExpanded(float u, float m) {
  if (0.0f == m)
    return (u + fabsf(u)) / 2.0f;

  float m2 = m * m;

  if (1.0f == u)
    return 1.0f - 0.5f * m2;

  float expansionTerm = -0.25f * m2 * (u + fabsf(u));  // accurate approximation for m < 0.25 that avoids numerical issues

  float u2 = u * u;
  return ((expf(u2 / (m2 * (-1.0f + u2))) * m * sqrtf(1.0f - u2)) / sqrtf(kPi) + u * (1.0f + erf(u / (m * sqrtf(1.0f - u2))))) / 2.0f + expansionTerm;
}

inline float coth(float x) {
  return (expf(-x) + expf(x)) / (-expf(-x) + expf(x));
}

// vmf sigma (cross section)
inline float sigmaVMF(float u, float m) {
  if (m < 0.25f)
    return sigmaBeckmannExpanded(u, m);

  float m2 = m * m;
  float m4 = m2 * m2;
  float m8 = m4 * m4;

  float u2 = u * u;
  float u4 = u2 * u2;
  float u6 = u2 * u4;
  float u8 = u4 * u4;
  float u10 = u6 * u4;
  float u12 = u6 * u6;

  float coth2m2 = coth(2.0f / m2);
  float sinh2m2 = sinhf(2.0f / m2);

  if (m > 0.9f)
    return 0.25f - 0.25f * u * (m2 - 2.0f * coth2m2) + 0.0390625f * (-1.0f + 3.0f * u2) * (4.0f + 3.0f * m4 - 6.0f * m2 * coth2m2);

  float q2 =
    1.0132789611816406e-6f * (35.0f - 1260.0f * u2 + 6930.0f * u4 - 12012.0f * u6 + 6435.0f * u8) * (1.0f + coth2m2) *
    (-256.0f - 315.0f * m4 * (128.0f + 33.0f * m4 * (80.0f + 364.0f * m4 + 195.0f * m8)) + 18.0f * m2 * (256.0f + 385.0f * m4 * (32.0f + 312.0f * m4 + 585.0f * m8)) * coth2m2) *
    sinh2m2;

  float q1 = 9.12696123123169e-8f * (-63.0f + 3465.0f * u2 - 30030.0f * u4 + 90090.0f * u6 - 109395.0f * u8 + 46189.0f * u10) * (1.0f + coth2m2) *
             (-1024.0f - 495.0f * m4 * (768.0f + 91.0f * m4 * (448.0f + 15.0f * m4 * (448.0f + 1836.0f * m4 + 969.0f * m8))) +
               110.0f * m2 * (256.0f + 117.0f * m4 * (256.0f + 21.0f * m4 * (336.0f + 85.0f * m4 * (32.0f + 57.0f * m4)))) * coth2m2) *
             sinh2m2;

  float q0 = 4.3655745685100555e-9f * (231.0f - 18018.0f * u2 + 225225.0f * u4 - 1.02102e6f * u6 + 2.078505e6f * u8 - 1.939938e6f * u10 + 676039.0f * u12) * (1.0f + coth2m2) *
             (-4096.0f - 3003.0f * m4 * (1024.0f + 45.0f * m4 * (2560.0f + 51.0f * m4 * (1792.0f + 285.0f * m4 * (80.0f + 308.0f * m4 + 161.0f * m8)))) +
               78.0f * m2 * (2048.0f + 385.0f * m4 * (1280.0f + 153.0f * m4 * (512.0f + 57.0f * m4 * (192.0f + 35.0f * m4 * (40.0f + 69.0f * m4))))) * coth2m2) *
             sinh2m2;

  return 0.25f - 0.25f * u * (m2 - 2.0f * coth2m2) + 0.0390625f * (-1.0f + 3.0f * u2) * (4.0f + 3.0f * m4 - 6.0f * m2 * coth2m2) -
         0.000732421875f * (3.0f - 30.0f * u2 + 35.0f * u4) * (16.0f + 180.0f * m4 + 105.0f * m8 - 10.0f * m2 * (8.0f + 21.0f * m4) * coth2m2) +
         0.000049591064453125f * (-5.0f + 105.0f * u2 - 315.0f * u4 + 231.0f * u6) *
           (64.0f + 105.0f * m4 * (32.0f + 180.0f * m4 + 99.0f * m8) - 42.0f * m2 * (16.0f + 240.0f * m4 + 495.0f * m8) * coth2m2) +
         (q2 / expf(2.0f / m2)) - (q1 / expf(2.0f / m2)) + (q0 / expf(2.0f / m2));
}

inline SpectralResponse vMFdiffuseBRDF(const float3& w_i, const float3& w_o, const float2& roughness, SpectralResponse albedo_in) {
  float r = clamp(sqrtf(roughness.x * roughness.y), 0.0f, 1.0f - 4.0f * kEpsilon);
  if (r == 0.0f)
    return albedo_in * kInvPi;

  SpectralResponse albedo = albedo_in;
  // SpectralResponse albedo = {};
  /*/
  {
    auto s = 0.64985f + 0.631112f * r + 1.38922f * r * r;
    ETX_VALIDATE(s);
    auto sterm = sqrt(1.0f - 2.0f * kd + sqr(kd) + 4.0f * sqr(s) * sqr(kd));
    auto nom = -1.0f + kd + sterm;
    auto denom = 2.0f * s * kd + kEpsilon;
    albedo = kd * (1.0f - sqrtf(r)) + sqrtf(r) * (nom / denom);
    ETX_VALIDATE(albedo);
  }
  // */

  float cosThetaI = w_i.z;
  float sinThetaI = sqrtf(1.0f - cosThetaI * cosThetaI);
  float cosThetaO = w_o.z;
  float sinThetaO = sqrtf(1.0f - cosThetaO * cosThetaO);

  float cosPhiDiff = 0.0;
  if (sinThetaI > 0.0f && sinThetaO > 0.0) {
    /* Compute cos(phiO-phiI) using the half-angle formulae */
    float sinPhiI = clamp(w_i.y / sinThetaI, -1.0f, 1.0f);
    float cosPhiI = clamp(w_i.x / sinThetaI, -1.0f, 1.0f);
    float sinPhiO = clamp(w_o.y / sinThetaO, -1.0f, 1.0f);
    float cosPhiO = clamp(w_o.x / sinThetaO, -1.0f, 1.0f);
    cosPhiDiff = clamp(cosPhiI * cosPhiO + sinPhiI * sinPhiO, -1.0f, 1.0f);
  }

  float phi = acosf(cosPhiDiff);
  float ui = w_i.z;
  float uo = w_o.z;
  float m = -logf(1.0f - sqrtf(r));
  float sigmai = sigmaVMF(ui, m);
  float sigmao = sigmaVMF(uo, m);
  float sigmano = sigmaVMF(-uo, m);
  float sigio = sigmai * sigmao;
  float sigdenom = uo * sigmai + ui * sigmano;

  float r2 = r * r;
  float r25 = r2 * sqrtf(r);
  float r3 = r * r2;
  float r4 = r2 * r2;
  float r45 = r4 * sqrtf(r);
  float r5 = r3 * r2;

  float ui2 = saturate(ui * ui);
  float uo2 = saturate(uo * uo);
  float sqrtuiuo = sqrtf((1.0f - ui2) * (1.0f - uo2));

  float C100 = 1.0f + (-0.1f * r + 0.84f * r4) / (1.0f + 9.0f * r3);
  float C101 = (0.0173f * r + 20.4f * r2 - 9.47f * r3) / (1.0f + 7.46f * r);
  float C102 = (-0.927f * r + 2.37f * r2) / (1.24f + r2);
  float C103 = (-0.110f * r - 1.54f * r2) / (1.0f - 1.05f * r + 7.1f * r2);
  float f10 = ((C100 + C101 * ui * uo + C102 * ui2 * uo2 + C103 * (ui2 + uo2)) * sigio) / sigdenom;

  float C110 = (0.54f * r - 0.182f * r3) / (1.0f + 1.32f * r2);
  float C111 = (-0.097f * r + 0.62f * r2 - 0.375f * r3) / (1.0f + 0.4f * r3);
  float C112 = 0.283f + 0.862f * r - 0.681f * r2;
  float f11 = (sqrtuiuo * (C110 + C111 * ui * uo)) * powf(sigio, C112) / sigdenom;

  float C120 = (2.25f * r + 5.1f * r2) / (1.0f + 9.8f * r + 32.4f * r2);
  float C121 = (-4.32f * r + 6.0f * r3) / (1.0f + 9.7f * r + 287.0f * r3);
  float f12 = ((1.0f - ui2) * (1.0f - uo2) * (C120 + C121 * uo) * (C120 + C121 * ui)) / (ui + uo);

  float C200 = (0.00056f * r + 0.226f * r2) / (1.0f + 7.07f * r2);
  float C201 = (-0.268f * r + 4.57f * r2 - 12.04f * r3) / (1.0f + 36.7f * r3);
  float C202 = (0.418f * r + 2.52f * r2 - 0.97f * r3) / (1.0f + 10.0f * r2);
  float C203 = (0.068f * r - 2.25f * r2 + 2.65f * r3) / (1.0f + 21.4f * r3);
  float C204 = (0.050f * r - 4.22f * r3) / (1.0f + 17.6f * r2 + 43.1f * r3);
  float f20 = (C200 + C201 * ui * uo + C203 * ui2 * uo2 + C202 * (ui + uo) + C204 * (ui2 + uo2)) / (ui + uo);

  float C210 = (-0.049f * r - 0.027f * r3) / (1.0f + 3.36f * r2);
  float C211 = (2.77f * r2 - 8.332f * r25 + 6.073f * r3) / (1.0f + 50.0f * r4);
  float C212 = (-0.431f * r2 - 0.295f * r3) / (1.0f + 23.9f * r3);
  float f21 = (sqrtuiuo * (C210 + C211 * ui * uo + C212 * (ui + uo))) / (ui + uo);

  float C300 = (-0.083f * r3 + 0.262f * r4) / (1.0f - 1.9f * r2 + 38.6f * r4);
  float C301 = (-0.627f * r2 + 4.95f * r25 - 2.44f * r3) / (1.0f + 31.5f * r4);
  float C302 = (0.33f * r2 + 0.31f * r25 + 1.4f * r3) / (1.0f + 20.0f * r3);
  float C303 = (-0.74f * r2 + 1.77f * r25 - 4.06f * r3) / (1.0f + 215.0f * r5);
  float C304 = (-1.026f * r3) / (1.0f + 5.81f * r2 + 13.2f * r3);
  float f30 = (C300 + C301 * ui * uo + C303 * ui2 * uo2 + C302 * (ui + uo) + C304 * (ui2 + uo2)) / (ui + uo);

  float C310 = (0.028f * r2 - 0.0132f * r3) / (1.0f + 7.46f * r2 - 3.315f * r4);
  float C311 = (-0.134f * r2 + 0.162f * r25 + 0.302f * r3) / (1.0f + 57.5f * r45);
  float C312 = (-0.119f * r2 + 0.5f * r25 - 0.207f * r3) / (1.0f + 18.7f * r3);
  float f31 = (sqrtuiuo * (C310 + C311 * ui * uo + C312 * (ui + uo))) / (ui + uo);

  auto t0 = albedo * max(0.0f, f10 + f11 * cosf(phi) * 2.0f + f12 * cosf(2.0f * phi) * 2.0f);
  ETX_VALIDATE(t0);

  auto t1 = albedo * albedo * max(0.0f, f20 + f21 * cosf(phi) * 2.0f);
  ETX_VALIDATE(t1);

  auto t2 = albedo * albedo * albedo * max(0.0f, f30 + f31 * cosf(phi) * 2.0f);
  ETX_VALIDATE(t2);

  auto t4 = fm(ui, uo, r, albedo);
  ETX_VALIDATE(t4);

  return kInvPi * (t0 + t1 + t2) + t4;
}

}  // namespace external
}  // namespace etx
