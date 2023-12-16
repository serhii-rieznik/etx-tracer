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
  float Lambda = {};
  float h = {};
  float C1 = {};
  float G1 = {};
  float pad = {};

  ETX_GPU_CODE RayInfo(const float3& w, const float alpha_x, const float alpha_y) {
    updateDirection(w, alpha_x, alpha_y);
  }

  ETX_GPU_CODE void updateDirection(const float3& in_w, const float alpha_x, const float alpha_y) {
    w = in_w;
    float theta = acosf(w.z);
    float cosTheta = w.z;
    float sinTheta = sinf(theta);
    float tanTheta = sinTheta / cosTheta;
    const float invSinTheta2 = 1.0f / (1.0f - w.z * w.z);
    const float cosPhi2 = w.x * w.x * invSinTheta2;
    const float sinPhi2 = w.y * w.y * invSinTheta2;
    float alpha = sqrtf(cosPhi2 * alpha_x * alpha_x + sinPhi2 * alpha_y * alpha_y);
    // Lambda
    if (w.z > 0.9999f)
      Lambda = 0.0f;
    else if (w.z < -0.9999f)
      Lambda = -1.0f;
    else {
      const float a = 1.0f / tanTheta / alpha;
      Lambda = 0.5f * (-1.0f + ((a > 0) ? 1.0f : -1.0f) * sqrtf(1.0f + 1.0f / (a * a)));
    }
  }

  ETX_GPU_CODE void updateHeight(const float& in_h) {
    h = in_h;
    ETX_ASSERT(isfinite(h));

    C1 = min(1.0f, max(0.0f, 0.5f * (h + 1.0f)));
    ETX_VALIDATE(C1);

    if (w.z > 0.9999f)
      G1 = 1.0f;
    else if (w.z <= 0.0f)
      G1 = 0.0f;
    else
      G1 = powf(C1, Lambda);
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

ETX_GPU_CODE float D_ggx(const float3& wm, const float alpha_x, const float alpha_y) {
  if (wm.z <= kEpsilon)
    return 0.0f;

  // slope of wm
  const float slope_x = -wm.x / wm.z;
  const float slope_y = -wm.y / wm.z;

  // P22
  const float tmp = 1.0f + slope_x * slope_x / (alpha_x * alpha_x) + slope_y * slope_y / (alpha_y * alpha_y);
  ETX_VALIDATE(tmp);

  const float P22 = 1.0f / (kPi * alpha_x * alpha_y) / (tmp * tmp);
  ETX_VALIDATE(P22);

  // value
  return P22 / (wm.z * wm.z * wm.z * wm.z);
}

ETX_GPU_CODE float2 sampleP22_11(const float theta_i, const float2& rnd, const float alpha_x, const float alpha_y) {
  float2 slope;

  if (theta_i < 0.0001f) {
    const float r = sqrtf(rnd.x / (1.0f - rnd.x));
    const float phi = 6.28318530718f * rnd.y;
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
  if (projectedarea < 0.0001f || projectedarea != projectedarea)
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

ETX_GPU_CODE float3 sampleVNDF(Sampler& smp, const float3& wi, const float alpha_x, const float alpha_y) {
  // sample D_wi
  // stretch to match configuration with alpha=1.0
  const float3 wi_11 = normalize(float3{alpha_x * wi.x, alpha_y * wi.y, wi.z});

  // sample visible slope with alpha=1.0
  float2 slope_11 = sampleP22_11(acosf(wi_11.z), smp.next_2d(), alpha_x, alpha_y);

  // align with view direction
  const float phi = atan2f(wi_11.y, wi_11.x);
  float2 slope = {
    cosf(phi) * slope_11.x - sinf(phi) * slope_11.y,
    sinf(phi) * slope_11.x + cosf(phi) * slope_11.y,
  };

  // stretch back
  slope.x *= alpha_x;
  slope.y *= alpha_y;

  // if numerical instability
  if ((slope.x != slope.x) || isinf(slope.x)) {
    if (wi.z > 0)
      return float3{0.0f, 0.0f, 1.0f};
    else
      return normalize(float3{wi.x, wi.y, 0.0f});
  }

  return normalize(float3{-slope.x, -slope.y, 1.0f});
}

ETX_GPU_CODE SpectralResponse evalPhaseFunction_conductor(SpectralQuery spect, const RayInfo& ray, const float3& wo, const float alpha_x, const float alpha_y,
  const RefractiveIndex::Sample& ext_ior, const RefractiveIndex::Sample& int_ior, const Thinfilm::Eval& thinfilm) {
  if (ray.w.z > 0.9999f)
    return {spect, 0.0f};

  // half float3
  const float3 wh = normalize(-ray.w + wo);
  if (wh.z < 0.0f)
    return {spect, 0.0f};

  float w_dot_h = dot(-ray.w, wh);
  if (w_dot_h < kEpsilon)
    return {spect, 0.0f};

  // projected area
  float projectedArea = (ray.w.z < -0.9999f) ? 1.0f : ray.Lambda * ray.w.z;
  if (projectedArea < kEpsilon)
    return {spect, 0.0f};

  // value
  auto f = fresnel::conductor(spect, -ray.w, wh, ext_ior, int_ior, thinfilm);
  ETX_VALIDATE(f);

  float d_ggx = D_ggx(wh, alpha_x, alpha_y);
  ETX_VALIDATE(d_ggx);

  float d = d_ggx / (4.0f * projectedArea);
  ETX_VALIDATE(d);

  return f * d;
}

ETX_GPU_CODE float3 samplePhaseFunction_conductor(SpectralQuery spect, Sampler& smp, const float3& wi, const float alpha_x, const float alpha_y,
  const RefractiveIndex::Sample& ext_ior, const RefractiveIndex::Sample& int_ior, const Thinfilm::Eval& thinfilm, SpectralResponse& weight) {
  // sample D_wi
  // stretch to match configuration with alpha=1.0
  const float3 wi_11 = normalize(float3{alpha_x * wi.x, alpha_y * wi.y, wi.z});

  // sample visible slope with alpha=1.0
  float2 slope_11 = sampleP22_11(acosf(wi_11.z), smp.next_2d(), alpha_x, alpha_y);

  // align with view direction
  const float phi = atan2f(wi_11.y, wi_11.x);
  float2 slope = {cosf(phi) * slope_11.x - sinf(phi) * slope_11.y, sinf(phi) * slope_11.x + cosf(phi) * slope_11.y};

  // stretch back
  slope.x *= alpha_x;
  slope.y *= alpha_y;

  // compute normal
  float3 wm = {};
  // if numerical instability
  if ((slope.x != slope.x) || isinf(slope.x)) {
    wm = (wi.z > 0) ? float3{0.0f, 0.0f, 1.0f} : normalize(float3{wi.x, wi.y, 0.0f});
  } else {
    wm = normalize(float3{-slope.x, -slope.y, 1.0f});
  }

  // reflect
  weight = fresnel::conductor(spect, wi, wm, ext_ior, int_ior, thinfilm);
  return -wi + 2.0f * wm * dot(wi, wm);
}

ETX_GPU_CODE float3 sample_conductor(SpectralQuery spect, Sampler& smp, const float3& wi, const float alpha_x, const float alpha_y, const RefractiveIndex::Sample& ext_ior,
  const RefractiveIndex::Sample& int_ior, const Thinfilm::Eval& thinfilm, SpectralResponse& energy) {
  energy = {spect, 1.0f};

  // init
  RayInfo ray = {-wi, alpha_x, alpha_y};
  ray.updateHeight(1.0f);

  // random walk
  uint32_t current_scatteringOrder = 0;
  while (true) {
    // next height
    ray.updateHeight(sampleHeight(ray, smp.next()));

    // leave the microsurface?
    if (ray.h == kMaxFloat)
      break;
    current_scatteringOrder++;

    // next direction
    SpectralResponse weight;
    ray.updateDirection(samplePhaseFunction_conductor(spect, smp, -ray.w, alpha_x, alpha_y, ext_ior, int_ior, thinfilm, weight), alpha_x, alpha_y);
    energy = energy * weight;
    ray.updateHeight(ray.h);

    if ((ray.h != ray.h) || (ray.w.x != ray.w.x) || (current_scatteringOrder > kScatteringOrderMax)) {
      energy = {spect, 0.0f};
      return float3{0, 0, 1};
    }
  }

  return ray.w;
}

// MIS weights for bidirectional path tracing on the microsurface
ETX_GPU_CODE float MISweight_conductor(const float3& wi, const float3& wo, const float alpha_x, const float alpha_y) {
  if (wi.x == -wo.x && wi.y == -wo.y && wi.z == -wo.z)
    return 1.0f;
  const float3 wh = normalize(wi + wo);
  return D_ggx((wh.z > 0) ? wh : -wh, alpha_x, alpha_y);
}

ETX_GPU_CODE SpectralResponse eval_conductor(SpectralQuery spect, Sampler& smp, const float3& wi, const float3& wo, const float alpha_x, const float alpha_y,
  const RefractiveIndex::Sample& ext_ior, const RefractiveIndex::Sample& int_ior, const Thinfilm::Eval& thinfilm) {
  if (wi.z <= 0 || wo.z <= 0)
    return {spect, 0.0f};

  // init
  RayInfo ray = {-wi, alpha_x, alpha_y};
  ray.updateHeight(1.0f);
  SpectralResponse energy = {spect, 1.0f};

  RayInfo ray_shadowing = {wo, alpha_x, alpha_y};

  // eval single scattering
  const float3 wh = normalize(wi + wo);
  const float D = D_ggx(wh, alpha_x, alpha_y);
  const float G2 = 1.0f / (1.0f + (-ray.Lambda - 1.0f) + ray_shadowing.Lambda);
  SpectralResponse singleScattering = fresnel::conductor(spect, ray.w, wh, ext_ior, int_ior, thinfilm) * D * G2 / (4.0f * wi.z);
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
      SpectralResponse phasefunction = evalPhaseFunction_conductor(spect, ray, wo, alpha_x, alpha_y, ext_ior, int_ior, thinfilm);
      ETX_VALIDATE(phasefunction);

      ray_shadowing.updateHeight(ray.h);
      float shadowing = ray_shadowing.G1;
      SpectralResponse I = energy * phasefunction * shadowing;
      ETX_VALIDATE(I);

      // MIS
      const float MIS = wi_MISweight / (wi_MISweight + MISweight_conductor(-ray.w, wo, alpha_x, alpha_y));
      ETX_VALIDATE(MIS);
      multipleScattering += I * MIS;
      ETX_VALIDATE(multipleScattering);
    }

    // next direction
    SpectralResponse weight;
    ray.updateDirection(samplePhaseFunction_conductor(spect, smp, -ray.w, alpha_x, alpha_y, ext_ior, int_ior, thinfilm, weight), alpha_x, alpha_y);
    energy = energy * weight;
    ray.updateHeight(ray.h);

    if (current_scatteringOrder == 1)
      wi_MISweight = MISweight_conductor(wi, ray.w, alpha_x, alpha_y);

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
  const RefractiveIndex::Sample& ext_ior, const RefractiveIndex::Sample& int_ior, const Thinfilm::Eval& thinfilm, const float alpha_x, const float alpha_y) {
  if (ray.w.z > 0.9999f)
    return {spect, 0.0f};

  float projectedArea = (ray.w.z < -0.9999f) ? 1.0f : ray.Lambda * ray.w.z;

  if (reflection) {
    const float3 wh = normalize(-ray.w + wo);
    if (wh.z < 0.0f)
      return {spect, 0.0f};

    SpectralResponse f = fresnel::dielectric(spect, ray.w, wh, ext_ior, int_ior, thinfilm);
    float i_dot_m = -dot(wh, ray.w);
    if (i_dot_m < 0)
      return {spect, 0.0f};

    return f * i_dot_m * D_ggx(wh, alpha_x, alpha_y) / (4.0f * projectedArea * i_dot_m);
  }

  float eta = int_ior.eta.monochromatic() / ext_ior.eta.monochromatic();
  float3 wh = normalize(-ray.w + wo * eta);
  wh *= (wh.z > 0) ? 1.0f : -1.0f;

  float i_dot_m = -dot(wh, ray.w);
  if (i_dot_m < 0)
    return {spect, 0.0f};

  float o_dot_m = dot(wo, wh);
  SpectralResponse f = fresnel::dielectric(spect, ray.w, wh, ext_ior, int_ior, thinfilm);
  return eta * eta * (1.0f - f) * i_dot_m * max(0.0f, -o_dot_m) * D_ggx(wh, alpha_x, alpha_y)  //
         / (projectedArea * sqr(i_dot_m + eta * o_dot_m));
}

// by convention, wi is always outside
ETX_GPU_CODE float3 samplePhaseFunction_dielectric(const SpectralQuery spect, Sampler& smp, const float3& wi, const float alpha_x, const float alpha_y,
  const RefractiveIndex::Sample& ext_ior, const RefractiveIndex::Sample& int_ior, const Thinfilm::Eval& thinfilm, bool& reflection) {
  // stretch to match configuration with alpha=1.0
  const float3 wi_11 = normalize(float3{alpha_x * wi.x, alpha_y * wi.y, wi.z});

  // sample visible slope with alpha=1.0
  float2 slope_11 = sampleP22_11(acosf(wi_11.z), smp.next_2d(), alpha_x, alpha_y);

  // align with view direction
  const float phi = atan2f(wi_11.y, wi_11.x);
  float2 slope = {
    cosf(phi) * slope_11.x - sinf(phi) * slope_11.y,
    sinf(phi) * slope_11.x + cosf(phi) * slope_11.y,
  };

  // stretch back
  slope.x *= alpha_x;
  slope.y *= alpha_y;

  // compute normal
  float3 wm = {};

  // if numerical instability
  if (isnan(slope.x) || isinf(slope.x)) {
    wm = (wi.z > 0) ? float3{0.0f, 0.0f, 1.0f} : normalize(float3{wi.x, wi.y, 0.0f});
  } else {
    wm = normalize(float3{-slope.x, -slope.y, 1.0f});
  }

  auto f = fresnel::dielectric(spect, wi, wm, ext_ior, int_ior, thinfilm);
  reflection = smp.next() < f.monochromatic();

  float eta = int_ior.eta.monochromatic() / ext_ior.eta.monochromatic();
  return reflection ? (-wi + 2.0f * wm * dot(wi, wm)) : normalize(refract(wi, wm, eta));
}

// MIS weights for bidirectional path tracing on the microsurface
ETX_GPU_CODE float MISweight_dielectric(const float3& wi, const float3& wo, const bool reflection, const float eta, const float alpha_x, const float alpha_y) {
  if (reflection) {
    if (wi.x == -wo.x && wi.y == -wo.y && wi.z == -wo.z)
      return 1.0f;
    const float3 wh = normalize(wi + wo);
    return D_ggx((wh.z > 0) ? wh : -wh, alpha_x, alpha_y);
  } else {
    const float3 wh = normalize(wi + wo * eta);
    return D_ggx((wh.z > 0) ? wh : -wh, alpha_x, alpha_y);
  }
}

ETX_GPU_CODE SpectralResponse eval_dielectric(const SpectralQuery spect, Sampler& smp, const float3& wi, const float3& wo, const bool wo_outside, const float alpha_x,
  const float alpha_y, const RefractiveIndex::Sample& ext_ior, const RefractiveIndex::Sample& int_ior, const Thinfilm::Eval& thinfilm) {
  if ((wi.z <= 0) || (wo.z <= 0 && wo_outside) || (wo.z >= 0 && !wo_outside))
    return {spect, 0.0f};

  // init
  RayInfo ray = {-wi, alpha_x, alpha_y};
  ray.updateHeight(1.0f);
  bool outside = true;

  RayInfo ray_shadowing = {wo_outside ? wo : -wo, alpha_x, alpha_y};

  SpectralResponse singleScattering = {spect, 0.0f};
  SpectralResponse multipleScattering = {spect, 0.0f};
  float wi_MISweight = 0.0f;

  float eta = int_ior.eta.monochromatic() / ext_ior.eta.monochromatic();

  // random walk
  int current_scatteringOrder = 0;
  while (current_scatteringOrder < kScatteringOrderMax) {
    // next height
    ray.updateHeight(sampleHeight(ray, smp.next()));

    // leave the microsurface?
    if (ray.h == kMaxFloat)
      break;
    else
      current_scatteringOrder++;

    // next event estimation
    if (current_scatteringOrder == 1)  // single scattering
    {
      auto phasefunction = evalPhaseFunction_dielectric(spect, ray, wo, wo_outside, ext_ior, int_ior, thinfilm, alpha_x, alpha_y);

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

    if (current_scatteringOrder > 1)  // multiple scattering
    {
      SpectralResponse phasefunction = {};
      float MIS = {};
      if (outside) {
        phasefunction = evalPhaseFunction_dielectric(spect, ray, wo, wo_outside, ext_ior, int_ior, thinfilm, alpha_x, alpha_y);
        MIS = wi_MISweight / (wi_MISweight + MISweight_dielectric(-ray.w, wo, wo_outside, eta, alpha_x, alpha_y));
      } else {
        phasefunction = evalPhaseFunction_dielectric(spect, ray, -wo, !wo_outside, int_ior, ext_ior, thinfilm, alpha_x, alpha_y);
        MIS = wi_MISweight / (wi_MISweight + MISweight_dielectric(-ray.w, -wo, !wo_outside, 1.0f / eta, alpha_x, alpha_y));
      }

      ray_shadowing.updateHeight((outside == wo_outside) ? ray.h : -ray.h);

      multipleScattering += phasefunction * ray_shadowing.G1 * MIS;
      ETX_VALIDATE(multipleScattering);
    }

    // next direction
    bool next_outside;
    float3 w = samplePhaseFunction_dielectric(spect, smp, -ray.w, alpha_x, alpha_y, (outside ? ext_ior : int_ior), (outside ? int_ior : ext_ior), thinfilm, next_outside);
    if (next_outside) {
      ray.updateDirection(w, alpha_x, alpha_y);
      ray.updateHeight(ray.h);
    } else {
      outside = !outside;
      ray.updateDirection(-w, alpha_x, alpha_y);
      ray.updateHeight(-ray.h);
    }

    if (current_scatteringOrder == 1)
      wi_MISweight = MISweight_dielectric(wi, ray.w, outside, eta, alpha_x, alpha_y);

    if ((ray.h != ray.h) || (ray.w.x != ray.w.x))
      return {spect, 0.0f};
  }

  // 0.5f = MIS weight of singleScattering
  // multipleScattering already weighted by MIS
  return 0.5f * singleScattering + multipleScattering;
}

ETX_GPU_CODE bool sample_dielectric(const SpectralQuery spect, Sampler& smp, const float3& wi, const float alpha_x, const float alpha_y, const RefractiveIndex::Sample& ext_ior,
  const RefractiveIndex::Sample& int_ior, const Thinfilm::Eval& thinfilm, float3& w_o) {
  // init
  RayInfo ray = {-wi, alpha_x, alpha_y};
  ray.updateHeight(1.0f);
  bool outside = true;

  // random walk
  int current_scatteringOrder = 0;
  while (true) {
    // next height
    ray.updateHeight(sampleHeight(ray, smp.next()));

    // leave the microsurface?
    if (ray.h == kMaxFloat)
      break;

    current_scatteringOrder++;

    // next direction
    bool next_outside;
    float3 w = samplePhaseFunction_dielectric(spect, smp, -ray.w, alpha_x, alpha_y, (outside ? ext_ior : int_ior), (outside ? int_ior : ext_ior), thinfilm, next_outside);
    if (next_outside) {
      ray.updateDirection(w, alpha_x, alpha_y);
      ray.updateHeight(ray.h);
    } else {
      outside = !outside;
      ray.updateDirection(-w, alpha_x, alpha_y);
      ray.updateHeight(-ray.h);
    }

    if (current_scatteringOrder > kScatteringOrderMax) {
      return false;
    }
  }

  w_o = (outside) ? ray.w : -ray.w;
  return true;
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

ETX_GPU_CODE SpectralResponse eval_diffuse(Sampler& smp, const float3& wi, const float3& wo, const float alpha_x, const float alpha_y, const SpectralResponse& albedo) {
  RayInfo ray_shadowing = {wo, alpha_x, alpha_y};

  RayInfo ray = {-wi, alpha_x, alpha_y};
  ray.updateHeight(1.0f);

  SpectralResponse res = {albedo, 0.0f};
  SpectralResponse energy = {albedo, 1.0f};

  // random walk
  int current_scatteringOrder = 0;
  while (current_scatteringOrder < kScatteringOrderMax) {
    // next height
    ray.updateHeight(sampleHeight(ray, smp.next()));
    // leave the microsurface?
    if (ray.h == kMaxFloat)
      break;

    current_scatteringOrder++;

    // sample VNDF
    float3 wm = sampleVNDF(smp, -ray.w, alpha_x, alpha_y);

    // next event estimation
    SpectralResponse phasefunction = energy * albedo * max(0.0f, dot(wm, wo) / kPi);
    ETX_VALIDATE(phasefunction);

    if (current_scatteringOrder == 1) {  // closed masking and shadowing (we compute G2 / G1 because G1 is  already in the phase function)
      float G2_G1 = (1.0f + (-ray.Lambda - 1.0f)) / (1.0f + (-ray.Lambda - 1.0f) + ray_shadowing.Lambda);
      if (isfinite(G2_G1)) {
        res += phasefunction * G2_G1;
        ETX_VALIDATE(res);
      }
    } else {
      ray_shadowing.updateHeight(ray.h);
      res += phasefunction * ray_shadowing.G1;
      ETX_VALIDATE(res);
    }

    // next direction
    ray.updateDirection(samplePhaseFunction_diffuse(smp, wm), alpha_x, alpha_y);
    ray.updateHeight(ray.h);

    energy = energy * albedo;

    // if NaN (should not happen, just in case)
    if ((ray.h != ray.h) || (ray.w.x != ray.w.x))
      return {albedo, 0.0f};
  }

  return res;
}

ETX_GPU_CODE float3 sample_diffuse(Sampler& smp, const float3& wi, const float alpha_x, const float alpha_y, const SpectralResponse& albedo, SpectralResponse& energy) {
  energy = {albedo, 1.0f};

  // init
  RayInfo ray = {-wi, alpha_x, alpha_y};
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
    float3 wm = sampleVNDF(smp, -ray.w, alpha_x, alpha_y);

    // next direction
    ray.updateDirection(samplePhaseFunction_diffuse(smp, wm), alpha_x, alpha_y);
    ray.updateHeight(ray.h);

    energy = energy * albedo;

    if (current_scatteringOrder > kScatteringOrderMax) {
      energy = {albedo, 0.0f};
      return float3{0, 0, 1};
    }
  }

  return ray.w;
}

}  // namespace external
}  // namespace etx
