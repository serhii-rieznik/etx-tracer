#pragma once

#include <etx/render/shared/material.hxx>
#include <etx/render/shared/sampler.hxx>

namespace etx {

// #define ETX_FORCED_BSDF PlasticBSDF

enum class PathSource : uint32_t {
  Undefined,
  Camera,
  Light,
};

struct BSDFData : public Vertex {
  ETX_GPU_CODE BSDFData(SpectralQuery spect, uint32_t medium, PathSource ps, const Vertex& av, const float3& awi, const float3& awo)
    : Vertex(av)
    , path_source(ps)
    , spectrum_sample(spect)
    , medium_index(medium)
    , w_i(awi)
    , w_o(awo) {
  }

  ETX_GPU_CODE BSDFData swap_directions() const {
    BSDFData result = *this;
    result.w_i = -w_o;
    result.w_o = -w_i;
    return result;
  }

  ETX_GPU_CODE struct {
    LocalFrame frame;
    bool entering_material;
  } get_normal_frame() const {
    bool entering_material = dot(nrm, w_i) < 0.0f;
    return {entering_material ? LocalFrame{tan, btn, nrm} : LocalFrame{-tan, -btn, -nrm}, entering_material};
  }

  PathSource path_source = PathSource::Undefined;
  SpectralQuery spectrum_sample;
  uint32_t medium_index = kInvalidIndex;
  float3 w_i = {};
  float3 w_o = {};
};

struct BSDFEval {
  BSDFEval() = default;

  ETX_GPU_CODE BSDFEval(float wl, float power)
    : weight(wl, power)
    , bsdf(wl, power) {
  }

  SpectralResponse func = {};
  SpectralResponse bsdf = {};
  SpectralResponse weight = {};
  float pdf = 0.0f;
  float eta = 1.0f;

  ETX_GPU_CODE bool valid() const {
    return (pdf > 0.0f);
  }
};

struct BSDFSample {
  enum Properties : uint32_t {
    Diffuse = 1u << 0u,
    DeltaReflection = 1u << 1u,
    DeltaTransmission = 1u << 2u,
    MediumChanged = 1u << 3u,
  };

  SpectralResponse weight = {};
  float3 w_o = {};
  float pdf = 0.0f;
  float eta = 1.0f;
  uint32_t properties = 0u;
  uint32_t medium_index = kInvalidIndex;

  BSDFSample() = default;

  ETX_GPU_CODE BSDFSample(const SpectralResponse& a_weight)
    : weight(a_weight) {
  }

  ETX_GPU_CODE BSDFSample(const float3& a_w_o, const SpectralResponse& a_weight, float a_pdf, float a_eta, uint32_t props)
    : weight(a_weight)
    , w_o(a_w_o)
    , pdf(a_pdf)
    , eta(a_eta)
    , properties(props) {
  }

  ETX_GPU_CODE BSDFSample(const float3& w, const BSDFEval& eval, uint32_t props)
    : weight(eval.weight)
    , w_o(w)
    , pdf(eval.pdf)
    , eta(eval.eta)
    , properties(props) {
  }

  ETX_GPU_CODE bool valid() const {
    return (pdf > 0.0f);
  }

  ETX_GPU_CODE bool is_diffuse() const {
    return (properties & Diffuse) != 0;
  }

  ETX_GPU_CODE bool is_delta() const {
    return ((properties & DeltaReflection) != 0) || ((properties & DeltaTransmission) != 0);
  }
};

struct NormalDistribution {
  struct Eval {
    float ndf = 0.0f;
    float g1_in = 0.0f;
    float visibility = 0.0f;
    float pdf = 0.0f;
  };

  ETX_GPU_CODE NormalDistribution(const LocalFrame& f, const float2& alpha)
    : _frame(f)
    , _alpha(alpha) {
  }

  [[nodiscard]] ETX_GPU_CODE float3 sample(Sampler& smp, const float3& in_w_i) const {
    auto w_i = _frame.to_local(-in_w_i);
    auto v_h = normalize(float3{_alpha.x * w_i.x, _alpha.y * w_i.y, w_i.z});

    float v_h_len = v_h.x * v_h.x + v_h.y * v_h.y;
    float3 u = v_h_len > 0.0f ? float3{-v_h.y, v_h.x, 0.0f} / sqrtf(v_h_len) : float3{1.0f, 0.0f, 0.0f};
    float3 v = cross(v_h, u);

    float r = sqrtf(smp.next());
    float phi = kDoublePi * smp.next();
    float t1 = r * cosf(phi);
    float t2 = r * sinf(phi);
    float s = 0.5f * (1.0f + v_h.z);
    t2 = (1.0f - s) * sqrtf(1.0f - t1 * t1) + s * t2;
    float3 n_h = t1 * u + t2 * v + sqrtf(max(0.0f, 1.0f - t1 * t1 - t2 * t2)) * v_h;
    float3 local_m = normalize(float3{_alpha.x * n_h.x, _alpha.y * n_h.y, n_h.z});

    return _frame.from_local(local_m);
  }

  [[nodiscard]] ETX_GPU_CODE Eval evaluate(const float3& in_m, const float3& in_w_i, const float3& in_w_o) const {
    auto local_w_i = _frame.to_local(-in_w_i);
    if (local_w_i.z <= kEpsilon) {
      return {};
    }

    auto local_w_o = _frame.to_local(in_w_o);
    auto local_m = _frame.to_local(in_m);

    Eval result = {};
    result.visibility = visibility_term_local(_alpha, local_m, local_w_i, local_w_o);
    ETX_VALIDATE(result.visibility);
    result.ndf = normal_distribution_local(_alpha, local_m);
    ETX_VALIDATE(result.ndf);
    result.g1_in = visibility_local(_alpha, local_m, local_w_i);
    ETX_VALIDATE(result.g1_in);
    {
      float s = fabsf(dot(local_w_i, local_m)) / local_w_i.z;
      ETX_VALIDATE(s);
      result.pdf = result.ndf * result.g1_in * s;
      ETX_VALIDATE(result.pdf);
    }
    return result;
  }

  [[nodiscard]] ETX_GPU_CODE float pdf(const float3& in_m, const float3& in_w_i, const float3& in_w_o) const {
    auto local_w_i = _frame.to_local(-in_w_i);
    if (local_w_i.z <= kEpsilon) {
      return 0.0f;
    }

    auto local_m = _frame.to_local(in_m);

    float g1 = visibility_local(_alpha, local_m, local_w_i);
    ETX_VALIDATE(g1);
    float d = normal_distribution_local(_alpha, local_m);
    ETX_VALIDATE(d);
    float s = fabsf(dot(local_w_i, local_m)) / local_w_i.z;
    ETX_VALIDATE(s);
    return g1 * d * s;
  }

 private:
  ETX_GPU_CODE float lambda_local(const float2& alpha, const float3& w) const {
    float a = sqrtf(alpha.x * alpha.y);
    float n_dot_w = fabsf(w.z);
    return 0.5f * (sqrtf(a + (1.0f - a) * n_dot_w * n_dot_w) / n_dot_w - 1.0f);
  }

  // G1
  ETX_GPU_CODE float visibility_local(const float2& alpha, const float3& m, const float3& w) const {
    float xy_alpha_2 = sqr(alpha.x * w.x) + sqr(alpha.y * w.y);
    if (xy_alpha_2 == 0.0f) {
      return 1.0f;
    }

    if (dot(w, m) * w.z <= 0.0f) {
      return 0.0f;
    }

    float tan_theta_alpha_2 = xy_alpha_2 / sqr(w.z);
    float result = 2.0f / (1.0f + sqrtf(1.0f + tan_theta_alpha_2));
    ETX_VALIDATE(result);
    return result;
  }

  // G
  ETX_GPU_CODE float visibility_term_local(const float2& alpha, const float3& m, const float3& w_i, const float3& w_o) const {
    return visibility_local(alpha, m, w_i) * visibility_local(alpha, m, w_o);
  }

  // D
  ETX_GPU_CODE float normal_distribution_local(const float2& alpha, const float3& m) const {
    float alpha_uv = alpha.x * alpha.y;
    float result = 1.0f / (kPi * alpha_uv * sqr(sqr(m.x / alpha.x) + sqr(m.y / alpha.y) + sqr(m.z)));
    ETX_VALIDATE(result);
    return result;
  }

 private:
  LocalFrame _frame;
  float2 _alpha = {};
};

ETX_GPU_CODE float fix_shading_normal(const float3& n_g, const float3& n_s, const float3& w_i, const float3& w_o) {
  float w_i_g = dot(w_i, n_g);
  float w_i_s = dot(w_i, n_s);
  float w_o_g = dot(w_o, n_g);
  float w_o_s = dot(w_o, n_s);
  return (fabsf(w_o_s * w_i_g) < kRayEpsilon) ? 0.0f : fabsf((w_o_g * w_i_s) / (w_o_s * w_i_g));
}

namespace fresnel {

ETX_GPU_CODE auto reflectance(const complex& ext_ior, const complex& cos_theta_i, const complex& int_ior, const complex& cos_theta_j) {
  struct result {
    complex rs, rp;
  };
  auto ni = ext_ior;
  auto nj = int_ior;
  auto rs = (ni * cos_theta_i - nj * cos_theta_j) / (ni * cos_theta_i + nj * cos_theta_j);
  ETX_CHECK_FINITE(rs);
  auto rp = (nj * cos_theta_i - ni * cos_theta_j) / (nj * cos_theta_i + ni * cos_theta_j);
  ETX_CHECK_FINITE(rp);
  return result{rs, rp};
}

ETX_GPU_CODE auto transmittance(const complex& ext_ior, const complex& cos_theta_i, const complex& int_ior, const complex& cos_theta_j) {
  struct result {
    complex ts, tp;
  };
  auto ni = ext_ior;
  auto nj = int_ior;
  auto ts = (2.0f * ni * cos_theta_i) / (ni * cos_theta_i + nj * cos_theta_j);
  ETX_CHECK_FINITE(ts);
  auto tp = (2.0f * ni * cos_theta_i) / (ni * cos_theta_j + nj * cos_theta_i);
  ETX_CHECK_FINITE(tp);
  return result{ts, tp};
}

ETX_GPU_CODE float fresnel_generic(const float cos_theta_i, const complex& ext_ior, const complex& int_ior) {
  auto sin_theta_o_squared = sqr(ext_ior / int_ior) * (1.0f - cos_theta_i * cos_theta_i);
  auto cos_theta_o = complex_sqrt(complex{1.0f, 0.0f} - sin_theta_o_squared);
  ETX_VALIDATE(cos_theta_o);
  auto rsrp = reflectance(ext_ior, cos_theta_i, int_ior, cos_theta_o);
  return saturate(0.5f * (complex_norm(rsrp.rs) + complex_norm(rsrp.rp)));
}

ETX_GPU_CODE float fresnel_thinfilm(float wavelength, const float cos_theta_0, const complex& ext_ior, const complex& film_ior, const complex& int_ior, float thickness) {
  auto sin_theta_1_squared = sqr(ext_ior / film_ior) * (1.0f - cos_theta_0 * cos_theta_0);
  auto cos_theta_1 = complex_sqrt(complex{1.0f, 0.0f} - sin_theta_1_squared);
  auto sin_theta_2_squared = sqr(film_ior / int_ior) * (1.0f - cos_theta_1 * cos_theta_1);
  auto cos_theta_2 = complex_sqrt(1.0f - sin_theta_2_squared);
  auto delta_10 = film_ior.real() > ext_ior.real() ? 0.0f : kPi;
  auto delta_21 = int_ior.real() > film_ior.real() ? 0.0f : kPi;
  auto phase_shift = delta_10 + delta_21;
  auto phi = (kDoublePi * 2.0f * thickness * cos_theta_1 + phase_shift * film_ior) / wavelength;
  auto r10 = reflectance(film_ior, cos_theta_1, ext_ior, cos_theta_0);
  auto r12 = reflectance(film_ior, cos_theta_1, int_ior, cos_theta_2);
  auto t01 = transmittance(ext_ior, cos_theta_0, film_ior, cos_theta_1);
  auto t12 = transmittance(film_ior, cos_theta_1, int_ior, cos_theta_2);
  auto alpha_p = r10.rp * r12.rp;
  auto beta_p = t01.tp * t12.tp;

  auto tp = (beta_p * beta_p) / ((alpha_p * alpha_p) - 2.0f * alpha_p * complex_cos(phi) + 1.0f);
  ETX_CHECK_FINITE(tp);

  auto alpha_s = r10.rs * r12.rs;
  auto beta_s = t01.ts * t12.ts;

  auto ts = (beta_s * beta_s) / ((alpha_s * alpha_s) - 2.0f * alpha_s * complex_cos(phi) + 1.0f);
  ETX_CHECK_FINITE(ts);

  auto ratio = (int_ior * cos_theta_2) / (ext_ior * cos_theta_0);
  ETX_CHECK_FINITE(ratio);

  return complex_abs(1.0f - ratio * 0.5f * (tp + ts));
}

ETX_GPU_CODE SpectralResponse conductor(SpectralQuery spect, const float3& i, const float3& m, const RefractiveIndex::Sample& ext_ior, const RefractiveIndex::Sample& int_ior,
  const Thinfilm::Eval& thinfilm) {
  ETX_ASSERT(spect.wavelength == ext_ior.wavelength);
  ETX_ASSERT(spect.wavelength == int_ior.wavelength);

  float cos_theta = fabsf(dot(i, m));
  if (thinfilm.thickness == 0.0f) {
    SpectralResponse result = {spect.wavelength, fresnel_generic(cos_theta, ext_ior.as_complex_x(), int_ior.as_complex_x())};
    if constexpr (spectrum::kSpectralRendering == false) {
      result.components.y = fresnel_generic(cos_theta, ext_ior.as_complex_y(), int_ior.as_complex_y());
      result.components.z = fresnel_generic(cos_theta, ext_ior.as_complex_z(), int_ior.as_complex_z());
    }
    return result;
  }

  const auto& flm_ior = thinfilm.ior;
  SpectralResponse result = {spect.wavelength, 0.0f};
  if constexpr (spectrum::kSpectralRendering) {
    result.components.x = fresnel_thinfilm(spect.wavelength, cos_theta, ext_ior.as_complex_x(), flm_ior.as_complex_x(), int_ior.as_complex_x(), thinfilm.thickness);
  } else {
    result.components.x = fresnel_thinfilm(690.0f, cos_theta, ext_ior.as_complex_x(), flm_ior.as_complex_x(), int_ior.as_complex_x(), thinfilm.thickness);
    result.components.y = fresnel_thinfilm(550.0f, cos_theta, ext_ior.as_complex_y(), flm_ior.as_complex_y(), int_ior.as_complex_y(), thinfilm.thickness);
    result.components.z = fresnel_thinfilm(430.0f, cos_theta, ext_ior.as_complex_z(), flm_ior.as_complex_z(), int_ior.as_complex_z(), thinfilm.thickness);
  }
  return result;
}

ETX_GPU_CODE SpectralResponse dielectric(SpectralQuery spect, const float3& i, const float3& m, const RefractiveIndex::Sample ext_ior, const RefractiveIndex::Sample int_ior,
  const Thinfilm::Eval& thinfilm) {
  float cos_theta = fabsf(dot(i, m));
  auto c_e = ext_ior.as_monochromatic_complex();
  auto c_i = int_ior.as_monochromatic_complex();
  if (thinfilm.thickness == 0.0f) {
    return {spect.wavelength, fresnel_generic(cos_theta, c_e, c_i)};
  }

  auto c_f = thinfilm.ior.as_monochromatic_complex();

  SpectralResponse result = {spect.wavelength, 0.0f};
  if constexpr (spectrum::kSpectralRendering) {
    result.components.x = fresnel_thinfilm(spect.wavelength, cos_theta, c_e, c_f, c_i, thinfilm.thickness);
  } else {
    result.components.x = fresnel_thinfilm(690.0f, cos_theta, c_e, c_f, c_i, thinfilm.thickness);
    result.components.y = fresnel_thinfilm(550.0f, cos_theta, c_e, c_f, c_i, thinfilm.thickness);
    result.components.z = fresnel_thinfilm(430.0f, cos_theta, c_e, c_f, c_i, thinfilm.thickness);
  }
  return result;
}

}  // namespace fresnel

}  // namespace etx
