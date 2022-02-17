#pragma once

#include <etx/render/shared/material.hxx>
#include <etx/render/shared/sampler.hxx>

namespace etx {

enum class PathSource : uint32_t {
  Undefined,
  Camera,
  Light,
};

struct BSDFData : public Vertex {
  ETX_GPU_CODE BSDFData(SpectralQuery spect, uint32_t medium, const Material& m, PathSource tm, const Vertex& av, const float3& awi, const float3& awo)
    : Vertex(av)
    , material(m)
    , mode(tm)
    , spectrum_sample(spect)
    , medium_index(medium)
    , w_i(awi)
    , w_o(awo) {
  }

  ETX_GPU_CODE BSDFData(const BSDFData& data, const Material& m)
    : Vertex(data)
    , material(m)
    , mode(data.mode)
    , spectrum_sample(data.spectrum_sample)
    , medium_index(data.medium_index)
    , w_i(data.w_i)
    , w_o(data.w_o) {
  }

  ETX_GPU_CODE BSDFData swap_directions() const {
    BSDFData result = *this;
    result.w_i = -w_o;
    result.w_o = -w_i;
    return result;
  }

  ETX_GPU_CODE bool check_side(Frame& f_out) const {
    float n_dot_i = dot(nrm, w_i);

    if (material.double_sided()) {
      float scale = (n_dot_i >= 0.0f ? -1.0f : +1.0f);
      f_out = {tan * scale, btn * scale, nrm * scale};
      return true;
    }

    f_out = {tan, btn, nrm};
    return (n_dot_i < 0.0f);
  }

  ETX_GPU_CODE bool get_normal_frame(Frame& f_out) const {
    bool entering_material = dot(nrm, w_i) < 0.0f;
    if (entering_material) {
      f_out = {tan, btn, nrm};
    } else {
      f_out = {-tan, -btn, -nrm};
    }
    return entering_material;
  }

  const Material& material;
  PathSource mode = PathSource::Undefined;
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
    return (pdf > 0.0f) && bsdf.valid();
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
    return (pdf > 0.0f) && weight.valid();
  }

  ETX_GPU_CODE bool is_diffuse() const {
    return (properties & Diffuse) != 0;
  }

  ETX_GPU_CODE bool is_delta() const {
    return ((properties & DeltaReflection) != 0) || ((properties & DeltaTransmission) != 0);
  }
};

struct LocalFrame : public Frame {
  ETX_GPU_CODE LocalFrame(const Frame& f)
    : Frame(f) {
    _to_local = {
      {tan.x, btn.x, nrm.x},
      {tan.y, btn.y, nrm.y},
      {tan.z, btn.z, nrm.z},
    };
    _from_local = transpose(_to_local);
  }

  ETX_GPU_CODE float3 to_local(const float3& v) const {
    return _to_local * v;
  }

  ETX_GPU_CODE float3 from_local(const float3& v) const {
    return _from_local * v;
  }

  ETX_GPU_CODE static float cos_theta(const float3& v) {
    return v.z;
  }

 private:
  float3x3 _to_local = {};
  float3x3 _from_local = {};
};

struct NormalDistribution {
  struct Eval {
    float ndf = 0.0f;
    float g1_in = 0.0f;
    float visibility = 0.0f;
    float pdf = 0.0f;
  };

  ETX_GPU_CODE NormalDistribution(const Frame& f, const float2& alpha)
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
  return (w_o_s * w_i_g == 0.0f) ? 0.0f : fabsf((w_o_g * w_i_s) / (w_o_s * w_i_g));
}

namespace fresnel {

ETX_GPU_CODE auto reflectance(float ext_ior, float cos_theta_i, float int_ior, float cos_theta_j) {
  struct result {
    float rs, rp;
  };
  auto ni = ext_ior;
  auto nj = int_ior;
  auto rs = (ni * cos_theta_i - nj * cos_theta_j) / (ni * cos_theta_i + nj * cos_theta_j);
  auto rp = (nj * cos_theta_i - ni * cos_theta_j) / (nj * cos_theta_i + ni * cos_theta_j);
  return result{rs, rp};
}

ETX_GPU_CODE auto transmittance(float ext_ior, float cos_theta_i, float int_ior, float cos_theta_j) {
  struct result {
    float ts, tp;
  };
  auto ni = ext_ior;
  auto nj = int_ior;
  auto ts = (2.0f * ni * cos_theta_i) / (ni * cos_theta_i + nj * cos_theta_j);
  auto tp = (2.0f * ni * cos_theta_i) / (ni * cos_theta_j + nj * cos_theta_i);
  return result{ts, tp};
}

ETX_GPU_CODE SpectralResponse dielectric(SpectralQuery spect, const float cos_theta_i, float ext_ior, float int_ior) {
  SpectralResponse result = {spect.wavelength, 1.0f};
  float eta = (ext_ior / int_ior);
  float sin_theta_o_squared = (eta * eta) * (1.0f - cos_theta_i * cos_theta_i);
  if (sin_theta_o_squared <= 1.0) {
    float cos_theta_o = sqrtf(clamp(1.0f - sin_theta_o_squared, 0.0f, 1.0f));
    auto rsrp = reflectance(ext_ior, cos_theta_i, int_ior, cos_theta_o);
    result = SpectralResponse{spect.wavelength, 0.5f * (rsrp.rs * rsrp.rs + rsrp.rp * rsrp.rp)};
  }
  return result;
}

ETX_GPU_CODE SpectralResponse dielectric(SpectralQuery spect, const float3& i, const float3& m, float ext_ior, float int_ior) {
  return dielectric(spect, fabsf(dot(i, m)), ext_ior, int_ior);
}

ETX_GPU_CODE SpectralResponse dielectric_thinfilm(SpectralQuery spect, const float cos_theta_0, float ext_ior, float film_ior, float int_ior, float thickness) {
  float eta_01 = (ext_ior / film_ior);
  float sin_theta_1_squared = sqr(eta_01) * (1.0f - cos_theta_0 * cos_theta_0);
  if (sin_theta_1_squared >= 1.0f) {
    return {spect.wavelength, 1.0f};
  }

  float cos_theta_1 = sqrtf(1.0f - sin_theta_1_squared);

  float eta_12 = (film_ior / int_ior);
  float sin_theta_2_squared = sqr(eta_12) * (1.0f - cos_theta_1 * cos_theta_1);
  if (sin_theta_2_squared >= 1.0f) {
    return {spect.wavelength, 1.0f};
  }

  float cos_theta_2 = sqrtf(1.0f - sin_theta_2_squared);

  float delta_10 = film_ior > ext_ior ? 0.0f : kPi;
  float delta_21 = int_ior > film_ior ? 0.0f : kPi;
  float phase_shift = delta_10 + delta_21;

  SpectralResponse phi = {spect.wavelength, kDoublePi * 2.0f * thickness * cos_theta_1 + phase_shift * film_ior};
  if constexpr (spectrum::kSpectralRendering) {
    phi /= spect.wavelength;
  } else {
    phi /= {spect.wavelength, {690.0f, 550.0f, 430.0f}};
  }

  auto r10 = reflectance(film_ior, cos_theta_1, ext_ior, cos_theta_0);
  auto r12 = reflectance(film_ior, cos_theta_1, int_ior, cos_theta_2);
  auto t01 = transmittance(ext_ior, cos_theta_0, film_ior, cos_theta_1);
  auto t12 = transmittance(film_ior, cos_theta_1, int_ior, cos_theta_2);

  auto alpha_p = r10.rp * r12.rp;
  auto beta_p = t01.tp * t12.tp;
  auto tp = (beta_p * beta_p) / ((alpha_p * alpha_p) - 2.0f * alpha_p * cos(phi) + 1.0f);

  auto alpha_s = r10.rs * r12.rs;
  auto beta_s = t01.ts * t12.ts;
  auto ts = (beta_s * beta_s) / ((alpha_s * alpha_s) - 2.0f * alpha_s * cos(phi) + 1.0f);
  auto ratio = (int_ior * cos_theta_2) / (ext_ior * cos_theta_0);

  return 1.0f - ratio * 0.5f * (tp + ts);
}

ETX_GPU_CODE SpectralResponse dielectric_thinfilm(SpectralQuery spect, const float3& i, const float3& m, float ext_ior, float film_ior, float int_ior, float thickness) {
  return dielectric_thinfilm(spect, fabsf(dot(i, m)), ext_ior, film_ior, int_ior, thickness);
}

ETX_GPU_CODE SpectralResponse conductor(SpectralQuery spect, const float cos_theta, const RefractiveIndex::Sample& sample_out, const RefractiveIndex::Sample& sample_in) {
  ETX_ASSERT(spect.wavelength == sample_in.wavelength);
  ETX_ASSERT(spect.wavelength == sample_out.wavelength);

  float cos_theta_2 = cos_theta * cos_theta;
  float sin_theta_2 = clamp(1.0f - cos_theta_2, 0.0f, 1.0f);

  bool entring_material = (cos_theta > 0.0f);
  RefractiveIndex::Sample ior_sample = (entring_material) ? (sample_in / sample_out) : (sample_out / sample_in);

  SpectralResponse eta_2 = ior_sample.eta * ior_sample.eta;
  SpectralResponse k_2 = ior_sample.k * ior_sample.k;
  SpectralResponse t0 = eta_2 - k_2 - sin_theta_2;
  SpectralResponse a2plusb2 = sqrt(t0 * t0 + 4.0f * eta_2 * k_2);

  SpectralResponse t1 = a2plusb2 + cos_theta_2;
  SpectralResponse t2 = 2.0f * sqrt(0.5f * (a2plusb2 + t0)) * fabsf(cos_theta);
  SpectralResponse Rs = abs((t1 - t2) / (t1 + t2));
  ETX_VALIDATE(Rs);

  SpectralResponse t3 = cos_theta_2 * a2plusb2 + sin_theta_2 * sin_theta_2;
  SpectralResponse t4 = t2 * sin_theta_2;
  SpectralResponse Rp = Rs * ((t3 + t4).is_zero() ? SpectralResponse(spect.wavelength, 1.0f) : (t3 - t4) / (t3 + t4));
  ETX_VALIDATE(Rp);

  return 0.5f * (Rp + Rs);
}

ETX_GPU_CODE SpectralResponse conductor(SpectralQuery spect, const float3& i, const float3& m, const RefractiveIndex::Sample& sample_out,
  const RefractiveIndex::Sample& sample_in) {
  return conductor(spect, fabsf(dot(i, m)), sample_out, sample_in);
}

}  // namespace fresnel

}  // namespace etx
