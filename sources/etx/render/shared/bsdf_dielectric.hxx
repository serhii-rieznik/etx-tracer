#include <stdlib.h>

namespace etx {

namespace DeltaDielectricBSDF {

ETX_GPU_CODE BSDFSample sample(const BSDFData& data, const Material& mtl, const Scene& scene, Sampler& smp) {
  auto frame_entering_material = data.get_normal_frame();
  auto frame = frame_entering_material.frame;
  auto entering_material = frame_entering_material.entering_material;

  float cos_theta_i = dot(frame.nrm, -data.w_i);
  if (cos_theta_i == 0.0f)
    return {{data.spectrum_sample.wavelength, 0.0f}};

  auto ior_i = (entering_material ? mtl.ext_ior : mtl.int_ior)(data.spectrum_sample);
  auto ior_o = (entering_material ? mtl.int_ior : mtl.ext_ior)(data.spectrum_sample);
  auto thinfilm = evaluate_thinfilm(data.spectrum_sample, mtl.thinfilm, data.tex, scene);
  SpectralResponse fr = fresnel::dielectric(data.spectrum_sample, data.w_i, frame.nrm, ior_i, ior_o, thinfilm);
  float f = fr.monochromatic();
  BSDFSample result;
  if (smp.next() <= f) {
    result.w_o = normalize(reflect(data.w_i, frame.nrm));
    result.pdf = f;
    ETX_VALIDATE(result.pdf);
    result.weight = (fr / f) * apply_image(data.spectrum_sample, mtl.specular, data.tex, scene);
    ETX_VALIDATE(result.weight);
    result.properties = BSDFSample::DeltaReflection;
    result.medium_index = entering_material ? mtl.ext_medium : mtl.int_medium;
  } else {
    float eta = ior_i.eta.monochromatic() / ior_o.eta.monochromatic();
    float sin_theta_o_squared = (eta * eta) * (1.0f - cos_theta_i * cos_theta_i);
    float cos_theta_o = sqrtf(clamp(1.0f - sin_theta_o_squared, 0.0f, 1.0f));
    result.w_o = normalize(eta * data.w_i + frame.nrm * (eta * cos_theta_i - cos_theta_o));

    result.pdf = 1.0f - f;
    ETX_VALIDATE(result.pdf);

    result.weight = (1.0f - fr) / (1.0f - f) * apply_image(data.spectrum_sample, mtl.transmittance, data.tex, scene);
    ETX_VALIDATE(result.weight);

    result.properties = BSDFSample::DeltaTransmission | BSDFSample::MediumChanged;
    result.medium_index = entering_material ? mtl.int_medium : mtl.ext_medium;
    if (data.path_source == PathSource::Camera) {
      result.weight *= eta * eta;
    }
  }
  return result;
}

ETX_GPU_CODE BSDFEval evaluate(const BSDFData& data, const Material& mtl, const Scene& scene, Sampler& smp) {
  return {data.spectrum_sample.wavelength, 0.0f};
}

ETX_GPU_CODE float pdf(const BSDFData& data, const Material& mtl, const Scene& scene, Sampler& smp) {
  return 0.0f;
}

}  // namespace DeltaDielectricBSDF
namespace DielectricBSDF {

ETX_GPU_CODE BSDFSample sample(const BSDFData& data, const Material& mtl, const Scene& scene, Sampler& smp) {
  if (is_delta(mtl, data.tex, scene, smp)) {
    return DeltaDielectricBSDF::sample(data, mtl, scene, smp);
  }

  auto frame_entering_material = data.get_normal_frame();
  auto frame = frame_entering_material.frame;
  auto entering_material = frame_entering_material.entering_material;
  auto ggx = NormalDistribution(frame, mtl.roughness);
  auto m = ggx.sample(smp, data.w_i);

  auto ior_i = (entering_material ? mtl.ext_ior : mtl.int_ior)(data.spectrum_sample);
  auto ior_o = (entering_material ? mtl.int_ior : mtl.ext_ior)(data.spectrum_sample);

  auto thinfilm = evaluate_thinfilm(data.spectrum_sample, mtl.thinfilm, data.tex, scene);
  SpectralResponse fr = fresnel::dielectric(data.spectrum_sample, data.w_i, m, ior_i, ior_o, thinfilm);

  bool reflection = smp.next() <= fr.monochromatic();
  BSDFData eval_data = data;
  if (reflection) {
    eval_data.w_o = reflect(data.w_i, m);
    if (dot(eval_data.w_o, frame.nrm) <= kEpsilon) {
      return {{data.spectrum_sample.wavelength, 0.0f}};
    }
    eval_data.w_o = normalize(eval_data.w_o);
  } else {
    float eta = ior_i.eta.monochromatic() / ior_o.eta.monochromatic();
    float cos_theta_i = dot(m, -data.w_i);
    float sin_theta_o_squared = (eta * eta) * (1.0f - cos_theta_i * cos_theta_i);
    float cos_theta_o = sqrtf(clamp(1.0f - sin_theta_o_squared, 0.0f, 1.0f));
    eval_data.w_o = eta * data.w_i + m * (eta * cos_theta_i - cos_theta_o);
    if (dot(eval_data.w_o, frame.nrm) >= -kEpsilon) {
      return {{data.spectrum_sample.wavelength, 0.0f}};
    }
    eval_data.w_o = normalize(eval_data.w_o);
  }

  BSDFSample result = {eval_data.w_o, evaluate(eval_data, mtl, scene, smp), reflection ? 0u : BSDFSample::MediumChanged};
  if (entering_material) {
    result.medium_index = reflection ? mtl.ext_medium : mtl.int_medium;
  } else {
    result.medium_index = reflection ? mtl.int_medium : mtl.ext_medium;
  }
  return result;
}

ETX_GPU_CODE BSDFEval evaluate(const BSDFData& data, const Material& mtl, const Scene& scene, Sampler& smp) {
  if (is_delta(mtl, data.tex, scene, smp)) {
    return DeltaDielectricBSDF::evaluate(data, mtl, scene, smp);
  }

  auto frame_entering_material = data.get_normal_frame();
  auto frame = frame_entering_material.frame;
  auto entering_material = frame_entering_material.entering_material;

  float n_dot_i = -dot(frame.nrm, data.w_i);
  float n_dot_o = dot(frame.nrm, data.w_o);
  if ((n_dot_o == 0.0f) || (n_dot_i == 0.0f)) {
    return {data.spectrum_sample.wavelength, 0.0f};
  }

  bool reflection = n_dot_o * n_dot_i >= 0.0f;
  auto ior_i = (entering_material ? mtl.ext_ior : mtl.int_ior)(data.spectrum_sample);
  auto eta_i = ior_i.eta.monochromatic();
  auto ior_o = (entering_material ? mtl.int_ior : mtl.ext_ior)(data.spectrum_sample);
  auto eta_o = ior_o.eta.monochromatic();

  float3 m = normalize(reflection ? (data.w_o - data.w_i) : (data.w_i * eta_i - data.w_o * eta_o));
  m *= (dot(frame.nrm, m) < 0.0f) ? -1.0f : 1.0f;

  float m_dot_i = -dot(m, data.w_i);
  float m_dot_o = dot(m, data.w_o);

  auto thinfilm = evaluate_thinfilm(data.spectrum_sample, mtl.thinfilm, data.tex, scene);
  SpectralResponse fr = fresnel::dielectric(data.spectrum_sample, data.w_i, m, ior_i, ior_o, thinfilm);
  float f = fr.monochromatic();

  auto ggx = NormalDistribution(frame, mtl.roughness);
  auto eval = ggx.evaluate(m, data.w_i, data.w_o);
  if (eval.pdf == 0.0f) {
    return {data.spectrum_sample.wavelength, 0.0f};
  }

  BSDFEval result = {};
  if (reflection) {
    auto specular = apply_image(data.spectrum_sample, mtl.specular, data.tex, scene);

    result.func = specular * fr * (eval.ndf * eval.visibility / (4.0f * n_dot_i * n_dot_o));
    ETX_VALIDATE(result.func);

    result.bsdf = specular * fr * (eval.ndf * eval.visibility / (4.0f * n_dot_i));
    ETX_VALIDATE(result.bsdf);

    result.weight = specular * (fr / f) * (eval.visibility / eval.g1_in);
    ETX_VALIDATE(result.weight);

    float j = 1.0f / fabsf(4.0f * m_dot_o);
    result.pdf = eval.pdf * f * j;
    ETX_VALIDATE(result.pdf);
  } else if (f < 1.0f) {
    auto transmittance = apply_image(data.spectrum_sample, mtl.transmittance, data.tex, scene);

    result.func = abs(transmittance * (1.0f - fr) * (m_dot_i * m_dot_o * sqr(eta_o) * eval.visibility * eval.ndf) / (n_dot_i * n_dot_o * sqr(m_dot_i * eta_i + m_dot_o * eta_o)));
    ETX_VALIDATE(result.func);

    result.bsdf = abs(transmittance * (1.0f - fr) * (m_dot_i * m_dot_o * sqr(eta_o) * eval.visibility * eval.ndf) / (n_dot_i * sqr(m_dot_i * eta_i + m_dot_o * eta_o)));
    ETX_VALIDATE(result.bsdf);

    result.weight = transmittance * ((1.0f - fr) / (1.0f - f)) * (eval.visibility / eval.g1_in);
    ETX_VALIDATE(result.weight);

    auto j = sqr(eta_o) * fabsf(m_dot_o) / sqr(m_dot_i * eta_i + m_dot_o * eta_o);
    result.pdf = eval.pdf * (1.0f - f) * j;
    ETX_VALIDATE(result.pdf);

    result.eta = eta_i / eta_o;
    if (data.path_source == PathSource::Camera) {
      result.bsdf *= result.eta * result.eta;
      result.weight *= result.eta * result.eta;
    }
  } else {
    result.bsdf = {data.spectrum_sample.wavelength, 0.0f};
    result.func = {data.spectrum_sample.wavelength, 0.0f};
    result.weight = {data.spectrum_sample.wavelength, 0.0f};
    result.pdf = 0.0f;
  }
  return result;
}

ETX_GPU_CODE float pdf(const BSDFData& data, const Material& mtl, const Scene& scene, Sampler& smp) {
  if (is_delta(mtl, data.tex, scene, smp)) {
    return DeltaDielectricBSDF::pdf(data, mtl, scene, smp);
  }

  auto frame_entering_material = data.get_normal_frame();
  auto frame = frame_entering_material.frame;
  auto entering_material = frame_entering_material.entering_material;
  auto ggx = NormalDistribution(frame, mtl.roughness);

  float n_dot_i = -dot(frame.nrm, data.w_i);
  float n_dot_o = dot(frame.nrm, data.w_o);
  bool reflection = n_dot_o * n_dot_i >= 0.0f;
  auto ior_i = (entering_material ? mtl.ext_ior : mtl.int_ior)(data.spectrum_sample);
  auto eta_i = ior_i.eta.monochromatic();
  auto ior_o = (entering_material ? mtl.int_ior : mtl.ext_ior)(data.spectrum_sample);
  auto eta_o = ior_o.eta.monochromatic();

  float3 m = normalize(reflection ? (data.w_o - data.w_i) : (data.w_i * eta_i - data.w_o * eta_o));
  m *= (dot(frame.nrm, m) < 0.0f) ? -1.0f : 1.0f;

  float m_dot_i = -dot(m, data.w_i);
  float m_dot_o = dot(m, data.w_o);

  auto thinfilm = evaluate_thinfilm(data.spectrum_sample, mtl.thinfilm, data.tex, scene);
  auto fr = fresnel::dielectric(data.spectrum_sample, data.w_i, frame.nrm, ior_i, ior_o, thinfilm);
  float f = fr.monochromatic();

  float pdf = ggx.pdf(m, data.w_i, data.w_o);

  if (reflection) {
    float j = 1.0f / fabsf(4.0f * m_dot_o);
    pdf *= f * j;
  } else {
    auto j = sqr(eta_o) * fabsf(m_dot_o) / sqr(m_dot_i * eta_i + m_dot_o * eta_o);
    pdf *= (1.0f - f) * j;
  }

  ETX_VALIDATE(pdf);
  return pdf;
}

ETX_GPU_CODE bool continue_tracing(const Material& material, const float2& tex, const Scene& scene, Sampler& smp) {
  return false;
}

ETX_GPU_CODE bool is_delta(const Material& material, const float2& tex, const Scene& scene, Sampler& smp) {
  return max(material.roughness.x, material.roughness.y) <= kDeltaAlphaTreshold;
}

}  // namespace DielectricBSDF

namespace ThinfilmBSDF {

ETX_GPU_CODE BSDFSample sample(const BSDFData& data, const Material& mtl, const Scene& scene, Sampler& smp) {
  auto frame_entering_material = data.get_normal_frame();
  auto frame = frame_entering_material.frame;
  auto entering_material = frame_entering_material.entering_material;
  auto ext_ior = mtl.ext_ior(data.spectrum_sample);
  auto int_ior = mtl.int_ior(data.spectrum_sample);
  auto thinfilm = evaluate_thinfilm(data.spectrum_sample, mtl.thinfilm, data.tex, scene);
  SpectralResponse fr = fresnel::dielectric(data.spectrum_sample, data.w_i, frame.nrm, ext_ior, int_ior, thinfilm);
  float f = fr.monochromatic();

  BSDFSample result = {};
  if (smp.next() <= f) {
    result.w_o = reflect(data.w_i, frame.nrm);
    if (dot(data.w_i, frame.nrm) * dot(result.w_o, frame.nrm) >= 0.0f) {
      return {{data.spectrum_sample.wavelength, 0.0f}};
    }
    result.w_o = normalize(result.w_o);
    result.pdf = f;
    result.weight = apply_image(data.spectrum_sample, mtl.specular, data.tex, scene);
    result.weight *= (fr / f);
    result.properties = BSDFSample::DeltaReflection;
    result.medium_index = entering_material ? mtl.ext_medium : mtl.int_medium;
  } else {
    result.w_o = data.w_i;
    result.pdf = 1.0f - f;
    result.weight = apply_image(data.spectrum_sample, mtl.transmittance, data.tex, scene);
    result.weight *= (1.0f - fr) / (1.0f - f);
    result.properties = BSDFSample::DeltaTransmission | BSDFSample::MediumChanged;
    result.medium_index = entering_material ? mtl.int_medium : mtl.ext_medium;
  }

  return result;
}

ETX_GPU_CODE BSDFEval evaluate(const BSDFData& data, const Material& mtl, const Scene& scene, Sampler& smp) {
  return {data.spectrum_sample.wavelength, 0.0f};
}

ETX_GPU_CODE float pdf(const BSDFData& data, const Material& mtl, const Scene& scene, Sampler& smp) {
  return 0.0f;
}

ETX_GPU_CODE bool continue_tracing(const Material& material, const float2& tex, const Scene& scene, Sampler& smp) {
  return false;
}

ETX_GPU_CODE bool is_delta(const Material& material, const float2& tex, const Scene& scene, Sampler& smp) {
  return true;
}

}  // namespace ThinfilmBSDF

namespace MultiscatteringDielectricBSDF {

ETX_GPU_CODE BSDFSample sample(const BSDFData& data, const Material& mtl, const Scene& scene, Sampler& smp) {
  if (is_delta(mtl, data.tex, scene, smp)) {
    return DeltaDielectricBSDF::sample(data, mtl, scene, smp);
  }

  LocalFrame local_frame = {data.tan, data.btn, data.nrm};
  auto w_i = local_frame.to_local(-data.w_i);
  auto ext_ior = mtl.ext_ior(data.spectrum_sample);
  auto int_ior = mtl.int_ior(data.spectrum_sample);
  auto thinfilm = evaluate_thinfilm(data.spectrum_sample, mtl.thinfilm, data.tex, scene);
  const float m_eta = int_ior.eta.monochromatic() / ext_ior.eta.monochromatic();
  const float m_invEta = 1.0f / m_eta;
  const float alpha_x = mtl.roughness.x;
  const float alpha_y = mtl.roughness.y;

  BSDFSample result = {};
  if (LocalFrame::cos_theta(w_i) > 0) {  // outside
    float weight = {};
    result.w_o = external::sample_dielectric(data.spectrum_sample, smp, w_i, alpha_x, alpha_y, ext_ior, int_ior, thinfilm, weight);

    auto a_data = data;
    a_data.w_o = normalize(local_frame.from_local(result.w_o));
    result.pdf = pdf(a_data, mtl, scene, smp);
    ETX_VALIDATE(result.pdf);

    if (LocalFrame::cos_theta(result.w_o) > 0) {  // reflection
      result.eta = 1.0f;
      result.weight = apply_image(data.spectrum_sample, mtl.specular, data.tex, scene) * weight;
    } else {  // refraction
      result.eta = m_eta;
      float factor = (data.path_source == PathSource::Camera) ? sqr(m_invEta) : 1.0f;
      result.weight = apply_image(data.spectrum_sample, mtl.transmittance, data.tex, scene) * factor * weight;
      result.properties |= BSDFSample::MediumChanged;
      result.medium_index = mtl.int_medium;
    }
  } else {  // inside
    float weight = {};
    result.w_o = -external::sample_dielectric(data.spectrum_sample, smp, -w_i, alpha_x, alpha_y, int_ior, ext_ior, thinfilm, weight);

    auto a_data = data;
    a_data.w_o = normalize(local_frame.from_local(result.w_o));
    result.pdf = pdf(a_data, mtl, scene, smp);
    ETX_VALIDATE(result.pdf);

    if (LocalFrame::cos_theta(result.w_o) > 0) {  // refraction
      result.eta = m_invEta;
      float factor = (data.path_source == PathSource::Camera) ? sqr(m_eta) : 1.0f;
      result.weight = apply_image(data.spectrum_sample, mtl.transmittance, data.tex, scene) * factor * weight;
      result.properties |= BSDFSample::MediumChanged;
      result.medium_index = mtl.ext_medium;
    } else {  // reflection
      result.eta = 1.0f;
      result.weight = apply_image(data.spectrum_sample, mtl.specular, data.tex, scene) * weight;
    }
  }

  result.w_o = normalize(local_frame.from_local(result.w_o));
  return result;
}

ETX_GPU_CODE BSDFEval evaluate(const BSDFData& data, const Material& mtl, const Scene& scene, Sampler& smp) {
  if (is_delta(mtl, data.tex, scene, smp)) {
    return DeltaDielectricBSDF::evaluate(data, mtl, scene, smp);
  }

  LocalFrame local_frame = {data.tan, data.btn, data.nrm};
  auto w_i = local_frame.to_local(-data.w_i);
  if (LocalFrame::cos_theta(w_i) == 0)
    return {data.spectrum_sample.wavelength, 0.0f};

  auto w_o = local_frame.to_local(data.w_o);
  auto ext_ior = mtl.ext_ior(data.spectrum_sample);
  auto int_ior = mtl.int_ior(data.spectrum_sample);
  auto thinfilm = evaluate_thinfilm(data.spectrum_sample, mtl.thinfilm, data.tex, scene);
  auto ext_eta = ext_ior.eta.monochromatic();
  auto int_eta = int_ior.eta.monochromatic();
  const float alpha_x = mtl.roughness.x;
  const float alpha_y = mtl.roughness.y;

  bool reflection = LocalFrame::cos_theta(w_i) * LocalFrame::cos_theta(w_o) > 0.0f;

  SpectralResponse value = {};
  if (LocalFrame::cos_theta(w_i) > 0) {
    if (LocalFrame::cos_theta(w_o) >= 0) {
      value = (smp.next() > 0.5f) ? external::eval_dielectric(data.spectrum_sample, smp, w_i, w_o, true, alpha_x, alpha_y, ext_ior, int_ior, thinfilm)
                                  : external::eval_dielectric(data.spectrum_sample, smp, w_o, w_i, true, alpha_x, alpha_y, ext_ior, int_ior, thinfilm) / LocalFrame::cos_theta(w_i);
    } else {
      value = (smp.next() > 0.5f)
                ? external::eval_dielectric(data.spectrum_sample, smp, w_i, w_o, false, alpha_x, alpha_y, ext_ior, int_ior, thinfilm)
                : external::eval_dielectric(data.spectrum_sample, smp, -w_o, -w_i, false, alpha_x, alpha_y, int_ior, ext_ior, thinfilm) / LocalFrame::cos_theta(w_i);
      value *= (data.path_source == PathSource::Camera) ? sqr(int_eta / ext_eta) : 1.0f;
    }
  } else if (LocalFrame::cos_theta(w_o) <= 0) {
    value = (smp.next() > 0.5f)
              ? external::eval_dielectric(data.spectrum_sample, smp, -w_i, -w_o, true, alpha_x, alpha_y, int_ior, ext_ior, thinfilm)
              : external::eval_dielectric(data.spectrum_sample, smp, -w_o, -w_i, true, alpha_x, alpha_y, int_ior, ext_ior, thinfilm) / LocalFrame::cos_theta(-w_i);
  } else {
    value = (smp.next() > 0.5f) ? external::eval_dielectric(data.spectrum_sample, smp, -w_i, -w_o, false, alpha_x, alpha_y, int_ior, ext_ior, thinfilm)
                                : external::eval_dielectric(data.spectrum_sample, smp, w_o, w_i, false, alpha_x, alpha_y, ext_ior, int_ior, thinfilm) / LocalFrame::cos_theta(-w_i);
    value *= (data.path_source == PathSource::Camera) ? sqr(ext_eta / int_eta) : 1.0f;
  }

  if (value.is_zero())
    return {data.spectrum_sample.wavelength, 0.0f};

  BSDFEval eval;
  eval.func = apply_image(data.spectrum_sample, reflection ? mtl.specular : mtl.transmittance, data.tex, scene) * (2.0f * value);
  ETX_VALIDATE(eval.func);
  eval.bsdf = eval.func * fabsf(LocalFrame::cos_theta(w_o));
  eval.pdf = pdf(data, mtl, scene, smp);
  eval.weight = eval.bsdf / eval.pdf;
  ETX_VALIDATE(eval.weight);
  return eval;
}

ETX_GPU_CODE float pdf(const BSDFData& data, const Material& mtl, const Scene& scene, Sampler& smp) {
  if (is_delta(mtl, data.tex, scene, smp)) {
    return DeltaDielectricBSDF::pdf(data, mtl, scene, smp);
  }

  LocalFrame local_frame = {data.tan, data.btn, data.nrm};
  auto w_i = local_frame.to_local(-data.w_i);
  if (LocalFrame::cos_theta(w_i) == 0.0f)
    return 0.0f;

  auto w_o = local_frame.to_local(data.w_o);
  auto ext_ior = mtl.ext_ior(data.spectrum_sample);
  auto int_ior = mtl.int_ior(data.spectrum_sample);
  auto thinfilm = evaluate_thinfilm(data.spectrum_sample, mtl.thinfilm, data.tex, scene);
  float m_eta = int_ior.eta.monochromatic() / ext_ior.eta.monochromatic();
  float m_invEta = 1.0f / m_eta;
  const float alpha_x = mtl.roughness.x;
  const float alpha_y = mtl.roughness.y;

  bool outside = LocalFrame::cos_theta(w_i) > 0;
  bool reflect = LocalFrame::cos_theta(w_i) * LocalFrame::cos_theta(w_o) > 0;

  float3 wh;
  float dwh_dwo;

  if (reflect) {
    wh = normalize(w_o + w_i);
    dwh_dwo = 1.0f / (4.0f * dot(w_o, wh));
  } else {
    float eta = LocalFrame::cos_theta(w_i) > 0 ? m_eta : m_invEta;
    wh = normalize(w_i + w_o * eta);
    float sqrtDenom = dot(w_i, wh) + eta * dot(w_o, wh);
    dwh_dwo = sqr(eta) * dot(w_o, wh) / sqr(sqrtDenom);
  }

  wh *= (LocalFrame::cos_theta(wh) >= 0.0f) ? 1.0f : -1.0f;

  external::RayInfo ray = {w_i * (outside ? 1.0f : -1.0f), alpha_x, alpha_y};

  auto d_ggx = external::D_ggx(wh, alpha_x, alpha_y);
  ETX_VALIDATE(d_ggx);

  float prob = max(0.0f, dot(wh, ray.w)) * d_ggx / (1.0f + ray.Lambda) / LocalFrame::cos_theta(ray.w);
  ETX_VALIDATE(prob);

  float F = fresnel::dielectric(data.spectrum_sample, w_i, wh, outside ? ext_ior : int_ior, outside ? int_ior : ext_ior, thinfilm).monochromatic();
  ETX_VALIDATE(F);

  prob *= reflect ? F : (1 - F);

  float result = fabsf(prob * dwh_dwo) + fabsf(LocalFrame::cos_theta(w_o));
  ETX_VALIDATE(result);

  return result;
}

ETX_GPU_CODE bool continue_tracing(const Material& material, const float2& tex, const Scene& scene, Sampler& smp) {
  return false;
}

ETX_GPU_CODE bool is_delta(const Material& material, const float2& tex, const Scene& scene, Sampler& smp) {
  return max(material.roughness.x, material.roughness.y) <= kDeltaAlphaTreshold;
}

}  // namespace MultiscatteringDielectricBSDF
}  // namespace etx
