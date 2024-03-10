namespace etx {

namespace DeltaDielectricBSDF {

ETX_GPU_CODE BSDFSample sample(const BSDFData& data, const Material& mtl, const Scene& scene, Sampler& smp) {
  auto frame = data.get_normal_frame();

  float cos_theta_i = dot(frame.nrm, -data.w_i);
  if (cos_theta_i == 0.0f)
    return {{data.spectrum_sample, 0.0f}};

  auto ior_i = (frame.entering_material() ? mtl.ext_ior : mtl.int_ior)(data.spectrum_sample);
  auto ior_o = (frame.entering_material() ? mtl.int_ior : mtl.ext_ior)(data.spectrum_sample);
  auto thinfilm = evaluate_thinfilm(data.spectrum_sample, mtl.thinfilm, data.tex, scene);
  SpectralResponse fr = fresnel::calculate(data.spectrum_sample, cos_theta_i, ior_i, ior_o, thinfilm);
  float f = fr.monochromatic();
  BSDFSample result;
  if (smp.next() <= f) {
    result.w_o = normalize(reflect(data.w_i, frame.nrm));
    result.pdf = f;
    ETX_VALIDATE(result.pdf);
    result.weight = (fr / f) * apply_image(data.spectrum_sample, mtl.specular, data.tex, scene, rgb::SpectrumClass::Reflection);
    ETX_VALIDATE(result.weight);
    result.properties = BSDFSample::Delta | BSDFSample::Reflection;
    result.medium_index = frame.entering_material() ? mtl.ext_medium : mtl.int_medium;
  } else {
    float eta = ior_i.eta.monochromatic() / ior_o.eta.monochromatic();
    float sin_theta_o_squared = (eta * eta) * (1.0f - cos_theta_i * cos_theta_i);
    float cos_theta_o = sqrtf(clamp(1.0f - sin_theta_o_squared, 0.0f, 1.0f));
    result.w_o = normalize(eta * data.w_i + frame.nrm * (eta * cos_theta_i - cos_theta_o));

    result.pdf = 1.0f - f;
    ETX_VALIDATE(result.pdf);

    result.weight = (1.0f - fr) / (1.0f - f) * apply_image(data.spectrum_sample, mtl.transmittance, data.tex, scene, rgb::SpectrumClass::Reflection);
    ETX_VALIDATE(result.weight);

    result.properties = BSDFSample::Delta | BSDFSample::Transmission | BSDFSample::MediumChanged;
    result.medium_index = frame.entering_material() ? mtl.int_medium : mtl.ext_medium;
    if (data.path_source == PathSource::Camera) {
      result.weight *= eta * eta;
    }
  }
  return result;
}

ETX_GPU_CODE BSDFEval evaluate(const BSDFData& data, const float3& w_o, const Material& mtl, const Scene& scene, Sampler& smp) {
  return {data.spectrum_sample, 0.0f};
}

ETX_GPU_CODE float pdf(const BSDFData& data, const float3& w_o, const Material& mtl, const Scene& scene, Sampler& smp) {
  return 0.0f;
}
}  // namespace DeltaDielectricBSDF

namespace ThinfilmBSDF {

ETX_GPU_CODE BSDFSample sample(const BSDFData& data, const Material& mtl, const Scene& scene, Sampler& smp) {
  auto frame = data.get_normal_frame();
  auto ext_ior = mtl.ext_ior(data.spectrum_sample);
  auto int_ior = mtl.int_ior(data.spectrum_sample);
  auto thinfilm = evaluate_thinfilm(data.spectrum_sample, mtl.thinfilm, data.tex, scene);
  float i_dot_n = dot(data.w_i, frame.nrm);
  SpectralResponse fr = fresnel::calculate(data.spectrum_sample, i_dot_n, ext_ior, int_ior, thinfilm);
  float f = fr.monochromatic();

  BSDFSample result = {};
  if (smp.next() <= f) {
    result.w_o = reflect(data.w_i, frame.nrm);
    if (i_dot_n * dot(result.w_o, frame.nrm) >= 0.0f) {
      return {{data.spectrum_sample, 0.0f}};
    }
    result.w_o = normalize(result.w_o);
    result.pdf = f;
    result.weight = apply_image(data.spectrum_sample, mtl.specular, data.tex, scene, rgb::SpectrumClass::Reflection);
    result.weight *= (fr / f);
    result.properties = BSDFSample::Delta | BSDFSample::Reflection;
    result.medium_index = frame.entering_material() ? mtl.ext_medium : mtl.int_medium;
  } else {
    result.w_o = data.w_i;
    result.pdf = 1.0f - f;
    result.weight = apply_image(data.spectrum_sample, mtl.transmittance, data.tex, scene, rgb::SpectrumClass::Reflection);
    result.weight *= (1.0f - fr) / (1.0f - f);
    result.properties = BSDFSample::Delta | BSDFSample::Transmission | BSDFSample::MediumChanged;
    result.medium_index = frame.entering_material() ? mtl.int_medium : mtl.ext_medium;
  }

  return result;
}

ETX_GPU_CODE BSDFEval evaluate(const BSDFData& data, const float3& w_o, const Material& mtl, const Scene& scene, Sampler& smp) {
  return {data.spectrum_sample, 0.0f};
}

ETX_GPU_CODE float pdf(const BSDFData& data, const float3& w_o, const Material& mtl, const Scene& scene, Sampler& smp) {
  return 0.0f;
}

ETX_GPU_CODE bool is_delta(const Material& material, const float2& tex, const Scene& scene, Sampler& smp) {
  return true;
}

}  // namespace ThinfilmBSDF

namespace DielectricBSDF {

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
    if (external::sample_dielectric(data.spectrum_sample, smp, w_i, alpha_x, alpha_y, ext_ior, int_ior, thinfilm, result.w_o) == false) {
      return {{data.spectrum_sample, 0.0f}};
    }

    float3 w_o = normalize(local_frame.from_local(result.w_o));
    result.pdf = pdf(data, w_o, mtl, scene, smp);
    ETX_VALIDATE(result.pdf);

    if (LocalFrame::cos_theta(result.w_o) > 0) {
      // reflection
      result.eta = 1.0f;
      result.weight = apply_image(data.spectrum_sample, mtl.specular, data.tex, scene, rgb::SpectrumClass::Reflection);
      result.properties = BSDFSample::Reflection;
      result.medium_index = mtl.ext_medium;
    } else {
      // refraction
      result.eta = m_eta;
      float factor = (data.path_source == PathSource::Camera) ? sqr(m_invEta) : 1.0f;
      result.weight = apply_image(data.spectrum_sample, mtl.transmittance, data.tex, scene, rgb::SpectrumClass::Reflection) * factor;
      result.properties = BSDFSample::Transmission | BSDFSample::MediumChanged;
      result.medium_index = mtl.int_medium;
    }
  } else {  // inside
    if (external::sample_dielectric(data.spectrum_sample, smp, -w_i, alpha_x, alpha_y, int_ior, ext_ior, thinfilm, result.w_o) == false) {
      return {{data.spectrum_sample, 0.0f}};
    }

    result.w_o = -result.w_o;

    float3 w_o = normalize(local_frame.from_local(result.w_o));
    result.pdf = pdf(data, w_o, mtl, scene, smp);
    ETX_VALIDATE(result.pdf);

    if (LocalFrame::cos_theta(result.w_o) > 0) {
      // refraction
      result.eta = m_invEta;
      float factor = (data.path_source == PathSource::Camera) ? sqr(m_eta) : 1.0f;
      result.weight = apply_image(data.spectrum_sample, mtl.transmittance, data.tex, scene, rgb::SpectrumClass::Reflection) * factor;
      result.properties = BSDFSample::Transmission | BSDFSample::MediumChanged;
      result.medium_index = mtl.ext_medium;
    } else {
      // reflection
      result.eta = 1.0f;
      result.weight = apply_image(data.spectrum_sample, mtl.specular, data.tex, scene, rgb::SpectrumClass::Reflection);
      result.properties = BSDFSample::Reflection;
      result.medium_index = mtl.int_medium;
    }
  }

  result.w_o = normalize(local_frame.from_local(result.w_o));
  return result;
}

ETX_GPU_CODE BSDFEval evaluate(const BSDFData& data, const float3& in_w_o, const Material& mtl, const Scene& scene, Sampler& smp) {
  if (is_delta(mtl, data.tex, scene, smp)) {
    return DeltaDielectricBSDF::evaluate(data, in_w_o, mtl, scene, smp);
  }

  LocalFrame local_frame = {data.tan, data.btn, data.nrm};
  auto w_i = local_frame.to_local(-data.w_i);
  if (LocalFrame::cos_theta(w_i) == 0)
    return {data.spectrum_sample, 0.0f};

  auto w_o = local_frame.to_local(in_w_o);
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
    return {data.spectrum_sample, 0.0f};

  BSDFEval eval;
  eval.func = apply_image(data.spectrum_sample, reflection ? mtl.specular : mtl.transmittance, data.tex, scene, rgb::SpectrumClass::Reflection) * (2.0f * value);
  ETX_VALIDATE(eval.func);
  eval.bsdf = eval.func * fabsf(LocalFrame::cos_theta(w_o));
  eval.pdf = pdf(data, w_o, mtl, scene, smp);
  eval.weight = eval.bsdf / eval.pdf;
  ETX_VALIDATE(eval.weight);
  return eval;
}

ETX_GPU_CODE float pdf(const BSDFData& data, const float3& in_w_o, const Material& mtl, const Scene& scene, Sampler& smp) {
  if (is_delta(mtl, data.tex, scene, smp)) {
    return DeltaDielectricBSDF::pdf(data, in_w_o, mtl, scene, smp);
  }

  LocalFrame local_frame = {data.tan, data.btn, data.nrm};
  auto w_i = local_frame.to_local(-data.w_i);
  if (LocalFrame::cos_theta(w_i) == 0.0f)
    return 0.0f;

  auto w_o = local_frame.to_local(in_w_o);
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

  float F = fresnel::calculate(data.spectrum_sample, dot(w_i, wh), outside ? ext_ior : int_ior, outside ? int_ior : ext_ior, thinfilm).monochromatic();
  ETX_VALIDATE(F);

  prob *= reflect ? F : (1 - F);

  float result = fabsf(prob * dwh_dwo) + fabsf(LocalFrame::cos_theta(w_o));
  ETX_VALIDATE(result);

  return result;
}

ETX_GPU_CODE bool is_delta(const Material& material, const float2& tex, const Scene& scene, Sampler& smp) {
  return max(material.roughness.x, material.roughness.y) <= kDeltaAlphaTreshold;
}

}  // namespace DielectricBSDF
}  // namespace etx
