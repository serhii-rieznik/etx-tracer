namespace etx {

namespace DeltaPlasticBSDF {

ETX_GPU_CODE BSDFSample sample_impl(const BSDFData& data, const Material& mtl, const Scene& scene, Sampler& smp) {
  auto frame = data.get_normal_frame().frame;

  auto eta_e = mtl.ext_ior(data.spectrum_sample);
  auto eta_i = mtl.int_ior(data.spectrum_sample);

  auto thinfilm = evaluate_thinfilm(data.spectrum_sample, mtl.thinfilm, data.tex, scene);
  auto fr = fresnel::dielectric(data.spectrum_sample, data.w_i, frame.nrm, eta_e, eta_i, thinfilm);
  auto f = fr.monochromatic();

  bool reflection = smp.next() <= f;

  BSDFSample result;
  result.properties = BSDFSample::Diffuse;
  if (reflection) {
    result.w_o = normalize(reflect(data.w_i, frame.nrm));
    result.properties = result.properties | BSDFSample::DeltaReflection;
  } else {
    result.w_o = sample_cosine_distribution(smp.next(), smp.next(), frame.nrm, 1.0f);
  }

  float n_dot_o = dot(frame.nrm, result.w_o);
  auto diffuse = apply_image(data.spectrum_sample, mtl.diffuse, data.tex, scene);
  auto specular = apply_image(data.spectrum_sample, mtl.specular, data.tex, scene);

  if (reflection) {
    auto bsdf = diffuse * (kInvPi * n_dot_o * (1.0f - fr)) + specular * fr;
    result.pdf = kInvPi * n_dot_o * (1.0f - f) + f;
    result.weight = bsdf / result.pdf;
  } else {
    result.pdf = kInvPi * n_dot_o;
    result.weight = diffuse;
  }

  return result;
}

ETX_GPU_CODE BSDFEval evaluate_impl(const BSDFData& data, const Material& mtl, const Scene& scene, Sampler& smp) {
  auto frame = data.get_normal_frame().frame;

  float n_dot_o = dot(frame.nrm, data.w_o);
  if (n_dot_o <= kEpsilon) {
    return {data.spectrum_sample.wavelength, 0.0f};
  }

  float3 m = normalize(data.w_o - data.w_i);
  auto eta_e = mtl.ext_ior(data.spectrum_sample);
  auto eta_i = mtl.int_ior(data.spectrum_sample);
  auto thinfilm = evaluate_thinfilm(data.spectrum_sample, mtl.thinfilm, data.tex, scene);
  auto fr = fresnel::dielectric(data.spectrum_sample, data.w_i, m, eta_e, eta_i, thinfilm);

  auto diffuse = apply_image(data.spectrum_sample, mtl.diffuse, data.tex, scene);

  BSDFEval result;
  result.func = diffuse * (kInvPi * (1.0f - fr));
  ETX_VALIDATE(result.func);
  result.bsdf = diffuse * (kInvPi * n_dot_o * (1.0f - fr));
  ETX_VALIDATE(result.bsdf);
  result.weight = diffuse;
  ETX_VALIDATE(result.weight);
  result.pdf = kInvPi * n_dot_o * (1.0f - fr.monochromatic());
  ETX_VALIDATE(result.pdf);
  return result;
}

ETX_GPU_CODE float pdf_impl(const BSDFData& data, const Material& mtl, const Scene& scene, Sampler& smp) {
  auto frame = data.get_normal_frame().frame;

  float n_dot_o = dot(frame.nrm, data.w_o);
  if (n_dot_o <= kEpsilon) {
    return 0.0f;
  }

  float3 m = normalize(data.w_o - data.w_i);
  auto eta_e = mtl.ext_ior(data.spectrum_sample);
  auto eta_i = mtl.int_ior(data.spectrum_sample);
  auto thinfilm = evaluate_thinfilm(data.spectrum_sample, mtl.thinfilm, data.tex, scene);
  auto fr = fresnel::dielectric(data.spectrum_sample, data.w_i, m, eta_e, eta_i, thinfilm);
  return kInvPi * n_dot_o * (1.0f - fr.monochromatic());
}
}  // namespace DeltaPlasticBSDF

namespace PlasticBSDF {

ETX_FORWARD_TO_IMPL;

ETX_GPU_CODE BSDFSample sample_impl(const BSDFData& data, const Material& mtl, const Scene& scene, Sampler& smp) {
  if (dot(mtl.roughness, float2{0.5f, 0.5f}) <= kDeltaAlphaTreshold) {
    return DeltaPlasticBSDF::sample_impl(data, mtl, scene, smp);
  }

  auto frame = data.get_normal_frame().frame;

  auto ggx = NormalDistribution(frame, mtl.roughness);
  auto m = ggx.sample(smp, data.w_i);

  auto eta_e = mtl.ext_ior(data.spectrum_sample);
  auto eta_i = mtl.int_ior(data.spectrum_sample);
  auto thinfilm = evaluate_thinfilm(data.spectrum_sample, mtl.thinfilm, data.tex, scene);
  auto f = fresnel::dielectric(data.spectrum_sample, data.w_i, m, eta_e, eta_i, thinfilm);

  uint32_t properties = 0;
  BSDFData eval_data = data;
  if (smp.next() <= f.monochromatic()) {
    eval_data.w_o = normalize(reflect(data.w_i, m));
  } else {
    eval_data.w_o = sample_cosine_distribution(smp.next(), smp.next(), frame.nrm, 1.0f);
    properties = BSDFSample::Diffuse;
  }

  return {eval_data.w_o, evaluate(eval_data, mtl, scene, smp), properties};
}

ETX_GPU_CODE BSDFEval evaluate_impl(const BSDFData& data, const Material& mtl, const Scene& scene, Sampler& smp) {
  if (dot(mtl.roughness, float2{0.5f, 0.5f}) <= kDeltaAlphaTreshold) {
    return DeltaPlasticBSDF::evaluate_impl(data, mtl, scene, smp);
  }

  auto frame = data.get_normal_frame().frame;

  float n_dot_o = dot(frame.nrm, data.w_o);
  float n_dot_i = -dot(frame.nrm, data.w_i);

  float3 m = normalize(data.w_o - data.w_i);
  float m_dot_o = dot(m, data.w_o);

  if ((n_dot_o <= kEpsilon) || (n_dot_i <= kEpsilon) || (m_dot_o <= kEpsilon)) {
    return {data.spectrum_sample.wavelength, 0.0f};
  }

  auto eta_e = mtl.ext_ior(data.spectrum_sample);
  auto eta_i = mtl.int_ior(data.spectrum_sample);
  auto thinfilm = evaluate_thinfilm(data.spectrum_sample, mtl.thinfilm, data.tex, scene);
  auto fr = fresnel::dielectric(data.spectrum_sample, data.w_i, m, eta_e, eta_i, thinfilm);
  auto f = fr.monochromatic();

  auto ggx = NormalDistribution(frame, mtl.roughness);
  auto eval = ggx.evaluate(m, data.w_i, data.w_o);
  float j = 1.0f / (4.0f * m_dot_o);

  auto diffuse = apply_image(data.spectrum_sample, mtl.diffuse, data.tex, scene);
  auto specular = apply_image(data.spectrum_sample, mtl.specular, data.tex, scene);

  BSDFEval result;
  result.func = diffuse * (kInvPi * (1.0f - fr)) + specular * (fr * eval.ndf * eval.visibility / (4.0f * n_dot_i * n_dot_o));
  ETX_VALIDATE(result.func);
  result.bsdf = diffuse * (kInvPi * n_dot_o * (1.0f - fr)) + specular * (fr * eval.ndf * eval.visibility / (4.0f * n_dot_i));
  ETX_VALIDATE(result.bsdf);
  result.pdf = kInvPi * n_dot_o * (1.0f - f) + eval.pdf * j * f;
  ETX_VALIDATE(result.pdf);
  result.weight = result.bsdf / result.pdf;
  ETX_VALIDATE(result.weight);
  return result;
}

ETX_GPU_CODE float pdf_impl(const BSDFData& data, const Material& mtl, const Scene& scene, Sampler& smp) {
  if (dot(mtl.roughness, float2{0.5f, 0.5f}) <= kDeltaAlphaTreshold) {
    return DeltaPlasticBSDF::pdf_impl(data, mtl, scene, smp);
  }

  auto frame = data.get_normal_frame().frame;

  float3 m = normalize(data.w_o - data.w_i);
  float m_dot_o = dot(m, data.w_o);
  float n_dot_o = dot(frame.nrm, data.w_o);

  if ((n_dot_o <= kEpsilon) || (m_dot_o <= kEpsilon)) {
    return 0.0f;
  }

  auto eta_e = mtl.ext_ior(data.spectrum_sample);
  auto eta_i = mtl.int_ior(data.spectrum_sample);
  auto thinfilm = evaluate_thinfilm(data.spectrum_sample, mtl.thinfilm, data.tex, scene);
  auto fr = fresnel::dielectric(data.spectrum_sample, data.w_i, m, eta_e, eta_i, thinfilm);
  auto f = fr.monochromatic();

  auto ggx = NormalDistribution(frame, mtl.roughness);

  float j = 1.0f / (4.0f * m_dot_o);
  float result = kInvPi * n_dot_o * (1.0f - f) + ggx.pdf(m, data.w_i, data.w_o) * j * f;
  ETX_VALIDATE(result);
  return result;
}

ETX_GPU_CODE bool continue_tracing_impl(const Material& material, const float2& tex, const Scene& scene, Sampler& smp) {
  return false;
}

ETX_GPU_CODE bool is_delta_impl(const Material& material, const float2& tex, const Scene& scene, Sampler& smp) {
  return false;
}

}  // namespace PlasticBSDF

}  // namespace etx
