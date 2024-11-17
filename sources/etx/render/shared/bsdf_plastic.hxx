namespace etx {

namespace PlasticBSDF {

ETX_GPU_CODE SpectralResponse spec_eval(const BSDFData& data, const float3& in_w_o, const Material& mtl, const Scene& scene, Sampler& smp) {
  LocalFrame local_frame = {data.tan, data.btn, data.nrm};

  auto w_i = local_frame.to_local(-data.w_i);
  if (LocalFrame::cos_theta(w_i) <= kEpsilon)
    return {data.spectrum_sample, 0.0f};

  auto w_o = local_frame.to_local(in_w_o);
  if (LocalFrame::cos_theta(w_o) <= kEpsilon)
    return {data.spectrum_sample, 0.0f};

  auto ext_ior = mtl.ext_ior(data.spectrum_sample);
  auto int_ior = mtl.int_ior(data.spectrum_sample);
  auto m_eta = (int_ior.eta / ext_ior.eta).monochromatic();
  auto thinfilm = evaluate_thinfilm(data.spectrum_sample, mtl.thinfilm, data.tex, scene, smp);

  bool forward_path = smp.next() > 0.5f;

  SpectralResponse value = external::eval_dielectric(data.spectrum_sample, smp,  //
    forward_path ? w_i : w_o, forward_path ? w_o : w_i, true, mtl.roughness, ext_ior, int_ior, thinfilm);

  if (value.is_zero())
    return {data.spectrum_sample, 0.0f};

  auto func = (2.0f * value) * apply_image(data.spectrum_sample, mtl.transmittance, data.tex, scene, nullptr);
  ETX_VALIDATE(func);

  return func;
}

ETX_GPU_CODE float spec_pdf(const BSDFData& data, const float3& in_w_o, const Material& mtl, const Scene& scene, Sampler& smp) {
  LocalFrame local_frame = {data.tan, data.btn, data.nrm};

  auto w_i = local_frame.to_local(-data.w_i);
  if (LocalFrame::cos_theta(w_i) <= kEpsilon)
    return 0.0f;

  auto w_o = local_frame.to_local(in_w_o);
  if (LocalFrame::cos_theta(w_o) <= kEpsilon)
    return 0.0f;

  auto ext_ior = mtl.ext_ior(data.spectrum_sample);
  auto int_ior = mtl.int_ior(data.spectrum_sample);
  auto thinfilm = evaluate_thinfilm(data.spectrum_sample, mtl.thinfilm, data.tex, scene, smp);

  float3 wh = normalize(w_o + w_i);
  float dwh_dwo = 1.0f / (4.0f * dot(w_o, wh));

  external::RayInfo ray = {w_i, mtl.roughness};

  auto d_ggx = external::D_ggx(wh, mtl.roughness);
  ETX_VALIDATE(d_ggx);

  float prob = max(0.0f, dot(wh, ray.w) * d_ggx / ((1.0f + ray.Lambda) * LocalFrame::cos_theta(ray.w)));
  ETX_VALIDATE(prob);

  auto fr = fresnel::calculate(data.spectrum_sample, dot(w_i, wh), ext_ior, int_ior, thinfilm);
  float f = fr.monochromatic();
  ETX_VALIDATE(f);

  prob *= f;

  float result = fabsf(prob * dwh_dwo);
  ETX_VALIDATE(result);

  return result;
}

ETX_GPU_CODE BSDFSample sample(const BSDFData& data, const Material& mtl, const Scene& scene, Sampler& smp) {
  auto frame = data.get_normal_frame();

  auto ggx = NormalDistribution(frame, mtl.roughness);
  auto m = ggx.sample(smp, data.w_i);

  auto ext_ior = mtl.ext_ior(data.spectrum_sample);
  auto int_ior = mtl.int_ior(data.spectrum_sample);
  auto thinfilm = evaluate_thinfilm(data.spectrum_sample, mtl.thinfilm, data.tex, scene, smp);

  auto f = fresnel::calculate(data.spectrum_sample, dot(data.w_i, m), ext_ior, int_ior, thinfilm);
  auto fr = f.monochromatic();

  auto w_i = frame.to_local(-data.w_i);
  if (w_i.z <= kEpsilon)
    return {data.spectrum_sample};

  float3 in_w_o = {};

  bool reflection = smp.next() <= f.monochromatic();
  bool sample_diffuse = false;

  if (reflection) {
    in_w_o = reflect(data.w_i, m);
  } else {
    sample_diffuse = true;
  }

  if (reflection && (dot(frame.nrm, in_w_o) <= kEpsilon)) {
    sample_diffuse = true;
  }

  if (sample_diffuse) {
    in_w_o = frame.from_local(external::sample_diffuse(smp, w_i, mtl.roughness));
    m = normalize(in_w_o - data.w_i);
  }

  {
    float n_dot_o = dot(frame.nrm, in_w_o);
    float m_dot_o = dot(m, in_w_o);

    if ((n_dot_o <= kEpsilon) || (m_dot_o <= kEpsilon)) {
      return {data.spectrum_sample};
    }

    auto fr = fresnel::calculate(data.spectrum_sample, dot(data.w_i, m), ext_ior, int_ior, thinfilm);
    auto f = fr.monochromatic();

    auto eval = ggx.evaluate(m, data.w_i, in_w_o);
    float j = 1.0f / (4.0f * m_dot_o);

    BSDFSample result = {};
    result.eta = 1.0f;
    result.properties = BSDFSample::Reflection | BSDFSample::Diffuse;
    result.medium_index = mtl.ext_medium;

    auto diffuse_layer = DiffuseBSDF::diffuse_layer(data, w_i, frame.to_local(in_w_o), mtl, scene, smp);

    auto specular_bsdf = spec_eval(data, in_w_o, mtl, scene, smp);
    auto bsdf = (1.0f - fr) * diffuse_layer.bsdf + fr * specular_bsdf;
    ETX_VALIDATE(bsdf);

    float specular_pdf = spec_pdf(data, in_w_o, mtl, scene, smp) * fr.monochromatic();
    result.pdf = diffuse_layer.pdf * (1.0f - fr).monochromatic() + specular_pdf;
    ETX_VALIDATE(result.pdf);

    if (result.pdf <= 0.0f) {
      return {data.spectrum_sample};
    }

    result.weight = bsdf / result.pdf;
    ETX_VALIDATE(result.weight);

    result.w_o = in_w_o;
    return result;
  }
}

ETX_GPU_CODE BSDFEval evaluate(const BSDFData& data, const float3& w_o, const Material& mtl, const Scene& scene, Sampler& smp) {
  auto frame = data.get_normal_frame();
  float3 m = normalize(w_o - data.w_i);

  float n_dot_o = dot(frame.nrm, w_o);
  float m_dot_o = dot(m, w_o);

  if ((n_dot_o <= kEpsilon) || (m_dot_o <= kEpsilon)) {
    return {data.spectrum_sample, 0.0f};
  }

  auto eta_e = mtl.ext_ior(data.spectrum_sample);
  auto eta_i = mtl.int_ior(data.spectrum_sample);
  auto thinfilm = evaluate_thinfilm(data.spectrum_sample, mtl.thinfilm, data.tex, scene, smp);
  auto fr = fresnel::calculate(data.spectrum_sample, dot(data.w_i, m), eta_e, eta_i, thinfilm);
  auto f = fr.monochromatic();

  auto local_w_i = frame.to_local(-data.w_i);
  auto local_w_o = frame.to_local(w_o);

  auto diffuse_layer = DiffuseBSDF::diffuse_layer(data, local_w_i, local_w_o, mtl, scene, smp);
  auto specular_func = spec_eval(data, w_o, mtl, scene, smp);

  BSDFEval result = {};

  result.func = diffuse_layer.func * (1.0f - fr) + specular_func * fr;
  ETX_VALIDATE(result.func);

  result.bsdf = result.func * n_dot_o;
  ETX_VALIDATE(result.bsdf);

  float specular_pdf = spec_pdf(data, w_o, mtl, scene, smp) * fr.monochromatic();
  result.pdf = diffuse_layer.pdf + specular_pdf;
  ETX_VALIDATE(result.pdf);

  result.weight = result.bsdf / result.pdf;
  ETX_VALIDATE(result.weight);

  return result;
}

ETX_GPU_CODE float pdf(const BSDFData& data, const float3& w_o, const Material& mtl, const Scene& scene, Sampler& smp) {
  auto frame = data.get_normal_frame();

  float3 m = normalize(w_o - data.w_i);
  float m_dot_o = dot(m, w_o);
  float n_dot_o = dot(frame.nrm, w_o);

  if ((n_dot_o <= kEpsilon) || (m_dot_o <= kEpsilon)) {
    return 0.0f;
  }

  auto eta_e = mtl.ext_ior(data.spectrum_sample);
  auto eta_i = mtl.int_ior(data.spectrum_sample);
  auto thinfilm = evaluate_thinfilm(data.spectrum_sample, mtl.thinfilm, data.tex, scene, smp);
  auto fr = fresnel::calculate(data.spectrum_sample, dot(data.w_i, m), eta_e, eta_i, thinfilm);

  float diffuse_pdf = kInvPi * n_dot_o * (1.0f - fr).monochromatic();
  float specular_pdf = spec_pdf(data, w_o, mtl, scene, smp) * fr.monochromatic();

  float result = diffuse_pdf + specular_pdf;
  ETX_VALIDATE(result);
  return result;
}

ETX_GPU_CODE bool is_delta(const Material& material, const float2& tex, const Scene& scene, Sampler& smp) {
  return false;
}

ETX_GPU_CODE SpectralResponse albedo(const BSDFData& data, const Material& mtl, const Scene& scene, Sampler& smp) {
  return apply_image(data.spectrum_sample, mtl.transmittance, data.tex, scene, nullptr);
}

}  // namespace PlasticBSDF

}  // namespace etx
