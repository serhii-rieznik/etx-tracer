namespace etx {

namespace PlasticBSDF {

ETX_GPU_CODE SpectralResponse specular_func(const BSDFData& data, const float3& in_w_o, const Material& mtl, const Scene& scene, Sampler& smp) {
  LocalFrame local_frame = {data.tan, data.btn, data.nrm};

  auto w_i = local_frame.to_local(-data.w_i);
  if (LocalFrame::cos_theta(w_i) <= kEpsilon)
    return {data.spectrum_sample, 0.0f};

  auto w_o = local_frame.to_local(in_w_o);
  if (LocalFrame::cos_theta(w_o) <= kEpsilon)
    return {data.spectrum_sample, 0.0f};

  auto roughness = evaluate_roughness(mtl, data.tex, scene);
  auto ext_ior = mtl.ext_ior(data.spectrum_sample);
  auto int_ior = mtl.int_ior(data.spectrum_sample);
  auto m_eta = (int_ior.eta / ext_ior.eta).monochromatic();
  auto thinfilm = evaluate_thinfilm(data.spectrum_sample, mtl.thinfilm, data.tex, scene, smp);

  SpectralResponse value = external::eval_dielectric(data.spectrum_sample, smp, w_i, w_o, true, roughness, ext_ior, int_ior, thinfilm);
  auto func = 2.0f * value * apply_image(data.spectrum_sample, mtl.reflectance, data.tex, scene, nullptr);
  ETX_VALIDATE(func);
  return func;
}

ETX_GPU_CODE float specular_pdf(const BSDFData& data, const float3& in_w_o, const Material& mtl, const Scene& scene, Sampler& smp) {
  LocalFrame local_frame = {data.tan, data.btn, data.nrm};

  auto w_i = local_frame.to_local(-data.w_i);
  if (LocalFrame::cos_theta(w_i) <= kEpsilon)
    return 0.0f;

  auto w_o = local_frame.to_local(in_w_o);
  if (LocalFrame::cos_theta(w_o) <= kEpsilon)
    return 0.0f;

  auto ext_ior = mtl.ext_ior(data.spectrum_sample);
  auto int_ior = mtl.int_ior(data.spectrum_sample);
  auto roughness = evaluate_roughness(mtl, data.tex, scene);
  auto thinfilm = evaluate_thinfilm(data.spectrum_sample, mtl.thinfilm, data.tex, scene, smp);

  float3 wh = normalize(w_o + w_i);
  float dwh_dwo = 1.0f / (4.0f * dot(w_o, wh));

  external::RayInfo ray = {w_i, roughness};

  auto d_ggx = external::D_ggx(wh, roughness);
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

  auto roughness = evaluate_roughness(mtl, data.tex, scene);
  auto ggx = NormalDistribution(frame, roughness);
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

  bool sample_diffuse = smp.next() > f.monochromatic();

  if (sample_diffuse == false) {
    in_w_o = reflect(data.w_i, m);
    sample_diffuse = dot(frame.nrm, in_w_o) <= kEpsilon;
  }

  if (sample_diffuse) {
    in_w_o = frame.from_local(sample_cosine_distribution(smp.next_2d(), 1.0f));
  }

  auto eval = evaluate(data, in_w_o, mtl, scene, smp);

  BSDFSample result = {};
  result.w_o = in_w_o;
  result.weight = eval.bsdf / eval.pdf;
  result.properties = BSDFSample::Reflection | (sample_diffuse ? BSDFSample::Diffuse : 0u);
  result.medium_index = data.current_medium;
  result.pdf = eval.pdf;
  return result;
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

  auto local_w_i = frame.to_local(-data.w_i);
  auto local_w_o = frame.to_local(w_o);

  auto diff_layer = DiffuseBSDF::diffuse_layer(data, local_w_i, local_w_o, mtl, scene, smp);
  auto spec_layer = specular_func(data, w_o, mtl, scene, smp);
  auto spec_pdf = specular_pdf(data, w_o, mtl, scene, smp);

  BSDFEval result = {};

  result.func = diff_layer.func * (1.0f - fr) + spec_layer / n_dot_o;
  ETX_VALIDATE(result.func);

  result.bsdf = diff_layer.func * (1.0f - fr) * n_dot_o + spec_layer;
  ETX_VALIDATE(result.bsdf);

  result.pdf = diff_layer.pdf * (1.0 - fr).monochromatic() + spec_pdf;
  ETX_VALIDATE(result.pdf);

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

  float diff_pdf = kInvPi * n_dot_o;
  float spec_pdf = specular_pdf(data, w_o, mtl, scene, smp);

  float result = diff_pdf * (1.0f - fr).monochromatic() + spec_pdf;
  ETX_VALIDATE(result);
  return result;
}

ETX_GPU_CODE bool is_delta(const Material& material, const float2& tex, const Scene& scene) {
  return false;
}

ETX_GPU_CODE SpectralResponse albedo(const BSDFData& data, const Material& mtl, const Scene& scene, Sampler& smp) {
  return apply_image(data.spectrum_sample, mtl.transmittance, data.tex, scene, nullptr);
}

}  // namespace PlasticBSDF

}  // namespace etx
