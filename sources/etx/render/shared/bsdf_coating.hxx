namespace etx {
namespace CoatingBSDF {

ETX_GPU_CODE float2 remap_alpha(float2 a) {
  return sqr(max(a, float2{1.0f / 16.0f, 1.0f / 16.0f}));
}

ETX_GPU_CODE BSDFSample sample(const BSDFData& data, const Material& mtl, const Scene& scene, Sampler& smp) {
  auto frame = data.get_normal_frame();

  uint32_t properties = 0;
  float3 w_o = {};
  if (smp.next() <= 0.5f) {
    auto ggx = NormalDistribution(frame, remap_alpha(mtl.roughness));
    auto m = ggx.sample(smp, data.w_i);
    w_o = normalize(reflect(data.w_i, m));
    properties = BSDFSample::Reflection;
  } else {
    w_o = sample_cosine_distribution(smp.next_2d(), frame.nrm, 1.0f);
    properties = BSDFSample::Diffuse | BSDFSample::Reflection;
  }

  return {w_o, evaluate(data, w_o, mtl, scene, smp), properties};
}

ETX_GPU_CODE BSDFEval evaluate(const BSDFData& data, const float3& w_o, const Material& mtl, const Scene& scene, Sampler& smp) {
  auto frame = data.get_normal_frame();

  auto pow5 = [](float value) {
    return sqr(value) * sqr(value) * fabsf(value);
  };

  float n_dot_o = dot(frame.nrm, w_o);
  float n_dot_i = -dot(frame.nrm, data.w_i);

  float3 m = normalize(w_o - data.w_i);
  float m_dot_o = dot(m, w_o);

  if ((n_dot_o <= kEpsilon) || (n_dot_i <= kEpsilon) || (m_dot_o <= kEpsilon)) {
    return {data.spectrum_sample.wavelength, 0.0f};
  }

  auto eta_e = mtl.ext_ior(data.spectrum_sample);
  auto eta_i = mtl.int_ior(data.spectrum_sample);
  auto thinfilm = evaluate_thinfilm(data.spectrum_sample, mtl.thinfilm, data.tex, scene);
  auto f = fresnel::dielectric(data.spectrum_sample, data.w_i, m, eta_e, eta_i, thinfilm);

  auto ggx = NormalDistribution(frame, remap_alpha(mtl.roughness));
  auto eval = ggx.evaluate(m, data.w_i, w_o);

  auto specular_value = apply_image(data.spectrum_sample, mtl.specular, data.tex, scene);
  auto diffuse_value = apply_image(data.spectrum_sample, mtl.diffuse, data.tex, scene);
  auto fresnel = specular_value + f * (1.0f - specular_value);

  auto diffuse_factor = (28.f / (23.f * kPi)) * (1.0 - specular_value) * (1.0f - pow5(1.0f - 0.5f * n_dot_i)) * (1.0f - pow5(1.0f - 0.5f * n_dot_o));
  auto specular = fresnel * eval.ndf / (4.0f * m_dot_o * m_dot_o);

  BSDFEval result;
  result.func = diffuse_value * diffuse_factor + specular;
  ETX_VALIDATE(result.func);
  result.bsdf = diffuse_value * diffuse_factor * n_dot_o + specular * n_dot_o;
  ETX_VALIDATE(result.bsdf);
  result.pdf = 0.5f * (kInvPi * n_dot_o + eval.pdf / (4.0f * m_dot_o));
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

  auto ggx = NormalDistribution(frame, remap_alpha(mtl.roughness));
  float result = 0.5f * (kInvPi * n_dot_o + ggx.pdf(m, data.w_i, w_o) / (4.0f * m_dot_o));
  ETX_VALIDATE(result);
  return result;
}

ETX_GPU_CODE bool is_delta(const Material& material, const float2& tex, const Scene& scene, Sampler& smp) {
  return false;  // TODO : check this
}

}  // namespace CoatingBSDF
}  // namespace etx
