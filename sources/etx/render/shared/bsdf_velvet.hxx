namespace etx {

namespace VelvetBSDF {

ETX_GPU_CODE BSDFSample sample(const BSDFData& data, const Material& mtl, const Scene& scene, Sampler& smp) {
  auto frame = data.get_normal_frame();
  float3 w_o = sample_cosine_distribution(smp.next_2d(), frame.nrm, 0.0f);
  return {w_o, evaluate(data, w_o, mtl, scene, smp), BSDFSample::Reflection, mtl.ext_medium};
}

ETX_GPU_CODE float lambda_velvet_l(float r, float x) {
  x = fmaxf(x, 0.0f);
  auto lerp_x = [](float a, float b, float t) {
    return sqr(1.0f - t) * a + (1.0f - sqr(1.0f - t)) * b;
  };
  float a = lerp_x(25.3245f, 21.5473f, r);
  float b = lerp_x(3.32435f, 3.82987f, r);
  float c = lerp_x(0.16801f, 0.19823f, r);
  float d = lerp_x(-1.27393f, -1.97760f, r);
  float e = lerp_x(-4.85967f, -4.32054f, r);
  float q = a / (1.0f + b * powf(x, c)) + d * x + e;
  ETX_VALIDATE(q);
  return q;
}

ETX_GPU_CODE float lambda_velvet(float r, float cos_t) {
  if (cos_t < 0.5f)
    return expf(lambda_velvet_l(r, cos_t));

  return expf(2.0f * lambda_velvet_l(r, 0.5f) - lambda_velvet_l(r, 1.0f - cos_t));
}

ETX_GPU_CODE float fresnel_approximate(float f0, float f90, float cos_t) {
  return f0 + (f90 - f0) * powf(fmaxf(1.0f - cos_t, 0.0f), 5.0f);
}

ETX_GPU_CODE float diffuse_burley(float alpha, float n_dot_i, float n_dot_o, float m_dot_o) {
  float f90 = 0.5f + 2.0f * alpha * m_dot_o * m_dot_o;
  float lightScatter = fresnel_approximate(1.0f, f90, n_dot_o);
  float viewScatter = fresnel_approximate(1.0f, f90, n_dot_i);
  return lightScatter * viewScatter * kInvPi;
}

ETX_GPU_CODE BSDFEval evaluate(const BSDFData& data, const float3& w_o, const Material& mtl, const Scene& scene, Sampler& smp) {
  auto frame = data.get_normal_frame();

  float n_dot_o = fmaxf(0.0f, dot(w_o, frame.nrm));
  float n_dot_i = fmaxf(0.0f, -dot(data.w_i, frame.nrm));
  if ((n_dot_o <= kEpsilon) || (n_dot_i <= kEpsilon))
    return {data.spectrum_sample, 0.0f};

  float3 m = normalize(w_o - data.w_i);
  float m_dot_o = fmaxf(0.0f, dot(w_o, m));
  float m_dot_i = fmaxf(0.0f, -dot(data.w_i, m));
  if ((m_dot_o <= kEpsilon) || (m_dot_i <= kEpsilon))
    return {data.spectrum_sample, 0.0f};

  float specular_scale_base = 0.0f;
  float alpha = 0.5f * (mtl.roughness.x + mtl.roughness.y);
  if (alpha > kEpsilon) {
    float inv_alpha = 1.0f / (kEpsilon + alpha);
    float m_dot_n = dot(m, frame.nrm);
    float sin_t = (1.0f - m_dot_n * m_dot_n);
    float d = (2.0f + inv_alpha) * powf(sin_t, 0.5f * inv_alpha) / kDoublePi;
    ETX_VALIDATE(d);
    float l_i = lambda_velvet(alpha, n_dot_i);
    ETX_VALIDATE(l_i);
    float l_o = lambda_velvet(alpha, n_dot_o);
    ETX_VALIDATE(l_o);
    float g = 1.0f / (1.0f + l_i + l_o);
    ETX_VALIDATE(g);
    specular_scale_base = 0.25f * d * g / n_dot_i;
    ETX_VALIDATE(specular_scale_base);
  }

  auto diffuse = apply_image(data.spectrum_sample, mtl.transmittance, data.tex, scene, nullptr);
  auto specular = apply_image(data.spectrum_sample, mtl.reflectance, data.tex, scene, nullptr);

  float diffuse_scale = diffuse_burley(alpha, n_dot_i, n_dot_o, m_dot_o);

  BSDFEval eval;
  eval.func = diffuse * diffuse_scale + specular * specular_scale_base / n_dot_o;
  ETX_VALIDATE(eval.func);

  eval.bsdf = diffuse * diffuse_scale * n_dot_o + specular * specular_scale_base;
  ETX_VALIDATE(eval.bsdf);

  eval.pdf = 1.0f / kDoublePi;

  eval.weight = eval.bsdf / eval.pdf;
  ETX_VALIDATE(eval.weight);
  return eval;
}

ETX_GPU_CODE float pdf(const BSDFData& data, const float3& w_o, const Material& mtl, const Scene& scene, Sampler& smp) {
  auto frame = data.get_normal_frame();
  if (frame.entering_material() == false)
    return 0.0f;

  return 1.0f / kDoublePi;
}

ETX_GPU_CODE bool is_delta(const Material& material, const float2& tex, const Scene& scene, Sampler& smp) {
  return false;
}

ETX_GPU_CODE SpectralResponse albedo(const BSDFData& data, const Material& mtl, const Scene& scene, Sampler& smp) {
  return apply_image(data.spectrum_sample, mtl.transmittance, data.tex, scene, nullptr);
}

}  // namespace VelvetBSDF
}  // namespace etx
