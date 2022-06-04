namespace etx {
namespace GenericBSDF {

ETX_FORWARD_TO_IMPL;

ETX_GPU_CODE float2 remap_alpha(float2 a) {
  return sqr(max(a, float2{1.0f / 16.0f, 1.0f / 16.0f}));
}

ETX_GPU_CODE float remap_metalness(float m) {
  return sqrtf(m);
}

ETX_GPU_CODE BSDFSample sample_impl(const BSDFData& data, const Material& mtl, const Scene& scene, Sampler& smp) {
  auto frame = data.get_normal_frame().frame;

  float2 alpha = mtl.roughness;
  float metalness = mtl.metalness;
  if (mtl.metal_roughness_image_index != kInvalidIndex) {
    auto value = scene.images[mtl.metal_roughness_image_index].evaluate(data.tex);
    alpha *= value.y;
    metalness *= value.z;
  }
  alpha = remap_alpha(alpha);
  metalness = remap_metalness(metalness);

  auto ggx = NormalDistribution(frame, alpha);
  auto m = ggx.sample(smp, data.w_i);

  auto diffuse = apply_image(data.spectrum_sample, mtl.diffuse, data.tex, scene);

  float t = fabsf(dot(data.w_i, m));
  SpectralResponse f0 = SpectralResponse{data.spectrum_sample.wavelength, 0.04f} * (1.0f - metalness) + diffuse * metalness;
  SpectralResponse f = f0 + (1.0f - f0) * powf(1.0f - t, 5.0f);

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
  auto frame = data.get_normal_frame().frame;

  float n_dot_o = dot(frame.nrm, data.w_o);
  float n_dot_i = -dot(frame.nrm, data.w_i);
  if ((n_dot_o <= kEpsilon) || (n_dot_i <= kEpsilon)) {
    return {data.spectrum_sample.wavelength, 0.0f};
  }

  float2 alpha = mtl.roughness;
  float metalness = mtl.metalness;
  if (mtl.metal_roughness_image_index != kInvalidIndex) {
    auto value = scene.images[mtl.metal_roughness_image_index].evaluate(data.tex);
    alpha *= value.y;
    metalness *= value.z;
  }
  alpha = remap_alpha(alpha);
  metalness = remap_metalness(metalness);

  float3 m = normalize(data.w_o - data.w_i);
  float m_dot_o = dot(m, data.w_o);

  auto diffuse = apply_image(data.spectrum_sample, mtl.diffuse, data.tex, scene);
  auto specular = apply_image(data.spectrum_sample, mtl.specular, data.tex, scene);

  float t = fabsf(dot(data.w_i, m));
  SpectralResponse f0 = SpectralResponse{data.spectrum_sample.wavelength, 0.04f * (1.0f - metalness)} + diffuse * metalness;
  SpectralResponse f = f0 + (1.0f - f0) * powf(1.0f - t, 5.0f);

  auto ggx = NormalDistribution(frame, alpha);
  auto eval = ggx.evaluate(m, data.w_i, data.w_o);
  float j = 1.0f / (4.0f * m_dot_o);

  BSDFEval result;
  result.func = diffuse * (kInvPi * (1.0f - metalness)) + specular * (f * eval.ndf * eval.visibility / (4.0f * n_dot_i * n_dot_o));
  result.bsdf = diffuse * (kInvPi * (1.0f - metalness) * n_dot_o) + specular * (f * eval.ndf * eval.visibility / (4.0f * n_dot_i));
  result.pdf = kInvPi * n_dot_o * (1.0f - f.monochromatic()) + eval.pdf * j * f.monochromatic();
  result.weight = result.bsdf / result.pdf;
  return result;
}

ETX_GPU_CODE float pdf_impl(const BSDFData& data, const Material& mtl, const Scene& scene, Sampler& smp) {
  auto frame = data.get_normal_frame().frame;

  float n_dot_o = dot(frame.nrm, data.w_o);
  if (n_dot_o <= kEpsilon) {
    return 0.0f;
  }

  float3 m = normalize(data.w_o - data.w_i);
  float m_dot_o = dot(m, data.w_o);

  auto diffuse = apply_image(data.spectrum_sample, mtl.diffuse, data.tex, scene);

  float2 alpha = mtl.roughness;
  float metalness = mtl.metalness;
  if (mtl.metal_roughness_image_index != kInvalidIndex) {
    auto value = scene.images[mtl.metal_roughness_image_index].evaluate(data.tex);
    alpha *= value.y;
    metalness *= value.z;
  }
  alpha = remap_alpha(alpha);
  metalness = remap_metalness(metalness);

  float t = fabsf(dot(data.w_i, m));
  SpectralResponse f0 = SpectralResponse{data.spectrum_sample.wavelength, 0.04f} * (1.0f - metalness) + diffuse * metalness;
  SpectralResponse f = f0 + (1.0f - f0) * powf(1.0f - t, 5.0f);

  auto ggx = NormalDistribution(frame, alpha);
  float j = 1.0f / (4.0f * m_dot_o);
  float result = kInvPi * n_dot_o * (1.0f - f.monochromatic()) + ggx.pdf(m, data.w_i, data.w_o) * j * f.monochromatic();
  ETX_VALIDATE(result);
  return result;
}

ETX_GPU_CODE bool continue_tracing_impl(const Material& material, const float2& tex, const Scene& scene, Sampler& smp) {
  return false;
}

ETX_GPU_CODE bool is_delta_impl(const Material& material, const float2& tex, const Scene& scene, Sampler& smp) {
  return max(material.roughness.x, material.roughness.y) <= kDeltaAlphaTreshold;
}

}  // namespace GenericBSDF
}  // namespace etx
