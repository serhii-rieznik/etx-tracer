namespace etx {
namespace GenericBSDF {

ETX_GPU_CODE float2 remap_alpha(float2 a) {
  return sqr(max(a, float2{1.0f / 16.0f, 1.0f / 16.0f}));
}

ETX_GPU_CODE float remap_metalness(float m) {
  return sqrtf(m);
}

ETX_GPU_CODE BSDFSample sample(const BSDFData& data, const Scene& scene, Sampler& smp) {
  Frame frame;
  if (data.check_side(frame) == false) {
    return {};
  }

  float2 alpha = data.material.roughness;
  float metalness = data.material.metalness;
  if (data.material.metal_roughness_image_index != kInvalidIndex) {
    auto value = scene.images[data.material.metal_roughness_image_index].evaluate(data.tex);
    alpha *= value.y;
    metalness *= value.z;
  }
  alpha = remap_alpha(alpha);
  metalness = remap_metalness(metalness);

  auto ggx = NormalDistribution(frame, alpha);
  auto m = ggx.sample(smp, data.w_i);

  auto diffuse = bsdf::apply_image(data.spectrum_sample, data.material.diffuse(data.spectrum_sample), data.material.diffuse_image_index, data.tex, scene);

  float t = fabsf(dot(data.w_i, m));
  SpectralResponse f0 = SpectralResponse{data.spectrum_sample.wavelength, 0.04f} * (1.0f - metalness) + diffuse * metalness;
  SpectralResponse f = f0 + (1.0f - f0) * std::pow(1.0f - t, 5.0f);

  uint32_t properties = 0;
  BSDFData eval_data = data;
  if (smp.next() <= f.monochromatic()) {
    eval_data.w_o = normalize(reflect(data.w_i, m));
  } else {
    eval_data.w_o = sample_cosine_distribution(smp.next(), smp.next(), frame.nrm, 1.0f);
    properties = BSDFSample::Diffuse;
  }

  return {eval_data.w_o, evaluate(eval_data, scene, smp), properties};
}

ETX_GPU_CODE BSDFEval evaluate(const BSDFData& data, const Scene& scene, Sampler& smp) {
  Frame frame;
  if (data.check_side(frame) == false) {
    return {data.spectrum_sample.wavelength, 0.0f};
  }

  float n_dot_o = dot(frame.nrm, data.w_o);
  float n_dot_i = -dot(frame.nrm, data.w_i);
  if ((n_dot_o <= kEpsilon) || (n_dot_i <= kEpsilon)) {
    return {data.spectrum_sample.wavelength, 0.0f};
  }

  float2 alpha = data.material.roughness;
  float metalness = data.material.metalness;
  if (data.material.metal_roughness_image_index != kInvalidIndex) {
    auto value = scene.images[data.material.metal_roughness_image_index].evaluate(data.tex);
    alpha *= value.y;
    metalness *= value.z;
  }
  alpha = remap_alpha(alpha);
  metalness = remap_metalness(metalness);

  float3 m = normalize(data.w_o - data.w_i);
  float m_dot_o = dot(m, data.w_o);

  auto diffuse = bsdf::apply_image(data.spectrum_sample, data.material.diffuse(data.spectrum_sample), data.material.diffuse_image_index, data.tex, scene);
  auto specular = bsdf::apply_image(data.spectrum_sample, data.material.specular(data.spectrum_sample), data.material.specular_image_index, data.tex, scene);

  float t = fabsf(dot(data.w_i, m));
  SpectralResponse f0 = SpectralResponse{data.spectrum_sample.wavelength, 0.04f * (1.0f - metalness)} + diffuse * metalness;
  SpectralResponse f = f0 + (1.0f - f0) * std::pow(1.0f - t, 5.0f);

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

ETX_GPU_CODE float pdf(const BSDFData& data, const Scene& scene) {
  Frame frame;
  if (data.check_side(frame) == false) {
    return 0.0f;
  }

  float n_dot_o = dot(frame.nrm, data.w_o);
  if (n_dot_o <= kEpsilon) {
    return 0.0f;
  }

  float3 m = normalize(data.w_o - data.w_i);
  float m_dot_o = dot(m, data.w_o);

  auto diffuse = bsdf::apply_image(data.spectrum_sample, data.material.diffuse(data.spectrum_sample), data.material.diffuse_image_index, data.tex, scene);

  float2 alpha = data.material.roughness;
  float metalness = data.material.metalness;
  if (data.material.metal_roughness_image_index != kInvalidIndex) {
    auto value = scene.images[data.material.metal_roughness_image_index].evaluate(data.tex);
    alpha *= value.y;
    metalness *= value.z;
  }
  alpha = remap_alpha(alpha);
  metalness = remap_metalness(metalness);

  float t = fabsf(dot(data.w_i, m));
  SpectralResponse f0 = SpectralResponse{data.spectrum_sample.wavelength, 0.04f} * (1.0f - metalness) + diffuse * metalness;
  SpectralResponse f = f0 + (1.0f - f0) * std::pow(1.0f - t, 5.0f);

  auto ggx = NormalDistribution(frame, alpha);
  float j = 1.0f / (4.0f * m_dot_o);
  float result = kInvPi * n_dot_o * (1.0f - f.monochromatic()) + ggx.pdf(m, data.w_i, data.w_o) * j * f.monochromatic();
  ETX_VALIDATE(result);
  return result;
}

ETX_GPU_CODE bool continue_tracing(const Material& material, const float2& tex, const Scene& scene, Sampler& smp) {
  return false;
}

}  // namespace GenericBSDF
}  // namespace etx
