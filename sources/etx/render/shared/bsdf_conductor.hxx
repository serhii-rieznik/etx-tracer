namespace etx {
namespace DeltaConductorBSDF {

ETX_GPU_CODE BSDFSample sample(struct Sampler& smp, const BSDFData& data, const Scene& scene) {
  Frame frame;
  if (data.check_side(frame) == false) {
    return {{data.spectrum_sample.wavelength, 0.0f}};
  }

  SpectralResponse f = fresnel::conductor(data.spectrum_sample, data.w_i, frame.nrm, data.material.ext_ior(data.spectrum_sample), data.material.int_ior(data.spectrum_sample));
  auto specular = bsdf::apply_image(data.spectrum_sample, data.material.specular(data.spectrum_sample), data.material.specular_image_index, data.tex, scene);

  BSDFSample result;
  result.w_o = normalize(reflect(data.w_i, frame.nrm));
  result.weight = specular * f;
  ETX_VALIDATE(result.weight);
  result.pdf = 1.0f;
  result.properties = BSDFSample::DeltaReflection;
  return result;
}

ETX_GPU_CODE BSDFEval evaluate(const BSDFData& data, const Scene& scene) {
  return {data.spectrum_sample.wavelength, 0.0f};
}

ETX_GPU_CODE float pdf(const BSDFData& data, const Scene& scene) {
  return 0.0f;
}

}  // namespace DeltaConductorBSDF

namespace ConductorBSDF {

ETX_GPU_CODE BSDFSample sample(Sampler& smp, const BSDFData& data, const Scene& scene) {
  if (data.material.is_delta()) {
    return DeltaConductorBSDF::sample(smp, data, scene);
  }

  Frame frame;
  if (data.check_side(frame) == false) {
    return {{data.spectrum_sample.wavelength, 0.0f}};
  }

  auto ggx = bsdf::NormalDistribution(frame, data.material.roughness);
  auto m = ggx.sample(smp, data.w_i);

  BSDFData eval_data = data;
  eval_data.w_o = normalize(reflect(data.w_i, m));
  return {eval_data.w_o, evaluate(eval_data, scene), 0};
}

ETX_GPU_CODE BSDFEval evaluate(const BSDFData& data, const Scene& scene) {
  if (data.material.is_delta()) {
    return DeltaConductorBSDF::evaluate(data, scene);
  }

  Frame frame;
  if (data.check_side(frame) == false) {
    return {data.spectrum_sample.wavelength, 0.0f};
  }

  float n_dot_i = -dot(frame.nrm, data.w_i);
  float n_dot_o = dot(frame.nrm, data.w_o);
  if ((n_dot_o <= kEpsilon) || (n_dot_i <= kEpsilon)) {
    return {data.spectrum_sample.wavelength, 0.0f};
  }

  float3 m = normalize(data.w_o - data.w_i);
  float m_dot_o = dot(m, data.w_o);
  if (m_dot_o <= kEpsilon) {
    return {data.spectrum_sample.wavelength, 0.0f};
  }

  auto ggx = bsdf::NormalDistribution(frame, data.material.roughness);
  auto eval = ggx.evaluate(m, data.w_i, data.w_o);

  SpectralResponse f = fresnel::conductor(data.spectrum_sample, data.w_i, frame.nrm, data.material.ext_ior(data.spectrum_sample), data.material.int_ior(data.spectrum_sample));

  auto specular = bsdf::apply_image(data.spectrum_sample, data.material.specular(data.spectrum_sample), data.material.specular_image_index, data.tex, scene);

  BSDFEval result;
  result.func = specular * (f * eval.ndf * eval.visibility / (4.0f * n_dot_i * n_dot_o));
  ETX_VALIDATE(result.func);

  result.bsdf = specular * (f * eval.ndf * eval.visibility / (4.0f * n_dot_i));
  ETX_VALIDATE(result.bsdf);

  result.pdf = eval.pdf / (4.0f * m_dot_o);
  ETX_VALIDATE(result.pdf);

  if (result.pdf > 0.0f) {
    result.weight = result.bsdf / result.pdf;
    ETX_VALIDATE(result.weight);
  }

  return result;
}

ETX_GPU_CODE float pdf(const BSDFData& data, const Scene& scene) {
  if (data.material.is_delta()) {
    return DeltaConductorBSDF::pdf(data, scene);
  }

  Frame frame;
  if (data.check_side(frame) == false) {
    return 0.0f;
  }

  float3 m = normalize(data.w_o - data.w_i);
  float m_dot_o = dot(m, data.w_o);
  if (m_dot_o <= kEpsilon) {
    return 0.0f;
  }

  auto ggx = bsdf::NormalDistribution(frame, data.material.roughness);
  return ggx.pdf(m, data.w_i, data.w_o) / (4.0f * m_dot_o);
}

ETX_GPU_CODE bool continue_tracing(const Material& material, const float2& tex, const Scene& scene, struct Sampler& smp) {
  if (material.specular_image_index == kInvalidIndex) {
    return false;
  }
  const auto& img = scene.images[material.specular_image_index];
  return (img.options & Image::HasAlphaChannel) && (img.evaluate(tex).w < smp.next());
}

}  // namespace ConductorBSDF

}  // namespace etx
