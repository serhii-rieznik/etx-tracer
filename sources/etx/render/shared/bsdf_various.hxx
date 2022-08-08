namespace etx {
namespace DiffuseBSDF {

ETX_FORWARD_TO_IMPL;

ETX_GPU_CODE BSDFSample sample_impl(const BSDFData& data, const Material& mtl, const Scene& scene, Sampler& smp) {
  auto frame = data.get_normal_frame().frame;

  BSDFData eval_data = data;
  eval_data.w_o = sample_cosine_distribution(smp.next_2d(), frame.nrm, 1.0f);
  if (dot(eval_data.w_o, data.nrm) < kEpsilon)
    return {};

  return {eval_data.w_o, evaluate(eval_data, mtl, scene, smp), BSDFSample::Diffuse};
}

ETX_GPU_CODE BSDFEval evaluate_impl(const BSDFData& data, const Material& mtl, const Scene& scene, Sampler& smp) {
  auto frame = data.get_normal_frame().frame;

  float n_dot_o = dot(frame.nrm, data.w_o);
  if (n_dot_o <= kEpsilon) {
    return {data.spectrum_sample.wavelength, 0.0f};
  }

  auto diffuse = apply_image(data.spectrum_sample, mtl.diffuse, data.tex, scene);

  BSDFEval result;
  result.func = diffuse * kInvPi;
  result.bsdf = diffuse * (kInvPi * n_dot_o);
  result.weight = diffuse;
  result.pdf = kInvPi * n_dot_o;
  ETX_VALIDATE(result.pdf);
  return result;
}

ETX_GPU_CODE float pdf_impl(const BSDFData& data, const Material& mtl, const Scene& scene, Sampler& smp) {
  auto frame = data.get_normal_frame().frame;

  float n_dot_o = dot(frame.nrm, data.w_o);
  if (n_dot_o <= kEpsilon) {
    return 0.0f;
  }
  float result = kInvPi * n_dot_o;
  ETX_VALIDATE(result);
  return result;
}

ETX_GPU_CODE bool continue_tracing_impl(const Material& material, const float2& tex, const Scene& scene, Sampler& smp) {
  if (material.diffuse.image_index == kInvalidIndex) {
    return false;
  }
  const auto& img = scene.images[material.diffuse.image_index];
  return (img.options & Image::HasAlphaChannel) ? (img.evaluate(tex).w <= smp.next()) : false;
}

ETX_GPU_CODE bool is_delta_impl(const Material& material, const float2& tex, const Scene& scene, Sampler& smp) {
  return false;
}

}  // namespace DiffuseBSDF

namespace MultiscatteringDiffuseBSDF {

ETX_FORWARD_TO_IMPL;

ETX_GPU_CODE BSDFSample sample_impl(const BSDFData& data, const Material& mtl, const Scene& scene, Sampler& smp) {
  auto frame = data.get_normal_frame().frame;

  LocalFrame local_frame(frame);
  auto w_i = local_frame.to_local(-data.w_i);
  auto alpha_x = mtl.roughness.x;
  auto alpha_y = mtl.roughness.y;
  auto albedo = apply_image(data.spectrum_sample, mtl.diffuse, data.tex, scene);

  BSDFSample result = {};
  result.eta = 1.0f;
  result.w_o = external::sample_diffuse(smp, w_i, alpha_x, alpha_y, albedo, result.weight);
  ETX_VALIDATE(result.weight);
  result.pdf = kInvPi * LocalFrame::cos_theta(result.w_o);
  result.w_o = normalize(local_frame.from_local(result.w_o));
  return result;
}

ETX_GPU_CODE BSDFEval evaluate_impl(const BSDFData& data, const Material& mtl, const Scene& scene, Sampler& smp) {
  auto frame = data.get_normal_frame().frame;

  LocalFrame local_frame(frame);
  auto w_i = local_frame.to_local(-data.w_i);
  if (LocalFrame::cos_theta(w_i) <= 0.0f) {
    return {data.spectrum_sample.wavelength, 0.0f};
  }

  auto w_o = local_frame.to_local(data.w_o);
  if (LocalFrame::cos_theta(w_o) <= 0.0f) {
    return {data.spectrum_sample.wavelength, 0.0f};
  }

  auto alpha_x = mtl.roughness.x;
  auto alpha_y = mtl.roughness.y;
  auto albedo = apply_image(data.spectrum_sample, mtl.diffuse, data.tex, scene);

  BSDFEval result = {};
  result.func = (smp.next() > 0.5f) ? external::eval_diffuse(smp, w_i, w_o, alpha_x, alpha_y, albedo)  //
                                    : external::eval_diffuse(smp, w_o, w_i, alpha_x, alpha_y, albedo) / LocalFrame::cos_theta(w_i);
  ETX_VALIDATE(result.func);
  result.bsdf = result.func * LocalFrame::cos_theta(w_o);
  ETX_VALIDATE(result.bsdf);
  result.pdf = kInvPi * LocalFrame::cos_theta(w_o);
  ETX_VALIDATE(result.pdf);
  result.weight = result.bsdf / result.pdf;
  ETX_VALIDATE(result.weight);
  result.eta = 1.0f;
  return result;
}

ETX_GPU_CODE float pdf_impl(const BSDFData& data, const Material& mtl, const Scene& scene, Sampler& smp) {
  auto frame = data.get_normal_frame().frame;
  return kInvPi * dot(frame.nrm, data.w_o);
}

ETX_GPU_CODE bool continue_tracing_impl(const Material& material, const float2& tex, const Scene& scene, Sampler& smp) {
  if (material.diffuse.image_index == kInvalidIndex) {
    return false;
  }
  const auto& img = scene.images[material.diffuse.image_index];
  return (img.options & Image::HasAlphaChannel) ? (img.evaluate(tex).w <= smp.next()) : false;
}

ETX_GPU_CODE bool is_delta_impl(const Material& material, const float2& tex, const Scene& scene, Sampler& smp) {
  return false;
}

}  // namespace MultiscatteringDiffuseBSDF

namespace TranslucentBSDF {

ETX_FORWARD_TO_IMPL;

ETX_GPU_CODE BSDFSample sample_impl(const BSDFData& data, const Material& mtl, const Scene& scene, Sampler& smp) {
  bool entering_material = dot(data.nrm, data.w_i) < 0.0f;
  float3 n = entering_material ? -data.nrm : data.nrm;

  BSDFData eval_data = data;
  eval_data.w_o = sample_cosine_distribution(smp.next_2d(), n, 1.0f);

  BSDFSample result = {eval_data.w_o, evaluate(eval_data, mtl, scene, smp), BSDFSample::Diffuse | BSDFSample::MediumChanged};
  result.medium_index = entering_material ? mtl.int_medium : mtl.ext_medium;
  return result;
}

ETX_GPU_CODE BSDFEval evaluate_impl(const BSDFData& data, const Material& mtl, const Scene& scene, Sampler& smp) {
  auto diffuse = apply_image(data.spectrum_sample, mtl.diffuse, data.tex, scene);

  bool entering_material = dot(data.nrm, data.w_i) < 0.0f;
  float3 n = entering_material ? -data.nrm : data.nrm;
  auto n_dot_o = fmaxf(0.0f, dot(n, data.w_o));

  BSDFEval result;
  result.func = diffuse * kInvPi;
  result.bsdf = diffuse * (kInvPi * n_dot_o);
  result.weight = diffuse;
  result.pdf = kInvPi * n_dot_o;
  return result;
}

ETX_GPU_CODE float pdf_impl(const BSDFData& data, const Material& mtl, const Scene& scene, Sampler& smp) {
  bool entering_material = dot(data.nrm, data.w_i) < 0.0f;
  float3 n = entering_material ? -data.nrm : data.nrm;
  auto n_dot_o = fmaxf(0.0f, dot(n, data.w_o));
  return kInvPi * n_dot_o;
}

ETX_GPU_CODE bool continue_tracing_impl(const Material& material, const float2& tex, const Scene& scene, Sampler& smp) {
  if (material.diffuse.image_index == kInvalidIndex) {
    return false;
  }
  const auto& img = scene.images[material.diffuse.image_index];
  return (img.options & Image::HasAlphaChannel) ? (img.evaluate(tex).w <= smp.next()) : false;
}

ETX_GPU_CODE bool is_delta_impl(const Material& material, const float2& tex, const Scene& scene, Sampler& smp) {
  return false;
}

}  // namespace TranslucentBSDF

namespace CoatingBSDF {

ETX_FORWARD_TO_IMPL;

ETX_GPU_CODE float2 remap_alpha(float2 a) {
  return sqr(max(a, float2{1.0f / 16.0f, 1.0f / 16.0f}));
}

ETX_GPU_CODE BSDFSample sample_impl(const BSDFData& data, const Material& mtl, const Scene& scene, Sampler& smp) {
  auto frame = data.get_normal_frame().frame;

  uint32_t properties = 0;
  BSDFData eval_data = data;
  if (smp.next() <= 0.5f) {
    auto ggx = NormalDistribution(frame, remap_alpha(mtl.roughness));
    auto m = ggx.sample(smp, data.w_i);
    eval_data.w_o = normalize(reflect(data.w_i, m));
  } else {
    eval_data.w_o = sample_cosine_distribution(smp.next_2d(), frame.nrm, 1.0f);
    properties = BSDFSample::Diffuse;
  }

  return {eval_data.w_o, evaluate(eval_data, mtl, scene, smp), properties};
}

ETX_GPU_CODE BSDFEval evaluate_impl(const BSDFData& data, const Material& mtl, const Scene& scene, Sampler& smp) {
  auto frame = data.get_normal_frame().frame;

  auto pow5 = [](float value) {
    return sqr(value) * sqr(value) * fabsf(value);
  };

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
  auto f = fresnel::dielectric(data.spectrum_sample, data.w_i, m, eta_e, eta_i, thinfilm);

  auto ggx = NormalDistribution(frame, remap_alpha(mtl.roughness));
  auto eval = ggx.evaluate(m, data.w_i, data.w_o);

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

ETX_GPU_CODE float pdf_impl(const BSDFData& data, const Material& mtl, const Scene& scene, Sampler& smp) {
  auto frame = data.get_normal_frame().frame;

  float3 m = normalize(data.w_o - data.w_i);
  float m_dot_o = dot(m, data.w_o);
  float n_dot_o = dot(frame.nrm, data.w_o);

  if ((n_dot_o <= kEpsilon) || (m_dot_o <= kEpsilon)) {
    return 0.0f;
  }

  auto ggx = NormalDistribution(frame, remap_alpha(mtl.roughness));
  float result = 0.5f * (kInvPi * n_dot_o + ggx.pdf(m, data.w_i, data.w_o) / (4.0f * m_dot_o));
  ETX_VALIDATE(result);
  return result;
}

ETX_GPU_CODE bool continue_tracing_impl(const Material& material, const float2& tex, const Scene& scene, Sampler& smp) {
  if (material.diffuse.image_index == kInvalidIndex) {
    return false;
  }
  const auto& img = scene.images[material.diffuse.image_index];
  return (img.options & Image::HasAlphaChannel) ? (img.evaluate(tex).w <= smp.next()) : false;
}

ETX_GPU_CODE bool is_delta_impl(const Material& material, const float2& tex, const Scene& scene, Sampler& smp) {
  return false;  // TODO : check this
}

}  // namespace CoatingBSDF

namespace MirrorBSDF {

ETX_FORWARD_TO_IMPL;

ETX_GPU_CODE BSDFSample sample_impl(const BSDFData& data, const Material& mtl, const Scene& scene, Sampler& smp) {
  auto frame = data.get_normal_frame().frame;

  BSDFSample result;
  result.w_o = normalize(reflect(data.w_i, frame.nrm));
  result.weight = apply_image(data.spectrum_sample, mtl.specular, data.tex, scene);
  result.pdf = 1.0f;
  result.properties = BSDFSample::DeltaReflection;
  return result;
}

ETX_GPU_CODE BSDFEval evaluate_impl(const BSDFData& data, const Material& mtl, const Scene& scene, Sampler& smp) {
  return {data.spectrum_sample.wavelength, 0.0f};
}

ETX_GPU_CODE float pdf_impl(const BSDFData& data, const Material& mtl, const Scene& scene, Sampler& smp) {
  return 0.0f;
}

ETX_GPU_CODE bool continue_tracing_impl(const Material& material, const float2& tex, const Scene& scene, Sampler& smp) {
  return false;
}

ETX_GPU_CODE bool is_delta_impl(const Material& material, const float2& tex, const Scene& scene, Sampler& smp) {
  return true;
}

}  // namespace MirrorBSDF

namespace BoundaryBSDF {

ETX_FORWARD_TO_IMPL;

ETX_GPU_CODE BSDFSample sample_impl(const BSDFData& data, const Material& mtl, const Scene& scene, Sampler& smp) {
  bool entering_material = dot(data.nrm, data.w_i) < 0.0f;

  BSDFSample result;
  result.w_o = data.w_i;
  result.pdf = 1.0f;
  result.weight = {data.spectrum_sample.wavelength, 1.0f};
  result.properties = BSDFSample::MediumChanged;
  result.medium_index = entering_material ? mtl.int_medium : mtl.ext_medium;
  return result;
}

ETX_GPU_CODE BSDFEval evaluate_impl(const BSDFData& data, const Material& mtl, const Scene& scene, Sampler& smp) {
  return {data.spectrum_sample.wavelength, 0.0f};
}

ETX_GPU_CODE float pdf_impl(const BSDFData& data, const Material& mtl, const Scene& scene, Sampler& smp) {
  return 0.0f;
}

ETX_GPU_CODE bool continue_tracing_impl(const Material& material, const float2& tex, const Scene& scene, Sampler& smp) {
  return false;
}

ETX_GPU_CODE bool is_delta_impl(const Material& material, const float2& tex, const Scene& scene, Sampler& smp) {
  return false;
}

}  // namespace BoundaryBSDF

namespace MixtureBSDF {

ETX_GPU_CODE float get_factor(const Material& material, const float2& tex, const Scene& scene, Sampler& smp) {
  float result = material.mixture;
  if (material.mixture_image_index != kInvalidIndex) {
    result *= scene.images[material.mixture_image_index].evaluate(tex).x;
  }
  return result;
}

ETX_GPU_CODE uint32_t select_material(const Material& material, const float2& tex, const Scene& scene, Sampler& smp) {
  if ((material.mixture_0 == kInvalidIndex) && (material.mixture_1 == kInvalidIndex)) {
    return kInvalidIndex;
  }
  float f = get_factor(material, tex, scene, smp);
  if ((f >= 1.0f) || (material.mixture_0 == kInvalidIndex)) {
    return material.mixture_1;
  }
  if ((f <= 0.0f) || (material.mixture_1 == kInvalidIndex)) {
    return material.mixture_0;
  }
  return f <= smp.next() ? material.mixture_0 : material.mixture_1;
}

ETX_GPU_CODE BSDFSample sample(const BSDFData& data, const Material& mtl, const Scene& scene, Sampler& smp) {
  uint32_t m = select_material(mtl, data.tex, scene, smp);
  if (m == kInvalidIndex) {
    return DiffuseBSDF::sample_impl(data, mtl, scene, smp);
  }
  return bsdf::sample_impl(data, scene.materials[m], scene, smp);
}

ETX_GPU_CODE BSDFEval evaluate(const BSDFData& data, const Material& mtl, const Scene& scene, Sampler& smp) {
  uint32_t m = select_material(mtl, data.tex, scene, smp);
  if (m == kInvalidIndex) {
    return DiffuseBSDF::evaluate_impl(data, mtl, scene, smp);
  }
  return bsdf::evaluate_impl(data, scene.materials[m], scene, smp);
}

ETX_GPU_CODE float pdf(const BSDFData& data, const Material& mtl, const Scene& scene, Sampler& smp) {
  uint32_t m = select_material(mtl, data.tex, scene, smp);
  if (m == kInvalidIndex) {
    return DiffuseBSDF::pdf_impl(data, mtl, scene, smp);
  }
  return bsdf::pdf_impl(data, scene.materials[m], scene, smp);
}

ETX_GPU_CODE bool continue_tracing(const Material& material, const float2& tex, const Scene& scene, Sampler& smp) {
  uint32_t m = select_material(material, tex, scene, smp);
  if (m == kInvalidIndex) {
    return DiffuseBSDF::continue_tracing_impl(material, tex, scene, smp);
  }
  return bsdf::continue_tracing_impl(scene.materials[m], tex, scene, smp);
}

ETX_GPU_CODE bool is_delta(const Material& material, const float2& tex, const Scene& scene, Sampler& smp) {
  uint32_t m = select_material(material, tex, scene, smp);
  if (m == kInvalidIndex) {
    return DiffuseBSDF::is_delta_impl(material, tex, scene, smp);
  }
  return bsdf::is_delta_impl(scene.materials[m], tex, scene, smp);
}

ETX_GPU_CODE BSDFSample sample_impl(const BSDFData& data, const Material& mtl, const Scene& scene, Sampler& smp) {
  return {};
}

ETX_GPU_CODE BSDFEval evaluate_impl(const BSDFData& data, const Material& mtl, const Scene& scene, Sampler& smp) {
  return {};
}

ETX_GPU_CODE float pdf_impl(const BSDFData& data, const Material& mtl, const Scene& scene, Sampler& smp) {
  return {};
}

ETX_GPU_CODE bool continue_tracing_impl(const Material& material, const float2& tex, const Scene& scene, Sampler& smp) {
  return {};
}

ETX_GPU_CODE bool is_delta_impl(const Material& material, const float2& tex, const Scene& scene, Sampler& smp) {
  return {};
}

}  // namespace MixtureBSDF
}  // namespace etx
