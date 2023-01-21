namespace etx {
namespace DeltaConductorBSDF {

ETX_GPU_CODE BSDFSample sample(const BSDFData& data, const Material& mtl, const Scene& scene, Sampler& smp) {
  auto frame = data.get_normal_frame().frame;
  auto thinfilm = evaluate_thinfilm(data.spectrum_sample, mtl.thinfilm, data.tex, scene);
  auto f = fresnel::conductor(data.spectrum_sample, data.w_i, frame.nrm, mtl.ext_ior(data.spectrum_sample), mtl.int_ior(data.spectrum_sample), thinfilm);
  auto specular = apply_image(data.spectrum_sample, mtl.specular, data.tex, scene);

  BSDFSample result;
  result.w_o = normalize(reflect(data.w_i, frame.nrm));
  result.weight = specular * f;
  ETX_VALIDATE(result.weight);
  result.pdf = 1.0f;
  result.properties = BSDFSample::Delta | BSDFSample::Reflection;
  return result;
}

ETX_GPU_CODE BSDFEval evaluate(const BSDFData& data, const Material& mtl, const Scene& scene, Sampler& smp) {
  return {data.spectrum_sample.wavelength, 0.0f};
}

ETX_GPU_CODE float pdf(const BSDFData& data, const Material& mtl, const Scene& scene, Sampler& smp) {
  return 0.0f;
}

}  // namespace DeltaConductorBSDF

namespace ConductorBSDF {

ETX_GPU_CODE BSDFSample sample(const BSDFData& data, const Material& mtl, const Scene& scene, Sampler& smp) {
  if (is_delta(mtl, data.tex, scene, smp)) {
    return DeltaConductorBSDF::sample(data, mtl, scene, smp);
  }

  auto frame = data.get_normal_frame().frame;

  LocalFrame local_frame(frame);
  auto w_i = local_frame.to_local(-data.w_i);
  auto alpha_x = mtl.roughness.x;
  auto alpha_y = mtl.roughness.y;
  auto thinfilm = evaluate_thinfilm(data.spectrum_sample, mtl.thinfilm, data.tex, scene);

  BSDFSample result;
  result.properties = BSDFSample::Reflection;
  result.medium_index = data.medium_index;
  result.eta = 1.0f;
  result.w_o = external::sample_conductor(data.spectrum_sample, smp, w_i, alpha_x, alpha_y,  //
    mtl.ext_ior(data.spectrum_sample), mtl.int_ior(data.spectrum_sample), thinfilm, result.weight);
  {
    external::RayInfo ray = {w_i, alpha_x, alpha_y};
    result.pdf = external::D_ggx(normalize(result.w_o + w_i), alpha_x, alpha_y) / (1.0f + ray.Lambda) / (4.0f * w_i.z) + result.w_o.z;
    ETX_VALIDATE(result.pdf);
  }
  result.weight *= apply_image(data.spectrum_sample, mtl.specular, data.tex, scene);
  ETX_VALIDATE(result.weight);

  result.w_o = normalize(local_frame.from_local(result.w_o));
  return result;
}

ETX_GPU_CODE BSDFEval evaluate(const BSDFData& data, const Material& mtl, const Scene& scene, Sampler& smp) {
  if (is_delta(mtl, data.tex, scene, smp)) {
    return DeltaConductorBSDF::evaluate(data, mtl, scene, smp);
  }

  auto frame = data.get_normal_frame().frame;

  LocalFrame local_frame(frame);
  auto w_o = local_frame.to_local(data.w_o);
  if (w_o.z <= kEpsilon) {
    return {data.spectrum_sample.wavelength, 0.0f};
  }
  auto w_i = local_frame.to_local(-data.w_i);
  if (w_i.z <= kEpsilon) {
    return {data.spectrum_sample.wavelength, 0.0f};
  }

  auto alpha_x = mtl.roughness.x;
  auto alpha_y = mtl.roughness.y;
  auto ext_ior = mtl.ext_ior(data.spectrum_sample);
  auto int_ior = mtl.int_ior(data.spectrum_sample);
  auto thinfilm = evaluate_thinfilm(data.spectrum_sample, mtl.thinfilm, data.tex, scene);

  BSDFEval result;
  if (smp.next() > 0.5f) {
    result.bsdf = 2.0f * external::eval_conductor(data.spectrum_sample, smp, w_i, w_o, alpha_x, alpha_y, ext_ior, int_ior, thinfilm);
    ETX_VALIDATE(result.bsdf);
  } else {
    result.bsdf = 2.0f * external::eval_conductor(data.spectrum_sample, smp, w_o, w_i, alpha_x, alpha_y, ext_ior, int_ior, thinfilm) / w_i.z * w_o.z;
    ETX_VALIDATE(result.bsdf);
  }
  result.bsdf *= apply_image(data.spectrum_sample, mtl.specular, data.tex, scene);
  ETX_VALIDATE(result.bsdf);

  result.func = result.bsdf / w_o.z;
  ETX_VALIDATE(result.func);

  {
    external::RayInfo ray = {w_i, alpha_x, alpha_y};
    result.pdf = external::D_ggx(normalize(w_o + w_i), alpha_x, alpha_y) / (1.0f + ray.Lambda) / (4.0f * w_i.z) + w_o.z;
    ETX_VALIDATE(result.pdf);
  }

  result.weight = result.bsdf / result.pdf;
  ETX_VALIDATE(result.weight);

  return result;
}

ETX_GPU_CODE float pdf(const BSDFData& data, const Material& mtl, const Scene& scene, Sampler& smp) {
  if (is_delta(mtl, data.tex, scene, smp)) {
    return DeltaConductorBSDF::pdf(data, mtl, scene, smp);
  }

  auto frame = data.get_normal_frame().frame;

  LocalFrame local_frame(frame);
  auto w_o = local_frame.to_local(data.w_o);
  if (w_o.z <= kEpsilon) {
    return 0.0f;
  }
  auto w_i = local_frame.to_local(-data.w_i);
  if (w_i.z <= kEpsilon) {
    return 0.0f;
  }

  auto alpha_x = mtl.roughness.x;
  auto alpha_y = mtl.roughness.y;
  external::RayInfo ray = {w_i, alpha_x, alpha_y};
  float result = external::D_ggx(normalize(w_o + w_i), alpha_x, alpha_y) / (1.0f + ray.Lambda) / (4.0f * w_i.z) + w_o.z;
  ETX_VALIDATE(result);
  return result;
}

ETX_GPU_CODE bool is_delta(const Material& material, const float2& tex, const Scene& scene, Sampler& smp) {
  return max(material.roughness.x, material.roughness.y) <= kDeltaAlphaTreshold;
}

}  // namespace ConductorBSDF
}  // namespace etx
