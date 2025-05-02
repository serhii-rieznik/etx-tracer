namespace etx {

namespace DiffuseBSDF {

ETX_GPU_CODE BSDFEval diffuse_layer(const BSDFData& data, const float3& local_w_i, const float3& local_w_o, const Material& mtl, const Scene& scene, Sampler& smp) {
  if (local_w_o.z <= 0.0f)
    return {data.spectrum_sample, 0.0f};

  SpectralResponse diffuse = apply_image(data.spectrum_sample, mtl.transmittance, data.tex, scene, nullptr);

  BSDFEval eval = {};
  eval.eta = 1.0f;

  auto roughness = evaluate_roughness(mtl, data.tex, scene);
  switch (mtl.diffuse_variation) {
    case 1: {
      eval.bsdf = external::eval_diffuse(smp, local_w_i, local_w_o, roughness, diffuse);
      eval.func = eval.bsdf / local_w_o.z;
      break;
    }
    case 2: {
      eval.func = external::vMFdiffuseBRDF(local_w_i, local_w_o, roughness, diffuse);
      eval.bsdf = eval.func * local_w_o.z;
      break;
    }
    default: {
      eval.func = diffuse / kPi;
      ETX_VALIDATE(eval.func);
      eval.bsdf = eval.func * local_w_o.z;
      ETX_VALIDATE(eval.bsdf);
      break;
    }
  }
  ETX_VALIDATE(eval.bsdf);

  eval.pdf = kInvPi * local_w_o.z;
  ETX_VALIDATE(eval.pdf);

  return eval;
}

ETX_GPU_CODE BSDFSample sample(const BSDFData& data, const Material& mtl, const Scene& scene, Sampler& smp) {
  auto frame = data.get_normal_frame();
  auto local_w_i = frame.to_local(-data.w_i);
  auto roughness = evaluate_roughness(mtl, data.tex, scene);

  BSDFSample result = {};
  result.eta = 1.0f;
  result.properties = BSDFSample::Reflection | BSDFSample::Diffuse;
  result.medium_index = mtl.ext_medium;

  float3 local_w_o = {};
  if (mtl.diffuse_variation == 1) {
    SpectralResponse diffuse = apply_image(data.spectrum_sample, mtl.transmittance, data.tex, scene, nullptr);
    local_w_o = external::sample_diffuse(smp, local_w_i, roughness, diffuse, result.weight);
    ETX_VALIDATE(result.weight);
    result.pdf = kInvPi * local_w_o.z;
    ETX_VALIDATE(result.pdf);
  } else {
    float2 cos_rnd = smp.has_fixed() ? float2{smp.fixed_u, smp.fixed_v} : smp.next_2d();
    local_w_o = sample_cosine_distribution(cos_rnd, 1.0f);
    auto dl = diffuse_layer(data, local_w_i, local_w_o, mtl, scene, smp);
    result.weight = dl.bsdf / dl.pdf;
    ETX_VALIDATE(result.weight);
    result.pdf = dl.pdf;
  }

  result.w_o = frame.from_local(local_w_o);
  return result;
}

ETX_GPU_CODE BSDFEval evaluate(const BSDFData& data, const float3& in_w_o, const Material& mtl, const Scene& scene, Sampler& smp) {
  auto frame = data.get_normal_frame();
  auto local_w_o = frame.to_local(in_w_o);

  if (local_w_o.z <= kEpsilon)
    return {data.spectrum_sample, 0.0f};

  auto local_w_i = frame.to_local(-data.w_i);
  return diffuse_layer(data, local_w_i, local_w_o, mtl, scene, smp);
}

ETX_GPU_CODE float pdf(const BSDFData& data, const float3& w_o, const Material& mtl, const Scene& scene, Sampler& smp) {
  float n_dot_o = dot(data.front_fracing_normal(), w_o);
  if (n_dot_o <= kEpsilon)
    return 0.0f;

  float result = kInvPi * n_dot_o;
  ETX_VALIDATE(result);
  return result;
}

ETX_GPU_CODE bool is_delta(const Material& material, const float2& tex, const Scene& scene, Sampler& smp) {
  return false;
}

ETX_GPU_CODE SpectralResponse albedo(const BSDFData& data, const Material& mtl, const Scene& scene, Sampler& smp) {
  return apply_image(data.spectrum_sample, mtl.transmittance, data.tex, scene, nullptr);
}

}  // namespace DiffuseBSDF

namespace TranslucentBSDF {

ETX_GPU_CODE BSDFSample sample(const BSDFData& data, const Material& mtl, const Scene& scene, Sampler& smp) {
  auto frame = data.get_normal_frame();
  auto tr = apply_image(data.spectrum_sample, mtl.transmittance, data.tex, scene, nullptr);
  auto rf = apply_image(data.spectrum_sample, mtl.reflectance, data.tex, scene, nullptr);

  float tr_value = tr.monochromatic();
  float rf_value = rf.monochromatic();
  float total = tr_value + rf_value;

  if (total == 0.0f)
    return {data.spectrum_sample};

  auto w_o = sample_cosine_distribution(smp.next_2d(), frame.nrm, 1.0f);
  float n_dot_o = fabsf(dot(w_o, frame.nrm));

  BSDFSample result = {};

  if (smp.next() < tr_value / total) {
    result.eta = 1.0f;
    result.w_o = -w_o;
    result.pdf = n_dot_o * kInvPi * (tr_value / total);
    result.properties = BSDFSample::Diffuse | BSDFSample::Transmission | BSDFSample::MediumChanged;
    result.medium_index = frame.entering_material() ? mtl.int_medium : mtl.ext_medium;
    result.weight = tr;
  } else {
    result.eta = 1.0f;
    result.w_o = w_o;
    result.pdf = n_dot_o * kInvPi * (rf_value / total);
    result.properties = BSDFSample::Diffuse | BSDFSample::Reflection;
    result.medium_index = data.medium_index;
    result.weight = rf;
  }

  return result;
}

ETX_GPU_CODE BSDFEval evaluate(const BSDFData& data, const float3& w_o, const Material& mtl, const Scene& scene, Sampler& smp) {
  auto frame = data.get_normal_frame();
  float n_dot_i = -dot(frame.nrm, data.w_i);
  float n_dot_o = dot(frame.nrm, w_o);

  bool reflection = n_dot_o * n_dot_i > 0.0f;

  auto tr = apply_image(data.spectrum_sample, mtl.transmittance, data.tex, scene, nullptr);
  auto rf = apply_image(data.spectrum_sample, mtl.reflectance, data.tex, scene, nullptr);

  float tr_value = tr.monochromatic();
  float rf_value = rf.monochromatic();
  float total = tr_value + rf_value;

  if (total == 0.0f)
    return {data.spectrum_sample, 0.0f};

  float scale = (total > 1.0f) ? 1.0f / total : 1.0f;

  n_dot_o = fabsf(n_dot_o);

  BSDFEval result;
  result.func = (reflection ? rf : tr) * (scale * kInvPi);
  result.bsdf = result.func * n_dot_o;
  result.pdf = kInvPi * n_dot_o * (reflection ? rf_value / total : tr_value / total);
  return result;
}

ETX_GPU_CODE float pdf(const BSDFData& data, const float3& w_o, const Material& mtl, const Scene& scene, Sampler& smp) {
  auto frame = data.get_normal_frame();
  float n_dot_i = -dot(frame.nrm, data.w_i);
  float n_dot_o = dot(frame.nrm, w_o);
  float tr_value = apply_image(data.spectrum_sample, mtl.transmittance, data.tex, scene, nullptr).monochromatic();
  float rf_value = apply_image(data.spectrum_sample, mtl.reflectance, data.tex, scene, nullptr).monochromatic();
  float total = tr_value + rf_value;
  bool reflection = n_dot_o * n_dot_i > 0.0f;
  return (total == 0.0f) ? 0.0f : kInvPi * fabsf(n_dot_o) * (reflection ? rf_value / total : tr_value / total);
}

ETX_GPU_CODE bool is_delta(const Material& material, const float2& tex, const Scene& scene, Sampler& smp) {
  return false;
}

ETX_GPU_CODE SpectralResponse albedo(const BSDFData& data, const Material& mtl, const Scene& scene, Sampler& smp) {
  return apply_image(data.spectrum_sample, mtl.transmittance, data.tex, scene, nullptr);
}

}  // namespace TranslucentBSDF

namespace MirrorBSDF {

ETX_GPU_CODE BSDFSample sample(const BSDFData& data, const Material& mtl, const Scene& scene, Sampler& smp) {
  auto frame = data.get_normal_frame();

  BSDFSample result;
  result.w_o = normalize(reflect(data.w_i, frame.nrm));
  result.weight = apply_image(data.spectrum_sample, mtl.transmittance, data.tex, scene, nullptr);
  result.pdf = 1.0f;
  result.properties = BSDFSample::Delta | BSDFSample::Reflection;
  return result;
}

ETX_GPU_CODE BSDFEval evaluate(const BSDFData& data, const float3& w_o, const Material& mtl, const Scene& scene, Sampler& smp) {
  return {data.spectrum_sample, 0.0f};
}

ETX_GPU_CODE float pdf(const BSDFData& data, const float3& w_o, const Material& mtl, const Scene& scene, Sampler& smp) {
  return 0.0f;
}

ETX_GPU_CODE bool is_delta(const Material& material, const float2& tex, const Scene& scene, Sampler& smp) {
  return true;
}

ETX_GPU_CODE SpectralResponse albedo(const BSDFData& data, const Material& mtl, const Scene& scene, Sampler& smp) {
  return {data.spectrum_sample, 1.0f};
}

}  // namespace MirrorBSDF

namespace BoundaryBSDF {

ETX_GPU_CODE BSDFSample sample(const BSDFData& data, const Material& mtl, const Scene& scene, Sampler& smp) {
  bool entering_material = dot(data.nrm, data.w_i) < 0.0f;

  BSDFSample result;
  result.w_o = data.w_i;
  result.pdf = 1.0f;
  result.weight = {data.spectrum_sample, 1.0f};
  result.properties = BSDFSample::Transmission | BSDFSample::MediumChanged;
  result.medium_index = entering_material ? mtl.int_medium : mtl.ext_medium;
  return result;
}

ETX_GPU_CODE BSDFEval evaluate(const BSDFData& data, const float3& w_o, const Material& mtl, const Scene& scene, Sampler& smp) {
  return {data.spectrum_sample, 0.0f};
}

ETX_GPU_CODE float pdf(const BSDFData& data, const float3& w_o, const Material& mtl, const Scene& scene, Sampler& smp) {
  return 0.0f;
}

ETX_GPU_CODE bool is_delta(const Material& material, const float2& tex, const Scene& scene, Sampler& smp) {
  return false;
}

ETX_GPU_CODE SpectralResponse albedo(const BSDFData& data, const Material& mtl, const Scene& scene, Sampler& smp) {
  return {data.spectrum_sample, 1.0f};
}

}  // namespace BoundaryBSDF

}  // namespace etx
