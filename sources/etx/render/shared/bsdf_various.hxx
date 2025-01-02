namespace etx {

namespace DiffuseBSDF {

ETX_GPU_CODE BSDFEval diffuse_layer(const BSDFData& data, const float3& local_w_i, const float3& local_w_o, const Material& mtl, const Scene& scene, Sampler& smp) {
  SpectralResponse diffuse = apply_image(data.spectrum_sample, mtl.transmittance, data.tex, scene, nullptr);

  BSDFEval eval = {};
  eval.eta = 1.0f;

  auto roughness = evaluate_roughness(mtl.roughness, data.tex, scene);
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
      eval.bsdf = eval.func * local_w_o.z;
      break;
    }
  }
  ETX_VALIDATE(eval.bsdf);

  eval.pdf = kInvPi * local_w_o.z;
  ETX_VALIDATE(eval.pdf);

  eval.weight = eval.pdf > 0.0f ? eval.bsdf / eval.pdf : SpectralResponse{data.spectrum_sample, 0.0f};
  ETX_VALIDATE(eval.weight);

  return eval;
}

ETX_GPU_CODE BSDFSample sample(const BSDFData& data, const Material& mtl, const Scene& scene, Sampler& smp) {
  auto frame = data.get_normal_frame();
  auto local_w_i = frame.to_local(-data.w_i);
  auto roughness = evaluate_roughness(mtl.roughness, data.tex, scene);

  BSDFSample result = {};
  result.eta = 1.0f;
  result.properties = BSDFSample::Reflection | BSDFSample::Diffuse;
  result.medium_index = mtl.ext_medium;

  float3 local_w_o = {};
  if (mtl.diffuse_variation == 1) {
    SpectralResponse diffuse = apply_image(data.spectrum_sample, mtl.transmittance, data.tex, scene, nullptr);
    local_w_o = external::sample_diffuse(smp, local_w_i, roughness, diffuse, result.weight);
    result.pdf = kInvPi * local_w_o.z;
    ETX_VALIDATE(result.pdf);
  } else {
    local_w_o = sample_cosine_distribution(smp.next_2d(), 1.0f);
    auto dl = diffuse_layer(data, local_w_i, local_w_o, mtl, scene, smp);
    result.weight = dl.weight;
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

  float t = apply_image(data.spectrum_sample, mtl.transmittance, data.tex, scene, nullptr).average();
  bool transmittance = (smp.next() < t);

  auto diffuse = apply_image(data.spectrum_sample, mtl.transmittance, data.tex, scene, nullptr);

  BSDFSample result;
  result.weight = diffuse;
  result.eta = 1.0f;

  if (transmittance) {
    result.w_o = -sample_cosine_distribution(smp.next_2d(), data.front_fracing_normal(), 1.0f);
    result.properties = BSDFSample::Diffuse | BSDFSample::Transmission | BSDFSample::MediumChanged;
    result.medium_index = frame.entering_material() ? mtl.int_medium : mtl.ext_medium;
  } else {
    result.w_o = sample_cosine_distribution(smp.next_2d(), data.front_fracing_normal(), 1.0f);
    result.properties = BSDFSample::Diffuse | BSDFSample::Reflection;
    result.medium_index = data.medium_index;
  }

  result.pdf = kInvPi * fabsf(dot(result.w_o, data.nrm));

  return result;
}

ETX_GPU_CODE BSDFEval evaluate(const BSDFData& data, const float3& w_o, const Material& mtl, const Scene& scene, Sampler& smp) {
  float n_dot_i = -dot(data.nrm, data.w_i);
  float n_dot_o = dot(data.nrm, w_o);

  float t = apply_image(data.spectrum_sample, mtl.transmittance, data.tex, scene, nullptr).average();
  bool transmittance = (smp.next() < t);
  bool reflection = n_dot_o * n_dot_i > 0.0f;

  if ((reflection && transmittance) || ((reflection == false) && (transmittance == false))) {
    return {data.spectrum_sample, 0.0f};
  }

  auto diffuse = apply_image(data.spectrum_sample, mtl.transmittance, data.tex, scene, nullptr);

  n_dot_o = fabsf(n_dot_o);

  BSDFEval result;
  result.func = diffuse * kInvPi;
  result.bsdf = diffuse * (kInvPi * n_dot_o);
  result.weight = diffuse;
  result.pdf = kInvPi * n_dot_o;
  return result;
}

ETX_GPU_CODE float pdf(const BSDFData& data, const float3& w_o, const Material& mtl, const Scene& scene, Sampler& smp) {
  float n_dot_o = fabsf(dot(data.front_fracing_normal(), w_o));
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
