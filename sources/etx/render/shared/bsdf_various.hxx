namespace etx {

namespace DiffuseBSDF {

ETX_GPU_CODE BSDFSample sample(const BSDFData& data, const Material& mtl, const Scene& scene, Sampler& smp) {
  float3 w_o = sample_cosine_distribution(smp.next_2d(), data.front_fracing_normal(), 1.0f);
  return {w_o, evaluate(data, w_o, mtl, scene, smp), BSDFSample::Diffuse | BSDFSample::Reflection};
}

ETX_GPU_CODE BSDFEval evaluate(const BSDFData& data, const float3& w_o, const Material& mtl, const Scene& scene, Sampler& smp) {
  float n_dot_o = dot(data.front_fracing_normal(), w_o);
  if (n_dot_o <= kEpsilon)
    return {data.spectrum_sample, 0.0f};

  auto diffuse = apply_image(data.spectrum_sample, mtl.diffuse, data.tex, scene, rgb::SpectrumClass::Reflection);

  BSDFEval result;
  result.func = diffuse * kInvPi;
  result.bsdf = diffuse * (kInvPi * n_dot_o);
  result.weight = diffuse;
  result.pdf = kInvPi * n_dot_o;
  return result;
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

}  // namespace DiffuseBSDF

namespace TranslucentBSDF {

ETX_GPU_CODE BSDFSample sample(const BSDFData& data, const Material& mtl, const Scene& scene, Sampler& smp) {
  auto frame = data.get_normal_frame();

  float t = apply_image(data.spectrum_sample, mtl.transmittance, data.tex, scene, rgb::SpectrumClass::Reflection).average();
  bool transmittance = (smp.next() < t);

  auto diffuse = apply_image(data.spectrum_sample, mtl.diffuse, data.tex, scene, rgb::SpectrumClass::Reflection);

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

  float t = apply_image(data.spectrum_sample, mtl.transmittance, data.tex, scene, rgb::SpectrumClass::Reflection).average();
  bool transmittance = (smp.next() < t);
  bool reflection = n_dot_o * n_dot_i > 0.0f;

  if ((reflection && transmittance) || ((reflection == false) && (transmittance == false))) {
    return {data.spectrum_sample, 0.0f};
  }

  auto diffuse = apply_image(data.spectrum_sample, mtl.diffuse, data.tex, scene, rgb::SpectrumClass::Reflection);

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

}  // namespace TranslucentBSDF

namespace MirrorBSDF {

ETX_GPU_CODE BSDFSample sample(const BSDFData& data, const Material& mtl, const Scene& scene, Sampler& smp) {
  auto frame = data.get_normal_frame();

  BSDFSample result;
  result.w_o = normalize(reflect(data.w_i, frame.nrm));
  result.weight = apply_image(data.spectrum_sample, mtl.specular, data.tex, scene, rgb::SpectrumClass::Reflection);
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

}  // namespace BoundaryBSDF

}  // namespace etx
