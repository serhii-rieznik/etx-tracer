namespace etx {
namespace DiffuseBSDF {

ETX_GPU_CODE BSDFSample sample(const BSDFData& data, const Material& mtl, const Scene& scene, Sampler& smp) {
  auto frame_info = data.get_normal_frame();
  if (frame_info.entering_material == false) {
    return {{data.spectrum_sample.wavelength, 0.0f}};
  }

  BSDFData eval_data = data;
  eval_data.w_o = sample_cosine_distribution(smp.next_2d(), frame_info.frame.nrm, 1.0f);
  if (dot(eval_data.w_o, data.nrm) < kEpsilon) {
    return {{data.spectrum_sample.wavelength, 0.0f}};
  }

  return {eval_data.w_o, evaluate(eval_data, mtl, scene, smp), BSDFSample::Diffuse | BSDFSample::Reflection};
}

ETX_GPU_CODE BSDFEval evaluate(const BSDFData& data, const Material& mtl, const Scene& scene, Sampler& smp) {
  auto frame_info = data.get_normal_frame();
  if (frame_info.entering_material == false)
    return {data.spectrum_sample.wavelength, 0.0f};

  float n_dot_o = dot(frame_info.frame.nrm, data.w_o);
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

ETX_GPU_CODE float pdf(const BSDFData& data, const Material& mtl, const Scene& scene, Sampler& smp) {
  auto frame_info = data.get_normal_frame();
  if (frame_info.entering_material == false)
    return 0.0f;

  float n_dot_o = dot(frame_info.frame.nrm, data.w_o);
  if (n_dot_o <= kEpsilon) {
    return 0.0f;
  }

  float result = kInvPi * n_dot_o;
  ETX_VALIDATE(result);
  return result;
}

ETX_GPU_CODE bool is_delta(const Material& material, const float2& tex, const Scene& scene, Sampler& smp) {
  return false;
}

}  // namespace DiffuseBSDF

namespace SubsurfaceBSDF {

ETX_GPU_CODE BSDFSample sample(const BSDFData& data, const Material& mtl, const Scene& scene, Sampler& smp) {
  auto frame_info = data.get_normal_frame();
  auto n = frame_info.entering_material ? frame_info.frame.nrm : -frame_info.frame.nrm;

  BSDFData eval_data = data;
  eval_data.w_o = sample_cosine_distribution(smp.next_2d(), n, 1.0f);
  return {eval_data.w_o, evaluate(eval_data, mtl, scene, smp), BSDFSample::Diffuse | BSDFSample::Reflection};
}

ETX_GPU_CODE BSDFEval evaluate(const BSDFData& data, const Material& mtl, const Scene& scene, Sampler& smp) {
  auto frame_info = data.get_normal_frame();
  float n_dot_o = fabsf(dot(frame_info.frame.nrm, data.w_o));
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

ETX_GPU_CODE float pdf(const BSDFData& data, const Material& mtl, const Scene& scene, Sampler& smp) {
  auto frame_info = data.get_normal_frame();
  float n_dot_o = fabsf(dot(frame_info.frame.nrm, data.w_o));
  if (n_dot_o <= kEpsilon) {
    return 0.0f;
  }

  float result = kInvPi * n_dot_o;
  ETX_VALIDATE(result);
  return result;
}

ETX_GPU_CODE bool is_delta(const Material& material, const float2& tex, const Scene& scene, Sampler& smp) {
  return false;
}

}  // namespace SubsurfaceBSDF

namespace MirrorBSDF {

ETX_GPU_CODE BSDFSample sample(const BSDFData& data, const Material& mtl, const Scene& scene, Sampler& smp) {
  auto frame = data.get_normal_frame().frame;

  BSDFSample result;
  result.w_o = normalize(reflect(data.w_i, frame.nrm));
  result.weight = apply_image(data.spectrum_sample, mtl.specular, data.tex, scene);
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
  result.weight = {data.spectrum_sample.wavelength, 1.0f};
  result.properties = BSDFSample::Transmission | BSDFSample::MediumChanged;
  result.medium_index = entering_material ? mtl.int_medium : mtl.ext_medium;
  return result;
}

ETX_GPU_CODE BSDFEval evaluate(const BSDFData& data, const Material& mtl, const Scene& scene, Sampler& smp) {
  return {data.spectrum_sample.wavelength, 0.0f};
}

ETX_GPU_CODE float pdf(const BSDFData& data, const Material& mtl, const Scene& scene, Sampler& smp) {
  return 0.0f;
}

ETX_GPU_CODE bool is_delta(const Material& material, const float2& tex, const Scene& scene, Sampler& smp) {
  return false;
}

}  // namespace BoundaryBSDF

}  // namespace etx
