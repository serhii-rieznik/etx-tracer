namespace etx {
namespace DiffuseBSDF {

ETX_GPU_CODE BSDFSample sample(struct Sampler& smp, const BSDFData& data, const Scene& scene) {
  Frame frame;
  if (data.check_side(frame) == false) {
    return {{data.spectrum_sample.wavelength, 0.0f}};
  }

  BSDFData eval_data = data;
  eval_data.w_o = sample_cosine_distribution(smp.next(), smp.next(), frame.nrm, 1.0f);
  return {eval_data.w_o, evaluate(eval_data, scene), BSDFSample::Diffuse};
}

ETX_GPU_CODE BSDFEval evaluate(const BSDFData& data, const Scene& scene) {
  Frame frame;
  if (data.check_side(frame) == false) {
    return {data.spectrum_sample.wavelength, 0.0f};
  }

  float n_dot_o = dot(frame.nrm, data.w_o);
  if (n_dot_o <= kEpsilon) {
    return {data.spectrum_sample.wavelength, 0.0f};
  }

  auto diffuse = bsdf::apply_image(data.spectrum_sample, data.material.diffuse(data.spectrum_sample), data.material.diffuse_image_index, data.tex, scene);

  BSDFEval result;
  result.func = diffuse * kInvPi;
  result.bsdf = diffuse * (kInvPi * n_dot_o);
  result.weight = diffuse;
  result.pdf = kInvPi * n_dot_o;
  ETX_VALIDATE(result.pdf);
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
  float result = kInvPi * n_dot_o;
  ETX_VALIDATE(result);
  return result;
}

ETX_GPU_CODE bool continue_tracing(const Material& material, const float2& tex, const Scene& scene, struct Sampler& smp) {
  if (material.diffuse_image_index == kInvalidIndex) {
    return false;
  }

  const auto& img = scene.images[material.diffuse_image_index];
  return (img.options & Image::HasAlphaChannel) && img.evaluate(tex).w < smp.next();
}

}  // namespace DiffuseBSDF

namespace TranslucentBSDF {

ETX_GPU_CODE BSDFSample sample(struct Sampler& smp, const BSDFData& data, const Scene& scene) {
  bool entering_material = dot(data.nrm, data.w_i) < 0.0f;
  float3 n = entering_material ? -data.nrm : data.nrm;

  BSDFData eval_data = data;
  eval_data.w_o = sample_cosine_distribution(smp.next(), smp.next(), n, 1.0f);

  BSDFSample result = {eval_data.w_o, evaluate(eval_data, scene), BSDFSample::Diffuse | BSDFSample::MediumChanged};
  result.medium_index = entering_material ? data.material.int_medium : data.material.ext_medium;
  return result;
}

ETX_GPU_CODE BSDFEval evaluate(const BSDFData& data, const Scene& scene) {
  auto diffuse = bsdf::apply_image(data.spectrum_sample, data.material.diffuse(data.spectrum_sample), data.material.diffuse_image_index, data.tex, scene);
  auto n_dot_o = fabsf(dot(data.nrm, data.w_o));

  BSDFEval result;
  result.func = diffuse * kInvPi;
  result.bsdf = diffuse * (kInvPi * n_dot_o);
  result.weight = diffuse;
  result.pdf = kInvPi * n_dot_o;
  return result;
}

ETX_GPU_CODE float pdf(const BSDFData& data, const Scene& scene) {
  auto n_dot_o = fabsf(dot(data.nrm, data.w_o));
  return kInvPi * n_dot_o;
}

ETX_GPU_CODE bool continue_tracing(const Material& material, const float2& tex, const Scene& scene, struct Sampler& smp) {
  if (material.diffuse_image_index == kInvalidIndex) {
    return false;
  }

  const auto& img = scene.images[material.diffuse_image_index];
  return (img.options & Image::HasAlphaChannel) && img.evaluate(tex).w < smp.next();
}

}  // namespace TranslucentBSDF

namespace MirrorBSDF {

ETX_GPU_CODE BSDFSample sample(struct Sampler& smp, const BSDFData& data, const Scene& scene) {
  Frame frame;
  if (data.check_side(frame) == false) {
    return {{data.spectrum_sample.wavelength, 0.0f}};
  }

  BSDFSample result;
  result.w_o = normalize(reflect(data.w_i, frame.nrm));
  result.weight = bsdf::apply_image(data.spectrum_sample, data.material.specular(data.spectrum_sample), data.material.specular_image_index, data.tex, scene);
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

ETX_GPU_CODE bool continue_tracing(const Material& material, const float2& tex, const Scene& scene, struct Sampler& smp) {
  return false;
}

}  // namespace MirrorBSDF

namespace BoundaryBSDF {

ETX_GPU_CODE BSDFSample sample(struct Sampler& smp, const BSDFData& data, const Scene& scene) {
  bool entering_material = dot(data.nrm, data.w_i) < 0.0f;

  BSDFSample result;
  result.w_o = data.w_i;
  result.pdf = 1.0f;
  result.weight = {data.spectrum_sample.wavelength, 1.0f};
  result.properties = BSDFSample::MediumChanged;
  result.medium_index = entering_material ? data.material.int_medium : data.material.ext_medium;
  return result;
}

ETX_GPU_CODE BSDFEval evaluate(const BSDFData& data, const Scene& scene) {
  return {data.spectrum_sample.wavelength, 0.0f};
}

ETX_GPU_CODE float pdf(const BSDFData& data, const Scene& scene) {
  return 0.0f;
}

ETX_GPU_CODE bool continue_tracing(const Material& material, const float2& tex, const Scene& scene, struct Sampler& smp) {
  return false;
}

}  // namespace BoundaryBSDF
}  // namespace etx