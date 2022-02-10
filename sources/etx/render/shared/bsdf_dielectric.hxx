#include <stdlib.h>

namespace etx {

namespace DeltaDielectricBSDF {

ETX_GPU_CODE BSDFSample sample(const BSDFData& data, const Scene& scene, Sampler& smp) {
  Frame frame;
  bool entering_material = data.get_normal_frame(frame);

  auto eta_i = (entering_material ? data.material.ext_ior : data.material.int_ior)(data.spectrum_sample).eta.monochromatic();
  auto eta_o = (entering_material ? data.material.int_ior : data.material.ext_ior)(data.spectrum_sample).eta.monochromatic();
  float eta = (eta_i / eta_o);

  SpectralResponse fr;
  if (data.material.thinfilm.image_index != kInvalidIndex) {
    auto t = scene.images[data.material.thinfilm.image_index].evaluate(data.tex);
    float thickness = lerp(data.material.thinfilm.min_thickness, data.material.thinfilm.max_thickness, t.x);
    auto ior_eta = scene.spectrums->thinfilm(data.spectrum_sample).eta.monochromatic();
    fr = fresnel::dielectric_thinfilm(data.spectrum_sample, data.w_i, frame.nrm, eta_i, ior_eta, eta_o, thickness);
  } else {
    fr = fresnel::dielectric(data.spectrum_sample, data.w_i, frame.nrm, eta_i, eta_o);
  }
  float f = fr.monochromatic();

  BSDFSample result;
  if (smp.next() <= f) {
    result.w_o = normalize(reflect(data.w_i, frame.nrm));
    result.pdf = f;
    ETX_VALIDATE(result.pdf);
    result.weight = (fr / f) * data.material.specular(data.spectrum_sample);
    ETX_VALIDATE(result.weight);
    result.properties = BSDFSample::DeltaReflection;
    result.medium_index = entering_material ? data.material.ext_medium : data.material.int_medium;
  } else {
    float cos_theta_i = dot(frame.nrm, -data.w_i);
    float sin_theta_o_squared = (eta * eta) * (1.0f - cos_theta_i * cos_theta_i);
    float cos_theta_o = sqrtf(clamp(1.0f - sin_theta_o_squared, 0.0f, 1.0f));
    result.w_o = normalize(eta * data.w_i + frame.nrm * (eta * cos_theta_i - cos_theta_o));
    result.pdf = 1.0f - f;
    ETX_VALIDATE(result.pdf);

    result.weight = bsdf::apply_image(data.spectrum_sample, data.material.transmittance(data.spectrum_sample), data.material.diffuse_image_index, data.tex, scene);
    ETX_VALIDATE(result.weight);

    result.weight *= (1.0f - fr) / (1.0f - f);
    ETX_VALIDATE(result.weight);

    result.properties = BSDFSample::DeltaTransmission | BSDFSample::MediumChanged;
    result.medium_index = entering_material ? data.material.int_medium : data.material.ext_medium;
    if (data.mode == PathSource::Camera) {
      result.weight *= eta * eta;
    }
  }
  return result;
}

ETX_GPU_CODE BSDFEval evaluate(const BSDFData& data, const Scene& scene, Sampler& smp) {
  return {data.spectrum_sample.wavelength, 0.0f};
}

ETX_GPU_CODE float pdf(const BSDFData& data, const Scene& scene) {
  return 0.0f;
}

}  // namespace DeltaDielectricBSDF
namespace DielectricBSDF {

ETX_GPU_CODE BSDFSample sample(const BSDFData& data, const Scene& scene, Sampler& smp) {
  if (data.material.is_delta()) {
    return DeltaDielectricBSDF::sample(data, scene, smp);
  }

  Frame frame;
  bool entering_material = data.get_normal_frame(frame);
  auto ggx = bsdf::NormalDistribution(frame, data.material.roughness);
  auto m = ggx.sample(smp, data.w_i);

  auto eta_i = (entering_material ? data.material.ext_ior : data.material.int_ior)(data.spectrum_sample).eta.monochromatic();
  auto eta_o = (entering_material ? data.material.int_ior : data.material.ext_ior)(data.spectrum_sample).eta.monochromatic();
  float eta = (eta_i / eta_o);

  SpectralResponse fr;
  if (data.material.thinfilm.image_index != kInvalidIndex) {
    auto t = scene.images[data.material.thinfilm.image_index].evaluate(data.tex);
    float thickness = lerp(data.material.thinfilm.min_thickness, data.material.thinfilm.max_thickness, t.x);
    auto ior_eta = scene.spectrums->thinfilm(data.spectrum_sample).eta.monochromatic();
    fr = fresnel::dielectric_thinfilm(data.spectrum_sample, data.w_i, m, eta_i, ior_eta, eta_o, thickness);
  } else {
    fr = fresnel::dielectric(data.spectrum_sample, data.w_i, m, eta_i, eta_o);
  }

  bool reflection = smp.next() <= fr.monochromatic();
  BSDFData eval_data = data;
  if (reflection) {
    eval_data.w_o = reflect(data.w_i, m);
    if (dot(eval_data.w_o, frame.nrm) <= kEpsilon) {
      return {{data.spectrum_sample.wavelength, 0.0f}};
    }
    eval_data.w_o = normalize(eval_data.w_o);
  } else {
    float cos_theta_i = dot(m, -data.w_i);
    float sin_theta_o_squared = (eta * eta) * (1.0f - cos_theta_i * cos_theta_i);
    float cos_theta_o = sqrtf(clamp(1.0f - sin_theta_o_squared, 0.0f, 1.0f));
    eval_data.w_o = eta * data.w_i + m * (eta * cos_theta_i - cos_theta_o);
    if (dot(eval_data.w_o, frame.nrm) >= -kEpsilon) {
      return {{data.spectrum_sample.wavelength, 0.0f}};
    }
    eval_data.w_o = normalize(eval_data.w_o);
  }

  BSDFSample result = {eval_data.w_o, evaluate(eval_data, scene, smp), reflection ? 0u : BSDFSample::MediumChanged};
  if (entering_material) {
    result.medium_index = reflection ? data.material.ext_medium : data.material.int_medium;
  } else {
    result.medium_index = reflection ? data.material.int_medium : data.material.ext_medium;
  }
  return result;
}

ETX_GPU_CODE BSDFEval evaluate(const BSDFData& data, const Scene& scene, Sampler& smp) {
  if (data.material.is_delta()) {
    return DeltaDielectricBSDF::evaluate(data, scene, smp);
  }

  Frame frame;
  bool entering_material = data.get_normal_frame(frame);

  float n_dot_i = -dot(frame.nrm, data.w_i);
  float n_dot_o = dot(frame.nrm, data.w_o);
  if ((n_dot_o == 0.0f) || (n_dot_i == 0.0f)) {
    return {data.spectrum_sample.wavelength, 0.0f};
  }

  bool reflection = n_dot_o * n_dot_i >= 0.0f;
  auto eta_i = (entering_material ? data.material.ext_ior : data.material.int_ior)(data.spectrum_sample).eta.monochromatic();
  auto eta_o = (entering_material ? data.material.int_ior : data.material.ext_ior)(data.spectrum_sample).eta.monochromatic();

  float3 m = normalize(reflection ? (data.w_o - data.w_i) : (data.w_i * eta_i - data.w_o * eta_o));
  m *= (dot(frame.nrm, m) < 0.0f) ? -1.0f : 1.0f;

  float m_dot_i = -dot(m, data.w_i);
  float m_dot_o = dot(m, data.w_o);

  SpectralResponse fr;
  if (data.material.thinfilm.image_index != kInvalidIndex) {
    auto t = scene.images[data.material.thinfilm.image_index].evaluate(data.tex);
    float thickness = lerp(data.material.thinfilm.min_thickness, data.material.thinfilm.max_thickness, t.x);
    auto ior_eta = scene.spectrums->thinfilm(data.spectrum_sample).eta.monochromatic();
    fr = fresnel::dielectric_thinfilm(data.spectrum_sample, data.w_i, m, eta_i, ior_eta, eta_o, thickness);
  } else {
    fr = fresnel::dielectric(data.spectrum_sample, data.w_i, m, eta_i, eta_o);
  }
  float f = fr.monochromatic();

  auto ggx = bsdf::NormalDistribution(frame, data.material.roughness);
  auto eval = ggx.evaluate(m, data.w_i, data.w_o);
  if (eval.pdf == 0.0f) {
    return {data.spectrum_sample.wavelength, 0.0f};
  }

  BSDFEval result = {};
  if (reflection) {
    auto specular = data.material.specular(data.spectrum_sample);

    result.func = specular * fr * (eval.ndf * eval.visibility / (4.0f * n_dot_i * n_dot_o));
    ETX_VALIDATE(result.func);

    result.bsdf = specular * fr * (eval.ndf * eval.visibility / (4.0f * n_dot_i));
    ETX_VALIDATE(result.bsdf);

    result.weight = specular * (fr / f) * (eval.visibility / eval.g1_in);
    ETX_VALIDATE(result.weight);

    float j = 1.0f / fabsf(4.0f * m_dot_o);
    result.pdf = eval.pdf * f * j;
    ETX_VALIDATE(result.pdf);
  } else if (f < 1.0f) {
    auto transmittance = bsdf::apply_image(data.spectrum_sample, data.material.transmittance(data.spectrum_sample), data.material.diffuse_image_index, data.tex, scene);

    result.func = abs(transmittance * (1.0f - fr) * (m_dot_i * m_dot_o * sqr(eta_o) * eval.visibility * eval.ndf) / (n_dot_i * n_dot_o * sqr(m_dot_i * eta_i + m_dot_o * eta_o)));
    ETX_VALIDATE(result.func);

    result.bsdf = abs(transmittance * (1.0f - fr) * (m_dot_i * m_dot_o * sqr(eta_o) * eval.visibility * eval.ndf) / (n_dot_i * sqr(m_dot_i * eta_i + m_dot_o * eta_o)));
    ETX_VALIDATE(result.bsdf);

    result.weight = transmittance * ((1.0f - fr) / (1.0f - f)) * (eval.visibility / eval.g1_in);
    ETX_VALIDATE(result.weight);

    auto j = sqr(eta_o) * fabsf(m_dot_o) / sqr(m_dot_i * eta_i + m_dot_o * eta_o);
    result.pdf = eval.pdf * (1.0f - f) * j;
    ETX_VALIDATE(result.pdf);

    result.eta = eta_i / eta_o;

    if (data.mode == PathSource::Camera) {
      result.bsdf *= result.eta * result.eta;
      result.weight *= result.eta * result.eta;
    }
  } else {
    result.bsdf = {data.spectrum_sample.wavelength, 0.0f};
    result.func = {data.spectrum_sample.wavelength, 0.0f};
    result.weight = {data.spectrum_sample.wavelength, 0.0f};
    result.pdf = 0.0f;
  }
  return result;
}

ETX_GPU_CODE float pdf(const BSDFData& data, const Scene& scene) {
  if (data.material.is_delta()) {
    return DeltaDielectricBSDF::pdf(data, scene);
  }

  Frame frame;
  bool entering_material = data.get_normal_frame(frame);
  auto ggx = bsdf::NormalDistribution(frame, data.material.roughness);

  float n_dot_i = -dot(frame.nrm, data.w_i);
  float n_dot_o = dot(frame.nrm, data.w_o);
  bool reflection = n_dot_o * n_dot_i >= 0.0f;
  auto eta_i = (entering_material ? data.material.ext_ior : data.material.int_ior)(data.spectrum_sample).eta.monochromatic();
  auto eta_o = (entering_material ? data.material.int_ior : data.material.ext_ior)(data.spectrum_sample).eta.monochromatic();

  float3 m = normalize(reflection ? (data.w_o - data.w_i) : (data.w_i * eta_i - data.w_o * eta_o));
  m *= (dot(frame.nrm, m) < 0.0f) ? -1.0f : 1.0f;

  float m_dot_i = -dot(m, data.w_i);
  float m_dot_o = dot(m, data.w_o);

  SpectralResponse fr;
  if (data.material.thinfilm.image_index != kInvalidIndex) {
    auto t = scene.images[data.material.thinfilm.image_index].evaluate(data.tex);
    float thickness = lerp(data.material.thinfilm.min_thickness, data.material.thinfilm.max_thickness, t.x);
    auto ior_eta = scene.spectrums->thinfilm(data.spectrum_sample).eta.monochromatic();
    fr = fresnel::dielectric_thinfilm(data.spectrum_sample, data.w_i, m, eta_i, ior_eta, eta_o, thickness);
  } else {
    fr = fresnel::dielectric(data.spectrum_sample, data.w_i, m, eta_i, eta_o);
  }
  float f = fr.monochromatic();

  float pdf = ggx.pdf(m, data.w_i, data.w_o);

  if (reflection) {
    float j = 1.0f / fabsf(4.0f * m_dot_o);
    pdf *= f * j;
  } else {
    auto j = sqr(eta_o) * fabsf(m_dot_o) / sqr(m_dot_i * eta_i + m_dot_o * eta_o);
    pdf *= (1.0f - f) * j;
  }

  ETX_VALIDATE(pdf);
  return pdf;
}

ETX_GPU_CODE bool continue_tracing(const Material& material, const float2& tex, const Scene& scene, Sampler& smp) {
  if (material.diffuse_image_index == kInvalidIndex) {
    return false;
  }
  const auto& img = scene.images[material.diffuse_image_index];
  return (img.options & Image::HasAlphaChannel) && (img.evaluate(tex).w < smp.next());
}

}  // namespace DielectricBSDF

namespace ThinfilmBSDF {

ETX_GPU_CODE BSDFSample sample(const BSDFData& data, const Scene& scene, Sampler& smp) {
  Frame frame;
  bool entering_material = data.get_normal_frame(frame);

  auto ext_ior = data.material.ext_ior(data.spectrum_sample).eta.monochromatic();
  auto int_ior = data.material.int_ior(data.spectrum_sample).eta.monochromatic();

  float thickness = spectrum::kLongestWavelength;
  if (data.material.thinfilm.image_index != kInvalidIndex) {
    const auto& img = scene.images[data.material.thinfilm.image_index];
    auto t = img.evaluate(data.tex);
    thickness = lerp(data.material.thinfilm.min_thickness, data.material.thinfilm.max_thickness, t.x);
  }

  SpectralResponse fr = fresnel::dielectric_thinfilm(data.spectrum_sample, data.w_i, frame.nrm, ext_ior, int_ior, ext_ior, thickness);
  float f = fr.monochromatic();

  BSDFSample result = {};
  if (smp.next() <= f) {
    result.w_o = reflect(data.w_i, frame.nrm);
    if (dot(data.w_i, frame.nrm) * dot(result.w_o, frame.nrm) >= 0.0f) {
      return {{data.spectrum_sample.wavelength, 0.0f}};
    }
    result.w_o = normalize(result.w_o);
    result.pdf = f;
    result.weight = data.material.specular(data.spectrum_sample);
    result.weight *= (fr / f);
    result.properties = BSDFSample::DeltaReflection;
    result.medium_index = entering_material ? data.material.ext_medium : data.material.int_medium;
  } else {
    result.w_o = data.w_i;
    result.pdf = 1.0f - f;
    result.weight = bsdf::apply_image(data.spectrum_sample, data.material.transmittance(data.spectrum_sample), data.material.diffuse_image_index, data.tex, scene);
    result.weight *= (1.0f - fr) / (1.0f - f);
    result.properties = BSDFSample::DeltaTransmission | BSDFSample::MediumChanged;
    result.medium_index = entering_material ? data.material.int_medium : data.material.ext_medium;
  }

  return result;
}

ETX_GPU_CODE BSDFEval evaluate(const BSDFData& data, const Scene& scene, Sampler& smp) {
  return {data.spectrum_sample.wavelength, 0.0f};
}

ETX_GPU_CODE float pdf(const BSDFData& data, const Scene& scene) {
  return 0.0f;
}

ETX_GPU_CODE bool continue_tracing(const Material& material, const float2& tex, const Scene& scene, Sampler& smp) {
  if (material.diffuse_image_index == kInvalidIndex) {
    return false;
  }

  const auto& img = scene.images[material.diffuse_image_index];
  return (img.options & Image::HasAlphaChannel) && img.evaluate(tex).w < smp.next();
}

}  // namespace ThinfilmBSDF

namespace MultiscatteringDielectricBSDF {

ETX_GPU_CODE BSDFSample sample(const BSDFData& data, const Scene& scene, Sampler& smp) {
  if (data.material.is_delta()) {
    return DeltaDielectricBSDF::sample(data, scene, smp);
  }

  bsdf::LocalFrame local_frame = {{data.tan, data.btn, data.nrm}};
  auto w_i = local_frame.to_local(-data.w_i);
  float ext_ior = data.material.ext_ior(data.spectrum_sample).eta.monochromatic();
  float int_ior = data.material.int_ior(data.spectrum_sample).eta.monochromatic();
  float m_eta = int_ior / ext_ior;
  float m_invEta = 1.0f / m_eta;
  const float alpha_x = data.material.roughness.x;
  const float alpha_y = data.material.roughness.y;

  BSDFSample result = {};
  if (bsdf::LocalFrame::cos_theta(w_i) > 0) {  // outside
    float weight = {};
    result.w_o = external::sample_dielectric(data.spectrum_sample, smp, w_i, alpha_x, alpha_y, ext_ior, int_ior, weight);

    auto a_data = data;
    a_data.w_o = normalize(local_frame.from_local(result.w_o));
    result.pdf = pdf(a_data, scene);

    if (bsdf::LocalFrame::cos_theta(result.w_o) > 0) {  // reflection
      result.eta = 1.0f;
      result.weight = data.material.specular(data.spectrum_sample) * weight;
    } else {  // refraction
      result.eta = m_eta;
      float factor = (data.mode == PathSource::Camera) ? m_invEta : 1.0f;
      result.weight = data.material.transmittance(data.spectrum_sample) * factor * factor * weight;
    }
  } else {  // inside
    float weight = {};
    result.w_o = -external::sample_dielectric(data.spectrum_sample, smp, -w_i, alpha_x, alpha_y, int_ior, ext_ior, weight);

    auto a_data = data;
    a_data.w_o = normalize(local_frame.from_local(result.w_o));
    result.pdf = pdf(a_data, scene);

    if (bsdf::LocalFrame::cos_theta(result.w_o) > 0) {  // refraction
      result.eta = m_invEta;
      float factor = (data.mode == PathSource::Camera) ? m_eta : 1.0f;
      result.weight = data.material.transmittance(data.spectrum_sample) * factor * factor * weight;
    } else {  // reflection
      result.eta = 1.0f;
      result.weight = data.material.specular(data.spectrum_sample) * weight;
    }
  }

  result.w_o = normalize(local_frame.from_local(result.w_o));
  return result;
}

ETX_GPU_CODE BSDFEval evaluate(const BSDFData& data, const Scene& scene, Sampler& smp) {
  if (data.material.is_delta()) {
    return DeltaDielectricBSDF::evaluate(data, scene, smp);
  }

  bsdf::LocalFrame local_frame = {{data.tan, data.btn, data.nrm}};
  auto w_i = local_frame.to_local(-data.w_i);
  if (bsdf::LocalFrame::cos_theta(w_i) == 0)
    return {};

  auto w_o = local_frame.to_local(data.w_o);

  float ext_ior = data.material.ext_ior(data.spectrum_sample).eta.monochromatic();
  float int_ior = data.material.int_ior(data.spectrum_sample).eta.monochromatic();
  const float alpha_x = data.material.roughness.x;
  const float alpha_y = data.material.roughness.y;

  bool reflection = bsdf::LocalFrame::cos_theta(w_i) * bsdf::LocalFrame::cos_theta(w_o) > 0.0f;

  // TODO : deal with solid angle compression

  SpectralResponse value = {};
  if (bsdf::LocalFrame::cos_theta(w_i) > 0) {
    if (bsdf::LocalFrame::cos_theta(w_o) >= 0) {
      value = (smp.next() > 0.5f) ? external::eval_dielectric(data.spectrum_sample, smp, w_i, w_o, true, alpha_x, alpha_y, ext_ior, int_ior)
                                  : external::eval_dielectric(data.spectrum_sample, smp, w_o, w_i, true, alpha_x, alpha_y, ext_ior, int_ior) / bsdf::LocalFrame::cos_theta(w_i);
    } else {
      value = (smp.next() > 0.5f) ? external::eval_dielectric(data.spectrum_sample, smp, w_i, w_o, false, alpha_x, alpha_y, ext_ior, int_ior)
                                  : external::eval_dielectric(data.spectrum_sample, smp, -w_o, -w_i, false, alpha_x, alpha_y, int_ior, ext_ior) / bsdf::LocalFrame::cos_theta(w_i);
    }
  } else if (bsdf::LocalFrame::cos_theta(w_o) <= 0) {
    value = (smp.next() > 0.5f) ? external::eval_dielectric(data.spectrum_sample, smp, -w_i, -w_o, true, alpha_x, alpha_y, int_ior, ext_ior)
                                : external::eval_dielectric(data.spectrum_sample, smp, -w_o, -w_i, true, alpha_x, alpha_y, int_ior, ext_ior) / bsdf::LocalFrame::cos_theta(-w_i);
  } else {
    value = (smp.next() > 0.5f) ? external::eval_dielectric(data.spectrum_sample, smp, -w_i, -w_o, false, alpha_x, alpha_y, int_ior, ext_ior)
                                : external::eval_dielectric(data.spectrum_sample, smp, w_o, w_i, false, alpha_x, alpha_y, ext_ior, int_ior) / bsdf::LocalFrame::cos_theta(-w_i);
  }

  BSDFEval eval;
  eval.func = (reflection ? data.material.specular : data.material.transmittance)(data.spectrum_sample) * (2.0f * value);
  ETX_VALIDATE(eval.func);
  eval.bsdf = eval.func * fabsf(bsdf::LocalFrame::cos_theta(w_o));
  eval.pdf = pdf(data, scene);
  eval.weight = eval.bsdf / eval.pdf;
  ETX_VALIDATE(eval.weight);
  return eval;
}

ETX_GPU_CODE float pdf(const BSDFData& data, const Scene& scene) {
  if (data.material.is_delta()) {
    return DeltaDielectricBSDF::pdf(data, scene);
  }

  bsdf::LocalFrame local_frame = {{data.tan, data.btn, data.nrm}};
  auto w_i = local_frame.to_local(-data.w_i);
  auto w_o = local_frame.to_local(data.w_o);
  float ext_ior = data.material.ext_ior(data.spectrum_sample).eta.monochromatic();
  float int_ior = data.material.int_ior(data.spectrum_sample).eta.monochromatic();
  float m_eta = int_ior / ext_ior;
  float m_invEta = 1.0f / m_eta;
  const float alpha_x = data.material.roughness.x;
  const float alpha_y = data.material.roughness.y;

  bool outside = bsdf::LocalFrame::cos_theta(w_i) > 0;
  bool reflect = bsdf::LocalFrame::cos_theta(w_i) * bsdf::LocalFrame::cos_theta(w_o) > 0;

  float3 wh;
  float dwh_dwo;

  if (reflect) {
    wh = normalize(w_o + w_i);
    dwh_dwo = 1.0f / (4.0f * dot(w_o, wh));
  } else {
    float eta = bsdf::LocalFrame::cos_theta(w_i) > 0 ? m_eta : m_invEta;
    wh = normalize(w_i + w_o * eta);
    float sqrtDenom = dot(w_i, wh) + eta * dot(w_o, wh);
    dwh_dwo = (eta * eta * dot(w_o, wh)) / (sqrtDenom * sqrtDenom);
  }

  wh *= (bsdf::LocalFrame::cos_theta(wh) >= 0.0f) ? 1.0f : -1.0f;

  external::RayInfo ray = {w_i * (outside ? 1.0f : -1.0f), alpha_x, alpha_y};
  float prob = max(0.0f, dot(wh, ray.w)) * external::D_ggx(wh, alpha_x, alpha_y) / (1.0f + ray.Lambda) / bsdf::LocalFrame::cos_theta(ray.w);

  float F = fresnel::dielectric(data.spectrum_sample, dot(w_i, wh), outside ? ext_ior : int_ior, outside ? int_ior : ext_ior).monochromatic();
  prob *= reflect ? F : (1 - F);
  // single-scattering PDF + diffuse
  // otherwise too many fireflies due to lack of multiple-scattering PDF
  // (MIS works even if the PDF is wrong and not normalized)
  return fabsf(prob * dwh_dwo) + fabsf(bsdf::LocalFrame::cos_theta(w_o));
}

ETX_GPU_CODE bool continue_tracing(const Material& material, const float2& tex, const Scene& scene, Sampler& smp) {
  if (material.diffuse_image_index == kInvalidIndex) {
    return false;
  }
  const auto& img = scene.images[material.diffuse_image_index];
  return (img.options & Image::HasAlphaChannel) && (img.evaluate(tex).w < smp.next());
}

}  // namespace MultiscatteringDielectricBSDF
}  // namespace etx
