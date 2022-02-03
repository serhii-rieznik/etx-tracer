namespace etx {

namespace DielectricBSDF {

namespace DeltaDielectricBSDF {

ETX_GPU_CODE BSDFSample sample(struct Sampler& smp, const BSDFData& data, const Scene& scene) {
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

ETX_GPU_CODE BSDFEval evaluate(const BSDFData& data, const Scene& scene) {
  return {data.spectrum_sample.wavelength, 0.0f};
}

ETX_GPU_CODE float pdf(const BSDFData& data, const Scene& scene) {
  return 0.0f;
}

}  // namespace DeltaDielectricBSDF

ETX_GPU_CODE BSDFSample sample(struct Sampler& smp, const BSDFData& data, const Scene& scene) {
  if (data.material.is_delta()) {
    return DeltaDielectricBSDF::sample(smp, data, scene);
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
      // if (dot(data.w_i, frame.nrm) * dot(eval_data.w_o, frame.nrm) >= 0.0f) {
      return {{data.spectrum_sample.wavelength, 0.0f}};
    }
    eval_data.w_o = normalize(eval_data.w_o);
  } else {
    float cos_theta_i = dot(m, -data.w_i);
    float sin_theta_o_squared = (eta * eta) * (1.0f - cos_theta_i * cos_theta_i);
    float cos_theta_o = sqrtf(clamp(1.0f - sin_theta_o_squared, 0.0f, 1.0f));
    eval_data.w_o = eta * data.w_i + m * (eta * cos_theta_i - cos_theta_o);
    if (dot(eval_data.w_o, frame.nrm) >= -kEpsilon) {
      //    if (dot(data.w_i, frame.nrm) * dot(eval_data.w_o, frame.nrm) < 0.0f) {
      return {{data.spectrum_sample.wavelength, 0.0f}};
    }
    eval_data.w_o = normalize(eval_data.w_o);
  }

  BSDFSample result = {eval_data.w_o, evaluate(eval_data, scene), reflection ? 0u : BSDFSample::MediumChanged};
  if (entering_material) {
    result.medium_index = reflection ? data.material.ext_medium : data.material.int_medium;
  } else {
    result.medium_index = reflection ? data.material.int_medium : data.material.ext_medium;
  }
  return result;
}

ETX_GPU_CODE BSDFEval evaluate(const BSDFData& data, const Scene& scene) {
  if (data.material.is_delta()) {
    return DeltaDielectricBSDF::evaluate(data, scene);
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
  } else {
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

ETX_GPU_CODE bool continue_tracing(const Material& material, const float2& tex, const Scene& scene, struct Sampler& smp) {
  if (material.diffuse_image_index == kInvalidIndex) {
    return false;
  }
  const auto& img = scene.images[material.diffuse_image_index];
  return (img.options & Image::HasAlphaChannel) && (img.evaluate(tex).w < smp.next());
}

}  // namespace DielectricBSDF

namespace ThinfilmBSDF {

ETX_GPU_CODE BSDFSample sample(struct Sampler& smp, const BSDFData& data, const Scene& scene) {
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

ETX_GPU_CODE BSDFEval evaluate(const BSDFData& data, const Scene& scene) {
  return {data.spectrum_sample.wavelength, 0.0f};
}

ETX_GPU_CODE float pdf(const BSDFData& data, const Scene& scene) {
  return 0.0f;
}

ETX_GPU_CODE bool continue_tracing(const Material& material, const float2& tex, const Scene& scene, struct Sampler& smp) {
  if (material.diffuse_image_index == kInvalidIndex) {
    return false;
  }

  const auto& img = scene.images[material.diffuse_image_index];
  return (img.options & Image::HasAlphaChannel) && img.evaluate(tex).w < smp.next();
}

}  // namespace ThinfilmBSDF
}  // namespace etx
