namespace etx {

namespace ThinfilmBSDF {

ETX_GPU_CODE BSDFSample sample(const BSDFData& data, const Material& mtl, const Scene& scene, Sampler& smp) {
  auto frame = data.get_normal_frame();
  auto ext_ior = mtl.ext_ior(data.spectrum_sample);
  auto int_ior = mtl.int_ior(data.spectrum_sample);
  auto thinfilm = evaluate_thinfilm(data.spectrum_sample, mtl.thinfilm, data.tex, scene, smp);
  SpectralResponse fr = fresnel::calculate(data.spectrum_sample, dot(data.w_i, data.nrm), ext_ior, int_ior, thinfilm);
  float f = fr.monochromatic();

  BSDFSample result = {};
  if (smp.next() <= f) {
    result.w_o = normalize(reflect(data.w_i, frame.nrm));
    result.pdf = f;
    result.weight = apply_image(data.spectrum_sample, mtl.reflectance, data.tex, scene, nullptr);
    result.weight *= (fr / f);
    result.properties = BSDFSample::Delta | BSDFSample::Reflection;
    result.medium_index = frame.entering_material() ? mtl.ext_medium : mtl.int_medium;
  } else {
    result.w_o = data.w_i;
    result.pdf = 1.0f - f;
    result.weight = apply_image(data.spectrum_sample, mtl.transmittance, data.tex, scene, nullptr);
    result.weight *= (1.0f - fr) / (1.0f - f);
    result.properties = BSDFSample::Delta | BSDFSample::Transmission | BSDFSample::MediumChanged;
    result.medium_index = frame.entering_material() ? mtl.int_medium : mtl.ext_medium;
  }

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
  return apply_image(data.spectrum_sample, mtl.transmittance, data.tex, scene, nullptr);
}

}  // namespace ThinfilmBSDF

namespace DielectricBSDF {

ETX_GPU_CODE BSDFSample sample(const BSDFData& data, const Material& mtl, const Scene& scene, Sampler& smp) {
  LocalFrame local_frame = {data.tan, data.btn, data.nrm};
  auto w_i = local_frame.to_local(-data.w_i);

  bool in_outside = LocalFrame::cos_theta(w_i) > 0;
  float direction_scale = in_outside ? 1.0f : -1.0f;

  auto ext_ior = in_outside ? mtl.ext_ior(data.spectrum_sample) : mtl.int_ior(data.spectrum_sample);
  auto int_ior = in_outside ? mtl.int_ior(data.spectrum_sample) : mtl.ext_ior(data.spectrum_sample);
  auto thinfilm = evaluate_thinfilm(data.spectrum_sample, mtl.thinfilm, data.tex, scene, smp);

  BSDFSample result = {};
  result.weight = {data.spectrum_sample, 1.0f};

  // init
  float2 roughness = evaluate_roughness(mtl.roughness, data.tex, scene);
  external::RayInfo ray = {-direction_scale * w_i, roughness};
  ray.updateHeight(1.0f);
  bool ray_outside = true;

  // random walk
  uint32_t scattering_order = 0;
  while (true) {
    float sampled_height = sampleHeight(ray, smp.next());
    if (sampled_height == kMaxFloat)
      break;

    ray.updateHeight(sampled_height);

    // next direction
    auto sample = external::samplePhaseFunction_dielectric(data.spectrum_sample, smp, -ray.w, roughness,  //
      (ray_outside ? ext_ior : int_ior), (ray_outside ? int_ior : ext_ior), thinfilm);

    result.weight *= sample.weight;

    if (sample.reflection) {
      ray.updateDirection(sample.w_o, roughness);
      ray.updateHeight(ray.h);
    } else {
      ray_outside = !ray_outside;
      ray.updateDirection(-sample.w_o, roughness);
      ray.updateHeight(-ray.h);
    }

    if (scattering_order++ > external::kScatteringOrderMax) {
      return {data.spectrum_sample};
    }
  }

  result.w_o = direction_scale * (ray_outside ? ray.w : -ray.w);

  if (LocalFrame::cos_theta(w_i) * LocalFrame::cos_theta(result.w_o) > 0.0f) {
    result.eta = 1.0f;
    result.weight = (result.weight / result.weight.monochromatic()) * apply_image(data.spectrum_sample, mtl.reflectance, data.tex, scene, nullptr);
    result.properties = BSDFSample::Reflection;
    result.medium_index = in_outside ? mtl.ext_medium : mtl.int_medium;
  } else {
    float eta = (int_ior.eta / ext_ior.eta).monochromatic();
    float factor = (data.path_source == PathSource::Camera) ? sqr(1.0f / eta) : 1.0f;
    result.eta = eta;
    result.weight = (result.weight / result.weight.monochromatic()) * apply_image(data.spectrum_sample, mtl.transmittance, data.tex, scene, nullptr) * factor;
    result.properties = BSDFSample::Transmission | BSDFSample::MediumChanged;
    result.medium_index = in_outside ? mtl.int_medium : mtl.ext_medium;
  }

  result.w_o = normalize(local_frame.from_local(result.w_o));
  result.pdf = pdf(data, result.w_o, mtl, scene, smp);
  ETX_VALIDATE(result.pdf);
  return result;
}

ETX_GPU_CODE BSDFEval evaluate(const BSDFData& data, const float3& in_w_o, const Material& mtl, const Scene& scene, Sampler& smp) {
  LocalFrame local_frame = {data.tan, data.btn, data.nrm};

  auto w_i = local_frame.to_local(-data.w_i);
  if (fabsf(LocalFrame::cos_theta(w_i)) <= kEpsilon)
    return {data.spectrum_sample, 0.0f};

  auto w_o = local_frame.to_local(in_w_o);
  if (fabsf(LocalFrame::cos_theta(w_o)) <= kEpsilon)
    return {data.spectrum_sample, 0.0f};

  auto roughness = evaluate_roughness(mtl.roughness, data.tex, scene);
  auto ext_ior = mtl.ext_ior(data.spectrum_sample);
  auto int_ior = mtl.int_ior(data.spectrum_sample);
  auto thinfilm = evaluate_thinfilm(data.spectrum_sample, mtl.thinfilm, data.tex, scene, smp);
  const float m_eta = (int_ior.eta / ext_ior.eta).monochromatic();

  bool forward_path = smp.next() > 0.5f;

  float factor = (data.path_source == PathSource::Camera) ? sqr(LocalFrame::cos_theta(w_i) < 0.0f ? 1.0f / m_eta : m_eta) : 1.0f;
  float backward_scale = fabsf(1.0f / LocalFrame::cos_theta(w_i));

  SpectralResponse value = {};
  if (LocalFrame::cos_theta(w_i) > 0) {
    if (LocalFrame::cos_theta(w_o) >= 0) {
      value = forward_path ? external::eval_dielectric(data.spectrum_sample, smp, w_i, w_o, true, roughness, ext_ior, int_ior, thinfilm)
                           : external::eval_dielectric(data.spectrum_sample, smp, w_o, w_i, true, roughness, ext_ior, int_ior, thinfilm) * backward_scale;
    } else {
      value = forward_path ? external::eval_dielectric(data.spectrum_sample, smp, w_i, w_o, false, roughness, ext_ior, int_ior, thinfilm)
                           : external::eval_dielectric(data.spectrum_sample, smp, -w_o, -w_i, false, roughness, int_ior, ext_ior, thinfilm) * backward_scale * factor;
      value *= factor;
    }
  } else if (LocalFrame::cos_theta(w_o) <= 0) {
    value = forward_path ? external::eval_dielectric(data.spectrum_sample, smp, -w_i, -w_o, true, roughness, int_ior, ext_ior, thinfilm)
                         : external::eval_dielectric(data.spectrum_sample, smp, -w_o, -w_i, true, roughness, int_ior, ext_ior, thinfilm) * backward_scale;
  } else {
    value = forward_path ? external::eval_dielectric(data.spectrum_sample, smp, -w_i, -w_o, false, roughness, int_ior, ext_ior, thinfilm)
                         : external::eval_dielectric(data.spectrum_sample, smp, w_o, w_i, false, roughness, ext_ior, int_ior, thinfilm) * backward_scale * factor;
    value *= factor;
  }

  if (value.is_zero())
    return {data.spectrum_sample, 0.0f};

  bool reflection = LocalFrame::cos_theta(w_i) * LocalFrame::cos_theta(w_o) > 0.0f;

  BSDFEval eval;
  eval.func = (2.0f * value) * apply_image(data.spectrum_sample, reflection ? mtl.reflectance : mtl.transmittance, data.tex, scene, nullptr);
  ETX_VALIDATE(eval.func);
  eval.bsdf = eval.func * fabsf(LocalFrame::cos_theta(w_o));
  eval.pdf = pdf(data, in_w_o, mtl, scene, smp);
  eval.weight = eval.bsdf / eval.pdf;
  ETX_VALIDATE(eval.weight);
  return eval;
}

ETX_GPU_CODE float pdf(const BSDFData& data, const float3& in_w_o, const Material& mtl, const Scene& scene, Sampler& smp) {
  LocalFrame local_frame = {data.tan, data.btn, data.nrm};

  auto w_i = local_frame.to_local(-data.w_i);
  if (fabsf(LocalFrame::cos_theta(w_i)) <= kEpsilon)
    return 0.0f;

  auto w_o = local_frame.to_local(in_w_o);
  if (fabsf(LocalFrame::cos_theta(w_o)) <= kEpsilon)
    return 0.0f;

  auto roughness = evaluate_roughness(mtl.roughness, data.tex, scene);
  auto ext_ior = mtl.ext_ior(data.spectrum_sample);
  auto int_ior = mtl.int_ior(data.spectrum_sample);
  auto thinfilm = evaluate_thinfilm(data.spectrum_sample, mtl.thinfilm, data.tex, scene, smp);

  const bool outside = LocalFrame::cos_theta(w_i) > 0;
  const bool reflection = LocalFrame::cos_theta(w_i) * LocalFrame::cos_theta(w_o) > 0.0f;

  float3 wh = {};
  float dwh_dwo = 0.0f;

  if (reflection) {
    wh = normalize(w_o + w_i);
    dwh_dwo = 1.0f / (4.0f * dot(w_o, wh));
  } else {
    auto eta = outside ? (int_ior.eta / ext_ior.eta).monochromatic() : (ext_ior.eta / int_ior.eta).monochromatic();
    wh = normalize(w_i + w_o * eta);
    float sqrt_denom = dot(w_i, wh) + eta * dot(w_o, wh);
    dwh_dwo = sqr(eta) * dot(w_o, wh) / sqr(sqrt_denom);
  }

  wh *= (LocalFrame::cos_theta(wh) >= 0.0f) ? 1.0f : -1.0f;

  external::RayInfo ray = {w_i * (outside ? 1.0f : -1.0f), roughness};

  auto d_ggx = external::D_ggx(wh, roughness);
  ETX_VALIDATE(d_ggx);

  float prob = max(0.0f, dot(wh, ray.w) * d_ggx / ((1.0f + ray.Lambda) * LocalFrame::cos_theta(ray.w)));
  ETX_VALIDATE(prob);

  float f = fresnel::calculate(data.spectrum_sample, dot(w_i, wh), outside ? ext_ior : int_ior, outside ? int_ior : ext_ior, thinfilm).monochromatic();
  ETX_VALIDATE(f);

  prob *= reflection ? f : (1.0f - f);

  float result = fabsf(prob * dwh_dwo) + fabsf(LocalFrame::cos_theta(w_o));
  ETX_VALIDATE(result);

  return result;
}

ETX_GPU_CODE bool is_delta(const Material& mtl, const float2& tex, const Scene& scene, Sampler& smp) {
  auto roughness = evaluate_roughness(mtl.roughness, tex, scene);
  return max(roughness.x, roughness.y) <= kDeltaAlphaTreshold;
}

ETX_GPU_CODE SpectralResponse albedo(const BSDFData& data, const Material& mtl, const Scene& scene, Sampler& smp) {
  return apply_image(data.spectrum_sample, mtl.transmittance, data.tex, scene, nullptr);
}

}  // namespace DielectricBSDF
}  // namespace etx
