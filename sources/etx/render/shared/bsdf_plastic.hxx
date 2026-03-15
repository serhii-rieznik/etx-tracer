namespace etx {

namespace PlasticBSDF {

struct PlasticMaterial {
  SpectralImage scattering;
  SpectralImage reflectance;
  SampledImage roughness;
  Thinfilm thinfilm;
  RefractiveIndex ext_ior;
  RefractiveIndex int_ior;
};

ETX_SHARED_INLINE SpectralResponse specular_func(const BSDFData& data, const float3& in_w_o, const Material& mtl, Sampler& smp) {
  LocalFrame local_frame = data.get_normal_frame(mtl);

  auto w_i = local_frame_to_local(local_frame, -data.w_i);
  if (LocalFrame::cos_theta(w_i) <= kEpsilon)
    return {data.spectrum_sample, 0.0f};

  auto w_o = local_frame_to_local(local_frame, in_w_o);
  if (LocalFrame::cos_theta(w_o) <= kEpsilon)
    return {data.spectrum_sample, 0.0f};

  auto roughness = evaluate_roughness(mtl, data.tex);
  auto ext_ior = evaluate_refractive_index(mtl.ext_ior, data.spectrum_sample);
  auto int_ior = evaluate_refractive_index(mtl.int_ior, data.spectrum_sample);
  auto m_eta = spectral_response_monochromatic(spectral_response_div(int_ior.eta, ext_ior.eta));
  auto thinfilm = evaluate_thinfilm(data.spectrum_sample, mtl.thinfilm, data.tex, smp);

  SpectralResponse value = external::eval_dielectric(data.spectrum_sample, smp, w_i, w_o, true, roughness, ext_ior, int_ior, thinfilm);
  auto func = 2.0f * value * apply_image(data.spectrum_sample, mtl.reflectance, data.tex);
  ETX_VALIDATE(func);
  return func;
}

ETX_SHARED_INLINE float specular_pdf(const BSDFData& data, const float3& in_w_o, const Material& mtl, Sampler& smp) {
  LocalFrame local_frame = data.get_normal_frame(mtl);

  auto w_i = local_frame_to_local(local_frame, -data.w_i);
  if (LocalFrame::cos_theta(w_i) <= kEpsilon)
    return 0.0f;

  auto w_o = local_frame_to_local(local_frame, in_w_o);
  if (LocalFrame::cos_theta(w_o) <= kEpsilon)
    return 0.0f;

  auto ext_ior = evaluate_refractive_index(mtl.ext_ior, data.spectrum_sample);
  auto int_ior = evaluate_refractive_index(mtl.int_ior, data.spectrum_sample);
  auto roughness = evaluate_roughness(mtl, data.tex);
  auto thinfilm = evaluate_thinfilm(data.spectrum_sample, mtl.thinfilm, data.tex, smp);

  float3 wh = normalize(w_o + w_i);
  float dwh_dwo = 1.0f / (4.0f * dot(w_o, wh));

  external::RayInfo ray = {w_i, roughness};

  auto d_ggx = external::D_ggx(wh, roughness);
  ETX_VALIDATE(d_ggx);

  float prob = max(0.0f, dot(wh, ray.w) * d_ggx / ((1.0f + ray.Lambda) * LocalFrame::cos_theta(ray.w)));
  ETX_VALIDATE(prob);

  auto fr = fresnel::calculate(data.spectrum_sample, dot(w_i, wh), ext_ior, int_ior, thinfilm);
  float f = fr.monochromatic();
  ETX_VALIDATE(f);

  prob *= f;

  float result = fabsf(prob * dwh_dwo);
  ETX_VALIDATE(result);

  return result;
}

ETX_SHARED_INLINE BSDFSample sample(const BSDFData& data, const Material& mtl, Sampler& smp) {
  auto frame = data.get_normal_frame(mtl);

  auto roughness = evaluate_roughness(mtl, data.tex);
  auto ggx = NormalDistribution(frame, roughness);
  auto m = ggx.sample(smp, data.w_i);

  auto ext_ior = evaluate_refractive_index(mtl.ext_ior, data.spectrum_sample);
  auto int_ior = evaluate_refractive_index(mtl.int_ior, data.spectrum_sample);
  auto thinfilm = evaluate_thinfilm(data.spectrum_sample, mtl.thinfilm, data.tex, smp);
  auto f = fresnel::calculate(data.spectrum_sample, dot(data.w_i, m), ext_ior, int_ior, thinfilm);
  auto fr = f.monochromatic();

  auto w_i = local_frame_to_local(frame, -data.w_i);
  if (w_i.z <= kEpsilon)
    return {data.spectrum_sample};

  float3 in_w_o = {};

  bool sample_diffuse = smp.next() > f.monochromatic();

  if (sample_diffuse == false) {
    in_w_o = reflect(data.w_i, m);
    sample_diffuse = dot(frame.nrm, in_w_o) <= kEpsilon;
  }

  if (sample_diffuse) {
    in_w_o = local_frame_from_local(frame, sample_cosine_distribution(smp.next_2d(), 1.0f));
  }

  auto eval = evaluate(data, in_w_o, mtl, smp);

  BSDFSample result = {};
  result.w_o = in_w_o;
  result.weight = eval.bsdf / eval.pdf;
  result.properties = BSDFSample::Reflection | (sample_diffuse ? BSDFSample::Diffuse : 0u);
  result.medium_index = data.current_medium;
  result.pdf = eval.pdf;
  return result;
}

ETX_SHARED_INLINE BSDFEval evaluate(const BSDFData& data, const float3& w_o, const Material& mtl, Sampler& smp) {
  auto frame = data.get_normal_frame(mtl);
  float3 m = normalize(w_o - data.w_i);

  float n_dot_o = dot(frame.nrm, w_o);
  float m_dot_o = dot(m, w_o);

  if ((n_dot_o <= kEpsilon) || (m_dot_o <= kEpsilon)) {
    return {data.spectrum_sample, 0.0f};
  }

  auto eta_e = evaluate_refractive_index(mtl.ext_ior, data.spectrum_sample);
  auto eta_i = evaluate_refractive_index(mtl.int_ior, data.spectrum_sample);
  auto thinfilm = evaluate_thinfilm(data.spectrum_sample, mtl.thinfilm, data.tex, smp);
  auto fr = fresnel::calculate(data.spectrum_sample, dot(data.w_i, m), eta_e, eta_i, thinfilm);

  auto local_w_i = local_frame_to_local(frame, -data.w_i);
  auto local_w_o = local_frame_to_local(frame, w_o);

  auto diff_layer = DiffuseBSDF::diffuse_layer(data, local_w_i, local_w_o, mtl, smp);
  auto spec_layer = specular_func(data, w_o, mtl, smp);
  auto spec_pdf = specular_pdf(data, w_o, mtl, smp);

  BSDFEval result = {};

  result.func = diff_layer.func * (1.0f - fr) + spec_layer / n_dot_o;
  ETX_VALIDATE(result.func);

  result.bsdf = diff_layer.func * (1.0f - fr) * n_dot_o + spec_layer;
  ETX_VALIDATE(result.bsdf);

  result.pdf = diff_layer.pdf * (1.0 - fr).monochromatic() + spec_pdf;
  ETX_VALIDATE(result.pdf);

  return result;
}

ETX_SHARED_INLINE float pdf(const BSDFData& data, const float3& w_o, const Material& mtl, Sampler& smp) {
  auto frame = data.get_normal_frame();

  float3 m = normalize(w_o - data.w_i);
  float m_dot_o = dot(m, w_o);
  float n_dot_o = dot(frame.nrm, w_o);

  if ((n_dot_o <= kEpsilon) || (m_dot_o <= kEpsilon)) {
    return 0.0f;
  }

  auto eta_e = evaluate_refractive_index(mtl.ext_ior, data.spectrum_sample);
  auto eta_i = evaluate_refractive_index(mtl.int_ior, data.spectrum_sample);
  auto thinfilm = evaluate_thinfilm(data.spectrum_sample, mtl.thinfilm, data.tex, smp);
  auto fr = fresnel::calculate(data.spectrum_sample, dot(data.w_i, m), eta_e, eta_i, thinfilm);

  float diff_pdf = kInvPi * n_dot_o;
  float spec_pdf = specular_pdf(data, w_o, mtl, smp);

  float result = diff_pdf * (1.0f - fr).monochromatic() + spec_pdf;
  ETX_VALIDATE(result);
  return result;
}

ETX_SHARED_INLINE bool is_delta(const Material& material, const float2& tex, Sampler& smp) {
  return false;
}

ETX_SHARED_INLINE SpectralResponse albedo(const BSDFData& data, const Material& mtl, Sampler& smp) {
  return apply_image(data.spectrum_sample, mtl.scattering, data.tex);
}

}  // namespace PlasticBSDF

}  // namespace etx
