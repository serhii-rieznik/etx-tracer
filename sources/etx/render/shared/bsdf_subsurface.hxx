namespace etx {
namespace SubsurfaceBSDF {

ETX_GPU_CODE float fresnel_moment_1(float eta) {
  float eta2 = eta * eta;
  float eta3 = eta2 * eta;
  float eta4 = eta3 * eta;
  float eta5 = eta4 * eta;
  if (eta < 1.0f)
    return 0.45966f - 1.73965f * eta + 3.37668f * eta2 - 3.904945f * eta3 + 2.49277f * eta4 - 0.68441f * eta5;

  return -4.61686f + 11.1136f * eta - 10.4646f * eta2 + 5.11455f * eta3 - 1.27198f * eta4 + 0.12746f * eta5;
}

ETX_GPU_CODE float fresnel_moment_2(float eta) {
  float eta2 = eta * eta;
  float eta3 = eta2 * eta;
  float eta4 = eta3 * eta;
  float eta5 = eta4 * eta;
  if (eta < 1.0f) {
    return 0.27614f - 0.87350f * eta + 1.12077f * eta2 - 0.65095f * eta3 + 0.07883f * eta4 + 0.04860f * eta5;
  }

  float r_eta = 1.0f / eta, r_eta2 = r_eta * r_eta, r_eta3 = r_eta2 * r_eta;
  return -547.033f + 45.3087f * r_eta3 - 218.725f * r_eta2 + 458.843f * r_eta + 404.557f * eta - 189.519f * eta2 + 54.9327f * eta3 - 9.00603f * eta4 + 0.63942f * eta5;
}

ETX_GPU_CODE BSDFSample sample(const BSDFData& data, const Material& mtl, const Scene& scene, Sampler& smp) {
  auto eval_data = data;
  auto frame = data.get_normal_frame().frame;
  eval_data.w_o = sample_cosine_distribution(smp.next_2d(), frame.nrm, 0.0f);
  return {eval_data.w_o, evaluate(eval_data, mtl, scene, smp), BSDFSample::Diffuse};
}

ETX_GPU_CODE BSDFEval evaluate(const BSDFData& data, const Material& mtl, const Scene& scene, Sampler& smp) {
  auto eta_e = mtl.ext_ior(data.spectrum_sample);
  auto eta_i = mtl.int_ior(data.spectrum_sample);
  float eta = eta_i.eta.monochromatic() / eta_e.eta.monochromatic();
  float c = 1.0f - 2.0f * fresnel_moment_1(1.0f / eta);
  auto thinfilm = evaluate_thinfilm(data.spectrum_sample, mtl.thinfilm, data.tex, scene);
  float n_dot_o = fabsf(dot(data.nrm, data.w_o));
  BSDFEval eval;
  eval.func = (1.0f - fresnel::dielectric(data.spectrum_sample, data.w_i, data.w_o, eta_e, eta_i, thinfilm)) / (c * kPi);
  eval.bsdf = eval.func * n_dot_o;
  eval.pdf = n_dot_o / kPi;
  eval.weight = eval.bsdf / eval.pdf;
  return eval;
}

ETX_GPU_CODE float pdf(const BSDFData& data, const Material& mtl, const Scene& scene, Sampler& smp) {
  float n_dot_o = fabsf(dot(data.nrm, data.w_o));
  return n_dot_o / kPi;
}

ETX_GPU_CODE bool is_delta(const Material& material, const float2& tex, const Scene& scene, Sampler& smp) {
  return false;
}

}  // namespace SubsurfaceBSDF
}  // namespace etx
