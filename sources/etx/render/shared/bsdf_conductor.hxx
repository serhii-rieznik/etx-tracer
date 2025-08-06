namespace etx {

namespace ConductorBSDF {

ETX_GPU_CODE BSDFSample sample(const BSDFData& data, const Material& mtl, const Scene& scene, Sampler& smp) {
  auto frame = data.get_normal_frame();

  LocalFrame local_frame(frame);
  auto w_i = local_frame.to_local(-data.w_i);
  auto ext_ior = mtl.ext_ior(data.spectrum_sample);
  auto int_ior = mtl.int_ior(data.spectrum_sample);
  auto thinfilm = evaluate_thinfilm(data.spectrum_sample, mtl.thinfilm, data.tex, scene, smp);

  uint32_t delta_sample = is_delta(mtl, data.tex, scene) ? BSDFSample::Delta : 0u;

  BSDFSample result;
  result.properties = BSDFSample::Reflection | delta_sample;
  result.medium_index = data.current_medium;
  result.eta = 1.0f;

  result.weight = {data.spectrum_sample, 1.0f};

  // init
  float2 roughness = evaluate_roughness(mtl, data.tex, scene);
  external::RayInfo ray = {-w_i, roughness};
  ray.updateHeight(1.0f);

  uint32_t scattering_order = 0;
  while (true) {
    ray.updateHeight(external::sampleHeight(ray, smp.next()));
    if (ray.h == kMaxFloat)
      break;

    float2 slope_rnd = (scattering_order == 0) && smp.has_fixed() ? float2{smp.fixed_u, smp.fixed_v} : smp.next_2d();

    SpectralResponse weight = {data.spectrum_sample, 1.0f};
    ray.updateDirection(external::samplePhaseFunction_conductor(data.spectrum_sample, slope_rnd, -ray.w, roughness, ext_ior, int_ior, thinfilm, weight), roughness);
    ray.updateHeight(ray.h);

    result.weight *= weight;

    if ((scattering_order++ > external::kScatteringOrderMax) || (ray.h != ray.h) || (ray.w.x != ray.w.x)) {
      result.weight = {data.spectrum_sample, 0.0f};
      ray.w = float3{0, 0, 1};
      break;
    }
  }

  result.w_o = ray.w;

  result.weight *= apply_image(data.spectrum_sample, mtl.reflectance, data.tex, scene, nullptr);
  ETX_VALIDATE(result.weight);

  {
    external::RayInfo ray = {w_i, roughness};
    result.pdf = external::D_ggx(normalize(result.w_o + w_i), roughness) / (1.0f + ray.Lambda) / (4.0f * w_i.z) + result.w_o.z;
    ETX_VALIDATE(result.pdf);
  }

  result.w_o = normalize(local_frame.from_local(result.w_o));
  return result;
}

ETX_GPU_CODE BSDFEval evaluate(const BSDFData& data, const float3& in_w_o, const Material& mtl, const Scene& scene, Sampler& smp) {
  auto frame = data.get_normal_frame();

  LocalFrame local_frame(frame);
  auto w_o = local_frame.to_local(in_w_o);
  if (w_o.z <= kEpsilon) {
    return {data.spectrum_sample, 0.0f};
  }
  auto w_i = local_frame.to_local(-data.w_i);
  if (w_i.z <= kEpsilon) {
    return {data.spectrum_sample, 0.0f};
  }

  float2 roughness = evaluate_roughness(mtl, data.tex, scene);
  auto alpha_x = roughness.x;
  auto alpha_y = roughness.y;
  auto ext_ior = mtl.ext_ior(data.spectrum_sample);
  auto int_ior = mtl.int_ior(data.spectrum_sample);
  auto thinfilm = evaluate_thinfilm(data.spectrum_sample, mtl.thinfilm, data.tex, scene, smp);

  auto value = external::eval_conductor(data.spectrum_sample, smp, w_i, w_o, roughness, ext_ior, int_ior, thinfilm);

  BSDFEval result = {};
  result.bsdf = value * apply_image(data.spectrum_sample, mtl.reflectance, data.tex, scene, nullptr);
  ETX_VALIDATE(result.bsdf);
  result.func = result.bsdf / w_o.z;
  ETX_VALIDATE(result.func);
  {
    external::RayInfo ray = {w_i, roughness};
    result.pdf = external::D_ggx(normalize(w_o + w_i), roughness) / (1.0f + ray.Lambda) / (4.0f * w_i.z) + w_o.z;
    ETX_VALIDATE(result.pdf);
  }
  return result;
}

ETX_GPU_CODE float pdf(const BSDFData& data, const float3& in_w_o, const Material& mtl, const Scene& scene, Sampler& smp) {
  auto frame = data.get_normal_frame();

  LocalFrame local_frame(frame);
  auto w_o = local_frame.to_local(in_w_o);
  if (w_o.z <= kEpsilon) {
    return 0.0f;
  }
  auto w_i = local_frame.to_local(-data.w_i);
  if (w_i.z <= kEpsilon) {
    return 0.0f;
  }

  float2 roughness = evaluate_roughness(mtl, data.tex, scene);
  auto alpha_x = roughness.x;
  auto alpha_y = roughness.y;
  external::RayInfo ray = {w_i, roughness};
  float result = external::D_ggx(normalize(w_o + w_i), roughness) / (1.0f + ray.Lambda) / (4.0f * w_i.z) + w_o.z;
  ETX_VALIDATE(result);
  return result;
}

ETX_GPU_CODE bool is_delta(const Material& mtl, const float2& tex, const Scene& scene) {
  float2 roughness = evaluate_roughness(mtl, tex, scene);
  return max(roughness.x, roughness.y) <= kDeltaAlphaTreshold;
}

ETX_GPU_CODE SpectralResponse albedo(const BSDFData& data, const Material& mtl, const Scene& scene, Sampler& smp) {
  return apply_image(data.spectrum_sample, mtl.reflectance, data.tex, scene, nullptr);
}

}  // namespace ConductorBSDF
}  // namespace etx
