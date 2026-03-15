namespace etx {

namespace ConductorBSDF {

struct ConductorMaterial {
  SpectralImage reflectance;
  SampledImage roughness;
  Thinfilm thinfilm;
  RefractiveIndex ext_ior;
  RefractiveIndex int_ior;
};

ETX_SHARED_INLINE float conductor_pdf(ETX_IN(float3, w_i), ETX_IN(float3, w_o), ETX_IN(float2, roughness)) {
  float3 half_vector = w_o + w_i;
  float half_vector_length_sq = dot(half_vector, half_vector);
  if (half_vector_length_sq <= kEpsilon) {
    return 0.0f;
  }

  half_vector *= 1.0f / sqrt(half_vector_length_sq);

  external::RayInfo ray = {w_i, roughness};
  float result = external::D_ggx(half_vector, roughness) / (1.0f + ray.Lambda) / (4.0f * w_i.z) + w_o.z;
  ETX_VALIDATE(result);
  return result;
}

ETX_SHARED_INLINE BSDFSample sample(const BSDFData& data, const Material& mtl, Sampler& smp) {
  auto frame = data.get_normal_frame(mtl);

  LocalFrame local_frame(frame);
  auto w_i = local_frame_to_local(local_frame, -data.w_i);
  if (w_i.z <= kEpsilon) {
    return {data.spectrum_sample};
  }
  auto ext_ior = evaluate_refractive_index(mtl.ext_ior, data.spectrum_sample);
  auto int_ior = evaluate_refractive_index(mtl.int_ior, data.spectrum_sample);
  auto thinfilm = evaluate_thinfilm(data.spectrum_sample, mtl.thinfilm, data.tex, smp);

  uint32_t delta_sample = is_delta(mtl, data.tex, smp) ? BSDFSample::Delta : 0u;

  BSDFSample result;
  result.properties = BSDFSample::Reflection | delta_sample;
  result.medium_index = data.current_medium;
  result.eta = 1.0f;

  result.weight = {data.spectrum_sample, 1.0f};

  // init
  float2 roughness = evaluate_roughness(mtl, data.tex);
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
  result.weight *= apply_image(data.spectrum_sample, mtl.reflectance, data.tex);
  ETX_VALIDATE(result.weight);

  result.pdf = conductor_pdf(w_i, result.w_o, roughness);

  result.w_o = normalize(local_frame_from_local(local_frame, result.w_o));
  return result;
}

ETX_SHARED_INLINE BSDFEval evaluate(const BSDFData& data, const float3& in_w_o, const Material& mtl, Sampler& smp) {
  auto frame = data.get_normal_frame(mtl);

  LocalFrame local_frame(frame);
  auto w_o = local_frame_to_local(local_frame, in_w_o);
  if (w_o.z <= kEpsilon) {
    return {data.spectrum_sample, 0.0f};
  }
  auto w_i = local_frame_to_local(local_frame, -data.w_i);
  if (w_i.z <= kEpsilon) {
    return {data.spectrum_sample, 0.0f};
  }

  float2 roughness = evaluate_roughness(mtl, data.tex);
  auto ext_ior = evaluate_refractive_index(mtl.ext_ior, data.spectrum_sample);
  auto int_ior = evaluate_refractive_index(mtl.int_ior, data.spectrum_sample);
  auto thinfilm = evaluate_thinfilm(data.spectrum_sample, mtl.thinfilm, data.tex, smp);

  auto value = external::eval_conductor(data.spectrum_sample, smp, w_i, w_o, roughness, ext_ior, int_ior, thinfilm);

  BSDFEval result = {};
  result.bsdf = value * apply_image(data.spectrum_sample, mtl.reflectance, data.tex);
  ETX_VALIDATE(result.bsdf);
  result.func = result.bsdf / w_o.z;
  ETX_VALIDATE(result.func);
  result.pdf = conductor_pdf(w_i, w_o, roughness);
  return result;
}

ETX_SHARED_INLINE float pdf(const BSDFData& data, const float3& in_w_o, const Material& mtl, Sampler& smp) {
  auto frame = data.get_normal_frame(mtl);

  LocalFrame local_frame(frame);
  auto w_o = local_frame_to_local(local_frame, in_w_o);
  if (w_o.z <= kEpsilon) {
    return 0.0f;
  }
  auto w_i = local_frame_to_local(local_frame, -data.w_i);
  if (w_i.z <= kEpsilon) {
    return 0.0f;
  }

  float2 roughness = evaluate_roughness(mtl, data.tex);
  return conductor_pdf(w_i, w_o, roughness);
}

ETX_SHARED_INLINE bool is_delta(const Material& mtl, const float2& tex, Sampler& smp) {
  float2 roughness = evaluate_roughness(mtl, tex);
  return max(roughness.x, roughness.y) <= kDeltaAlphaTreshold;
}

ETX_SHARED_INLINE SpectralResponse albedo(const BSDFData& data, const Material& mtl, Sampler& smp) {
  return apply_image(data.spectrum_sample, mtl.reflectance, data.tex);
}

}  // namespace ConductorBSDF
}  // namespace etx
