#pragma once

#include "bsdf_external_shared.hxx"
#include "bsdf_resource_shared.hxx"

ETX_SHARED_INLINE bool bsdf_conductor_is_delta(ETX_IN(Material, material), ETX_IN(float2, tex), ETX_INOUT(Sampler, sampler));
ETX_SHARED_INLINE bool bsdf_conductor_is_delta_with_context(ETX_IN(BSDFResourceContext, context), ETX_IN(Material, material), ETX_IN(float2, tex));

ETX_SHARED_INLINE SpectralResponse bsdf_conductor_delta_weight(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(Material, material),
  ETX_IN(RefractiveIndexSample, ext_ior), ETX_IN(RefractiveIndexSample, int_ior), ETX_IN(ThinfilmEval, thinfilm)) {
  LocalFrame frame = bsdf_data_get_normal_frame(data, material);
  float3 w_i = local_frame_to_local(frame, -data.w_i);
  if (w_i.z <= kEpsilon) {
    return spectral_response_make(data.spectrum_sample, 0.0f);
  }

  SpectralResponse fresnel = bsdf_fresnel_calculate(data.spectrum_sample, w_i.z, ext_ior, int_ior, thinfilm);
  SpectralResponse reflectance = bsdf_resource_apply_image(context, data.spectrum_sample, material.reflectance, data.tex);
  return spectral_response_mul(fresnel, reflectance);
}

ETX_SHARED_INLINE float3 bsdf_conductor_delta_reflect(ETX_IN(BSDFData, data), ETX_IN(Material, material)) {
  LocalFrame frame = bsdf_data_get_normal_frame(data, material);
  return normalize(reflect(data.w_i, frame.nrm));
}

ETX_SHARED_INLINE float bsdf_conductor_pdf_local(ETX_IN(float3, w_i), ETX_IN(float3, w_o), ETX_IN(float2, roughness)) {
  float3 half_vector = w_o + w_i;
  float half_vector_length_sq = dot(half_vector, half_vector);
  if (half_vector_length_sq <= kEpsilon) {
    return 0.0f;
  }

  half_vector *= 1.0f / sqrt(half_vector_length_sq);
  BSDFExternalRayInfo ray = bsdf_external_ray_info_make(w_i, roughness);
  float result = bsdf_external_d_ggx(half_vector, roughness) / (1.0f + ray.Lambda) / (4.0f * w_i.z) + w_o.z;
  return result;
}

ETX_SHARED_INLINE BSDFSample bsdf_conductor_sample(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(Material, material), ETX_INOUT(Sampler, sampler)) {
  LocalFrame frame = bsdf_data_get_normal_frame(data, material);
  float3 w_i = local_frame_to_local(frame, -data.w_i);
  if (w_i.z <= kEpsilon) {
    return bsdf_sample_zero(data.spectrum_sample);
  }

  RefractiveIndexSample ext_ior = bsdf_resource_evaluate_refractive_index(context, material.ext_ior, data.spectrum_sample);
  RefractiveIndexSample int_ior = bsdf_resource_evaluate_refractive_index(context, material.int_ior, data.spectrum_sample);
  ThinfilmEval thinfilm = bsdf_resource_evaluate_thinfilm(context, data.spectrum_sample, material.thinfilm, data.tex, sampler);
  bool is_delta = bsdf_conductor_is_delta_with_context(context, material, data.tex);

  BSDFSample result = ETX_ZERO(BSDFSample);
  result.properties = BSDFSample::Reflection | (is_delta ? BSDFSample::Delta : 0u);
  result.medium_index = data.current_medium;
  result.eta = 1.0f;
  if (is_delta) {
    result.w_o = bsdf_conductor_delta_reflect(data, material);
    result.weight = bsdf_conductor_delta_weight(context, data, material, ext_ior, int_ior, thinfilm);
    result.pdf = 1.0f;
    return result;
  }

  result.weight = spectral_response_make(data.spectrum_sample, 1.0f);

  float2 roughness = bsdf_resource_evaluate_roughness(context, material, data.tex);
  BSDFExternalRayInfo ray = bsdf_external_ray_info_make(-w_i, roughness);
  ray = bsdf_external_ray_info_update_height(ray, 1.0f);

  uint32_t scattering_order = 0u;
  while (true) {
    ray = bsdf_external_ray_info_update_height(ray, bsdf_external_sample_height(ray, bsdf_sampler_next(sampler)));
    if (ray.h == kMaxFloat) {
      break;
    }

    float2 slope_rnd = ((scattering_order == 0u) && bsdf_sampler_has_fixed(sampler)) ? float2(sampler.fixed_u, sampler.fixed_v) : bsdf_sampler_next_2d(sampler);
    SpectralResponse weight = spectral_response_make(data.spectrum_sample, 1.0f);
    ray = bsdf_external_ray_info_update_direction(ray,
      bsdf_external_sample_phase_function_conductor(data.spectrum_sample, slope_rnd, -ray.w, roughness, ext_ior, int_ior, thinfilm, weight), roughness);
    ray = bsdf_external_ray_info_update_height(ray, ray.h);
    result.weight = spectral_response_mul(result.weight, weight);

    scattering_order += 1u;
    if ((scattering_order > kBSDFExternalScatteringOrderMax) || ((ray.h != ray.h) == true) || ((ray.w.x != ray.w.x) == true)) {
      result.weight = spectral_response_make(data.spectrum_sample, 0.0f);
      ray.w = float3(0.0f, 0.0f, 1.0f);
      break;
    }
  }

  result.w_o = ray.w;
  result.weight = spectral_response_mul(result.weight, bsdf_resource_apply_image(context, data.spectrum_sample, material.reflectance, data.tex));
  result.pdf = bsdf_conductor_pdf_local(w_i, result.w_o, roughness);
  result.w_o = normalize(local_frame_from_local(frame, result.w_o));
  return result;
}

ETX_SHARED_INLINE BSDFEval bsdf_conductor_evaluate(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(float3, outgoing_direction), ETX_IN(Material, material),
  ETX_INOUT(Sampler, sampler)) {
  LocalFrame frame = bsdf_data_get_normal_frame(data, material);
  float3 w_o = local_frame_to_local(frame, outgoing_direction);
  if (w_o.z <= kEpsilon) {
    return bsdf_eval_zero(data.spectrum_sample);
  }

  float3 w_i = local_frame_to_local(frame, -data.w_i);
  if (w_i.z <= kEpsilon) {
    return bsdf_eval_zero(data.spectrum_sample);
  }

  float2 roughness = bsdf_resource_evaluate_roughness(context, material, data.tex);
  RefractiveIndexSample ext_ior = bsdf_resource_evaluate_refractive_index(context, material.ext_ior, data.spectrum_sample);
  RefractiveIndexSample int_ior = bsdf_resource_evaluate_refractive_index(context, material.int_ior, data.spectrum_sample);
  ThinfilmEval thinfilm = bsdf_resource_evaluate_thinfilm(context, data.spectrum_sample, material.thinfilm, data.tex, sampler);
  bool is_delta = bsdf_conductor_is_delta_with_context(context, material, data.tex);

  if (is_delta) {
    float3 ideal_w_o = bsdf_conductor_delta_reflect(data, material);
    float3 actual_w_o = normalize(outgoing_direction);
    if (direction_matches(ideal_w_o, actual_w_o, 1.0f) == false) {
      return bsdf_eval_zero(data.spectrum_sample);
    }

    BSDFEval result = ETX_ZERO(BSDFEval);
    result.bsdf = bsdf_conductor_delta_weight(context, data, material, ext_ior, int_ior, thinfilm);
    result.func = result.bsdf;
    result.pdf = 1.0f;
    result.eta = 1.0f;
    return result;
  }

  SpectralResponse value = bsdf_external_eval_conductor(data.spectrum_sample, sampler, w_i, w_o, roughness, ext_ior, int_ior, thinfilm);
  BSDFEval result = ETX_ZERO(BSDFEval);
  result.bsdf = spectral_response_mul(value, bsdf_resource_apply_image(context, data.spectrum_sample, material.reflectance, data.tex));
  if ((isfinite(result.bsdf.integrated.x) == false) || (isfinite(result.bsdf.integrated.y) == false) || (isfinite(result.bsdf.integrated.z) == false) ||
      (isfinite(result.bsdf.value) == false)) {
    return bsdf_eval_zero(data.spectrum_sample);
  }

  result.func = spectral_response_div(result.bsdf, w_o.z);
  if ((isfinite(result.func.integrated.x) == false) || (isfinite(result.func.integrated.y) == false) || (isfinite(result.func.integrated.z) == false) ||
      (isfinite(result.func.value) == false)) {
    return bsdf_eval_zero(data.spectrum_sample);
  }

  result.pdf = bsdf_conductor_pdf_local(w_i, w_o, roughness);
  if ((isfinite(result.pdf) == false) || (result.pdf <= 0.0f)) {
    return bsdf_eval_zero(data.spectrum_sample);
  }

  result.eta = 1.0f;
  return result;
}

ETX_SHARED_INLINE float bsdf_conductor_pdf(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(float3, outgoing_direction), ETX_IN(Material, material),
  ETX_INOUT(Sampler, sampler)) {
  (void)context;
  (void)sampler;

  LocalFrame frame = bsdf_data_get_normal_frame(data, material);
  float3 w_o = local_frame_to_local(frame, outgoing_direction);
  if (w_o.z <= kEpsilon) {
    return 0.0f;
  }

  float3 w_i = local_frame_to_local(frame, -data.w_i);
  if (w_i.z <= kEpsilon) {
    return 0.0f;
  }

  if (bsdf_conductor_is_delta_with_context(context, material, data.tex)) {
    float3 ideal_w_o = bsdf_conductor_delta_reflect(data, material);
    float3 actual_w_o = normalize(outgoing_direction);
    return direction_matches(ideal_w_o, actual_w_o, 1.0f) ? 1.0f : 0.0f;
  }

  float2 roughness = bsdf_resource_evaluate_roughness(context, material, data.tex);
  return bsdf_conductor_pdf_local(w_i, w_o, roughness);
}

ETX_SHARED_INLINE bool bsdf_conductor_is_delta(ETX_IN(Material, material), ETX_IN(float2, tex), ETX_INOUT(Sampler, sampler)) {
  (void)tex;
  (void)sampler;
  float2 roughness = float2(material.roughness.value.x, material.roughness.value.y);
  return max(roughness.x, roughness.y) <= kDeltaAlphaTreshold;
}

ETX_SHARED_INLINE bool bsdf_conductor_is_delta_with_context(ETX_IN(BSDFResourceContext, context), ETX_IN(Material, material), ETX_IN(float2, tex)) {
  float2 roughness = bsdf_resource_evaluate_roughness(context, material, tex);
  return max(roughness.x, roughness.y) <= kDeltaAlphaTreshold;
}

ETX_SHARED_INLINE SpectralResponse bsdf_conductor_albedo(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(Material, material), ETX_INOUT(Sampler, sampler)) {
  (void)sampler;
  return bsdf_resource_apply_image(context, data.spectrum_sample, material.reflectance, data.tex);
}
