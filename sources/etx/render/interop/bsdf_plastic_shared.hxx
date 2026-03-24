#pragma once

#include "bsdf_various_shared.hxx"

ETX_SHARED_INLINE BSDFEval bsdf_plastic_evaluate(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(float3, outgoing_direction), ETX_IN(Material, material),
  ETX_INOUT(Sampler, sampler));

ETX_SHARED_INLINE SpectralResponse bsdf_plastic_specular_func(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(float3, outgoing_direction),
  ETX_IN(Material, material), ETX_INOUT(Sampler, sampler)) {
  LocalFrame local_frame = bsdf_data_get_normal_frame(data, material);

  float3 w_i = local_frame_to_local(local_frame, -data.w_i);
  if (LocalFrame::cos_theta(w_i) <= kEpsilon) {
    return spectral_response_make(data.spectrum_sample, 0.0f);
  }

  float3 w_o = local_frame_to_local(local_frame, outgoing_direction);
  if (LocalFrame::cos_theta(w_o) <= kEpsilon) {
    return spectral_response_make(data.spectrum_sample, 0.0f);
  }

  float2 roughness = bsdf_resource_evaluate_roughness(context, material, data.tex);
  RefractiveIndexSample ext_ior = bsdf_resource_evaluate_refractive_index(context, material.ext_ior, data.spectrum_sample);
  RefractiveIndexSample int_ior = bsdf_resource_evaluate_refractive_index(context, material.int_ior, data.spectrum_sample);
  ThinfilmEval thinfilm = bsdf_resource_evaluate_thinfilm(context, data.spectrum_sample, material.thinfilm, data.tex, sampler);

  SpectralResponse value = bsdf_external_eval_dielectric(data.spectrum_sample, sampler, w_i, w_o, true, roughness, ext_ior, int_ior, thinfilm);
  SpectralResponse func = spectral_response_mul(spectral_response_mul(value, 2.0f), bsdf_resource_apply_image(context, data.spectrum_sample, material.reflectance, data.tex));
  return func;
}

ETX_SHARED_INLINE float bsdf_plastic_specular_pdf(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(float3, outgoing_direction), ETX_IN(Material, material),
  ETX_INOUT(Sampler, sampler)) {
  LocalFrame local_frame = bsdf_data_get_normal_frame(data, material);

  float3 w_i = local_frame_to_local(local_frame, -data.w_i);
  if (LocalFrame::cos_theta(w_i) <= kEpsilon) {
    return 0.0f;
  }

  float3 w_o = local_frame_to_local(local_frame, outgoing_direction);
  if (LocalFrame::cos_theta(w_o) <= kEpsilon) {
    return 0.0f;
  }

  RefractiveIndexSample ext_ior = bsdf_resource_evaluate_refractive_index(context, material.ext_ior, data.spectrum_sample);
  RefractiveIndexSample int_ior = bsdf_resource_evaluate_refractive_index(context, material.int_ior, data.spectrum_sample);
  float2 roughness = bsdf_resource_evaluate_roughness(context, material, data.tex);
  ThinfilmEval thinfilm = bsdf_resource_evaluate_thinfilm(context, data.spectrum_sample, material.thinfilm, data.tex, sampler);

  float3 wh = normalize(w_o + w_i);
  float dwh_dwo = 1.0f / (4.0f * dot(w_o, wh));

  BSDFExternalRayInfo ray = bsdf_external_ray_info_make(w_i, roughness);
  float d_ggx = bsdf_external_d_ggx(wh, roughness);
  float prob = max(0.0f, dot(wh, ray.w) * d_ggx / ((1.0f + ray.Lambda) * LocalFrame::cos_theta(ray.w)));
  SpectralResponse fr = bsdf_fresnel_calculate(data.spectrum_sample, dot(w_i, wh), ext_ior, int_ior, thinfilm);
  float fresnel_value = spectral_response_monochromatic(fr);
  prob *= fresnel_value;
  return abs(prob * dwh_dwo);
}

ETX_SHARED_INLINE BSDFSample bsdf_plastic_sample(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(Material, material), ETX_INOUT(Sampler, sampler)) {
  LocalFrame frame = bsdf_data_get_normal_frame(data, material);

  float2 roughness = bsdf_resource_evaluate_roughness(context, material, data.tex);
  float3 m = bsdf_normal_distribution_sample(frame, roughness, sampler, data.w_i);

  RefractiveIndexSample ext_ior = bsdf_resource_evaluate_refractive_index(context, material.ext_ior, data.spectrum_sample);
  RefractiveIndexSample int_ior = bsdf_resource_evaluate_refractive_index(context, material.int_ior, data.spectrum_sample);
  ThinfilmEval thinfilm = bsdf_resource_evaluate_thinfilm(context, data.spectrum_sample, material.thinfilm, data.tex, sampler);
  SpectralResponse fresnel = bsdf_fresnel_calculate(data.spectrum_sample, dot(data.w_i, m), ext_ior, int_ior, thinfilm);

  float3 w_i = local_frame_to_local(frame, -data.w_i);
  if (w_i.z <= kEpsilon) {
    return bsdf_sample_zero(data.spectrum_sample);
  }

  float3 outgoing_direction = float3(0.0f, 0.0f, 0.0f);
  bool sample_diffuse = bsdf_sampler_next(sampler) > spectral_response_monochromatic(fresnel);

  if (sample_diffuse == false) {
    outgoing_direction = reflect(data.w_i, m);
    sample_diffuse = dot(frame.nrm, outgoing_direction) <= kEpsilon;
  }

  if (sample_diffuse) {
    outgoing_direction = local_frame_from_local(frame, sample_cosine_distribution(bsdf_sampler_next_2d(sampler), 1.0f));
  }

  BSDFEval eval = bsdf_plastic_evaluate(context, data, outgoing_direction, material, sampler);
  BSDFSample result = ETX_ZERO(BSDFSample);
  result.w_o = outgoing_direction;
  result.weight = spectral_response_div(eval.bsdf, eval.pdf);
  result.properties = BSDFSample::Reflection | (sample_diffuse ? BSDFSample::Diffuse : 0u);
  result.medium_index = data.current_medium;
  result.pdf = eval.pdf;
  result.eta = 1.0f;
  return result;
}

ETX_SHARED_INLINE BSDFEval bsdf_plastic_evaluate(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(float3, outgoing_direction), ETX_IN(Material, material),
  ETX_INOUT(Sampler, sampler)) {
  LocalFrame frame = bsdf_data_get_normal_frame(data, material);
  float3 m = normalize(outgoing_direction - data.w_i);

  float n_dot_o = dot(frame.nrm, outgoing_direction);
  float m_dot_o = dot(m, outgoing_direction);
  if ((n_dot_o <= kEpsilon) || (m_dot_o <= kEpsilon)) {
    return bsdf_eval_zero(data.spectrum_sample);
  }

  RefractiveIndexSample eta_e = bsdf_resource_evaluate_refractive_index(context, material.ext_ior, data.spectrum_sample);
  RefractiveIndexSample eta_i = bsdf_resource_evaluate_refractive_index(context, material.int_ior, data.spectrum_sample);
  ThinfilmEval thinfilm = bsdf_resource_evaluate_thinfilm(context, data.spectrum_sample, material.thinfilm, data.tex, sampler);
  SpectralResponse fr = bsdf_fresnel_calculate(data.spectrum_sample, dot(data.w_i, m), eta_e, eta_i, thinfilm);

  float3 local_w_i = local_frame_to_local(frame, -data.w_i);
  float3 local_w_o = local_frame_to_local(frame, outgoing_direction);

  BSDFEval diff_layer = bsdf_diffuse_layer(context, data, local_w_i, local_w_o, material, sampler);
  SpectralResponse spec_layer = bsdf_plastic_specular_func(context, data, outgoing_direction, material, sampler);
  float spec_pdf = bsdf_plastic_specular_pdf(context, data, outgoing_direction, material, sampler);

  BSDFEval result = ETX_ZERO(BSDFEval);
  SpectralResponse one_minus_fr = spectral_response_sub(spectral_response_make(data.spectrum_sample, 1.0f), fr);
  result.func = spectral_response_add(spectral_response_mul(diff_layer.func, one_minus_fr), spectral_response_div(spec_layer, n_dot_o));
  result.bsdf = spectral_response_add(spectral_response_mul(spectral_response_mul(diff_layer.func, one_minus_fr), n_dot_o), spec_layer);
  result.pdf = diff_layer.pdf * spectral_response_monochromatic(one_minus_fr) + spec_pdf;
  result.eta = 1.0f;
  return result;
}

ETX_SHARED_INLINE float bsdf_plastic_pdf(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(float3, outgoing_direction), ETX_IN(Material, material),
  ETX_INOUT(Sampler, sampler)) {
  LocalFrame frame = bsdf_data_get_normal_frame(data);
  float3 m = normalize(outgoing_direction - data.w_i);
  float m_dot_o = dot(m, outgoing_direction);
  float n_dot_o = dot(frame.nrm, outgoing_direction);

  if ((n_dot_o <= kEpsilon) || (m_dot_o <= kEpsilon)) {
    return 0.0f;
  }

  RefractiveIndexSample eta_e = bsdf_resource_evaluate_refractive_index(context, material.ext_ior, data.spectrum_sample);
  RefractiveIndexSample eta_i = bsdf_resource_evaluate_refractive_index(context, material.int_ior, data.spectrum_sample);
  ThinfilmEval thinfilm = bsdf_resource_evaluate_thinfilm(context, data.spectrum_sample, material.thinfilm, data.tex, sampler);
  SpectralResponse fr = bsdf_fresnel_calculate(data.spectrum_sample, dot(data.w_i, m), eta_e, eta_i, thinfilm);
  SpectralResponse one_minus_fr = spectral_response_sub(spectral_response_make(data.spectrum_sample, 1.0f), fr);

  float diff_pdf = kInvPi * n_dot_o;
  float spec_pdf = bsdf_plastic_specular_pdf(context, data, outgoing_direction, material, sampler);
  float result = diff_pdf * spectral_response_monochromatic(one_minus_fr) + spec_pdf;
  return result;
}

ETX_SHARED_INLINE bool bsdf_plastic_is_delta(ETX_IN(Material, material), ETX_IN(float2, tex), ETX_INOUT(Sampler, sampler)) {
  (void)material;
  (void)tex;
  (void)sampler;
  return false;
}

ETX_SHARED_INLINE SpectralResponse bsdf_plastic_albedo(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(Material, material), ETX_INOUT(Sampler, sampler)) {
  (void)sampler;
  return bsdf_resource_apply_image(context, data.spectrum_sample, material.scattering, data.tex);
}
