#pragma once

#include "bsdf_external_shared.hxx"
#include "bsdf_resource_shared.hxx"

ETX_SHARED_INLINE bool bsdf_dielectric_is_delta(ETX_IN(Material, material), ETX_IN(float2, tex), ETX_INOUT(Sampler, sampler));
ETX_SHARED_INLINE bool bsdf_dielectric_is_delta_with_context(ETX_IN(BSDFResourceContext, context), ETX_IN(Material, material), ETX_IN(float2, tex));

ETX_SHARED_INLINE bool bsdf_dielectric_has_thinfilm(ETX_IN(Material, material)) {
  return (material.thinfilm.min_thickness * material.thinfilm.max_thickness) > 0.0f;
}

ETX_SHARED_INLINE bool bsdf_dielectric_equal_eta(ETX_IN(RefractiveIndexSample, ext_ior), ETX_IN(RefractiveIndexSample, int_ior)) {
  const float eta_ext = max(kEpsilon, spectral_response_monochromatic(ext_ior.eta));
  const float eta_int = max(kEpsilon, spectral_response_monochromatic(int_ior.eta));
  const float eta_scale = max(eta_ext, eta_int);
  const float tolerance = max(kEpsilon, 16.0f * kEpsilon * eta_scale);
  return abs(eta_ext - eta_int) <= tolerance;
}

ETX_SHARED_INLINE bool bsdf_dielectric_equal_eta_with_context(ETX_IN(BSDFResourceContext, context), ETX_IN(Material, material), ETX_IN(SpectralQuery, spect)) {
  if (bsdf_dielectric_has_thinfilm(material)) {
    return false;
  }

  const RefractiveIndexSample ext_ior = bsdf_resource_evaluate_refractive_index(context, material.ext_ior, spect);
  const RefractiveIndexSample int_ior = bsdf_resource_evaluate_refractive_index(context, material.int_ior, spect);
  return bsdf_dielectric_equal_eta(ext_ior, int_ior);
}

ETX_SHARED_INLINE BSDFSample bsdf_dielectric_equal_eta_sample(ETX_IN(BSDFData, data), ETX_IN(Material, material)) {
  const LocalFrame frame = bsdf_data_get_normal_frame(data);
  BSDFSample result = ETX_ZERO(BSDFSample);
  result.w_o = data.w_i;
  result.pdf = 1.0f;
  result.weight = spectral_response_make(data.spectrum_sample, 1.0f);
  result.properties = BSDFSample::Delta | BSDFSample::Transmission | BSDFSample::MediumChanged;
  result.medium_index = local_frame_entering_material(frame) ? material.int_medium : material.ext_medium;
  result.eta = 1.0f;
  return result;
}

ETX_SHARED_INLINE BSDFSample bsdf_dielectric_delta_sample(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(Material, material),
  ETX_INOUT(Sampler, sampler)) {
  LocalFrame frame = ETX_ZERO(LocalFrame);
  frame.tan = data.tan;
  frame.btn = data.btn;
  frame.nrm = data.nrm;
  const float3 w_i_local = local_frame_to_local(frame, -data.w_i);
  if (abs(w_i_local.z) <= kEpsilon) {
    return bsdf_sample_zero(data.spectrum_sample);
  }

  RefractiveIndexSample ext_ior = bsdf_resource_evaluate_refractive_index(context, material.ext_ior, data.spectrum_sample);
  RefractiveIndexSample int_ior = bsdf_resource_evaluate_refractive_index(context, material.int_ior, data.spectrum_sample);
  const ThinfilmEval thinfilm = bsdf_resource_evaluate_thinfilm(context, data.spectrum_sample, material.thinfilm, data.tex, sampler);
  if ((thinfilm.thickness <= 0.0f) && bsdf_dielectric_equal_eta(ext_ior, int_ior)) {
    return bsdf_dielectric_equal_eta_sample(data, material);
  }

  const bool outside = w_i_local.z > 0.0f;
  const float direction_scale = outside ? 1.0f : -1.0f;
  const float3 w_i = direction_scale * w_i_local;
  if (outside == false) {
    const RefractiveIndexSample original_ext_ior = ext_ior;
    ext_ior = int_ior;
    int_ior = original_ext_ior;
  }

  const SpectralResponse fresnel = bsdf_fresnel_calculate(data.spectrum_sample, w_i.z, ext_ior, int_ior, thinfilm);
  const float fresnel_probability = min(1.0f, max(0.0f, spectral_response_monochromatic(fresnel)));
  const float eta = max(kEpsilon, spectral_response_monochromatic(spectral_response_div(int_ior.eta, ext_ior.eta)));
  const float sin_theta_i2 = max(0.0f, 1.0f - w_i.z * w_i.z);
  const float sin_theta_t2 = sin_theta_i2 / max(kEpsilon, eta * eta);
  const bool total_internal_reflection = sin_theta_t2 >= 1.0f;
  const float selector = bsdf_sampler_next(sampler);

  BSDFSample result = ETX_ZERO(BSDFSample);
  if ((selector <= fresnel_probability) || total_internal_reflection) {
    const float3 local_w_o = direction_scale * float3(-w_i.x, -w_i.y, w_i.z);
    const float pdf = total_internal_reflection ? 1.0f : max(kEpsilon, fresnel_probability);
    result.w_o = normalize(local_frame_from_local(frame, local_w_o));
    result.pdf = pdf;
    result.weight = spectral_response_mul(bsdf_resource_apply_image(context, data.spectrum_sample, material.reflectance, data.tex), spectral_response_div(fresnel, pdf));
    result.properties = BSDFSample::Delta | BSDFSample::Reflection;
    result.medium_index = data.current_medium;
    result.eta = 1.0f;
    return result;
  }

  const float3 local_w_o = direction_scale * normalize(bsdf_external_refract(w_i, float3(0.0f, 0.0f, 1.0f), eta));
  const float pdf = max(kEpsilon, 1.0f - fresnel_probability);
  const SpectralResponse one_minus_fresnel = spectral_response_sub(spectral_response_make(data.spectrum_sample, 1.0f), fresnel);
  const float eta_factor = eta * eta;
  result.w_o = normalize(local_frame_from_local(frame, local_w_o));
  result.pdf = pdf;
  result.weight = spectral_response_mul(bsdf_resource_apply_image(context, data.spectrum_sample, material.scattering, data.tex),
    spectral_response_div(spectral_response_mul(one_minus_fresnel, eta_factor), pdf));
  result.properties = BSDFSample::Delta | BSDFSample::Transmission | BSDFSample::MediumChanged;
  result.medium_index = outside ? material.int_medium : material.ext_medium;
  result.eta = eta;
  return result;
}

ETX_SHARED_INLINE BSDFSample bsdf_thinfilm_sample(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(Material, material), ETX_INOUT(Sampler, sampler)) {
  LocalFrame frame = bsdf_data_get_normal_frame(data);
  RefractiveIndexSample ext_ior = bsdf_resource_evaluate_refractive_index(context, material.ext_ior, data.spectrum_sample);
  RefractiveIndexSample int_ior = bsdf_resource_evaluate_refractive_index(context, material.int_ior, data.spectrum_sample);
  ThinfilmEval thinfilm = bsdf_resource_evaluate_thinfilm(context, data.spectrum_sample, material.thinfilm, data.tex, sampler);
  SpectralResponse fr = bsdf_fresnel_calculate(data.spectrum_sample, dot(data.w_i, data.nrm), ext_ior, int_ior, thinfilm);
  float f = spectral_response_monochromatic(fr);

  BSDFSample result = ETX_ZERO(BSDFSample);
  if (bsdf_sampler_next(sampler) <= f) {
    result.w_o = normalize(reflect(data.w_i, frame.nrm));
    result.pdf = f;
    result.weight = spectral_response_mul(bsdf_resource_apply_image(context, data.spectrum_sample, material.reflectance, data.tex), spectral_response_div(fr, f));
    result.properties = BSDFSample::Delta | BSDFSample::Reflection;
    result.medium_index = data.current_medium;
  } else {
    result.w_o = data.w_i;
    result.pdf = 1.0f - f;
    SpectralResponse one_minus_fr = spectral_response_sub(spectral_response_make(data.spectrum_sample, 1.0f), fr);
    result.weight = spectral_response_mul(bsdf_resource_apply_image(context, data.spectrum_sample, material.scattering, data.tex), spectral_response_div(one_minus_fr, 1.0f - f));
    result.properties = BSDFSample::Delta | BSDFSample::Transmission | BSDFSample::MediumChanged;
    result.medium_index = local_frame_entering_material(frame) ? material.int_medium : material.ext_medium;
  }

  result.eta = 1.0f;
  return result;
}

ETX_SHARED_INLINE BSDFEval bsdf_thinfilm_evaluate(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(float3, outgoing_direction), ETX_IN(Material, material),
  ETX_INOUT(Sampler, sampler)) {
  (void)context;
  (void)outgoing_direction;
  (void)material;
  (void)sampler;
  return bsdf_eval_zero(data.spectrum_sample);
}

ETX_SHARED_INLINE float bsdf_thinfilm_pdf(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(float3, outgoing_direction), ETX_IN(Material, material),
  ETX_INOUT(Sampler, sampler)) {
  (void)context;
  (void)data;
  (void)outgoing_direction;
  (void)material;
  (void)sampler;
  return 0.0f;
}

ETX_SHARED_INLINE bool bsdf_thinfilm_is_delta(ETX_IN(Material, material), ETX_IN(float2, tex), ETX_INOUT(Sampler, sampler)) {
  (void)material;
  (void)tex;
  (void)sampler;
  return true;
}

ETX_SHARED_INLINE SpectralResponse bsdf_thinfilm_albedo(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(Material, material), ETX_INOUT(Sampler, sampler)) {
  (void)sampler;
  return bsdf_resource_apply_image(context, data.spectrum_sample, material.scattering, data.tex);
}

ETX_SHARED_INLINE bool bsdf_dielectric_is_delta(ETX_IN(Material, material), ETX_IN(float2, tex), ETX_INOUT(Sampler, sampler)) {
  (void)tex;
  (void)sampler;
  float2 roughness = float2(material.roughness.value.x, material.roughness.value.y);
  return max(roughness.x, roughness.y) <= kDeltaAlphaTreshold;
}

ETX_SHARED_INLINE bool bsdf_dielectric_is_delta_with_context(ETX_IN(BSDFResourceContext, context), ETX_IN(Material, material), ETX_IN(float2, tex)) {
  float2 roughness = bsdf_resource_evaluate_roughness(context, material, tex);
  SpectralQuery spect = ETX_ZERO(SpectralQuery);
  return ((max(roughness.x, roughness.y) <= kDeltaAlphaTreshold) || bsdf_dielectric_equal_eta_with_context(context, material, spect));
}

ETX_SHARED_INLINE SpectralResponse bsdf_dielectric_albedo(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(Material, material), ETX_INOUT(Sampler, sampler)) {
  (void)sampler;
  return bsdf_resource_apply_image(context, data.spectrum_sample, material.scattering, data.tex);
}
