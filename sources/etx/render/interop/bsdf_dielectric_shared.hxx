#pragma once

#include "bsdf_external_shared.hxx"
#include "bsdf_resource_shared.hxx"

ETX_SHARED_INLINE bool bsdf_dielectric_is_delta(ETX_IN(Material, material), ETX_IN(float2, tex), ETX_INOUT(Sampler, sampler));
ETX_SHARED_INLINE bool bsdf_dielectric_is_delta_with_context(ETX_IN(BSDFResourceContext, context), ETX_IN(Material, material), ETX_IN(float2, tex));

ETX_SHARED_INLINE bool bsdf_dielectric_has_thinfilm(ETX_IN(Material, material)) {
  return bsdf_resource_thinfilm_enabled(material.thinfilm);
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
  if (((thinfilm.weight <= 0.0f) || (thinfilm.thickness <= 0.0f)) && bsdf_dielectric_equal_eta(ext_ior, int_ior)) {
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

struct BSDFThinfilmInterface {
  LocalFrame frame ETX_INIT({});
  SpectralResponse reflection ETX_INIT({});
  SpectralResponse transmission ETX_INIT({});
  float reflection_probability ETX_INIT(0.0f);
  float transmission_probability ETX_INIT(0.0f);
  uint32_t transmission_medium ETX_INIT(kInvalidIndex);
  bool medium_changed ETX_INIT(false);
};

ETX_SHARED_INLINE BSDFThinfilmInterface bsdf_thinfilm_interface(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(Material, material),
  ETX_INOUT(Sampler, sampler)) {
  BSDFThinfilmInterface result = ETX_ZERO(BSDFThinfilmInterface);
  result.frame = bsdf_data_get_normal_frame(data);

  const float3 local_w_i = local_frame_to_local(result.frame, -data.w_i);
  if (local_w_i.z <= kEpsilon) {
    return result;
  }

  const bool entering = local_frame_entering_material(result.frame);
  const RefractiveIndexSample material_ext_ior = bsdf_resource_evaluate_refractive_index(context, material.ext_ior, data.spectrum_sample);
  const RefractiveIndexSample material_int_ior = bsdf_resource_evaluate_refractive_index(context, material.int_ior, data.spectrum_sample);
  const bool standalone_sheet = material.cls == MaterialClass::Thinfilm;
  RefractiveIndexSample phase_ext_ior = material_ext_ior;
  RefractiveIndexSample phase_int_ior = material_ext_ior;
  if (standalone_sheet == false) {
    phase_int_ior = material_int_ior;
    if (entering == false) {
      phase_ext_ior = material_int_ior;
      phase_int_ior = material_ext_ior;
    }
  }
  const ThinfilmEval thinfilm = bsdf_resource_evaluate_thinfilm(context, data.spectrum_sample, material.thinfilm, data.tex, sampler);
  const SpectralResponse fresnel = bsdf_fresnel_calculate(data.spectrum_sample, local_w_i.z, phase_ext_ior, phase_int_ior, thinfilm);
  const SpectralResponse one_minus_fresnel = spectral_response_sub(spectral_response_make(data.spectrum_sample, 1.0f), fresnel);
  result.reflection = spectral_response_mul(bsdf_resource_apply_image(context, data.spectrum_sample, material.reflectance, data.tex), fresnel);
  result.transmission = spectral_response_mul(bsdf_resource_apply_image(context, data.spectrum_sample, material.scattering, data.tex), one_minus_fresnel);
  result.reflection_probability = min(1.0f, max(0.0f, spectral_response_monochromatic(fresnel)));
  result.transmission_probability = max(0.0f, 1.0f - result.reflection_probability);
  result.transmission_medium = standalone_sheet ? data.current_medium : (entering ? material.int_medium : material.ext_medium);
  result.medium_changed = standalone_sheet == false;
  return result;
}

ETX_SHARED_INLINE BSDFSample bsdf_thinfilm_sample(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(Material, material), ETX_INOUT(Sampler, sampler)) {
  const BSDFThinfilmInterface interface_data = bsdf_thinfilm_interface(context, data, material, sampler);
  if ((interface_data.reflection_probability <= kEpsilon) && (interface_data.transmission_probability <= kEpsilon)) {
    return bsdf_sample_zero(data.spectrum_sample);
  }

  BSDFSample result = ETX_ZERO(BSDFSample);
  if (bsdf_sampler_next(sampler) <= interface_data.reflection_probability) {
    result.w_o = normalize(reflect(data.w_i, interface_data.frame.nrm));
    result.pdf = max(kEpsilon, interface_data.reflection_probability);
    result.weight = spectral_response_div(interface_data.reflection, result.pdf);
    result.properties = BSDFSample::Delta | BSDFSample::Reflection;
    result.medium_index = data.current_medium;
  } else {
    result.w_o = data.w_i;
    result.pdf = max(kEpsilon, interface_data.transmission_probability);
    result.weight = spectral_response_div(interface_data.transmission, result.pdf);
    result.properties = BSDFSample::Delta | BSDFSample::Transmission;
    if (interface_data.medium_changed) {
      result.properties |= BSDFSample::MediumChanged;
    }
    result.medium_index = interface_data.transmission_medium;
  }

  result.eta = 1.0f;
  return result;
}

ETX_SHARED_INLINE BSDFEval bsdf_thinfilm_evaluate(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(float3, outgoing_direction), ETX_IN(Material, material),
  ETX_INOUT(Sampler, sampler)) {
  const BSDFThinfilmInterface interface_data = bsdf_thinfilm_interface(context, data, material, sampler);
  const float3 actual_w_o = normalize(outgoing_direction);
  const float3 reflection_w_o = normalize(reflect(data.w_i, interface_data.frame.nrm));
  BSDFEval result = bsdf_eval_zero(data.spectrum_sample);
  if (direction_matches(reflection_w_o, actual_w_o, 1.0f)) {
    result.bsdf = interface_data.reflection;
    result.func = result.bsdf;
    result.pdf = interface_data.reflection_probability;
    result.properties = BSDFSample::Delta | BSDFSample::Reflection;
    result.medium_index = data.current_medium;
    return result;
  }

  if (direction_matches(data.w_i, actual_w_o, 1.0f)) {
    result.bsdf = interface_data.transmission;
    result.func = result.bsdf;
    result.pdf = interface_data.transmission_probability;
    result.properties = BSDFSample::Delta | BSDFSample::Transmission;
    if (interface_data.medium_changed) {
      result.properties |= BSDFSample::MediumChanged;
    }
    result.medium_index = interface_data.transmission_medium;
    return result;
  }

  return result;
}

ETX_SHARED_INLINE float bsdf_thinfilm_pdf(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(float3, outgoing_direction), ETX_IN(Material, material),
  ETX_INOUT(Sampler, sampler)) {
  const BSDFThinfilmInterface interface_data = bsdf_thinfilm_interface(context, data, material, sampler);
  const float3 actual_w_o = normalize(outgoing_direction);
  const float3 reflection_w_o = normalize(reflect(data.w_i, interface_data.frame.nrm));
  if (direction_matches(reflection_w_o, actual_w_o, 1.0f)) {
    return interface_data.reflection_probability;
  }
  if (direction_matches(data.w_i, actual_w_o, 1.0f)) {
    return interface_data.transmission_probability;
  }
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
