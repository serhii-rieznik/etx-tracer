#pragma once

#include "bsdf_external_shared.hxx"
#include "bsdf_resource_shared.hxx"

ETX_SHARED_INLINE bool bsdf_dielectric_is_delta(ETX_IN(Material, material), ETX_IN(float2, tex), ETX_INOUT(Sampler, sampler));
ETX_SHARED_INLINE bool bsdf_dielectric_is_delta_with_context(ETX_IN(BSDFResourceContext, context), ETX_IN(Material, material), ETX_IN(float2, tex));

ETX_SHARED_INLINE bool bsdf_plastic_has_subsurface(ETX_IN(Material, material)) {
  return (material.cls == MaterialClass::Plastic) && (material.subsurface_cls != SubsurfaceMaterial::Disabled);
}

ETX_SHARED_INLINE SpectralResponse bsdf_dielectric_boundary_texture(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(Material, material), bool reflection) {
  if ((reflection == false) && bsdf_plastic_has_subsurface(material)) {
    return bsdf_resource_subsurface_boundary_color(context, data.spectrum_sample, material, data.tex);
  }
  SpectralImage texture = material.reflectance;
  if (reflection == false) {
    texture = material.scattering;
  }
  return bsdf_resource_apply_image(context, data.spectrum_sample, texture, data.tex);
}

ETX_SHARED_INLINE bool bsdf_dielectric_has_thinfilm(ETX_IN(Material, material)) {
#if ETX_ENABLE_COHERENT_COATING
  return (bsdf_resource_thinfilm_enabled(material.thinfilm)) || (material.coherent_coating.max_thickness_nm > 0.0f);
#else
  return bsdf_resource_thinfilm_enabled(material.thinfilm);
#endif
}

ETX_SHARED_INLINE bool bsdf_dielectric_equal_eta(ETX_IN(RefractiveIndexSample, ext_ior), ETX_IN(RefractiveIndexSample, int_ior)) {
  const float eta_ext = max(kEpsilon, spectral_response_monochromatic(ext_ior.eta));
  const float eta_int = max(kEpsilon, spectral_response_monochromatic(int_ior.eta));
  // Equality of the common direction index does not imply equality of RGB Fresnel weights.
  return eta_ext == eta_int;
}

ETX_SHARED_INLINE bool bsdf_dielectric_identity_with_context(ETX_IN(BSDFResourceContext, context), ETX_IN(Material, material), ETX_IN(float2, tex)) {
  // Without a wavelength, shared spectrum resources prove identity for every query.
  return (material.ext_ior.eta_index == material.int_ior.eta_index) && (material.ext_ior.k_index == material.int_ior.k_index) &&
         (bsdf_resource_interface_has_layers(context, material, tex) == false);
}

ETX_SHARED_INLINE BSDFSample bsdf_dielectric_equal_eta_sample(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(Material, material)) {
  const LocalFrame frame = bsdf_data_get_normal_frame(data);
  BSDFSample result = ETX_ZERO(BSDFSample);
  result.w_o = data.w_i;
  result.pdf = 1.0f;
  result.weight = bsdf_dielectric_boundary_texture(context, data, material, false);
  result.properties = BSDFSample::Delta | BSDFSample::Transmission | BSDFSample::MediumChanged;
  result.medium_index = local_frame_entering_material(frame) ? material.int_medium : material.ext_medium;
  result.eta = 1.0f;
  return result;
}

ETX_SHARED_INLINE float2 bsdf_dielectric_atom_probabilities(ETX_IN(SpectralResponse, reflection), ETX_IN(SpectralResponse, transmission), ETX_IN(RefractiveIndexSample, exterior),
  ETX_IN(RefractiveIndexSample, interior), bool transmission_possible) {
  const bool dispersive = (spectral_response_is_spectral(exterior.eta) == false) &&
                          ((exterior.eta.integrated.x != exterior.eta.integrated.y) || (exterior.eta.integrated.y != exterior.eta.integrated.z) ||
                            (interior.eta.integrated.x != interior.eta.integrated.y) || (interior.eta.integrated.y != interior.eta.integrated.z));
  if (dispersive) {
    return transmission_possible ? float2(0.5f, 0.5f) : float2(1.0f, 0.0f);
  }
  const float cell = 1.0f / 8388608.0f;
  const float reflected = spectral_response_maximum(reflection) > 0.0f ? max(cell, spectral_response_monochromatic(reflection)) : 0.0f;
  const float transmitted = transmission_possible && (spectral_response_maximum(transmission) > 0.0f) ? max(cell, spectral_response_monochromatic(transmission)) : 0.0f;
  const float3 proposal = bsdf_external_atom_proposal(reflected, transmitted, 0.0f);
  return float2(proposal.x, proposal.y);
}

ETX_SHARED_INLINE BSDFSample bsdf_dielectric_delta_sample_event(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(Material, material), uint32_t required_event,
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
  const ThinfilmEval optics = bsdf_resource_evaluate_interface_optics(context, data.spectrum_sample, material, data.tex, sampler);
  if ((bsdf_resource_interface_has_layers(optics) == false) && bsdf_fresnel_equal_indices(ext_ior, int_ior)) {
    if (required_event == BSDFSample::Reflection) {
      return bsdf_sample_zero(data.spectrum_sample);
    }
    return bsdf_dielectric_equal_eta_sample(context, data, material);
  }

  const bool outside = w_i_local.z > 0.0f;
  const ThinfilmEval thinfilm = bsdf_fresnel_orient_coherent_interface(optics, outside);
  const float direction_scale = outside ? 1.0f : -1.0f;
  const float3 w_i = direction_scale * w_i_local;
  if (outside == false) {
    const RefractiveIndexSample original_ext_ior = ext_ior;
    ext_ior = int_ior;
    int_ior = original_ext_ior;
  }

  const SpectralResponse fresnel = bsdf_fresnel_calculate(data.spectrum_sample, w_i.z, ext_ior, int_ior, thinfilm);
  const SpectralResponse transmission = bsdf_fresnel_transmission_calculate(data.spectrum_sample, w_i.z, ext_ior, int_ior, thinfilm);
  const bool matching_media = bsdf_dielectric_equal_eta(ext_ior, int_ior);
  const float eta = matching_media ? 1.0f : bsdf_dielectric_directional_eta(ext_ior, int_ior);
  const bool total_internal_reflection = (matching_media == false) && (bsdf_external_transmission_possible(w_i.z, ext_ior, int_ior) == false);
  const float2 proposal = bsdf_dielectric_atom_probabilities(fresnel, transmission, ext_ior, int_ior, total_internal_reflection == false);
  const float selector = required_event == 0u ? bsdf_sampler_next(sampler) : 0.0f;
  if ((required_event == 0u) && (selector >= (proposal.x + proposal.y))) {
    return bsdf_sample_zero(data.spectrum_sample);
  }
  const bool reflection = required_event == 0u ? (selector < proposal.x) : (required_event == BSDFSample::Reflection);
  const float probability = required_event == 0u ? (reflection ? proposal.x : proposal.y) : 1.0f;
  if (probability <= 0.0f) {
    return bsdf_sample_zero(data.spectrum_sample);
  }

  BSDFSample result = ETX_ZERO(BSDFSample);
  if (reflection) {
    const float3 local_w_o = direction_scale * float3(-w_i.x, -w_i.y, w_i.z);
    const float pdf = probability;
    result.w_o = normalize(local_frame_from_local(frame, local_w_o));
    result.pdf = pdf;
    result.weight = spectral_response_mul(bsdf_resource_apply_image(context, data.spectrum_sample, material.reflectance, data.tex), spectral_response_div(fresnel, pdf));
    result.properties = BSDFSample::Delta | BSDFSample::Reflection;
    result.medium_index = data.current_medium;
    result.eta = 1.0f;
    return result;
  }
  if (total_internal_reflection) {
    return bsdf_sample_zero(data.spectrum_sample);
  }

  const float3 local_w_o = matching_media ? -w_i_local : direction_scale * normalize(bsdf_external_refract(w_i, float3(0.0f, 0.0f, 1.0f), ext_ior, int_ior));
  const float pdf = probability;
  const float eta_factor = data.path_source == PathSource::Light ? 1.0f / (eta * eta) : 1.0f;
  result.w_o = matching_media ? data.w_i : normalize(local_frame_from_local(frame, local_w_o));
  result.pdf = pdf;
  result.weight =
    spectral_response_mul(bsdf_dielectric_boundary_texture(context, data, material, false), spectral_response_div(spectral_response_mul(transmission, eta_factor), pdf));
  result.properties = BSDFSample::Delta | BSDFSample::Transmission | BSDFSample::MediumChanged;
  if (matching_media == false) {
    result.properties |= BSDFSample::WavelengthDependentDirection;
  }
  result.medium_index = outside ? material.int_medium : material.ext_medium;
  result.eta = eta;
  return result;
}

ETX_SHARED_INLINE BSDFSample bsdf_dielectric_delta_sample(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(Material, material), ETX_INOUT(Sampler, sampler)) {
  return bsdf_dielectric_delta_sample_event(context, data, material, 0u, sampler);
}

struct BSDFThinfilmInterface {
  LocalFrame frame ETX_INIT({});
  SpectralResponse reflection ETX_INIT({});
  SpectralResponse transmission ETX_INIT({});
  float reflection_probability ETX_INIT(0.0f);
  float transmission_probability ETX_INIT(0.0f);
  float3 transmission_direction ETX_INIT({});
  float eta ETX_INIT(1.0f);
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
  RefractiveIndexSample phase_ext_ior = bsdf_resource_evaluate_refractive_index(context, material.ext_ior, data.spectrum_sample);
  RefractiveIndexSample phase_int_ior = bsdf_resource_evaluate_refractive_index(context, material.int_ior, data.spectrum_sample);
  if (entering == false) {
    const RefractiveIndexSample original = phase_ext_ior;
    phase_ext_ior = phase_int_ior;
    phase_int_ior = original;
  }
  const ThinfilmEval thinfilm =
    bsdf_fresnel_orient_coherent_interface(bsdf_resource_evaluate_interface_optics(context, data.spectrum_sample, material, data.tex, sampler), entering);
  const SpectralResponse fresnel = bsdf_fresnel_calculate(data.spectrum_sample, local_w_i.z, phase_ext_ior, phase_int_ior, thinfilm);
  const SpectralResponse transmission = bsdf_fresnel_transmission_calculate(data.spectrum_sample, local_w_i.z, phase_ext_ior, phase_int_ior, thinfilm);
  result.reflection = spectral_response_mul(bsdf_resource_apply_image(context, data.spectrum_sample, material.reflectance, data.tex), fresnel);
  result.eta = bsdf_dielectric_directional_eta(phase_ext_ior, phase_int_ior);
  const float eta_factor = data.path_source == PathSource::Light ? 1.0f / (result.eta * result.eta) : 1.0f;
  const bool transmission_possible = bsdf_external_transmission_possible(local_w_i.z, phase_ext_ior, phase_int_ior);
  if (transmission_possible) {
    result.transmission_direction = normalize(local_frame_from_local(result.frame, bsdf_external_refract(local_w_i, float3(0.0f, 0.0f, 1.0f), phase_ext_ior, phase_int_ior)));
    result.transmission = spectral_response_mul(bsdf_dielectric_boundary_texture(context, data, material, false), spectral_response_mul(transmission, eta_factor));
  } else {
    result.transmission = spectral_response_zero(data.spectrum_sample);
  }
  const float2 proposal = bsdf_dielectric_atom_probabilities(fresnel, transmission, phase_ext_ior, phase_int_ior, transmission_possible);
  result.reflection_probability = proposal.x;
  result.transmission_probability = proposal.y;
  result.transmission_medium = entering ? material.int_medium : material.ext_medium;
  result.medium_changed = true;
  return result;
}

ETX_SHARED_INLINE BSDFSample bsdf_thinfilm_sample(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(Material, material), ETX_INOUT(Sampler, sampler)) {
  const BSDFThinfilmInterface interface_data = bsdf_thinfilm_interface(context, data, material, sampler);
  if ((interface_data.reflection_probability <= 0.0f) && (interface_data.transmission_probability <= 0.0f)) {
    return bsdf_sample_zero(data.spectrum_sample);
  }

  BSDFSample result = ETX_ZERO(BSDFSample);
  const float selector = bsdf_sampler_next(sampler);
  if (selector < interface_data.reflection_probability) {
    result.w_o = normalize(reflect(data.w_i, interface_data.frame.nrm));
    result.pdf = interface_data.reflection_probability;
    result.weight = spectral_response_div(interface_data.reflection, result.pdf);
    result.properties = BSDFSample::Delta | BSDFSample::Reflection;
    result.medium_index = data.current_medium;
  } else {
    if (selector >= (interface_data.reflection_probability + interface_data.transmission_probability)) {
      return bsdf_sample_zero(data.spectrum_sample);
    }
    result.w_o = interface_data.transmission_direction;
    result.pdf = interface_data.transmission_probability;
    result.weight = spectral_response_div(interface_data.transmission, result.pdf);
    result.properties = BSDFSample::Delta | BSDFSample::Transmission | BSDFSample::WavelengthDependentDirection;
    if (interface_data.medium_changed) {
      result.properties |= BSDFSample::MediumChanged;
    }
    result.medium_index = interface_data.transmission_medium;
  }

  result.eta = (result.properties & BSDFSample::Transmission) != 0u ? interface_data.eta : 1.0f;
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

  if (direction_matches(interface_data.transmission_direction, actual_w_o, 1.0f)) {
    result.bsdf = interface_data.transmission;
    result.func = result.bsdf;
    result.pdf = interface_data.transmission_probability;
    result.eta = interface_data.eta;
    result.properties = BSDFSample::Delta | BSDFSample::Transmission | BSDFSample::WavelengthDependentDirection;
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
  if (direction_matches(interface_data.transmission_direction, actual_w_o, 1.0f)) {
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
  return bsdf_dielectric_boundary_texture(context, data, material, false);
}

ETX_SHARED_INLINE bool bsdf_dielectric_is_delta(ETX_IN(Material, material), ETX_IN(float2, tex), ETX_INOUT(Sampler, sampler)) {
  (void)tex;
  (void)sampler;
  const float2 roughness = ((material.cls == MaterialClass::Plastic) && (material.layer_base_class == (MaterialClass::Diffuse + 1u)))
                             ? float2(material.coating.roughness.value.x, material.coating.roughness.value.y)
                             : float2(material.roughness.value.x, material.roughness.value.y);
  return max(roughness.x, roughness.y) <= kDeltaAlphaTreshold;
}

ETX_SHARED_INLINE bool bsdf_dielectric_is_delta_with_context(ETX_IN(BSDFResourceContext, context), ETX_IN(Material, material), ETX_IN(float2, tex)) {
  const float2 roughness = bsdf_resource_evaluate_roughness(context, material, tex);
  return ((max(roughness.x, roughness.y) <= kDeltaAlphaTreshold) || bsdf_dielectric_identity_with_context(context, material, tex));
}

ETX_SHARED_INLINE SpectralResponse bsdf_dielectric_albedo(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(Material, material), ETX_INOUT(Sampler, sampler)) {
  (void)sampler;
  return bsdf_dielectric_boundary_texture(context, data, material, false);
}
