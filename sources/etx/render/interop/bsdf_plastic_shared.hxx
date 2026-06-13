#pragma once

#include "bsdf_various_shared.hxx"
#include "bsdf_energy_compensated_shared.hxx"

struct BSDFPlasticCoatingReflectionProposal {
  float probability ETX_INIT(0.0f);
  float base_probability ETX_INIT(0.0f);
  float compensation_probability ETX_INIT(0.0f);
};

struct BSDFPlasticExternalAlbedos {
  SpectralResponse reflection ETX_INIT({});
  SpectralResponse transmission ETX_INIT({});
};

struct BSDFPlasticDeltaThinfilmTerms {
  LocalFrame frame ETX_INIT({});
  float3 local_w_i ETX_INIT({});
  SpectralResponse reflection ETX_INIT({});
  SpectralResponse diffuse_scale ETX_INIT({});
  float specular_probability ETX_INIT(0.0f);
  bool valid ETX_INIT(false);
};

ETX_SHARED_NOINLINE float bsdf_plastic_pdf(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(float3, outgoing_direction), ETX_IN(Material, material),
  ETX_INOUT(Sampler, sampler));

ETX_SHARED_INLINE BSDFEval bsdf_plastic_delta_thinfilm_evaluate(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(float3, outgoing_direction),
  ETX_IN(Material, material), ETX_INOUT(Sampler, sampler));

ETX_SHARED_INLINE BSDFSample bsdf_plastic_delta_thinfilm_sample(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(Material, material),
  ETX_INOUT(Sampler, sampler));

ETX_SHARED_INLINE float bsdf_plastic_delta_thinfilm_pdf(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(float3, outgoing_direction),
  ETX_IN(Material, material), ETX_INOUT(Sampler, sampler));

ETX_SHARED_INLINE bool bsdf_plastic_supported(ETX_IN(BSDFResourceContext, context), ETX_IN(Material, material), ETX_IN(float2, uv)) {
  if (bsdf_energy_compensated_material_supported(context, material, uv) == false) {
    return false;
  }
  return bsdf_energy_compensated_material_interface_valid(context, material, MaterialClass::Dielectric);
}

ETX_SHARED_INLINE LocalFrame bsdf_plastic_coating_frame(ETX_IN(BSDFData, data), ETX_IN(Material, material)) {
  return bsdf_data_get_normal_frame(data, material);
}

ETX_SHARED_INLINE SpectralResponse bsdf_plastic_one(ETX_IN(SpectralQuery, spect)) {
  return spectral_response_make(spect, 1.0f);
}

ETX_SHARED_INLINE bool bsdf_plastic_delta_thinfilm_supported(ETX_IN(BSDFResourceContext, context), ETX_IN(Material, material), ETX_IN(float2, uv)) {
  if (bsdf_resource_thinfilm_enabled(material.thinfilm) == false) {
    return false;
  }

  const float2 roughness = bsdf_resource_evaluate_roughness(context, material, uv);
  return max(roughness.x, roughness.y) <= kDeltaAlphaTreshold;
}

ETX_SHARED_NOINLINE SpectralResponse bsdf_plastic_dielectric_total_branch_albedo(ETX_IN(BSDFResourceContext, context), ETX_IN(SpectralQuery, spect),
  ETX_IN(Material, material), float mu, float alpha, bool incident_outside, bool outgoing_outside, float thinfilm_lut_value) {
  const BSDFEnergyCompensatedDielectricBranchPair pair =
    bsdf_energy_compensated_dielectric_branch_pair_value(context, spect, material, mu, alpha, incident_outside, thinfilm_lut_value);
  const SpectralResponse single = bsdf_energy_compensated_dielectric_branch_pair_selected_albedo(spect, pair, outgoing_outside);
  const SpectralResponse e_i = bsdf_energy_compensated_dielectric_branch_pair_albedo(spect, pair);
  const SpectralResponse d_i = spectral_response_max(spectral_response_sub(bsdf_plastic_one(spect), e_i), 0.0f);
  const SpectralResponse d_o = bsdf_energy_compensated_dielectric_average_residual(context, spect, material, alpha, outgoing_outside, thinfilm_lut_value);
  const SpectralResponse coefficient =
    bsdf_energy_compensated_dielectric_branch_coefficient(context, spect, material, alpha, incident_outside, outgoing_outside, thinfilm_lut_value);
  return spectral_response_min(spectral_response_add(single, spectral_response_mul(spectral_response_mul(coefficient, d_i), d_o)), 1.0f);
}

ETX_SHARED_NOINLINE SpectralResponse bsdf_plastic_dielectric_total_branch_average_albedo(ETX_IN(BSDFResourceContext, context), ETX_IN(SpectralQuery, spect),
  ETX_IN(Material, material), float alpha, bool incident_outside, bool outgoing_outside, float thinfilm_lut_value) {
  const SpectralResponse single =
    bsdf_energy_compensated_dielectric_branch_average_albedo(context, spect, material, alpha, incident_outside, outgoing_outside, thinfilm_lut_value);
  const SpectralResponse d_i = bsdf_energy_compensated_dielectric_average_residual(context, spect, material, alpha, incident_outside, thinfilm_lut_value);
  const SpectralResponse d_o = bsdf_energy_compensated_dielectric_average_residual(context, spect, material, alpha, outgoing_outside, thinfilm_lut_value);
  const SpectralResponse coefficient =
    bsdf_energy_compensated_dielectric_branch_coefficient(context, spect, material, alpha, incident_outside, outgoing_outside, thinfilm_lut_value);
  return spectral_response_min(spectral_response_add(single, spectral_response_mul(spectral_response_mul(coefficient, d_i), d_o)), 1.0f);
}

ETX_SHARED_INLINE SpectralResponse bsdf_plastic_external_reflection_albedo(ETX_IN(BSDFResourceContext, context), ETX_IN(SpectralQuery, spect), ETX_IN(Material, material),
  float mu, float alpha, float thinfilm_lut_value) {
  return bsdf_plastic_dielectric_total_branch_albedo(context, spect, material, mu, alpha, true, true, thinfilm_lut_value);
}

ETX_SHARED_INLINE SpectralResponse bsdf_plastic_external_transmission_albedo(ETX_IN(BSDFResourceContext, context), ETX_IN(SpectralQuery, spect), ETX_IN(Material, material),
  float mu, float alpha, float thinfilm_lut_value) {
  return bsdf_plastic_dielectric_total_branch_albedo(context, spect, material, mu, alpha, true, false, thinfilm_lut_value);
}

ETX_SHARED_NOINLINE BSDFPlasticExternalAlbedos bsdf_plastic_external_albedos(ETX_IN(BSDFResourceContext, context), ETX_IN(SpectralQuery, spect), ETX_IN(Material, material),
  float mu, float alpha, float thinfilm_lut_value) {
  BSDFPlasticExternalAlbedos result = ETX_ZERO(BSDFPlasticExternalAlbedos);
  const BSDFEnergyCompensatedDielectricBranchPair pair =
    bsdf_energy_compensated_dielectric_branch_pair_value(context, spect, material, mu, alpha, true, thinfilm_lut_value);
  const SpectralResponse e_i = bsdf_energy_compensated_dielectric_branch_pair_albedo(spect, pair);
  const SpectralResponse d_i = spectral_response_max(spectral_response_sub(bsdf_plastic_one(spect), e_i), 0.0f);
  const SpectralResponse d_o_reflection = bsdf_energy_compensated_dielectric_average_residual(context, spect, material, alpha, true, thinfilm_lut_value);
  const SpectralResponse d_o_transmission = bsdf_energy_compensated_dielectric_average_residual(context, spect, material, alpha, false, thinfilm_lut_value);
  const SpectralResponse reflection_coefficient =
    bsdf_energy_compensated_dielectric_branch_coefficient(context, spect, material, alpha, true, true, thinfilm_lut_value);
  const SpectralResponse transmission_coefficient =
    bsdf_energy_compensated_dielectric_branch_coefficient(context, spect, material, alpha, true, false, thinfilm_lut_value);
  const SpectralResponse single_reflection = bsdf_energy_compensated_dielectric_branch_pair_selected_albedo(spect, pair, true);
  const SpectralResponse single_transmission = bsdf_energy_compensated_dielectric_branch_pair_selected_albedo(spect, pair, false);
  result.reflection =
    spectral_response_min(spectral_response_add(single_reflection, spectral_response_mul(spectral_response_mul(reflection_coefficient, d_i), d_o_reflection)), 1.0f);
  result.transmission =
    spectral_response_min(spectral_response_add(single_transmission, spectral_response_mul(spectral_response_mul(transmission_coefficient, d_i), d_o_transmission)), 1.0f);
  return result;
}

ETX_SHARED_INLINE SpectralResponse bsdf_plastic_internal_transmission_albedo(ETX_IN(BSDFResourceContext, context), ETX_IN(SpectralQuery, spect), ETX_IN(Material, material),
  float mu, float alpha, float thinfilm_lut_value) {
  return bsdf_plastic_dielectric_total_branch_albedo(context, spect, material, mu, alpha, false, true, thinfilm_lut_value);
}

ETX_SHARED_INLINE SpectralResponse bsdf_plastic_internal_average_transmission_albedo(ETX_IN(BSDFResourceContext, context), ETX_IN(SpectralQuery, spect),
  ETX_IN(Material, material), float alpha, float thinfilm_lut_value) {
  return bsdf_plastic_dielectric_total_branch_average_albedo(context, spect, material, alpha, false, true, thinfilm_lut_value);
}

ETX_SHARED_INLINE SpectralResponse bsdf_plastic_internal_average_reflection_albedo(ETX_IN(BSDFResourceContext, context), ETX_IN(SpectralQuery, spect),
  ETX_IN(Material, material), float alpha, float thinfilm_lut_value) {
  return bsdf_plastic_dielectric_total_branch_average_albedo(context, spect, material, alpha, false, false, thinfilm_lut_value);
}

ETX_SHARED_NOINLINE SpectralResponse bsdf_plastic_internal_bounce_denominator(ETX_IN(BSDFResourceContext, context), ETX_IN(SpectralQuery, spect), ETX_IN(Material, material),
  ETX_IN(SpectralResponse, substrate), float alpha, float thinfilm_lut_value) {
  const SpectralResponse one = bsdf_plastic_one(spect);
  const SpectralResponse r_internal_average = bsdf_plastic_internal_average_reflection_albedo(context, spect, material, alpha, thinfilm_lut_value);
  return spectral_response_max(spectral_response_sub(one, spectral_response_mul(substrate, r_internal_average)), kEpsilon);
}

ETX_SHARED_INLINE BSDFPlasticDeltaThinfilmTerms bsdf_plastic_delta_thinfilm_terms(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(Material, material),
  ETX_INOUT(Sampler, sampler)) {
  BSDFPlasticDeltaThinfilmTerms result = ETX_ZERO(BSDFPlasticDeltaThinfilmTerms);
  result.frame = bsdf_plastic_coating_frame(data, material);
  result.local_w_i = local_frame_to_local(result.frame, -data.w_i);
  if (result.local_w_i.z <= kEpsilon) {
    return result;
  }

  const RefractiveIndexSample ext_ior = bsdf_resource_evaluate_refractive_index(context, material.ext_ior, data.spectrum_sample);
  const RefractiveIndexSample int_ior = bsdf_resource_evaluate_refractive_index(context, material.int_ior, data.spectrum_sample);
  const ThinfilmEval thinfilm = bsdf_resource_evaluate_thinfilm(context, data.spectrum_sample, material.thinfilm, data.tex, sampler);
  const SpectralResponse fresnel_i = bsdf_fresnel_calculate(data.spectrum_sample, result.local_w_i.z, ext_ior, int_ior, thinfilm);
  const SpectralResponse one = bsdf_plastic_one(data.spectrum_sample);
  const SpectralResponse substrate = bsdf_resource_apply_image(context, data.spectrum_sample, material.scattering, data.tex);
  const SpectralResponse reflectance = bsdf_resource_apply_image(context, data.spectrum_sample, material.reflectance, data.tex);
  const SpectralResponse internal_reflection_average = bsdf_fresnel_average(data.spectrum_sample, int_ior, ext_ior, thinfilm);
  const SpectralResponse internal_transmission_average = spectral_response_sub(one, internal_reflection_average);
  const SpectralResponse denominator = spectral_response_max(spectral_response_sub(one, spectral_response_mul(substrate, internal_reflection_average)), kEpsilon);
  const SpectralResponse t_i = spectral_response_sub(one, fresnel_i);
  result.reflection = spectral_response_mul(reflectance, fresnel_i);
  result.diffuse_scale = spectral_response_div(spectral_response_mul(t_i, internal_transmission_average), denominator);

  const float reflection_energy = max(0.0f, spectral_response_monochromatic(result.reflection));
  const float substrate_energy = max(0.0f, spectral_response_monochromatic(spectral_response_mul(substrate, result.diffuse_scale)));
  const float total_energy = reflection_energy + substrate_energy;
  result.specular_probability = (total_energy > kEpsilon) ? bsdf_energy_compensated_saturate(reflection_energy / total_energy) : 0.0f;
  result.valid = true;
  return result;
}

ETX_SHARED_INLINE BSDFEval bsdf_plastic_delta_thinfilm_evaluate(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(float3, outgoing_direction),
  ETX_IN(Material, material), ETX_INOUT(Sampler, sampler)) {
  const BSDFPlasticDeltaThinfilmTerms terms = bsdf_plastic_delta_thinfilm_terms(context, data, material, sampler);
  if (terms.valid == false) {
    return bsdf_eval_zero(data.spectrum_sample);
  }

  const float3 local_w_o = local_frame_to_local(terms.frame, outgoing_direction);
  if (local_w_o.z <= kEpsilon) {
    return bsdf_eval_zero(data.spectrum_sample);
  }

  const float3 ideal_reflection_w_o = normalize(reflect(data.w_i, terms.frame.nrm));
  const float3 actual_w_o = normalize(outgoing_direction);
  BSDFEval result = ETX_ZERO(BSDFEval);
  if (direction_matches(ideal_reflection_w_o, actual_w_o, 1.0f)) {
    result.bsdf = terms.reflection;
    result.func = result.bsdf;
    result.pdf = terms.specular_probability;
    result.eta = 1.0f;
    result.properties = BSDFSample::Delta | BSDFSample::Reflection;
    result.medium_index = data.current_medium;
    return result;
  }

  const BSDFEval substrate_eval = bsdf_diffuse_layer(context, data, terms.local_w_i, local_w_o, material, sampler);
  if (bsdf_eval_valid(substrate_eval) == false) {
    return bsdf_eval_zero(data.spectrum_sample);
  }

  const float diffuse_probability = max(0.0f, 1.0f - terms.specular_probability);
  result.bsdf = spectral_response_mul(substrate_eval.bsdf, terms.diffuse_scale);
  result.func = spectral_response_div(result.bsdf, local_w_o.z);
  result.pdf = diffuse_probability * substrate_eval.pdf;
  result.eta = 1.0f;
  result.properties = BSDFSample::Diffuse | BSDFSample::Reflection;
  result.medium_index = data.current_medium;
  return result;
}

ETX_SHARED_INLINE BSDFSample bsdf_plastic_delta_thinfilm_sample(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(Material, material),
  ETX_INOUT(Sampler, sampler)) {
  const BSDFPlasticDeltaThinfilmTerms terms = bsdf_plastic_delta_thinfilm_terms(context, data, material, sampler);
  if (terms.valid == false) {
    return bsdf_sample_zero(data.spectrum_sample);
  }

  const bool has_fixed = bsdf_sampler_has_fixed(sampler);
  float selector = sampler.fixed_w;
  float2 rnd = float2(sampler.fixed_u, sampler.fixed_v);
  if (has_fixed == false) {
    selector = bsdf_sampler_next(sampler);
    rnd = bsdf_sampler_next_2d(sampler);
  }
  BSDFSample result = ETX_ZERO(BSDFSample);
  if (selector < terms.specular_probability) {
    result.w_o = normalize(reflect(data.w_i, terms.frame.nrm));
    result.pdf = max(kEpsilon, terms.specular_probability);
    result.weight = spectral_response_div(terms.reflection, result.pdf);
    result.properties = BSDFSample::Delta | BSDFSample::Reflection;
    result.medium_index = data.current_medium;
    result.eta = 1.0f;
    return result;
  }

  const float3 local_w_o = sample_cosine_distribution(rnd, 1.0f);
  const float3 world_w_o = normalize(local_frame_from_local(terms.frame, local_w_o));
  const BSDFEval eval = bsdf_plastic_delta_thinfilm_evaluate(context, data, world_w_o, material, sampler);
  if (bsdf_eval_valid(eval) == false) {
    return bsdf_sample_zero(data.spectrum_sample);
  }

  result.w_o = world_w_o;
  result.pdf = eval.pdf;
  result.weight = spectral_response_div(eval.bsdf, eval.pdf);
  result.properties = BSDFSample::Diffuse | BSDFSample::Reflection;
  result.medium_index = data.current_medium;
  result.eta = 1.0f;
  return result;
}

ETX_SHARED_INLINE float bsdf_plastic_delta_thinfilm_pdf(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(float3, outgoing_direction),
  ETX_IN(Material, material), ETX_INOUT(Sampler, sampler)) {
  const BSDFPlasticDeltaThinfilmTerms terms = bsdf_plastic_delta_thinfilm_terms(context, data, material, sampler);
  if (terms.valid == false) {
    return 0.0f;
  }

  const float3 ideal_reflection_w_o = normalize(reflect(data.w_i, terms.frame.nrm));
  const float3 actual_w_o = normalize(outgoing_direction);
  if (direction_matches(ideal_reflection_w_o, actual_w_o, 1.0f)) {
    return terms.specular_probability;
  }

  const float3 local_w_o = local_frame_to_local(terms.frame, outgoing_direction);
  if (local_w_o.z <= kEpsilon) {
    return 0.0f;
  }
  return max(0.0f, 1.0f - terms.specular_probability) * local_w_o.z * kInvPi;
}

ETX_SHARED_NOINLINE SpectralResponse bsdf_plastic_coated_diffuse_func(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(Material, material),
  ETX_IN(float3, local_w_i), ETX_IN(float3, local_w_o), float alpha, float thinfilm_lut_value, ETX_INOUT(Sampler, sampler)) {
  const BSDFEval substrate_eval = bsdf_diffuse_layer(context, data, local_w_i, local_w_o, material, sampler);
  if (bsdf_eval_valid(substrate_eval) == false) {
    return spectral_response_zero(data.spectrum_sample);
  }

  const SpectralResponse substrate = bsdf_resource_apply_image(context, data.spectrum_sample, material.scattering, data.tex);
  const SpectralResponse t_i = bsdf_plastic_external_transmission_albedo(context, data.spectrum_sample, material, local_w_i.z, alpha, thinfilm_lut_value);
  const SpectralResponse t_o = bsdf_plastic_internal_average_transmission_albedo(context, data.spectrum_sample, material, alpha, thinfilm_lut_value);
  const SpectralResponse denominator = bsdf_plastic_internal_bounce_denominator(context, data.spectrum_sample, material, substrate, alpha, thinfilm_lut_value);
  const SpectralResponse scale = spectral_response_div(spectral_response_mul(t_i, t_o), denominator);
  return spectral_response_mul(substrate_eval.func, scale);
}

ETX_SHARED_NOINLINE float bsdf_plastic_specular_sample_probability(ETX_IN(BSDFResourceContext, context), ETX_IN(SpectralQuery, spect), ETX_IN(Material, material),
  ETX_IN(SpectralResponse, substrate), float mu_i, float alpha, float thinfilm_lut_value) {
  const BSDFPlasticExternalAlbedos external_albedos = bsdf_plastic_external_albedos(context, spect, material, mu_i, alpha, thinfilm_lut_value);
  const SpectralResponse transmission_average = bsdf_plastic_internal_average_transmission_albedo(context, spect, material, alpha, thinfilm_lut_value);
  const SpectralResponse denominator = bsdf_plastic_internal_bounce_denominator(context, spect, material, substrate, alpha, thinfilm_lut_value);
  const SpectralResponse diffuse_energy = spectral_response_div(spectral_response_mul(spectral_response_mul(external_albedos.transmission, substrate), transmission_average), denominator);
  const float reflection_energy = max(0.0f, spectral_response_monochromatic(external_albedos.reflection));
  const float substrate_energy = max(0.0f, spectral_response_monochromatic(diffuse_energy));
  const float total_energy = reflection_energy + substrate_energy;
  if (total_energy <= kEpsilon) {
    return 0.0f;
  }
  return bsdf_energy_compensated_saturate(reflection_energy / total_energy);
}

ETX_SHARED_NOINLINE BSDFPlasticCoatingReflectionProposal bsdf_plastic_coating_reflection_proposal(ETX_IN(BSDFResourceContext, context), ETX_IN(SpectralQuery, spect),
  ETX_IN(Material, material), float mu_i, float alpha, float thinfilm_lut_value) {
  BSDFPlasticCoatingReflectionProposal result = ETX_ZERO(BSDFPlasticCoatingReflectionProposal);
  const BSDFEnergyCompensatedDielectricBranchPair pair =
    bsdf_energy_compensated_dielectric_branch_pair_value(context, spect, material, mu_i, alpha, true, thinfilm_lut_value);
  const float visible_probability = bsdf_energy_compensated_dielectric_branch_pair_visible_probability(pair);
  const float reflection_visible_probability = bsdf_energy_compensated_dielectric_branch_pair_selected_visible_probability(pair, true);
  const SpectralResponse e_i = bsdf_energy_compensated_dielectric_branch_pair_albedo(spect, pair);
  const float base_probability = bsdf_energy_compensated_saturate(spectral_response_monochromatic(e_i));
  const float base_reflection_probability = (visible_probability > kEpsilon) ? bsdf_energy_compensated_saturate(reflection_visible_probability / visible_probability) : 0.0f;
  const float compensation_branch_probability =
    bsdf_energy_compensated_dielectric_compensation_branch_probability(context, spect, material, alpha, true, true, thinfilm_lut_value);
  const float base_mass = base_probability * base_reflection_probability;
  const float compensation_mass = max(0.0f, 1.0f - base_probability) * compensation_branch_probability;
  const float total_mass = base_mass + compensation_mass;
  if (total_mass <= kEpsilon) {
    return result;
  }

  result.probability = bsdf_energy_compensated_saturate(total_mass);
  result.base_probability = bsdf_energy_compensated_saturate(base_mass / total_mass);
  result.compensation_probability = max(0.0f, 1.0f - result.base_probability);
  return result;
}

ETX_SHARED_INLINE BSDFPlasticCoatingReflectionProposal bsdf_plastic_coating_reflection_proposal(ETX_IN(BSDFResourceContext, context), ETX_IN(SpectralQuery, spect),
  ETX_IN(Material, material), float mu_i, float alpha, float thinfilm_lut_value, ETX_IN(RefractiveIndexSample, ext_ior), ETX_IN(RefractiveIndexSample, int_ior)) {
  (void)ext_ior;
  (void)int_ior;
  return bsdf_plastic_coating_reflection_proposal(context, spect, material, mu_i, alpha, thinfilm_lut_value);
}

ETX_SHARED_INLINE BSDFPlasticCoatingReflectionProposal bsdf_plastic_coating_reflection_proposal(ETX_IN(BSDFResourceContext, context), ETX_IN(SpectralQuery, spect),
  ETX_IN(Material, material), ETX_IN(float3, local_w_i), float alpha, float thinfilm_lut_value) {
  return bsdf_plastic_coating_reflection_proposal(context, spect, material, local_w_i.z, alpha, thinfilm_lut_value);
}

ETX_SHARED_NOINLINE BSDFEval bsdf_plastic_evaluate(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(float3, outgoing_direction), ETX_IN(Material, material),
  ETX_INOUT(Sampler, sampler)) {
  if (bsdf_plastic_delta_thinfilm_supported(context, material, data.tex)) {
    return bsdf_plastic_delta_thinfilm_evaluate(context, data, outgoing_direction, material, sampler);
  }

  if (bsdf_plastic_supported(context, material, data.tex) == false) {
    return bsdf_diffuse_evaluate(context, data, outgoing_direction, material, sampler);
  }

  const LocalFrame frame = bsdf_plastic_coating_frame(data, material);
  const float3 local_w_i = local_frame_to_local(frame, -data.w_i);
  const float3 local_w_o = local_frame_to_local(frame, outgoing_direction);
  if ((local_w_i.z <= kEpsilon) || (local_w_o.z <= kEpsilon)) {
    return bsdf_eval_zero(data.spectrum_sample);
  }

  const float alpha = bsdf_energy_compensated_scalar_roughness(context, material, data.tex);
  const RefractiveIndexSample ext_ior = bsdf_resource_evaluate_refractive_index(context, material.ext_ior, data.spectrum_sample);
  const RefractiveIndexSample int_ior = bsdf_resource_evaluate_refractive_index(context, material.int_ior, data.spectrum_sample);
  const ThinfilmEval thinfilm = bsdf_resource_evaluate_thinfilm(context, data.spectrum_sample, material.thinfilm, data.tex, sampler);
  const float thinfilm_lut_value = bsdf_energy_compensated_thinfilm_lut_value(material, thinfilm);
  const SpectralResponse reflectance = bsdf_resource_apply_image(context, data.spectrum_sample, material.reflectance, data.tex);
  const SpectralResponse coating_bsdf =
    bsdf_energy_compensated_dielectric_bsdf_local(context, data.spectrum_sample, material, local_w_i, local_w_o, alpha, ext_ior, int_ior, thinfilm, reflectance);
  const SpectralResponse diffuse_func = bsdf_plastic_coated_diffuse_func(context, data, material, local_w_i, local_w_o, alpha, thinfilm_lut_value, sampler);
  const SpectralResponse diffuse_bsdf = spectral_response_mul(diffuse_func, local_w_o.z);

  BSDFEval result = ETX_ZERO(BSDFEval);
  result.bsdf = spectral_response_add(coating_bsdf, diffuse_bsdf);
  if (bsdf_energy_compensated_spectral_response_finite(result.bsdf) == false) {
    return bsdf_eval_zero(data.spectrum_sample);
  }
  result.func = spectral_response_div(result.bsdf, local_w_o.z);
  if (bsdf_energy_compensated_spectral_response_finite(result.func) == false) {
    return bsdf_eval_zero(data.spectrum_sample);
  }
  result.pdf = bsdf_plastic_pdf(context, data, outgoing_direction, material, sampler);
  result.eta = 1.0f;
  result.properties = BSDFSample::Reflection;
  result.medium_index = data.current_medium;
  return result;
}

ETX_SHARED_NOINLINE BSDFSample bsdf_plastic_sample(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(Material, material), ETX_INOUT(Sampler, sampler)) {
  if (bsdf_plastic_delta_thinfilm_supported(context, material, data.tex)) {
    return bsdf_plastic_delta_thinfilm_sample(context, data, material, sampler);
  }

  if (bsdf_plastic_supported(context, material, data.tex) == false) {
    return bsdf_diffuse_sample(context, data, material, sampler);
  }

  const LocalFrame frame = bsdf_plastic_coating_frame(data, material);
  const float3 local_w_i = local_frame_to_local(frame, -data.w_i);
  if (local_w_i.z <= kEpsilon) {
    return bsdf_sample_zero(data.spectrum_sample);
  }

  const float alpha = bsdf_energy_compensated_scalar_roughness(context, material, data.tex);
  const SpectralResponse substrate = bsdf_resource_apply_image(context, data.spectrum_sample, material.scattering, data.tex);
  const ThinfilmEval thinfilm = bsdf_resource_evaluate_thinfilm(context, data.spectrum_sample, material.thinfilm, data.tex, sampler);
  const float thinfilm_lut_value = bsdf_energy_compensated_thinfilm_lut_value(material, thinfilm);
  const float specular_probability = bsdf_plastic_specular_sample_probability(context, data.spectrum_sample, material, substrate, local_w_i.z, alpha, thinfilm_lut_value);
  const BSDFPlasticCoatingReflectionProposal coating_proposal =
    bsdf_plastic_coating_reflection_proposal(context, data.spectrum_sample, material, local_w_i, alpha, thinfilm_lut_value);
  const bool has_fixed = bsdf_sampler_has_fixed(sampler);
  float selector = sampler.fixed_w;
  float2 rnd = float2(sampler.fixed_u, sampler.fixed_v);
  if (has_fixed == false) {
    selector = bsdf_sampler_next(sampler);
    rnd = bsdf_sampler_next_2d(sampler);
  }

  float3 local_w_o = float3(0.0f, 0.0f, 0.0f);
  bool sampled_diffuse = (selector >= specular_probability) || (coating_proposal.probability <= kEpsilon);
  if (sampled_diffuse == false) {
    const float coating_selector = bsdf_sampler_next(sampler);
    if (coating_selector < coating_proposal.base_probability) {
      const RefractiveIndexSample ext_ior = bsdf_resource_evaluate_refractive_index(context, material.ext_ior, data.spectrum_sample);
      const RefractiveIndexSample int_ior = bsdf_resource_evaluate_refractive_index(context, material.int_ior, data.spectrum_sample);
      bool candidate_valid = false;
      bool first_attempt = true;
      while (candidate_valid == false) {
        float2 attempt_rnd = rnd;
        if ((first_attempt == false) || (has_fixed == false)) {
          attempt_rnd = bsdf_sampler_next_2d(sampler);
        }
        const float3 m = bsdf_energy_compensated_sample_vndf_local(local_w_i, alpha, attempt_rnd);
        const float i_dot_m = dot(local_w_i, m);
        if ((m.z > kEpsilon) && (i_dot_m > kEpsilon)) {
          const SpectralResponse fresnel = bsdf_fresnel_calculate(data.spectrum_sample, i_dot_m, ext_ior, int_ior, thinfilm);
          const float fresnel_probability = spectral_response_monochromatic(fresnel);
          local_w_o = -local_w_i + 2.0f * m * i_dot_m;
          candidate_valid = (local_w_o.z > kEpsilon) && (bsdf_sampler_next(sampler) < fresnel_probability);
        }
        first_attempt = false;
      }
    } else {
      local_w_o = sample_cosine_distribution(rnd, 1.0f);
    }
  } else {
    local_w_o = sample_cosine_distribution(rnd, 1.0f);
  }

  const float3 world_w_o = normalize(local_frame_from_local(frame, local_w_o));
  const BSDFEval eval = bsdf_plastic_evaluate(context, data, world_w_o, material, sampler);
  if (bsdf_eval_valid(eval) == false) {
    return bsdf_sample_zero(data.spectrum_sample);
  }
  if ((isfinite(eval.pdf) == false) || (eval.pdf <= kEpsilon)) {
    return bsdf_sample_zero(data.spectrum_sample);
  }

  BSDFSample result = ETX_ZERO(BSDFSample);
  result.w_o = world_w_o;
  result.pdf = eval.pdf;
  result.weight = spectral_response_div(eval.bsdf, eval.pdf);
  if (bsdf_energy_compensated_spectral_response_finite(result.weight) == false) {
    return bsdf_sample_zero(data.spectrum_sample);
  }
  result.properties = BSDFSample::Reflection | (sampled_diffuse ? BSDFSample::Diffuse : 0u);
  result.medium_index = data.current_medium;
  result.eta = 1.0f;
  return result;
}

ETX_SHARED_NOINLINE float bsdf_plastic_pdf(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(float3, outgoing_direction), ETX_IN(Material, material),
  ETX_INOUT(Sampler, sampler)) {
  (void)sampler;
  if (bsdf_plastic_delta_thinfilm_supported(context, material, data.tex)) {
    return bsdf_plastic_delta_thinfilm_pdf(context, data, outgoing_direction, material, sampler);
  }

  if (bsdf_plastic_supported(context, material, data.tex) == false) {
    return bsdf_diffuse_pdf(context, data, outgoing_direction, material, sampler);
  }

  const LocalFrame frame = bsdf_plastic_coating_frame(data, material);
  const float3 local_w_i = local_frame_to_local(frame, -data.w_i);
  const float3 local_w_o = local_frame_to_local(frame, outgoing_direction);
  if ((local_w_i.z <= kEpsilon) || (local_w_o.z <= kEpsilon)) {
    return 0.0f;
  }

  const float alpha = bsdf_energy_compensated_scalar_roughness(context, material, data.tex);
  const SpectralResponse substrate = bsdf_resource_apply_image(context, data.spectrum_sample, material.scattering, data.tex);
  const RefractiveIndexSample ext_ior = bsdf_resource_evaluate_refractive_index(context, material.ext_ior, data.spectrum_sample);
  const RefractiveIndexSample int_ior = bsdf_resource_evaluate_refractive_index(context, material.int_ior, data.spectrum_sample);
  const ThinfilmEval thinfilm = bsdf_resource_evaluate_thinfilm(context, data.spectrum_sample, material.thinfilm, data.tex, sampler);
  const float thinfilm_lut_value = bsdf_energy_compensated_thinfilm_lut_value(material, thinfilm);
  const float specular_probability =
    bsdf_plastic_specular_sample_probability(context, data.spectrum_sample, material, substrate, local_w_i.z, alpha, thinfilm_lut_value);
  const BSDFPlasticCoatingReflectionProposal coating_proposal =
    bsdf_plastic_coating_reflection_proposal(context, data.spectrum_sample, material, local_w_i.z, alpha, thinfilm_lut_value, ext_ior, int_ior);
  float specular_pdf = 0.0f;
  if (coating_proposal.probability > kEpsilon) {
    const float full_dielectric_pdf =
      bsdf_energy_compensated_dielectric_pdf_local(context, data.spectrum_sample, material, local_w_i, local_w_o, alpha, ext_ior, int_ior, thinfilm);
    specular_pdf = full_dielectric_pdf / coating_proposal.probability;
  }
  const float diffuse_pdf = local_w_o.z * kInvPi;
  return specular_probability * specular_pdf + (1.0f - specular_probability) * diffuse_pdf;
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
