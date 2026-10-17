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
  SpectralResponse internal_average_reflection ETX_INIT({});
  SpectralResponse internal_average_transmission ETX_INIT({});
  float compensation_reflection_probability ETX_INIT(0.0f);
};

struct BSDFPlasticDeltaThinfilmTerms {
  LocalFrame frame ETX_INIT({});
  float3 local_w_i ETX_INIT({});
  SpectralResponse substrate ETX_INIT({});
  SpectralResponse reflection ETX_INIT({});
  SpectralResponse diffuse_scale ETX_INIT({});
  float diffuse_roughness ETX_INIT(0.0f);
  float specular_probability ETX_INIT(0.0f);
  bool valid ETX_INIT(false);
};

struct BSDFPlasticIncidentTerms {
  SpectralResponse diffuse_scale ETX_INIT({});
  BSDFPlasticCoatingReflectionProposal coating_proposal ETX_INIT({});
  float specular_probability ETX_INIT(0.0f);
};

ETX_SHARED_NOINLINE float bsdf_plastic_opaque_pdf(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(float3, outgoing_direction), ETX_IN(Material, material),
  ETX_INOUT(Sampler, sampler));

ETX_SHARED_NOINLINE float bsdf_plastic_pdf_prepared(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(Material, material), ETX_IN(float3, local_w_i),
  ETX_IN(float3, local_w_o), ETX_IN(float2, roughness), float alpha, ETX_IN(SpectralResponse, substrate), float2 interface_lut_value);

ETX_SHARED_NOINLINE float bsdf_plastic_pdf_prepared_impl(ETX_IN(float3, local_w_i), ETX_IN(float3, local_w_o), ETX_IN(float2, roughness),
  ETX_IN(BSDFPlasticIncidentTerms, incident_terms));

ETX_SHARED_INLINE BSDFEval bsdf_plastic_delta_thinfilm_evaluate(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(float3, outgoing_direction),
  ETX_IN(Material, material), ETX_IN(float2, roughness), ETX_INOUT(Sampler, sampler));

ETX_SHARED_INLINE BSDFSample bsdf_plastic_delta_thinfilm_sample(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(Material, material), ETX_IN(float2, roughness),
  ETX_INOUT(Sampler, sampler));

ETX_SHARED_INLINE float bsdf_plastic_delta_thinfilm_pdf(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(float3, outgoing_direction),
  ETX_IN(Material, material), ETX_IN(float2, roughness), ETX_INOUT(Sampler, sampler));

ETX_SHARED_INLINE bool bsdf_plastic_supported(ETX_IN(BSDFResourceContext, context), ETX_IN(Material, material), ETX_IN(float2, uv)) {
  (void)uv;
  return bsdf_energy_compensated_material_interface_valid(context, material, MaterialClass::Dielectric);
}

ETX_SHARED_INLINE LocalFrame bsdf_plastic_coating_frame(ETX_IN(BSDFData, data), ETX_IN(Material, material)) {
  return bsdf_data_get_normal_frame(data, material);
}

ETX_SHARED_INLINE float2 bsdf_plastic_coating_roughness(ETX_IN(float2, roughness)) {
  return float2(max(kBSDFNormalDistributionMinAlpha, roughness.x), max(kBSDFNormalDistributionMinAlpha, roughness.y));
}

ETX_SHARED_INLINE SpectralResponse bsdf_plastic_one(ETX_IN(SpectralQuery, spect)) {
  return spectral_response_make(spect, 1.0f);
}

ETX_SHARED_INLINE bool bsdf_plastic_delta_thinfilm_supported(ETX_IN(Material, material), ETX_IN(float2, roughness)) {
  if (bsdf_dielectric_has_thinfilm(material) == false) {
    return false;
  }

  return max(roughness.x, roughness.y) <= kDeltaAlphaTreshold;
}

ETX_SHARED_INLINE SpectralResponse bsdf_plastic_dielectric_total_branch_albedo_from_pair(ETX_IN(BSDFResourceContext, context), ETX_IN(SpectralQuery, spect),
  ETX_IN(Material, material), ETX_IN(BSDFEnergyCompensatedDielectricBranchPair, pair), float alpha, bool incident_outside, bool outgoing_outside, float2 interface_lut_value) {
  const SpectralResponse single = bsdf_energy_compensated_dielectric_branch_pair_selected_albedo(spect, pair, outgoing_outside);
  const SpectralResponse e_i = bsdf_energy_compensated_dielectric_branch_pair_albedo(spect, pair);
  const SpectralResponse d_i = spectral_response_max(spectral_response_sub(bsdf_plastic_one(spect), e_i), 0.0f);
  const SpectralResponse d_o = bsdf_energy_compensated_dielectric_average_residual(context, spect, material, alpha, outgoing_outside, interface_lut_value);
  const SpectralResponse coefficient =
    bsdf_energy_compensated_dielectric_branch_coefficient(context, spect, material, alpha, incident_outside, outgoing_outside, interface_lut_value);
  return spectral_response_min(spectral_response_add(single, spectral_response_mul(spectral_response_mul(coefficient, d_i), d_o)), 1.0f);
}

ETX_SHARED_NOINLINE SpectralResponse bsdf_plastic_dielectric_total_branch_albedo(ETX_IN(BSDFResourceContext, context), ETX_IN(SpectralQuery, spect), ETX_IN(Material, material),
  float mu, float alpha, bool incident_outside, bool outgoing_outside, float2 interface_lut_value) {
  const BSDFEnergyCompensatedDielectricBranchPair pair =
    bsdf_energy_compensated_dielectric_branch_pair_value(context, spect, material, mu, alpha, incident_outside, interface_lut_value);
  return bsdf_plastic_dielectric_total_branch_albedo_from_pair(context, spect, material, pair, alpha, incident_outside, outgoing_outside, interface_lut_value);
}

ETX_SHARED_NOINLINE SpectralResponse bsdf_plastic_dielectric_total_branch_albedo(ETX_IN(BSDFResourceContext, context), ETX_IN(SpectralQuery, spect), ETX_IN(Material, material),
  ETX_IN(float3, direction), float alpha, bool outgoing_outside, float2 interface_lut_value) {
  const BSDFEnergyCompensatedDielectricBranchPair pair = bsdf_energy_compensated_dielectric_branch_pair_value(context, spect, material, direction, alpha, interface_lut_value);
  return bsdf_plastic_dielectric_total_branch_albedo_from_pair(context, spect, material, pair, alpha, direction.z > 0.0f, outgoing_outside, interface_lut_value);
}

ETX_SHARED_INLINE SpectralResponse bsdf_plastic_dielectric_total_branch_average_albedo_from_residuals(ETX_IN(BSDFResourceContext, context), ETX_IN(SpectralQuery, spect),
  ETX_IN(Material, material), float alpha, bool incident_outside, bool outgoing_outside, float2 interface_lut_value, ETX_IN(SpectralResponse, d_i), ETX_IN(SpectralResponse, d_o)) {
  const SpectralResponse single =
    bsdf_energy_compensated_dielectric_branch_average_albedo(context, spect, material, alpha, incident_outside, outgoing_outside, interface_lut_value);
  const SpectralResponse coefficient = bsdf_energy_compensated_dielectric_branch_coefficient(context, spect, material.energy_compensation_interface_index, alpha, incident_outside,
    outgoing_outside, interface_lut_value, d_i, d_o);
  return spectral_response_min(spectral_response_add(single, spectral_response_mul(spectral_response_mul(coefficient, d_i), d_o)), 1.0f);
}

ETX_SHARED_INLINE SpectralResponse bsdf_plastic_dielectric_total_branch_average_albedo(ETX_IN(BSDFResourceContext, context), ETX_IN(SpectralQuery, spect),
  ETX_IN(Material, material), float alpha, bool incident_outside, bool outgoing_outside, float2 interface_lut_value) {
  const SpectralResponse d_i = bsdf_energy_compensated_dielectric_average_residual(context, spect, material, alpha, incident_outside, interface_lut_value);
  SpectralResponse d_o = d_i;
  if (incident_outside != outgoing_outside) {
    d_o = bsdf_energy_compensated_dielectric_average_residual(context, spect, material, alpha, outgoing_outside, interface_lut_value);
  }
  return bsdf_plastic_dielectric_total_branch_average_albedo_from_residuals(context, spect, material, alpha, incident_outside, outgoing_outside, interface_lut_value, d_i, d_o);
}

ETX_SHARED_INLINE SpectralResponse bsdf_plastic_external_reflection_albedo(ETX_IN(BSDFResourceContext, context), ETX_IN(SpectralQuery, spect), ETX_IN(Material, material), float mu,
  float alpha, float2 interface_lut_value) {
  return bsdf_plastic_dielectric_total_branch_albedo(context, spect, material, mu, alpha, true, true, interface_lut_value);
}

ETX_SHARED_INLINE BSDFPlasticExternalAlbedos bsdf_plastic_external_albedos_from_pair(ETX_IN(BSDFResourceContext, context), ETX_IN(SpectralQuery, spect), ETX_IN(Material, material),
  ETX_IN(BSDFEnergyCompensatedDielectricBranchPair, pair), float alpha, float2 interface_lut_value) {
  BSDFPlasticExternalAlbedos result = ETX_ZERO(BSDFPlasticExternalAlbedos);
  const SpectralResponse e_i = bsdf_energy_compensated_dielectric_branch_pair_albedo(spect, pair);
  const SpectralResponse d_i = spectral_response_max(spectral_response_sub(bsdf_plastic_one(spect), e_i), 0.0f);
  const SpectralResponse d_o_reflection = bsdf_energy_compensated_dielectric_average_residual(context, spect, material, alpha, true, interface_lut_value);
  const SpectralResponse d_o_transmission = bsdf_energy_compensated_dielectric_average_residual(context, spect, material, alpha, false, interface_lut_value);
  const SpectralResponse reflection_coefficient = bsdf_energy_compensated_dielectric_branch_coefficient(context, spect, material.energy_compensation_interface_index, alpha, true,
    true, interface_lut_value, d_o_reflection, d_o_reflection);
  const SpectralResponse transmission_coefficient = bsdf_energy_compensated_dielectric_branch_coefficient(context, spect, material.energy_compensation_interface_index, alpha, true,
    false, interface_lut_value, d_o_reflection, d_o_transmission);
  const SpectralResponse single_reflection = bsdf_energy_compensated_dielectric_branch_pair_selected_albedo(spect, pair, true);
  const SpectralResponse single_transmission = bsdf_energy_compensated_dielectric_branch_pair_selected_albedo(spect, pair, false);
  result.reflection =
    spectral_response_min(spectral_response_add(single_reflection, spectral_response_mul(spectral_response_mul(reflection_coefficient, d_i), d_o_reflection)), 1.0f);
  result.transmission =
    spectral_response_min(spectral_response_add(single_transmission, spectral_response_mul(spectral_response_mul(transmission_coefficient, d_i), d_o_transmission)), 1.0f);
  result.compensation_reflection_probability =
    bsdf_energy_compensated_dielectric_compensation_branch_proposal(reflection_coefficient, transmission_coefficient, d_o_reflection, d_o_transmission).x;
  result.internal_average_reflection =
    bsdf_plastic_dielectric_total_branch_average_albedo_from_residuals(context, spect, material, alpha, false, false, interface_lut_value, d_o_transmission, d_o_transmission);
  result.internal_average_transmission =
    bsdf_plastic_dielectric_total_branch_average_albedo_from_residuals(context, spect, material, alpha, false, true, interface_lut_value, d_o_transmission, d_o_reflection);
  return result;
}

ETX_SHARED_INLINE SpectralResponse bsdf_plastic_internal_transmission_albedo(ETX_IN(BSDFResourceContext, context), ETX_IN(SpectralQuery, spect), ETX_IN(Material, material),
  float mu, float alpha, float2 interface_lut_value) {
  return bsdf_plastic_dielectric_total_branch_albedo(context, spect, material, mu, alpha, false, true, interface_lut_value);
}

ETX_SHARED_INLINE BSDFPlasticDeltaThinfilmTerms bsdf_plastic_delta_thinfilm_terms(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(Material, material),
  ETX_IN(float2, roughness), ETX_INOUT(Sampler, sampler)) {
  BSDFPlasticDeltaThinfilmTerms result = ETX_ZERO(BSDFPlasticDeltaThinfilmTerms);
  result.frame = bsdf_plastic_coating_frame(data, material);
  result.local_w_i = local_frame_to_local(result.frame, -data.w_i);
  if (result.local_w_i.z <= kEpsilon) {
    return result;
  }

  const RefractiveIndexSample ext_ior = bsdf_resource_evaluate_refractive_index(context, material.ext_ior, data.spectrum_sample);
  const RefractiveIndexSample int_ior = bsdf_resource_evaluate_refractive_index(context, material.int_ior, data.spectrum_sample);
  const ThinfilmEval thinfilm = bsdf_resource_evaluate_interface_optics(context, data.spectrum_sample, material, data.tex, sampler);
  const SpectralResponse fresnel_i = bsdf_fresnel_calculate(data.spectrum_sample, result.local_w_i.z, ext_ior, int_ior, thinfilm);
  const SpectralResponse one = bsdf_plastic_one(data.spectrum_sample);
  result.substrate = bsdf_resource_apply_image(context, data.spectrum_sample, material.scattering, data.tex);
  const SpectralResponse reflectance = bsdf_resource_apply_image(context, data.spectrum_sample, material.reflectance, data.tex);
  const SpectralResponse internal_reflection_average = bsdf_fresnel_film_internal_average(data.spectrum_sample, ext_ior, int_ior, thinfilm);
  const SpectralResponse internal_transmission_average =
    bsdf_fresnel_transmission_average(data.spectrum_sample, int_ior, ext_ior, bsdf_fresnel_orient_coherent_interface(thinfilm, false));
  const SpectralResponse denominator = spectral_response_max(spectral_response_sub(one, spectral_response_mul(result.substrate, internal_reflection_average)), kEpsilon);
  const SpectralResponse t_i = bsdf_fresnel_transmission_from_reflection(data.spectrum_sample, result.local_w_i.z, ext_ior, int_ior, thinfilm, fresnel_i);
  result.reflection = spectral_response_mul(reflectance, fresnel_i);
  const SpectralResponse average_diffuse_scale = spectral_response_div(spectral_response_mul(t_i, internal_transmission_average), denominator);
  const SpectralResponse eta = spectral_response_div(int_ior.eta, ext_ior.eta);
  result.diffuse_scale = spectral_response_div(t_i, spectral_response_mul(denominator, spectral_response_mul(eta, eta)));
  const float2 base_roughness = material.layer_base_class != 0u ? bsdf_resource_evaluate_base_roughness(context, material, data.tex) : roughness;
  result.diffuse_roughness = saturate(0.5f * (base_roughness.x + base_roughness.y));

  const float reflection_energy = max(0.0f, spectral_response_monochromatic(result.reflection));
  const float substrate_energy = max(0.0f, spectral_response_monochromatic(spectral_response_mul(result.substrate, average_diffuse_scale)));
  const float total_energy = reflection_energy + substrate_energy;
  result.specular_probability = (total_energy > kEpsilon) ? bsdf_energy_compensated_saturate(reflection_energy / total_energy) : 0.0f;
  result.valid = true;
  return result;
}

ETX_SHARED_INLINE BSDFEval bsdf_plastic_delta_thinfilm_evaluate_prepared(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(float3, outgoing_direction),
  ETX_IN(Material, material), ETX_INOUT(Sampler, sampler), ETX_IN(BSDFPlasticDeltaThinfilmTerms, terms)) {
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

  const float diffuse_probability = max(0.0f, 1.0f - terms.specular_probability);
  const SpectralResponse substrate_func = bsdf_diffuse_eon_brdf(data.spectrum_sample, terms.substrate, terms.local_w_i, local_w_o, terms.diffuse_roughness);
  const RefractiveIndexSample ext_ior = bsdf_resource_evaluate_refractive_index(context, material.ext_ior, data.spectrum_sample);
  const RefractiveIndexSample int_ior = bsdf_resource_evaluate_refractive_index(context, material.int_ior, data.spectrum_sample);
  const ThinfilmEval film = bsdf_resource_evaluate_interface_optics(context, data.spectrum_sample, material, data.tex, sampler);
  const SpectralResponse transmission = bsdf_fresnel_transmission_calculate(data.spectrum_sample, local_w_o.z, ext_ior, int_ior, film);
  result.bsdf = spectral_response_mul(spectral_response_mul(spectral_response_mul(substrate_func, local_w_o.z), terms.diffuse_scale), transmission);
  result.func = spectral_response_div(result.bsdf, local_w_o.z);
  result.pdf = diffuse_probability * local_w_o.z * kInvPi;
  result.eta = 1.0f;
  result.properties = BSDFSample::Diffuse | BSDFSample::Reflection;
  result.medium_index = data.current_medium;
  return result;
}

ETX_SHARED_INLINE BSDFEval bsdf_plastic_delta_thinfilm_evaluate(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(float3, outgoing_direction),
  ETX_IN(Material, material), ETX_IN(float2, roughness), ETX_INOUT(Sampler, sampler)) {
  const BSDFPlasticDeltaThinfilmTerms terms = bsdf_plastic_delta_thinfilm_terms(context, data, material, roughness, sampler);
  return bsdf_plastic_delta_thinfilm_evaluate_prepared(context, data, outgoing_direction, material, sampler, terms);
}

ETX_SHARED_INLINE BSDFSample bsdf_plastic_delta_thinfilm_sample(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(Material, material), ETX_IN(float2, roughness),
  ETX_INOUT(Sampler, sampler)) {
  const BSDFPlasticDeltaThinfilmTerms terms = bsdf_plastic_delta_thinfilm_terms(context, data, material, roughness, sampler);
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
  const BSDFEval eval = bsdf_plastic_delta_thinfilm_evaluate_prepared(context, data, world_w_o, material, sampler, terms);
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
  ETX_IN(Material, material), ETX_IN(float2, roughness), ETX_INOUT(Sampler, sampler)) {
  const BSDFPlasticDeltaThinfilmTerms terms = bsdf_plastic_delta_thinfilm_terms(context, data, material, roughness, sampler);
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

ETX_SHARED_INLINE SpectralResponse bsdf_plastic_coated_diffuse_func_prepared(ETX_IN(BSDFData, data), ETX_IN(float3, local_w_i), ETX_IN(float3, local_w_o),
  ETX_IN(SpectralResponse, substrate), float diffuse_roughness, ETX_IN(SpectralResponse, diffuse_scale)) {
  if ((local_w_i.z <= kEpsilon) || (local_w_o.z <= kEpsilon)) {
    return spectral_response_zero(data.spectrum_sample);
  }

  const SpectralResponse substrate_func = bsdf_diffuse_eon_brdf(data.spectrum_sample, substrate, local_w_i, local_w_o, diffuse_roughness);
  return spectral_response_mul(substrate_func, diffuse_scale);
}

ETX_SHARED_INLINE BSDFPlasticCoatingReflectionProposal bsdf_plastic_coating_reflection_proposal_from_pair(ETX_IN(SpectralQuery, spect),
  ETX_IN(BSDFEnergyCompensatedDielectricBranchPair, pair), float compensation_branch_probability) {
  BSDFPlasticCoatingReflectionProposal result = ETX_ZERO(BSDFPlasticCoatingReflectionProposal);
  const float visible_probability = bsdf_energy_compensated_dielectric_branch_pair_visible_probability(pair);
  const float reflection_visible_probability = bsdf_energy_compensated_dielectric_branch_pair_selected_visible_probability(pair, true);
  const SpectralResponse e_i = bsdf_energy_compensated_dielectric_branch_pair_albedo(spect, pair);
  const float base_probability = bsdf_energy_compensated_saturate(spectral_response_monochromatic(e_i));
  const float base_reflection_probability = (visible_probability > kEpsilon) ? bsdf_energy_compensated_saturate(reflection_visible_probability / visible_probability) : 0.0f;
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

ETX_SHARED_INLINE BSDFPlasticCoatingReflectionProposal bsdf_plastic_coating_reflection_proposal_from_pair(ETX_IN(BSDFResourceContext, context), ETX_IN(SpectralQuery, spect),
  ETX_IN(Material, material), ETX_IN(BSDFEnergyCompensatedDielectricBranchPair, pair), float alpha, float2 interface_lut_value) {
  const float compensation_branch_probability =
    bsdf_energy_compensated_dielectric_compensation_branch_probability(context, spect, material, alpha, true, true, interface_lut_value);
  return bsdf_plastic_coating_reflection_proposal_from_pair(spect, pair, compensation_branch_probability);
}

ETX_SHARED_NOINLINE BSDFPlasticCoatingReflectionProposal bsdf_plastic_coating_reflection_proposal(ETX_IN(BSDFResourceContext, context), ETX_IN(SpectralQuery, spect),
  ETX_IN(Material, material), float mu_i, float alpha, float2 interface_lut_value) {
  const BSDFEnergyCompensatedDielectricBranchPair pair = bsdf_energy_compensated_dielectric_branch_pair_value(context, spect, material, mu_i, alpha, true, interface_lut_value);
  return bsdf_plastic_coating_reflection_proposal_from_pair(context, spect, material, pair, alpha, interface_lut_value);
}

ETX_SHARED_INLINE BSDFPlasticCoatingReflectionProposal bsdf_plastic_coating_reflection_proposal(ETX_IN(BSDFResourceContext, context), ETX_IN(SpectralQuery, spect),
  ETX_IN(Material, material), ETX_IN(float3, local_w_i), float alpha, float2 interface_lut_value) {
  const BSDFEnergyCompensatedDielectricBranchPair pair = bsdf_energy_compensated_dielectric_branch_pair_value(context, spect, material, local_w_i, alpha, interface_lut_value);
  return bsdf_plastic_coating_reflection_proposal_from_pair(context, spect, material, pair, alpha, interface_lut_value);
}

ETX_SHARED_INLINE BSDFPlasticIncidentTerms bsdf_plastic_prepare_incident_terms(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(Material, material),
  ETX_IN(float3, local_w_i), ETX_IN(SpectralResponse, substrate), float alpha, float2 interface_lut_value) {
  BSDFPlasticIncidentTerms result = ETX_ZERO(BSDFPlasticIncidentTerms);
  const BSDFEnergyCompensatedDielectricBranchPair pair =
    bsdf_energy_compensated_dielectric_branch_pair_value(context, data.spectrum_sample, material, local_w_i, alpha, interface_lut_value);
  const BSDFPlasticExternalAlbedos external_albedos = bsdf_plastic_external_albedos_from_pair(context, data.spectrum_sample, material, pair, alpha, interface_lut_value);
  const SpectralResponse denominator =
    spectral_response_max(spectral_response_sub(bsdf_plastic_one(data.spectrum_sample), spectral_response_mul(substrate, external_albedos.internal_average_reflection)), kEpsilon);
  result.diffuse_scale = spectral_response_div(spectral_response_mul(external_albedos.transmission, external_albedos.internal_average_transmission), denominator);

  const SpectralResponse diffuse_energy = spectral_response_mul(result.diffuse_scale, substrate);
  const float reflection_energy = max(0.0f, spectral_response_monochromatic(external_albedos.reflection));
  const float substrate_energy = max(0.0f, spectral_response_monochromatic(diffuse_energy));
  const float total_energy = reflection_energy + substrate_energy;
  result.specular_probability = (total_energy > kEpsilon) ? bsdf_energy_compensated_saturate(reflection_energy / total_energy) : 0.0f;
  result.coating_proposal = bsdf_plastic_coating_reflection_proposal_from_pair(data.spectrum_sample, pair, external_albedos.compensation_reflection_probability);
  return result;
}

ETX_SHARED_NOINLINE BSDFEval bsdf_plastic_evaluate_prepared_impl(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(Material, material),
  ETX_IN(float3, local_w_i), ETX_IN(float3, local_w_o), ETX_IN(float2, roughness), float alpha, ETX_IN(SpectralResponse, substrate), ETX_IN(SpectralResponse, reflectance),
  ETX_IN(RefractiveIndexSample, ext_ior), ETX_IN(RefractiveIndexSample, int_ior), ETX_IN(ThinfilmEval, thinfilm), ETX_IN(BSDFPlasticIncidentTerms, incident_terms)) {
  if ((local_w_i.z <= kEpsilon) || (local_w_o.z <= kEpsilon)) {
    return bsdf_eval_zero(data.spectrum_sample);
  }

  const float2 coating_roughness = bsdf_plastic_coating_roughness(roughness);
  const SpectralResponse coating_bsdf =
    bsdf_energy_compensated_dielectric_bsdf_local(context, data.spectrum_sample, material, local_w_i, local_w_o, alpha, coating_roughness, ext_ior, int_ior, thinfilm, reflectance);
  const float2 base_roughness = material.layer_base_class != 0u ? bsdf_resource_evaluate_base_roughness(context, material, data.tex) : roughness;
  const float diffuse_roughness = saturate(0.5f * (base_roughness.x + base_roughness.y));
  const float2 interface_lut_value = bsdf_energy_compensated_interface_lut_value(material, thinfilm);
  const SpectralResponse outgoing_transmission = bsdf_plastic_dielectric_total_branch_albedo(context, data.spectrum_sample, material, local_w_o, alpha, false, interface_lut_value);
  const SpectralResponse transmission_average =
    bsdf_plastic_dielectric_total_branch_average_albedo(context, data.spectrum_sample, material, alpha, true, false, interface_lut_value);
  const SpectralResponse escape_scale = spectral_response_div(outgoing_transmission, spectral_response_max(transmission_average, kEpsilon));
  const SpectralResponse diffuse_func =
    bsdf_plastic_coated_diffuse_func_prepared(data, local_w_i, local_w_o, substrate, diffuse_roughness, spectral_response_mul(incident_terms.diffuse_scale, escape_scale));
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
  result.pdf = bsdf_plastic_pdf_prepared_impl(local_w_i, local_w_o, roughness, incident_terms);
  result.eta = 1.0f;
  result.properties = BSDFSample::Reflection;
  result.medium_index = data.current_medium;
  return result;
}

ETX_SHARED_NOINLINE BSDFEval bsdf_plastic_evaluate_prepared(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(Material, material), ETX_IN(float3, local_w_i),
  ETX_IN(float3, local_w_o), ETX_IN(float2, roughness), float alpha, ETX_IN(SpectralResponse, substrate), ETX_IN(SpectralResponse, reflectance),
  ETX_IN(RefractiveIndexSample, ext_ior), ETX_IN(RefractiveIndexSample, int_ior), ETX_IN(ThinfilmEval, thinfilm), float2 interface_lut_value) {
  const BSDFPlasticIncidentTerms incident_terms = bsdf_plastic_prepare_incident_terms(context, data, material, local_w_i, substrate, alpha, interface_lut_value);
  return bsdf_plastic_evaluate_prepared_impl(context, data, material, local_w_i, local_w_o, roughness, alpha, substrate, reflectance, ext_ior, int_ior, thinfilm, incident_terms);
}

ETX_SHARED_NOINLINE BSDFEval bsdf_plastic_opaque_evaluate(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(float3, outgoing_direction),
  ETX_IN(Material, material), ETX_INOUT(Sampler, sampler)) {
  const bool thinfilm_enabled = bsdf_dielectric_has_thinfilm(material);
  float2 roughness = float2(0.0f, 0.0f);
  if (thinfilm_enabled) {
    roughness = bsdf_resource_evaluate_roughness(context, material, data.tex);
    if (bsdf_plastic_delta_thinfilm_supported(material, roughness)) {
      return bsdf_plastic_delta_thinfilm_evaluate(context, data, outgoing_direction, material, roughness, sampler);
    }
  }

  if (bsdf_plastic_supported(context, material, data.tex) == false) {
    return bsdf_diffuse_evaluate(context, data, outgoing_direction, material, sampler);
  }

  if (thinfilm_enabled == false) {
    roughness = bsdf_resource_evaluate_roughness(context, material, data.tex);
  }

  const LocalFrame frame = bsdf_plastic_coating_frame(data, material);
  const float3 local_w_i = local_frame_to_local(frame, -data.w_i);
  const float3 local_w_o = local_frame_to_local(frame, outgoing_direction);
  if ((local_w_i.z <= kEpsilon) || (local_w_o.z <= kEpsilon)) {
    return bsdf_eval_zero(data.spectrum_sample);
  }

  const float alpha = bsdf_energy_compensated_scalar_roughness_from_value(roughness);
  const RefractiveIndexSample ext_ior = bsdf_resource_evaluate_refractive_index(context, material.ext_ior, data.spectrum_sample);
  const RefractiveIndexSample int_ior = bsdf_resource_evaluate_refractive_index(context, material.int_ior, data.spectrum_sample);
  const ThinfilmEval thinfilm = bsdf_resource_evaluate_interface_optics(context, data.spectrum_sample, material, data.tex, sampler);
  const float2 interface_lut_value = bsdf_energy_compensated_interface_lut_value(material, thinfilm);
  const SpectralResponse substrate = bsdf_resource_apply_image(context, data.spectrum_sample, material.scattering, data.tex);
  const SpectralResponse reflectance = bsdf_resource_apply_image(context, data.spectrum_sample, material.reflectance, data.tex);
  return bsdf_plastic_evaluate_prepared(context, data, material, local_w_i, local_w_o, roughness, alpha, substrate, reflectance, ext_ior, int_ior, thinfilm, interface_lut_value);
}

ETX_SHARED_NOINLINE BSDFSample bsdf_plastic_opaque_sample(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(Material, material), ETX_INOUT(Sampler, sampler)) {
  const bool thinfilm_enabled = bsdf_dielectric_has_thinfilm(material);
  float2 roughness = float2(0.0f, 0.0f);
  if (thinfilm_enabled) {
    roughness = bsdf_resource_evaluate_roughness(context, material, data.tex);
    if (bsdf_plastic_delta_thinfilm_supported(material, roughness)) {
      return bsdf_plastic_delta_thinfilm_sample(context, data, material, roughness, sampler);
    }
  }

  if (bsdf_plastic_supported(context, material, data.tex) == false) {
    return bsdf_diffuse_sample(context, data, material, sampler);
  }

  if (thinfilm_enabled == false) {
    roughness = bsdf_resource_evaluate_roughness(context, material, data.tex);
  }

  const LocalFrame frame = bsdf_plastic_coating_frame(data, material);
  const float3 local_w_i = local_frame_to_local(frame, -data.w_i);
  if (local_w_i.z <= kEpsilon) {
    return bsdf_sample_zero(data.spectrum_sample);
  }

  const float alpha = bsdf_energy_compensated_scalar_roughness_from_value(roughness);
  const SpectralResponse substrate = bsdf_resource_apply_image(context, data.spectrum_sample, material.scattering, data.tex);
  const float2 coating_roughness = bsdf_plastic_coating_roughness(roughness);
  const RefractiveIndexSample ext_ior = bsdf_resource_evaluate_refractive_index(context, material.ext_ior, data.spectrum_sample);
  const RefractiveIndexSample int_ior = bsdf_resource_evaluate_refractive_index(context, material.int_ior, data.spectrum_sample);
  const ThinfilmEval thinfilm = bsdf_resource_evaluate_interface_optics(context, data.spectrum_sample, material, data.tex, sampler);
  const float2 interface_lut_value = bsdf_energy_compensated_interface_lut_value(material, thinfilm);
  const BSDFPlasticIncidentTerms incident_terms = bsdf_plastic_prepare_incident_terms(context, data, material, local_w_i, substrate, alpha, interface_lut_value);
  const bool has_fixed = bsdf_sampler_has_fixed(sampler);
  float selector = sampler.fixed_w;
  float2 rnd = float2(sampler.fixed_u, sampler.fixed_v);
  if (has_fixed == false) {
    selector = bsdf_sampler_next(sampler);
    rnd = bsdf_sampler_next_2d(sampler);
  }

  float3 local_w_o = float3(0.0f, 0.0f, 0.0f);
  bool sampled_diffuse = (selector >= incident_terms.specular_probability) || (incident_terms.coating_proposal.probability <= kEpsilon);
  if (sampled_diffuse == false) {
    const float coating_selector = bsdf_sampler_next(sampler);
    if (coating_selector < incident_terms.coating_proposal.base_probability) {
      const float3 m = bsdf_energy_compensated_sample_vndf_local(local_w_i, coating_roughness, rnd);
      local_w_o = -local_w_i + 2.0f * m * dot(local_w_i, m);
      if (local_w_o.z <= kEpsilon) {
        return bsdf_sample_zero(data.spectrum_sample);
      }
    } else {
      local_w_o = sample_cosine_distribution(rnd, 1.0f);
    }
  } else {
    local_w_o = sample_cosine_distribution(rnd, 1.0f);
  }

  const float3 world_w_o = normalize(local_frame_from_local(frame, local_w_o));
  const SpectralResponse reflectance = bsdf_resource_apply_image(context, data.spectrum_sample, material.reflectance, data.tex);
  const BSDFEval eval =
    bsdf_plastic_evaluate_prepared_impl(context, data, material, local_w_i, local_w_o, roughness, alpha, substrate, reflectance, ext_ior, int_ior, thinfilm, incident_terms);
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

ETX_SHARED_NOINLINE float bsdf_plastic_pdf_prepared_impl(ETX_IN(float3, local_w_i), ETX_IN(float3, local_w_o), ETX_IN(float2, roughness),
  ETX_IN(BSDFPlasticIncidentTerms, incident_terms)) {
  if ((local_w_i.z <= kEpsilon) || (local_w_o.z <= kEpsilon)) {
    return 0.0f;
  }

  float base_pdf = 0.0f;
  if (incident_terms.coating_proposal.base_probability > kEpsilon) {
    const float2 coating_roughness = bsdf_plastic_coating_roughness(roughness);
    const float3 m = normalize(local_w_i + local_w_o);
    base_pdf = bsdf_energy_compensated_vndf_pdf(local_w_i, m, coating_roughness) / max(kEpsilon, 4.0f * dot(local_w_o, m));
  }
  const float diffuse_pdf = local_w_o.z * kInvPi;
  const float base_mass = incident_terms.specular_probability * incident_terms.coating_proposal.base_probability;
  const float compensation_mass = incident_terms.specular_probability * incident_terms.coating_proposal.compensation_probability;
  const float diffuse_mass = max(0.0f, 1.0f - incident_terms.specular_probability);
  return base_mass * base_pdf + (compensation_mass + diffuse_mass) * diffuse_pdf;
}

ETX_SHARED_NOINLINE float bsdf_plastic_pdf_prepared(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(Material, material), ETX_IN(float3, local_w_i),
  ETX_IN(float3, local_w_o), ETX_IN(float2, roughness), float alpha, ETX_IN(SpectralResponse, substrate), float2 interface_lut_value) {
  const BSDFPlasticIncidentTerms incident_terms = bsdf_plastic_prepare_incident_terms(context, data, material, local_w_i, substrate, alpha, interface_lut_value);
  return bsdf_plastic_pdf_prepared_impl(local_w_i, local_w_o, roughness, incident_terms);
}

ETX_SHARED_NOINLINE float bsdf_plastic_opaque_pdf(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(float3, outgoing_direction), ETX_IN(Material, material),
  ETX_INOUT(Sampler, sampler)) {
  (void)sampler;
  const bool thinfilm_enabled = bsdf_dielectric_has_thinfilm(material);
  float2 roughness = float2(0.0f, 0.0f);
  if (thinfilm_enabled) {
    roughness = bsdf_resource_evaluate_roughness(context, material, data.tex);
    if (bsdf_plastic_delta_thinfilm_supported(material, roughness)) {
      return bsdf_plastic_delta_thinfilm_pdf(context, data, outgoing_direction, material, roughness, sampler);
    }
  }

  if (bsdf_plastic_supported(context, material, data.tex) == false) {
    return bsdf_diffuse_pdf(context, data, outgoing_direction, material, sampler);
  }

  if (thinfilm_enabled == false) {
    roughness = bsdf_resource_evaluate_roughness(context, material, data.tex);
  }

  const LocalFrame frame = bsdf_plastic_coating_frame(data, material);
  const float3 local_w_i = local_frame_to_local(frame, -data.w_i);
  const float3 local_w_o = local_frame_to_local(frame, outgoing_direction);
  if ((local_w_i.z <= kEpsilon) || (local_w_o.z <= kEpsilon)) {
    return 0.0f;
  }

  const float alpha = bsdf_energy_compensated_scalar_roughness_from_value(roughness);
  const SpectralResponse substrate = bsdf_resource_apply_image(context, data.spectrum_sample, material.scattering, data.tex);
  const ThinfilmEval thinfilm = bsdf_resource_evaluate_interface_optics(context, data.spectrum_sample, material, data.tex, sampler);
  const float2 interface_lut_value = bsdf_energy_compensated_interface_lut_value(material, thinfilm);
  return bsdf_plastic_pdf_prepared(context, data, material, local_w_i, local_w_o, roughness, alpha, substrate, interface_lut_value);
}

ETX_SHARED_INLINE bool bsdf_plastic_is_delta(ETX_IN(Material, material), ETX_IN(float2, tex), ETX_INOUT(Sampler, sampler)) {
  if (bsdf_plastic_has_subsurface(material)) {
    return bsdf_dielectric_energy_compensated_is_delta(material, tex, sampler);
  }
  return false;
}

ETX_SHARED_INLINE SpectralResponse bsdf_plastic_albedo(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(Material, material), ETX_INOUT(Sampler, sampler)) {
  (void)sampler;
  return bsdf_resource_apply_image(context, data.spectrum_sample, material.scattering, data.tex);
}

ETX_SHARED_NOINLINE BSDFEval bsdf_plastic_evaluate(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(float3, outgoing_direction), ETX_IN(Material, material),
  ETX_INOUT(Sampler, sampler)) {
  if (bsdf_plastic_has_subsurface(material)) {
    return bsdf_dielectric_energy_compensated_evaluate(context, data, outgoing_direction, material, sampler);
  }
  return bsdf_plastic_opaque_evaluate(context, data, outgoing_direction, material, sampler);
}

ETX_SHARED_NOINLINE BSDFSample bsdf_plastic_sample(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(Material, material), ETX_INOUT(Sampler, sampler)) {
  if (bsdf_plastic_has_subsurface(material)) {
    return bsdf_dielectric_energy_compensated_sample(context, data, material, sampler);
  }
  return bsdf_plastic_opaque_sample(context, data, material, sampler);
}

ETX_SHARED_NOINLINE float bsdf_plastic_pdf(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(float3, outgoing_direction), ETX_IN(Material, material),
  ETX_INOUT(Sampler, sampler)) {
  if (bsdf_plastic_has_subsurface(material)) {
    return bsdf_dielectric_energy_compensated_pdf(context, data, outgoing_direction, material, sampler);
  }
  return bsdf_plastic_opaque_pdf(context, data, outgoing_direction, material, sampler);
}
