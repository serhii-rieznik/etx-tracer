#pragma once

// Included after the shared coating state: physical z>0 is exterior and z<0 is interior.
ETX_SHARED_INLINE Material bsdf_coating_transmission_interface(ETX_IN(BSDFCoatedConductor, layer), bool top) {
  if (top) {
    return bsdf_coating_interface_material(layer.interface_material, MaterialClass::Dielectric);
  }
  return bsdf_coating_interface_material(layer.substrate_material, layer.substrate_class);
}

ETX_SHARED_INLINE bool bsdf_coating_transmission_port_delta(ETX_IN(BSDFCoatedConductor, layer), bool top) {
  return top ? layer.interface_delta : layer.substrate_delta;
}

ETX_SHARED_INLINE bool bsdf_coating_transmission_exit_top(ETX_IN(BSDFCoatingWalk, state)) {
  return state.outgoing.z > 0.0f;
}

ETX_SHARED_INLINE bool bsdf_coating_transmission_entrance_top(ETX_IN(BSDFCoatedConductor, layer)) {
  return layer.data.w_i.z < 0.0f;
}

ETX_SHARED_INLINE bool bsdf_coating_transmission_sample_top(ETX_IN(BSDFCoatedConductor, layer), ETX_IN(BSDFCoatingWalk, state)) {
  if (state.sample_target == BSDFCoatingWalk::Entrance) {
    return bsdf_coating_transmission_entrance_top(layer);
  }
  if (state.sample_target == BSDFCoatingWalk::ReverseExit) {
    return bsdf_coating_transmission_exit_top(state);
  }
  return state.sample_target == BSDFCoatingWalk::Top;
}

ETX_SHARED_INLINE BSDFData bsdf_coating_transmission_sample_data(ETX_IN(BSDFCoatedConductor, layer), ETX_IN(BSDFCoatingWalk, state)) {
  BSDFData data = layer.data;
  if (state.sample_target == BSDFCoatingWalk::ReverseExit) {
    data.w_i = -state.outgoing;
    data.path_source = (data.path_source == PathSource::Light) ? PathSource::Camera : PathSource::Light;
  } else if (state.sample_target != BSDFCoatingWalk::Entrance) {
    data.w_i = state.direction;
  }
  return data;
}

ETX_SHARED_INLINE bool bsdf_coating_transmission_reverse_valid(ETX_IN(BSDFCoatingWalk, state)) {
  return bsdf_sample_valid(state.reverse_exit) && ((state.reverse_exit.properties & BSDFSample::Transmission) != 0u);
}

ETX_SHARED_INLINE bool bsdf_coating_transmission_has_continuous(ETX_IN(BSDFCoatedConductor, layer)) {
  return (layer.interface_delta == false) || (layer.substrate_delta == false) || ((layer.thickness > 0.0f) && (spectral_response_is_zero(layer.scattering) == false));
}

ETX_SHARED_INLINE float2 bsdf_coating_transmission_indices(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFCoatedConductor, layer)) {
  const RefractiveIndexSample exterior = bsdf_resource_evaluate_refractive_index(context, layer.interface_material.ext_ior, layer.data.spectrum_sample);
  const RefractiveIndexSample interior = bsdf_resource_evaluate_refractive_index(context, layer.substrate_material.int_ior, layer.data.spectrum_sample);
  const float exterior_eta = max(kEpsilon, spectral_response_monochromatic(exterior.eta));
  const float interior_eta = max(kEpsilon, spectral_response_monochromatic(interior.eta));
  if (bsdf_coating_transmission_entrance_top(layer)) {
    return float2(exterior_eta, interior_eta);
  }
  return float2(interior_eta, exterior_eta);
}

ETX_SHARED_INLINE float bsdf_coating_transmission_eta(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFCoatedConductor, layer)) {
  const float2 indices = bsdf_coating_transmission_indices(context, layer);
  return indices.y / indices.x;
}

ETX_SHARED_INLINE BSDFEval bsdf_coating_transmission_interface_evaluate(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFCoatedConductor, layer), bool top, ETX_IN(BSDFData, data),
  ETX_IN(float3, outgoing), ETX_INOUT(Sampler, sampler)) {
  const Material material = bsdf_coating_transmission_interface(layer, top);
  if (material.cls == MaterialClass::Translucent) {
    return bsdf_translucent_evaluate(context, data, outgoing, material, sampler);
  }
  return bsdf_dielectric_energy_compensated_evaluate(context, data, outgoing, material, sampler);
}

struct BSDFCoatingAtomInterface {
  SpectralResponse reflection ETX_INIT({});
  SpectralResponse transmission ETX_INIT({});
};

ETX_SHARED_INLINE bool bsdf_coating_transmission_port_atom(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFCoatedConductor, layer), bool top) {
  const Material material = bsdf_coating_transmission_interface(layer, top);
  if (material.cls != MaterialClass::Dielectric) {
    return false;
  }
  if (bsdf_coating_transmission_port_delta(layer, top)) {
    return true;
  }
  const RefractiveIndexSample exterior = bsdf_resource_evaluate_refractive_index(context, material.ext_ior, layer.data.spectrum_sample);
  const RefractiveIndexSample interior = bsdf_resource_evaluate_refractive_index(context, material.int_ior, layer.data.spectrum_sample);
  return bsdf_dielectric_equal_eta(exterior, interior);
}

ETX_SHARED_INLINE BSDFCoatingAtomInterface bsdf_coating_transmission_atom_interface(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFCoatedConductor, layer), bool top,
  ETX_IN(float3, direction), ETX_INOUT(Sampler, sampler)) {
  const SpectralQuery spect = layer.data.spectrum_sample;
  BSDFCoatingAtomInterface result = ETX_ZERO(BSDFCoatingAtomInterface);
  result.reflection = spectral_response_zero(spect);
  result.transmission = spectral_response_zero(spect);
  const Material material = bsdf_coating_transmission_interface(layer, top);
  if (material.cls != MaterialClass::Dielectric) {
    return result;
  }
  const bool outside = direction.z < 0.0f;
  RefractiveIndexSample incident = bsdf_resource_evaluate_refractive_index(context, material.ext_ior, spect);
  RefractiveIndexSample transmitted = bsdf_resource_evaluate_refractive_index(context, material.int_ior, spect);
  if (outside == false) {
    const RefractiveIndexSample original = incident;
    incident = transmitted;
    transmitted = original;
  }
  const InterfaceOptics film = bsdf_fresnel_orient_coherent_interface(bsdf_resource_evaluate_interface_optics(context, spect, material, layer.data.tex, sampler), outside);
  const bool matching = bsdf_dielectric_equal_eta(incident, transmitted);
  const SpectralResponse transmission_texture = bsdf_resource_apply_image(context, spect, material.scattering, layer.data.tex);
  if (bsdf_fresnel_equal_indices(incident, transmitted) && (bsdf_resource_interface_has_layers(film) == false)) {
    result.transmission = transmission_texture;
    return result;
  }
  const float3 incoming = outside ? -direction : direction;
  if (bsdf_coating_transmission_port_delta(layer, top)) {
    result.reflection = spectral_response_mul(bsdf_fresnel_calculate(spect, incoming.z, incident, transmitted, film),
      bsdf_resource_apply_image(context, spect, material.reflectance, layer.data.tex));
    if (bsdf_external_transmission_possible(incoming.z, incident, transmitted)) {
      result.transmission = spectral_response_mul(bsdf_fresnel_transmission_calculate(spect, incoming.z, incident, transmitted, film), transmission_texture);
    }
    return result;
  }
  if (matching == false) {
    return result;
  }
  const float2 alpha = top ? layer.interface_roughness : layer.substrate_roughness;
  const float3 micro_normal = bsdf_energy_compensated_sample_vndf_local(incoming, alpha, bsdf_sampler_next_2d(sampler));
  const float lambda = bsdf_external_ray_info_make(incoming, alpha).Lambda;
  const float geometric_weight = (1.0f + lambda) * bsdf_external_beta(1.0f + lambda, 1.0f + lambda);
  result.transmission = spectral_response_mul(
    spectral_response_mul(bsdf_fresnel_transmission_calculate(spect, dot(incoming, micro_normal), incident, transmitted, film), transmission_texture), geometric_weight);
  return result;
}

ETX_SHARED_INLINE float3 bsdf_coating_transmission_reflected_direction(ETX_IN(float3, direction)) {
  return float3(direction.x, direction.y, -direction.z);
}

ETX_SHARED_INLINE bool bsdf_coating_transmission_inside_direction(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFCoatedConductor, layer), ETX_OUT(float3, direction)) {
  const bool top = bsdf_coating_transmission_entrance_top(layer);
  const RefractiveIndexSample coating = bsdf_resource_evaluate_refractive_index(context, layer.interface_material.int_ior, layer.data.spectrum_sample);
  RefractiveIndexSample incident = bsdf_resource_evaluate_refractive_index(context, layer.interface_material.ext_ior, layer.data.spectrum_sample);
  if (top == false) {
    incident = bsdf_resource_evaluate_refractive_index(context, layer.substrate_material.int_ior, layer.data.spectrum_sample);
  }
  if (bsdf_dielectric_equal_eta(incident, coating)) {
    direction = layer.data.w_i;
    return abs(direction.z) > 0.0f;
  }
  const float eta = bsdf_dielectric_directional_eta(incident, coating);
  const float inside_cosine = bsdf_external_transmitted_cosine(abs(layer.data.w_i.z), incident, coating);
  direction = float3(0.0f, 0.0f, 0.0f);
  if (inside_cosine <= 0.0f) {
    return false;
  }
  direction = float3(layer.data.w_i.x / eta, layer.data.w_i.y / eta, (top ? -1.0f : 1.0f) * inside_cosine);
  return true;
}

ETX_SHARED_NOINLINE BSDFCoatingAtomInterface bsdf_coating_transmission_atoms(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFCoatedConductor, layer),
  ETX_INOUT(Sampler, sampler)) {
  const bool entrance_top = bsdf_coating_transmission_entrance_top(layer);
  const BSDFCoatingAtomInterface entrance = bsdf_coating_transmission_atom_interface(context, layer, entrance_top, layer.data.w_i, sampler);
  BSDFCoatingAtomInterface result = entrance;
  result.transmission = spectral_response_zero(layer.data.spectrum_sample);
  float3 inside;
  if ((bsdf_coating_transmission_port_atom(context, layer, entrance_top) == false) || (bsdf_coating_transmission_inside_direction(context, layer, inside) == false)) {
    return result;
  }
  const BSDFCoatingAtomInterface other = bsdf_coating_transmission_atom_interface(context, layer, entrance_top == false, inside, sampler);
  const BSDFCoatingAtomInterface reverse = bsdf_coating_transmission_atom_interface(context, layer, entrance_top, bsdf_coating_transmission_reflected_direction(inside), sampler);
  const SpectralResponse attenuation = bsdf_coating_transmittance(layer.extinction, layer.thickness / abs(inside.z));
  SpectralResponse denominator = spectral_response_make(layer.data.spectrum_sample, 1.0f);
  if (layer.interface_delta && layer.substrate_delta) {
    denominator = bsdf_coating_series_denominator(reverse.reflection, reverse.transmission, other.reflection, attenuation);
  }
  const SpectralResponse entrance_attenuated = spectral_response_mul(entrance.transmission, attenuation);
  const SpectralResponse roundtrip = spectral_response_mul(spectral_response_mul(other.reflection, attenuation), reverse.transmission);
  result.reflection = spectral_response_add(result.reflection, spectral_response_mul(entrance_attenuated, bsdf_coating_series_ratio(roundtrip, denominator)));
  result.transmission = spectral_response_mul(entrance_attenuated, bsdf_coating_series_ratio(other.transmission, denominator));
  return result;
}

ETX_SHARED_INLINE bool bsdf_coating_transmission_rgb_dispersive(ETX_IN(RefractiveIndexSample, ior)) {
  if (spectral_response_is_spectral(ior.eta)) {
    return false;
  }
  return (ior.eta.integrated.x != ior.eta.integrated.y) || (ior.eta.integrated.y != ior.eta.integrated.z);
}

ETX_SHARED_INLINE bool bsdf_coating_transmission_fixed_atoms(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFCoatedConductor, layer)) {
  if (((layer.interface_delta == false) && bsdf_coating_transmission_port_atom(context, layer, true)) ||
      ((layer.substrate_delta == false) && bsdf_coating_transmission_port_atom(context, layer, false))) {
    return true;
  }
  return bsdf_coating_transmission_rgb_dispersive(bsdf_resource_evaluate_refractive_index(context, layer.interface_material.ext_ior, layer.data.spectrum_sample)) ||
         bsdf_coating_transmission_rgb_dispersive(bsdf_resource_evaluate_refractive_index(context, layer.interface_material.int_ior, layer.data.spectrum_sample)) ||
         bsdf_coating_transmission_rgb_dispersive(bsdf_resource_evaluate_refractive_index(context, layer.substrate_material.int_ior, layer.data.spectrum_sample));
}

ETX_SHARED_NOINLINE float3 bsdf_coating_transmission_probabilities(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFCoatedConductor, layer)) {
  float reflection = 0.0f;
  float transmission = 0.0f;
  const bool fixed = bsdf_coating_transmission_fixed_atoms(context, layer);
  if (fixed == false) {
    Sampler deterministic_sampler = ETX_ZERO(Sampler);
    const BSDFCoatingAtomInterface atoms = bsdf_coating_transmission_atoms(context, layer, deterministic_sampler);
    reflection = saturate(spectral_response_monochromatic(atoms.reflection));
    transmission = saturate(spectral_response_monochromatic(atoms.transmission));
  }
  if (fixed) {
    const bool entrance_top = bsdf_coating_transmission_entrance_top(layer);
    const bool entrance_atom = bsdf_coating_transmission_port_atom(context, layer, entrance_top);
    const bool other_atom = bsdf_coating_transmission_port_atom(context, layer, entrance_top == false);
    reflection =
      (bsdf_coating_transmission_port_delta(layer, entrance_top) || (entrance_atom && bsdf_coating_transmission_port_delta(layer, entrance_top == false))) ? 0.25f : 0.0f;
    transmission = (entrance_atom && other_atom) ? 0.25f : 0.0f;
  }
  const float continuous = bsdf_coating_transmission_has_continuous(layer) ? max(1.0f / 8388608.0f, 1.0f - reflection - transmission) : 0.0f;
  return fixed ? float3(reflection, transmission, continuous) : bsdf_external_atom_proposal(reflection, transmission, continuous);
}

ETX_SHARED_INLINE uint32_t bsdf_coating_transmission_proposal_count(ETX_IN(BSDFCoatedConductor, layer), float eta) {
  uint32_t count = 2u;
  const uint32_t interface_components = eta != 1.0f ? 2u : 1u;
  if (layer.interface_delta == false) {
    count += interface_components;
  }
  if ((layer.substrate_class == MaterialClass::Dielectric) && (layer.substrate_delta == false)) {
    count += interface_components;
  }
  return count;
}

ETX_SHARED_INLINE float bsdf_coating_transmission_component_probability(uint32_t component, uint32_t count) {
  const uint32_t cells = 8388608u;
  const uint32_t begin = (component * cells + count - 1u) / count;
  const uint32_t end = ((component + 1u) * cells + count - 1u) / count;
  return float(end - begin) / float(cells);
}

ETX_SHARED_INLINE float bsdf_coating_transmission_vndf_pdf(ETX_IN(float3, incoming), ETX_IN(float3, outgoing), ETX_IN(float2, alpha), float eta) {
  const bool reflection = outgoing.z > 0.0f;
  const float3 sum = reflection ? (incoming + outgoing) : (incoming + outgoing * eta);
  float3 normal;
  if (bsdf_external_normalize_half_vector(sum, normal) == false) {
    return 0.0f;
  }
  if (normal.z < 0.0f) {
    normal = -normal;
  }
  const float incident_cosine = dot(incoming, normal);
  const float outgoing_cosine = dot(outgoing, normal);
  if ((incident_cosine <= 0.0f) || (reflection ? (outgoing_cosine <= 0.0f) : (outgoing_cosine >= 0.0f))) {
    return 0.0f;
  }
  const float normal_pdf = bsdf_energy_compensated_vndf_pdf(incoming, normal, alpha);
  if (reflection) {
    return normal_pdf / (4.0f * outgoing_cosine);
  }
  const float denominator = incident_cosine + eta * outgoing_cosine;
  return denominator == 0.0f ? 0.0f : normal_pdf * (eta * eta) * abs(outgoing_cosine) / (denominator * denominator);
}

ETX_SHARED_INLINE uint32_t bsdf_coating_transmission_required_event(ETX_IN(BSDFCoatingWalk, state)) {
  return ((state.sample_target == BSDFCoatingWalk::Entrance) || (state.sample_target == BSDFCoatingWalk::ReverseExit)) ? BSDFSample::Transmission : BSDFSample::Reflection;
}

ETX_SHARED_INLINE float bsdf_coating_transmission_conditioned_base_probability(ETX_IN(SpectralQuery, spect), ETX_IN(BSDFEnergyCompensatedDielectricBranchPair, pair)) {
  return bsdf_external_component_proposal(saturate(spectral_response_monochromatic(bsdf_energy_compensated_dielectric_branch_pair_albedo(spect, pair))));
}

ETX_SHARED_INLINE float bsdf_coating_transmission_conditioned_pdf_prepared(ETX_IN(SpectralQuery, spect), ETX_IN(BSDFEnergyCompensatedPreparedMaterial, prepared),
  ETX_IN(BSDFEnergyCompensatedDielectricBranchPair, pair), ETX_IN(float3, incoming_local), ETX_IN(float3, outgoing_local), uint32_t required_event) {
  const bool reflection = required_event == BSDFSample::Reflection;
  if (((incoming_local.z * outgoing_local.z) > 0.0f) != reflection) {
    return 0.0f;
  }
  const float sign = incoming_local.z > 0.0f ? 1.0f : -1.0f;
  const float3 incoming = incoming_local * sign;
  const float3 outgoing = outgoing_local * sign;
  const float eta = bsdf_energy_compensated_dielectric_continuation_eta(prepared.ext_ior, prepared.int_ior, incoming_local.z > 0.0f);
  const float2 alpha = prepared.supported ? float2(prepared.alpha, prepared.alpha) : prepared.roughness;
  const float base_probability = bsdf_coating_transmission_conditioned_base_probability(spect, pair);
  const bool matched_transmission = (reflection == false) && bsdf_dielectric_equal_eta(prepared.ext_ior, prepared.int_ior);
  const float base_pdf = matched_transmission ? 0.0f : bsdf_coating_transmission_vndf_pdf(incoming, outgoing, alpha, eta);
  return base_probability * base_pdf + (1.0f - base_probability) * abs(outgoing.z) * kInvPi;
}

ETX_SHARED_NOINLINE float bsdf_coating_transmission_conditioned_pdf(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFCoatedConductor, layer), bool top, ETX_IN(BSDFData, data),
  ETX_IN(float3, outgoing), uint32_t required_event, ETX_INOUT(Sampler, sampler)) {
  const float3 incoming_local = -data.w_i;
  if ((abs(incoming_local.z) <= kEpsilon) || (abs(outgoing.z) <= kEpsilon) || ((((incoming_local.z * outgoing.z) > 0.0f)) != (required_event == BSDFSample::Reflection))) {
    return 0.0f;
  }
  const Material material = bsdf_coating_transmission_interface(layer, top);
  if (material.cls == MaterialClass::Translucent) {
    return abs(outgoing.z) * kInvPi;
  }
  if (bsdf_coating_transmission_port_delta(layer, top)) {
    return 0.0f;
  }
  const BSDFEnergyCompensatedPreparedMaterial prepared = bsdf_energy_compensated_prepare_material(context, data.spectrum_sample, material, data.tex, sampler);
  const BSDFEnergyCompensatedDielectricBranchPair pair = bsdf_energy_compensated_dielectric_branch_pair_value(context, data.spectrum_sample, material, abs(incoming_local.z),
    prepared.alpha, incoming_local.z > 0.0f, prepared.interface_lut_value);
  return bsdf_coating_transmission_conditioned_pdf_prepared(data.spectrum_sample, prepared, pair, incoming_local, outgoing, required_event);
}

ETX_SHARED_INLINE float bsdf_coating_transmission_continuous_pdf(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFCoatedConductor, layer), ETX_IN(float3, outgoing)) {
  const float sign = bsdf_coating_transmission_entrance_top(layer) ? 1.0f : -1.0f;
  const float3 incoming = -layer.data.w_i * sign;
  const float3 local_outgoing = outgoing * sign;
  const float eta = bsdf_coating_transmission_eta(context, layer);
  const bool reflection = local_outgoing.z > 0.0f;
  const uint32_t count = bsdf_coating_transmission_proposal_count(layer, eta);
  float pdf = bsdf_coating_transmission_component_probability(reflection ? 0u : 1u, count) * abs(local_outgoing.z) * kInvPi;
  if (reflection || (eta != 1.0f)) {
    const uint32_t interface_components = eta != 1.0f ? 2u : 1u;
    uint32_t component = 2u + (reflection ? 0u : 1u);
    if (layer.interface_delta == false) {
      pdf += bsdf_coating_transmission_component_probability(component, count) * bsdf_coating_transmission_vndf_pdf(incoming, local_outgoing, layer.interface_roughness, eta);
      component += interface_components;
    }
    if ((layer.substrate_class == MaterialClass::Dielectric) && (layer.substrate_delta == false)) {
      pdf += bsdf_coating_transmission_component_probability(component, count) * bsdf_coating_transmission_vndf_pdf(incoming, local_outgoing, layer.substrate_roughness, eta);
    }
  }
  return pdf;
}

ETX_SHARED_INLINE float bsdf_coating_transmission_pdf(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFCoatedConductor, layer), ETX_IN(float3, outgoing)) {
  if ((abs(layer.data.w_i.z) <= kEpsilon) || (abs(outgoing.z) <= kEpsilon)) {
    return 0.0f;
  }
  return bsdf_coating_transmission_probabilities(context, layer).z * bsdf_coating_transmission_continuous_pdf(context, layer, outgoing);
}

ETX_SHARED_INLINE BSDFEval bsdf_coating_transmission_resolve_evaluation(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFCoatedConductor, layer), ETX_IN(float3, outgoing),
  ETX_IN(SpectralResponse, function)) {
  BSDFEval result = bsdf_eval_zero(layer.data.spectrum_sample);
  result.func = function;
  result.bsdf = spectral_response_mul(function, abs(outgoing.z));
  result.pdf = bsdf_coating_transmission_pdf(context, layer, outgoing);
  const bool reflection = (layer.data.w_i.z * outgoing.z) < 0.0f;
  result.eta = reflection ? 1.0f : bsdf_coating_transmission_eta(context, layer);
  result.properties = reflection ? BSDFSample::Reflection : (BSDFSample::Transmission | BSDFSample::MediumChanged);
  result.medium_index = reflection ? layer.data.current_medium : (outgoing.z < 0.0f ? layer.int_medium : layer.ext_medium);
  return result;
}

ETX_SHARED_NOINLINE BSDFCoatingProposal bsdf_coating_transmission_propose(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFCoatedConductor, layer), ETX_INOUT(Sampler, sampler)) {
  BSDFCoatingProposal result = ETX_ZERO(BSDFCoatingProposal);
  result.sample = bsdf_sample_zero(layer.data.spectrum_sample);
  if (abs(layer.data.w_i.z) <= kEpsilon) {
    return result;
  }
  const float3 probabilities = bsdf_coating_transmission_probabilities(context, layer);
  const float selector = bsdf_sampler_has_fixed(sampler) ? sampler.fixed_w : bsdf_sampler_next(sampler);
  if (selector < (probabilities.x + probabilities.y)) {
    const bool reflection = selector < probabilities.x;
    Sampler atom_sampler = sampler;
    bsdf_sampler_pop_fixed(atom_sampler);
    const BSDFCoatingAtomInterface atoms = bsdf_coating_transmission_atoms(context, layer, atom_sampler);
    sampler.seed = atom_sampler.seed;
    result.sample.properties = BSDFSample::Delta | (reflection ? BSDFSample::Reflection : (BSDFSample::Transmission | BSDFSample::MediumChanged));
    result.sample.pdf = reflection ? probabilities.x : probabilities.y;
    result.sample.eta = reflection ? 1.0f : bsdf_coating_transmission_eta(context, layer);
    SpectralResponse value = atoms.reflection;
    float3 outgoing = bsdf_coating_transmission_reflected_direction(layer.data.w_i);
    if (reflection == false) {
      value = atoms.transmission;
      const float eta = result.sample.eta;
      const float2 indices = bsdf_coating_transmission_indices(context, layer);
      const float transmitted_cosine = bsdf_external_transmitted_cosine(abs(layer.data.w_i.z), indices.x, indices.y);
      if (transmitted_cosine <= 0.0f) {
        result.sample = bsdf_sample_zero(layer.data.spectrum_sample);
        return result;
      }
      outgoing = float3(layer.data.w_i.x / eta, layer.data.w_i.y / eta, (layer.data.w_i.z < 0.0f ? -1.0f : 1.0f) * transmitted_cosine);
      if (layer.data.path_source == PathSource::Light) {
        value = spectral_response_div(value, eta * eta);
      }
      if (eta != 1.0f) {
        result.sample.properties |= BSDFSample::WavelengthDependentDirection;
      }
    }
    result.sample.w_o = normalize(local_frame_from_local(layer.frame, outgoing));
    result.sample.weight = spectral_response_div(value, result.sample.pdf);
    result.sample.medium_index = reflection ? layer.data.current_medium : (outgoing.z < 0.0f ? layer.int_medium : layer.ext_medium);
    return result;
  }
  if ((probabilities.z <= 0.0f) || (selector >= (probabilities.x + probabilities.y + probabilities.z))) {
    return result;
  }
  const float eta = bsdf_coating_transmission_eta(context, layer);
  const uint32_t count = bsdf_coating_transmission_proposal_count(layer, eta);
  const uint32_t component_cell = uint32_t(bsdf_sampler_next(sampler) * 8388608.0f);
  uint32_t component = (component_cell * count) / 8388608u;
  const float2 random = bsdf_sampler_has_fixed(sampler) ? float2(sampler.fixed_u, sampler.fixed_v) : bsdf_sampler_next_2d(sampler);
  float3 outgoing;
  const float sign = bsdf_coating_transmission_entrance_top(layer) ? 1.0f : -1.0f;
  if (component < 2u) {
    outgoing = sample_cosine_distribution(random, 1.0f);
    if (component == 1u) {
      outgoing.z = -outgoing.z;
    }
  } else {
    component -= 2u;
    const uint32_t interface_components = eta != 1.0f ? 2u : 1u;
    float2 alpha = layer.substrate_roughness;
    if (layer.interface_delta == false) {
      if (component < interface_components) {
        alpha = layer.interface_roughness;
      } else {
        component -= interface_components;
      }
    }
    const float3 incoming = -layer.data.w_i * sign;
    const float3 normal = bsdf_energy_compensated_sample_vndf_local(incoming, alpha, random);
    if (component == 0u) {
      outgoing = reflect(-incoming, normal);
      if (outgoing.z <= kEpsilon) {
        return result;
      }
    } else {
      const float2 indices = bsdf_coating_transmission_indices(context, layer);
      outgoing = bsdf_external_refract(incoming, normal, indices.x, indices.y);
      if ((dot(outgoing, outgoing) <= kEpsilon) || (outgoing.z >= -kEpsilon)) {
        return result;
      }
      result.sample.properties |= BSDFSample::WavelengthDependentDirection;
    }
  }
  result.outgoing = normalize(local_frame_from_local(layer.frame, outgoing * sign));
  result.kind = BSDFCoatingProposal::Direction;
  return result;
}

ETX_SHARED_INLINE BSDFCoatingWalk bsdf_coating_transmission_walk_begin(ETX_IN(BSDFCoatedConductor, layer), ETX_IN(float3, outgoing)) {
  BSDFCoatingWalk state = ETX_ZERO(BSDFCoatingWalk);
  state.result = spectral_response_zero(layer.data.spectrum_sample);
  state.throughput = spectral_response_zero(layer.data.spectrum_sample);
  state.reverse_exit = bsdf_sample_zero(layer.data.spectrum_sample);
  state.outgoing = outgoing;
  state.sample_target = BSDFCoatingWalk::ReverseExit;
  const bool direct_reflection = bsdf_coating_transmission_entrance_top(layer) == bsdf_coating_transmission_exit_top(state);
  state.operation = (direct_reflection && (bsdf_coating_transmission_port_delta(layer, bsdf_coating_transmission_entrance_top(layer)) == false)) ? BSDFCoatingWalk::Direct
                                                                                                                                                 : BSDFCoatingWalk::SampleBegin;
  return state;
}

ETX_SHARED_NOINLINE void bsdf_coating_transmission_direct(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFCoatedConductor, layer), ETX_INOUT(BSDFCoatingWalk, state),
  ETX_INOUT(Sampler, sampler)) {
  state.result = bsdf_coating_transmission_interface_evaluate(context, layer, bsdf_coating_transmission_entrance_top(layer), layer.data, state.outgoing, sampler).func;
  state.operation = BSDFCoatingWalk::SampleBegin;
}

ETX_SHARED_INLINE void bsdf_coating_transmission_after_sample(ETX_IN(BSDFCoatedConductor, layer), ETX_IN(BSDFSample, sample), ETX_INOUT(BSDFCoatingWalk, state)) {
  if (state.sample_target == BSDFCoatingWalk::ReverseExit) {
    state.reverse_exit = sample;
    if (bsdf_coating_transmission_entrance_top(layer) != bsdf_coating_transmission_exit_top(state)) {
      state.operation = BSDFCoatingWalk::EntranceEvaluate;
    } else {
      state.sample_target = BSDFCoatingWalk::Entrance;
      state.operation = BSDFCoatingWalk::SampleBegin;
    }
    return;
  }
  const bool entrance = state.sample_target == BSDFCoatingWalk::Entrance;
  const uint32_t required_event = entrance ? BSDFSample::Transmission : BSDFSample::Reflection;
  if ((bsdf_sample_valid(sample) == false) || ((sample.properties & required_event) == 0u)) {
    state.operation = BSDFCoatingWalk::Complete;
    return;
  }
  if (entrance) {
    state.throughput = sample.weight;
    state.height = bsdf_coating_transmission_entrance_top(layer) ? layer.thickness : 0.0f;
  } else {
    state.throughput = spectral_response_mul(state.throughput, sample.weight);
  }
  state.direction = sample.w_o;
  state.continuation_pdf = sample.pdf;
  state.continuation_delta = (sample.properties & BSDFSample::Delta) != 0u;
  state.exit_from_phase = false;
  const bool toward_exit = (state.direction.z > 0.0f) == bsdf_coating_transmission_exit_top(state);
  state.operation =
    (toward_exit && (bsdf_coating_transmission_port_delta(layer, bsdf_coating_transmission_exit_top(state)) == false)) ? BSDFCoatingWalk::ExitEvaluate : BSDFCoatingWalk::Advance;
}

ETX_SHARED_NOINLINE BSDFEnergyCompensatedDielectricSampleState bsdf_coating_transmission_sample_interface_begin(ETX_IN(BSDFResourceContext, context),
  ETX_IN(BSDFCoatedConductor, layer), bool top, ETX_IN(BSDFData, data), uint32_t required_event, ETX_INOUT(Sampler, sampler)) {
  BSDFEnergyCompensatedDielectricSampleState state = ETX_ZERO(BSDFEnergyCompensatedDielectricSampleState);
  state.proposal.sample = bsdf_sample_zero(data.spectrum_sample);
  const Material material = bsdf_coating_transmission_interface(layer, top);
  const bool reflection = required_event == BSDFSample::Reflection;
  const float3 incoming_local = -data.w_i;
  if (abs(incoming_local.z) <= kEpsilon) {
    return state;
  }
  if (bsdf_coating_transmission_port_delta(layer, top)) {
    state.stage = BSDFEnergyCompensatedDielectricSampleState::Delta;
    return state;
  }
  const float sign = incoming_local.z > 0.0f ? 1.0f : -1.0f;
  const bool fixed = bsdf_sampler_has_fixed(sampler);
  const float2 random = fixed ? float2(sampler.fixed_u, sampler.fixed_v) : bsdf_sampler_next_2d(sampler);
  if (material.cls == MaterialClass::Translucent) {
    const float3 direction = sample_cosine_distribution(random, 1.0f) * (reflection ? sign : -sign);
    const BSDFEval evaluation = bsdf_translucent_evaluate(context, data, direction, material, sampler);
    BSDFSample sample = bsdf_sample_zero(data.spectrum_sample);
    const float pdf = abs(direction.z) * kInvPi;
    if ((pdf > 0.0f) && (spectral_response_is_zero(evaluation.bsdf) == false)) {
      sample.w_o = direction;
      sample.pdf = pdf;
      sample.weight = spectral_response_div(evaluation.bsdf, pdf);
      sample.eta = 1.0f;
      sample.properties = evaluation.properties;
      sample.medium_index = evaluation.medium_index;
    }
    state.proposal.sample = sample;
    return state;
  }
  state.w_i_local = incoming_local;
  state.prepared = bsdf_energy_compensated_prepare_material(context, data.spectrum_sample, material, data.tex, sampler);
  const BSDFEnergyCompensatedPreparedMaterial prepared = state.prepared;
  state.incident_pair = bsdf_energy_compensated_dielectric_branch_pair_value(context, data.spectrum_sample, material, abs(incoming_local.z), prepared.alpha,
    incoming_local.z > 0.0f, prepared.interface_lut_value);
  const float base_probability = bsdf_coating_transmission_conditioned_base_probability(data.spectrum_sample, state.incident_pair);
  const float selector = fixed ? sampler.fixed_w : bsdf_sampler_next(sampler);
  float3 outgoing;
  if (selector < base_probability) {
    const bool matching_media = bsdf_dielectric_equal_eta(prepared.ext_ior, prepared.int_ior);
    if ((reflection == false) && matching_media) {
      const BSDFCoatingAtomInterface atom = bsdf_coating_transmission_atom_interface(context, layer, top, data.w_i, sampler);
      BSDFSample sample = bsdf_sample_zero(data.spectrum_sample);
      sample.w_o = data.w_i;
      sample.pdf = base_probability;
      sample.weight = spectral_response_div(atom.transmission, base_probability);
      sample.eta = 1.0f;
      sample.properties = BSDFSample::Delta | BSDFSample::Transmission | BSDFSample::MediumChanged;
      sample.medium_index = incoming_local.z > 0.0f ? material.int_medium : material.ext_medium;
      state.proposal.sample = sample;
      return state;
    }
    const float2 alpha = prepared.supported ? float2(prepared.alpha, prepared.alpha) : prepared.roughness;
    const float3 incoming = incoming_local * sign;
    const float3 normal = bsdf_energy_compensated_sample_vndf_local(incoming, alpha, random);
    if (reflection) {
      outgoing = sign * reflect(-incoming, normal);
    } else {
      RefractiveIndexSample incident = prepared.ext_ior;
      RefractiveIndexSample transmitted = prepared.int_ior;
      if (incoming_local.z < 0.0f) {
        incident = prepared.int_ior;
        transmitted = prepared.ext_ior;
      }
      if (bsdf_external_transmission_possible(dot(incoming, normal), incident, transmitted) == false) {
        return state;
      }
      outgoing = sign * normalize(bsdf_external_refract(incoming, normal, incident, transmitted));
    }
  } else {
    outgoing = sample_cosine_distribution(random, 1.0f) * (reflection ? sign : -sign);
  }
  if ((abs(outgoing.z) <= kEpsilon) || ((((incoming_local.z * outgoing.z) > 0.0f)) != reflection)) {
    return state;
  }
  state.proposal.kind = BSDFEnergyCompensatedDielectricProposal::Direction;
  state.proposal.local_w_o = outgoing;
  state.proposal.wavelength_dependent_direction = (reflection == false) && (selector < base_probability);
  state.stage = BSDFEnergyCompensatedDielectricSampleState::Resolve;
  return state;
}

ETX_SHARED_NOINLINE BSDFSample bsdf_coating_transmission_sample_interface_resolve(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFCoatedConductor, layer), bool top,
  ETX_IN(BSDFData, data), uint32_t required_event, ETX_IN(BSDFEnergyCompensatedDielectricSampleState, state)) {
  const Material material = bsdf_coating_transmission_interface(layer, top);
  const float pdf =
    bsdf_coating_transmission_conditioned_pdf_prepared(data.spectrum_sample, state.prepared, state.incident_pair, state.w_i_local, state.proposal.local_w_o, required_event);
  BSDFSample sample = bsdf_sample_zero(data.spectrum_sample);
  if (pdf > 0.0f) {
    const LocalFrame frame = bsdf_local_frame_make(data.tan, data.btn, data.nrm, true);
    sample = bsdf_energy_compensated_dielectric_resolve_with_pdf(context, data, bsdf_interface_material(material), state.prepared, frame, state.w_i_local, state.incident_pair,
      state.proposal.local_w_o, state.proposal.wavelength_dependent_direction, pdf);
  }
  return sample;
}

ETX_SHARED_NOINLINE void bsdf_coating_transmission_sample_begin(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFCoatedConductor, layer), ETX_INOUT(BSDFCoatingWalk, state),
  ETX_INOUT(Sampler, sampler)) {
  state.interface_sample = bsdf_coating_transmission_sample_interface_begin(context, layer, bsdf_coating_transmission_sample_top(layer, state),
    bsdf_coating_transmission_sample_data(layer, state), bsdf_coating_transmission_required_event(state), sampler);
  if (state.interface_sample.stage == BSDFEnergyCompensatedDielectricSampleState::Delta) {
    state.operation = BSDFCoatingWalk::SampleDelta;
    return;
  }
  if (state.interface_sample.stage == BSDFEnergyCompensatedDielectricSampleState::Resolve) {
    state.operation = BSDFCoatingWalk::SampleResolve;
    return;
  }
  bsdf_coating_transmission_after_sample(layer, state.interface_sample.proposal.sample, state);
}

ETX_SHARED_NOINLINE void bsdf_coating_transmission_sample_resolve(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFCoatedConductor, layer), ETX_INOUT(BSDFCoatingWalk, state)) {
  const BSDFSample sample = bsdf_coating_transmission_sample_interface_resolve(context, layer, bsdf_coating_transmission_sample_top(layer, state),
    bsdf_coating_transmission_sample_data(layer, state), bsdf_coating_transmission_required_event(state), state.interface_sample);
  bsdf_coating_transmission_after_sample(layer, sample, state);
}

ETX_SHARED_NOINLINE void bsdf_coating_transmission_sample_delta(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFCoatedConductor, layer), ETX_INOUT(BSDFCoatingWalk, state),
  ETX_INOUT(Sampler, sampler)) {
  const Material material = bsdf_coating_transmission_interface(layer, bsdf_coating_transmission_sample_top(layer, state));
  const BSDFSample sample =
    bsdf_dielectric_delta_sample_event(context, bsdf_coating_transmission_sample_data(layer, state), material, bsdf_coating_transmission_required_event(state), sampler);
  bsdf_coating_transmission_after_sample(layer, sample, state);
}

ETX_SHARED_NOINLINE void bsdf_coating_transmission_entrance_evaluate(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFCoatedConductor, layer), ETX_INOUT(BSDFCoatingWalk, state),
  ETX_INOUT(Sampler, sampler)) {
  if (bsdf_coating_transmission_reverse_valid(state) && (bsdf_coating_transmission_port_delta(layer, bsdf_coating_transmission_entrance_top(layer)) == false)) {
    const float3 direction = -state.reverse_exit.w_o;
    const BSDFEval entrance = bsdf_coating_transmission_interface_evaluate(context, layer, bsdf_coating_transmission_entrance_top(layer), layer.data, direction, sampler);
    const bool reverse_delta = (state.reverse_exit.properties & BSDFSample::Delta) != 0u;
    const float mis = reverse_delta
                        ? 1.0f
                        : bsdf_coating_mis(state.reverse_exit.pdf, bsdf_coating_transmission_conditioned_pdf(context, layer, bsdf_coating_transmission_entrance_top(layer),
                                                                     layer.data, direction, BSDFSample::Transmission, sampler));
    const SpectralResponse transmittance = bsdf_coating_transmittance(layer.extinction, layer.thickness / abs(direction.z));
    const SpectralResponse connection = spectral_response_mul(spectral_response_mul(entrance.func, transmittance), spectral_response_mul(state.reverse_exit.weight, mis));
    state.result = spectral_response_add(state.result, connection);
  }
  state.sample_target = BSDFCoatingWalk::Entrance;
  state.operation = BSDFCoatingWalk::SampleBegin;
}

ETX_SHARED_NOINLINE void bsdf_coating_transmission_advance(ETX_IN(BSDFCoatedConductor, layer), ETX_INOUT(BSDFCoatingWalk, state), ETX_INOUT(Sampler, sampler)) {
  if (spectral_response_is_zero(state.throughput)) {
    state.operation = BSDFCoatingWalk::Complete;
    return;
  }
  if (state.depth++ >= 4u) {
    const float survival = bsdf_external_continuation_probability(state.throughput);
    if ((survival <= 0.0f) || ((survival < 1.0f) && (bsdf_sampler_next(sampler) >= survival))) {
      state.operation = BSDFCoatingWalk::Complete;
      return;
    }
    state.throughput = spectral_response_div(state.throughput, survival);
  }
  if (abs(state.direction.z) <= kEpsilon) {
    state.operation = BSDFCoatingWalk::Complete;
    return;
  }
  const bool top = state.direction.z > 0.0f;
  const float boundary_height = top ? layer.thickness : 0.0f;
  const float boundary_distance = max(0.0f, (boundary_height - state.height) / state.direction.z);
  float distance = 0.0f;
  const bool scattered = bsdf_coating_free_flight(layer, boundary_distance, sampler, state.throughput, distance);
  state.height = clamp(state.height + state.direction.z * distance, 0.0f, layer.thickness);
  if (spectral_response_is_zero(state.throughput)) {
    state.operation = BSDFCoatingWalk::Complete;
    return;
  }
  if (scattered) {
    state.operation = BSDFCoatingWalk::Phase;
    return;
  }
  state.sample_target = top ? BSDFCoatingWalk::Top : BSDFCoatingWalk::Bottom;
  if (top == bsdf_coating_transmission_exit_top(state)) {
    state.operation = BSDFCoatingWalk::SampleBegin;
    return;
  }
  state.operation =
    ((bsdf_coating_transmission_port_delta(layer, top) == false) && bsdf_coating_transmission_reverse_valid(state)) ? BSDFCoatingWalk::BaseEvaluate : BSDFCoatingWalk::BaseSample;
}

ETX_SHARED_INLINE SpectralResponse bsdf_coating_transmission_phase_atom_escape(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFCoatedConductor, layer),
  ETX_IN(BSDFCoatingWalk, state), ETX_IN(float3, direction), ETX_INOUT(Sampler, sampler)) {
  const bool exit_top = bsdf_coating_transmission_exit_top(state);
  const float cosine = abs(direction.z);
  const float to_exit_height = exit_top ? (layer.thickness - state.height) : state.height;
  const float3 reflected = bsdf_coating_transmission_reflected_direction(direction);
  BSDFCoatingAtomInterface exit = ETX_ZERO(BSDFCoatingAtomInterface);
  BSDFCoatingAtomInterface other = ETX_ZERO(BSDFCoatingAtomInterface);
  exit.reflection = spectral_response_zero(layer.data.spectrum_sample);
  exit.transmission = spectral_response_zero(layer.data.spectrum_sample);
  other.reflection = spectral_response_zero(layer.data.spectrum_sample);
  other.transmission = spectral_response_zero(layer.data.spectrum_sample);
  if (bsdf_coating_transmission_port_delta(layer, exit_top)) {
    exit = bsdf_coating_transmission_atom_interface(context, layer, exit_top, direction, sampler);
  }
  if (bsdf_coating_transmission_port_delta(layer, exit_top == false)) {
    other = bsdf_coating_transmission_atom_interface(context, layer, exit_top == false, reflected, sampler);
  }
  const float direct_phase = medium_phase_shared_henyey_greenstein(state.direction, direction, layer.anisotropy);
  const float reflected_phase = medium_phase_shared_henyey_greenstein(state.direction, reflected, layer.anisotropy);
  const SpectralResponse direct = spectral_response_mul(bsdf_coating_transmittance(layer.extinction, to_exit_height / cosine), direct_phase);
  const SpectralResponse indirect = spectral_response_mul(other.reflection,
    spectral_response_mul(bsdf_coating_transmittance(layer.extinction, (2.0f * layer.thickness - to_exit_height) / cosine), reflected_phase));
  SpectralResponse denominator = spectral_response_make(layer.data.spectrum_sample, 1.0f);
  if (layer.interface_delta && layer.substrate_delta) {
    denominator = bsdf_coating_series_denominator(exit.reflection, exit.transmission, other.reflection, bsdf_coating_transmittance(layer.extinction, layer.thickness / cosine));
  }
  return spectral_response_mul(spectral_response_add(direct, indirect), bsdf_coating_series_ratio(state.reverse_exit.weight, denominator));
}

ETX_SHARED_NOINLINE void bsdf_coating_transmission_phase(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFCoatedConductor, layer), ETX_INOUT(BSDFCoatingWalk, state),
  ETX_INOUT(Sampler, sampler)) {
  if (bsdf_coating_transmission_reverse_valid(state)) {
    const float3 exit_direction = -state.reverse_exit.w_o;
    const bool reverse_delta = (state.reverse_exit.properties & BSDFSample::Delta) != 0u;
    SpectralResponse escape;
    if (reverse_delta) {
      escape = bsdf_coating_transmission_phase_atom_escape(context, layer, state, exit_direction, sampler);
    } else {
      const float phase = medium_phase_shared_henyey_greenstein(state.direction, exit_direction, layer.anisotropy);
      const float mis = bsdf_coating_mis(state.reverse_exit.pdf, phase);
      const float distance = (bsdf_coating_transmission_exit_top(state) ? (layer.thickness - state.height) : state.height) / abs(exit_direction.z);
      escape = spectral_response_mul(bsdf_coating_transmittance(layer.extinction, distance), spectral_response_mul(state.reverse_exit.weight, phase * mis));
    }
    state.result = spectral_response_add(state.result, spectral_response_mul(state.throughput, spectral_response_div(escape, abs(exit_direction.z))));
  }
  const float3 incident = state.direction;
  state.direction = medium_phase_shared_sample_henyey_greenstein(incident, layer.anisotropy, bsdf_sampler_next_2d(sampler));
  state.operation = BSDFCoatingWalk::Advance;
  if (((state.direction.z > 0.0f) == bsdf_coating_transmission_exit_top(state)) &&
      (bsdf_coating_transmission_port_delta(layer, bsdf_coating_transmission_exit_top(state)) == false)) {
    state.continuation_pdf = medium_phase_shared_henyey_greenstein(incident, state.direction, layer.anisotropy);
    state.continuation_delta = false;
    state.exit_from_phase = true;
    state.operation = BSDFCoatingWalk::ExitEvaluate;
  }
}

ETX_SHARED_NOINLINE void bsdf_coating_transmission_base_evaluate(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFCoatedConductor, layer), ETX_INOUT(BSDFCoatingWalk, state),
  ETX_INOUT(Sampler, sampler)) {
  BSDFData data = layer.data;
  data.w_i = state.direction;
  const float3 exit_direction = -state.reverse_exit.w_o;
  const bool top = state.sample_target == BSDFCoatingWalk::Top;
  const BSDFEval evaluation = bsdf_coating_transmission_interface_evaluate(context, layer, top, data, exit_direction, sampler);
  const bool reverse_delta = (state.reverse_exit.properties & BSDFSample::Delta) != 0u;
  const float mis =
    reverse_delta ? 1.0f
                  : bsdf_coating_mis(state.reverse_exit.pdf, bsdf_coating_transmission_conditioned_pdf(context, layer, top, data, exit_direction, BSDFSample::Reflection, sampler));
  const SpectralResponse transmittance = bsdf_coating_transmittance(layer.extinction, layer.thickness / abs(exit_direction.z));
  const SpectralResponse connection = spectral_response_mul(spectral_response_mul(evaluation.func, transmittance), spectral_response_mul(state.reverse_exit.weight, mis));
  state.result = spectral_response_add(state.result, spectral_response_mul(state.throughput, connection));
  state.operation = BSDFCoatingWalk::BaseSample;
}

ETX_SHARED_NOINLINE void bsdf_coating_transmission_base_sample(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFCoatedConductor, layer), ETX_INOUT(BSDFCoatingWalk, state),
  ETX_INOUT(Sampler, sampler)) {
  bsdf_coating_transmission_sample_begin(context, layer, state, sampler);
}

ETX_SHARED_NOINLINE void bsdf_coating_transmission_exit_evaluate(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFCoatedConductor, layer), ETX_INOUT(BSDFCoatingWalk, state),
  ETX_INOUT(Sampler, sampler)) {
  BSDFData data = layer.data;
  data.w_i = state.direction;
  state.exit_func = bsdf_coating_transmission_interface_evaluate(context, layer, bsdf_coating_transmission_exit_top(state), data, state.outgoing, sampler).func;
  state.operation = BSDFCoatingWalk::ExitPdf;
}

ETX_SHARED_NOINLINE void bsdf_coating_transmission_exit_pdf(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFCoatedConductor, layer), ETX_INOUT(BSDFCoatingWalk, state),
  ETX_INOUT(Sampler, sampler)) {
  BSDFData reverse_data = layer.data;
  reverse_data.w_i = -state.outgoing;
  reverse_data.path_source = (layer.data.path_source == PathSource::Light) ? PathSource::Camera : PathSource::Light;
  const float reverse_pdf =
    bsdf_coating_transmission_conditioned_pdf(context, layer, bsdf_coating_transmission_exit_top(state), reverse_data, -state.direction, BSDFSample::Transmission, sampler);
  const float mis = state.continuation_delta ? 1.0f : bsdf_coating_mis(state.continuation_pdf, reverse_pdf);
  const float distance = (bsdf_coating_transmission_exit_top(state) ? (layer.thickness - state.height) : state.height) / abs(state.direction.z);
  const SpectralResponse connection = spectral_response_mul(state.exit_func, spectral_response_mul(bsdf_coating_transmittance(layer.extinction, distance), mis));
  state.result = spectral_response_add(state.result, spectral_response_mul(state.throughput, connection));
  state.operation = BSDFCoatingWalk::Advance;
}
