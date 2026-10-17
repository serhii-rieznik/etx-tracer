#pragma once

#include "bsdf_coating_substrate_shared.hxx"
#include "bsdf_velvet_shared.hxx"
#include "medium_phase_shared.hxx"
#include "scene_math_shared.hxx"

#ifndef ETX_ENABLE_COATED_TRANSMISSION_SUBSTRATE
# define ETX_ENABLE_COATED_TRANSMISSION_SUBSTRATE 1
#endif

struct BSDFCoatingInterface {
  SpectralImage reflectance ETX_INIT({});
  SpectralImage scattering ETX_INIT({});
  SampledImage roughness ETX_INIT({});
  RefractiveIndex ext_ior ETX_INIT({});
  RefractiveIndex int_ior ETX_INIT({});
  Thinfilm thinfilm ETX_INIT({});
  CoherentCoating coherent_coating ETX_INIT({});
  uint32_t energy_compensation_interface_index ETX_INIT(kInvalidIndex);
  uint32_t thermal_conductor_image_index ETX_INIT(kInvalidIndex);
  float thermal_conductor_average_albedo ETX_INIT(0.0f);
};

ETX_SHARED_INLINE BSDFCoatingInterface bsdf_coating_interface_optics(ETX_IN(Material, material)) {
  BSDFCoatingInterface result = ETX_ZERO(BSDFCoatingInterface);
  result.reflectance = material.reflectance;
  result.scattering = material.scattering;
  result.roughness = material.roughness;
  result.ext_ior = material.ext_ior;
  result.int_ior = material.int_ior;
  result.thinfilm = material.thinfilm;
  result.coherent_coating = material.coherent_coating;
  result.energy_compensation_interface_index = material.energy_compensation_interface_index;
  result.thermal_conductor_image_index = material.thermal_conductor_image_index;
  result.thermal_conductor_average_albedo = material.thermal_conductor_average_albedo;
  return result;
}

ETX_SHARED_INLINE BSDFCoatingInterface bsdf_coating_top_interface_optics(ETX_IN(Material, material)) {
  BSDFCoatingInterface result = bsdf_coating_interface_optics(material);
  result.int_ior = material.coating.ior;
  result.roughness = material.coating.roughness;
  result.reflectance.spectrum_index = material.coating.unit_spectrum_index;
  result.reflectance.image_index = kInvalidIndex;
  result.scattering = result.reflectance;
  result.energy_compensation_interface_index = material.coating.energy_compensation_interface_index;
  return result;
}

ETX_SHARED_INLINE Material bsdf_coating_interface_material(ETX_IN(BSDFCoatingInterface, optics), uint32_t cls) {
  Material result = ETX_ZERO(Material);
  result.cls = cls;
  result.reflectance = optics.reflectance;
  result.scattering = optics.scattering;
  result.roughness = optics.roughness;
  result.ext_ior = optics.ext_ior;
  result.int_ior = optics.int_ior;
  result.energy_compensation_interface_index = optics.energy_compensation_interface_index;
  result.thermal_conductor_image_index = optics.thermal_conductor_image_index;
  result.thermal_conductor_average_albedo = optics.thermal_conductor_average_albedo;
  result.int_medium = kInvalidIndex;
  result.ext_medium = kInvalidIndex;
  result.thinfilm = optics.thinfilm;
  result.coherent_coating = optics.coherent_coating;
  return result;
}

struct BSDFCoatedConductor {
  BSDFCoatingInterface interface_material ETX_INIT({});
  BSDFCoatingInterface substrate_material ETX_INIT({});
  BSDFData data ETX_INIT({});
  LocalFrame frame ETX_INIT({});
  SpectralResponse extinction ETX_INIT({});
  SpectralResponse scattering ETX_INIT({});
  float2 interface_roughness ETX_INIT({});
  float2 substrate_roughness ETX_INIT({});
  uint32_t int_medium ETX_INIT(kInvalidIndex);
  uint32_t ext_medium ETX_INIT(kInvalidIndex);
  float thickness ETX_INIT(0.0f);
  float anisotropy ETX_INIT(0.0f);
  float emission_collimation ETX_INIT(0.0f);
  uint32_t thermal_spectrum_index ETX_INIT(kInvalidIndex);
  uint32_t substrate_class ETX_INIT(MaterialClass::Conductor);
  bool interface_delta ETX_INIT(false);
  bool substrate_delta ETX_INIT(false);
  bool source_two_sided ETX_INIT(false);
};

struct BSDFCoatedConductorMaterial {
  BSDFCoatingInterface substrate ETX_INIT({});
  Coating coating ETX_INIT({});
  uint32_t two_sided ETX_INIT(0u);
  uint32_t substrate_class ETX_INIT(MaterialClass::Conductor);
  uint32_t int_medium ETX_INIT(kInvalidIndex);
  uint32_t ext_medium ETX_INIT(kInvalidIndex);
  float emission_collimation ETX_INIT(0.0f);
};

ETX_SHARED_INLINE BSDFCoatedConductorMaterial bsdf_coated_conductor_material_parameters(ETX_IN(Material, material)) {
  BSDFCoatedConductorMaterial result = ETX_ZERO(BSDFCoatedConductorMaterial);
  result.substrate = bsdf_coating_interface_optics(material);
  result.coating = material.coating;
  result.two_sided = material.two_sided;
  result.substrate_class = material_coating_base_class(material);
  result.int_medium = material.int_medium;
  result.ext_medium = material.ext_medium;
  result.emission_collimation = material.emission_collimation;
  return result;
}

ETX_SHARED_INLINE Material bsdf_coated_conductor_material(ETX_IN(BSDFCoatedConductorMaterial, parameters)) {
  Material result = bsdf_coating_interface_material(parameters.substrate, MaterialClass::CoatedConductor);
  result.coating = parameters.coating;
  result.two_sided = parameters.two_sided;
  result.layer_base_class = parameters.substrate_class + 1u;
  result.int_medium = parameters.int_medium;
  result.ext_medium = parameters.ext_medium;
  result.emission_collimation = parameters.emission_collimation;
  return result;
}

ETX_SHARED_INLINE bool bsdf_coating_transmission_base(uint32_t cls) {
#if (ETX_CPP == 1) || ETX_ENABLE_COATED_TRANSMISSION_SUBSTRATE
  return (cls == MaterialClass::Dielectric) || (cls == MaterialClass::Translucent);
#else
  return false;
#endif
}

ETX_SHARED_INLINE bool bsdf_coating_transmission_enabled(ETX_IN(BSDFCoatedConductor, layer)) {
  return bsdf_coating_transmission_base(layer.substrate_class);
}

ETX_SHARED_INLINE bool bsdf_coating_interface_delta_at_query(ETX_IN(BSDFResourceContext, context), ETX_IN(SpectralQuery, spect), ETX_IN(Material, material), ETX_IN(float2, tex)) {
  const float2 roughness = bsdf_resource_evaluate_roughness(context, material, tex);
  if (max(roughness.x, roughness.y) <= kDeltaAlphaTreshold) {
    return true;
  }
  return (bsdf_resource_interface_has_layers(context, material, tex) == false) &&
         bsdf_fresnel_equal_indices(bsdf_resource_evaluate_refractive_index(context, material.ext_ior, spect),
           bsdf_resource_evaluate_refractive_index(context, material.int_ior, spect));
}

ETX_SHARED_INLINE BSDFCoatedConductor bsdf_coated_conductor_prepare(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(Material, material)) {
  BSDFCoatedConductor result = ETX_ZERO(BSDFCoatedConductor);
  result.frame = bsdf_data_get_normal_frame(data, material);
  if (bsdf_coating_transmission_base(material_coating_base_class(material))) {
    result.frame = bsdf_local_frame_make(data.tan, data.btn, data.nrm, true);
  }
  result.int_medium = material.int_medium;
  result.ext_medium = material.ext_medium;
  result.data = data;
  result.data.nrm = float3(0.0f, 0.0f, 1.0f);
  result.data.tan = float3(1.0f, 0.0f, 0.0f);
  result.data.btn = float3(0.0f, 1.0f, 0.0f);
  result.data.w_i = local_frame_to_local(result.frame, data.w_i);
  result.interface_material = bsdf_coating_top_interface_optics(material);
  const float interface_roughness_scale = bsdf_resource_evaluate_sampled_image(context, material.coating.roughness, data.tex, 1.0f);
  result.interface_roughness = max(float2(material.coating.roughness.value.x, material.coating.roughness.value.y) * interface_roughness_scale, float2(kEpsilon, kEpsilon));
  result.substrate_material = bsdf_coating_interface_optics(material_coating_substrate(material));
  result.substrate_class = material_coating_base_class(material);
  if (bsdf_coating_transmission_enabled(result)) {
    result.substrate_roughness = bsdf_resource_evaluate_roughness(context, material, data.tex);
  }
  result.emission_collimation = material.emission_collimation;
  result.source_two_sided = material.two_sided != 0u;
  result.thermal_spectrum_index = material.coating.thermal_spectrum_index;
  result.extinction = bsdf_resource_apply_image(context, data.spectrum_sample, material.coating.absorption, data.tex);
  result.scattering = bsdf_resource_apply_image(context, data.spectrum_sample, material.coating.scattering, data.tex);
  result.extinction = spectral_response_add(result.extinction, result.scattering);
  result.thickness = material.coating.thickness.value.x * bsdf_resource_evaluate_sampled_image(context, material.coating.thickness, data.tex, 1.0f);
  result.anisotropy = material.coating.anisotropy;
  result.interface_delta =
    bsdf_dielectric_energy_compensated_is_delta_with_context(context, bsdf_coating_interface_material(result.interface_material, MaterialClass::Dielectric), data.tex);
  result.substrate_delta = bsdf_coating_substrate_is_delta(context, bsdf_coating_interface_material(result.substrate_material, result.substrate_class), data.tex);
  if (bsdf_coating_transmission_enabled(result)) {
    result.interface_delta =
      bsdf_coating_interface_delta_at_query(context, data.spectrum_sample, bsdf_coating_interface_material(result.interface_material, MaterialClass::Dielectric), data.tex);
    result.substrate_delta =
      (result.substrate_class == MaterialClass::Dielectric) &&
      bsdf_coating_interface_delta_at_query(context, data.spectrum_sample, bsdf_coating_interface_material(result.substrate_material, result.substrate_class), data.tex);
  }
  return result;
}

ETX_SHARED_INLINE SpectralResponse bsdf_coating_transmittance(ETX_IN(SpectralResponse, extinction), float distance) {
  return spectral_response_exp(spectral_response_mul(extinction, -distance));
}

ETX_SHARED_INLINE float bsdf_coating_probability(ETX_IN(SpectralResponse, response)) {
  if (spectral_response_is_spectral(response)) {
    return response.value;
  }
  return (response.integrated.x + response.integrated.y + response.integrated.z) / 3.0f;
}

ETX_SHARED_INLINE float bsdf_coating_mis(float first, float second) {
  const float scale = max(first, second);
  const bool inactive = scale <= 0.0f;
  const float safe_scale = inactive ? 1.0f : scale;
#if ETX_CPP
  const float normalized_first = (inactive ? 0.0f : first) / safe_scale;
  const float normalized_second = (inactive ? 0.0f : second) / safe_scale;
#else
  const float normalized_first = select(inactive, 0.0f, first) / safe_scale;
  const float normalized_second = select(inactive, 0.0f, second) / safe_scale;
#endif
  const float a = normalized_first * normalized_first;
  const float b = normalized_second * normalized_second;
#if ETX_CPP
  return a / (inactive ? 1.0f : (a + b));
#else
  return a / select(inactive, 1.0f, a + b);
#endif
}

ETX_SHARED_INLINE SpectralResponse bsdf_coating_series_ratio(ETX_IN(SpectralResponse, numerator), ETX_IN(SpectralResponse, denominator)) {
  const SpectralQuery spect = spectral_response_as_query(numerator);
  if (spectral_response_is_spectral(numerator)) {
    return spectral_response_make(spect, (denominator.value > 0.0f) ? (numerator.value / denominator.value) : 0.0f);
  }
  const float3 a = numerator.integrated;
  const float3 b = denominator.integrated;
  return spectral_response_make(spect, float3((b.x > 0.0f) ? (a.x / b.x) : 0.0f, (b.y > 0.0f) ? (a.y / b.y) : 0.0f, (b.z > 0.0f) ? (a.z / b.z) : 0.0f));
}

ETX_SHARED_INLINE SpectralResponse bsdf_coating_series_denominator(ETX_IN(SpectralResponse, top_reflection), ETX_IN(SpectralResponse, top_transmission),
  ETX_IN(SpectralResponse, base_reflection), ETX_IN(SpectralResponse, transmittance)) {
  const SpectralResponse one = spectral_response_make(spectral_response_as_query(top_reflection), 1.0f);
  const SpectralResponse top_absorption = spectral_response_max(spectral_response_sub(spectral_response_sub(one, top_reflection), top_transmission), 0.0f);
  const SpectralResponse base_absorption = spectral_response_sub(one, base_reflection);
  const SpectralResponse volume_loss = spectral_response_mul(spectral_response_sub(one, transmittance), spectral_response_add(one, transmittance));
  const SpectralResponse round_trip_loss = spectral_response_add(base_absorption, spectral_response_mul(base_reflection, volume_loss));
  // Resolve escape and absorption separately: 1 - R rounds to zero before a tiny T does.
  return spectral_response_add(spectral_response_add(top_transmission, top_absorption), spectral_response_mul(top_reflection, round_trip_loss));
}

// Smooth, uncollided paths share one direction; sum every internal reflection analytically.
ETX_SHARED_NOINLINE SpectralResponse bsdf_coated_conductor_delta_weight(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFCoatedConductor, layer), bool substrate_absorption) {
  const SpectralQuery spect = layer.data.spectrum_sample;
  const SpectralResponse one = spectral_response_make(spect, 1.0f);
  const float cosine = -layer.data.w_i.z;
  const RefractiveIndexSample exterior = bsdf_resource_evaluate_refractive_index(context, layer.interface_material.ext_ior, spect);
  const RefractiveIndexSample coating = bsdf_resource_evaluate_refractive_index(context, layer.interface_material.int_ior, spect);
  Sampler sampler = ETX_ZERO(Sampler);
  const ThinfilmEval film =
    bsdf_resource_evaluate_interface_optics(context, spect, bsdf_coating_interface_material(layer.interface_material, MaterialClass::Dielectric), layer.data.tex, sampler);
  const SpectralResponse f = bsdf_fresnel_calculate(spect, cosine, exterior, coating, film);
  if (layer.substrate_delta == false) {
    if (substrate_absorption) {
      return spectral_response_make(spect, 0.0f);
    }
    return f;
  }
  const float cosine_inside = bsdf_external_transmitted_cosine(cosine, spectral_response_monochromatic(exterior.eta), spectral_response_monochromatic(coating.eta));
  if (cosine_inside <= 0.0f) {
    if (substrate_absorption) {
      return spectral_response_make(spect, 0.0f);
    }
    return f;
  }
  const ThinfilmEval reverse_optics = bsdf_fresnel_orient_coherent_interface(film, false);
  const SpectralResponse reverse_reflection = bsdf_fresnel_calculate(spect, cosine_inside, coating, exterior, reverse_optics);
  const SpectralResponse transmission = bsdf_fresnel_transmission_from_reflection(spect, cosine, exterior, coating, film, f);
  const SpectralResponse reverse_transmission = bsdf_fresnel_transmission_from_reflection(spect, cosine_inside, coating, exterior, reverse_optics, reverse_reflection);
  const SpectralResponse r =
    bsdf_coating_substrate_specular_reflectance(context, spect, bsdf_coating_interface_material(layer.substrate_material, layer.substrate_class), layer.data.tex, cosine_inside);
  const SpectralResponse tr = bsdf_coating_transmittance(layer.extinction, layer.thickness / cosine_inside);
  const SpectralResponse round_trip = spectral_response_mul(r, spectral_response_mul(tr, tr));
  const SpectralResponse base_absorption = spectral_response_sub(one, r);
  const SpectralResponse denominator = bsdf_coating_series_denominator(reverse_reflection, reverse_transmission, r, tr);
  if (substrate_absorption) {
    return spectral_response_mul(spectral_response_mul(transmission, tr), bsdf_coating_series_ratio(base_absorption, denominator));
  }
  // A closed channel cannot be entered; its internal contribution is zero.
  const SpectralResponse escape = bsdf_coating_series_ratio(reverse_transmission, denominator);
  return spectral_response_add(f, spectral_response_mul(spectral_response_mul(transmission, round_trip), escape));
}

ETX_SHARED_INLINE float bsdf_coated_conductor_delta_probability(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFCoatedConductor, layer)) {
  if (layer.interface_delta == false) {
    return 0.0f;
  }
  if (layer.substrate_delta && spectral_response_is_zero(layer.scattering)) {
    return 1.0f;
  }
  return saturate(spectral_response_monochromatic(bsdf_coated_conductor_delta_weight(context, layer, false)));
}

ETX_SHARED_INLINE bool bsdf_coating_free_flight(ETX_IN(BSDFCoatedConductor, layer), float boundary_distance, ETX_INOUT(Sampler, sampler), ETX_INOUT(SpectralResponse, throughput),
  ETX_OUT(float, distance)) {
  distance = boundary_distance;
  if (spectral_response_is_zero(layer.scattering)) {
    throughput = spectral_response_mul(throughput, bsdf_coating_transmittance(layer.extinction, distance));
    return false;
  }
  float coefficient = layer.extinction.value;
  if (spectral_response_is_spectral(layer.extinction) == false) {
    const uint32_t channel = min(2u, uint32_t(bsdf_sampler_next(sampler) * 3.0f));
    coefficient = (channel == 0u) ? layer.extinction.integrated.x : ((channel == 1u) ? layer.extinction.integrated.y : layer.extinction.integrated.z);
  }
  const float sampled_distance = (coefficient > 0.0f) ? (-log(1.0f - bsdf_sampler_next(sampler)) / coefficient) : kMaxFloat;
  const bool scattered = sampled_distance < boundary_distance;
  distance = scattered ? sampled_distance : boundary_distance;
  if (spectral_response_is_spectral(layer.extinction)) {
    if (scattered) {
      throughput = spectral_response_mul(throughput, layer.scattering.value / coefficient);
    }
    return scattered;
  }
  const SpectralResponse tr = bsdf_coating_transmittance(layer.extinction, distance);
  SpectralResponse density = tr;
  SpectralResponse attenuation = tr;
  if (scattered) {
    density = spectral_response_mul(tr, layer.extinction);
    attenuation = spectral_response_mul(tr, layer.scattering);
  }
  const float pdf = bsdf_coating_probability(density);
  if (pdf > 0.0f) {
    throughput = spectral_response_mul(throughput, spectral_response_div(attenuation, pdf));
  } else {
    throughput = spectral_response_make(layer.data.spectrum_sample, 0.0f);
  }
  return scattered;
}

// Sum collision-to-exit paths through smooth interfaces, including substrate reflections.
ETX_SHARED_INLINE SpectralResponse bsdf_coating_specular_phase_escape(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFCoatedConductor, layer), ETX_IN(float3, incident),
  ETX_IN(float3, outgoing), float height, ETX_IN(SpectralResponse, exit_weight)) {
  const SpectralQuery spect = layer.data.spectrum_sample;
  const RefractiveIndexSample exterior = bsdf_resource_evaluate_refractive_index(context, layer.interface_material.ext_ior, spect);
  const RefractiveIndexSample coating = bsdf_resource_evaluate_refractive_index(context, layer.interface_material.int_ior, spect);
  Sampler sampler = ETX_ZERO(Sampler);
  const ThinfilmEval film = bsdf_fresnel_orient_coherent_interface(
    bsdf_resource_evaluate_interface_optics(context, spect, bsdf_coating_interface_material(layer.interface_material, MaterialClass::Dielectric), layer.data.tex, sampler), false);
  const SpectralResponse top = bsdf_fresnel_calculate(spect, outgoing.z, coating, exterior, film);
  const SpectralResponse top_transmission = bsdf_fresnel_transmission_calculate(spect, outgoing.z, coating, exterior, film);
  const SpectralResponse base =
    bsdf_coating_substrate_specular_reflectance(context, spect, bsdf_coating_interface_material(layer.substrate_material, layer.substrate_class), layer.data.tex, outgoing.z);
  const float direct_phase = medium_phase_shared_henyey_greenstein(incident, outgoing, layer.anisotropy);
  const float reflected_phase = medium_phase_shared_henyey_greenstein(incident, float3(outgoing.x, outgoing.y, -outgoing.z), layer.anisotropy);
  const SpectralResponse direct = spectral_response_mul(bsdf_coating_transmittance(layer.extinction, (layer.thickness - height) / outgoing.z), direct_phase);
  const SpectralResponse reflected =
    spectral_response_mul(base, spectral_response_mul(bsdf_coating_transmittance(layer.extinction, (layer.thickness + height) / outgoing.z), reflected_phase));
  const SpectralResponse denominator = bsdf_coating_series_denominator(top, top_transmission, base, bsdf_coating_transmittance(layer.extinction, layer.thickness / outgoing.z));
  return spectral_response_mul(spectral_response_add(direct, reflected), bsdf_coating_series_ratio(exit_weight, denominator));
}

struct BSDFCoatingWalk {
  enum : uint32_t {
    Complete,
    Direct,
    SampleBegin,
    SampleDelta,
    SampleResolve,
    Advance,
    Phase,
    BaseEvaluate,
    BaseSample,
    ExitEvaluate,
    ExitPdf,
    VelvetBaseEvaluate,
    VelvetBaseSample,
    VelvetProposal,
    EntranceEvaluate
  };
  enum : uint32_t { Entrance, ReverseExit, Top, Bottom };
  BSDFEnergyCompensatedDielectricSampleState interface_sample ETX_INIT({});
  BSDFSample reverse_exit ETX_INIT({});
  SpectralResponse result ETX_INIT({});
  SpectralResponse throughput ETX_INIT({});
  SpectralResponse exit_func ETX_INIT({});
  float3 outgoing ETX_INIT({});
  float3 direction ETX_INIT({});
  float height ETX_INIT(0.0f);
  float continuation_pdf ETX_INIT(0.0f);
  uint32_t depth ETX_INIT(0u);
  uint32_t operation ETX_INIT(Complete);
  uint32_t sample_target ETX_INIT(Entrance);
  bool exit_from_phase ETX_INIT(false);
#if (ETX_CPP == 1) || ETX_ENABLE_COATED_VELVET_SUBSTRATE
  BSDFVelvetWalk velvet ETX_INIT({});
  float3 velvet_outgoing ETX_INIT({});
#endif
  bool continuation_delta ETX_INIT(false);
};

struct BSDFCoatingProposal {
  enum : uint32_t { Complete, Direction, Velvet };
  BSDFSample sample ETX_INIT({});
  float3 outgoing ETX_INIT({});
  uint32_t kind ETX_INIT(Complete);
};

#include "bsdf_coating_transmission_shared.hxx"

ETX_SHARED_INLINE BSDFCoatingWalk bsdf_coating_walk_begin(ETX_IN(BSDFCoatedConductor, layer), ETX_IN(float3, outgoing)) {
  if (bsdf_coating_transmission_enabled(layer)) {
    return bsdf_coating_transmission_walk_begin(layer, outgoing);
  }
  BSDFCoatingWalk state = ETX_ZERO(BSDFCoatingWalk);
  state.result = spectral_response_make(layer.data.spectrum_sample, 0.0f);
  state.outgoing = outgoing;
  state.operation = layer.interface_delta ? BSDFCoatingWalk::SampleBegin : BSDFCoatingWalk::Direct;
  return state;
}

ETX_SHARED_INLINE BSDFData bsdf_coating_walk_reverse_exit_data(ETX_IN(BSDFCoatedConductor, layer), ETX_IN(BSDFCoatingWalk, state)) {
  BSDFData data = layer.data;
  data.w_i = -state.outgoing;
  data.path_source = (layer.data.path_source == PathSource::Light) ? PathSource::Camera : PathSource::Light;
  return data;
}

ETX_SHARED_INLINE BSDFData bsdf_coating_walk_sample_data(ETX_IN(BSDFCoatedConductor, layer), ETX_IN(BSDFCoatingWalk, state)) {
  if (state.sample_target == BSDFCoatingWalk::ReverseExit) {
    return bsdf_coating_walk_reverse_exit_data(layer, state);
  }
  BSDFData data = layer.data;
  if (state.sample_target == BSDFCoatingWalk::Top) {
    data.w_i = state.direction;
  }
  return data;
}

ETX_SHARED_INLINE void bsdf_coating_walk_accept_sample(ETX_IN(BSDFCoatedConductor, layer), ETX_IN(BSDFSample, sample), ETX_INOUT(BSDFCoatingWalk, state)) {
  if (state.sample_target == BSDFCoatingWalk::Entrance) {
    if ((bsdf_sample_valid(sample) == false) || ((sample.properties & BSDFSample::Transmission) == 0u)) {
      state.operation = BSDFCoatingWalk::Complete;
      return;
    }
    state.throughput = sample.weight;
    state.direction = sample.w_o;
    state.height = layer.thickness;
    state.sample_target = BSDFCoatingWalk::ReverseExit;
    state.operation = BSDFCoatingWalk::SampleBegin;
    return;
  }
  if (state.sample_target == BSDFCoatingWalk::ReverseExit) {
    // A null reverse proposal still permits forward-sampled connections.
    state.reverse_exit = sample;
    state.operation = BSDFCoatingWalk::Advance;
    return;
  }
  if ((bsdf_sample_valid(sample) == false) || ((sample.properties & BSDFSample::Reflection) == 0u)) {
    state.operation = BSDFCoatingWalk::Complete;
    return;
  }
  state.throughput = spectral_response_mul(state.throughput, sample.weight);
  state.direction = sample.w_o;
  state.operation = BSDFCoatingWalk::Advance;
}

ETX_SHARED_INLINE SpectralResponse bsdf_coating_direct_function(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, local_data), ETX_IN(BSDFCoatingInterface, optics),
  ETX_IN(float3, local_outgoing), ETX_INOUT(Sampler, sampler)) {
  const Material interface_material = bsdf_coating_interface_material(optics, MaterialClass::Dielectric);
  return bsdf_dielectric_energy_compensated_evaluate(context, local_data, local_outgoing, interface_material, sampler).func;
}

ETX_SHARED_NOINLINE void bsdf_coating_walk_direct(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFCoatedConductor, layer), ETX_INOUT(BSDFCoatingWalk, state),
  ETX_INOUT(Sampler, sampler)) {
  if (bsdf_coating_transmission_enabled(layer)) {
    bsdf_coating_transmission_direct(context, layer, state, sampler);
    return;
  }
  state.result = bsdf_coating_direct_function(context, layer.data, layer.interface_material, state.outgoing, sampler);
  state.operation = BSDFCoatingWalk::SampleBegin;
}

ETX_SHARED_NOINLINE void bsdf_coating_walk_sample_begin(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFCoatedConductor, layer), ETX_INOUT(BSDFCoatingWalk, state),
  ETX_INOUT(Sampler, sampler)) {
  if (bsdf_coating_transmission_enabled(layer)) {
    bsdf_coating_transmission_sample_begin(context, layer, state, sampler);
    return;
  }
  const BSDFData data = bsdf_coating_walk_sample_data(layer, state);
  state.interface_sample = bsdf_coating_transmission_sample_interface_begin(context, layer, true, data, bsdf_coating_transmission_required_event(state), sampler);
  if (state.interface_sample.stage == BSDFEnergyCompensatedDielectricSampleState::Delta) {
    state.operation = BSDFCoatingWalk::SampleDelta;
    return;
  }
  if (state.interface_sample.stage == BSDFEnergyCompensatedDielectricSampleState::Resolve) {
    state.operation = BSDFCoatingWalk::SampleResolve;
    return;
  }
  bsdf_coating_walk_accept_sample(layer, state.interface_sample.proposal.sample, state);
}

ETX_SHARED_NOINLINE void bsdf_coating_walk_sample_delta(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFCoatedConductor, layer), ETX_INOUT(BSDFCoatingWalk, state),
  ETX_INOUT(Sampler, sampler)) {
  if (bsdf_coating_transmission_enabled(layer)) {
    bsdf_coating_transmission_sample_delta(context, layer, state, sampler);
    return;
  }
  const Material interface_material = bsdf_coating_interface_material(layer.interface_material, MaterialClass::Dielectric);
  const BSDFSample sample =
    bsdf_dielectric_delta_sample_event(context, bsdf_coating_walk_sample_data(layer, state), interface_material, bsdf_coating_transmission_required_event(state), sampler);
  bsdf_coating_walk_accept_sample(layer, sample, state);
}

ETX_SHARED_NOINLINE void bsdf_coating_walk_sample_resolve(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFCoatedConductor, layer), ETX_INOUT(BSDFCoatingWalk, state)) {
  if (bsdf_coating_transmission_enabled(layer)) {
    bsdf_coating_transmission_sample_resolve(context, layer, state);
    return;
  }
  const BSDFSample sample = bsdf_coating_transmission_sample_interface_resolve(context, layer, true, bsdf_coating_walk_sample_data(layer, state),
    bsdf_coating_transmission_required_event(state), state.interface_sample);
  bsdf_coating_walk_accept_sample(layer, sample, state);
}

ETX_SHARED_INLINE bool bsdf_coating_walk_reverse_exit_valid(ETX_IN(BSDFCoatingWalk, state)) {
  return bsdf_sample_valid(state.reverse_exit) && ((state.reverse_exit.properties & BSDFSample::Transmission) != 0u);
}

ETX_SHARED_NOINLINE void bsdf_coating_walk_advance(ETX_IN(BSDFCoatedConductor, layer), ETX_INOUT(BSDFCoatingWalk, state), ETX_INOUT(Sampler, sampler)) {
  if (bsdf_coating_transmission_enabled(layer)) {
    bsdf_coating_transmission_advance(layer, state, sampler);
    return;
  }
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
  const bool hit_top = state.direction.z > 0.0f;
  const float boundary_height = hit_top ? layer.thickness : 0.0f;
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
  if (hit_top) {
    state.sample_target = BSDFCoatingWalk::Top;
    state.operation = BSDFCoatingWalk::SampleBegin;
    return;
  }
  state.operation = ((layer.substrate_delta == false) && bsdf_coating_walk_reverse_exit_valid(state)) ? BSDFCoatingWalk::BaseEvaluate : BSDFCoatingWalk::BaseSample;
}

ETX_SHARED_NOINLINE void bsdf_coating_walk_phase(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFCoatedConductor, layer), ETX_INOUT(BSDFCoatingWalk, state),
  ETX_INOUT(Sampler, sampler)) {
  if (bsdf_coating_transmission_enabled(layer)) {
    bsdf_coating_transmission_phase(context, layer, state, sampler);
    return;
  }
  const float3 exit_direction = -state.reverse_exit.w_o;
  if (bsdf_coating_walk_reverse_exit_valid(state)) {
    const bool reverse_exit_delta = (state.reverse_exit.properties & BSDFSample::Delta) != 0u;
    const float phase = medium_phase_shared_henyey_greenstein(state.direction, exit_direction, layer.anisotropy);
    const float mis = reverse_exit_delta ? 1.0f : bsdf_coating_mis(state.reverse_exit.pdf, phase);
    SpectralResponse phase_escape;
    if (layer.interface_delta && layer.substrate_delta) {
      phase_escape = bsdf_coating_specular_phase_escape(context, layer, state.direction, exit_direction, state.height, state.reverse_exit.weight);
    } else {
      phase_escape = spectral_response_mul(spectral_response_mul(bsdf_coating_transmittance(layer.extinction, (layer.thickness - state.height) / exit_direction.z), phase * mis),
        state.reverse_exit.weight);
    }
    const SpectralResponse connection = spectral_response_div(phase_escape, exit_direction.z);
    state.result = spectral_response_add(state.result, spectral_response_mul(state.throughput, connection));
  }
  const float3 incident_direction = state.direction;
  state.direction = medium_phase_shared_sample_henyey_greenstein(incident_direction, layer.anisotropy, bsdf_sampler_next_2d(sampler));
  state.operation = BSDFCoatingWalk::Advance;
  if ((layer.interface_delta == false) && (state.direction.z > kEpsilon)) {
    state.continuation_pdf = medium_phase_shared_henyey_greenstein(incident_direction, state.direction, layer.anisotropy);
    state.exit_from_phase = true;
    state.operation = BSDFCoatingWalk::ExitEvaluate;
  }
}

ETX_SHARED_INLINE void bsdf_coating_walk_accept_base_evaluation(ETX_IN(BSDFCoatedConductor, layer), ETX_IN(BSDFEval, base), ETX_INOUT(BSDFCoatingWalk, state)) {
  const float3 exit_direction = -state.reverse_exit.w_o;
  const bool reverse_exit_delta = (state.reverse_exit.properties & BSDFSample::Delta) != 0u;
  const float mis = reverse_exit_delta ? 1.0f : bsdf_coating_mis(state.reverse_exit.pdf, base.pdf);
  const SpectralResponse connection = spectral_response_mul(base.func,
    spectral_response_mul(bsdf_coating_transmittance(layer.extinction, layer.thickness / exit_direction.z), spectral_response_mul(state.reverse_exit.weight, mis)));
  state.result = spectral_response_add(state.result, spectral_response_mul(state.throughput, connection));
  state.operation = BSDFCoatingWalk::BaseSample;
}

ETX_SHARED_INLINE void bsdf_coating_walk_accept_base_sample(ETX_IN(BSDFCoatedConductor, layer), ETX_IN(BSDFSample, base), ETX_INOUT(BSDFCoatingWalk, state)) {
  if (bsdf_sample_valid(base) == false) {
    state.operation = BSDFCoatingWalk::Complete;
    return;
  }
  state.throughput = spectral_response_mul(state.throughput, base.weight);
  state.direction = base.w_o;
  state.continuation_pdf = base.pdf;
  state.exit_from_phase = false;
  state.operation = layer.interface_delta ? BSDFCoatingWalk::Advance : BSDFCoatingWalk::ExitEvaluate;
}

#if (ETX_CPP == 1) || ETX_ENABLE_COATED_VELVET_SUBSTRATE
ETX_SHARED_INLINE void bsdf_coating_walk_begin_velvet(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(Material, material), ETX_IN(float3, outgoing),
  uint32_t operation, ETX_INOUT(BSDFCoatingWalk, state)) {
  const LocalFrame frame = bsdf_data_get_normal_frame(data, material);
  const float3 incoming = local_frame_to_local(frame, -data.w_i);
  const float3 local_outgoing = local_frame_to_local(frame, outgoing);
  state.velvet = bsdf_velvet_walk_begin_evaluation(context, data.spectrum_sample, bsdf_velvet_prepare(context, data, material), incoming, local_outgoing);
  state.velvet_outgoing = outgoing;
  state.operation = operation;
}

ETX_SHARED_INLINE BSDFEval bsdf_coating_walk_velvet_evaluation(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFCoatedConductor, layer), ETX_INOUT(BSDFCoatingWalk, state),
  ETX_INOUT(Sampler, sampler)) {
  BSDFData data = layer.data;
  if (state.operation != BSDFCoatingWalk::VelvetProposal) {
    data.w_i = state.direction;
  }
  const Material material = bsdf_coating_interface_material(layer.substrate_material, layer.substrate_class);
  if (state.velvet.operation != BSDFVelvetWalk::Complete) {
    Sampler walk_sampler = sampler;
    bsdf_sampler_pop_fixed(walk_sampler);
    bsdf_velvet_walk_evaluate_step(context, data.spectrum_sample, bsdf_velvet_prepare(context, data, material), state.velvet, walk_sampler);
    sampler.seed = walk_sampler.seed;
  }
  const LocalFrame frame = bsdf_data_get_normal_frame(data, material);
  return bsdf_velvet_evaluation_resolve(data, local_frame_to_local(frame, -data.w_i), state.velvet.outgoing, state.velvet.value);
}

ETX_SHARED_NOINLINE void bsdf_coating_walk_velvet(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFCoatedConductor, layer), ETX_INOUT(BSDFCoatingWalk, state),
  ETX_INOUT(Sampler, sampler)) {
  const BSDFEval evaluation = bsdf_coating_walk_velvet_evaluation(context, layer, state, sampler);
  if (state.velvet.operation != BSDFVelvetWalk::Complete) {
    return;
  }
  if (state.operation == BSDFCoatingWalk::VelvetBaseEvaluate) {
    bsdf_coating_walk_accept_base_evaluation(layer, evaluation, state);
  } else {
    bsdf_coating_walk_accept_base_sample(layer, bsdf_velvet_sample_resolve(layer.data.spectrum_sample, state.velvet_outgoing, evaluation), state);
  }
}
#endif

ETX_SHARED_NOINLINE void bsdf_coating_walk_base_evaluate(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFCoatedConductor, layer), ETX_INOUT(BSDFCoatingWalk, state),
  ETX_INOUT(Sampler, sampler)) {
  if (bsdf_coating_transmission_enabled(layer)) {
    bsdf_coating_transmission_base_evaluate(context, layer, state, sampler);
    return;
  }
  BSDFData data = layer.data;
  data.w_i = state.direction;
  const float3 exit_direction = -state.reverse_exit.w_o;
  const Material substrate_material = bsdf_coating_interface_material(layer.substrate_material, layer.substrate_class);
#if (ETX_CPP == 1) || ETX_ENABLE_COATED_VELVET_SUBSTRATE
  if (layer.substrate_class == MaterialClass::Velvet) {
    bsdf_coating_walk_begin_velvet(context, data, substrate_material, exit_direction, BSDFCoatingWalk::VelvetBaseEvaluate, state);
    return;
  }
#endif
  const BSDFEval base = bsdf_coating_substrate_evaluate_non_velvet(context, data, exit_direction, substrate_material, sampler);
  bsdf_coating_walk_accept_base_evaluation(layer, base, state);
}

ETX_SHARED_NOINLINE void bsdf_coating_walk_base_sample(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFCoatedConductor, layer), ETX_INOUT(BSDFCoatingWalk, state),
  ETX_INOUT(Sampler, sampler)) {
  if (bsdf_coating_transmission_enabled(layer)) {
    bsdf_coating_transmission_base_sample(context, layer, state, sampler);
    return;
  }
  BSDFData data = layer.data;
  data.w_i = state.direction;
  const Material substrate_material = bsdf_coating_interface_material(layer.substrate_material, layer.substrate_class);
#if (ETX_CPP == 1) || ETX_ENABLE_COATED_VELVET_SUBSTRATE
  if (layer.substrate_class == MaterialClass::Velvet) {
    const LocalFrame frame = bsdf_data_get_normal_frame(data, substrate_material);
    const float2 random = bsdf_sampler_has_fixed(sampler) ? float2(sampler.fixed_u, sampler.fixed_v) : bsdf_sampler_next_2d(sampler);
    const float3 outgoing = local_frame_from_local(frame, sample_cosine_distribution(random, 0.0f));
    bsdf_coating_walk_begin_velvet(context, data, substrate_material, outgoing, BSDFCoatingWalk::VelvetBaseSample, state);
    return;
  }
#endif
  const BSDFSample base = bsdf_coating_substrate_sample_non_velvet(context, data, substrate_material, sampler);
  bsdf_coating_walk_accept_base_sample(layer, base, state);
}

ETX_SHARED_NOINLINE void bsdf_coating_walk_exit_evaluate(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFCoatedConductor, layer), ETX_INOUT(BSDFCoatingWalk, state),
  ETX_INOUT(Sampler, sampler)) {
  if (bsdf_coating_transmission_enabled(layer)) {
    bsdf_coating_transmission_exit_evaluate(context, layer, state, sampler);
    return;
  }
  BSDFData data = layer.data;
  data.w_i = state.direction;
  const Material interface_material = bsdf_coating_interface_material(layer.interface_material, MaterialClass::Dielectric);
  state.exit_func = bsdf_dielectric_energy_compensated_evaluate(context, data, state.outgoing, interface_material, sampler).func;
  state.operation = BSDFCoatingWalk::ExitPdf;
}

ETX_SHARED_NOINLINE void bsdf_coating_walk_exit_pdf(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFCoatedConductor, layer), ETX_INOUT(BSDFCoatingWalk, state),
  ETX_INOUT(Sampler, sampler)) {
  if (bsdf_coating_transmission_enabled(layer)) {
    bsdf_coating_transmission_exit_pdf(context, layer, state, sampler);
    return;
  }
  const float reverse_pdf =
    bsdf_coating_transmission_conditioned_pdf(context, layer, true, bsdf_coating_walk_reverse_exit_data(layer, state), -state.direction, BSDFSample::Transmission, sampler);
  const float mis = ((state.exit_from_phase == false) && layer.substrate_delta) ? 1.0f : bsdf_coating_mis(state.continuation_pdf, reverse_pdf);
  const float distance = (state.exit_from_phase ? (layer.thickness - state.height) : layer.thickness) / state.direction.z;
  const SpectralResponse transmittance = bsdf_coating_transmittance(layer.extinction, distance);
  SpectralResponse connection;
  if (state.exit_from_phase) {
    connection = spectral_response_mul(transmittance, spectral_response_mul(state.exit_func, mis));
  } else {
    connection = spectral_response_mul(state.exit_func, spectral_response_mul(transmittance, mis));
  }
  state.result = spectral_response_add(state.result, spectral_response_mul(state.throughput, connection));
  state.operation = BSDFCoatingWalk::Advance;
}

struct BSDFCoatingRenewalPrepared {
  SpectralResponse reflectance ETX_INIT({});
  SpectralResponse function ETX_INIT({});
  SpectralResponse top_reflectance ETX_INIT({});
  RefractiveIndexSample exterior ETX_INIT({});
  RefractiveIndexSample interior ETX_INIT({});
  ThinfilmEval reverse_optics ETX_INIT({});
  bool identity ETX_INIT(false);
};

ETX_SHARED_INLINE bool bsdf_coating_renewal_response_in_range(ETX_IN(SpectralResponse, value), float lower, float upper) {
  if (bsdf_energy_compensated_spectral_response_finite(value) == false) {
    return false;
  }
  if (spectral_response_is_spectral(value)) {
    return (value.value >= lower) && (value.value <= upper);
  }
  return (value.integrated.x >= lower) && (value.integrated.x <= upper) && (value.integrated.y >= lower) && (value.integrated.y <= upper) && (value.integrated.z >= lower) &&
         (value.integrated.z <= upper);
}

ETX_SHARED_INLINE bool bsdf_coating_renewal_index_valid(ETX_IN(RefractiveIndexSample, value)) {
  return bsdf_coating_renewal_response_in_range(value.eta, kEpsilon, kMaxFloat) && bsdf_coating_renewal_response_in_range(value.k, 0.0f, kMaxFloat);
}

ETX_SHARED_INLINE bool bsdf_coating_renewal_prepare(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFCoatedConductor, layer), ETX_IN(Sampler, sampler),
  ETX_OUT(BSDFCoatingRenewalPrepared, prepared)) {
  prepared = ETX_ZERO(BSDFCoatingRenewalPrepared);
  if ((layer.substrate_class != MaterialClass::Diffuse) || (layer.interface_delta == false) || bsdf_sampler_has_fixed(sampler) ||
      (spectral_response_is_zero(layer.scattering) == false) || (isfinite(layer.thickness) == false) || (layer.thickness < 0.0f) ||
      ((isfinite(layer.data.w_i.x) && isfinite(layer.data.w_i.y) && isfinite(layer.data.w_i.z)) == false) ||
      (bsdf_coating_renewal_response_in_range(layer.extinction, 0.0f, kMaxFloat) == false)) {
    return false;
  }
  const Material substrate = bsdf_coating_interface_material(layer.substrate_material, MaterialClass::Diffuse);
  const float roughness = bsdf_diffuse_scalar_roughness(context, substrate, layer.data.tex);
  if ((isfinite(roughness) == false) || (roughness > kEpsilon)) {
    return false;
  }
  prepared.reflectance = bsdf_resource_apply_image(context, layer.data.spectrum_sample, substrate.scattering, layer.data.tex);
  if (bsdf_coating_renewal_response_in_range(prepared.reflectance, 0.0f, 1.0f) == false) {
    return false;
  }
  const Material top = bsdf_coating_interface_material(layer.interface_material, MaterialClass::Dielectric);
  prepared.top_reflectance = bsdf_resource_apply_image(context, layer.data.spectrum_sample, top.reflectance, layer.data.tex);
  prepared.exterior = bsdf_resource_evaluate_refractive_index(context, top.ext_ior, layer.data.spectrum_sample);
  prepared.interior = bsdf_resource_evaluate_refractive_index(context, top.int_ior, layer.data.spectrum_sample);
  if ((bsdf_coating_renewal_response_in_range(prepared.top_reflectance, 0.0f, 1.0f) == false) || (bsdf_coating_renewal_index_valid(prepared.exterior) == false) ||
      (bsdf_coating_renewal_index_valid(prepared.interior) == false)) {
    return false;
  }
  Sampler optics_sampler = ETX_ZERO(Sampler);
  const ThinfilmEval optics = bsdf_resource_evaluate_interface_optics(context, layer.data.spectrum_sample, top, layer.data.tex, optics_sampler);
  if ((isfinite(optics.thickness) == false) || (isfinite(optics.coating_thickness_nm) == false) || (isfinite(optics.weight) == false) ||
      ((optics.thickness > 0.0f) && (optics.weight > 0.0f) && (bsdf_coating_renewal_index_valid(optics.ior) == false)) ||
      ((optics.coating_thickness_nm > 0.0f) && (bsdf_coating_renewal_index_valid(optics.coating_ior) == false))) {
    return false;
  }
  prepared.function = spectral_response_mul(prepared.reflectance, kInvPi);
  prepared.reverse_optics = bsdf_fresnel_orient_coherent_interface(optics, false);
  prepared.identity = (bsdf_resource_interface_has_layers(optics) == false) && bsdf_fresnel_equal_indices(prepared.exterior, prepared.interior);
  return true;
}

ETX_SHARED_INLINE BSDFSample bsdf_coating_renewal_base_sample(ETX_IN(BSDFCoatingRenewalPrepared, prepared), ETX_IN(BSDFData, data), ETX_IN(float3, incident),
  ETX_INOUT(Sampler, sampler)) {
  BSDFSample sample = bsdf_sample_zero(data.spectrum_sample);
  sample.w_o = sample_cosine_distribution(bsdf_sampler_next_2d(sampler), 1.0f);
  sample.properties = BSDFSample::Reflection | BSDFSample::Diffuse;
  sample.eta = 1.0f;
  sample.medium_index = data.current_medium;
  if ((incident.z < -kEpsilon) && (sample.w_o.z > kEpsilon)) {
    sample.pdf = kInvPi * sample.w_o.z;
    sample.weight = spectral_response_div(spectral_response_mul(prepared.function, sample.w_o.z), sample.pdf);
  }
  return sample;
}

ETX_SHARED_INLINE BSDFSample bsdf_coating_renewal_top_sample(ETX_IN(BSDFCoatingRenewalPrepared, prepared), ETX_IN(BSDFData, data), ETX_IN(float3, incident)) {
  if (prepared.identity || (incident.z <= kEpsilon)) {
    return bsdf_sample_zero(data.spectrum_sample);
  }
  BSDFSample sample = ETX_ZERO(BSDFSample);
  sample.w_o = normalize(float3(incident.x, incident.y, -incident.z));
  sample.pdf = 1.0f;
  sample.weight =
    spectral_response_mul(prepared.top_reflectance, bsdf_fresnel_calculate(data.spectrum_sample, incident.z, prepared.interior, prepared.exterior, prepared.reverse_optics));
  sample.properties = BSDFSample::Delta | BSDFSample::Reflection;
  sample.medium_index = data.current_medium;
  sample.eta = 1.0f;
  return sample;
}

ETX_SHARED_INLINE bool bsdf_coating_renewal_walk_step(ETX_IN(BSDFCoatingRenewalPrepared, prepared), ETX_IN(BSDFCoatedConductor, layer), ETX_INOUT(BSDFCoatingWalk, state),
  ETX_INOUT(Sampler, sampler)) {
  if ((state.operation != BSDFCoatingWalk::BaseEvaluate) || (state.direction.z >= -kEpsilon) || (bsdf_coating_walk_reverse_exit_valid(state) == false) ||
      ((isfinite(state.direction.x) && isfinite(state.direction.y) && isfinite(state.direction.z)) == false) ||
      ((isfinite(state.reverse_exit.w_o.x) && isfinite(state.reverse_exit.w_o.y) && isfinite(state.reverse_exit.w_o.z)) == false) ||
      (bsdf_energy_compensated_spectral_response_finite(state.throughput) == false) || ((state.reverse_exit.properties & BSDFSample::Delta) == 0u)) {
    return false;
  }
  BSDFEval base = bsdf_eval_zero(layer.data.spectrum_sample);
  if (-state.reverse_exit.w_o.z > kEpsilon) {
    base.func = prepared.function;
    base.pdf = kInvPi * -state.reverse_exit.w_o.z;
  }
  bsdf_coating_walk_accept_base_evaluation(layer, base, state);
  bsdf_coating_walk_accept_base_sample(layer, bsdf_coating_renewal_base_sample(prepared, layer.data, state.direction, sampler), state);
  if (state.operation == BSDFCoatingWalk::Complete) {
    return true;
  }
  bsdf_coating_walk_advance(layer, state, sampler);
  if (state.operation == BSDFCoatingWalk::SampleBegin) {
    bsdf_coating_walk_accept_sample(layer, bsdf_coating_renewal_top_sample(prepared, layer.data, state.direction), state);
    if (state.operation == BSDFCoatingWalk::Advance) {
      bsdf_coating_walk_advance(layer, state, sampler);
    }
  }
  return true;
}

// The internal walk estimates f. Its random choices never enter the directional proposal PDF.
ETX_SHARED_NOINLINE SpectralResponse bsdf_coated_conductor_function(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFCoatedConductor, layer), ETX_IN(float3, outgoing),
  ETX_INOUT(Sampler, sampler)) {
  BSDFCoatingWalk state = bsdf_coating_walk_begin(layer, outgoing);
  BSDFCoatingRenewalPrepared renewal = ETX_ZERO(BSDFCoatingRenewalPrepared);
  bool renewal_checked = false;
  bool renewal_ready = false;
  while (state.operation != BSDFCoatingWalk::Complete) {
    switch (state.operation) {
      case BSDFCoatingWalk::EntranceEvaluate:
        bsdf_coating_transmission_entrance_evaluate(context, layer, state, sampler);
        break;
      case BSDFCoatingWalk::Direct:
        bsdf_coating_walk_direct(context, layer, state, sampler);
        break;
      case BSDFCoatingWalk::SampleBegin:
        bsdf_coating_walk_sample_begin(context, layer, state, sampler);
        break;
      case BSDFCoatingWalk::SampleDelta:
        bsdf_coating_walk_sample_delta(context, layer, state, sampler);
        break;
      case BSDFCoatingWalk::SampleResolve:
        bsdf_coating_walk_sample_resolve(context, layer, state);
        break;
      case BSDFCoatingWalk::Advance:
        bsdf_coating_walk_advance(layer, state, sampler);
        break;
      case BSDFCoatingWalk::Phase:
        bsdf_coating_walk_phase(context, layer, state, sampler);
        break;
      case BSDFCoatingWalk::BaseEvaluate: {
        if (renewal_checked == false) {
          renewal_ready = bsdf_coating_renewal_prepare(context, layer, sampler, renewal);
          renewal_checked = true;
        }
        bool renewed = false;
        if (renewal_ready) {
          renewed = bsdf_coating_renewal_walk_step(renewal, layer, state, sampler);
        }
        if (renewed == false) {
          bsdf_coating_walk_base_evaluate(context, layer, state, sampler);
        }
        break;
      }
      case BSDFCoatingWalk::BaseSample:
        bsdf_coating_walk_base_sample(context, layer, state, sampler);
        break;
#if (ETX_CPP == 1) || ETX_ENABLE_COATED_VELVET_SUBSTRATE
      case BSDFCoatingWalk::VelvetBaseEvaluate:
      case BSDFCoatingWalk::VelvetBaseSample:
        bsdf_coating_walk_velvet(context, layer, state, sampler);
        break;
#endif
      case BSDFCoatingWalk::ExitEvaluate:
        bsdf_coating_walk_exit_evaluate(context, layer, state, sampler);
        break;
      case BSDFCoatingWalk::ExitPdf:
        bsdf_coating_walk_exit_pdf(context, layer, state, sampler);
        break;
    }
  }
  return state.result;
}

ETX_SHARED_INLINE float bsdf_coating_reflection_proposal_pdf(ETX_IN(BSDFCoatedConductor, layer), ETX_IN(float3, outgoing)) {
  const float3 incident = -layer.data.w_i;
  const float3 normal = normalize(incident + outgoing);
  return bsdf_energy_compensated_vndf_pdf(incident, normal, layer.interface_roughness) / max(kEpsilon, 4.0f * dot(outgoing, normal));
}

ETX_SHARED_INLINE float bsdf_coated_conductor_continuous_pdf(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFCoatedConductor, layer), ETX_IN(float3, outgoing),
  ETX_INOUT(Sampler, sampler)) {
  float pdf = outgoing.z / kPi;
  float count = 1.0f;
  if (layer.interface_delta == false) {
    pdf += bsdf_coating_reflection_proposal_pdf(layer, outgoing);
    count += 1.0f;
  }
  if (layer.substrate_delta == false) {
    pdf += bsdf_coating_substrate_pdf(context, layer.data, outgoing, bsdf_coating_interface_material(layer.substrate_material, layer.substrate_class), sampler);
    count += 1.0f;
  }
  return pdf / count;
}

ETX_SHARED_NOINLINE float bsdf_coated_conductor_pdf(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(float3, outgoing),
  ETX_IN(BSDFCoatedConductorMaterial, parameters), ETX_INOUT(Sampler, sampler)) {
  const Material material = bsdf_coated_conductor_material(parameters);
  const BSDFCoatedConductor layer = bsdf_coated_conductor_prepare(context, data, material);
  if (bsdf_coating_transmission_enabled(layer)) {
    return bsdf_coating_transmission_pdf(context, layer, local_frame_to_local(layer.frame, outgoing));
  }
  const float3 local_outgoing = local_frame_to_local(layer.frame, outgoing);
  if ((layer.data.w_i.z >= -kEpsilon) || (local_outgoing.z <= kEpsilon)) {
    return 0.0f;
  }
  const float delta_probability = bsdf_coated_conductor_delta_probability(context, layer);
  if (delta_probability >= 1.0f) {
    return 0.0f;
  }
  return (1.0f - delta_probability) * bsdf_coated_conductor_continuous_pdf(context, layer, local_outgoing, sampler);
}

ETX_SHARED_INLINE float bsdf_coated_conductor_pdf(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(float3, outgoing), ETX_IN(Material, material),
  ETX_INOUT(Sampler, sampler)) {
  return bsdf_coated_conductor_pdf(context, data, outgoing, bsdf_coated_conductor_material_parameters(material), sampler);
}

ETX_SHARED_INLINE BSDFEval bsdf_coated_conductor_resolve_evaluation(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFCoatedConductor, layer), ETX_IN(float3, local_outgoing),
  ETX_IN(SpectralResponse, function), ETX_INOUT(Sampler, sampler)) {
  if (bsdf_coating_transmission_enabled(layer)) {
    return bsdf_coating_transmission_resolve_evaluation(context, layer, local_outgoing, function);
  }
  BSDFEval result = bsdf_eval_zero(layer.data.spectrum_sample);
  result.func = function;
  result.bsdf = spectral_response_mul(result.func, local_outgoing.z);
  result.pdf = (1.0f - bsdf_coated_conductor_delta_probability(context, layer)) * bsdf_coated_conductor_continuous_pdf(context, layer, local_outgoing, sampler);
  result.properties = BSDFSample::Reflection;
  result.medium_index = layer.data.current_medium;
  return result;
}

ETX_SHARED_NOINLINE BSDFEval bsdf_coated_conductor_evaluate_prepared(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFCoatedConductor, layer), ETX_IN(float3, local_outgoing),
  ETX_INOUT(Sampler, sampler)) {
  if (bsdf_coating_transmission_enabled(layer)) {
    if ((abs(layer.data.w_i.z) <= kEpsilon) || (abs(local_outgoing.z) <= kEpsilon)) {
      return bsdf_eval_zero(layer.data.spectrum_sample);
    }
    Sampler walk_sampler = sampler;
    bsdf_sampler_pop_fixed(walk_sampler);
    const SpectralResponse function = bsdf_coated_conductor_function(context, layer, local_outgoing, walk_sampler);
    sampler.seed = walk_sampler.seed;
    return bsdf_coating_transmission_resolve_evaluation(context, layer, local_outgoing, function);
  }
  BSDFEval result = bsdf_eval_zero(layer.data.spectrum_sample);
  if ((layer.data.w_i.z >= -kEpsilon) || (local_outgoing.z <= kEpsilon)) {
    return result;
  }
  Sampler walk_sampler = sampler;
  bsdf_sampler_pop_fixed(walk_sampler);
  const SpectralResponse function = bsdf_coated_conductor_function(context, layer, local_outgoing, walk_sampler);
  sampler.seed = walk_sampler.seed;
  return bsdf_coated_conductor_resolve_evaluation(context, layer, local_outgoing, function, sampler);
}

ETX_SHARED_NOINLINE BSDFEval bsdf_coated_conductor_evaluate(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(float3, outgoing),
  ETX_IN(BSDFCoatedConductorMaterial, parameters), ETX_INOUT(Sampler, sampler)) {
  const Material material = bsdf_coated_conductor_material(parameters);
  const BSDFCoatedConductor layer = bsdf_coated_conductor_prepare(context, data, material);
  return bsdf_coated_conductor_evaluate_prepared(context, layer, local_frame_to_local(layer.frame, outgoing), sampler);
}

ETX_SHARED_INLINE BSDFEval bsdf_coated_conductor_evaluate(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(float3, outgoing), ETX_IN(Material, material),
  ETX_INOUT(Sampler, sampler)) {
  return bsdf_coated_conductor_evaluate(context, data, outgoing, bsdf_coated_conductor_material_parameters(material), sampler);
}

ETX_SHARED_NOINLINE BSDFCoatingProposal bsdf_coated_conductor_propose_begin(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(BSDFCoatedConductor, layer),
  ETX_INOUT(Sampler, sampler)) {
  if (bsdf_coating_transmission_enabled(layer)) {
    return bsdf_coating_transmission_propose(context, layer, sampler);
  }
  BSDFCoatingProposal result = ETX_ZERO(BSDFCoatingProposal);
  result.sample = bsdf_sample_zero(data.spectrum_sample);
  if (layer.data.w_i.z >= -kEpsilon) {
    return result;
  }
  const float delta_probability = bsdf_coated_conductor_delta_probability(context, layer);
  const float selector = bsdf_sampler_has_fixed(sampler) ? sampler.fixed_w : bsdf_sampler_next(sampler);
  if (selector < delta_probability) {
    result.sample.w_o = normalize(reflect(data.w_i, layer.frame.nrm));
    result.sample.pdf = delta_probability;
    result.sample.weight = spectral_response_div(bsdf_coated_conductor_delta_weight(context, layer, false), delta_probability);
    result.sample.properties = BSDFSample::Reflection | BSDFSample::Delta;
    result.sample.medium_index = data.current_medium;
    return result;
  }
  const uint32_t count = 1u + uint32_t(layer.interface_delta == false) + uint32_t(layer.substrate_delta == false);
  const uint32_t proposal = min(count - 1u, uint32_t((selector - delta_probability) * float(count) / (1.0f - delta_probability)));
  float3 local_outgoing;
  if (proposal == 0u) {
    const float2 rnd = bsdf_sampler_has_fixed(sampler) ? float2(sampler.fixed_u, sampler.fixed_v) : bsdf_sampler_next_2d(sampler);
    local_outgoing = sample_cosine_distribution(rnd, 1.0f);
  } else {
    Sampler proposal_sampler = sampler;
    bsdf_sampler_pop_fixed(proposal_sampler);
    BSDFSample proposal_sample = bsdf_sample_zero(data.spectrum_sample);
    if ((proposal == 1u) && (layer.interface_delta == false)) {
      const float2 rnd = bsdf_sampler_has_fixed(sampler) ? float2(sampler.fixed_u, sampler.fixed_v) : bsdf_sampler_next_2d(proposal_sampler);
      const float3 normal = bsdf_energy_compensated_sample_vndf_local(-layer.data.w_i, layer.interface_roughness, rnd);
      proposal_sample.w_o = reflect(layer.data.w_i, normal);
      proposal_sample.pdf = proposal_sample.w_o.z > kEpsilon ? bsdf_coating_reflection_proposal_pdf(layer, proposal_sample.w_o) : 0.0f;
      proposal_sample.weight = spectral_response_make(data.spectrum_sample, 1.0f);
    } else {
      const Material substrate_material = bsdf_coating_interface_material(layer.substrate_material, layer.substrate_class);
#if (ETX_CPP == 1) || ETX_ENABLE_COATED_VELVET_SUBSTRATE
      if (layer.substrate_class == MaterialClass::Velvet) {
        const LocalFrame frame = bsdf_data_get_normal_frame(layer.data, substrate_material);
        result.outgoing = local_frame_from_local(frame, sample_cosine_distribution(bsdf_sampler_next_2d(proposal_sampler), 0.0f));
        result.kind = BSDFCoatingProposal::Velvet;
        sampler.seed = proposal_sampler.seed;
        return result;
      }
#endif
      proposal_sample = bsdf_coating_substrate_sample_non_velvet(context, layer.data, substrate_material, proposal_sampler);
    }
    sampler.seed = proposal_sampler.seed;
    if ((bsdf_sample_valid(proposal_sample) == false) || (proposal_sample.w_o.z <= kEpsilon)) {
      return result;
    }
    local_outgoing = proposal_sample.w_o;
  }
  result.outgoing = normalize(local_frame_from_local(layer.frame, local_outgoing));
  result.kind = BSDFCoatingProposal::Direction;
  return result;
}

ETX_SHARED_INLINE BSDFCoatingProposal bsdf_coated_conductor_proposal_resolve(ETX_IN(BSDFCoatedConductor, layer), ETX_IN(BSDFSample, sample)) {
  BSDFCoatingProposal result = ETX_ZERO(BSDFCoatingProposal);
  result.sample = bsdf_sample_zero(layer.data.spectrum_sample);
  if ((bsdf_sample_valid(sample) == false) || (sample.w_o.z <= kEpsilon)) {
    return result;
  }
  result.outgoing = normalize(local_frame_from_local(layer.frame, sample.w_o));
  result.kind = BSDFCoatingProposal::Direction;
  return result;
}

ETX_SHARED_NOINLINE BSDFCoatingProposal bsdf_coated_conductor_propose(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(BSDFCoatedConductor, layer),
  ETX_INOUT(Sampler, sampler)) {
  BSDFCoatingProposal proposal = bsdf_coated_conductor_propose_begin(context, data, layer, sampler);
#if (ETX_CPP == 1) || ETX_ENABLE_COATED_VELVET_SUBSTRATE
  if (proposal.kind == BSDFCoatingProposal::Velvet) {
    BSDFCoatingWalk state = ETX_ZERO(BSDFCoatingWalk);
    const Material substrate_material = bsdf_coating_interface_material(layer.substrate_material, layer.substrate_class);
    bsdf_coating_walk_begin_velvet(context, layer.data, substrate_material, proposal.outgoing, BSDFCoatingWalk::VelvetProposal, state);
    BSDFEval evaluation;
    do {
      evaluation = bsdf_coating_walk_velvet_evaluation(context, layer, state, sampler);
    } while (state.velvet.operation != BSDFVelvetWalk::Complete);
    proposal = bsdf_coated_conductor_proposal_resolve(layer, bsdf_velvet_sample_resolve(data.spectrum_sample, state.velvet_outgoing, evaluation));
  }
#endif
  return proposal;
}

ETX_SHARED_INLINE BSDFSample bsdf_coated_conductor_resolve_sample(ETX_IN(BSDFData, data), ETX_IN(float3, outgoing), ETX_IN(BSDFEval, eval)) {
  if (bsdf_eval_valid(eval) == false) {
    return bsdf_sample_zero(data.spectrum_sample);
  }
  BSDFSample result = bsdf_sample_zero(data.spectrum_sample);
  result.w_o = outgoing;
  result.pdf = eval.pdf;
  result.weight = spectral_response_div(eval.bsdf, eval.pdf);
  result.properties = eval.properties;
  result.medium_index = eval.medium_index;
  result.eta = eval.eta;
  return result;
}

ETX_SHARED_NOINLINE BSDFSample bsdf_coated_conductor_sample(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(BSDFCoatedConductorMaterial, parameters),
  ETX_INOUT(Sampler, sampler)) {
  const Material material = bsdf_coated_conductor_material(parameters);
  const BSDFCoatedConductor layer = bsdf_coated_conductor_prepare(context, data, material);
  const BSDFCoatingProposal proposal = bsdf_coated_conductor_propose(context, data, layer, sampler);
  if (proposal.kind == BSDFCoatingProposal::Complete) {
    return proposal.sample;
  }
  const BSDFEval eval = bsdf_coated_conductor_evaluate_prepared(context, layer, local_frame_to_local(layer.frame, proposal.outgoing), sampler);
  BSDFSample result = bsdf_coated_conductor_resolve_sample(data, proposal.outgoing, eval);
  result.properties |= proposal.sample.properties & BSDFSample::WavelengthDependentDirection;
  return result;
}

ETX_SHARED_INLINE BSDFSample bsdf_coated_conductor_sample(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(Material, material), ETX_INOUT(Sampler, sampler)) {
  return bsdf_coated_conductor_sample(context, data, bsdf_coated_conductor_material_parameters(material), sampler);
}

ETX_SHARED_INLINE bool bsdf_coated_conductor_is_delta_with_context(ETX_IN(BSDFResourceContext, context), ETX_IN(Material, material), ETX_IN(float2, tex)) {
  Material interface_material = material;
  interface_material.cls = MaterialClass::Dielectric;
  interface_material.int_ior = material.coating.ior;
  interface_material.roughness = material.coating.roughness;
  float3 scattering = bsdf_resource_load_spectrum_integrated(context, material.coating.scattering.spectrum_index);
  float4 image_value;
  if ((material.coating.scattering.image_index != kInvalidIndex) &&
      bsdf_resource_image_try_evaluate_rgba_no_pdf(context, material.coating.scattering.image_index, tex, image_value)) {
    scattering *= float3(image_value.x, image_value.y, image_value.z);
  }
  const Material substrate_material = material_coating_substrate(material);
  if (bsdf_coating_transmission_base(substrate_material.cls)) {
    if (substrate_material.cls == MaterialClass::Translucent) {
      return false;
    }
    const float thickness = material.coating.thickness.value.x * bsdf_resource_evaluate_sampled_image(context, material.coating.thickness, tex, 1.0f);
    return bsdf_dielectric_energy_compensated_is_delta_with_context(context, interface_material, tex) && bsdf_coating_substrate_is_delta(context, substrate_material, tex) &&
           ((thickness <= 0.0f) || (dot(scattering, scattering) == 0.0f));
  }
  return bsdf_dielectric_energy_compensated_is_delta_with_context(context, interface_material, tex) && bsdf_coating_substrate_is_delta(context, substrate_material, tex) &&
         (dot(scattering, scattering) == 0.0f);
}

struct BSDFCoatingSubstrateEscape {
  SpectralResponse source ETX_INIT({});
  SpectralResponse absorption ETX_INIT({});
};

ETX_SHARED_NOINLINE SpectralResponse bsdf_coating_substrate_absorptance(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(Material, material),
  ETX_INOUT(Sampler, sampler)) {
  const SpectralQuery spect = data.spectrum_sample;
  const SpectralResponse one = spectral_response_make(spect, 1.0f);
  if (material.cls == MaterialClass::Mirror) {
    return spectral_response_max(spectral_response_sub(one, bsdf_resource_apply_image(context, spect, material.scattering, data.tex)), 0.0f);
  }
  if (material.cls == MaterialClass::Diffuse) {
    const float roughness = bsdf_diffuse_scalar_roughness(context, material, data.tex);
    const float e = bsdf_diffuse_eon_directional_albedo(-data.w_i.z, roughness);
    const float e_avg = bsdf_diffuse_eon_average_albedo(roughness);
    const SpectralResponse albedo = bsdf_resource_apply_image(context, spect, material.scattering, data.tex);
    const SpectralResponse multiple =
      spectral_response_div(spectral_response_mul(spectral_response_mul(albedo, albedo), e_avg), spectral_response_sub(one, spectral_response_mul(albedo, 1.0f - e_avg)));
    return spectral_response_max(spectral_response_sub(one, spectral_response_add(spectral_response_mul(albedo, e), spectral_response_mul(multiple, 1.0f - e))), 0.0f);
  }
  const BSDFEnergyCompensatedPreparedMaterial prepared = bsdf_energy_compensated_prepare_material(context, spect, material, data.tex, sampler);
  if (prepared.conductor_delta) {
    return spectral_response_sub(one, bsdf_conductor_delta_weight(context, data, material, prepared.ext_ior, prepared.int_ior, prepared.thinfilm));
  }
  const float3 incoming = -data.w_i;
  SpectralResponse single = spectral_response_make(spect, 0.0f);
  if (prepared.supported) {
    single = bsdf_energy_compensated_conductor_directional_albedo(context, spect, material, incoming.z, prepared.alpha);
  } else {
    // Integrate the anisotropic single-scattering lobe in its VNDF sample domain.
    ETX_STATIC_CONST uint32_t count = 512u;
#if (ETX_CPP == 0)
    [loop]
#endif
      for (uint32_t index = 0u; index < count; ++index) {
      const uint32_t bits = sampler_reverse_bits_32(index);
      const float2 uv = float2((float(index) + 0.5f) / float(count), float(bits) * 2.3283064365386963e-10f);
      const float3 normal = bsdf_energy_compensated_sample_vndf_local(incoming, prepared.roughness, uv);
      const float3 outgoing = reflect(-incoming, normal);
      if (outgoing.z <= 0.0f) {
        continue;
      }
      const BSDFEnergyCompensatedLobe lobe =
        bsdf_energy_compensated_conductor_base_lobe(context, material, spect, incoming, outgoing, prepared.roughness, prepared.ext_ior, prepared.int_ior, prepared.thinfilm, one);
      if (lobe.pdf > 0.0f) {
        single = spectral_response_add(single, spectral_response_div(lobe.bsdf, lobe.pdf * float(count)));
      }
    }
  }
  const SpectralResponse tint = bsdf_resource_apply_image(context, spect, material.reflectance, data.tex);
  const BSDFEnergyCompensatedConductorIncident incident = bsdf_energy_compensated_conductor_prepare_incident(context, spect, material, incoming.z, prepared, tint);
  const SpectralResponse multiple = spectral_response_mul(incident.multiple_scattering_fresnel, 1.0f - incident.geometric_albedo);
  return spectral_response_max(spectral_response_sub(one, spectral_response_add(spectral_response_mul(tint, single), multiple)), 0.0f);
}

struct BSDFCoatingEscapeState {
  enum : uint32_t { Complete, SampleBegin, SampleDelta, SampleResolve, Advance, Phase, BaseSample, VelvetSample, BaseEmission, VelvetEscape, BaseContinue };
  enum : uint32_t { Entrance, Top, Bottom };
  SpectralResponse source ETX_INIT({});
  SpectralResponse absorption ETX_INIT({});
  SpectralResponse throughput ETX_INIT({});
  BSDFSample base ETX_INIT({});
  float3 direction ETX_INIT({});
  float height ETX_INIT(0.0f);
  uint32_t depth ETX_INIT(0u);
  uint32_t operation ETX_INIT(Complete);
  uint32_t sample_target ETX_INIT(Entrance);
};

ETX_SHARED_INLINE BSDFCoatingEscapeState bsdf_coating_escape_begin(ETX_IN(SpectralQuery, spect)) {
  BSDFCoatingEscapeState state = ETX_ZERO(BSDFCoatingEscapeState);
  state.source = spectral_response_make(spect, 0.0f);
  state.absorption = spectral_response_make(spect, 0.0f);
  state.operation = BSDFCoatingEscapeState::SampleBegin;
  return state;
}

ETX_SHARED_INLINE BSDFCoatingEscapeState bsdf_coating_escape_begin(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFCoatedConductor, layer)) {
  BSDFCoatingEscapeState state = bsdf_coating_escape_begin(layer.data.spectrum_sample);
  if (bsdf_coating_transmission_enabled(layer) && (layer.data.w_i.z > 0.0f)) {
    if (layer.source_two_sided) {
      const float exponent = scene_math_shared_collimation_to_exponent(layer.emission_collimation);
      state.source = spectral_response_make(layer.data.spectrum_sample, scene_math_shared_collimated_emission_scale(layer.data.w_i.z, exponent));
    }
    if ((layer.substrate_class == MaterialClass::Translucent) && (layer.thermal_spectrum_index != kInvalidIndex)) {
      const Material substrate = bsdf_coating_interface_material(layer.substrate_material, layer.substrate_class);
      state.absorption = bsdf_translucent_absorptance(context, layer.data, substrate);
    }
  }
  return state;
}

ETX_SHARED_INLINE bool bsdf_coating_escape_sample_top(ETX_IN(BSDFCoatedConductor, layer), ETX_IN(BSDFCoatingEscapeState, state)) {
  if (bsdf_coating_transmission_enabled(layer) == false) {
    return true;
  }
  if (state.sample_target == BSDFCoatingEscapeState::Entrance) {
    return layer.data.w_i.z < 0.0f;
  }
  return state.sample_target == BSDFCoatingEscapeState::Top;
}

ETX_SHARED_INLINE BSDFData bsdf_coating_escape_boundary_data(ETX_IN(BSDFCoatedConductor, layer), ETX_IN(BSDFCoatingEscapeState, state)) {
  BSDFData data = layer.data;
  data.w_i = state.direction;
  return data;
}

ETX_SHARED_INLINE BSDFData bsdf_coating_escape_sample_data(ETX_IN(BSDFCoatedConductor, layer), ETX_IN(BSDFCoatingEscapeState, state)) {
  if (state.sample_target == BSDFCoatingEscapeState::Entrance) {
    return layer.data;
  }
  return bsdf_coating_escape_boundary_data(layer, state);
}

ETX_SHARED_INLINE void bsdf_coating_escape_accept_sample(ETX_IN(BSDFCoatedConductor, layer), ETX_IN(BSDFSample, sample), ETX_INOUT(BSDFCoatingEscapeState, state)) {
  if (state.sample_target == BSDFCoatingEscapeState::Bottom) {
    state.base = sample;
    state.operation = BSDFCoatingEscapeState::BaseEmission;
    return;
  }
  const bool entrance = state.sample_target == BSDFCoatingEscapeState::Entrance;
  const uint32_t required = entrance ? BSDFSample::Transmission : BSDFSample::Reflection;
  if ((bsdf_sample_valid(sample) == false) || ((sample.properties & required) == 0u)) {
    state.operation = BSDFCoatingEscapeState::Complete;
    return;
  }
  if (entrance) {
    state.throughput = sample.weight;
  } else {
    state.throughput = spectral_response_mul(state.throughput, sample.weight);
  }
  state.direction = sample.w_o;
  if (entrance) {
    state.height = bsdf_coating_escape_sample_top(layer, state) ? layer.thickness : 0.0f;
  }
  state.operation = BSDFCoatingEscapeState::Advance;
}

ETX_SHARED_NOINLINE void bsdf_coating_escape_sample_begin(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFCoatedConductor, layer), ETX_INOUT(BSDFCoatingEscapeState, state),
  ETX_OUT(BSDFEnergyCompensatedDielectricSampleState, sample_state), ETX_INOUT(Sampler, sampler)) {
  const uint32_t required_event = state.sample_target == BSDFCoatingEscapeState::Entrance ? BSDFSample::Transmission : BSDFSample::Reflection;
  sample_state = bsdf_coating_transmission_sample_interface_begin(context, layer, bsdf_coating_escape_sample_top(layer, state), bsdf_coating_escape_sample_data(layer, state),
    required_event, sampler);
  if (sample_state.stage == BSDFEnergyCompensatedDielectricSampleState::Delta) {
    state.operation = BSDFCoatingEscapeState::SampleDelta;
    return;
  }
  if (sample_state.stage == BSDFEnergyCompensatedDielectricSampleState::Resolve) {
    state.operation = BSDFCoatingEscapeState::SampleResolve;
    return;
  }
  bsdf_coating_escape_accept_sample(layer, sample_state.proposal.sample, state);
}

ETX_SHARED_NOINLINE void bsdf_coating_escape_sample_delta(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFCoatedConductor, layer), ETX_INOUT(BSDFCoatingEscapeState, state),
  ETX_INOUT(Sampler, sampler)) {
  const Material material = bsdf_coating_transmission_interface(layer, bsdf_coating_escape_sample_top(layer, state));
  const uint32_t required_event = state.sample_target == BSDFCoatingEscapeState::Entrance ? BSDFSample::Transmission : BSDFSample::Reflection;
  bsdf_coating_escape_accept_sample(layer, bsdf_dielectric_delta_sample_event(context, bsdf_coating_escape_sample_data(layer, state), material, required_event, sampler), state);
}

ETX_SHARED_NOINLINE void bsdf_coating_escape_sample_resolve(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFCoatedConductor, layer), ETX_INOUT(BSDFCoatingEscapeState, state),
  ETX_IN(BSDFEnergyCompensatedDielectricSampleState, sample_state)) {
  const uint32_t required_event = state.sample_target == BSDFCoatingEscapeState::Entrance ? BSDFSample::Transmission : BSDFSample::Reflection;
  const BSDFSample sample = bsdf_coating_transmission_sample_interface_resolve(context, layer, bsdf_coating_escape_sample_top(layer, state),
    bsdf_coating_escape_sample_data(layer, state), required_event, sample_state);
  bsdf_coating_escape_accept_sample(layer, sample, state);
}

ETX_SHARED_INLINE void bsdf_coating_escape_advance(ETX_IN(BSDFCoatedConductor, layer), ETX_INOUT(BSDFCoatingEscapeState, state), ETX_INOUT(Sampler, sampler)) {
  if (spectral_response_is_zero(state.throughput)) {
    state.operation = BSDFCoatingEscapeState::Complete;
    return;
  }
  if (state.depth++ >= 4u) {
    const float survival = bsdf_external_continuation_probability(state.throughput);
    if ((survival <= 0.0f) || ((survival < 1.0f) && (bsdf_sampler_next(sampler) >= survival))) {
      state.operation = BSDFCoatingEscapeState::Complete;
      return;
    }
    state.throughput = spectral_response_div(state.throughput, survival);
  }
  if (abs(state.direction.z) <= kEpsilon) {
    state.operation = BSDFCoatingEscapeState::Complete;
    return;
  }
  const bool hit_top = state.direction.z > 0.0f;
  const float boundary_height = hit_top ? layer.thickness : 0.0f;
  float distance = 0.0f;
  const bool scattered = bsdf_coating_free_flight(layer, max(0.0f, (boundary_height - state.height) / state.direction.z), sampler, state.throughput, distance);
  state.height = clamp(state.height + distance * state.direction.z, 0.0f, layer.thickness);
  if (scattered) {
    state.operation = BSDFCoatingEscapeState::Phase;
    return;
  }
  state.sample_target = BSDFCoatingEscapeState::Top;
  if ((hit_top == false) && bsdf_coating_transmission_enabled(layer)) {
    state.sample_target = BSDFCoatingEscapeState::Bottom;
  }
  state.operation = (hit_top || bsdf_coating_transmission_enabled(layer)) ? BSDFCoatingEscapeState::SampleBegin : BSDFCoatingEscapeState::BaseSample;
}

ETX_SHARED_INLINE void bsdf_coating_escape_phase(ETX_IN(BSDFCoatedConductor, layer), ETX_INOUT(BSDFCoatingEscapeState, state), ETX_INOUT(Sampler, sampler)) {
  state.direction = medium_phase_shared_sample_henyey_greenstein(state.direction, layer.anisotropy, bsdf_sampler_next_2d(sampler));
  state.operation = BSDFCoatingEscapeState::Advance;
}

ETX_SHARED_NOINLINE void bsdf_coating_escape_base_sample(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFCoatedConductor, layer), ETX_INOUT(BSDFCoatingEscapeState, state),
  ETX_OUT(BSDFVelvetWalk, child), ETX_INOUT(Sampler, sampler)) {
  const BSDFData data = bsdf_coating_escape_boundary_data(layer, state);
  const Material material = bsdf_coating_interface_material(layer.substrate_material, layer.substrate_class);
  child = ETX_ZERO(BSDFVelvetWalk);
#if (ETX_CPP == 1) || ETX_ENABLE_COATED_VELVET_SUBSTRATE
  if (material.cls == MaterialClass::Velvet) {
    const LocalFrame frame = bsdf_data_get_normal_frame(data, material);
    const float2 random = bsdf_sampler_has_fixed(sampler) ? float2(sampler.fixed_u, sampler.fixed_v) : bsdf_sampler_next_2d(sampler);
    state.base = bsdf_sample_zero(data.spectrum_sample);
    state.base.w_o = local_frame_from_local(frame, sample_cosine_distribution(random, 0.0f));
    const BSDFVelvetPrepared prepared = bsdf_velvet_prepare(context, data, material);
    child = bsdf_velvet_walk_begin_evaluation(context, data.spectrum_sample, prepared, local_frame_to_local(frame, -data.w_i), local_frame_to_local(frame, state.base.w_o));
    state.operation = BSDFCoatingEscapeState::VelvetSample;
    return;
  }
#endif
  state.base = bsdf_coating_substrate_sample_non_velvet(context, data, material, sampler);
  state.operation = BSDFCoatingEscapeState::BaseEmission;
}

ETX_SHARED_INLINE void bsdf_coating_escape_velvet_sample(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFCoatedConductor, layer), ETX_INOUT(BSDFCoatingEscapeState, state),
  ETX_INOUT(BSDFVelvetWalk, child), ETX_INOUT(Sampler, sampler)) {
  const BSDFData data = bsdf_coating_escape_boundary_data(layer, state);
  const Material material = bsdf_coating_interface_material(layer.substrate_material, layer.substrate_class);
  const BSDFVelvetPrepared prepared = bsdf_velvet_prepare(context, data, material);
  if (child.operation != BSDFVelvetWalk::Complete) {
    Sampler walk_sampler = sampler;
    bsdf_sampler_pop_fixed(walk_sampler);
    bsdf_velvet_walk_evaluate_step(context, data.spectrum_sample, prepared, child, walk_sampler);
    sampler.seed = walk_sampler.seed;
  }
  if (child.operation == BSDFVelvetWalk::Complete) {
    const LocalFrame frame = bsdf_data_get_normal_frame(data, material);
    const BSDFEval eval = bsdf_velvet_evaluation_resolve(data, local_frame_to_local(frame, -data.w_i), local_frame_to_local(frame, state.base.w_o), child.value);
    state.base = bsdf_velvet_sample_resolve(data.spectrum_sample, state.base.w_o, eval);
    state.operation = BSDFCoatingEscapeState::BaseEmission;
  }
}

ETX_SHARED_NOINLINE void bsdf_coating_escape_base_emission(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFCoatedConductor, layer), ETX_INOUT(BSDFCoatingEscapeState, state),
  ETX_OUT(BSDFVelvetWalk, child), ETX_INOUT(Sampler, sampler)) {
  const BSDFData data = bsdf_coating_escape_boundary_data(layer, state);
  const Material material = bsdf_coating_interface_material(layer.substrate_material, layer.substrate_class);
  child = ETX_ZERO(BSDFVelvetWalk);
#if (ETX_CPP == 1) || ETX_ENABLE_COATED_VELVET_SUBSTRATE
  if (material.cls == MaterialClass::Velvet) {
    const BSDFVelvetPrepared prepared = bsdf_velvet_prepare(context, data, material);
    child = bsdf_velvet_walk_begin_escape(context, data.spectrum_sample, prepared, -data.w_i, layer.emission_collimation);
    state.operation = BSDFCoatingEscapeState::VelvetEscape;
    return;
  }
#endif
  const float exponent = scene_math_shared_collimation_to_exponent(layer.emission_collimation);
  state.source = spectral_response_add(state.source, spectral_response_mul(state.throughput, scene_math_shared_collimated_emission_scale(-data.w_i.z, exponent)));
  if (layer.thermal_spectrum_index != kInvalidIndex) {
    SpectralResponse absorption = spectral_response_zero(data.spectrum_sample);
    if (material.cls == MaterialClass::Translucent) {
      absorption = bsdf_translucent_absorptance(context, data, material);
    } else if (material.cls != MaterialClass::Dielectric) {
      absorption = bsdf_coating_substrate_absorptance(context, data, material, sampler);
    }
    state.absorption = spectral_response_add(state.absorption, spectral_response_mul(state.throughput, absorption));
  }
  state.operation = BSDFCoatingEscapeState::BaseContinue;
}

ETX_SHARED_INLINE void bsdf_coating_escape_velvet_escape(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFCoatedConductor, layer), ETX_INOUT(BSDFCoatingEscapeState, state),
  ETX_INOUT(BSDFVelvetWalk, child), ETX_INOUT(Sampler, sampler)) {
  const BSDFData data = bsdf_coating_escape_boundary_data(layer, state);
  const Material material = bsdf_coating_interface_material(layer.substrate_material, layer.substrate_class);
  const BSDFVelvetPrepared prepared = bsdf_velvet_prepare(context, data, material);
  bsdf_velvet_walk_escape_step(context, data.spectrum_sample, prepared, child, sampler);
  if (child.operation == BSDFVelvetWalk::Complete) {
    state.source = spectral_response_add(state.source, spectral_response_mul(state.throughput, child.value));
    state.absorption = spectral_response_add(state.absorption, spectral_response_mul(state.throughput, child.absorption));
    state.operation = BSDFCoatingEscapeState::BaseContinue;
  }
}

ETX_SHARED_INLINE void bsdf_coating_escape_base_continue(ETX_IN(BSDFCoatedConductor, layer), ETX_INOUT(BSDFCoatingEscapeState, state)) {
  if ((bsdf_sample_valid(state.base) == false) || (bsdf_coating_transmission_enabled(layer) && ((state.base.properties & BSDFSample::Reflection) == 0u))) {
    state.operation = BSDFCoatingEscapeState::Complete;
    return;
  }
  state.throughput = spectral_response_mul(state.throughput, state.base.weight);
  state.direction = state.base.w_o;
  state.operation = BSDFCoatingEscapeState::Advance;
}

ETX_SHARED_INLINE bool bsdf_coating_renewal_escape_step(ETX_IN(BSDFCoatingRenewalPrepared, prepared), ETX_IN(BSDFCoatedConductor, layer), ETX_INOUT(BSDFCoatingEscapeState, state),
  ETX_INOUT(Sampler, sampler)) {
  if ((state.operation != BSDFCoatingEscapeState::BaseEmission) || (state.direction.z >= -kEpsilon) ||
      ((isfinite(state.direction.x) && isfinite(state.direction.y) && isfinite(state.direction.z)) == false) ||
      (bsdf_energy_compensated_spectral_response_finite(state.throughput) == false)) {
    return false;
  }
  const float exponent = scene_math_shared_collimation_to_exponent(layer.emission_collimation);
  state.source = spectral_response_add(state.source, spectral_response_mul(state.throughput, scene_math_shared_collimated_emission_scale(-state.direction.z, exponent)));
  if (layer.thermal_spectrum_index != kInvalidIndex) {
    const SpectralResponse absorption = spectral_response_sub(spectral_response_make(layer.data.spectrum_sample, 1.0f), prepared.reflectance);
    state.absorption = spectral_response_add(state.absorption, spectral_response_mul(state.throughput, absorption));
  }
  bsdf_coating_escape_base_continue(layer, state);
  if (state.operation == BSDFCoatingEscapeState::Complete) {
    return true;
  }
  bsdf_coating_escape_advance(layer, state, sampler);
  if (state.operation == BSDFCoatingEscapeState::SampleBegin) {
    bsdf_coating_escape_accept_sample(layer, bsdf_coating_renewal_top_sample(prepared, layer.data, state.direction), state);
    if (state.operation == BSDFCoatingEscapeState::Advance) {
      bsdf_coating_escape_advance(layer, state, sampler);
    }
  }
  if (state.operation == BSDFCoatingEscapeState::BaseSample) {
    // The generic escape walk samples the base even when downward Beer attenuation underflows.
    state.base = bsdf_coating_renewal_base_sample(prepared, layer.data, state.direction, sampler);
    state.operation = BSDFCoatingEscapeState::BaseEmission;
  }
  return true;
}

ETX_SHARED_NOINLINE BSDFCoatingSubstrateEscape bsdf_coated_conductor_substrate_escape(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFCoatedConductor, layer),
  ETX_INOUT(Sampler, sampler)) {
  BSDFCoatingEscapeState state = bsdf_coating_escape_begin(context, layer);
  BSDFEnergyCompensatedDielectricSampleState interface_sample = ETX_ZERO(BSDFEnergyCompensatedDielectricSampleState);
  BSDFVelvetWalk child = ETX_ZERO(BSDFVelvetWalk);
  BSDFCoatingRenewalPrepared renewal = ETX_ZERO(BSDFCoatingRenewalPrepared);
  bool renewal_checked = false;
  bool renewal_ready = false;
  while (state.operation != BSDFCoatingEscapeState::Complete) {
    switch (state.operation) {
      case BSDFCoatingEscapeState::SampleBegin:
        bsdf_coating_escape_sample_begin(context, layer, state, interface_sample, sampler);
        break;
      case BSDFCoatingEscapeState::SampleDelta:
        bsdf_coating_escape_sample_delta(context, layer, state, sampler);
        break;
      case BSDFCoatingEscapeState::SampleResolve:
        bsdf_coating_escape_sample_resolve(context, layer, state, interface_sample);
        break;
      case BSDFCoatingEscapeState::Advance:
        bsdf_coating_escape_advance(layer, state, sampler);
        break;
      case BSDFCoatingEscapeState::Phase:
        bsdf_coating_escape_phase(layer, state, sampler);
        break;
      case BSDFCoatingEscapeState::BaseSample:
        bsdf_coating_escape_base_sample(context, layer, state, child, sampler);
        break;
      case BSDFCoatingEscapeState::VelvetSample:
        bsdf_coating_escape_velvet_sample(context, layer, state, child, sampler);
        break;
      case BSDFCoatingEscapeState::BaseEmission: {
        if (renewal_checked == false) {
          renewal_ready = bsdf_coating_renewal_prepare(context, layer, sampler, renewal);
          renewal_checked = true;
        }
        bool renewed = false;
        if (renewal_ready) {
          renewed = bsdf_coating_renewal_escape_step(renewal, layer, state, sampler);
        }
        if (renewed == false) {
          bsdf_coating_escape_base_emission(context, layer, state, child, sampler);
        }
        break;
      }
      case BSDFCoatingEscapeState::VelvetEscape:
        bsdf_coating_escape_velvet_escape(context, layer, state, child, sampler);
        break;
      case BSDFCoatingEscapeState::BaseContinue:
        bsdf_coating_escape_base_continue(layer, state);
        break;
    }
  }
  BSDFCoatingSubstrateEscape result = ETX_ZERO(BSDFCoatingSubstrateEscape);
  result.source = state.source;
  result.absorption = state.absorption;
  return result;
}
