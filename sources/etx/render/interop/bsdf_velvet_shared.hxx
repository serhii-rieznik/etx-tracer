#pragma once

#include "bsdf_various_shared.hxx"
#include "bsdf_velvet_constants_shared.hxx"
#include "bsdf_energy_compensation_interface_shared.hxx"
#include "scene_math_shared.hxx"

struct BSDFVelvetPrepared {
  SpectralResponse diffuse ETX_INIT({});
  SpectralResponse sheen ETX_INIT({});
  float roughness ETX_INIT(0.0f);
  uint32_t image_index ETX_INIT(kInvalidIndex);
};

ETX_SHARED_INLINE BSDFVelvetPrepared bsdf_velvet_prepare(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(Material, material)) {
  BSDFVelvetPrepared result = ETX_ZERO(BSDFVelvetPrepared);
  result.diffuse = bsdf_resource_apply_image(context, data.spectrum_sample, material.scattering, data.tex);
  result.sheen = bsdf_resource_apply_image(context, data.spectrum_sample, material.reflectance, data.tex);
  const float2 roughness = bsdf_resource_evaluate_roughness(context, material, data.tex);
  result.roughness = saturate(0.5f * (roughness.x + roughness.y));
  BSDFEnergyCompensationInterfaceData lookup = ETX_ZERO(BSDFEnergyCompensationInterfaceData);
  if (bsdf_resource_energy_compensation_interface_try_load(context, material.energy_compensation_interface_index, lookup) && (lookup.cls == MaterialClass::Velvet)) {
    result.image_index = lookup.directional_lut;
  }
  return result;
}

ETX_SHARED_INLINE float bsdf_velvet_lookup(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFVelvetPrepared, prepared), float mu, bool projected_area) {
  const float alpha = max(kEpsilon, prepared.roughness);
  const float root_alpha = sqrt(alpha);
  const float row = log(alpha / kEpsilon) / log(1.0f / kEpsilon);
  const float coordinate = projected_area ? (asin(saturate(mu)) * (2.0f / kPi)) : sqrt(log(1.0f + saturate(mu) / root_alpha) / log(1.0f + 1.0f / root_alpha));
  const float scale = float(kVelvetPhysicalLutSize - 1u) / float(kVelvetPhysicalLutSize);
  const float4 value = bsdf_resource_image_evaluate_rgba_no_pdf_or_zero(context, prepared.image_index, float2(coordinate * scale, row * scale));
  return projected_area ? (value.y / root_alpha) : value.x;
}

ETX_SHARED_INLINE float bsdf_velvet_sheen_function(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFVelvetPrepared, prepared), ETX_IN(float3, incoming), ETX_IN(float3, outgoing)) {
  if ((incoming.z <= kEpsilon) || (outgoing.z <= kEpsilon) || (prepared.roughness <= kEpsilon)) {
    return 0.0f;
  }
  const float3 half_vector = normalize(incoming + outgoing);
  const float inverse_alpha = 1.0f / prepared.roughness;
  const float half_z_squared = min(1.0f, half_vector.z * half_vector.z);
  const float sine_squared = 1.0f - half_z_squared;
  // Compensate cancellation in log(1 - z^2) for narrow grazing distributions.
  const float logarithm = sine_squared == 1.0f ? -half_z_squared : log(sine_squared) * half_z_squared / (1.0f - sine_squared);
  const float density = (2.0f + inverse_alpha) * exp(0.5f * inverse_alpha * logarithm) / kDoublePi;
  const float area_i = bsdf_velvet_lookup(context, prepared, incoming.z, true);
  const float area_o = bsdf_velvet_lookup(context, prepared, outgoing.z, true);
  return density / (4.0f * (area_i * outgoing.z + area_o * incoming.z - incoming.z * outgoing.z));
}

ETX_SHARED_INLINE SpectralResponse bsdf_velvet_transmission(ETX_IN(BSDFResourceContext, context), ETX_IN(SpectralQuery, spect), ETX_IN(BSDFVelvetPrepared, prepared), float mu) {
  if ((prepared.roughness <= kEpsilon) || spectral_response_is_zero(prepared.sheen)) {
    return spectral_response_make(spect, 1.0f);
  }
  return spectral_response_sub(spectral_response_make(spect, 1.0f), spectral_response_mul(prepared.sheen, bsdf_velvet_lookup(context, prepared, mu, false)));
}

ETX_SHARED_INLINE SpectralResponse bsdf_velvet_return_sample(ETX_IN(BSDFResourceContext, context), ETX_IN(SpectralQuery, spect), ETX_IN(BSDFVelvetPrepared, prepared),
  ETX_INOUT(float3, incoming), ETX_INOUT(Sampler, sampler)) {
  if ((prepared.roughness <= kEpsilon) || spectral_response_is_zero(prepared.sheen)) {
    return spectral_response_zero(spect);
  }
  const float3 from_base = sample_cosine_distribution(bsdf_sampler_next_2d(sampler), 1.0f);
  const SpectralResponse bottom = spectral_response_mul(bsdf_diffuse_eon_brdf(spect, prepared.diffuse, incoming, from_base, prepared.roughness), kPi);
  const float3 next_incoming = sample_cosine_distribution(bsdf_sampler_next_2d(sampler), 0.0f);
  const float reflection = bsdf_velvet_sheen_function(context, prepared, from_base, next_incoming) * next_incoming.z * kDoublePi;
  incoming = next_incoming;
  return spectral_response_mul(bottom, spectral_response_mul(prepared.sheen, reflection));
}

struct BSDFVelvetWalk {
  enum : uint32_t { Complete, Evaluate, SourceEscape };
  SpectralResponse value ETX_INIT({});
  SpectralResponse absorption ETX_INIT({});
  SpectralResponse throughput ETX_INIT({});
  SpectralResponse transmission ETX_INIT({});
  SpectralResponse multiple ETX_INIT({});
  float3 direction ETX_INIT({});
  float3 outgoing ETX_INIT({});
  float exponent ETX_INIT(0.0f);
  uint32_t depth ETX_INIT(0u);
  uint32_t operation ETX_INIT(Complete);
};

ETX_SHARED_INLINE BSDFVelvetWalk bsdf_velvet_walk_begin_function(ETX_IN(BSDFResourceContext, context), ETX_IN(SpectralQuery, spect), ETX_IN(BSDFVelvetPrepared, prepared),
  ETX_IN(float3, incoming), ETX_IN(float3, outgoing)) {
  BSDFVelvetWalk state = ETX_ZERO(BSDFVelvetWalk);
  state.value = spectral_response_mul(prepared.sheen, bsdf_velvet_sheen_function(context, prepared, incoming, outgoing));
  state.transmission = bsdf_velvet_transmission(context, spect, prepared, outgoing.z);
  state.throughput = bsdf_velvet_transmission(context, spect, prepared, incoming.z);
  state.direction = incoming;
  state.outgoing = outgoing;
  state.operation = BSDFVelvetWalk::Evaluate;
  return state;
}

ETX_SHARED_INLINE bool bsdf_velvet_walk_continue(ETX_INOUT(BSDFVelvetWalk, state), ETX_INOUT(Sampler, sampler)) {
  if (spectral_response_is_zero(state.throughput)) {
    state.operation = BSDFVelvetWalk::Complete;
    return false;
  }
  if (state.depth++ >= 4u) {
    const float survival = min(0.95f, spectral_response_maximum(state.throughput));
    if ((survival <= 0.0f) || (bsdf_sampler_next(sampler) >= survival)) {
      state.operation = BSDFVelvetWalk::Complete;
      return false;
    }
    state.throughput = spectral_response_div(state.throughput, survival);
  }
  return true;
}

ETX_SHARED_INLINE void bsdf_velvet_walk_evaluate_step(ETX_IN(BSDFResourceContext, context), ETX_IN(SpectralQuery, spect), ETX_IN(BSDFVelvetPrepared, prepared),
  ETX_INOUT(BSDFVelvetWalk, state), ETX_INOUT(Sampler, sampler)) {
  if (bsdf_velvet_walk_continue(state, sampler) == false) {
    return;
  }
  const SpectralResponse bottom = bsdf_diffuse_eon_brdf(spect, prepared.diffuse, state.direction, state.outgoing, prepared.roughness);
  state.value = spectral_response_add(state.value, spectral_response_mul(state.throughput, spectral_response_mul(state.transmission, bottom)));
  state.throughput = spectral_response_mul(state.throughput, bsdf_velvet_return_sample(context, spect, prepared, state.direction, sampler));
}

ETX_SHARED_NOINLINE SpectralResponse bsdf_velvet_function(ETX_IN(BSDFResourceContext, context), ETX_IN(SpectralQuery, spect), ETX_IN(BSDFVelvetPrepared, prepared),
  ETX_IN(float3, incoming), ETX_IN(float3, outgoing), ETX_INOUT(Sampler, sampler)) {
  BSDFVelvetWalk state = bsdf_velvet_walk_begin_function(context, spect, prepared, incoming, outgoing);
  while (state.operation != BSDFVelvetWalk::Complete) {
    bsdf_velvet_walk_evaluate_step(context, spect, prepared, state, sampler);
  }
  return state.value;
}

ETX_SHARED_INLINE BSDFVelvetWalk bsdf_velvet_walk_begin_evaluation(ETX_IN(BSDFResourceContext, context), ETX_IN(SpectralQuery, spect), ETX_IN(BSDFVelvetPrepared, prepared),
  ETX_IN(float3, incoming), ETX_IN(float3, outgoing)) {
  BSDFVelvetWalk state = ETX_ZERO(BSDFVelvetWalk);
  state.value = spectral_response_zero(spect);
  state.outgoing = outgoing;
  if ((incoming.z <= kEpsilon) || (outgoing.z <= kEpsilon)) {
    return state;
  }
  if ((prepared.roughness <= kEpsilon) || spectral_response_is_zero(prepared.sheen)) {
    state.value = bsdf_diffuse_eon_brdf(spect, prepared.diffuse, incoming, outgoing, prepared.roughness);
    return state;
  }
  return bsdf_velvet_walk_begin_function(context, spect, prepared, incoming, outgoing);
}

ETX_SHARED_INLINE BSDFEval bsdf_velvet_evaluation_resolve(ETX_IN(BSDFData, data), ETX_IN(float3, incoming), ETX_IN(float3, outgoing), ETX_IN(SpectralResponse, function)) {
  BSDFEval result = bsdf_eval_zero(data.spectrum_sample);
  if ((incoming.z <= kEpsilon) || (outgoing.z <= kEpsilon)) {
    return result;
  }
  result.func = function;
  result.bsdf = spectral_response_mul(result.func, outgoing.z);
  result.pdf = 1.0f / kDoublePi;
  result.properties = BSDFSample::Reflection | BSDFSample::Diffuse;
  result.medium_index = data.current_medium;
  result.eta = 1.0f;
  return result;
}

ETX_SHARED_INLINE BSDFSample bsdf_velvet_sample_resolve(ETX_IN(SpectralQuery, spect), ETX_IN(float3, outgoing), ETX_IN(BSDFEval, eval)) {
  BSDFSample result = bsdf_sample_zero(spect);
  result.w_o = outgoing;
  result.pdf = eval.pdf;
  result.properties = eval.properties;
  result.medium_index = eval.medium_index;
  result.eta = 1.0f;
  if (result.pdf > 0.0f) {
    result.weight = spectral_response_div(eval.bsdf, result.pdf);
  }
  return result;
}

ETX_SHARED_INLINE BSDFEval bsdf_velvet_evaluate(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(float3, outgoing_direction), ETX_IN(Material, material),
  ETX_INOUT(Sampler, sampler)) {
  const LocalFrame frame = bsdf_data_get_normal_frame(data, material);
  const float3 incoming = local_frame_to_local(frame, -data.w_i);
  const float3 outgoing = local_frame_to_local(frame, outgoing_direction);
  if ((incoming.z <= kEpsilon) || (outgoing.z <= kEpsilon)) {
    return bsdf_eval_zero(data.spectrum_sample);
  }
  const BSDFVelvetPrepared prepared = bsdf_velvet_prepare(context, data, material);
  BSDFEval result = bsdf_eval_zero(data.spectrum_sample);
  if ((prepared.roughness <= kEpsilon) || spectral_response_is_zero(prepared.sheen)) {
    result.func = bsdf_diffuse_eon_brdf(data.spectrum_sample, prepared.diffuse, incoming, outgoing, prepared.roughness);
  } else {
    Sampler walk_sampler = sampler;
    bsdf_sampler_pop_fixed(walk_sampler);
    result.func = bsdf_velvet_function(context, data.spectrum_sample, prepared, incoming, outgoing, walk_sampler);
    sampler.seed = walk_sampler.seed;
  }
  result.bsdf = spectral_response_mul(result.func, outgoing.z);
  result.pdf = 1.0f / kDoublePi;
  result.properties = BSDFSample::Reflection | BSDFSample::Diffuse;
  result.medium_index = data.current_medium;
  result.eta = 1.0f;
  return result;
}

ETX_SHARED_INLINE BSDFSample bsdf_velvet_sample(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(Material, material), ETX_INOUT(Sampler, sampler)) {
  const LocalFrame frame = bsdf_data_get_normal_frame(data, material);
  const float2 random = bsdf_sampler_has_fixed(sampler) ? float2(sampler.fixed_u, sampler.fixed_v) : bsdf_sampler_next_2d(sampler);
  const float3 outgoing = local_frame_from_local(frame, sample_cosine_distribution(random, 0.0f));
  const BSDFEval eval = bsdf_velvet_evaluate(context, data, outgoing, material, sampler);
  BSDFSample result = bsdf_sample_zero(data.spectrum_sample);
  result.w_o = outgoing;
  result.pdf = eval.pdf;
  result.properties = eval.properties;
  result.medium_index = eval.medium_index;
  result.eta = 1.0f;
  if (result.pdf > 0.0f) {
    result.weight = spectral_response_div(eval.bsdf, result.pdf);
  }
  return result;
}

struct BSDFVelvetSubstrateEscape {
  SpectralResponse source ETX_INIT({});
  SpectralResponse absorption ETX_INIT({});
};

ETX_SHARED_INLINE BSDFVelvetWalk bsdf_velvet_walk_begin_escape(ETX_IN(BSDFResourceContext, context), ETX_IN(SpectralQuery, spect), ETX_IN(BSDFVelvetPrepared, prepared),
  ETX_IN(float3, outgoing), float emission_collimation) {
  BSDFVelvetWalk state = ETX_ZERO(BSDFVelvetWalk);
  state.value = spectral_response_zero(spect);
  state.absorption = spectral_response_zero(spect);
  state.throughput = bsdf_velvet_transmission(context, spect, prepared, outgoing.z);
  state.direction = outgoing;
  state.exponent = scene_math_shared_collimation_to_exponent(emission_collimation);
  const float average = bsdf_diffuse_eon_average_albedo(prepared.roughness);
  const SpectralResponse one = spectral_response_make(spect, 1.0f);
  state.multiple = spectral_response_div(spectral_response_mul(spectral_response_mul(prepared.diffuse, prepared.diffuse), average),
    spectral_response_sub(one, spectral_response_mul(prepared.diffuse, 1.0f - average)));
  state.operation = BSDFVelvetWalk::SourceEscape;
  return state;
}

ETX_SHARED_INLINE void bsdf_velvet_walk_escape_step(ETX_IN(BSDFResourceContext, context), ETX_IN(SpectralQuery, spect), ETX_IN(BSDFVelvetPrepared, prepared),
  ETX_INOUT(BSDFVelvetWalk, state), ETX_INOUT(Sampler, sampler)) {
  if (bsdf_velvet_walk_continue(state, sampler) == false) {
    return;
  }
  state.value = spectral_response_add(state.value, spectral_response_mul(state.throughput, scene_math_shared_collimated_emission_scale(state.direction.z, state.exponent)));
  const float single = bsdf_diffuse_eon_directional_albedo(state.direction.z, prepared.roughness);
  const SpectralResponse one = spectral_response_make(spect, 1.0f);
  const SpectralResponse absorptance =
    spectral_response_sub(one, spectral_response_add(spectral_response_mul(prepared.diffuse, single), spectral_response_mul(state.multiple, 1.0f - single)));
  state.absorption = spectral_response_add(state.absorption, spectral_response_mul(state.throughput, absorptance));
  state.throughput = spectral_response_mul(state.throughput, bsdf_velvet_return_sample(context, spect, prepared, state.direction, sampler));
}

ETX_SHARED_NOINLINE BSDFVelvetSubstrateEscape bsdf_velvet_substrate_escape(ETX_IN(BSDFResourceContext, context), ETX_IN(SpectralQuery, spect), ETX_IN(BSDFVelvetPrepared, prepared),
  ETX_IN(float3, outgoing), float emission_collimation, ETX_INOUT(Sampler, sampler)) {
  BSDFVelvetWalk state = bsdf_velvet_walk_begin_escape(context, spect, prepared, outgoing, emission_collimation);
  while (state.operation != BSDFVelvetWalk::Complete) {
    bsdf_velvet_walk_escape_step(context, spect, prepared, state, sampler);
  }
  BSDFVelvetSubstrateEscape result = ETX_ZERO(BSDFVelvetSubstrateEscape);
  result.source = state.value;
  result.absorption = state.absorption;
  return result;
}

ETX_SHARED_INLINE float bsdf_velvet_pdf(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(float3, outgoing_direction), ETX_IN(Material, material),
  ETX_INOUT(Sampler, sampler)) {
  (void)context;
  (void)material;
  (void)sampler;

  const LocalFrame frame = bsdf_data_get_normal_frame(data, material);
  const float3 local_w_i = local_frame_to_local(frame, -data.w_i);
  const float3 local_w_o = local_frame_to_local(frame, outgoing_direction);
  if ((local_w_i.z <= kEpsilon) || (local_w_o.z <= kEpsilon)) {
    return 0.0f;
  }

  return 1.0f / kDoublePi;
}

ETX_SHARED_INLINE bool bsdf_velvet_is_delta(ETX_IN(Material, material), ETX_IN(float2, tex), ETX_INOUT(Sampler, sampler)) {
  (void)material;
  (void)tex;
  (void)sampler;
  return false;
}

ETX_SHARED_INLINE SpectralResponse bsdf_velvet_albedo(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(Material, material), ETX_INOUT(Sampler, sampler)) {
  (void)sampler;
  const SpectralResponse base = bsdf_resource_apply_image(context, data.spectrum_sample, material.scattering, data.tex);
  const SpectralResponse sheen = bsdf_resource_apply_image(context, data.spectrum_sample, material.reflectance, data.tex);
  return spectral_response_min(spectral_response_add(base, sheen), 1.0f);
}
