#pragma once

#include "bsdf_resource_shared.hxx"
#include "bsdf_diffraction_grating_shared.hxx"

ETX_SHARED_INLINE BSDFSample bsdf_void_sample(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(Material, material), ETX_INOUT(Sampler, sampler)) {
  (void)context;
  (void)material;
  (void)sampler;

  BSDFSample result = ETX_ZERO(BSDFSample);
  result.w_o = data.w_i;
  result.weight = spectral_response_zero(data.spectrum_sample);
  result.pdf = 0.0f;
  result.properties = BSDFSample::Delta;
  result.medium_index = data.current_medium;
  result.eta = 1.0f;
  return result;
}

ETX_SHARED_INLINE BSDFEval bsdf_void_evaluate(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(float3, outgoing_direction), ETX_IN(Material, material),
  ETX_INOUT(Sampler, sampler)) {
  (void)context;
  (void)outgoing_direction;
  (void)material;
  (void)sampler;
  return bsdf_eval_zero(data.spectrum_sample);
}

ETX_SHARED_INLINE float bsdf_void_pdf(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(float3, outgoing_direction), ETX_IN(Material, material),
  ETX_INOUT(Sampler, sampler)) {
  (void)context;
  (void)data;
  (void)outgoing_direction;
  (void)material;
  (void)sampler;
  return 0.0f;
}

ETX_SHARED_INLINE bool bsdf_void_is_delta(ETX_IN(Material, material), ETX_IN(float2, tex), ETX_INOUT(Sampler, sampler)) {
  (void)material;
  (void)tex;
  (void)sampler;
  return true;
}

ETX_SHARED_INLINE SpectralResponse bsdf_void_albedo(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(Material, material), ETX_INOUT(Sampler, sampler)) {
  (void)context;
  (void)material;
  (void)sampler;
  return spectral_response_zero(data.spectrum_sample);
}

ETX_SHARED_INLINE float bsdf_diffuse_scalar_roughness(ETX_IN(BSDFResourceContext, context), ETX_IN(Material, material), ETX_IN(float2, uv)) {
  const float2 roughness = bsdf_resource_evaluate_roughness(context, material, uv);
  return saturate(0.5f * (roughness.x + roughness.y));
}

ETX_SHARED_INLINE float bsdf_diffuse_eon_a(float roughness) {
  const float coefficient = 0.5f - 2.0f * kInvPi / 3.0f;
  return 1.0f / (1.0f + coefficient * roughness);
}

ETX_SHARED_INLINE float bsdf_diffuse_eon_directional_albedo(float mu, float roughness) {
  if (roughness <= kEpsilon) {
    return 1.0f;
  }

  const float clamped_mu = min(1.0f, max(0.0f, mu));
  const float a = bsdf_diffuse_eon_a(roughness);
  if (clamped_mu <= kEpsilon) {
    return a * (1.0f + roughness * kInvPi * (0.5f * kPi - 2.0f / 3.0f));
  }

  const float sin_theta = sqrt(max(0.0f, 1.0f - clamped_mu * clamped_mu));
  const float sin_theta_3 = sin_theta * sin_theta * sin_theta;
  const float grazing_term = sin_theta * (acos(clamped_mu) - sin_theta * clamped_mu) + (2.0f / 3.0f) * (((sin_theta / clamped_mu) * (1.0f - sin_theta_3)) - sin_theta);
  return a * (1.0f + roughness * kInvPi * grazing_term);
}

ETX_SHARED_INLINE float bsdf_diffuse_eon_average_albedo(float roughness) {
  const float a = bsdf_diffuse_eon_a(roughness);
  return a * (1.0f + roughness * (2.0f / 3.0f - 28.0f * kInvPi / 15.0f));
}

ETX_SHARED_INLINE SpectralResponse bsdf_diffuse_eon_brdf(ETX_IN(SpectralQuery, spect), ETX_IN(SpectralResponse, albedo), ETX_IN(float3, local_w_i),
  ETX_IN(float3, local_w_o), float roughness) {
  if (roughness <= kEpsilon) {
    return spectral_response_mul(albedo, kInvPi);
  }

  const float mu_i = max(0.0f, local_w_i.z);
  const float mu_o = max(0.0f, local_w_o.z);
  const float s = dot(local_w_i, local_w_o) - mu_i * mu_o;
  const float t = max(mu_i, mu_o);
  const float s_over_t = (s > 0.0f) ? (s / max(kEpsilon, t)) : s;
  const float a = bsdf_diffuse_eon_a(roughness);

  const SpectralResponse f_ss = spectral_response_mul(albedo, kInvPi * a * (1.0f + roughness * s_over_t));

  const float e_i = bsdf_diffuse_eon_directional_albedo(mu_i, roughness);
  const float e_o = bsdf_diffuse_eon_directional_albedo(mu_o, roughness);
  const float e_avg = bsdf_diffuse_eon_average_albedo(roughness);
  const SpectralResponse one = spectral_response_make(spect, 1.0f);
  const SpectralResponse numerator = spectral_response_mul(spectral_response_mul(albedo, albedo), e_avg);
  const SpectralResponse denominator = spectral_response_sub(one, spectral_response_mul(albedo, 1.0f - e_avg));
  const SpectralResponse rho_ms = spectral_response_div(numerator, denominator);
  const float f_ms_scalar = ((1.0f - e_i) * (1.0f - e_o)) / (1.0f - e_avg);
  const SpectralResponse f_ms = spectral_response_mul(rho_ms, kInvPi * f_ms_scalar);
  return spectral_response_add(f_ss, f_ms);
}

ETX_SHARED_INLINE BSDFEval bsdf_diffuse_layer(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(float3, local_w_i), ETX_IN(float3, local_w_o),
  ETX_IN(Material, material), ETX_INOUT(Sampler, sampler)) {
  (void)sampler;

  if ((local_w_i.z <= kEpsilon) || (local_w_o.z <= kEpsilon)) {
    return bsdf_eval_zero(data.spectrum_sample);
  }

  const SpectralResponse diffuse = bsdf_resource_apply_image(context, data.spectrum_sample, material.scattering, data.tex);
  const float roughness = bsdf_diffuse_scalar_roughness(context, material, data.tex);
  BSDFEval result = ETX_ZERO(BSDFEval);
  result.func = bsdf_diffuse_eon_brdf(data.spectrum_sample, diffuse, local_w_i, local_w_o, roughness);
  result.bsdf = spectral_response_mul(result.func, local_w_o.z);
  result.pdf = kInvPi * local_w_o.z;
  result.eta = 1.0f;
  return result;
}

ETX_SHARED_INLINE BSDFSample bsdf_diffuse_sample(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(Material, material), ETX_INOUT(Sampler, sampler)) {
  const LocalFrame frame = bsdf_data_get_normal_frame(data, material);
  const float3 local_w_i = local_frame_to_local(frame, -data.w_i);

  BSDFSample result = ETX_ZERO(BSDFSample);
  result.weight = spectral_response_zero(data.spectrum_sample);
  result.eta = 1.0f;
  result.properties = BSDFSample::Reflection | BSDFSample::Diffuse;

  const bool has_fixed_sample = bsdf_sampler_has_fixed(sampler);
  float2 cosine_rnd = float2(sampler.fixed_u, sampler.fixed_v);
  if (has_fixed_sample == false) {
    cosine_rnd = bsdf_sampler_next_2d(sampler);
  }
  const float3 local_w_o = sample_cosine_distribution(cosine_rnd, 1.0f);
  const BSDFEval layer = bsdf_diffuse_layer(context, data, local_w_i, local_w_o, material, sampler);
  if (layer.pdf > 0.0f) {
    result.weight = spectral_response_div(layer.bsdf, layer.pdf);
  }
  result.pdf = layer.pdf;

  result.w_o = local_frame_from_local(frame, local_w_o);
  result.medium_index = data.current_medium;
  return result;
}

ETX_SHARED_INLINE BSDFEval bsdf_diffuse_evaluate(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(float3, outgoing_direction), ETX_IN(Material, material),
  ETX_INOUT(Sampler, sampler)) {
  const LocalFrame frame = bsdf_data_get_normal_frame(data, material);
  const float3 local_w_o = local_frame_to_local(frame, outgoing_direction);
  if (local_w_o.z <= kEpsilon) {
    return bsdf_eval_zero(data.spectrum_sample);
  }

  const float3 local_w_i = local_frame_to_local(frame, -data.w_i);
  return bsdf_diffuse_layer(context, data, local_w_i, local_w_o, material, sampler);
}

ETX_SHARED_INLINE float bsdf_diffuse_pdf(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(float3, outgoing_direction), ETX_IN(Material, material),
  ETX_INOUT(Sampler, sampler)) {
  (void)context;
  (void)sampler;

  const LocalFrame frame = bsdf_data_get_normal_frame(data, material);
  const float3 local_w_o = local_frame_to_local(frame, outgoing_direction);
  if (local_w_o.z <= kEpsilon) {
    return 0.0f;
  }

  return kInvPi * local_w_o.z;
}

ETX_SHARED_INLINE bool bsdf_diffuse_is_delta(ETX_IN(Material, material), ETX_IN(float2, tex), ETX_INOUT(Sampler, sampler)) {
  (void)material;
  (void)tex;
  (void)sampler;
  return false;
}

ETX_SHARED_INLINE SpectralResponse bsdf_diffuse_albedo(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(Material, material), ETX_INOUT(Sampler, sampler)) {
  (void)sampler;
  return bsdf_resource_apply_image(context, data.spectrum_sample, material.scattering, data.tex);
}

ETX_SHARED_INLINE BSDFSample bsdf_translucent_sample(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(Material, material), ETX_INOUT(Sampler, sampler)) {
  const LocalFrame frame = bsdf_data_get_normal_frame(data, material);
  const float3 local_w_i = local_frame_to_local(frame, -data.w_i);
  if (local_w_i.z <= kEpsilon) {
    return bsdf_sample_zero(data.spectrum_sample);
  }

  const SpectralResponse transmission = bsdf_resource_apply_image(context, data.spectrum_sample, material.scattering, data.tex);
  const SpectralResponse reflection = bsdf_resource_apply_image(context, data.spectrum_sample, material.reflectance, data.tex);

  const float transmission_value = spectral_response_monochromatic(transmission);
  const float reflection_value = spectral_response_monochromatic(reflection);
  const float total = transmission_value + reflection_value;
  if (total == 0.0f) {
    return bsdf_sample_zero(data.spectrum_sample);
  }

  const bool has_fixed_sample = bsdf_sampler_has_fixed(sampler);
  float2 cosine_rnd = float2(sampler.fixed_u, sampler.fixed_v);
  if (has_fixed_sample == false) {
    cosine_rnd = bsdf_sampler_next_2d(sampler);
  }
  const float3 local_sampled_w_o = sample_cosine_distribution(cosine_rnd, 1.0f);
  const float n_dot_o = local_sampled_w_o.z;

  BSDFSample result = ETX_ZERO(BSDFSample);
  result.weight = spectral_response_zero(data.spectrum_sample);
  result.eta = 1.0f;
  const float scale = (total > 1.0f) ? (1.0f / total) : 1.0f;
  const float transmission_probability = transmission_value / total;
  const float reflection_probability = reflection_value / total;
  float branch_rnd = sampler.fixed_w;
  if (has_fixed_sample == false) {
    branch_rnd = bsdf_sampler_next(sampler);
  }
  const bool sample_transmission = (reflection_probability == 0.0f) || (branch_rnd < transmission_probability);
  const float branch_probability = sample_transmission ? transmission_probability : reflection_probability;
  SpectralResponse branch_response = reflection;
  if (sample_transmission) {
    branch_response = transmission;
  }
  branch_response = spectral_response_mul(branch_response, scale);
  const float roughness = bsdf_diffuse_scalar_roughness(context, material, data.tex);
  const SpectralResponse unit_response = spectral_response_make(data.spectrum_sample, 1.0f);
  const SpectralResponse unit_func = bsdf_diffuse_eon_brdf(data.spectrum_sample, unit_response, local_w_i, local_sampled_w_o, roughness);
  const SpectralResponse func = spectral_response_mul(unit_func, branch_response);
  const SpectralResponse bsdf = spectral_response_mul(func, n_dot_o);

  if (sample_transmission) {
    result.w_o = local_frame_from_local(frame, -local_sampled_w_o);
    result.pdf = n_dot_o * kInvPi * branch_probability;
    result.properties = BSDFSample::Diffuse | BSDFSample::Transmission | BSDFSample::MediumChanged;
    result.medium_index = local_frame_entering_material(frame) ? material.int_medium : material.ext_medium;
  } else {
    result.w_o = local_frame_from_local(frame, local_sampled_w_o);
    result.pdf = n_dot_o * kInvPi * branch_probability;
    result.properties = BSDFSample::Diffuse | BSDFSample::Reflection;
    result.medium_index = data.current_medium;
  }
  if (result.pdf > 0.0f) {
    result.weight = spectral_response_div(bsdf, result.pdf);
  }

  return result;
}

ETX_SHARED_INLINE BSDFEval bsdf_translucent_evaluate(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(float3, outgoing_direction), ETX_IN(Material, material),
  ETX_INOUT(Sampler, sampler)) {
  (void)sampler;

  const LocalFrame frame = bsdf_data_get_normal_frame(data, material);
  const float3 local_w_i = local_frame_to_local(frame, -data.w_i);
  if (local_w_i.z <= kEpsilon) {
    return bsdf_eval_zero(data.spectrum_sample);
  }

  const float3 local_w_o = local_frame_to_local(frame, outgoing_direction);
  const bool reflection = local_w_o.z > 0.0f;
  const float abs_n_dot_o = abs(local_w_o.z);
  if (abs_n_dot_o <= kEpsilon) {
    return bsdf_eval_zero(data.spectrum_sample);
  }

  const SpectralResponse transmission = bsdf_resource_apply_image(context, data.spectrum_sample, material.scattering, data.tex);
  const SpectralResponse reflection_value = bsdf_resource_apply_image(context, data.spectrum_sample, material.reflectance, data.tex);

  const float transmission_strength = spectral_response_monochromatic(transmission);
  const float reflection_strength = spectral_response_monochromatic(reflection_value);
  const float total = transmission_strength + reflection_strength;
  if (total == 0.0f) {
    return bsdf_eval_zero(data.spectrum_sample);
  }

  const float scale = (total > 1.0f) ? (1.0f / total) : 1.0f;
  const float branch_probability = reflection ? (reflection_strength / total) : (transmission_strength / total);
  SpectralResponse branch_response = transmission;
  if (reflection) {
    branch_response = reflection_value;
  }
  branch_response = spectral_response_mul(branch_response, scale);
  const float roughness = bsdf_diffuse_scalar_roughness(context, material, data.tex);
  const float3 lobe_w_o = reflection ? local_w_o : -local_w_o;
  const SpectralResponse unit_response = spectral_response_make(data.spectrum_sample, 1.0f);
  const SpectralResponse unit_func = bsdf_diffuse_eon_brdf(data.spectrum_sample, unit_response, local_w_i, lobe_w_o, roughness);
  BSDFEval result = ETX_ZERO(BSDFEval);
  result.func = spectral_response_mul(unit_func, branch_response);
  result.bsdf = spectral_response_mul(result.func, abs_n_dot_o);
  result.pdf = kInvPi * abs_n_dot_o * branch_probability;
  result.eta = 1.0f;
  return result;
}

ETX_SHARED_INLINE float bsdf_translucent_pdf(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(float3, outgoing_direction), ETX_IN(Material, material),
  ETX_INOUT(Sampler, sampler)) {
  (void)sampler;

  const LocalFrame frame = bsdf_data_get_normal_frame(data, material);
  const float3 local_w_i = local_frame_to_local(frame, -data.w_i);
  if (local_w_i.z <= kEpsilon) {
    return 0.0f;
  }

  const float3 local_w_o = local_frame_to_local(frame, outgoing_direction);
  const float abs_n_dot_o = abs(local_w_o.z);
  if (abs_n_dot_o <= kEpsilon) {
    return 0.0f;
  }

  const float transmission_value = spectral_response_monochromatic(bsdf_resource_apply_image(context, data.spectrum_sample, material.scattering, data.tex));
  const float reflection_value = spectral_response_monochromatic(bsdf_resource_apply_image(context, data.spectrum_sample, material.reflectance, data.tex));
  const float total = transmission_value + reflection_value;
  const bool reflection = local_w_o.z > 0.0f;
  if (total == 0.0f) {
    return 0.0f;
  }

  return kInvPi * abs_n_dot_o * (reflection ? (reflection_value / total) : (transmission_value / total));
}

ETX_SHARED_INLINE bool bsdf_translucent_is_delta(ETX_IN(Material, material), ETX_IN(float2, tex), ETX_INOUT(Sampler, sampler)) {
  (void)material;
  (void)tex;
  (void)sampler;
  return false;
}

ETX_SHARED_INLINE SpectralResponse bsdf_translucent_albedo(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(Material, material), ETX_INOUT(Sampler, sampler)) {
  (void)sampler;
  const SpectralResponse transmission = bsdf_resource_apply_image(context, data.spectrum_sample, material.scattering, data.tex);
  const SpectralResponse reflection = bsdf_resource_apply_image(context, data.spectrum_sample, material.reflectance, data.tex);
  const float total = spectral_response_monochromatic(transmission) + spectral_response_monochromatic(reflection);
  if (total == 0.0f) {
    return spectral_response_zero(data.spectrum_sample);
  }
  const float scale = (total > 1.0f) ? (1.0f / total) : 1.0f;
  return spectral_response_mul(spectral_response_add(transmission, reflection), scale);
}

ETX_SHARED_INLINE BSDFSample bsdf_mirror_sample(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(Material, material), ETX_INOUT(Sampler, sampler)) {
  (void)sampler;

  LocalFrame frame = bsdf_data_get_normal_frame(data, material);
  BSDFSample result = ETX_ZERO(BSDFSample);
  result.w_o = normalize(reflect(data.w_i, frame.nrm));
  result.weight = bsdf_resource_apply_image(context, data.spectrum_sample, material.scattering, data.tex);
  result.pdf = 1.0f;
  result.properties = BSDFSample::Delta | BSDFSample::Reflection;
  result.medium_index = data.current_medium;
  result.eta = 1.0f;
  return result;
}

ETX_SHARED_INLINE BSDFEval bsdf_mirror_evaluate(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(float3, outgoing_direction), ETX_IN(Material, material),
  ETX_INOUT(Sampler, sampler)) {
  (void)sampler;

  BSDFEval result = bsdf_eval_zero(data.spectrum_sample);
  LocalFrame frame = bsdf_data_get_normal_frame(data, material);
  float3 ideal_w_o = normalize(reflect(data.w_i, frame.nrm));
  float3 actual_w_o = normalize(outgoing_direction);
  if (direction_matches(ideal_w_o, actual_w_o, 1.0f)) {
    result.func = bsdf_resource_apply_image(context, data.spectrum_sample, material.scattering, data.tex);
    result.bsdf = result.func;
    result.pdf = 1.0f;
  }
  return result;
}

ETX_SHARED_INLINE float bsdf_mirror_pdf(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(float3, outgoing_direction), ETX_IN(Material, material),
  ETX_INOUT(Sampler, sampler)) {
  (void)context;
  (void)sampler;

  LocalFrame frame = bsdf_data_get_normal_frame(data, material);
  float3 ideal_w_o = normalize(reflect(data.w_i, frame.nrm));
  float3 actual_w_o = normalize(outgoing_direction);
  return direction_matches(ideal_w_o, actual_w_o, 1.0f) ? 1.0f : 0.0f;
}

ETX_SHARED_INLINE bool bsdf_mirror_is_delta(ETX_IN(Material, material), ETX_IN(float2, tex), ETX_INOUT(Sampler, sampler)) {
  (void)material;
  (void)tex;
  (void)sampler;
  return true;
}

ETX_SHARED_INLINE SpectralResponse bsdf_mirror_albedo(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(Material, material), ETX_INOUT(Sampler, sampler)) {
  (void)context;
  (void)material;
  (void)sampler;
  return spectral_response_make(data.spectrum_sample, 1.0f);
}

ETX_SHARED_INLINE BSDFSample bsdf_boundary_sample(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(Material, material), ETX_INOUT(Sampler, sampler)) {
  (void)context;
  (void)sampler;

  bool entering_material = dot(data.nrm, data.w_i) < 0.0f;
  BSDFSample result = ETX_ZERO(BSDFSample);
  result.w_o = data.w_i;
  result.pdf = 1.0f;
  result.weight = spectral_response_make(data.spectrum_sample, 1.0f);
  result.properties = BSDFSample::Transmission | BSDFSample::MediumChanged;
  result.medium_index = entering_material ? material.int_medium : material.ext_medium;
  result.eta = 1.0f;
  return result;
}

ETX_SHARED_INLINE BSDFEval bsdf_boundary_evaluate(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(float3, outgoing_direction), ETX_IN(Material, material),
  ETX_INOUT(Sampler, sampler)) {
  (void)context;
  (void)outgoing_direction;
  (void)material;
  (void)sampler;
  return bsdf_eval_zero(data.spectrum_sample);
}

ETX_SHARED_INLINE float bsdf_boundary_pdf(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(float3, outgoing_direction), ETX_IN(Material, material),
  ETX_INOUT(Sampler, sampler)) {
  (void)context;
  (void)data;
  (void)outgoing_direction;
  (void)material;
  (void)sampler;
  return 0.0f;
}

ETX_SHARED_INLINE bool bsdf_boundary_is_delta(ETX_IN(Material, material), ETX_IN(float2, tex), ETX_INOUT(Sampler, sampler)) {
  (void)material;
  (void)tex;
  (void)sampler;
  return false;
}

ETX_SHARED_INLINE SpectralResponse bsdf_boundary_albedo(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(Material, material), ETX_INOUT(Sampler, sampler)) {
  (void)context;
  (void)material;
  (void)sampler;
  return spectral_response_make(data.spectrum_sample, 1.0f);
}
