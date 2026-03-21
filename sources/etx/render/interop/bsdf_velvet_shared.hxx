#pragma once

#include "bsdf_resource_shared.hxx"

ETX_SHARED_INLINE float bsdf_velvet_lambda_l(float roughness, float x) {
  float clamped_x = max(x, 0.0f);
  float inverse_lerp = 1.0f - roughness;
  float inverse_lerp_squared = inverse_lerp * inverse_lerp;
  float a = inverse_lerp_squared * 25.3245f + (1.0f - inverse_lerp_squared) * 21.5473f;
  float b = inverse_lerp_squared * 3.32435f + (1.0f - inverse_lerp_squared) * 3.82987f;
  float c = inverse_lerp_squared * 0.16801f + (1.0f - inverse_lerp_squared) * 0.19823f;
  float d = inverse_lerp_squared * (-1.27393f) + (1.0f - inverse_lerp_squared) * (-1.97760f);
  float e = inverse_lerp_squared * (-4.85967f) + (1.0f - inverse_lerp_squared) * (-4.32054f);
  float q = a / (1.0f + b * pow(clamped_x, c)) + d * clamped_x + e;
  return q;
}

ETX_SHARED_INLINE float bsdf_velvet_lambda(float roughness, float cos_theta) {
  if (cos_theta < 0.5f) {
    return exp(bsdf_velvet_lambda_l(roughness, cos_theta));
  }

  return exp(2.0f * bsdf_velvet_lambda_l(roughness, 0.5f) - bsdf_velvet_lambda_l(roughness, 1.0f - cos_theta));
}

ETX_SHARED_INLINE float bsdf_velvet_fresnel_approximate(float f0, float f90, float cos_theta) {
  return f0 + (f90 - f0) * pow(max(1.0f - cos_theta, 0.0f), 5.0f);
}

ETX_SHARED_INLINE float bsdf_velvet_diffuse_burley(float alpha, float n_dot_i, float n_dot_o, float m_dot_o) {
  float f90 = 0.5f + 2.0f * alpha * m_dot_o * m_dot_o;
  float light_scatter = bsdf_velvet_fresnel_approximate(1.0f, f90, n_dot_o);
  float view_scatter = bsdf_velvet_fresnel_approximate(1.0f, f90, n_dot_i);
  return light_scatter * view_scatter * kInvPi;
}

ETX_SHARED_INLINE BSDFEval bsdf_velvet_evaluate(
  ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(float3, outgoing_direction), ETX_IN(Material, material), ETX_INOUT(Sampler, sampler));

ETX_SHARED_INLINE BSDFSample bsdf_velvet_sample(
  ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(Material, material), ETX_INOUT(Sampler, sampler)) {
  LocalFrame frame = bsdf_data_get_normal_frame(data);
  float3 sampled_direction = sample_cosine_distribution(bsdf_sampler_next_2d(sampler), frame.nrm, 0.0f);
  BSDFEval eval = bsdf_velvet_evaluate(context, data, sampled_direction, material, sampler);

  BSDFSample result = ETX_ZERO(BSDFSample);
  result.w_o = sampled_direction;
  result.properties = BSDFSample::Reflection | BSDFSample::Diffuse;
  result.medium_index = data.current_medium;
  result.eta = 1.0f;
  result.pdf = eval.pdf;
  result.weight = spectral_response_div(eval.bsdf, eval.pdf);
  return result;
}

ETX_SHARED_INLINE BSDFEval bsdf_velvet_evaluate(
  ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(float3, outgoing_direction), ETX_IN(Material, material), ETX_INOUT(Sampler, sampler)) {
  (void)sampler;

  LocalFrame frame = bsdf_data_get_normal_frame(data);
  float n_dot_o = max(0.0f, dot(outgoing_direction, frame.nrm));
  float n_dot_i = max(0.0f, -dot(data.w_i, frame.nrm));
  if ((n_dot_o <= kEpsilon) || (n_dot_i <= kEpsilon)) {
    return bsdf_eval_zero(data.spectrum_sample);
  }

  float3 half_vector = normalize(outgoing_direction - data.w_i);
  float m_dot_o = max(0.0f, dot(outgoing_direction, half_vector));
  float m_dot_i = max(0.0f, -dot(data.w_i, half_vector));
  if ((m_dot_o <= kEpsilon) || (m_dot_i <= kEpsilon)) {
    return bsdf_eval_zero(data.spectrum_sample);
  }

  float2 roughness = bsdf_resource_evaluate_roughness(context, material, data.tex);
  float specular_scale_base = 0.0f;
  float alpha = 0.5f * (roughness.x + roughness.y);
  if (alpha > kEpsilon) {
    float inv_alpha = 1.0f / (kEpsilon + alpha);
    float m_dot_n = dot(half_vector, frame.nrm);
    float sin_theta = 1.0f - m_dot_n * m_dot_n;
    float d = (2.0f + inv_alpha) * pow(sin_theta, 0.5f * inv_alpha) / kDoublePi;
    float lambda_i = bsdf_velvet_lambda(alpha, n_dot_i);
    float lambda_o = bsdf_velvet_lambda(alpha, n_dot_o);
    float g = 1.0f / (1.0f + lambda_i + lambda_o);
    specular_scale_base = 0.25f * d * g / n_dot_i;
  }

  SpectralResponse diffuse = bsdf_resource_apply_image(context, data.spectrum_sample, material.scattering, data.tex);
  SpectralResponse specular = bsdf_resource_apply_image(context, data.spectrum_sample, material.reflectance, data.tex);
  float diffuse_scale = bsdf_velvet_diffuse_burley(alpha, n_dot_i, n_dot_o, m_dot_o);

  BSDFEval result = ETX_ZERO(BSDFEval);
  result.func = spectral_response_add(spectral_response_mul(diffuse, diffuse_scale), spectral_response_mul(specular, specular_scale_base / n_dot_o));
  result.bsdf = spectral_response_add(spectral_response_mul(diffuse, diffuse_scale * n_dot_o), spectral_response_mul(specular, specular_scale_base));
  result.pdf = 1.0f / kDoublePi;
  result.eta = 1.0f;
  return result;
}

ETX_SHARED_INLINE float bsdf_velvet_pdf(
  ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(float3, outgoing_direction), ETX_IN(Material, material), ETX_INOUT(Sampler, sampler)) {
  (void)context;
  (void)outgoing_direction;
  (void)material;
  (void)sampler;

  LocalFrame frame = bsdf_data_get_normal_frame(data);
  if (local_frame_entering_material(frame) == false) {
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

ETX_SHARED_INLINE SpectralResponse bsdf_velvet_albedo(
  ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(Material, material), ETX_INOUT(Sampler, sampler)) {
  (void)sampler;
  return bsdf_resource_apply_image(context, data.spectrum_sample, material.scattering, data.tex);
}
