#pragma once

#include "bsdf_external_shared.hxx"
#include "bsdf_resource_shared.hxx"

ETX_SHARED_INLINE BSDFSample bsdf_void_sample(
  ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(Material, material), ETX_INOUT(Sampler, sampler)) {
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

ETX_SHARED_INLINE BSDFEval bsdf_void_evaluate(
  ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(float3, outgoing_direction), ETX_IN(Material, material), ETX_INOUT(Sampler, sampler)) {
  (void)context;
  (void)outgoing_direction;
  (void)material;
  (void)sampler;
  return bsdf_eval_zero(data.spectrum_sample);
}

ETX_SHARED_INLINE float bsdf_void_pdf(
  ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(float3, outgoing_direction), ETX_IN(Material, material), ETX_INOUT(Sampler, sampler)) {
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

ETX_SHARED_INLINE SpectralResponse bsdf_void_albedo(
  ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(Material, material), ETX_INOUT(Sampler, sampler)) {
  (void)context;
  (void)material;
  (void)sampler;
  return spectral_response_zero(data.spectrum_sample);
}

ETX_SHARED_INLINE BSDFEval bsdf_diffuse_layer(
  ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(float3, local_w_i), ETX_IN(float3, local_w_o), ETX_IN(Material, material),
  ETX_INOUT(Sampler, sampler)) {
  if (local_w_o.z <= 0.0f) {
    return bsdf_eval_zero(data.spectrum_sample);
  }

  SpectralResponse diffuse = bsdf_resource_apply_image(context, data.spectrum_sample, material.scattering, data.tex);
  BSDFEval result = ETX_ZERO(BSDFEval);
  result.func = spectral_response_zero(data.spectrum_sample);
  result.bsdf = spectral_response_zero(data.spectrum_sample);
  result.eta = 1.0f;

  float2 roughness = bsdf_resource_evaluate_roughness(context, material, data.tex);
  if (material.diffuse_variation == 1u) {
    result.bsdf = bsdf_external_eval_diffuse(sampler, local_w_i, local_w_o, roughness, diffuse);
    if (local_w_o.z > 0.0f) {
      result.func = spectral_response_div(result.bsdf, local_w_o.z);
    }
  } else if (material.diffuse_variation == 2u) {
    result.func = bsdf_external_vmf_diffuse_brdf(local_w_i, local_w_o, roughness, diffuse);
    result.bsdf = spectral_response_mul(result.func, local_w_o.z);
  } else {
    result.func = spectral_response_mul(diffuse, kInvPi);
    result.bsdf = spectral_response_mul(result.func, local_w_o.z);
  }

  result.pdf = kInvPi * local_w_o.z;
  return result;
}

ETX_SHARED_INLINE BSDFSample bsdf_diffuse_sample(
  ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(Material, material), ETX_INOUT(Sampler, sampler)) {
  LocalFrame frame = bsdf_data_get_normal_frame(data, material);
  float3 local_w_i = local_frame_to_local(frame, -data.w_i);
  float2 roughness = bsdf_resource_evaluate_roughness(context, material, data.tex);

  BSDFSample result = ETX_ZERO(BSDFSample);
  result.weight = spectral_response_zero(data.spectrum_sample);
  result.eta = 1.0f;
  result.properties = BSDFSample::Reflection | BSDFSample::Diffuse;

  float3 local_w_o = float3(0.0f, 0.0f, 0.0f);
  if (material.diffuse_variation == 1u) {
    SpectralResponse diffuse = bsdf_resource_apply_image(context, data.spectrum_sample, material.scattering, data.tex);
    local_w_o = bsdf_external_sample_diffuse(sampler, local_w_i, roughness, diffuse, result.weight);
    result.pdf = kInvPi * local_w_o.z;
  } else {
    float2 cosine_rnd = bsdf_sampler_has_fixed(sampler) ? float2(sampler.fixed_u, sampler.fixed_v) : bsdf_sampler_next_2d(sampler);
    local_w_o = sample_cosine_distribution(cosine_rnd, 1.0f);
    BSDFEval layer = bsdf_diffuse_layer(context, data, local_w_i, local_w_o, material, sampler);
    if (layer.pdf > 0.0f) {
      result.weight = spectral_response_div(layer.bsdf, layer.pdf);
    }
    result.pdf = layer.pdf;
  }

  result.w_o = local_frame_from_local(frame, local_w_o);
  result.medium_index = data.current_medium;
  return result;
}

ETX_SHARED_INLINE BSDFEval bsdf_diffuse_evaluate(
  ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(float3, outgoing_direction), ETX_IN(Material, material), ETX_INOUT(Sampler, sampler)) {
  LocalFrame frame = bsdf_data_get_normal_frame(data, material);
  float3 local_w_o = local_frame_to_local(frame, outgoing_direction);
  if (local_w_o.z <= kEpsilon) {
    return bsdf_eval_zero(data.spectrum_sample);
  }

  float3 local_w_i = local_frame_to_local(frame, -data.w_i);
  return bsdf_diffuse_layer(context, data, local_w_i, local_w_o, material, sampler);
}

ETX_SHARED_INLINE float bsdf_diffuse_pdf(
  ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(float3, outgoing_direction), ETX_IN(Material, material), ETX_INOUT(Sampler, sampler)) {
  (void)context;
  (void)sampler;

  LocalFrame frame = bsdf_data_get_normal_frame(data, material);
  float n_dot_o = dot(frame.nrm, outgoing_direction);
  if (n_dot_o <= kEpsilon) {
    return 0.0f;
  }

  return kInvPi * n_dot_o;
}

ETX_SHARED_INLINE bool bsdf_diffuse_is_delta(ETX_IN(Material, material), ETX_IN(float2, tex), ETX_INOUT(Sampler, sampler)) {
  (void)material;
  (void)tex;
  (void)sampler;
  return false;
}

ETX_SHARED_INLINE SpectralResponse bsdf_diffuse_albedo(
  ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(Material, material), ETX_INOUT(Sampler, sampler)) {
  (void)sampler;
  return bsdf_resource_apply_image(context, data.spectrum_sample, material.scattering, data.tex);
}

ETX_SHARED_INLINE BSDFSample bsdf_translucent_sample(
  ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(Material, material), ETX_INOUT(Sampler, sampler)) {
  LocalFrame frame = bsdf_data_get_normal_frame(data);
  SpectralResponse transmission = bsdf_resource_apply_image(context, data.spectrum_sample, material.scattering, data.tex);
  SpectralResponse reflection = bsdf_resource_apply_image(context, data.spectrum_sample, material.reflectance, data.tex);

  float transmission_value = spectral_response_monochromatic(transmission);
  float reflection_value = spectral_response_monochromatic(reflection);
  float total = transmission_value + reflection_value;
  if (total == 0.0f) {
    return bsdf_sample_zero(data.spectrum_sample);
  }

  float3 sampled_direction = sample_cosine_distribution(bsdf_sampler_next_2d(sampler), frame.nrm, 1.0f);
  float n_dot_o = abs(dot(sampled_direction, frame.nrm));

  BSDFSample result = ETX_ZERO(BSDFSample);
  result.weight = spectral_response_zero(data.spectrum_sample);
  result.eta = 1.0f;
  if (bsdf_sampler_next(sampler) < (transmission_value / total)) {
    result.w_o = -sampled_direction;
    result.pdf = n_dot_o * kInvPi * (transmission_value / total);
    result.properties = BSDFSample::Diffuse | BSDFSample::Transmission | BSDFSample::MediumChanged;
    result.medium_index = local_frame_entering_material(frame) ? material.int_medium : material.ext_medium;
    result.weight = transmission;
  } else {
    result.w_o = sampled_direction;
    result.pdf = n_dot_o * kInvPi * (reflection_value / total);
    result.properties = BSDFSample::Diffuse | BSDFSample::Reflection;
    result.medium_index = data.current_medium;
    result.weight = reflection;
  }

  return result;
}

ETX_SHARED_INLINE BSDFEval bsdf_translucent_evaluate(
  ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(float3, outgoing_direction), ETX_IN(Material, material), ETX_INOUT(Sampler, sampler)) {
  (void)sampler;

  LocalFrame frame = bsdf_data_get_normal_frame(data);
  float n_dot_i = -dot(frame.nrm, data.w_i);
  float n_dot_o = dot(frame.nrm, outgoing_direction);
  bool reflection = (n_dot_o * n_dot_i) > 0.0f;

  SpectralResponse transmission = bsdf_resource_apply_image(context, data.spectrum_sample, material.scattering, data.tex);
  SpectralResponse reflection_value = bsdf_resource_apply_image(context, data.spectrum_sample, material.reflectance, data.tex);

  float transmission_strength = spectral_response_monochromatic(transmission);
  float reflection_strength = spectral_response_monochromatic(reflection_value);
  float total = transmission_strength + reflection_strength;
  if (total == 0.0f) {
    return bsdf_eval_zero(data.spectrum_sample);
  }

  float scale = (total > 1.0f) ? (1.0f / total) : 1.0f;
  float abs_n_dot_o = abs(n_dot_o);

  BSDFEval result = ETX_ZERO(BSDFEval);
  if (reflection) {
    result.func = spectral_response_mul(reflection_value, scale * kInvPi);
  } else {
    result.func = spectral_response_mul(transmission, scale * kInvPi);
  }
  result.bsdf = spectral_response_mul(result.func, abs_n_dot_o);
  result.pdf = kInvPi * abs_n_dot_o * (reflection ? (reflection_strength / total) : (transmission_strength / total));
  result.eta = 1.0f;
  return result;
}

ETX_SHARED_INLINE float bsdf_translucent_pdf(
  ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(float3, outgoing_direction), ETX_IN(Material, material), ETX_INOUT(Sampler, sampler)) {
  (void)sampler;

  LocalFrame frame = bsdf_data_get_normal_frame(data);
  float n_dot_i = -dot(frame.nrm, data.w_i);
  float n_dot_o = dot(frame.nrm, outgoing_direction);
  float transmission_value = spectral_response_monochromatic(bsdf_resource_apply_image(context, data.spectrum_sample, material.scattering, data.tex));
  float reflection_value = spectral_response_monochromatic(bsdf_resource_apply_image(context, data.spectrum_sample, material.reflectance, data.tex));
  float total = transmission_value + reflection_value;
  bool reflection = (n_dot_o * n_dot_i) > 0.0f;
  if (total == 0.0f) {
    return 0.0f;
  }

  return kInvPi * abs(n_dot_o) * (reflection ? (reflection_value / total) : (transmission_value / total));
}

ETX_SHARED_INLINE bool bsdf_translucent_is_delta(ETX_IN(Material, material), ETX_IN(float2, tex), ETX_INOUT(Sampler, sampler)) {
  (void)material;
  (void)tex;
  (void)sampler;
  return false;
}

ETX_SHARED_INLINE SpectralResponse bsdf_translucent_albedo(
  ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(Material, material), ETX_INOUT(Sampler, sampler)) {
  (void)sampler;
  return bsdf_resource_apply_image(context, data.spectrum_sample, material.scattering, data.tex);
}

ETX_SHARED_INLINE BSDFSample bsdf_mirror_sample(
  ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(Material, material), ETX_INOUT(Sampler, sampler)) {
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

ETX_SHARED_INLINE BSDFEval bsdf_mirror_evaluate(
  ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(float3, outgoing_direction), ETX_IN(Material, material), ETX_INOUT(Sampler, sampler)) {
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

ETX_SHARED_INLINE float bsdf_mirror_pdf(
  ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(float3, outgoing_direction), ETX_IN(Material, material), ETX_INOUT(Sampler, sampler)) {
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

ETX_SHARED_INLINE SpectralResponse bsdf_mirror_albedo(
  ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(Material, material), ETX_INOUT(Sampler, sampler)) {
  (void)context;
  (void)material;
  (void)sampler;
  return spectral_response_make(data.spectrum_sample, 1.0f);
}

ETX_SHARED_INLINE BSDFSample bsdf_boundary_sample(
  ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(Material, material), ETX_INOUT(Sampler, sampler)) {
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

ETX_SHARED_INLINE BSDFEval bsdf_boundary_evaluate(
  ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(float3, outgoing_direction), ETX_IN(Material, material), ETX_INOUT(Sampler, sampler)) {
  (void)context;
  (void)outgoing_direction;
  (void)material;
  (void)sampler;
  return bsdf_eval_zero(data.spectrum_sample);
}

ETX_SHARED_INLINE float bsdf_boundary_pdf(
  ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(float3, outgoing_direction), ETX_IN(Material, material), ETX_INOUT(Sampler, sampler)) {
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

ETX_SHARED_INLINE SpectralResponse bsdf_boundary_albedo(
  ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(Material, material), ETX_INOUT(Sampler, sampler)) {
  (void)context;
  (void)material;
  (void)sampler;
  return spectral_response_make(data.spectrum_sample, 1.0f);
}
