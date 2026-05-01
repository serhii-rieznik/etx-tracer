#pragma once

#include "bsdf_external_shared.hxx"
#include "bsdf_resource_shared.hxx"

ETX_SHARED_INLINE bool bsdf_dielectric_is_delta(ETX_IN(Material, material), ETX_IN(float2, tex), ETX_INOUT(Sampler, sampler));
ETX_SHARED_INLINE bool bsdf_dielectric_is_delta_with_context(ETX_IN(BSDFResourceContext, context), ETX_IN(Material, material), ETX_IN(float2, tex));

ETX_SHARED_INLINE float bsdf_dielectric_pdf(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(float3, outgoing_direction), ETX_IN(Material, material),
  ETX_INOUT(Sampler, sampler));

ETX_SHARED_INLINE bool bsdf_dielectric_has_thinfilm(ETX_IN(Material, material)) {
  return (material.thinfilm.min_thickness * material.thinfilm.max_thickness) > 0.0f;
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

ETX_SHARED_INLINE BSDFSample bsdf_thinfilm_sample(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(Material, material), ETX_INOUT(Sampler, sampler)) {
  LocalFrame frame = bsdf_data_get_normal_frame(data);
  RefractiveIndexSample ext_ior = bsdf_resource_evaluate_refractive_index(context, material.ext_ior, data.spectrum_sample);
  RefractiveIndexSample int_ior = bsdf_resource_evaluate_refractive_index(context, material.int_ior, data.spectrum_sample);
  ThinfilmEval thinfilm = bsdf_resource_evaluate_thinfilm(context, data.spectrum_sample, material.thinfilm, data.tex, sampler);
  SpectralResponse fr = bsdf_fresnel_calculate(data.spectrum_sample, dot(data.w_i, data.nrm), ext_ior, int_ior, thinfilm);
  float f = spectral_response_monochromatic(fr);

  BSDFSample result = ETX_ZERO(BSDFSample);
  if (bsdf_sampler_next(sampler) <= f) {
    result.w_o = normalize(reflect(data.w_i, frame.nrm));
    result.pdf = f;
    result.weight = spectral_response_mul(bsdf_resource_apply_image(context, data.spectrum_sample, material.reflectance, data.tex), spectral_response_div(fr, f));
    result.properties = BSDFSample::Delta | BSDFSample::Reflection;
    result.medium_index = data.current_medium;
  } else {
    result.w_o = data.w_i;
    result.pdf = 1.0f - f;
    SpectralResponse one_minus_fr = spectral_response_sub(spectral_response_make(data.spectrum_sample, 1.0f), fr);
    result.weight = spectral_response_mul(bsdf_resource_apply_image(context, data.spectrum_sample, material.scattering, data.tex), spectral_response_div(one_minus_fr, 1.0f - f));
    result.properties = BSDFSample::Delta | BSDFSample::Transmission | BSDFSample::MediumChanged;
    result.medium_index = local_frame_entering_material(frame) ? material.int_medium : material.ext_medium;
  }

  result.eta = 1.0f;
  return result;
}

ETX_SHARED_INLINE BSDFEval bsdf_thinfilm_evaluate(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(float3, outgoing_direction), ETX_IN(Material, material),
  ETX_INOUT(Sampler, sampler)) {
  (void)context;
  (void)outgoing_direction;
  (void)material;
  (void)sampler;
  return bsdf_eval_zero(data.spectrum_sample);
}

ETX_SHARED_INLINE float bsdf_thinfilm_pdf(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(float3, outgoing_direction), ETX_IN(Material, material),
  ETX_INOUT(Sampler, sampler)) {
  (void)context;
  (void)data;
  (void)outgoing_direction;
  (void)material;
  (void)sampler;
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

ETX_SHARED_INLINE BSDFSample bsdf_dielectric_sample(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(Material, material), ETX_INOUT(Sampler, sampler)) {
  LocalFrame local_frame = ETX_ZERO(LocalFrame);
  local_frame.tan = data.tan;
  local_frame.btn = data.btn;
  local_frame.nrm = data.nrm;

  float3 w_i = local_frame_to_local(local_frame, -data.w_i);
  bool in_outside = LocalFrame::cos_theta(w_i) > 0.0f;
  float direction_scale = in_outside ? 1.0f : -1.0f;

  RefractiveIndexSample ext_ior = bsdf_resource_evaluate_refractive_index(context, material.ext_ior, data.spectrum_sample);
  RefractiveIndexSample int_ior = bsdf_resource_evaluate_refractive_index(context, material.int_ior, data.spectrum_sample);
  if (in_outside == false) {
    ext_ior = bsdf_resource_evaluate_refractive_index(context, material.int_ior, data.spectrum_sample);
    int_ior = bsdf_resource_evaluate_refractive_index(context, material.ext_ior, data.spectrum_sample);
  }
  ThinfilmEval thinfilm = bsdf_resource_evaluate_thinfilm(context, data.spectrum_sample, material.thinfilm, data.tex, sampler);

  if ((thinfilm.thickness <= 0.0f) && bsdf_dielectric_equal_eta(ext_ior, int_ior)) {
    return bsdf_dielectric_equal_eta_sample(data, material);
  }

  BSDFSample result = ETX_ZERO(BSDFSample);
  result.weight = spectral_response_make(data.spectrum_sample, 1.0f);

  float2 roughness = bsdf_resource_evaluate_roughness(context, material, data.tex);
  BSDFExternalRayInfo ray = bsdf_external_ray_info_make(-direction_scale * w_i, roughness);
  ray = bsdf_external_ray_info_update_height(ray, 1.0f);
  bool ray_outside = true;

  uint32_t scattering_order = 0u;
  while (true) {
    float sampled_height = bsdf_external_sample_height(ray, bsdf_sampler_next(sampler));
    if (sampled_height == kMaxFloat) {
      break;
    }

    ray = bsdf_external_ray_info_update_height(ray, sampled_height);

    float2 rnd_slope = ((scattering_order == 0u) && bsdf_sampler_has_fixed(sampler)) ? float2(sampler.fixed_u, sampler.fixed_v) : bsdf_sampler_next_2d(sampler);
    float rnd_reflection = ((scattering_order == 0u) && bsdf_sampler_has_fixed(sampler)) ? sampler.fixed_w : bsdf_sampler_next(sampler);
    RefractiveIndexSample phase_ext_ior = ext_ior;
    RefractiveIndexSample phase_int_ior = int_ior;
    if (ray_outside == false) {
      phase_ext_ior = int_ior;
      phase_int_ior = ext_ior;
    }
    BSDFExternalDielectricSample sample =
      bsdf_external_sample_phase_function_dielectric(data.spectrum_sample, rnd_slope, rnd_reflection, -ray.w, roughness, phase_ext_ior, phase_int_ior, thinfilm);

    result.weight = spectral_response_mul(result.weight, sample.weight);

    if (sample.reflection) {
      ray = bsdf_external_ray_info_update_direction(ray, sample.w_o, roughness);
      ray = bsdf_external_ray_info_update_height(ray, ray.h);
    } else {
      ray_outside = (ray_outside == false);
      ray = bsdf_external_ray_info_update_direction(ray, -sample.w_o, roughness);
      ray = bsdf_external_ray_info_update_height(ray, -ray.h);
    }

    scattering_order += 1u;
    if (scattering_order > kBSDFExternalScatteringOrderMax) {
      return bsdf_sample_zero(data.spectrum_sample);
    }
  }

  result.w_o = direction_scale * (ray_outside ? ray.w : -ray.w);
  uint32_t delta_sample = bsdf_dielectric_is_delta_with_context(context, material, data.tex) ? BSDFSample::Delta : 0u;

  if ((LocalFrame::cos_theta(w_i) * LocalFrame::cos_theta(result.w_o)) > 0.0f) {
    result.eta = 1.0f;
    result.weight = spectral_response_mul(spectral_response_div(result.weight, spectral_response_monochromatic(result.weight)),
      bsdf_resource_apply_image(context, data.spectrum_sample, material.reflectance, data.tex));
    result.properties = BSDFSample::Reflection | delta_sample;
    result.medium_index = data.current_medium;
  } else {
    float eta = spectral_response_monochromatic(spectral_response_div(int_ior.eta, ext_ior.eta));
    result.eta = eta;
    result.weight = spectral_response_mul(spectral_response_div(result.weight, spectral_response_monochromatic(result.weight)),
      bsdf_resource_apply_image(context, data.spectrum_sample, material.scattering, data.tex));
    result.properties = BSDFSample::Transmission | BSDFSample::MediumChanged | delta_sample;
    result.medium_index = in_outside ? material.int_medium : material.ext_medium;
  }

  result.w_o = normalize(local_frame_from_local(local_frame, result.w_o));
  result.pdf = bsdf_dielectric_pdf(context, data, result.w_o, material, sampler);
  return result;
}

ETX_SHARED_INLINE BSDFEval bsdf_dielectric_evaluate(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(float3, outgoing_direction), ETX_IN(Material, material),
  ETX_INOUT(Sampler, sampler)) {
  LocalFrame local_frame = ETX_ZERO(LocalFrame);
  local_frame.tan = data.tan;
  local_frame.btn = data.btn;
  local_frame.nrm = data.nrm;

  float3 w_i = local_frame_to_local(local_frame, -data.w_i);
  if (abs(LocalFrame::cos_theta(w_i)) <= kEpsilon) {
    return bsdf_eval_zero(data.spectrum_sample);
  }

  float3 w_o = local_frame_to_local(local_frame, outgoing_direction);
  if (abs(LocalFrame::cos_theta(w_o)) <= kEpsilon) {
    return bsdf_eval_zero(data.spectrum_sample);
  }

  float2 roughness = bsdf_resource_evaluate_roughness(context, material, data.tex);
  RefractiveIndexSample ext_ior = bsdf_resource_evaluate_refractive_index(context, material.ext_ior, data.spectrum_sample);
  RefractiveIndexSample int_ior = bsdf_resource_evaluate_refractive_index(context, material.int_ior, data.spectrum_sample);
  ThinfilmEval thinfilm = bsdf_resource_evaluate_thinfilm(context, data.spectrum_sample, material.thinfilm, data.tex, sampler);
  if ((thinfilm.thickness <= 0.0f) && bsdf_dielectric_equal_eta(ext_ior, int_ior)) {
    return bsdf_eval_zero(data.spectrum_sample);
  }

  bool forward_path = data.path_source == PathSource::Camera;
  float backward_scale = abs(1.0f / LocalFrame::cos_theta(w_i));

  SpectralResponse value = spectral_response_make(data.spectrum_sample, 0.0f);
  if (LocalFrame::cos_theta(w_i) > 0.0f) {
    if (LocalFrame::cos_theta(w_o) >= 0.0f) {
      if (forward_path) {
        value = bsdf_external_eval_dielectric(data.spectrum_sample, sampler, w_i, w_o, true, roughness, ext_ior, int_ior, thinfilm);
      } else {
        value = spectral_response_mul(bsdf_external_eval_dielectric(data.spectrum_sample, sampler, w_o, w_i, true, roughness, ext_ior, int_ior, thinfilm), backward_scale);
      }
    } else {
      if (forward_path) {
        value = bsdf_external_eval_dielectric(data.spectrum_sample, sampler, w_i, w_o, false, roughness, ext_ior, int_ior, thinfilm);
      } else {
        value = spectral_response_mul(bsdf_external_eval_dielectric(data.spectrum_sample, sampler, -w_o, -w_i, false, roughness, int_ior, ext_ior, thinfilm), backward_scale);
      }
    }
  } else if (LocalFrame::cos_theta(w_o) <= 0.0f) {
    if (forward_path) {
      value = bsdf_external_eval_dielectric(data.spectrum_sample, sampler, -w_i, -w_o, true, roughness, int_ior, ext_ior, thinfilm);
    } else {
      value = spectral_response_mul(bsdf_external_eval_dielectric(data.spectrum_sample, sampler, -w_o, -w_i, true, roughness, int_ior, ext_ior, thinfilm), backward_scale);
    }
  } else {
    if (forward_path) {
      value = bsdf_external_eval_dielectric(data.spectrum_sample, sampler, -w_i, -w_o, false, roughness, int_ior, ext_ior, thinfilm);
    } else {
      value = spectral_response_mul(bsdf_external_eval_dielectric(data.spectrum_sample, sampler, w_o, w_i, false, roughness, ext_ior, int_ior, thinfilm), backward_scale);
    }
  }

  if (spectral_response_is_zero(value)) {
    return bsdf_eval_zero(data.spectrum_sample);
  }

  bool reflection = (LocalFrame::cos_theta(w_i) * LocalFrame::cos_theta(w_o)) > 0.0f;
  SpectralImage scattering_image = material.scattering;
  if (reflection) {
    scattering_image = material.reflectance;
  }

  BSDFEval eval = ETX_ZERO(BSDFEval);
  const float abs_cos_theta_o = abs(LocalFrame::cos_theta(w_o));
  eval.bsdf = spectral_response_mul(spectral_response_mul(value, 2.0f), bsdf_resource_apply_image(context, data.spectrum_sample, scattering_image, data.tex));
  eval.func = spectral_response_div(eval.bsdf, abs_cos_theta_o);
  eval.pdf = bsdf_dielectric_pdf(context, data, outgoing_direction, material, sampler);
  eval.eta = 1.0f;
  return eval;
}

ETX_SHARED_INLINE float bsdf_dielectric_pdf(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(float3, outgoing_direction), ETX_IN(Material, material),
  ETX_INOUT(Sampler, sampler)) {
  LocalFrame local_frame = ETX_ZERO(LocalFrame);
  local_frame.tan = data.tan;
  local_frame.btn = data.btn;
  local_frame.nrm = data.nrm;

  float3 w_i = local_frame_to_local(local_frame, -data.w_i);
  if (abs(LocalFrame::cos_theta(w_i)) <= kEpsilon) {
    return 0.0f;
  }

  float3 w_o = local_frame_to_local(local_frame, outgoing_direction);
  if (abs(LocalFrame::cos_theta(w_o)) <= kEpsilon) {
    return 0.0f;
  }

  float2 roughness = bsdf_resource_evaluate_roughness(context, material, data.tex);
  RefractiveIndexSample ext_ior = bsdf_resource_evaluate_refractive_index(context, material.ext_ior, data.spectrum_sample);
  RefractiveIndexSample int_ior = bsdf_resource_evaluate_refractive_index(context, material.int_ior, data.spectrum_sample);
  ThinfilmEval thinfilm = bsdf_resource_evaluate_thinfilm(context, data.spectrum_sample, material.thinfilm, data.tex, sampler);
  if ((thinfilm.thickness <= 0.0f) && bsdf_dielectric_equal_eta(ext_ior, int_ior)) {
    return 0.0f;
  }

  bool outside = LocalFrame::cos_theta(w_i) > 0.0f;
  bool reflection = (LocalFrame::cos_theta(w_i) * LocalFrame::cos_theta(w_o)) > 0.0f;

  float3 wh = float3(0.0f, 0.0f, 0.0f);
  float dwh_dwo = 0.0f;
  if (reflection) {
    wh = normalize(w_o + w_i);
    dwh_dwo = 1.0f / (4.0f * dot(w_o, wh));
  } else {
    float eta =
      outside ? spectral_response_monochromatic(spectral_response_div(int_ior.eta, ext_ior.eta)) : spectral_response_monochromatic(spectral_response_div(ext_ior.eta, int_ior.eta));
    wh = normalize(w_i + w_o * eta);
    float sqrt_denom = dot(w_i, wh) + eta * dot(w_o, wh);
    dwh_dwo = (eta * eta) * dot(w_o, wh) / (sqrt_denom * sqrt_denom);
  }

  wh *= (LocalFrame::cos_theta(wh) >= 0.0f) ? 1.0f : -1.0f;

  BSDFExternalRayInfo ray = bsdf_external_ray_info_make(w_i * (outside ? 1.0f : -1.0f), roughness);
  float d_ggx = bsdf_external_d_ggx(wh, roughness);
  float prob = max(0.0f, dot(wh, ray.w) * d_ggx / ((1.0f + ray.Lambda) * LocalFrame::cos_theta(ray.w)));
  RefractiveIndexSample fresnel_ext_ior = ext_ior;
  RefractiveIndexSample fresnel_int_ior = int_ior;
  if (outside == false) {
    fresnel_ext_ior = int_ior;
    fresnel_int_ior = ext_ior;
  }
  float f = spectral_response_monochromatic(bsdf_fresnel_calculate(data.spectrum_sample, dot(w_i, wh), fresnel_ext_ior, fresnel_int_ior, thinfilm));

  if (reflection) {
    prob *= f;
  } else {
    prob *= 1.0f - f;
  }

  float result = abs(prob * dwh_dwo) + abs(LocalFrame::cos_theta(w_o));
  return result;
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
