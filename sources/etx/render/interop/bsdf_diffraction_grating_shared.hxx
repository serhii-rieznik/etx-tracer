#pragma once

#include "bsdf_resource_shared.hxx"

struct BSDFDiffractionOrderRange {
  int minimum ETX_INIT(0);
  int maximum ETX_INIT(-1);
};

ETX_SHARED_INLINE bool bsdf_diffraction_grating_wavelength_valid(float wavelength_nm) {
  return isfinite(wavelength_nm) && (wavelength_nm >= kShortestWavelength) && (wavelength_nm <= kLongestWavelength);
}

ETX_SHARED_INLINE bool bsdf_diffraction_grating_period_valid(float period_nm) {
  return isfinite(period_nm) && (period_nm >= kDiffractionGratingMinimumPeriodNm) && (period_nm <= kDiffractionGratingMaximumPeriodNm);
}

ETX_SHARED_INLINE bool bsdf_diffraction_grating_optical_path_difference_valid(float optical_path_difference_nm) {
  return isfinite(optical_path_difference_nm) && (optical_path_difference_nm >= kDiffractionGratingMinimumOpticalPathDifferenceNm) &&
         (optical_path_difference_nm <= kDiffractionGratingMaximumOpticalPathDifferenceNm);
}

ETX_SHARED_INLINE bool bsdf_diffraction_grating_direction_finite(ETX_IN(float3, direction)) {
  return isfinite(direction.x) && isfinite(direction.y) && isfinite(direction.z);
}

ETX_SHARED_INLINE bool bsdf_diffraction_grating_material_valid(ETX_IN(Material, material)) {
  return bsdf_diffraction_grating_period_valid(material.diffraction_grating.period_nm) &&
         bsdf_diffraction_grating_optical_path_difference_valid(material.diffraction_grating.optical_path_difference_nm) && isfinite(material.diffraction_grating.duty_cycle) &&
         (material.diffraction_grating.duty_cycle >= 0.0f) && (material.diffraction_grating.duty_cycle <= 1.0f) && isfinite(material.diffraction_grating.rotation);
}

ETX_SHARED_INLINE LocalFrame bsdf_diffraction_grating_frame(ETX_IN(BSDFData, data), ETX_IN(Material, material)) {
  LocalFrame result = bsdf_data_get_normal_frame(data, material);
  const float angle = fmod(material.diffraction_grating.rotation, kDoublePi);
  const float cosine = cos(angle);
  const float sine = sin(angle);
  const float3 original_tangent = result.tan;
  const float3 original_bitangent = result.btn;
  result.tan = normalize(cosine * original_tangent + sine * original_bitangent);
  result.btn = normalize(-sine * original_tangent + cosine * original_bitangent);
  return result;
}

ETX_SHARED_INLINE BSDFDiffractionOrderRange bsdf_diffraction_grating_order_range(ETX_IN(float3, local_w_i), float wavelength_nm, float period_nm) {
  BSDFDiffractionOrderRange result = ETX_ZERO(BSDFDiffractionOrderRange);
  result.maximum = -1;
  if ((bsdf_diffraction_grating_direction_finite(local_w_i) == false) || (local_w_i.z <= kEpsilon) || (bsdf_diffraction_grating_wavelength_valid(wavelength_nm) == false) ||
      (bsdf_diffraction_grating_period_valid(period_nm) == false)) {
    return result;
  }

  const float remaining_tangent = 1.0f - local_w_i.y * local_w_i.y;
  if (remaining_tangent <= 0.0f) {
    return result;
  }

  const float tangent_limit = max(0.0f, sqrt(remaining_tangent) - kEpsilon);
  const float period_over_wavelength = period_nm / wavelength_nm;
  result.minimum = int(ceil((local_w_i.x - tangent_limit) * period_over_wavelength));
  result.maximum = int(floor((local_w_i.x + tangent_limit) * period_over_wavelength));
  return result;
}

ETX_SHARED_INLINE bool bsdf_diffraction_grating_order_direction(ETX_IN(float3, local_w_i), float wavelength_nm, float period_nm, int order, ETX_OUT(float3, local_w_o)) {
  if ((bsdf_diffraction_grating_direction_finite(local_w_i) == false) || (local_w_i.z <= kEpsilon) || (bsdf_diffraction_grating_wavelength_valid(wavelength_nm) == false) ||
      (bsdf_diffraction_grating_period_valid(period_nm) == false)) {
    local_w_o = float3(0.0f, 0.0f, 0.0f);
    return false;
  }

  const float order_shift = float(order) * wavelength_nm / period_nm;
  const float tangent_x = -local_w_i.x + order_shift;
  const float tangent_y = -local_w_i.y;
  if (order == 0) {
    local_w_o = float3(tangent_x, tangent_y, local_w_i.z);
    return local_w_o.z > kEpsilon;
  }
  const float normal_squared = 1.0f - tangent_x * tangent_x - tangent_y * tangent_y;
  if (normal_squared < 0.0f) {
    local_w_o = float3(0.0f, 0.0f, 0.0f);
    return false;
  }

  local_w_o = float3(tangent_x, tangent_y, sqrt(max(0.0f, normal_squared)));
  return local_w_o.z > kEpsilon;
}

// Fourier power of a lossless binary phase profile with ridge fraction
// `duty_cycle` and ridge-to-groove phase difference `phase_difference`.
// For a single phase difference, Parseval's identity makes the sum over all
// integer orders exactly one.
ETX_SHARED_INLINE float bsdf_diffraction_grating_binary_phase_fourier_power(int order, float duty_cycle, float phase_difference) {
  if ((isfinite(duty_cycle) == false) || (isfinite(phase_difference) == false) || (duty_cycle < 0.0f) || (duty_cycle > 1.0f)) {
    return 0.0f;
  }

  const float phase_sine = sin(0.5f * phase_difference);
  const float phase_contrast = 4.0f * phase_sine * phase_sine;
  if (order == 0) {
    return max(0.0f, 1.0f - phase_contrast * duty_cycle * (1.0f - duty_cycle));
  }

  const float order_value = float(order);
  const float aperture = sin(kPi * order_value * duty_cycle) / (kPi * order_value);
  return max(0.0f, phase_contrast * aperture * aperture);
}

// The ideal binary phase mask is defined directly by an optical path
// difference between its two regions.
// Its phase contrast is independent of the selected output order, so all
// Fourier coefficients belong to one phase function and satisfy Parseval's
// identity. This is a reciprocal scalar phase-mask model, not an
// electromagnetic solution for a geometrical surface-relief profile.
ETX_SHARED_INLINE float bsdf_diffraction_grating_phase_difference(ETX_IN(float3, local_w_i), ETX_IN(float3, local_w_o), float wavelength_nm, float optical_path_difference_nm) {
  if ((bsdf_diffraction_grating_direction_finite(local_w_i) == false) || (bsdf_diffraction_grating_direction_finite(local_w_o) == false) ||
      (bsdf_diffraction_grating_wavelength_valid(wavelength_nm) == false) || (bsdf_diffraction_grating_optical_path_difference_valid(optical_path_difference_nm) == false)) {
    return 0.0f;
  }
  return kDoublePi * optical_path_difference_nm / wavelength_nm;
}

ETX_SHARED_INLINE float bsdf_diffraction_grating_sideband_efficiency(ETX_IN(float3, local_w_i), ETX_IN(float3, local_w_o), float wavelength_nm, int order,
  ETX_IN(Material, material)) {
  if (order == 0) {
    return 0.0f;
  }
  const float phase_difference = bsdf_diffraction_grating_phase_difference(local_w_i, local_w_o, wavelength_nm, material.diffraction_grating.optical_path_difference_nm);
  return bsdf_diffraction_grating_binary_phase_fourier_power(order, material.diffraction_grating.duty_cycle, phase_difference);
}

ETX_SHARED_INLINE float bsdf_diffraction_grating_propagating_sideband_efficiency(ETX_IN(BSDFDiffractionOrderRange, range), ETX_IN(float3, local_w_i), float wavelength_nm,
  float period_nm, ETX_IN(Material, material)) {
  float result = 0.0f;
  for (int order = range.minimum; order <= range.maximum; ++order) {
    if (order == 0) {
      continue;
    }
    float3 local_w_o = float3(0.0f, 0.0f, 0.0f);
    if (bsdf_diffraction_grating_order_direction(local_w_i, wavelength_nm, period_nm, order, local_w_o)) {
      result += bsdf_diffraction_grating_sideband_efficiency(local_w_i, local_w_o, wavelength_nm, order, material);
    }
  }
  return min(1.0f, max(0.0f, result));
}

ETX_SHARED_INLINE float bsdf_diffraction_grating_zero_order_efficiency(ETX_IN(BSDFDiffractionOrderRange, range), ETX_IN(float3, local_w_i), float wavelength_nm, float period_nm,
  ETX_IN(Material, material)) {
  (void)range;
  float3 local_w_o = float3(0.0f, 0.0f, 0.0f);
  if (bsdf_diffraction_grating_order_direction(local_w_i, wavelength_nm, period_nm, 0, local_w_o) == false) {
    return 0.0f;
  }
  const float phase_difference = bsdf_diffraction_grating_phase_difference(local_w_i, local_w_o, wavelength_nm, material.diffraction_grating.optical_path_difference_nm);
  return bsdf_diffraction_grating_binary_phase_fourier_power(0, material.diffraction_grating.duty_cycle, phase_difference);
}

ETX_SHARED_INLINE float bsdf_diffraction_grating_order_efficiency(ETX_IN(BSDFDiffractionOrderRange, range), ETX_IN(float3, local_w_i), ETX_IN(float3, local_w_o),
  float wavelength_nm, float period_nm, int order, ETX_IN(Material, material)) {
  if (order == 0) {
    return bsdf_diffraction_grating_zero_order_efficiency(range, local_w_i, wavelength_nm, period_nm, material);
  }
  return bsdf_diffraction_grating_sideband_efficiency(local_w_i, local_w_o, wavelength_nm, order, material);
}

ETX_SHARED_INLINE float bsdf_diffraction_grating_propagating_efficiency(ETX_IN(BSDFDiffractionOrderRange, range), ETX_IN(float3, local_w_i), float wavelength_nm, float period_nm,
  ETX_IN(Material, material)) {
  if (range.maximum < range.minimum) {
    return 0.0f;
  }
  const float sidebands = bsdf_diffraction_grating_propagating_sideband_efficiency(range, local_w_i, wavelength_nm, period_nm, material);
  const float zero_order = bsdf_diffraction_grating_zero_order_efficiency(range, local_w_i, wavelength_nm, period_nm, material);
  return min(1.0f, max(0.0f, zero_order + sidebands));
}

ETX_SHARED_INLINE bool bsdf_diffraction_grating_match_order(ETX_IN(float3, local_w_i), ETX_IN(float3, local_w_o), float wavelength_nm, float period_nm,
  ETX_OUT(int, matched_order)) {
  matched_order = 0;
  if ((bsdf_diffraction_grating_direction_finite(local_w_i) == false) || (bsdf_diffraction_grating_direction_finite(local_w_o) == false) ||
      (bsdf_diffraction_grating_wavelength_valid(wavelength_nm) == false) || (bsdf_diffraction_grating_period_valid(period_nm) == false)) {
    return false;
  }

  const float continuous_order = (local_w_o.x + local_w_i.x) * period_nm / wavelength_nm;
  matched_order = int(round(continuous_order));
  if (abs(continuous_order - float(matched_order)) > 1.0e-4f) {
    return false;
  }

  float3 expected_direction = float3(0.0f, 0.0f, 0.0f);
  if (bsdf_diffraction_grating_order_direction(local_w_i, wavelength_nm, period_nm, matched_order, expected_direction) == false) {
    return false;
  }
  const float tangent_error = max(abs(expected_direction.x - local_w_o.x), abs(expected_direction.y - local_w_o.y));
  return (tangent_error <= 1.0e-4f) && (abs(expected_direction.z - local_w_o.z) <= 1.0e-4f);
}

ETX_SHARED_INLINE float bsdf_diffraction_grating_rgb_component(ETX_IN(float3, value), uint32_t channel) {
  if (channel == 0u) {
    return value.x;
  }
  if (channel == 1u) {
    return value.y;
  }
  return value.z;
}

ETX_SHARED_INLINE float3 bsdf_diffraction_grating_rgb_set_component(ETX_IN(float3, value), uint32_t channel, float component) {
  float3 result = value;
  if (channel == 0u) {
    result.x = component;
  } else if (channel == 1u) {
    result.y = component;
  } else {
    result.z = component;
  }
  return result;
}

ETX_SHARED_INLINE float bsdf_diffraction_grating_rgb_wavelength(uint32_t channel) {
  return bsdf_diffraction_grating_rgb_component(kRGBWavelengths, channel);
}

ETX_SHARED_INLINE float bsdf_diffraction_grating_rgb_wavelength_span(uint32_t channel) {
  return bsdf_diffraction_grating_rgb_component(kRGBWavelengthsSpan, channel);
}

ETX_SHARED_INLINE float bsdf_diffraction_grating_rgb_sample_wavelength(uint32_t channel, float sample) {
  const float wavelength = bsdf_diffraction_grating_rgb_wavelength(channel);
  const float span = bsdf_diffraction_grating_rgb_wavelength_span(channel);
  const float clamped_sample = clamp(sample, 0.0f, 1.0f - kEpsilon);
  return wavelength + span * (2.0f * clamped_sample - 1.0f);
}

ETX_SHARED_INLINE float3 bsdf_diffraction_grating_rgb_efficiencies(ETX_IN(float3, local_w_i), ETX_IN(Material, material)) {
  float3 result = float3(0.0f, 0.0f, 0.0f);
  for (uint32_t channel = 0u; channel < 3u; ++channel) {
    const float wavelength_nm = bsdf_diffraction_grating_rgb_wavelength(channel);
    const BSDFDiffractionOrderRange range = bsdf_diffraction_grating_order_range(local_w_i, wavelength_nm, material.diffraction_grating.period_nm);
    const float efficiency = bsdf_diffraction_grating_propagating_efficiency(range, local_w_i, wavelength_nm, material.diffraction_grating.period_nm, material);
    result = bsdf_diffraction_grating_rgb_set_component(result, channel, efficiency);
  }
  return result;
}

struct BSDFDiffractionRGBEvaluation {
  float3 value ETX_INIT({});
  float pdf ETX_INIT(0.0f);
};

ETX_SHARED_INLINE BSDFDiffractionRGBEvaluation bsdf_diffraction_grating_rgb_evaluate_local(ETX_IN(float3, local_w_i), ETX_IN(float3, local_w_o), ETX_IN(float3, reflectance),
  ETX_IN(Material, material)) {
  BSDFDiffractionRGBEvaluation result = ETX_ZERO(BSDFDiffractionRGBEvaluation);
  const float3 total_efficiency = bsdf_diffraction_grating_rgb_efficiencies(local_w_i, material);
  const float3 channel_weight = reflectance * total_efficiency;
  const float total_weight = channel_weight.x + channel_weight.y + channel_weight.z;
  if (total_weight <= kEpsilon) {
    return result;
  }

  for (uint32_t channel = 0u; channel < 3u; ++channel) {
    const float channel_total_efficiency = bsdf_diffraction_grating_rgb_component(total_efficiency, channel);
    const float channel_sampling_weight = bsdf_diffraction_grating_rgb_component(channel_weight, channel);
    if ((channel_total_efficiency <= kEpsilon) || (channel_sampling_weight <= 0.0f)) {
      continue;
    }

    const float wavelength_nm = bsdf_diffraction_grating_rgb_wavelength(channel);
    const BSDFDiffractionOrderRange range = bsdf_diffraction_grating_order_range(local_w_i, wavelength_nm, material.diffraction_grating.period_nm);
    int matched_order = 0;
    if ((bsdf_diffraction_grating_match_order(local_w_i, local_w_o, wavelength_nm, material.diffraction_grating.period_nm, matched_order) == false) ||
        (matched_order < range.minimum) || (matched_order > range.maximum)) {
      continue;
    }

    const float efficiency = bsdf_diffraction_grating_order_efficiency(range, local_w_i, local_w_o, wavelength_nm, material.diffraction_grating.period_nm, matched_order, material);
    if (efficiency <= 0.0f) {
      continue;
    }

    const float channel_reflectance = bsdf_diffraction_grating_rgb_component(reflectance, channel);
    result.value = bsdf_diffraction_grating_rgb_set_component(result.value, channel, channel_reflectance * efficiency);
    result.pdf += (channel_sampling_weight / total_weight) * (efficiency / channel_total_efficiency);
  }
  return result;
}

ETX_SHARED_INLINE BSDFSample bsdf_diffraction_grating_sample_spectral(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(Material, material),
  ETX_INOUT(Sampler, sampler)) {
  if ((spectral_query_is_spectral(data.spectrum_sample) == false) || (bsdf_diffraction_grating_wavelength_valid(data.spectrum_sample.wavelength) == false) ||
      (bsdf_diffraction_grating_material_valid(material) == false)) {
    return bsdf_sample_zero(data.spectrum_sample);
  }

  const LocalFrame frame = bsdf_diffraction_grating_frame(data, material);
  const float3 local_w_i = local_frame_to_local(frame, -data.w_i);
  const float wavelength_nm = data.spectrum_sample.wavelength;
  const float period_nm = material.diffraction_grating.period_nm;
  const BSDFDiffractionOrderRange range = bsdf_diffraction_grating_order_range(local_w_i, wavelength_nm, period_nm);
  const float total_efficiency = bsdf_diffraction_grating_propagating_efficiency(range, local_w_i, wavelength_nm, period_nm, material);
  const SpectralResponse reflectance = spectral_response_saturate(bsdf_resource_apply_image(context, data.spectrum_sample, material.reflectance, data.tex));
  if ((range.maximum < range.minimum) || (total_efficiency <= kEpsilon) || (spectral_response_monochromatic(reflectance) <= 0.0f)) {
    return bsdf_sample_zero(data.spectrum_sample);
  }

  const float target = min(bsdf_sampler_next(sampler), 1.0f - kEpsilon) * total_efficiency;
  float accumulated = 0.0f;
  float selected_efficiency = 0.0f;
  float3 selected_w_o = float3(0.0f, 0.0f, 0.0f);
  float last_efficiency = 0.0f;
  float3 last_w_o = float3(0.0f, 0.0f, 0.0f);
  for (int order = range.minimum; order <= range.maximum; ++order) {
    float3 local_w_o = float3(0.0f, 0.0f, 0.0f);
    if (bsdf_diffraction_grating_order_direction(local_w_i, wavelength_nm, period_nm, order, local_w_o) == false) {
      continue;
    }
    const float efficiency = bsdf_diffraction_grating_order_efficiency(range, local_w_i, local_w_o, wavelength_nm, period_nm, order, material);
    if (efficiency <= 0.0f) {
      continue;
    }
    last_efficiency = efficiency;
    last_w_o = local_w_o;
    accumulated += efficiency;
    if (target < accumulated) {
      selected_efficiency = efficiency;
      selected_w_o = local_w_o;
      break;
    }
  }

  // The analytic order powers sum to total_efficiency. Retain the last
  // positive order if floating-point accumulation ends a few ulps below the
  // sampled target.
  if ((selected_efficiency <= 0.0f) && (last_efficiency > 0.0f)) {
    selected_efficiency = last_efficiency;
    selected_w_o = last_w_o;
  }

  if (selected_efficiency <= 0.0f) {
    return bsdf_sample_zero(data.spectrum_sample);
  }

  BSDFSample result = ETX_ZERO(BSDFSample);
  result.w_o = normalize(local_frame_from_local(frame, selected_w_o));
  result.pdf = selected_efficiency / total_efficiency;
  result.weight = spectral_response_mul(reflectance, total_efficiency);
  result.properties = BSDFSample::Delta | BSDFSample::Reflection | BSDFSample::WavelengthDependentDirection;
  result.medium_index = data.current_medium;
  result.eta = 1.0f;
  return result;
}

ETX_SHARED_INLINE BSDFEval bsdf_diffraction_grating_evaluate_spectral(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(float3, outgoing_direction),
  ETX_IN(Material, material), ETX_INOUT(Sampler, sampler)) {
  (void)sampler;
  BSDFEval result = bsdf_eval_zero(data.spectrum_sample);
  if ((spectral_query_is_spectral(data.spectrum_sample) == false) || (bsdf_diffraction_grating_wavelength_valid(data.spectrum_sample.wavelength) == false) ||
      (bsdf_diffraction_grating_material_valid(material) == false)) {
    return result;
  }

  const LocalFrame frame = bsdf_diffraction_grating_frame(data, material);
  const float3 local_w_i = local_frame_to_local(frame, -data.w_i);
  const float3 local_w_o = local_frame_to_local(frame, normalize(outgoing_direction));
  if ((local_w_i.z <= kEpsilon) || (local_w_o.z <= kEpsilon)) {
    return result;
  }

  const float wavelength_nm = data.spectrum_sample.wavelength;
  const float period_nm = material.diffraction_grating.period_nm;
  const BSDFDiffractionOrderRange range = bsdf_diffraction_grating_order_range(local_w_i, wavelength_nm, period_nm);
  const float total_efficiency = bsdf_diffraction_grating_propagating_efficiency(range, local_w_i, wavelength_nm, period_nm, material);
  int matched_order = 0;
  if ((total_efficiency <= kEpsilon) || (bsdf_diffraction_grating_match_order(local_w_i, local_w_o, wavelength_nm, period_nm, matched_order) == false) ||
      (matched_order < range.minimum) || (matched_order > range.maximum)) {
    return result;
  }

  const float efficiency = bsdf_diffraction_grating_order_efficiency(range, local_w_i, local_w_o, wavelength_nm, period_nm, matched_order, material);
  if (efficiency <= 0.0f) {
    return result;
  }

  const SpectralResponse reflectance = spectral_response_saturate(bsdf_resource_apply_image(context, data.spectrum_sample, material.reflectance, data.tex));
  result.func = spectral_response_mul(reflectance, efficiency);
  result.bsdf = result.func;
  result.pdf = efficiency / total_efficiency;
  result.properties = BSDFSample::Delta | BSDFSample::Reflection;
  result.medium_index = data.current_medium;
  result.eta = 1.0f;
  return result;
}

ETX_SHARED_INLINE float bsdf_diffraction_grating_pdf_spectral(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(float3, outgoing_direction),
  ETX_IN(Material, material), ETX_INOUT(Sampler, sampler)) {
  (void)context;
  (void)sampler;
  if ((spectral_query_is_spectral(data.spectrum_sample) == false) || (bsdf_diffraction_grating_wavelength_valid(data.spectrum_sample.wavelength) == false) ||
      (bsdf_diffraction_grating_material_valid(material) == false)) {
    return 0.0f;
  }

  const LocalFrame frame = bsdf_diffraction_grating_frame(data, material);
  const float3 local_w_i = local_frame_to_local(frame, -data.w_i);
  const float3 local_w_o = local_frame_to_local(frame, normalize(outgoing_direction));
  if ((local_w_i.z <= kEpsilon) || (local_w_o.z <= kEpsilon)) {
    return 0.0f;
  }

  const float wavelength_nm = data.spectrum_sample.wavelength;
  const float period_nm = material.diffraction_grating.period_nm;
  const BSDFDiffractionOrderRange range = bsdf_diffraction_grating_order_range(local_w_i, wavelength_nm, period_nm);
  const float total_efficiency = bsdf_diffraction_grating_propagating_efficiency(range, local_w_i, wavelength_nm, period_nm, material);
  int matched_order = 0;
  if ((total_efficiency <= kEpsilon) || (bsdf_diffraction_grating_match_order(local_w_i, local_w_o, wavelength_nm, period_nm, matched_order) == false) ||
      (matched_order < range.minimum) || (matched_order > range.maximum)) {
    return 0.0f;
  }

  return bsdf_diffraction_grating_order_efficiency(range, local_w_i, local_w_o, wavelength_nm, period_nm, matched_order, material) / total_efficiency;
}

ETX_SHARED_INLINE bool bsdf_diffraction_grating_is_delta(ETX_IN(Material, material), ETX_IN(float2, tex), ETX_INOUT(Sampler, sampler)) {
  (void)material;
  (void)tex;
  (void)sampler;
  return true;
}

ETX_SHARED_INLINE SpectralResponse bsdf_diffraction_grating_albedo_spectral(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(Material, material),
  ETX_INOUT(Sampler, sampler)) {
  (void)sampler;
  if ((spectral_query_is_spectral(data.spectrum_sample) == false) || (bsdf_diffraction_grating_wavelength_valid(data.spectrum_sample.wavelength) == false) ||
      (bsdf_diffraction_grating_material_valid(material) == false)) {
    return spectral_response_zero(data.spectrum_sample);
  }
  const LocalFrame frame = bsdf_diffraction_grating_frame(data, material);
  const float3 local_w_i = local_frame_to_local(frame, -data.w_i);
  const float wavelength_nm = data.spectrum_sample.wavelength;
  const float period_nm = material.diffraction_grating.period_nm;
  const BSDFDiffractionOrderRange range = bsdf_diffraction_grating_order_range(local_w_i, wavelength_nm, period_nm);
  const float total_efficiency = bsdf_diffraction_grating_propagating_efficiency(range, local_w_i, wavelength_nm, period_nm, material);
  const SpectralResponse reflectance = spectral_response_saturate(bsdf_resource_apply_image(context, data.spectrum_sample, material.reflectance, data.tex));
  return spectral_response_mul(reflectance, total_efficiency);
}

ETX_SHARED_INLINE BSDFSample bsdf_diffraction_grating_sample_rgb(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(Material, material),
  ETX_INOUT(Sampler, sampler)) {
  if (bsdf_diffraction_grating_material_valid(material) == false) {
    return bsdf_sample_zero(data.spectrum_sample);
  }

  const LocalFrame frame = bsdf_diffraction_grating_frame(data, material);
  const float3 local_w_i = local_frame_to_local(frame, -data.w_i);
  const SpectralResponse reflectance_response = spectral_response_saturate(bsdf_resource_apply_image(context, data.spectrum_sample, material.reflectance, data.tex));
  const float3 reflectance = reflectance_response.integrated;
  const float total_reflectance = reflectance.x + reflectance.y + reflectance.z;
  if (total_reflectance <= kEpsilon) {
    return bsdf_sample_zero(data.spectrum_sample);
  }

  const float channel_target = min(bsdf_sampler_next(sampler), 1.0f - kEpsilon) * total_reflectance;
  float accumulated_reflectance = 0.0f;
  uint32_t selected_channel = 0u;
  for (uint32_t channel = 0u; channel < 3u; ++channel) {
    const float channel_reflectance = bsdf_diffraction_grating_rgb_component(reflectance, channel);
    if (channel_reflectance <= 0.0f) {
      continue;
    }
    selected_channel = channel;
    accumulated_reflectance += channel_reflectance;
    if (channel_target < accumulated_reflectance) {
      break;
    }
  }

  const float selected_reflectance = bsdf_diffraction_grating_rgb_component(reflectance, selected_channel);
  const float selected_channel_probability = selected_reflectance / total_reflectance;
  const float wavelength_nm = bsdf_diffraction_grating_rgb_sample_wavelength(selected_channel, bsdf_sampler_next(sampler));
  const BSDFDiffractionOrderRange range = bsdf_diffraction_grating_order_range(local_w_i, wavelength_nm, material.diffraction_grating.period_nm);
  const float channel_total_efficiency = bsdf_diffraction_grating_propagating_efficiency(range, local_w_i, wavelength_nm, material.diffraction_grating.period_nm, material);
  if (channel_total_efficiency <= kEpsilon) {
    return bsdf_sample_zero(data.spectrum_sample);
  }
  const float order_target = min(bsdf_sampler_next(sampler), 1.0f - kEpsilon) * channel_total_efficiency;
  float accumulated_order_efficiency = 0.0f;
  float selected_efficiency = 0.0f;
  float3 selected_w_o = float3(0.0f, 0.0f, 0.0f);
  float last_efficiency = 0.0f;
  float3 last_w_o = float3(0.0f, 0.0f, 0.0f);
  for (int order = range.minimum; order <= range.maximum; ++order) {
    float3 local_w_o = float3(0.0f, 0.0f, 0.0f);
    if (bsdf_diffraction_grating_order_direction(local_w_i, wavelength_nm, material.diffraction_grating.period_nm, order, local_w_o) == false) {
      continue;
    }
    const float efficiency = bsdf_diffraction_grating_order_efficiency(range, local_w_i, local_w_o, wavelength_nm, material.diffraction_grating.period_nm, order, material);
    if (efficiency <= 0.0f) {
      continue;
    }
    last_efficiency = efficiency;
    last_w_o = local_w_o;
    accumulated_order_efficiency += efficiency;
    if (order_target < accumulated_order_efficiency) {
      selected_efficiency = efficiency;
      selected_w_o = local_w_o;
      break;
    }
  }

  if ((selected_efficiency <= 0.0f) && (last_efficiency > 0.0f)) {
    selected_efficiency = last_efficiency;
    selected_w_o = last_w_o;
  }
  if (selected_efficiency <= 0.0f) {
    return bsdf_sample_zero(data.spectrum_sample);
  }

  const float sample_pdf = selected_channel_probability * selected_efficiency / channel_total_efficiency;
  if (sample_pdf <= kEpsilon) {
    return bsdf_sample_zero(data.spectrum_sample);
  }

  BSDFSample result = ETX_ZERO(BSDFSample);
  result.w_o = normalize(local_frame_from_local(frame, selected_w_o));
  result.pdf = sample_pdf;
  const float selected_weight = selected_reflectance * channel_total_efficiency / selected_channel_probability;
  result.weight = spectral_response_make(data.spectrum_sample, bsdf_diffraction_grating_rgb_set_component(float3(0.0f, 0.0f, 0.0f), selected_channel, selected_weight));
  result.properties = BSDFSample::Delta | BSDFSample::Reflection | BSDFSample::WavelengthDependentDirection;
  result.medium_index = data.current_medium;
  result.eta = 1.0f;
  return result;
}

ETX_SHARED_INLINE BSDFEval bsdf_diffraction_grating_evaluate_rgb(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(float3, outgoing_direction),
  ETX_IN(Material, material)) {
  BSDFEval result = bsdf_eval_zero(data.spectrum_sample);
  if (bsdf_diffraction_grating_material_valid(material) == false) {
    return result;
  }

  const LocalFrame frame = bsdf_diffraction_grating_frame(data, material);
  const float3 local_w_i = local_frame_to_local(frame, -data.w_i);
  const float3 local_w_o = local_frame_to_local(frame, normalize(outgoing_direction));
  if ((local_w_i.z <= kEpsilon) || (local_w_o.z <= kEpsilon)) {
    return result;
  }

  const SpectralResponse reflectance_response = spectral_response_saturate(bsdf_resource_apply_image(context, data.spectrum_sample, material.reflectance, data.tex));
  const BSDFDiffractionRGBEvaluation evaluation = bsdf_diffraction_grating_rgb_evaluate_local(local_w_i, local_w_o, reflectance_response.integrated, material);
  if (evaluation.pdf <= 0.0f) {
    return result;
  }

  result.func = spectral_response_make(data.spectrum_sample, evaluation.value);
  result.bsdf = result.func;
  result.pdf = evaluation.pdf;
  result.properties = BSDFSample::Delta | BSDFSample::Reflection;
  result.medium_index = data.current_medium;
  result.eta = 1.0f;
  return result;
}

ETX_SHARED_INLINE BSDFSample bsdf_diffraction_grating_sample(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(Material, material),
  ETX_INOUT(Sampler, sampler)) {
  if (spectral_query_is_spectral(data.spectrum_sample)) {
    return bsdf_diffraction_grating_sample_spectral(context, data, material, sampler);
  }
  return bsdf_diffraction_grating_sample_rgb(context, data, material, sampler);
}

ETX_SHARED_INLINE BSDFEval bsdf_diffraction_grating_evaluate(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(float3, outgoing_direction),
  ETX_IN(Material, material), ETX_INOUT(Sampler, sampler)) {
  if (spectral_query_is_spectral(data.spectrum_sample)) {
    return bsdf_diffraction_grating_evaluate_spectral(context, data, outgoing_direction, material, sampler);
  }
  return bsdf_diffraction_grating_evaluate_rgb(context, data, outgoing_direction, material);
}

ETX_SHARED_INLINE float bsdf_diffraction_grating_pdf(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(float3, outgoing_direction), ETX_IN(Material, material),
  ETX_INOUT(Sampler, sampler)) {
  if (spectral_query_is_spectral(data.spectrum_sample)) {
    return bsdf_diffraction_grating_pdf_spectral(context, data, outgoing_direction, material, sampler);
  }
  return bsdf_diffraction_grating_evaluate_rgb(context, data, outgoing_direction, material).pdf;
}

ETX_SHARED_INLINE SpectralResponse bsdf_diffraction_grating_albedo(ETX_IN(BSDFResourceContext, context), ETX_IN(BSDFData, data), ETX_IN(Material, material),
  ETX_INOUT(Sampler, sampler)) {
  if (spectral_query_is_spectral(data.spectrum_sample)) {
    return bsdf_diffraction_grating_albedo_spectral(context, data, material, sampler);
  }
  if (bsdf_diffraction_grating_material_valid(material) == false) {
    return spectral_response_zero(data.spectrum_sample);
  }

  const LocalFrame frame = bsdf_diffraction_grating_frame(data, material);
  const float3 local_w_i = local_frame_to_local(frame, -data.w_i);
  const SpectralResponse reflectance = spectral_response_saturate(bsdf_resource_apply_image(context, data.spectrum_sample, material.reflectance, data.tex));
  return spectral_response_make(data.spectrum_sample, reflectance.integrated * bsdf_diffraction_grating_rgb_efficiencies(local_w_i, material));
}
