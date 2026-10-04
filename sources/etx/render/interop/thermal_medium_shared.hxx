#pragma once

#include "thermal_radiation_shared.hxx"
#include "medium_transmittance_shared.hxx"

#if defined(__cplusplus)
# define ETX_THERMAL_MEDIUM_SPECTRAL_RESPONSE ::SpectralResponse
# define ETX_THERMAL_MEDIUM_SPECTRAL_QUERY    ::SpectralQuery
# define ETX_THERMAL_MEDIUM_PRECISE
#else
# define ETX_THERMAL_MEDIUM_SPECTRAL_RESPONSE SpectralResponse
# define ETX_THERMAL_MEDIUM_SPECTRAL_QUERY    SpectralQuery
# define ETX_THERMAL_MEDIUM_PRECISE           precise
#endif

ETX_SHARED_INLINE float thermal_medium_absorbed_fraction(float extinction, float distance) {
  if (extinction == 0.0f) {
    return 0.0f;
  }
  if (distance >= kMaxFloat) {
    return 1.0f;
  }
  const float optical_depth = extinction * distance;
  return (optical_depth < 0.1f) ? extinction * thermal_medium_segment_factor(extinction, distance) : (1.0f - exp(-optical_depth));
}

ETX_SHARED_INLINE ETX_THERMAL_MEDIUM_SPECTRAL_RESPONSE thermal_medium_segment_radiance(ETX_INOUT(MediumSharedContext, context), ETX_IN(ETX_THERMAL_MEDIUM_SPECTRAL_QUERY, spect),
  ETX_IN(ETX_THERMAL_MEDIUM_SPECTRAL_RESPONSE, source), ETX_IN(ETX_THERMAL_MEDIUM_SPECTRAL_RESPONSE, extinction), ETX_IN(float3, origin), ETX_IN(float3, direction),
  float distance) {
  if ((distance <= 0.0f) || spectral_response_is_zero(source)) {
    return spectral_response_zero(spect);
  }
  if (context.medium_class == Medium::Homogeneous) {
    if (spectral_query_is_spectral(spect)) {
      return spectral_response_make(spect, source.value * thermal_medium_absorbed_fraction(extinction.value, distance));
    }
    const float3 fraction = float3(thermal_medium_absorbed_fraction(extinction.integrated.x, distance), thermal_medium_absorbed_fraction(extinction.integrated.y, distance),
      thermal_medium_absorbed_fraction(extinction.integrated.z, distance));
    return spectral_response_make(spect, source.integrated * fraction);
  }
  if (context.has_grid_data == 0u) {
    return spectral_response_zero(spect);
  }
  const float max_sigma = spectral_response_maximum(extinction);
  if (max_sigma <= 0.0f) {
    return spectral_response_zero(spect);
  }
  if (distance >= kMaxFloat) {
    const float3 center = 0.5f * (context.bounds_min + context.bounds_max);
    distance = (length(origin - center) + length(context.bounds_max - center)) * (1.0f + 2.0f * medium_shared_gamma(3u));
  }
  MediumSharedIntersection intersection = medium_shared_zero_intersection();
  if (medium_shared_intersects_bounds(context.bounds_min, context.bounds_max, origin, direction, distance, intersection) == false) {
    return spectral_response_zero(spect);
  }
  const ETX_THERMAL_MEDIUM_SPECTRAL_RESPONSE one = spectral_response_make(spect, 1.0f);
  ETX_THERMAL_MEDIUM_SPECTRAL_RESPONSE transmittance = one;
  const float3 segment_origin = origin + intersection.world_dir_normalized * intersection.t_min;
  const float segment_length = intersection.t_max - intersection.t_min;
  float t = 0.0f;
  float distance_error = 0.0f;
  // Ratio tracking without roulette keeps 0 <= T <= 1, so the complementary source remains nonnegative.
  while (true) {
    const float random_value = medium_shared_rnd(context);
    const float step = -medium_shared_log(1.0f - random_value) / max_sigma;
    // Retain sub-ULP steps through empty regions instead of rounding the marching position to a fixed value.
    ETX_THERMAL_MEDIUM_PRECISE float corrected_step = step - distance_error;
    ETX_THERMAL_MEDIUM_PRECISE float next_t = t + corrected_step;
    distance_error = (next_t - t) - corrected_step;
    t = next_t;
    if (t >= segment_length) {
      break;
    }
    const float density = medium_shared_density(context, segment_origin + intersection.world_dir_normalized * t);
    const ETX_THERMAL_MEDIUM_SPECTRAL_RESPONSE null_weight = spectral_response_sub(one, spectral_response_div(spectral_response_mul(extinction, density), max_sigma));
    transmittance = spectral_response_mul(transmittance, null_weight);
    if (spectral_response_is_zero(spectral_response_mul(source, transmittance))) {
      break;
    }
  }
  return spectral_response_mul(source, spectral_response_sub(one, transmittance));
}

ETX_SHARED_INLINE float medium_emission_component_integral(ETX_INOUT(MediumSharedContext, context), float emission, float extinction, ETX_IN(float3, origin),
  ETX_IN(float3, direction), float distance, ETX_INOUT(bool, valid)) {
  if (emission <= 0.0f) {
    return 0.0f;
  }
  const float optical_depth = extinction * distance;
  const float rate = max(1.0f, optical_depth);
  if (isfinite(rate) == false) {
    valid = false;
    return 0.0f;
  }
  const float source_scale = (optical_depth >= 1.0f) ? (emission / extinction) : (emission * distance);
  const float null_probability_scale = min(1.0f, optical_depth);
  float transmittance = 1.0f;
  float radiance = 0.0f;
  float radiance_error = 0.0f;
  float t = 0.0f;
  float distance_error = 0.0f;
  // Poisson track-length integration uses j*rho*T/rate before each null-collision update.
  // Independent RGB rates preserve transparent channels without the other channels' extinction cost.
  while (transmittance > 0.0f) {
    const float step = -medium_shared_log(1.0f - medium_shared_rnd(context)) / rate;
    ETX_THERMAL_MEDIUM_PRECISE float corrected_step = step - distance_error;
    ETX_THERMAL_MEDIUM_PRECISE float next_t = t + corrected_step;
    distance_error = (next_t - t) - corrected_step;
    t = next_t;
    if (t >= 1.0f) {
      break;
    }
    const float density = medium_shared_density(context, origin + direction * (t * distance));
    if (density > 0.0f) {
      const float contribution = source_scale * (density * transmittance);
      ETX_THERMAL_MEDIUM_PRECISE float corrected_contribution = contribution - radiance_error;
      ETX_THERMAL_MEDIUM_PRECISE float next_radiance = radiance + corrected_contribution;
      radiance_error = (next_radiance - radiance) - corrected_contribution;
      radiance = next_radiance;
    }
    transmittance *= 1.0f - density * null_probability_scale;
  }
  valid = valid && isfinite(radiance);
  return radiance;
}

ETX_SHARED_INLINE float medium_emission_homogeneous_component(float emission, float extinction, float distance) {
  if (emission <= 0.0f) {
    return 0.0f;
  }
  if (distance >= kMaxFloat) {
    return emission / extinction;
  }
  const float optical_depth = extinction * distance;
  return (optical_depth < 0.1f) ? emission * thermal_medium_segment_factor(extinction, distance) : (emission * (1.0f - exp(-optical_depth))) / extinction;
}

ETX_SHARED_INLINE ETX_THERMAL_MEDIUM_SPECTRAL_RESPONSE medium_authored_segment_radiance(ETX_INOUT(MediumSharedContext, context), ETX_IN(ETX_THERMAL_MEDIUM_SPECTRAL_QUERY, spect),
  ETX_IN(ETX_THERMAL_MEDIUM_SPECTRAL_RESPONSE, emission), ETX_IN(ETX_THERMAL_MEDIUM_SPECTRAL_RESPONSE, extinction), ETX_IN(float3, origin), ETX_IN(float3, direction),
  float distance, ETX_OUT(bool, valid)) {
  valid = true;
  if ((distance <= 0.0f) || spectral_response_is_zero(emission)) {
    return spectral_response_zero(spect);
  }
  if (context.medium_class == Medium::Homogeneous) {
    if (spectral_query_is_spectral(spect)) {
      const float radiance = medium_emission_homogeneous_component(emission.value, extinction.value, distance);
      valid = isfinite(radiance);
      return spectral_response_make(spect, radiance);
    }
    const float3 radiance = float3(medium_emission_homogeneous_component(emission.integrated.x, extinction.integrated.x, distance),
      medium_emission_homogeneous_component(emission.integrated.y, extinction.integrated.y, distance),
      medium_emission_homogeneous_component(emission.integrated.z, extinction.integrated.z, distance));
    valid = isfinite(radiance.x) && isfinite(radiance.y) && isfinite(radiance.z);
    return spectral_response_make(spect, radiance);
  }
  if (context.has_grid_data == 0u) {
    return spectral_response_zero(spect);
  }
  if (distance >= kMaxFloat) {
    const float3 center = 0.5f * (context.bounds_min + context.bounds_max);
    distance = (length(origin - center) + length(context.bounds_max - center)) * (1.0f + 2.0f * medium_shared_gamma(3u));
  }
  MediumSharedIntersection intersection = medium_shared_zero_intersection();
  if (medium_shared_intersects_bounds(context.bounds_min, context.bounds_max, origin, direction, distance, intersection) == false) {
    return spectral_response_zero(spect);
  }
  const float3 segment_origin = origin + intersection.world_dir_normalized * intersection.t_min;
  const float segment_length = intersection.t_max - intersection.t_min;
  if (spectral_query_is_spectral(spect)) {
    return spectral_response_make(spect,
      medium_emission_component_integral(context, emission.value, extinction.value, segment_origin, intersection.world_dir_normalized, segment_length, valid));
  }
  const float red =
    medium_emission_component_integral(context, emission.integrated.x, extinction.integrated.x, segment_origin, intersection.world_dir_normalized, segment_length, valid);
  const float green =
    medium_emission_component_integral(context, emission.integrated.y, extinction.integrated.y, segment_origin, intersection.world_dir_normalized, segment_length, valid);
  const float blue =
    medium_emission_component_integral(context, emission.integrated.z, extinction.integrated.z, segment_origin, intersection.world_dir_normalized, segment_length, valid);
  return spectral_response_make(spect, float3(red, green, blue));
}

#undef ETX_THERMAL_MEDIUM_SPECTRAL_QUERY
#undef ETX_THERMAL_MEDIUM_SPECTRAL_RESPONSE
#undef ETX_THERMAL_MEDIUM_PRECISE
