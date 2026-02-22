#pragma once

#include "medium_transmittance_shared.hxx"

#if defined(__cplusplus)
# define ETX_MEDIUM_SAMPLE_SHARED_MEDIUM_TYPE ::Medium
# define ETX_MEDIUM_SAMPLE_SHARED_SPECTRAL_RESPONSE ::SpectralResponse
# define ETX_MEDIUM_SAMPLE_SHARED_SPECTRAL_QUERY ::SpectralQuery
# define ETX_MEDIUM_SAMPLE_SHARED_SAMPLE ::MediumSample
# define ETX_MEDIUM_SAMPLE_SHARED_SPECTRAL_IS_SPECTRAL ::spectral_response_is_spectral
# define ETX_MEDIUM_SAMPLE_SHARED_QUERY_IS_SPECTRAL ::spectral_query_is_spectral
# define ETX_MEDIUM_SAMPLE_SHARED_SPECTRAL_MAKE ::spectral_response_make
# define ETX_MEDIUM_SAMPLE_SHARED_SPECTRAL_ADD ::spectral_response_add
# define ETX_MEDIUM_SAMPLE_SHARED_SPECTRAL_SUB ::spectral_response_sub
# define ETX_MEDIUM_SAMPLE_SHARED_SPECTRAL_MUL ::spectral_response_mul
# define ETX_MEDIUM_SAMPLE_SHARED_SPECTRAL_DIV ::spectral_response_div
# define ETX_MEDIUM_SAMPLE_SHARED_SPECTRAL_EXP ::spectral_response_exp
# define ETX_MEDIUM_SAMPLE_SHARED_SPECTRAL_IS_ZERO ::spectral_response_is_zero
# define ETX_MEDIUM_SAMPLE_SHARED_SPECTRAL_MAXIMUM ::spectral_response_maximum
# define ETX_MEDIUM_SAMPLE_SHARED_SPECTRAL_CLAMP_NON_NEGATIVE ::spectral_response_clamp_non_negative
#else
# define ETX_MEDIUM_SAMPLE_SHARED_MEDIUM_TYPE Medium
# define ETX_MEDIUM_SAMPLE_SHARED_SPECTRAL_RESPONSE SpectralResponse
# define ETX_MEDIUM_SAMPLE_SHARED_SPECTRAL_QUERY SpectralQuery
# define ETX_MEDIUM_SAMPLE_SHARED_SAMPLE MediumSample
# define ETX_MEDIUM_SAMPLE_SHARED_SPECTRAL_IS_SPECTRAL spectral_response_is_spectral
# define ETX_MEDIUM_SAMPLE_SHARED_QUERY_IS_SPECTRAL spectral_query_is_spectral
# define ETX_MEDIUM_SAMPLE_SHARED_SPECTRAL_MAKE spectral_response_make
# define ETX_MEDIUM_SAMPLE_SHARED_SPECTRAL_ADD spectral_response_add
# define ETX_MEDIUM_SAMPLE_SHARED_SPECTRAL_SUB spectral_response_sub
# define ETX_MEDIUM_SAMPLE_SHARED_SPECTRAL_MUL spectral_response_mul
# define ETX_MEDIUM_SAMPLE_SHARED_SPECTRAL_DIV spectral_response_div
# define ETX_MEDIUM_SAMPLE_SHARED_SPECTRAL_EXP spectral_response_exp
# define ETX_MEDIUM_SAMPLE_SHARED_SPECTRAL_IS_ZERO spectral_response_is_zero
# define ETX_MEDIUM_SAMPLE_SHARED_SPECTRAL_MAXIMUM spectral_response_maximum
# define ETX_MEDIUM_SAMPLE_SHARED_SPECTRAL_CLAMP_NON_NEGATIVE spectral_response_clamp_non_negative
#endif

#ifndef ETX_MEDIUM_SAMPLE_SHARED_CONTEXT_TYPE
# error "ETX_MEDIUM_SAMPLE_SHARED_CONTEXT_TYPE must be defined before including medium_sample_shared.hxx"
#endif

#ifndef ETX_MEDIUM_SAMPLE_SHARED_RND
# error "ETX_MEDIUM_SAMPLE_SHARED_RND must be defined before including medium_sample_shared.hxx"
#endif

#ifndef ETX_MEDIUM_SAMPLE_SHARED_DENSITY
# error "ETX_MEDIUM_SAMPLE_SHARED_DENSITY must be defined before including medium_sample_shared.hxx"
#endif

#ifndef ETX_MEDIUM_SAMPLE_SHARED_LOAD_MEDIUM_CLASS
# error "ETX_MEDIUM_SAMPLE_SHARED_LOAD_MEDIUM_CLASS must be defined before including medium_sample_shared.hxx"
#endif

#ifndef ETX_MEDIUM_SAMPLE_SHARED_HAS_GRID_DATA
# error "ETX_MEDIUM_SAMPLE_SHARED_HAS_GRID_DATA must be defined before including medium_sample_shared.hxx"
#endif

#ifndef ETX_MEDIUM_SAMPLE_SHARED_BOUNDS_MIN
# error "ETX_MEDIUM_SAMPLE_SHARED_BOUNDS_MIN must be defined before including medium_sample_shared.hxx"
#endif

#ifndef ETX_MEDIUM_SAMPLE_SHARED_BOUNDS_MAX
# error "ETX_MEDIUM_SAMPLE_SHARED_BOUNDS_MAX must be defined before including medium_sample_shared.hxx"
#endif

ETX_SHARED_INLINE float medium_sample_shared_response_component(ETX_IN(ETX_MEDIUM_SAMPLE_SHARED_SPECTRAL_RESPONSE, value), uint32_t component_index) {
  if (ETX_MEDIUM_SAMPLE_SHARED_SPECTRAL_IS_SPECTRAL(value)) {
    return value.value;
  }
  if (component_index == 0u) {
    return value.integrated.x;
  }
  if (component_index == 1u) {
    return value.integrated.y;
  }
  return value.integrated.z;
}

ETX_SHARED_INLINE float medium_sample_shared_response_sum(ETX_IN(ETX_MEDIUM_SAMPLE_SHARED_SPECTRAL_RESPONSE, value)) {
  if (ETX_MEDIUM_SAMPLE_SHARED_SPECTRAL_IS_SPECTRAL(value)) {
    return value.value;
  }
  return value.integrated.x + value.integrated.y + value.integrated.z;
}

ETX_SHARED_INLINE ETX_MEDIUM_SAMPLE_SHARED_SPECTRAL_RESPONSE medium_sample_shared_calculate_albedo(ETX_IN(ETX_MEDIUM_SAMPLE_SHARED_SPECTRAL_QUERY, spect),
  ETX_IN(ETX_MEDIUM_SAMPLE_SHARED_SPECTRAL_RESPONSE, scattering), ETX_IN(ETX_MEDIUM_SAMPLE_SHARED_SPECTRAL_RESPONSE, extinction)) {
  ETX_MEDIUM_SAMPLE_SHARED_SPECTRAL_RESPONSE result = ETX_MEDIUM_SAMPLE_SHARED_SPECTRAL_MAKE(spect, 0.0f);
  if ((ETX_MEDIUM_SAMPLE_SHARED_SPECTRAL_IS_SPECTRAL(extinction)) && (ETX_MEDIUM_SAMPLE_SHARED_SPECTRAL_IS_SPECTRAL(scattering))) {
    result = ETX_MEDIUM_SAMPLE_SHARED_SPECTRAL_MAKE(spect, (extinction.value > 0.0f) ? (scattering.value / extinction.value) : 0.0f);
  } else {
    float3 value = make_float3(0.0f, 0.0f, 0.0f);
    value.x = (extinction.integrated.x > 0.0f) ? (scattering.integrated.x / extinction.integrated.x) : 0.0f;
    value.y = (extinction.integrated.y > 0.0f) ? (scattering.integrated.y / extinction.integrated.y) : 0.0f;
    value.z = (extinction.integrated.z > 0.0f) ? (scattering.integrated.z / extinction.integrated.z) : 0.0f;
    result = ETX_MEDIUM_SAMPLE_SHARED_SPECTRAL_MAKE(spect, value);
  }

  return result;
}

ETX_SHARED_INLINE uint32_t medium_sample_shared_sample_spectrum_component(ETX_IN(ETX_MEDIUM_SAMPLE_SHARED_SPECTRAL_QUERY, spect),
  ETX_IN(ETX_MEDIUM_SAMPLE_SHARED_SPECTRAL_RESPONSE, albedo), ETX_IN(ETX_MEDIUM_SAMPLE_SHARED_SPECTRAL_RESPONSE, throughput), float random_value,
  ETX_OUT(ETX_MEDIUM_SAMPLE_SHARED_SPECTRAL_RESPONSE, pdf)) {
  if (ETX_MEDIUM_SAMPLE_SHARED_QUERY_IS_SPECTRAL(spect)) {
    pdf = ETX_MEDIUM_SAMPLE_SHARED_SPECTRAL_MAKE(spect, 1.0f);
    return 0u;
  }

  ETX_MEDIUM_SAMPLE_SHARED_SPECTRAL_RESPONSE at = ETX_MEDIUM_SAMPLE_SHARED_SPECTRAL_MUL(albedo, throughput);
  if (ETX_MEDIUM_SAMPLE_SHARED_SPECTRAL_IS_ZERO(at)) {
    pdf = ETX_MEDIUM_SAMPLE_SHARED_SPECTRAL_MAKE(spect, 1.0f / 3.0f);
    return uint32_t(3.0f * random_value);
  }

  float at_sum = medium_sample_shared_response_sum(at);
  pdf = ETX_MEDIUM_SAMPLE_SHARED_SPECTRAL_DIV(at, at_sum);
  return 2u - uint32_t(random_value < (pdf.integrated.x + pdf.integrated.y)) - uint32_t(random_value < pdf.integrated.x);
}

ETX_SHARED_INLINE ETX_MEDIUM_SAMPLE_SHARED_SAMPLE medium_sample_shared_zero_sample(ETX_IN(ETX_MEDIUM_SAMPLE_SHARED_SPECTRAL_QUERY, spect), ETX_IN(float3, pos),
  float sampled_medium_t) {
  ETX_ZERO_INIT(ETX_MEDIUM_SAMPLE_SHARED_SAMPLE, result);
  result.weight = ETX_MEDIUM_SAMPLE_SHARED_SPECTRAL_MAKE(spect, 0.0f);
  result.pos = pos;
  result.sampled_medium_t = sampled_medium_t;
  return result;
}

ETX_SHARED_INLINE ETX_MEDIUM_SAMPLE_SHARED_SAMPLE medium_sample_shared_sample(ETX_INOUT(ETX_MEDIUM_SAMPLE_SHARED_CONTEXT_TYPE, context),
  ETX_IN(ETX_MEDIUM_SAMPLE_SHARED_SPECTRAL_QUERY, spect), ETX_IN(ETX_MEDIUM_SAMPLE_SHARED_SPECTRAL_RESPONSE, throughput),
  ETX_IN(ETX_MEDIUM_SAMPLE_SHARED_SPECTRAL_RESPONSE, scattering_value), ETX_IN(ETX_MEDIUM_SAMPLE_SHARED_SPECTRAL_RESPONSE, absorption_value), ETX_IN(float3, pos),
  ETX_IN(float3, w_i), float max_t) {
  ETX_MEDIUM_SAMPLE_SHARED_SPECTRAL_RESPONSE extinction_value = ETX_MEDIUM_SAMPLE_SHARED_SPECTRAL_ADD(scattering_value, absorption_value);
  ETX_MEDIUM_SAMPLE_SHARED_SPECTRAL_RESPONSE albedo = medium_sample_shared_calculate_albedo(spect, scattering_value, extinction_value);
  uint32_t medium_class = ETX_MEDIUM_SAMPLE_SHARED_LOAD_MEDIUM_CLASS(context);

  if (medium_class == ETX_MEDIUM_SAMPLE_SHARED_MEDIUM_TYPE::Homogeneous) {
    float t = 0.0f;
    ETX_MEDIUM_SAMPLE_SHARED_SPECTRAL_RESPONSE pdf = ETX_MEDIUM_SAMPLE_SHARED_SPECTRAL_MAKE(spect, 0.0f);
    while (t < kRayEpsilon) {
      uint32_t channel = medium_sample_shared_sample_spectrum_component(spect, albedo, throughput, ETX_MEDIUM_SAMPLE_SHARED_RND(context), pdf);
      float sample_t = medium_sample_shared_response_component(extinction_value, channel);
      t = (sample_t > 0.0f) ? (-medium_shared_log(1.0f - ETX_MEDIUM_SAMPLE_SHARED_RND(context)) / sample_t) : max_t;
    }

    t = min(t, max_t);
    bool sampled_medium = t < max_t;
    ETX_MEDIUM_SAMPLE_SHARED_SPECTRAL_RESPONSE transmittance = ETX_MEDIUM_SAMPLE_SHARED_SPECTRAL_EXP(ETX_MEDIUM_SAMPLE_SHARED_SPECTRAL_MUL(extinction_value, -t));
    ETX_MEDIUM_SAMPLE_SHARED_SPECTRAL_RESPONSE pdf_contribution = transmittance;
    if (sampled_medium) {
      pdf_contribution = ETX_MEDIUM_SAMPLE_SHARED_SPECTRAL_MUL(transmittance, extinction_value);
    }
    pdf = ETX_MEDIUM_SAMPLE_SHARED_SPECTRAL_MUL(pdf, pdf_contribution);

    if (ETX_MEDIUM_SAMPLE_SHARED_SPECTRAL_IS_ZERO(pdf)) {
      return medium_sample_shared_zero_sample(spect, pos + w_i * t, 0.0f);
    }

    ETX_MEDIUM_SAMPLE_SHARED_SPECTRAL_RESPONSE weight_numerator = transmittance;
    if (sampled_medium) {
      weight_numerator = ETX_MEDIUM_SAMPLE_SHARED_SPECTRAL_MUL(transmittance, scattering_value);
    }
    float pdf_sum = medium_sample_shared_response_sum(pdf);

    ETX_ZERO_INIT(ETX_MEDIUM_SAMPLE_SHARED_SAMPLE, result);
    result.pos = pos + w_i * t;
    result.sampled_medium_t = sampled_medium ? t : 0.0f;
    result.weight = ETX_MEDIUM_SAMPLE_SHARED_SPECTRAL_DIV(weight_numerator, pdf_sum);
    return result;
  }

  if (medium_class == ETX_MEDIUM_SAMPLE_SHARED_MEDIUM_TYPE::Heterogeneous) {
    float max_sigma = ETX_MEDIUM_SAMPLE_SHARED_SPECTRAL_MAXIMUM(extinction_value);
    if ((max_sigma <= 0.0f) || (ETX_MEDIUM_SAMPLE_SHARED_HAS_GRID_DATA(context) == false)) {
      return medium_sample_shared_zero_sample(spect, pos + w_i * max_t, 0.0f);
    }

    float3 bounds_min = ETX_MEDIUM_SAMPLE_SHARED_BOUNDS_MIN(context);
    float3 bounds_max = ETX_MEDIUM_SAMPLE_SHARED_BOUNDS_MAX(context);
    MediumSharedIntersection medium_intersection = medium_shared_zero_intersection();
    if (medium_shared_intersects_bounds(bounds_min, bounds_max, pos, w_i, max_t, medium_intersection) == false) {
      ETX_ZERO_INIT(ETX_MEDIUM_SAMPLE_SHARED_SAMPLE, result);
      result.weight = ETX_MEDIUM_SAMPLE_SHARED_SPECTRAL_MAKE(spect, 1.0f);
      result.pos = pos + w_i * max_t;
      result.sampled_medium_t = 0.0f;
      return result;
    }

    ETX_MEDIUM_SAMPLE_SHARED_SPECTRAL_RESPONSE pdf = ETX_MEDIUM_SAMPLE_SHARED_SPECTRAL_MAKE(spect, 0.0f);
    uint32_t channel = medium_sample_shared_sample_spectrum_component(spect, albedo, throughput, ETX_MEDIUM_SAMPLE_SHARED_RND(context), pdf);

    ETX_MEDIUM_SAMPLE_SHARED_SPECTRAL_RESPONSE transmittance = ETX_MEDIUM_SAMPLE_SHARED_SPECTRAL_MAKE(spect, 1.0f);
    float t_world = 0.0f;
    float segment_length = medium_intersection.t_max - medium_intersection.t_min;
    const float rr_threshold = 0.1f;
    while (true) {
      t_world += -medium_shared_log(1.0f - ETX_MEDIUM_SAMPLE_SHARED_RND(context)) / max_sigma;

      float3 world_pos_at_t = pos + medium_intersection.world_dir_normalized * t_world;
      float3 local_pos = medium_shared_bounds_to_local(world_pos_at_t, bounds_min, bounds_max);
      float t_local_along_dir = dot(local_pos - medium_intersection.medium_pos, medium_intersection.medium_dir);
      if (t_local_along_dir >= segment_length) {
        pdf = ETX_MEDIUM_SAMPLE_SHARED_SPECTRAL_MUL(pdf, transmittance);
        ETX_ZERO_INIT(ETX_MEDIUM_SAMPLE_SHARED_SAMPLE, result);
        result.pos = pos + medium_intersection.world_dir_normalized * min(t_world, max_t);
        result.sampled_medium_t = 0.0f;
        if (ETX_MEDIUM_SAMPLE_SHARED_SPECTRAL_IS_ZERO(pdf)) {
          result.weight = ETX_MEDIUM_SAMPLE_SHARED_SPECTRAL_MAKE(spect, 0.0f);
        } else {
          result.weight = ETX_MEDIUM_SAMPLE_SHARED_SPECTRAL_DIV(transmittance, medium_sample_shared_response_sum(pdf));
        }
        return result;
      }

      float density_value = ETX_MEDIUM_SAMPLE_SHARED_DENSITY(context, local_pos);
      ETX_MEDIUM_SAMPLE_SHARED_SPECTRAL_RESPONSE extinction_at_point = ETX_MEDIUM_SAMPLE_SHARED_SPECTRAL_MUL(extinction_value, density_value);
      float sigma_t_channel = medium_sample_shared_response_component(extinction_at_point, channel);
      if ((sigma_t_channel > 0.0f) && (ETX_MEDIUM_SAMPLE_SHARED_RND(context) < (sigma_t_channel / max_sigma))) {
        ETX_MEDIUM_SAMPLE_SHARED_SPECTRAL_RESPONSE scattering_at_point = ETX_MEDIUM_SAMPLE_SHARED_SPECTRAL_MUL(scattering_value, density_value);
        pdf = ETX_MEDIUM_SAMPLE_SHARED_SPECTRAL_MUL(pdf, ETX_MEDIUM_SAMPLE_SHARED_SPECTRAL_MUL(transmittance, extinction_at_point));
        if (ETX_MEDIUM_SAMPLE_SHARED_SPECTRAL_IS_ZERO(pdf)) {
          return medium_sample_shared_zero_sample(spect, world_pos_at_t, 0.0f);
        }

        ETX_ZERO_INIT(ETX_MEDIUM_SAMPLE_SHARED_SAMPLE, result);
        result.pos = world_pos_at_t;
        result.sampled_medium_t = t_world;
        result.weight = ETX_MEDIUM_SAMPLE_SHARED_SPECTRAL_DIV(
          ETX_MEDIUM_SAMPLE_SHARED_SPECTRAL_MUL(transmittance, scattering_at_point), medium_sample_shared_response_sum(pdf));
        return result;
      }

      ETX_MEDIUM_SAMPLE_SHARED_SPECTRAL_RESPONSE step_weight =
        ETX_MEDIUM_SAMPLE_SHARED_SPECTRAL_SUB(ETX_MEDIUM_SAMPLE_SHARED_SPECTRAL_MAKE(spect, 1.0f), ETX_MEDIUM_SAMPLE_SHARED_SPECTRAL_DIV(extinction_at_point, max_sigma));
      step_weight = ETX_MEDIUM_SAMPLE_SHARED_SPECTRAL_CLAMP_NON_NEGATIVE(step_weight);
      transmittance = ETX_MEDIUM_SAMPLE_SHARED_SPECTRAL_MUL(transmittance, step_weight);

      float transmittance_max = ETX_MEDIUM_SAMPLE_SHARED_SPECTRAL_MAXIMUM(transmittance);
      if (transmittance_max < rr_threshold) {
        float p = clamp(transmittance_max, 0.01f, 0.95f);
        if (ETX_MEDIUM_SAMPLE_SHARED_RND(context) > p) {
          return medium_sample_shared_zero_sample(spect, world_pos_at_t, 0.0f);
        }
        transmittance = ETX_MEDIUM_SAMPLE_SHARED_SPECTRAL_MUL(transmittance, 1.0f / p);
      }
    }
  }

  return medium_sample_shared_zero_sample(spect, pos + w_i * max_t, 0.0f);
}

#undef ETX_MEDIUM_SAMPLE_SHARED_SPECTRAL_CLAMP_NON_NEGATIVE
#undef ETX_MEDIUM_SAMPLE_SHARED_SPECTRAL_MAXIMUM
#undef ETX_MEDIUM_SAMPLE_SHARED_SPECTRAL_IS_ZERO
#undef ETX_MEDIUM_SAMPLE_SHARED_SPECTRAL_EXP
#undef ETX_MEDIUM_SAMPLE_SHARED_SPECTRAL_DIV
#undef ETX_MEDIUM_SAMPLE_SHARED_SPECTRAL_MUL
#undef ETX_MEDIUM_SAMPLE_SHARED_SPECTRAL_SUB
#undef ETX_MEDIUM_SAMPLE_SHARED_SPECTRAL_ADD
#undef ETX_MEDIUM_SAMPLE_SHARED_SPECTRAL_MAKE
#undef ETX_MEDIUM_SAMPLE_SHARED_QUERY_IS_SPECTRAL
#undef ETX_MEDIUM_SAMPLE_SHARED_SPECTRAL_IS_SPECTRAL
#undef ETX_MEDIUM_SAMPLE_SHARED_SAMPLE
#undef ETX_MEDIUM_SAMPLE_SHARED_SPECTRAL_QUERY
#undef ETX_MEDIUM_SAMPLE_SHARED_SPECTRAL_RESPONSE
#undef ETX_MEDIUM_SAMPLE_SHARED_MEDIUM_TYPE
