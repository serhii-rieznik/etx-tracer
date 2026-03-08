#pragma once

#include "interop.hxx"
#include "spectrum.hxx"

#if defined(__cplusplus)
# define ETX_MEDIUM_SHARED_SPECTRAL_RESPONSE ::SpectralResponse
# define ETX_MEDIUM_SHARED_SPECTRAL_QUERY ::SpectralQuery
# define ETX_MEDIUM_SHARED_SPECTRAL_CLAMP_NON_NEGATIVE ::spectral_response_clamp_non_negative
# define ETX_MEDIUM_SHARED_SPECTRAL_EXP ::spectral_response_exp
# define ETX_MEDIUM_SHARED_SPECTRAL_MUL ::spectral_response_mul
# define ETX_MEDIUM_SHARED_SPECTRAL_DIV ::spectral_response_div
# define ETX_MEDIUM_SHARED_SPECTRAL_SUB ::spectral_response_sub
# define ETX_MEDIUM_SHARED_SPECTRAL_MAXIMUM ::spectral_response_maximum
# define ETX_MEDIUM_SHARED_SPECTRAL_MAKE ::spectral_response_make
#else
# define ETX_MEDIUM_SHARED_SPECTRAL_RESPONSE SpectralResponse
# define ETX_MEDIUM_SHARED_SPECTRAL_QUERY SpectralQuery
# define ETX_MEDIUM_SHARED_SPECTRAL_CLAMP_NON_NEGATIVE spectral_response_clamp_non_negative
# define ETX_MEDIUM_SHARED_SPECTRAL_EXP spectral_response_exp
# define ETX_MEDIUM_SHARED_SPECTRAL_MUL spectral_response_mul
# define ETX_MEDIUM_SHARED_SPECTRAL_DIV spectral_response_div
# define ETX_MEDIUM_SHARED_SPECTRAL_SUB spectral_response_sub
# define ETX_MEDIUM_SHARED_SPECTRAL_MAXIMUM spectral_response_maximum
# define ETX_MEDIUM_SHARED_SPECTRAL_MAKE spectral_response_make
#endif

struct MediumSharedIntersection {
  float3 medium_pos ETX_INIT({});
  float3 medium_dir ETX_INIT({});
  float t_min ETX_INIT(0.0f);
  float t_max ETX_INIT(0.0f);
  float3 world_dir_normalized ETX_INIT({});
};

ETX_SHARED_INLINE MediumSharedIntersection medium_shared_zero_intersection() {
  MediumSharedIntersection result;
  result.medium_pos = float3(0.0f, 0.0f, 0.0f);
  result.medium_dir = float3(0.0f, 0.0f, 0.0f);
  result.t_min = 0.0f;
  result.t_max = 0.0f;
  result.world_dir_normalized = float3(0.0f, 0.0f, 0.0f);
  return result;
}

ETX_SHARED_INLINE float3 medium_shared_bounds_to_local(ETX_IN(float3, p), ETX_IN(float3, bounds_min), ETX_IN(float3, bounds_max)) {
  float3 size = bounds_max - bounds_min;
  float3 result = float3(0.0f, 0.0f, 0.0f);
  result.x = (size.x > kEpsilon) ? ((p.x - bounds_min.x) / size.x) : 0.0f;
  result.y = (size.y > kEpsilon) ? ((p.y - bounds_min.y) / size.y) : 0.0f;
  result.z = (size.z > kEpsilon) ? ((p.z - bounds_min.z) / size.z) : 0.0f;
  return result;
}

ETX_SHARED_INLINE float3 medium_shared_bounds_from_local(ETX_IN(float3, p), ETX_IN(float3, bounds_min), ETX_IN(float3, bounds_max)) {
  float3 size = bounds_max - bounds_min;
  return float3(p.x * size.x + bounds_min.x, p.y * size.y + bounds_min.y, p.z * size.z + bounds_min.z);
}

ETX_SHARED_INLINE float medium_shared_gamma(uint32_t n) {
  float e = kEpsilon * 0.5f;
  float n_f = float(n);
  return (n_f * e) / (1.0f - n_f * e);
}

ETX_SHARED_INLINE float medium_shared_log(float value) {
#if defined(__cplusplus)
  return ::logf(value);
#else
  return log(value);
#endif
}

ETX_SHARED_INLINE float medium_shared_sqrt(float value) {
#if defined(__cplusplus)
  return ::sqrtf(value);
#else
  return sqrt(value);
#endif
}

ETX_SHARED_INLINE bool medium_shared_bounds(ETX_IN(float3, in_pos), ETX_IN(float3, in_dir), float max_t, ETX_OUT(float, t_min), ETX_OUT(float, t_max)) {
  const float g3 = 1.0f + 2.0f * medium_shared_gamma(3u);

  t_min = 0.0f;
  t_max = max_t;

  float t_near_x = (0.0f - in_pos.x) / in_dir.x;
  float t_far_x = (1.0f - in_pos.x) / in_dir.x;
  if (t_near_x > t_far_x) {
    float t_swap = t_far_x;
    t_far_x = t_near_x;
    t_near_x = t_swap;
  }
  t_far_x *= g3;
  t_min = max(t_min, t_near_x);
  t_max = min(t_max, t_far_x);
  if (t_min > t_max) {
    return false;
  }

  float t_near_y = (0.0f - in_pos.y) / in_dir.y;
  float t_far_y = (1.0f - in_pos.y) / in_dir.y;
  if (t_near_y > t_far_y) {
    float t_swap = t_far_y;
    t_far_y = t_near_y;
    t_near_y = t_swap;
  }
  t_far_y *= g3;
  t_min = max(t_min, t_near_y);
  t_max = min(t_max, t_far_y);
  if (t_min > t_max) {
    return false;
  }

  float t_near_z = (0.0f - in_pos.z) / in_dir.z;
  float t_far_z = (1.0f - in_pos.z) / in_dir.z;
  if (t_near_z > t_far_z) {
    float t_swap = t_far_z;
    t_far_z = t_near_z;
    t_near_z = t_swap;
  }
  t_far_z *= g3;
  t_min = max(t_min, t_near_z);
  t_max = min(t_max, t_far_z);
  if (t_min > t_max) {
    return false;
  }

  return true;
}

ETX_SHARED_INLINE bool medium_shared_intersects_bounds(ETX_IN(float3, bounds_min), ETX_IN(float3, bounds_max), ETX_IN(float3, in_pos), ETX_IN(float3, in_direction),
  float in_max_t, ETX_OUT(MediumSharedIntersection, result)) {
  result = medium_shared_zero_intersection();
  if (in_max_t >= kMaxFloat) {
    return false;
  }

  result.medium_pos = medium_shared_bounds_to_local(in_pos, bounds_min, bounds_max);

  float3 end_pos = in_pos + in_direction * in_max_t;
  float3 medium_end_pos = medium_shared_bounds_to_local(end_pos, bounds_min, bounds_max);

  result.medium_dir = medium_end_pos - result.medium_pos;
  float d_len = dot(result.medium_dir, result.medium_dir);
  const float threshold = kRayEpsilon * kRayEpsilon;
  if (d_len <= threshold) {
    return false;
  }

  result.medium_dir *= 1.0f / medium_shared_sqrt(d_len);
  result.world_dir_normalized = normalize(in_direction);

  float segment = length(medium_end_pos - result.medium_pos);
  return medium_shared_bounds(result.medium_pos, result.medium_dir, segment, result.t_min, result.t_max);
}

ETX_SHARED_INLINE float3 medium_shared_transmittance_homogeneous_integrated(ETX_IN(float3, extinction), float distance) {
  return exp(-max(extinction, float3(0.0f, 0.0f, 0.0f)) * distance);
}

ETX_SHARED_INLINE ETX_MEDIUM_SHARED_SPECTRAL_RESPONSE medium_shared_transmittance_homogeneous_spectral(
  ETX_IN(ETX_MEDIUM_SHARED_SPECTRAL_RESPONSE, extinction), float distance) {
  ETX_MEDIUM_SHARED_SPECTRAL_RESPONSE extinction_non_negative = ETX_MEDIUM_SHARED_SPECTRAL_CLAMP_NON_NEGATIVE(extinction);
  return ETX_MEDIUM_SHARED_SPECTRAL_EXP(ETX_MEDIUM_SHARED_SPECTRAL_MUL(extinction_non_negative, -distance));
}

ETX_SHARED_INLINE float3 medium_shared_transmittance_heterogeneous_integrated(ETX_IN(float3, base_extinction), ETX_IN(float3, origin), ETX_IN(float3, direction),
  float distance, ETX_IN(float3, bounds_min), ETX_IN(float3, bounds_max), ETX_INOUT(MediumSharedContext, context)) {
  float3 extinction = max(base_extinction, float3(0.0f, 0.0f, 0.0f));
  float max_sigma = max(extinction.x, max(extinction.y, extinction.z));
  if (max_sigma <= 0.0f) {
    return float3(1.0f, 1.0f, 1.0f);
  }

  MediumSharedIntersection intersection = medium_shared_zero_intersection();
  if (medium_shared_intersects_bounds(bounds_min, bounds_max, origin, direction, distance, intersection) == false) {
    return float3(1.0f, 1.0f, 1.0f);
  }

  float3 transmittance = float3(1.0f, 1.0f, 1.0f);
  float t_world = 0.0f;
  float segment_length = intersection.t_max - intersection.t_min;
  const float rr_threshold = 0.1f;
  const uint32_t max_delta_tracking_steps = 4096u;
  for (uint32_t step = 0u; step < max_delta_tracking_steps; ++step) {
    float random_value = min(medium_shared_rnd(context), 1.0f - kEpsilon);
    t_world += -medium_shared_log(1.0f - random_value) / max_sigma;

    float3 world_pos_at_t = origin + intersection.world_dir_normalized * t_world;
    float3 local_pos = medium_shared_bounds_to_local(world_pos_at_t, bounds_min, bounds_max);
    float t_local_along_dir = dot(local_pos - intersection.medium_pos, intersection.medium_dir);
    if (t_local_along_dir >= segment_length) {
      break;
    }

    float density_value = medium_shared_density(context, local_pos);
    float3 extinction_at_point = extinction * density_value;
    float3 weight = float3(1.0f, 1.0f, 1.0f) - (extinction_at_point / max_sigma);
    transmittance *= max(weight, float3(0.0f, 0.0f, 0.0f));

    float transmittance_max = max(transmittance.x, max(transmittance.y, transmittance.z));
    if (transmittance_max < rr_threshold) {
      float p = clamp(transmittance_max, 0.01f, 0.95f);
      if (medium_shared_rnd(context) > p) {
        return float3(0.0f, 0.0f, 0.0f);
      }
      transmittance *= (1.0f / p);
    }
  }

  return transmittance;
}

ETX_SHARED_INLINE ETX_MEDIUM_SHARED_SPECTRAL_RESPONSE medium_shared_transmittance_heterogeneous_spectral(
  ETX_IN(ETX_MEDIUM_SHARED_SPECTRAL_RESPONSE, base_extinction), ETX_IN(float3, origin), ETX_IN(float3, direction), float distance, ETX_IN(float3, bounds_min),
  ETX_IN(float3, bounds_max), ETX_INOUT(MediumSharedContext, context), ETX_IN(ETX_MEDIUM_SHARED_SPECTRAL_QUERY, spect)) {
  ETX_MEDIUM_SHARED_SPECTRAL_RESPONSE one = ETX_MEDIUM_SHARED_SPECTRAL_MAKE(spect, 1.0f);
  ETX_MEDIUM_SHARED_SPECTRAL_RESPONSE extinction = ETX_MEDIUM_SHARED_SPECTRAL_CLAMP_NON_NEGATIVE(base_extinction);
  float max_sigma = ETX_MEDIUM_SHARED_SPECTRAL_MAXIMUM(extinction);
  if (max_sigma <= 0.0f) {
    return one;
  }

  MediumSharedIntersection intersection = medium_shared_zero_intersection();
  if (medium_shared_intersects_bounds(bounds_min, bounds_max, origin, direction, distance, intersection) == false) {
    return one;
  }

  ETX_MEDIUM_SHARED_SPECTRAL_RESPONSE transmittance = one;
  float t_world = 0.0f;
  float segment_length = intersection.t_max - intersection.t_min;
  const float rr_threshold = 0.1f;
  const uint32_t max_delta_tracking_steps = 4096u;
  for (uint32_t step = 0u; step < max_delta_tracking_steps; ++step) {
    float random_value = min(medium_shared_rnd(context), 1.0f - kEpsilon);
    t_world += -medium_shared_log(1.0f - random_value) / max_sigma;

    float3 world_pos_at_t = origin + intersection.world_dir_normalized * t_world;
    float3 local_pos = medium_shared_bounds_to_local(world_pos_at_t, bounds_min, bounds_max);
    float t_local_along_dir = dot(local_pos - intersection.medium_pos, intersection.medium_dir);
    if (t_local_along_dir >= segment_length) {
      break;
    }

    float density_value = medium_shared_density(context, local_pos);
    ETX_MEDIUM_SHARED_SPECTRAL_RESPONSE extinction_at_point = ETX_MEDIUM_SHARED_SPECTRAL_MUL(extinction, density_value);
    ETX_MEDIUM_SHARED_SPECTRAL_RESPONSE weight = ETX_MEDIUM_SHARED_SPECTRAL_SUB(one, ETX_MEDIUM_SHARED_SPECTRAL_DIV(extinction_at_point, max_sigma));
    weight = ETX_MEDIUM_SHARED_SPECTRAL_CLAMP_NON_NEGATIVE(weight);
    transmittance = ETX_MEDIUM_SHARED_SPECTRAL_MUL(transmittance, weight);

    float transmittance_max = ETX_MEDIUM_SHARED_SPECTRAL_MAXIMUM(transmittance);
    if (transmittance_max < rr_threshold) {
      float p = clamp(transmittance_max, 0.01f, 0.95f);
      if (medium_shared_rnd(context) > p) {
        return ETX_MEDIUM_SHARED_SPECTRAL_MAKE(spect, 0.0f);
      }
      transmittance = ETX_MEDIUM_SHARED_SPECTRAL_MUL(transmittance, 1.0f / p);
    }
  }

  return transmittance;
}

#undef ETX_MEDIUM_SHARED_SPECTRAL_MAKE
#undef ETX_MEDIUM_SHARED_SPECTRAL_MAXIMUM
#undef ETX_MEDIUM_SHARED_SPECTRAL_SUB
#undef ETX_MEDIUM_SHARED_SPECTRAL_DIV
#undef ETX_MEDIUM_SHARED_SPECTRAL_MUL
#undef ETX_MEDIUM_SHARED_SPECTRAL_EXP
#undef ETX_MEDIUM_SHARED_SPECTRAL_CLAMP_NON_NEGATIVE
#undef ETX_MEDIUM_SHARED_SPECTRAL_QUERY
#undef ETX_MEDIUM_SHARED_SPECTRAL_RESPONSE
