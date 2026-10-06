#pragma once

#include "interop_constants.hxx"

struct SubsurfaceFreePath {
  float survival;
  float density;
};

// Equilibrium boundary flights and shifted exponential collision flights.
// Packing is the exclusion distance divided by the mean collision flight.
ETX_SHARED_INLINE float subsurface_free_path_sample(float extinction, float packing, bool correlated, float random) {
  if (extinction <= 0.0f) {
    return kMaxFloat;
  }
  if ((correlated == false) && (random < packing)) {
    return random / extinction;
  }
  const float tail = 1.0f - packing;
  const float survival = correlated ? (1.0f - random) : ((1.0f - random) / tail);
#if ETX_CPP
  const float exclusion_distance = packing / extinction;
  const float distance = exclusion_distance - (tail / extinction) * ETX_STD log(survival);
#else
  precise float exclusion_distance = packing / extinction;
  precise float distance = exclusion_distance - (tail / extinction) * log(survival);
#endif
  return distance;
}

ETX_SHARED_INLINE SubsurfaceFreePath subsurface_free_path_evaluate(float extinction, float packing, bool correlated, float distance) {
  SubsurfaceFreePath result;
  result.survival = 1.0f;
  result.density = 0.0f;
  if (extinction <= 0.0f) {
    return result;
  }
#if ETX_CPP
  const float exclusion_distance = packing / extinction;
#else
  precise float exclusion_distance = packing / extinction;
#endif
  if (distance < exclusion_distance) {
    if (correlated == false) {
      result.survival = 1.0f - extinction * distance;
      result.density = extinction;
    }
    return result;
  }
  const float tail = 1.0f - packing;
#if ETX_CPP
  const float excess_distance = distance - exclusion_distance;
  const float exponent = -excess_distance * extinction / tail;
#else
  precise float excess_distance = distance - exclusion_distance;
  precise float exponent = -excess_distance * extinction / tail;
#endif
  const float correlated_survival = ETX_STD exp(exponent);
  result.survival = correlated ? correlated_survival : (tail * correlated_survival);
  result.density = correlated ? (extinction * correlated_survival / tail) : (extinction * correlated_survival);
  return result;
}

ETX_SHARED_INLINE float subsurface_free_path_channel_weight(float extinction, float packing, bool correlated, float boundary_distance, float albedo) {
  const SubsurfaceFreePath flight = subsurface_free_path_evaluate(extinction, packing, correlated, boundary_distance);
  // Integrate the boundary-escape and scattering contributions of this flight.
  return flight.survival + albedo * (1.0f - flight.survival);
}
