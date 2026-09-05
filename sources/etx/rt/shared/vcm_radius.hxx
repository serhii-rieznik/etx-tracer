#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace etx {

inline float vcm_iteration_radius(float initial_radius, float bounding_sphere_radius, uint32_t max_dimension, uint32_t iteration) {
  if (initial_radius == 0.0f) {
    initial_radius = 2.0f * bounding_sphere_radius / static_cast<float>(std::max(1u, max_dimension));
  }

  // Progressive surface-density radius: r_n = r_0 (n + 1)^(-(1 - alpha) / 2).
  constexpr float kRadiusAlpha = 0.75f;
  return initial_radius * std::pow(static_cast<float>(iteration) + 1.0f, -0.5f * (1.0f - kRadiusAlpha));
}

}  // namespace etx
