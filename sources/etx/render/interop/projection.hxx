#pragma once

#include "interop.hxx"
#include "math_shared.hxx"

struct Projection {
  enum : uint32_t {
    Equirectangular,
    EqualArea,
  };
};

#if defined(__cplusplus)
static_assert(static_cast<uint32_t>(ProjectionType::Equirectangular) == Projection::Equirectangular,
  "Projection::Equirectangular changed; update ProjectionType::Equirectangular to keep CPU/GPU projection ABI aligned");
static_assert(static_cast<uint32_t>(ProjectionType::EqualArea) == Projection::EqualArea,
  "Projection::EqualArea changed; update ProjectionType::EqualArea to keep CPU/GPU projection ABI aligned");
#endif

ETX_SHARED_INLINE float projection_sin_theta_for_pdf(ETX_IN(float2, uv), uint32_t projection) {
  float sin_theta = max(kEpsilon, sin(uv.y * kPi));
  if (projection == Projection::EqualArea) {
    sin_theta = max(kEpsilon, abs(2.0f * uv.y - 1.0f));
  }
  return sin_theta;
}
