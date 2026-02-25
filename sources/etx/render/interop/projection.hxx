#pragma once

#include "interop.hxx"
#include "math_shared.hxx"

struct Projection {
  enum : uint32_t {
    Equirectangular,
    EqualArea,
  };
};

ETX_STATIC_ASSERT((ETX_ENUM_U32_TO_UINT32(ProjectionType::Equirectangular) == Projection::Equirectangular),
  "Projection::Equirectangular changed; update ProjectionType::Equirectangular to keep CPU/GPU projection ABI aligned");
ETX_STATIC_ASSERT((ETX_ENUM_U32_TO_UINT32(ProjectionType::EqualArea) == Projection::EqualArea),
  "Projection::EqualArea changed; update ProjectionType::EqualArea to keep CPU/GPU projection ABI aligned");

ETX_SHARED_INLINE uint32_t projection_environment_mode(bool is_atmosphere) {
  return is_atmosphere ? Projection::EqualArea : Projection::Equirectangular;
}

ETX_SHARED_INLINE float2 projection_environment_direction_to_uv(ETX_IN(float3, direction), ETX_IN(float2, image_offset), float image_u_scale, bool is_atmosphere) {
  uint32_t projection_mode = projection_environment_mode(is_atmosphere);
  return direction_to_uv(direction, image_offset, image_u_scale, projection_mode);
}

ETX_SHARED_INLINE float projection_sin_theta_for_pdf(ETX_IN(float2, uv), uint32_t projection) {
  float sin_theta = max(kEpsilon, sin(uv.y * kPi));
  if (projection == Projection::EqualArea) {
    sin_theta = max(kEpsilon, abs(2.0f * uv.y - 1.0f));
  }
  return sin_theta;
}

ETX_SHARED_INLINE float projection_environment_image_pdf_to_solid_angle(float image_pdf, ETX_IN(float2, uv), uint32_t projection) {
  float sin_theta = projection_sin_theta_for_pdf(uv, projection);
  return image_pdf / (2.0f * kPi * kPi * sin_theta);
}
