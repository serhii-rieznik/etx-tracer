#pragma once

#include "interop.hxx"
#include "math_shared.hxx"

struct DirectionalEmissionDomain {
  OrthonormalBasis basis;
  float2 extent;
  float area;
  bool disk;
};

ETX_SHARED_INLINE float directional_emission_projected_extent(ETX_IN(float3, axis), ETX_IN(float3, half_extent)) {
  return ETX_STD abs(axis.x) * half_extent.x + ETX_STD abs(axis.y) * half_extent.y + ETX_STD abs(axis.z) * half_extent.z;
}

ETX_SHARED_INLINE DirectionalEmissionDomain directional_emission_domain(ETX_IN(float3, direction), float angular_cosine, ETX_IN(float3, half_extent), float radius) {
  DirectionalEmissionDomain result;
  result.basis = orthonormal_basis(direction);
  result.extent = float2(radius, radius);
  result.area = kPi * radius * radius;
  result.disk = true;

  const float tangent = (angular_cosine > kEpsilon) ? ETX_STD sqrt(max(0.0f, 1.0f - angular_cosine * angular_cosine)) / angular_cosine : 0.0f;
  // Project every supported sun direction back onto the nominal launch plane.
  const float padding = (radius + directional_emission_projected_extent(direction, half_extent)) * tangent;
  for (uint32_t axis_index = 0u; axis_index < 3u; ++axis_index) {
    const float3 axis = axis_index == 0u ? float3(1.0f, 0.0f, 0.0f) : (axis_index == 1u ? float3(0.0f, 1.0f, 0.0f) : float3(0.0f, 0.0f, 1.0f));
    const float3 projected = cross(direction, axis);
    const float length_squared = dot(projected, projected);
    if (length_squared == 0.0f) {
      continue;
    }
    const float3 u = projected / ETX_STD sqrt(length_squared);
    const float3 v = cross(direction, u);
    const float2 extent = float2(directional_emission_projected_extent(u, half_extent) + padding, directional_emission_projected_extent(v, half_extent) + padding);
    const float area = 4.0f * extent.x * extent.y;
    if ((area > 0.0f) && (area < result.area)) {
      result.basis.u = u;
      result.basis.v = v;
      result.extent = extent;
      result.area = area;
      result.disk = false;
    }
  }
  return result;
}

ETX_SHARED_INLINE float3 directional_emission_position(ETX_IN(DirectionalEmissionDomain, domain), ETX_IN(float2, rnd)) {
  const float2 position = domain.disk ? sample_disk(rnd) : 2.0f * rnd - float2(1.0f, 1.0f);
  return (position.x * domain.extent.x) * domain.basis.u + (position.y * domain.extent.y) * domain.basis.v;
}
