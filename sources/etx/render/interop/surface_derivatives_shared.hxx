#pragma once

#include "math_shared.hxx"

// World-space derivatives before material normal mapping or view-dependent normal correction.
struct SurfaceDerivatives {
  float3 normal ETX_INIT({});
  float3 dpdu ETX_INIT({});
  float3 dpdv ETX_INIT({});
  float3 dndu ETX_INIT({});
  float3 dndv ETX_INIT({});
};

ETX_SHARED_INLINE float3 surface_derivatives_shared_transform_vector(ETX_IN(AffineTransform, transform), ETX_IN(float3, vector)) {
  return float3(dot(float3(transform.rows[0].x, transform.rows[0].y, transform.rows[0].z), vector),
    dot(float3(transform.rows[1].x, transform.rows[1].y, transform.rows[1].z), vector), dot(float3(transform.rows[2].x, transform.rows[2].y, transform.rows[2].z), vector));
}

ETX_SHARED_INLINE float3 surface_derivatives_shared_transform_normal(ETX_IN(AffineTransform, world_to_object), ETX_IN(float3, normal)) {
  return float3(world_to_object.rows[0].x * normal.x + world_to_object.rows[1].x * normal.y + world_to_object.rows[2].x * normal.z,
    world_to_object.rows[0].y * normal.x + world_to_object.rows[1].y * normal.y + world_to_object.rows[2].y * normal.z,
    world_to_object.rows[0].z * normal.x + world_to_object.rows[1].z * normal.y + world_to_object.rows[2].z * normal.z);
}

ETX_SHARED_INLINE bool surface_derivatives_shared_finite(ETX_IN(float3, value)) {
  return ETX_STD isfinite(value.x) && ETX_STD isfinite(value.y) && ETX_STD isfinite(value.z);
}

ETX_SHARED_INLINE bool surface_derivatives_shared_compute_triangle(ETX_IN(float3, p0), ETX_IN(float3, p1), ETX_IN(float3, p2), ETX_IN(float3, n0), ETX_IN(float3, n1),
  ETX_IN(float3, n2), ETX_IN(float2, uv0), ETX_IN(float2, uv1), ETX_IN(float2, uv2), ETX_IN(float3, barycentric), ETX_IN(AffineTransform, object_to_world),
  ETX_IN(AffineTransform, world_to_object), float orientation, ETX_OUT(SurfaceDerivatives, derivatives)) {
  derivatives = ETX_ZERO(SurfaceDerivatives);
  const float2 uv_edge1 = uv1 - uv0;
  const float2 uv_edge2 = uv2 - uv0;
  const float u_scale = max(abs(uv_edge1.x), abs(uv_edge2.x));
  const float v_scale = max(abs(uv_edge1.y), abs(uv_edge2.y));
  if ((u_scale == 0.0f) || (v_scale == 0.0f) || (ETX_STD isfinite(u_scale) == false) || (ETX_STD isfinite(v_scale) == false)) {
    return false;
  }
  // Scale each UV axis separately so small or highly anisotropic mappings remain usable.
  const float2 a = float2(uv_edge1.x / u_scale, uv_edge1.y / v_scale);
  const float2 b = float2(uv_edge2.x / u_scale, uv_edge2.y / v_scale);
  const float determinant = a.x * b.y - a.y * b.x;
  if ((determinant == 0.0f) || (ETX_STD isfinite(determinant) == false)) {
    return false;
  }
  const float inverse_determinant = 1.0f / determinant;
  // Transform edges directly; a large instance translation must not erase their lengths.
  const float3 edge1 = surface_derivatives_shared_transform_vector(object_to_world, p1 - p0);
  const float3 edge2 = surface_derivatives_shared_transform_vector(object_to_world, p2 - p0);
  const float3 normal0 = surface_derivatives_shared_transform_normal(world_to_object, n0) * orientation;
  const float3 normal1 = surface_derivatives_shared_transform_normal(world_to_object, n1) * orientation;
  const float3 normal2 = surface_derivatives_shared_transform_normal(world_to_object, n2) * orientation;
  const float normal_scale = max(max(max(abs(normal0.x), abs(normal0.y)), abs(normal0.z)),
    max(max(max(abs(normal1.x), abs(normal1.y)), abs(normal1.z)), max(max(abs(normal2.x), abs(normal2.y)), abs(normal2.z))));
  if ((normal_scale == 0.0f) || (ETX_STD isfinite(normal_scale) == false)) {
    return false;
  }
  // Interpolate unnormalized transformed normals, matching normalize(M^-T * normalize(interpolate(n))).
  const float3 scaled_n0 = normal0 / normal_scale;
  const float3 scaled_n1 = normal1 / normal_scale;
  const float3 scaled_n2 = normal2 / normal_scale;
  const float3 interpolated_normal = scaled_n0 * barycentric.x + scaled_n1 * barycentric.y + scaled_n2 * barycentric.z;
  const float normal_length_squared = dot(interpolated_normal, interpolated_normal);
  if ((normal_length_squared <= 0.0f) || (ETX_STD isfinite(normal_length_squared) == false)) {
    return false;
  }
  const float inverse_normal_length = 1.0f / sqrt(normal_length_squared);
  const float3 normal = interpolated_normal * inverse_normal_length;
  const float3 normal_edge1 = scaled_n1 - scaled_n0;
  const float3 normal_edge2 = scaled_n2 - scaled_n0;
  const float3 normal_du = ((b.y * normal_edge1 - a.y * normal_edge2) * inverse_determinant) / u_scale;
  const float3 normal_dv = ((a.x * normal_edge2 - b.x * normal_edge1) * inverse_determinant) / v_scale;
  SurfaceDerivatives result = ETX_ZERO(SurfaceDerivatives);
  result.normal = normal;
  result.dpdu = ((b.y * edge1 - a.y * edge2) * inverse_determinant) / u_scale;
  result.dpdv = ((a.x * edge2 - b.x * edge1) * inverse_determinant) / v_scale;
  result.dndu = (normal_du - normal * dot(normal, normal_du)) * inverse_normal_length;
  result.dndv = (normal_dv - normal * dot(normal, normal_dv)) * inverse_normal_length;
  if ((surface_derivatives_shared_finite(result.dpdu) == false) || (surface_derivatives_shared_finite(result.dpdv) == false) ||
      (surface_derivatives_shared_finite(result.dndu) == false) || (surface_derivatives_shared_finite(result.dndv) == false)) {
    return false;
  }
  derivatives = result;
  return true;
}
