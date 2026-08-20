#pragma once

#include "math_shared.hxx"

ETX_SHARED_INLINE void surface_point_shared_build_frame_with_handedness(ETX_IN(float3, normal), ETX_IN(float3, tangent_hint), ETX_IN(float3, bitangent_hint), float handedness,
  ETX_OUT(float3, normalized_normal), ETX_OUT(float3, tangent), ETX_OUT(float3, bitangent)) {
  normalized_normal = normalize(normal);
  const float3 tangent_projected = tangent_hint - dot(tangent_hint, normalized_normal) * normalized_normal;
  const float3 bitangent_projected = bitangent_hint - dot(bitangent_hint, normalized_normal) * normalized_normal;
  const float tangent_length_sq = dot(tangent_projected, tangent_projected);
  const float bitangent_length_sq = dot(bitangent_projected, bitangent_projected);
  if (tangent_length_sq > kEpsilon) {
    tangent = tangent_projected / sqrt(tangent_length_sq);
    bitangent = normalize(cross(normalized_normal, tangent)) * handedness;
  } else if (bitangent_length_sq > kEpsilon) {
    bitangent = bitangent_projected / sqrt(bitangent_length_sq);
    tangent = normalize(cross(bitangent, normalized_normal)) * handedness;
  } else {
    const OrthonormalBasis basis = orthonormal_basis(normalized_normal);
    tangent = basis.u;
    bitangent = basis.v * handedness;
  }
}

ETX_SHARED_INLINE void surface_point_shared_orthogonalize_frame(ETX_IN(float3, normal), ETX_IN(float3, tangent_hint), ETX_IN(float3, bitangent_hint), ETX_OUT(float3, tangent),
  ETX_OUT(float3, bitangent)) {
  const float handedness = (dot(cross(normal, tangent_hint), bitangent_hint) >= 0.0f) ? 1.0f : -1.0f;
  float3 normalized_normal = float3(0.0f, 0.0f, 0.0f);
  surface_point_shared_build_frame_with_handedness(normal, tangent_hint, bitangent_hint, handedness, normalized_normal, tangent, bitangent);
}

ETX_SHARED_INLINE void surface_point_shared_interpolate_vertex(ETX_IN(float3, pos_0), ETX_IN(float3, pos_1), ETX_IN(float3, pos_2), ETX_IN(float3, nrm_0), ETX_IN(float3, nrm_1),
  ETX_IN(float3, nrm_2), ETX_IN(float3, tan_0), ETX_IN(float3, tan_1), ETX_IN(float3, tan_2), ETX_IN(float3, btn_0), ETX_IN(float3, btn_1), ETX_IN(float3, btn_2),
  ETX_IN(float2, tex_0), ETX_IN(float2, tex_1), ETX_IN(float2, tex_2), ETX_IN(float3, bc), bool has_surface_frame, bool has_texcoords, ETX_OUT(Vertex, vertex)) {
  vertex.pos = pos_0 * bc.x + pos_1 * bc.y + pos_2 * bc.z;
  vertex.nrm = normalize(nrm_0 * bc.x + nrm_1 * bc.y + nrm_2 * bc.z);

  if (has_surface_frame == false) {
    vertex.tan = float3(0.0f, 0.0f, 0.0f);
    vertex.btn = float3(0.0f, 0.0f, 0.0f);
  } else {
    float3 tangent_hint = tan_0 * bc.x + tan_1 * bc.y + tan_2 * bc.z;
    float3 bitangent_hint = btn_0 * bc.x + btn_1 * bc.y + btn_2 * bc.z;
    surface_point_shared_orthogonalize_frame(vertex.nrm, tangent_hint, bitangent_hint, vertex.tan, vertex.btn);
  }

  if (has_texcoords == false) {
    vertex.tex = float2(0.0f, 0.0f);
  } else {
    vertex.tex = tex_0 * bc.x + tex_1 * bc.y + tex_2 * bc.z;
  }
}

ETX_SHARED_INLINE float3 surface_point_shared_orient_geo_normal(ETX_IN(float3, geo_normal), ETX_IN(float3, ray_direction)) {
  float3 result = normalize(geo_normal);
  if (dot(result, ray_direction) > 0.0f) {
    result = -result;
  }
  return result;
}

ETX_SHARED_INLINE float3 surface_point_shared_orient_shading_normal(ETX_IN(float3, shading_normal), ETX_IN(float3, geo_normal)) {
  if (dot(shading_normal, geo_normal) < 0.0f) {
    return -shading_normal;
  }
  return shading_normal;
}
