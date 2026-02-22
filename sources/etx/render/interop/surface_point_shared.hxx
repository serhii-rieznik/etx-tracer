#pragma once

#include "math_shared.hxx"

ETX_SHARED_INLINE float3 surface_point_shared_barycentrics(ETX_IN(float2, bary)) {
  return barycentrics(bary);
}

ETX_SHARED_INLINE float3 surface_point_shared_lerp_float3(
  ETX_IN(float3, value_0), ETX_IN(float3, value_1), ETX_IN(float3, value_2), ETX_IN(float3, bc)) {
  return value_0 * bc.x + value_1 * bc.y + value_2 * bc.z;
}

ETX_SHARED_INLINE float2 surface_point_shared_lerp_float2(
  ETX_IN(float2, value_0), ETX_IN(float2, value_1), ETX_IN(float2, value_2), ETX_IN(float3, bc)) {
  return value_0 * bc.x + value_1 * bc.y + value_2 * bc.z;
}

ETX_SHARED_INLINE void surface_point_shared_orthogonalize_frame(
  ETX_IN(float3, normal), ETX_IN(float3, tangent_hint), ETX_IN(float3, bitangent_hint), ETX_OUT(float3, tangent), ETX_OUT(float3, bitangent)) {
  tangent = normalize(tangent_hint - dot(tangent_hint, normal) * normal);
  bitangent = normalize(cross(normal, tangent));
  bitangent = bitangent * ((dot(bitangent, bitangent_hint) > 0.0f) ? 1.0f : -1.0f);
}

ETX_SHARED_INLINE void surface_point_shared_interpolate_vertex(
  ETX_IN(float3, pos_0), ETX_IN(float3, pos_1), ETX_IN(float3, pos_2), ETX_IN(float3, nrm_0), ETX_IN(float3, nrm_1), ETX_IN(float3, nrm_2), ETX_IN(float3, tan_0),
  ETX_IN(float3, tan_1), ETX_IN(float3, tan_2), ETX_IN(float3, btn_0), ETX_IN(float3, btn_1), ETX_IN(float3, btn_2), ETX_IN(float2, tex_0), ETX_IN(float2, tex_1),
  ETX_IN(float2, tex_2), ETX_IN(float3, bc), bool has_surface_frame, bool has_texcoords, ETX_OUT(float3, pos), ETX_OUT(float3, nrm), ETX_OUT(float3, tan),
  ETX_OUT(float3, btn), ETX_OUT(float2, tex)) {
  pos = surface_point_shared_lerp_float3(pos_0, pos_1, pos_2, bc);
  nrm = normalize(surface_point_shared_lerp_float3(nrm_0, nrm_1, nrm_2, bc));

  if (has_surface_frame == false) {
    tan = float3(0.0f, 0.0f, 0.0f);
    btn = float3(0.0f, 0.0f, 0.0f);
  } else {
    float3 tangent_hint = surface_point_shared_lerp_float3(tan_0, tan_1, tan_2, bc);
    float3 bitangent_hint = surface_point_shared_lerp_float3(btn_0, btn_1, btn_2, bc);
    surface_point_shared_orthogonalize_frame(nrm, tangent_hint, bitangent_hint, tan, btn);
  }

  if (has_texcoords == false) {
    tex = float2(0.0f, 0.0f);
  } else {
    tex = surface_point_shared_lerp_float2(tex_0, tex_1, tex_2, bc);
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
