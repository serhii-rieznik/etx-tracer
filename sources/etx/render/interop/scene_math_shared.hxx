#pragma once

#include "surface_point_shared.hxx"

ETX_SHARED_INLINE float scene_math_shared_collimation_to_exponent(float normalized) {
  float t = saturate(normalized);
  float one_minus_t = 1.0f - t;
  float denom = one_minus_t * one_minus_t;
  denom *= denom;
  return 1.0f / max(kEpsilon, denom);
}

ETX_SHARED_INLINE float3 scene_math_shared_shading_pos_project(ETX_IN(float3, position), ETX_IN(float3, origin), ETX_IN(float3, normal)) {
  return position - dot(position - origin, normal) * normal;
}

ETX_SHARED_INLINE float3 scene_math_shared_shading_pos(ETX_IN(float3, g0), ETX_IN(float3, g1), ETX_IN(float3, g2), ETX_IN(float3, n0), ETX_IN(float3, n1), ETX_IN(float3, n2),
  ETX_IN(float3, geo_normal), ETX_IN(float3, bc), ETX_IN(float3, w_o)) {
  float3 geo_pos = g0 * bc.x + g1 * bc.y + g2 * bc.z;
  float3 sh_normal = normalize(n0 * bc.x + n1 * bc.y + n2 * bc.z);
  float direction = (dot(sh_normal, w_o) >= 0.0f) ? 1.0f : -1.0f;

  float3 p0 = scene_math_shared_shading_pos_project(geo_pos, g0, direction * n0);
  float3 p1 = scene_math_shared_shading_pos_project(geo_pos, g1, direction * n1);
  float3 p2 = scene_math_shared_shading_pos_project(geo_pos, g2, direction * n2);
  float3 sh_pos = p0 * bc.x + p1 * bc.y + p2 * bc.z;

  bool convex = dot(sh_pos - geo_pos, sh_normal) * direction > 0.0f;
  return offset_ray((convex ? sh_pos : geo_pos), geo_normal * direction);
}

ETX_SHARED_INLINE float3 scene_math_shared_orient_normals_to_hemisphere(ETX_IN(float3, shading_normal), ETX_IN(float3, geo_normal), ETX_IN(float3, view_direction)) {
  const uint32_t max_attempts = 16u;
  float3 normalized_geo_normal = normalize(geo_normal);
  float i_dot_g = dot(view_direction, normalized_geo_normal);

  float shading_normal_length_sq = dot(shading_normal, shading_normal);
  if ((shading_normal_length_sq > kEpsilon) == false) {
    return normalized_geo_normal;
  }
  float3 result = shading_normal / sqrt(shading_normal_length_sq);
  if (dot(result, normalized_geo_normal) < 0.0f) {
    result = -result;
  }
  float i_dot_s = dot(view_direction, result);
  for (uint32_t i = 0u; ((i_dot_s * i_dot_g) <= kEpsilon) && (i < max_attempts); ++i) {
    result = normalize(8.0f * result + normalized_geo_normal);
    i_dot_s = dot(view_direction, result);
  }

  if ((i_dot_s * i_dot_g) <= kEpsilon) {
    result = normalized_geo_normal;
  }
  return result;
}

ETX_SHARED_INLINE void scene_math_shared_build_sampling_frame_with_handedness(ETX_IN(float3, normal), ETX_IN(float3, tangent_hint), ETX_IN(float3, bitangent_hint),
  float handedness, ETX_OUT(float3, normalized_normal), ETX_OUT(float3, tangent), ETX_OUT(float3, bitangent)) {
  surface_point_shared_build_frame_with_handedness(normal, tangent_hint, bitangent_hint, handedness, normalized_normal, tangent, bitangent);
}

ETX_SHARED_INLINE void scene_math_shared_build_sampling_frame(ETX_IN(float3, normal), ETX_IN(float3, tangent_hint), ETX_IN(float3, bitangent_hint),
  ETX_OUT(float3, normalized_normal), ETX_OUT(float3, tangent), ETX_OUT(float3, bitangent)) {
  float handedness = (dot(cross(normal, tangent_hint), bitangent_hint) >= 0.0f) ? 1.0f : -1.0f;
  scene_math_shared_build_sampling_frame_with_handedness(normal, tangent_hint, bitangent_hint, handedness, normalized_normal, tangent, bitangent);
}

ETX_SHARED_INLINE void scene_math_shared_finalize_shading_frame(ETX_IN(float3, shading_normal), ETX_IN(float3, frame_normal), ETX_IN(float3, tangent_hint),
  ETX_IN(float3, bitangent_hint), ETX_IN(float3, geo_normal), ETX_IN(float3, view_direction), ETX_OUT(float3, normal), ETX_OUT(float3, tangent), ETX_OUT(float3, bitangent)) {
  float handedness = (dot(cross(frame_normal, tangent_hint), bitangent_hint) >= 0.0f) ? 1.0f : -1.0f;
  normal = scene_math_shared_orient_normals_to_hemisphere(shading_normal, geo_normal, view_direction);
  scene_math_shared_build_sampling_frame_with_handedness(normal, tangent_hint, bitangent_hint, handedness, normal, tangent, bitangent);
}

ETX_SHARED_INLINE float3 scene_math_shared_local_to_world(ETX_IN(float3, normal), ETX_IN(float3, tangent), ETX_IN(float3, bitangent), ETX_IN(float3, local_direction)) {
  return normalize(tangent * local_direction.x + bitangent * local_direction.y + normal * local_direction.z);
}

ETX_SHARED_INLINE float3 scene_math_shared_default_ao_shading(ETX_IN(float3, hit_normal), float ao) {
  float n_dot_up = saturate(dot(hit_normal, float3(0.0f, 1.0f, 0.0f)));
  float3 base = lerp(float3(0.35f, 0.37f, 0.42f), float3(0.85f, 0.87f, 0.9f), n_dot_up);
  return base * ao;
}
