#pragma once

#include "image_filter_shared.hxx"
#include "surface_derivatives_shared.hxx"
#include "scene_math_shared.hxx"

ETX_SHARED_INLINE float bump_mapping_shared_channel(ETX_IN(float4, value), uint32_t channel) {
  if (channel == 4u)
    return (value.x + value.y + value.z) / 3.0f;
  if (channel == 0u)
    return value.x;
  if (channel == 1u)
    return value.y;
  return channel == 2u ? value.z : value.w;
}

// Differentiate the same bilinear reconstruction used for texture evaluation.
ETX_SHARED_INLINE void bump_mapping_shared_sample(ETX_IN(float4, p00), ETX_IN(float4, p01), ETX_IN(float4, p10), ETX_IN(float4, p11), ETX_IN(ImageFilterSharedAddress, address),
  ETX_IN(float2, uv), ETX_IN(float2, fsize), ETX_IN(float2, uv_scale), uint32_t options, uint32_t channel, ETX_OUT(float, height), ETX_OUT(float2, gradient)) {
  const float h00 = bump_mapping_shared_channel(p00, channel);
  const float h01 = bump_mapping_shared_channel(p01, channel);
  const float h10 = bump_mapping_shared_channel(p10, channel);
  const float h11 = bump_mapping_shared_channel(p11, channel);
  const float top = h00 + (h01 - h00) * address.dx;
  const float bottom = h10 + (h11 - h10) * address.dx;
  height = top + (bottom - top) * address.dy;
  gradient = float2((h01 - h00) * (1.0f - address.dy) + (h11 - h10) * address.dy, (h10 - h00) * (1.0f - address.dx) + (h11 - h01) * address.dx) * fsize * uv_scale;
  if (((options & Image::RepeatU) == 0u) && (uv.x < 0.0f))
    gradient.x = 0.0f;
  if (((options & Image::RepeatV) == 0u) && (uv.y < 0.0f))
    gradient.y = 0.0f;
}

ETX_SHARED_INLINE void bump_mapping_shared_apply(ETX_IN(SurfaceDerivatives, derivatives), float height, ETX_IN(float2, gradient), ETX_IN(float3, geo_normal),
  ETX_IN(float3, incoming_direction), ETX_INOUT(float3, normal), ETX_INOUT(float3, tangent), ETX_INOUT(float3, bitangent)) {
  if ((height == 0.0f) && (gradient.x == 0.0f) && (gradient.y == 0.0f))
    return;
  const float orientation = dot(derivatives.normal, normal) < 0.0f ? -1.0f : 1.0f;
  const float3 dpdu = derivatives.dpdu - normal * dot(normal, derivatives.dpdu);
  const float3 dpdv = derivatives.dpdv - normal * dot(normal, derivatives.dpdv);
  float3 bumped_du = dpdu + gradient.x * normal + height * orientation * derivatives.dndu;
  float3 bumped_dv = dpdv + gradient.y * normal + height * orientation * derivatives.dndv;
  const float u_scale = max(max(abs(bumped_du.x), abs(bumped_du.y)), abs(bumped_du.z));
  const float v_scale = max(max(abs(bumped_dv.x), abs(bumped_dv.y)), abs(bumped_dv.z));
  if ((u_scale == 0.0f) || (v_scale == 0.0f) || (surface_derivatives_shared_finite(bumped_du) == false) || (surface_derivatives_shared_finite(bumped_dv) == false))
    return;
  bumped_du /= u_scale;
  bumped_dv /= v_scale;
  float3 bumped_normal = cross(bumped_du, bumped_dv);
  const float scale = max(max(abs(bumped_normal.x), abs(bumped_normal.y)), abs(bumped_normal.z));
  if ((scale == 0.0f) || (surface_derivatives_shared_finite(bumped_normal) == false))
    return;
  bumped_normal /= scale;
  if (dot(bumped_normal, normal) < 0.0f)
    bumped_normal = -bumped_normal;
  scene_math_shared_finalize_shading_frame(bumped_normal, normal, bumped_du, bumped_dv, geo_normal, incoming_direction, normal, tangent, bitangent);
}
