#pragma once

#include "math_shared.hxx"

ETX_SHARED_INLINE float3 medium_position_before_surface(ETX_IN(float3, position), ETX_IN(float3, surface_position), ETX_IN(float3, geometric_normal),
  ETX_IN(float3, incident_direction)) {
  const float3 incident_normal = geometric_normal * ((dot(geometric_normal, incident_direction) > 0.0f) ? -1.0f : 1.0f);
  const float separation = dot(position - surface_position, incident_normal);
  const float surface_offset = dot(offset_ray(surface_position, incident_normal) - surface_position, incident_normal);
  if (separation > surface_offset) {
    return position;
  }

  // A medium event can round onto or across its terminal surface. Use the same
  // representable, incident-side offset as surface rays, without changing its sampled distance or PDF.
  return offset_ray(position + incident_normal * max(0.0f, -separation), incident_normal);
}
