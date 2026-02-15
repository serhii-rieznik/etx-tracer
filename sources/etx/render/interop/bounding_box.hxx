#pragma once

#include "interop.hxx"

struct ETX_ALIGNED BoundingBox {
  float3 p_min ETX_INIT({});
  float pad0 ETX_INIT(0.0f);
  float3 p_max ETX_INIT({});
  float pad1 ETX_INIT(0.0f);
};

ETX_SHARED_INLINE float3 bounding_box_to_local(ETX_IN(BoundingBox, bbox), ETX_IN(float3, p)) {
  float3 size = bbox.p_max - bbox.p_min;
  float3 result = float3(0.0f, 0.0f, 0.0f);
  result.x = (size.x > kEpsilon) ? ((p.x - bbox.p_min.x) / size.x) : 0.0f;
  result.y = (size.y > kEpsilon) ? ((p.y - bbox.p_min.y) / size.y) : 0.0f;
  result.z = (size.z > kEpsilon) ? ((p.z - bbox.p_min.z) / size.z) : 0.0f;
  return result;
}

ETX_SHARED_INLINE float3 bounding_box_from_local(ETX_IN(BoundingBox, bbox), ETX_IN(float3, p)) {
  float3 size = bbox.p_max - bbox.p_min;
  return float3(p.x * size.x + bbox.p_min.x, p.y * size.y + bbox.p_min.y, p.z * size.z + bbox.p_min.z);
}

ETX_SHARED_INLINE bool bounding_box_contains(ETX_IN(BoundingBox, bbox), ETX_IN(float3, p)) {
  return (p.x >= bbox.p_min.x) && (p.y >= bbox.p_min.y) && (p.z >= bbox.p_min.z) &&  //
         (p.x <= bbox.p_max.x) && (p.y <= bbox.p_max.y) && (p.z <= bbox.p_max.z);
}
