#pragma once

ETX_SHARED_INLINE bool pixel_filter_contains_uv(ETX_IN(float2, uv)) {
  return (uv.x >= -1.0f) && (uv.y >= -1.0f) && (uv.x < 1.0f) && (uv.y < 1.0f);
}

ETX_SHARED_INLINE float2 pixel_filter_sample_uv(ETX_IN(uint2, pixel), ETX_IN(uint2, dimensions), ETX_IN(float2, pixel_sample), ETX_IN(float2, filter_offset)) {
  return float2((float(pixel.x) + pixel_sample.x + filter_offset.x) / float(dimensions.x) * 2.0f - 1.0f,
    (float(pixel.y) + pixel_sample.y + filter_offset.y) / float(dimensions.y) * 2.0f - 1.0f);
}

// Sampling opposite offsets gives camera samples and light splats the same box-convolved filter.
ETX_SHARED_INLINE float2 pixel_filter_splat_uv(ETX_IN(float2, uv), ETX_IN(uint2, dimensions), ETX_IN(float2, filter_offset)) {
  return uv - float2(2.0f * filter_offset.x / float(dimensions.x), 2.0f * filter_offset.y / float(dimensions.y));
}
