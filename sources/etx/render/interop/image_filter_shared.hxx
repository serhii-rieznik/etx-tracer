#pragma once

#include "image.hxx"

struct ETX_ALIGNED ImageFilterSharedAddress {
  uint32_t row_0;
  uint32_t row_1;
  uint32_t col_0;
  uint32_t col_1;
  float dx;
  float dy;
};

ETX_SHARED_INLINE ImageFilterSharedAddress image_filter_shared_address(ETX_IN(float2, uv), ETX_IN(float2, fsize), ETX_IN(uint2, size), uint32_t options) {
  ETX_ZERO_INIT(ImageFilterSharedAddress, result);
  if ((size.x == 0u) || (size.y == 0u)) {
    return result;
  }

  float2 image_uv = uv * fsize;
  float x0 = image_tex_coord_u(image_uv.x, fsize.x, options);
  float y0 = image_tex_coord_v(image_uv.y, fsize.y, options);

#if defined(__cplusplus)
  result.dx = x0 - floorf(x0);
  result.dy = y0 - floorf(y0);
#else
  result.dx = x0 - floor(x0);
  result.dy = y0 - floor(y0);
#endif

  result.row_0 = min(uint32_t(y0), size.y - 1u);
  result.row_1 = min(result.row_0 + 1u, size.y - 1u);
  result.col_0 = min(uint32_t(x0), size.x - 1u);
  result.col_1 = min(result.col_0 + 1u, size.x - 1u);
  return result;
}

ETX_SHARED_INLINE float4 image_filter_shared_bilinear(
  ETX_IN(float4, p00), ETX_IN(float4, p01), ETX_IN(float4, p10), ETX_IN(float4, p11), float dx, float dy) {
  return p00 * (1.0f - dx) * (1.0f - dy) + p01 * dx * (1.0f - dy) + p10 * (1.0f - dx) * dy + p11 * dx * dy;
}
