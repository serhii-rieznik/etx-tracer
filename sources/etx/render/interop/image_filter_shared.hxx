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

struct ETX_ALIGNED ImageFilterSharedAddress3D {
  uint32_t slice_0;
  uint32_t slice_1;
  uint32_t row_0;
  uint32_t row_1;
  uint32_t col_0;
  uint32_t col_1;
  float dx;
  float dy;
  float dz;
};

ETX_SHARED_INLINE uint32_t image_filter_shared_next_coord(uint32_t value, uint32_t size, uint32_t options, uint32_t repeat_option) {
  if (size == 0u) {
    return 0u;
  }

  if ((options & repeat_option) != 0u) {
    return (value + 1u) % size;
  }

  return min(value + 1u, size - 1u);
}

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
  result.row_1 = image_filter_shared_next_coord(result.row_0, size.y, options, Image::RepeatV);
  result.col_0 = min(uint32_t(x0), size.x - 1u);
  result.col_1 = image_filter_shared_next_coord(result.col_0, size.x, options, Image::RepeatU);
  return result;
}

ETX_SHARED_INLINE ImageFilterSharedAddress3D image_filter_shared_address_3d(ETX_IN(float3, uvw), ETX_IN(float3, fsize), ETX_IN(uint3, size), uint32_t options) {
  ETX_ZERO_INIT(ImageFilterSharedAddress3D, result);
  if ((size.x == 0u) || (size.y == 0u) || (size.z == 0u)) {
    return result;
  }

  float3 image_uvw = uvw * fsize;
  float x0 = image_tex_coord_u(image_uvw.x, fsize.x, options);
  float y0 = image_tex_coord_v(image_uvw.y, fsize.y, options);
  float z0 = image_tex_coord_w(image_uvw.z, fsize.z, options);

#if defined(__cplusplus)
  result.dx = x0 - floorf(x0);
  result.dy = y0 - floorf(y0);
  result.dz = z0 - floorf(z0);
#else
  result.dx = x0 - floor(x0);
  result.dy = y0 - floor(y0);
  result.dz = z0 - floor(z0);
#endif

  result.slice_0 = min(uint32_t(z0), size.z - 1u);
  result.slice_1 = image_filter_shared_next_coord(result.slice_0, size.z, options, Image::RepeatW);
  result.row_0 = min(uint32_t(y0), size.y - 1u);
  result.row_1 = image_filter_shared_next_coord(result.row_0, size.y, options, Image::RepeatV);
  result.col_0 = min(uint32_t(x0), size.x - 1u);
  result.col_1 = image_filter_shared_next_coord(result.col_0, size.x, options, Image::RepeatU);
  return result;
}

ETX_SHARED_INLINE float4 image_filter_shared_bilinear(ETX_IN(float4, p00), ETX_IN(float4, p01), ETX_IN(float4, p10), ETX_IN(float4, p11), float dx, float dy) {
  return p00 * (1.0f - dx) * (1.0f - dy) + p01 * dx * (1.0f - dy) + p10 * (1.0f - dx) * dy + p11 * dx * dy;
}

ETX_SHARED_INLINE float4 image_filter_shared_trilinear(ETX_IN(float4, p000), ETX_IN(float4, p001), ETX_IN(float4, p010), ETX_IN(float4, p011), ETX_IN(float4, p100),
  ETX_IN(float4, p101), ETX_IN(float4, p110), ETX_IN(float4, p111), float dx, float dy, float dz) {
  const float4 bottom = image_filter_shared_bilinear(p000, p001, p010, p011, dx, dy);
  const float4 top = image_filter_shared_bilinear(p100, p101, p110, p111, dx, dy);
  return bottom * (1.0f - dz) + top * dz;
}
