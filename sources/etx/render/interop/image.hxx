#pragma once

#include "interop.hxx"

struct ETX_ALIGNED Image {
  enum class Format : uint32_t {
    Undefined,
    RGBA32F,
    RGBA8,
    BC1,
    BC1_SRGB,
    BC2,
    BC2_SRGB,
    BC3,
    BC3_SRGB,
    BC4,
    BC5,
    BC6H,
    BC6H_SIGNED,
    BC7,
    BC7_SRGB,
  };

  enum : uint32_t {
    Regular = 0u,
    BuildSamplingTable = 1u << 0u,
    RepeatU = 1u << 1u,
    RepeatV = 1u << 2u,
    SkipSRGBConversion = 1u << 3u,
    HasAlphaChannel = 1u << 4u,
    UniformSamplingTable = 1u << 5u,

    Committed = 1u << 6u,
  };

  float2 fsize ETX_INIT({});
  float2 offset ETX_INIT({});
  float2 scale ETX_INIT((float2{1.0f, 1.0f}));
  float normalization ETX_INIT(0.0f);

  uint2 isize ETX_INIT({});
  uint32_t options ETX_INIT(0u);
  Format format ETX_INIT(Format::Undefined);
  uint32_t data_size ETX_INIT(0u);

  uint32_t pixel_data_offset ETX_INIT(kInvalidIndex);
  uint32_t x_distribution_entries_offset ETX_INIT(kInvalidIndex);
  uint32_t y_distribution_entries_offset ETX_INIT(kInvalidIndex);
  uint32_t x_entries_stride ETX_INIT(0u);

  uint32_t x_distribution_count ETX_INIT(0u);
  uint32_t y_entries_count ETX_INIT(0u);
  float y_distribution_total_weight ETX_INIT(0.0f);
  uint32_t pixel_data_stride ETX_INIT(0u);

  // Chunk indices for packed payload streams.
  uint32_t pixel_data_chunk_index ETX_INIT(kInvalidIndex);
  uint32_t x_distribution_chunk_index ETX_INIT(kInvalidIndex);
  uint32_t y_distribution_chunk_index ETX_INIT(kInvalidIndex);
};

struct ETX_ALIGNED ImageSample {
  float2 uv ETX_INIT({});
  float pdf ETX_INIT(0.0f);
  uint2 location ETX_INIT({});
  float4 eval ETX_INIT((float4{1.0f, 1.0f, 1.0f, 1.0f}));
};

ETX_SHARED_INLINE float image_tex_coord_repeat(float u, float size) {
  if (size <= 0.0f) {
    return 0.0f;
  }

  float x = fmod(u, size);
  return (x < 0.0f) ? (x + size) : x;
}

ETX_SHARED_INLINE float image_tex_coord_clamp(float u, float size) {
#if defined(__cplusplus)
  return clamp(u, 0.0f, nextafterf(size, 0.0f));
#else
  float max_u = 0.0f;
  if (size > 0.0f) {
    max_u = asfloat(asuint(size) - 1u);
  }
  return clamp(u, 0.0f, max_u);
#endif
}

ETX_SHARED_INLINE float image_tex_coord_u(float u, float size, uint32_t options) {
  return ((options & Image::RepeatU) != 0u) ? image_tex_coord_repeat(u, size) : image_tex_coord_clamp(u, size);
}

ETX_SHARED_INLINE float image_tex_coord_v(float u, float size, uint32_t options) {
  return ((options & Image::RepeatV) != 0u) ? image_tex_coord_repeat(u, size) : image_tex_coord_clamp(u, size);
}

ETX_SHARED_INLINE float image_sample_interpolate(float rnd, float cdf_0, float cdf_1) {
  float result = rnd - cdf_0;
  float cdf_delta = cdf_1 - cdf_0;
  if (cdf_delta > 0.0f) {
    result = result / cdf_delta;
  }
  return result;
}

ETX_SHARED_INLINE float2 image_sample_uv_from_distribution(
  ETX_IN(float2, rnd), ETX_IN(uint2, location), ETX_IN(float2, image_fsize), float x_cdf_0, float x_cdf_1, float y_cdf_0, float y_cdf_1) {
  float dx = image_sample_interpolate(rnd.x, x_cdf_0, x_cdf_1);
  float dy = image_sample_interpolate(rnd.y, y_cdf_0, y_cdf_1);

  float inv_width = (image_fsize.x > 0.0f) ? (1.0f / image_fsize.x) : 0.0f;
  float inv_height = (image_fsize.y > 0.0f) ? (1.0f / image_fsize.y) : 0.0f;
  return float2((float(location.x) + dx) * inv_width, (float(location.y) + dy) * inv_height);
}
