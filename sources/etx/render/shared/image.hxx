#pragma once

#include <etx/render/interop/image.hxx>
#include <etx/render/interop/image_filter_shared.hxx>
#include <etx/render/shared/distribution.hxx>
#include <etx/render/shared/buffer_view.hxx>
#include <etx/render/shared/spectrum.hxx>

extern "C" {
#define BCDEC_BC4BC5_PRECISE
#include <bcdec.h>
}

// Compile-time flag to control DDS BC texture loading behavior
// Default: disabled (decompress BC to RGBA8/RGBA32F at load time)
// When enabled: store compressed BC data directly (decompress at runtime during sampling)
// Note: BC format support is always available for runtime sampling
#define ETX_STORE_COMPRESSED_BC 0

namespace etx {

struct Image;

ETX_SHARED_INLINE bool image_sample_distribution_cpu(ETX_IN(Image, image), ETX_IN(float2, rnd), ETX_OUT(float, image_pdf), ETX_OUT(uint2, location), ETX_OUT(float2, uv));

struct Image {
  using Format = ::Image::Format;
  enum : uint32_t {
    Regular = ::Image::Regular,
    BuildSamplingTable = ::Image::BuildSamplingTable,
    RepeatU = ::Image::RepeatU,
    RepeatV = ::Image::RepeatV,
    RepeatW = ::Image::RepeatW,
    SkipSRGBConversion = ::Image::SkipSRGBConversion,
    HasAlphaChannel = ::Image::HasAlphaChannel,
    UniformSamplingTable = ::Image::UniformSamplingTable,
    Committed = ::Image::Committed,
  };

  float3 fsize = {};
  float3 offset = {};
  float3 scale = float3{1.0f, 1.0f, 1.0f};
  float normalization = 0.0f;

  uint3 isize = {};
  uint32_t options = 0u;
  Format format = Format::Undefined;
  uint32_t data_size = 0u;

  struct Gather {
    float4 p00 = {};
    float4 p01 = {};
    float4 p10 = {};
    float4 p11 = {};
    uint32_t row_0 = 0;
    uint32_t row_1 = 0;
  };

  // Runtime pixel access (f32/u8/compressed views into external storage) - RENDERING
  // Compressed BC data view is always available for sampling support
  union {
    ArrayView<float4> f32;
    ArrayView<float> r32;
    ArrayView<ubyte4> u8;
    ArrayView<uint8_t> compressed;  // BC compressed data
  } pixels = {};

  // View to x distribution data (points to external storage)
  ArrayView<Distribution> x_distributions;

  // View to y distribution data (points to external storage)
  Distribution y_distribution = {};

  // Backing buffer references for CPU-side storage.
  BufferHandle pixel_buffer = {};
  BufferHandle distribution_buffer = {};

  BufferView data = {};
  BufferView x_distributions_storage = {};
  BufferView y_distribution_storage = {};
  BufferView x_distributions_buffer = {};

  ETX_SHARED_INLINE uint32_t next_coord_u(uint32_t value) const {
    if (isize.x == 0u) {
      return 0u;
    }

    if ((options & RepeatU) != 0u) {
      return (value + 1u) % isize.x;
    }

    return min(value + 1u, isize.x - 1u);
  }

  ETX_SHARED_INLINE uint32_t next_coord_v(uint32_t value) const {
    if (isize.y == 0u) {
      return 0u;
    }

    if ((options & RepeatV) != 0u) {
      return (value + 1u) % isize.y;
    }

    return min(value + 1u, isize.y - 1u);
  }

  ETX_SHARED_INLINE uint32_t next_coord_w(uint32_t value) const {
    if (isize.z == 0u) {
      return 0u;
    }

    if ((options & RepeatW) != 0u) {
      return (value + 1u) % isize.z;
    }

    return min(value + 1u, isize.z - 1u);
  }

  ETX_SHARED_INLINE Gather gather(const float2& in_uv) const {
    const float2 uv = in_uv * float2{fsize.x, fsize.y};
    float x0 = tex_coord_u(uv.x, fsize.x);
    float y0 = tex_coord_v(uv.y, fsize.y);
    float dx = x0 - floorf(x0);
    float dy = y0 - floorf(y0);

    uint32_t row_0 = clamp(static_cast<uint32_t>(y0), 0u, isize.y - 1u);
    uint32_t row_1 = next_coord_v(row_0);
    uint32_t col_0 = clamp(static_cast<uint32_t>(x0), 0u, isize.x - 1u);
    uint32_t col_1 = next_coord_u(col_0);

    const auto& p00 = pixel(col_0, row_0) * (1.0f - dx) * (1.0f - dy);
    ETX_VALIDATE(p00);
    const auto& p01 = pixel(col_1, row_0) * (dx) * (1.0f - dy);
    ETX_VALIDATE(p01);
    const auto& p10 = pixel(col_0, row_1) * (1.0f - dx) * (dy);
    ETX_VALIDATE(p10);
    const auto& p11 = pixel(col_1, row_1) * (dx) * (dy);
    ETX_VALIDATE(p11);

    return {p00, p01, p10, p11, row_0, row_1};
  }

  ETX_SHARED_INLINE float4 evaluate(const float2& in_uv, float* pdf) const {
    auto g = gather(in_uv);

    if (pdf) {
      *pdf = 0.0f;
      if (normalization > 0.0f) {
        float s_t = ((options & UniformSamplingTable) || (isize.y == 1u) ? 1.0f : max(0.0f, sinf(kPi * saturate(in_uv.y + 0.0f / fsize.y))));
        auto t = luminance(to_float3(g.p00 + g.p01)) * s_t;
        float s_b = ((options & UniformSamplingTable) || (isize.y == 1u) ? 1.0f : max(0.0f, sinf(kPi * saturate(in_uv.y + 1.0f / fsize.y))));
        auto b = luminance(to_float3(g.p10 + g.p11)) * s_b;
        *pdf = (t + b) / normalization;
      }
      ETX_VALIDATE(*pdf);
    }

    return g.p00 + g.p01 + g.p10 + g.p11;
  }

  ETX_SHARED_INLINE float4 evaluate_rgba32f_fast_3d(const float3& in_uvw) const {
    ETX_ASSERT(format == Format::RGBA32F);
    ETX_ASSERT(pixels.f32.a != nullptr);
    ETX_ASSERT((isize.x > 0u) && (isize.y > 0u) && (isize.z > 0u));

    const float3 uvw = in_uvw * fsize;
    const float x0 = tex_coord_u(uvw.x, fsize.x);
    const float y0 = tex_coord_v(uvw.y, fsize.y);
    const float z0 = tex_coord_w(uvw.z, fsize.z);
    const float dx = x0 - floorf(x0);
    const float dy = y0 - floorf(y0);
    const float dz = z0 - floorf(z0);

    const uint32_t slice_0 = clamp(static_cast<uint32_t>(z0), 0u, isize.z - 1u);
    const uint32_t slice_1 = next_coord_w(slice_0);
    const uint32_t row_0 = clamp(static_cast<uint32_t>(y0), 0u, isize.y - 1u);
    const uint32_t row_1 = next_coord_v(row_0);
    const uint32_t col_0 = clamp(static_cast<uint32_t>(x0), 0u, isize.x - 1u);
    const uint32_t col_1 = next_coord_u(col_0);

    const uint32_t row_stride = isize.x;
    const uint32_t slice_stride = isize.x * isize.y;
    const uint32_t slice_offset_0 = slice_0 * slice_stride;
    const uint32_t slice_offset_1 = slice_1 * slice_stride;
    const uint32_t row_offset_0 = row_0 * row_stride;
    const uint32_t row_offset_1 = row_1 * row_stride;
    const float4 p000 = pixels.f32.a[slice_offset_0 + row_offset_0 + col_0];
    const float4 p001 = pixels.f32.a[slice_offset_0 + row_offset_0 + col_1];
    const float4 p010 = pixels.f32.a[slice_offset_0 + row_offset_1 + col_0];
    const float4 p011 = pixels.f32.a[slice_offset_0 + row_offset_1 + col_1];
    const float4 p100 = pixels.f32.a[slice_offset_1 + row_offset_0 + col_0];
    const float4 p101 = pixels.f32.a[slice_offset_1 + row_offset_0 + col_1];
    const float4 p110 = pixels.f32.a[slice_offset_1 + row_offset_1 + col_0];
    const float4 p111 = pixels.f32.a[slice_offset_1 + row_offset_1 + col_1];

    const float4 bottom = image_filter_shared_bilinear(p000, p001, p010, p011, dx, dy);
    const float4 top = image_filter_shared_bilinear(p100, p101, p110, p111, dx, dy);
    return bottom * (1.0f - dz) + top * dz;
  }

  ETX_SHARED_INLINE float4 evaluate_rgba32f_fast(const float2& in_uv) const {
    ETX_ASSERT(format == Format::RGBA32F);
    ETX_ASSERT(pixels.f32.a != nullptr);
    ETX_ASSERT((isize.x > 0u) && (isize.y > 0u));

    const float2 uv = in_uv * float2{fsize.x, fsize.y};
    const float x0 = tex_coord_u(uv.x, fsize.x);
    const float y0 = tex_coord_v(uv.y, fsize.y);
    const float dx = x0 - floorf(x0);
    const float dy = y0 - floorf(y0);

    const uint32_t row_0 = clamp(static_cast<uint32_t>(y0), 0u, isize.y - 1u);
    const uint32_t row_1 = next_coord_v(row_0);
    const uint32_t col_0 = clamp(static_cast<uint32_t>(x0), 0u, isize.x - 1u);
    const uint32_t col_1 = next_coord_u(col_0);

    const float wx0 = 1.0f - dx;
    const float wy0 = 1.0f - dy;
    const uint32_t row_offset_0 = row_0 * isize.x;
    const uint32_t row_offset_1 = row_1 * isize.x;
    const float4 p00 = pixels.f32.a[row_offset_0 + col_0] * (wx0 * wy0);
    const float4 p01 = pixels.f32.a[row_offset_0 + col_1] * (dx * wy0);
    const float4 p10 = pixels.f32.a[row_offset_1 + col_0] * (wx0 * dy);
    const float4 p11 = pixels.f32.a[row_offset_1 + col_1] * (dx * dy);
    return p00 + p01 + p10 + p11;
  }

  ETX_SHARED_INLINE float evaluate_r32f_fast_3d(const float3& in_uvw) const {
    ETX_ASSERT(format == Format::R32F);
    ETX_ASSERT(pixels.r32.a != nullptr);
    ETX_ASSERT((isize.x > 0u) && (isize.y > 0u) && (isize.z > 0u));

    const float3 uvw = in_uvw * fsize;
    const float x0 = tex_coord_u(uvw.x, fsize.x);
    const float y0 = tex_coord_v(uvw.y, fsize.y);
    const float z0 = tex_coord_w(uvw.z, fsize.z);
    const float dx = x0 - floorf(x0);
    const float dy = y0 - floorf(y0);
    const float dz = z0 - floorf(z0);

    const uint32_t slice_0 = clamp(static_cast<uint32_t>(z0), 0u, isize.z - 1u);
    const uint32_t slice_1 = next_coord_w(slice_0);
    const uint32_t row_0 = clamp(static_cast<uint32_t>(y0), 0u, isize.y - 1u);
    const uint32_t row_1 = next_coord_v(row_0);
    const uint32_t col_0 = clamp(static_cast<uint32_t>(x0), 0u, isize.x - 1u);
    const uint32_t col_1 = next_coord_u(col_0);

    const uint32_t row_stride = isize.x;
    const uint32_t slice_stride = isize.x * isize.y;
    const uint32_t slice_offset_0 = slice_0 * slice_stride;
    const uint32_t slice_offset_1 = slice_1 * slice_stride;
    const uint32_t row_offset_0 = row_0 * row_stride;
    const uint32_t row_offset_1 = row_1 * row_stride;
    const float p000 = pixels.r32.a[slice_offset_0 + row_offset_0 + col_0];
    const float p001 = pixels.r32.a[slice_offset_0 + row_offset_0 + col_1];
    const float p010 = pixels.r32.a[slice_offset_0 + row_offset_1 + col_0];
    const float p011 = pixels.r32.a[slice_offset_0 + row_offset_1 + col_1];
    const float p100 = pixels.r32.a[slice_offset_1 + row_offset_0 + col_0];
    const float p101 = pixels.r32.a[slice_offset_1 + row_offset_0 + col_1];
    const float p110 = pixels.r32.a[slice_offset_1 + row_offset_1 + col_0];
    const float p111 = pixels.r32.a[slice_offset_1 + row_offset_1 + col_1];

    const float bottom = p000 * (1.0f - dx) * (1.0f - dy) + p001 * dx * (1.0f - dy) + p010 * (1.0f - dx) * dy + p011 * dx * dy;
    const float top = p100 * (1.0f - dx) * (1.0f - dy) + p101 * dx * (1.0f - dy) + p110 * (1.0f - dx) * dy + p111 * dx * dy;
    return bottom * (1.0f - dz) + top * dz;
  }

  ETX_SHARED_INLINE float evaluate_alpha(const float2& in_uv) const {
    auto g = gather(in_uv);
    return g.p00.w + g.p01.w + g.p10.w + g.p11.w;
  }

  static ETX_SHARED_INLINE bool is_compressed_bc_format(Format format) {
    return format == Format::BC1 || format == Format::BC1_SRGB || format == Format::BC2 || format == Format::BC2_SRGB || format == Format::BC3 || format == Format::BC3_SRGB ||
           format == Format::BC4 || format == Format::BC5 || format == Format::BC6H || format == Format::BC6H_SIGNED || format == Format::BC7 || format == Format::BC7_SRGB;
  }

  static ETX_SHARED_INLINE uint32_t get_bc_block_size(Format format) {
    switch (format) {
      case Format::BC1:
      case Format::BC1_SRGB:
      case Format::BC4:
        return 8;  // 8 bytes
      case Format::BC2:
      case Format::BC2_SRGB:
      case Format::BC3:
      case Format::BC3_SRGB:
      case Format::BC5:
      case Format::BC6H:
      case Format::BC6H_SIGNED:
      case Format::BC7:
      case Format::BC7_SRGB:
        return 16;  // 16 bytes
      default:
        return 0;
    }
  }

  static ETX_SHARED_INLINE bool is_bc_srgb_format(Format format) {
    return format == Format::BC1_SRGB || format == Format::BC2_SRGB || format == Format::BC3_SRGB || format == Format::BC7_SRGB;
  }

  static ETX_SHARED_INLINE bool is_bc_signed_format(Format format) {
    return format == Format::BC6H_SIGNED;
  }

  static ETX_SHARED_INLINE void decompress_bc_to_rgba(Format format, const uint8_t* block_data, uint8_t decompressed_rgba[64], bool is_signed = false) {
    float decompressed_float[48] = {};

    switch (format) {
      case Format::BC1:
      case Format::BC1_SRGB:
        bcdec_bc1(block_data, decompressed_rgba, 4 * 4);
        break;
      case Format::BC2:
      case Format::BC2_SRGB:
        bcdec_bc2(block_data, decompressed_rgba, 4 * 4);
        break;
      case Format::BC3:
      case Format::BC3_SRGB:
        bcdec_bc3(block_data, decompressed_rgba, 4 * 4);
        break;
      case Format::BC4:
        bcdec_bc4(block_data, decompressed_rgba, 4 * 1, is_signed ? 1 : 0);
        for (uint32_t p = 0; p < 16; ++p) {
          uint8_t r = decompressed_rgba[p];
          decompressed_rgba[p * 4 + 0] = r;
          decompressed_rgba[p * 4 + 1] = r;
          decompressed_rgba[p * 4 + 2] = r;
          decompressed_rgba[p * 4 + 3] = 255;
        }
        break;
      case Format::BC5: {
        float bc5_float[32] = {};
        bcdec_bc5_float(block_data, bc5_float, 4 * 2, is_signed ? 1 : 0);
        for (uint32_t p = 0; p < 16; ++p) {
          float r = bc5_float[p * 2 + 0];
          float g = bc5_float[p * 2 + 1];
          decompressed_rgba[p * 4 + 0] = static_cast<uint8_t>(max(0.0f, min(255.0f, r * 255.0f)));
          decompressed_rgba[p * 4 + 1] = static_cast<uint8_t>(max(0.0f, min(255.0f, g * 255.0f)));
          decompressed_rgba[p * 4 + 2] = 0;
          decompressed_rgba[p * 4 + 3] = 255;
        }
        break;
      }
      case Format::BC6H:
      case Format::BC6H_SIGNED:
        for (uint32_t i = 0; i < 64; ++i) {
          decompressed_rgba[i] = 128;
        }
        break;
      case Format::BC7:
      case Format::BC7_SRGB:
        bcdec_bc7(block_data, decompressed_rgba, 4 * 4);
        break;
      default:
        // Unknown format - fill with black
        for (uint32_t i = 0; i < 64; ++i) {
          decompressed_rgba[i] = (i % 4 == 3) ? 255 : 0;  // RGBA(0,0,0,255)
        }
        break;
    }
  }

  ETX_SHARED_INLINE float4 decompress_bc_pixel(uint32_t pixel_index) const {
    uint32_t pixel_x = pixel_index % isize.x;
    uint32_t pixel_y = pixel_index / isize.x;

    uint32_t corrected_pixel_y = pixel_y;

    if (format == Format::BC5) {
      uint32_t blocks_y = (isize.y + 3) / 4;
      uint32_t block_y = pixel_y / 4;
      uint32_t local_y = pixel_y % 4;
      corrected_pixel_y = (blocks_y - 1 - block_y) * 4 + (3 - local_y);
    } else {
      corrected_pixel_y = isize.y - 1 - pixel_y;
    }

    uint32_t block_x = pixel_x / 4;
    uint32_t block_y = corrected_pixel_y / 4;
    uint32_t local_x = pixel_x % 4;
    uint32_t local_y = corrected_pixel_y % 4;

    uint32_t blocks_per_row = (isize.x + 3) / 4;
    uint32_t block_index = block_y * blocks_per_row + block_x;
    uint32_t block_size = get_bc_block_size(format);
    uint32_t block_offset = block_index * block_size;

    const uint8_t* block_data = pixels.compressed.a + block_offset;

    float4 result;
    if (format == Format::BC6H || format == Format::BC6H_SIGNED) {
      // Use the correct signed/unsigned decompression based on format detected during loading
      float decompressed_float[48] = {};
      bcdec_bc6h_float(block_data, decompressed_float, 4 * 3, is_bc_signed_format(format) ? 1 : 0);

      uint32_t pixel_offset = (local_y * 4 + local_x) * 3;
      result = {decompressed_float[pixel_offset + 0], decompressed_float[pixel_offset + 1], decompressed_float[pixel_offset + 2], 1.0f};
    } else {
      uint8_t decompressed_rgba[64] = {};
      decompress_bc_to_rgba(format, block_data, decompressed_rgba, is_bc_signed_format(format));

      uint32_t pixel_offset = (local_y * 4 + local_x) * 4;
      result = {decompressed_rgba[pixel_offset + 0] / 255.0f, decompressed_rgba[pixel_offset + 1] / 255.0f, decompressed_rgba[pixel_offset + 2] / 255.0f,
        decompressed_rgba[pixel_offset + 3] / 255.0f};
    }

    if (format == Format::BC5) {
      float r = result.x;
      float g = result.y;
      float nx = r * 2.0f - 1.0f;
      float ny = g * 2.0f - 1.0f;
      float nz = 1.0f;

      float length = sqrtf(nx * nx + ny * ny + nz * nz);
      if (length > 0.0f) {
        nx /= length;
        ny /= length;
        nz /= length;
      }

      result = {nx * 0.5f + 0.5f, ny * 0.5f + 0.5f, nz * 0.5f + 0.5f, 1.0f};
    }

    // Apply sRGB to linear conversion if needed (but not for BC5 normal maps)
    if (is_bc_srgb_format(format)) {
      result.x = gamma_to_linear(result.x);
      result.y = gamma_to_linear(result.y);
      result.z = gamma_to_linear(result.z);
    }

    return result;
  }

  ETX_SHARED_INLINE float4 pixel(uint32_t i) const {
    ETX_ASSERT(format != Format::Undefined);

    if (format == Format::RGBA8)
      return to_float4(pixels.u8[i]);

    // Compressed BC formats - decompress on-the-fly for sampling
    if (is_compressed_bc_format(format)) {
      return decompress_bc_pixel(i);
    }

    if (format == Format::R32F) {
      const float value = pixels.r32[i];
      return float4(value, value, value, 1.0f);
    }

    return pixels.f32[i];
  }

  ETX_SHARED_INLINE float4 pixel(uint32_t x, uint32_t y) const {
    uint32_t i = min(x + y * isize.x, isize.x * isize.y - 1u);
    return pixel(i);
  }

  ETX_SHARED_INLINE float pixel_r32(uint32_t x, uint32_t y, uint32_t z) const {
    ETX_ASSERT(format == Format::R32F);
    ETX_ASSERT(pixels.r32.a != nullptr);
    const uint32_t i = min(x + y * isize.x + z * isize.x * isize.y, isize.x * isize.y * isize.z - 1u);
    return pixels.r32[i];
  }

  ETX_SHARED_INLINE float3 evaluate_normal(const float2& uv, float scale) const {
    float4 value = evaluate(uv, nullptr);
    return {
      scale * (value.x * 2.0f - 1.0f),
      scale * (value.y * 2.0f - 1.0f),
      scale * (value.z * 2.0f - 1.0f) + (1.0f - scale),
    };
  }

  ETX_SHARED_INLINE float2 sample(const float2& rnd, float& image_pdf, uint2& location, float4& eval) const {
    float y_pdf = 0.0f;
    location.y = y_distribution.sample(rnd.y, y_pdf);

    float x_pdf = 0.0f;
    const auto& x_distribution = x_distributions[location.y];
    location.x = x_distribution.sample(rnd.x, x_pdf);

    const auto& x0 = x_distribution.values[location.x];
    const auto& x1 = x_distribution.values[min(location.x + 1u, uint32_t(x_distribution.values.count) - 1u)];
    float dx = (rnd.x - x0.cdf);
    if ((x1.cdf - x0.cdf) > 0.0f) {
      dx /= (x1.cdf - x0.cdf);
    }

    const auto& y0 = y_distribution.values[location.y];
    const auto& y1 = y_distribution.values[min(location.y + 1u, uint32_t(y_distribution.values.count) - 1u)];
    float dy = (rnd.y - y0.cdf);
    if ((y1.cdf - y0.cdf) > 0.0f) {
      dy /= (y1.cdf - y0.cdf);
    }

    float2 uv = {
      (float(location.x) + dx) / fsize.x,
      (float(location.y) + dy) / fsize.y,
    };

    eval = evaluate(uv, &image_pdf);
    return uv;
  }

  ETX_SHARED_INLINE float2 sample(const float2& rnd) const {
    float image_pdf = 0.0f;
    uint2 location = {};
    float4 eval = {};
    return sample(rnd, image_pdf, location, eval);
  }

  ETX_SHARED_INLINE float tex_coord_repeat(float u, float size) const {
    float x = fmodf(u, size);
    return x < 0.0f ? (x + size) : x;
  }

  ETX_SHARED_INLINE float tex_coord_clamp(float u, float size) const {
    return clamp(u, 0.0f, nextafterf(size, 0.0f));
  }

  ETX_SHARED_INLINE float tex_coord_u(float u, float size) const {
    return (options & RepeatU) ? tex_coord_repeat(u, size) : tex_coord_clamp(u, size);
  }

  ETX_SHARED_INLINE float tex_coord_v(float u, float size) const {
    return (options & RepeatV) ? tex_coord_repeat(u, size) : tex_coord_clamp(u, size);
  }

  ETX_SHARED_INLINE float tex_coord_w(float u, float size) const {
    return (options & RepeatW) ? tex_coord_repeat(u, size) : tex_coord_clamp(u, size);
  }

  ETX_SHARED_INLINE float4 read(const float2& uv) const {
    float x0 = tex_coord_u(uv.x - 0.0f, fsize.x);
    float x1 = tex_coord_u(uv.x + 1.0f, fsize.x);
    float y0 = tex_coord_v(uv.y - 0.0f, fsize.y);
    float y1 = tex_coord_v(uv.y + 1.0f, fsize.y);
    float dx = x0 - floorf(x0);
    float dy = y0 - floorf(y0);
    const auto& p00 = pixel(uint32_t(x0), uint32_t(y0)) * (1.0f - dx) * (1.0f - dy);
    const auto& p01 = pixel(uint32_t(x1), uint32_t(y0)) * (dx) * (1.0f - dy);
    const auto& p10 = pixel(uint32_t(x0), uint32_t(y1)) * (1.0f - dx) * (dy);
    const auto& p11 = pixel(uint32_t(x1), uint32_t(y1)) * (dx) * (dy);
    return p00 + p01 + p10 + p11;
  }
};

ETX_SHARED_INLINE bool image_sample_distribution_cpu(ETX_IN(Image, image), ETX_IN(float2, rnd), ETX_OUT(float, image_pdf), ETX_OUT(uint2, location), ETX_OUT(float2, uv)) {
  image_pdf = 0.0f;
  location = uint2(0u, 0u);
  uv = rnd;

  const uint32_t y_count = static_cast<uint32_t>(image.y_distribution.values.count);
  if (y_count == 0u) {
    return false;
  }

  float y_pdf = 0.0f;
  location.y = image.y_distribution.sample(rnd.y, y_pdf);
  if ((location.y == kInvalidIndex) || (location.y >= y_count)) {
    return false;
  }

  ETX_ASSERT(location.y < image.x_distributions.count);
  const uint32_t x_count = static_cast<uint32_t>(image.x_distributions[location.y].values.count);
  if (x_count == 0u) {
    return false;
  }

  float x_pdf = 0.0f;
  location.x = image.x_distributions[location.y].sample(rnd.x, x_pdf);
  if ((location.x == kInvalidIndex) || (location.x >= x_count)) {
    return false;
  }

  const uint32_t x1_index = min(location.x + 1u, x_count);
  const uint32_t y1_index = min(location.y + 1u, y_count);

  ETX_ASSERT(location.y < image.x_distributions.count);
  ETX_ASSERT(location.x < image.x_distributions[location.y].values.count);
  ETX_ASSERT(x1_index <= image.x_distributions[location.y].values.count);
  ETX_ASSERT(location.y < image.y_distribution.values.count);
  ETX_ASSERT(y1_index <= image.y_distribution.values.count);
  ETX_ASSERT(image.x_distributions[location.y].values.a != nullptr);
  ETX_ASSERT(image.y_distribution.values.a != nullptr);

  const float x0_cdf = image.x_distributions[location.y].values[location.x].cdf;
  const float x1_cdf = image.x_distributions[location.y].values.a[x1_index].cdf;
  const float y0_cdf = image.y_distribution.values[location.y].cdf;
  const float y1_cdf = image.y_distribution.values.a[y1_index].cdf;

  uv = image_sample_uv_from_distribution(rnd, location, float2{image.fsize.x, image.fsize.y}, x0_cdf, x1_cdf, y0_cdf, y1_cdf);
  image_pdf = x_pdf * y_pdf;
  return true;
}

}  // namespace etx
