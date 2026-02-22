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

struct ImageSampleSharedCPUContext {
  const Image& image;
};

ETX_SHARED_INLINE uint32_t image_sample_shared_cpu_y_count(ETX_IN(ImageSampleSharedCPUContext, context));
ETX_SHARED_INLINE uint32_t image_sample_shared_cpu_x_count(ETX_IN(ImageSampleSharedCPUContext, context), uint32_t y_index);
ETX_SHARED_INLINE float2 image_sample_shared_cpu_image_fsize(ETX_IN(ImageSampleSharedCPUContext, context));
ETX_SHARED_INLINE uint32_t image_sample_shared_cpu_sample_y(ETX_INOUT(ImageSampleSharedCPUContext, context), float rnd, ETX_OUT(float, pdf));
ETX_SHARED_INLINE uint32_t image_sample_shared_cpu_sample_x(ETX_INOUT(ImageSampleSharedCPUContext, context), uint32_t y_index, float rnd, ETX_OUT(float, pdf));
ETX_SHARED_INLINE float image_sample_shared_cpu_cdf_y(ETX_IN(ImageSampleSharedCPUContext, context), uint32_t y_index);
ETX_SHARED_INLINE float image_sample_shared_cpu_cdf_x(ETX_IN(ImageSampleSharedCPUContext, context), uint32_t y_index, uint32_t x_index);
ETX_SHARED_INLINE bool image_sample_shared_distribution(
  ETX_INOUT(ImageSampleSharedCPUContext, context), ETX_IN(float2, rnd), ETX_OUT(float, image_pdf), ETX_OUT(uint2, location), ETX_OUT(float2, uv));

struct Image : public ::Image {
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

  ETX_SHARED_INLINE Gather gather(const float2& in_uv) const {
    ImageFilterSharedAddress sample = image_filter_shared_address(in_uv, fsize, isize, options);

    const auto& p00 = pixel(sample.col_0, sample.row_0) * (1.0f - sample.dx) * (1.0f - sample.dy);
    ETX_VALIDATE(p00);
    const auto& p01 = pixel(sample.col_1, sample.row_0) * (sample.dx) * (1.0f - sample.dy);
    ETX_VALIDATE(p01);
    const auto& p10 = pixel(sample.col_0, sample.row_1) * (1.0f - sample.dx) * (sample.dy);
    ETX_VALIDATE(p10);
    const auto& p11 = pixel(sample.col_1, sample.row_1) * (sample.dx) * (sample.dy);
    ETX_VALIDATE(p11);

    return {p00, p01, p10, p11, sample.row_0, sample.row_1};
  }

  ETX_SHARED_INLINE float4 evaluate(const float2& in_uv, float* pdf) const {
    auto g = gather(in_uv);

    if (pdf) {
      float s_t = ((options & UniformSamplingTable) || (isize.y == 1u) ? 1.0f : max(0.0f, sinf(kPi * saturate(in_uv.y + 0.0f / fsize.y))));
      auto t = luminance(to_float3(g.p00 + g.p01)) * s_t;
      float s_b = ((options & UniformSamplingTable) || (isize.y == 1u) ? 1.0f : max(0.0f, sinf(kPi * saturate(in_uv.y + 1.0f / fsize.y))));
      auto b = luminance(to_float3(g.p10 + g.p11)) * s_b;
      *pdf = (t + b) / normalization;
      ETX_VALIDATE(*pdf);
    }

    return g.p00 + g.p01 + g.p10 + g.p11;
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

    return pixels.f32[i];
  }

  ETX_SHARED_INLINE float4 pixel(uint32_t x, uint32_t y) const {
    uint32_t i = min(x + y * isize.x, isize.x * isize.y - 1u);
    return pixel(i);
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
    ImageSampleSharedCPUContext context = {*this};
    float2 uv = rnd;
    bool sampled = image_sample_shared_distribution(context, rnd, image_pdf, location, uv);
    (void)sampled;
    eval = evaluate(uv, &image_pdf);
    return uv;
  }

  ETX_SHARED_INLINE float2 sample(const float2& rnd) const {
    float image_pdf = 0.0f;
    uint2 location = {};
    float4 eval = {};
    return sample(rnd, image_pdf, location, eval);
  }

  ETX_SHARED_INLINE float4 read(const float2& uv) const {
    if ((isize.x == 0u) || (isize.y == 0u)) {
      return float4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    float2 normalized_uv = {
      (fsize.x > 0.0f) ? (uv.x / fsize.x) : 0.0f,
      (fsize.y > 0.0f) ? (uv.y / fsize.y) : 0.0f,
    };
    ImageFilterSharedAddress sample = image_filter_shared_address(normalized_uv, fsize, isize, options);

    float4 p00 = pixel(sample.col_0, sample.row_0);
    float4 p01 = pixel(sample.col_1, sample.row_0);
    float4 p10 = pixel(sample.col_0, sample.row_1);
    float4 p11 = pixel(sample.col_1, sample.row_1);
    return image_filter_shared_bilinear(p00, p01, p10, p11, sample.dx, sample.dy);
  }
};

ETX_SHARED_INLINE uint32_t image_sample_shared_cpu_y_count(ETX_IN(ImageSampleSharedCPUContext, context)) {
  return static_cast<uint32_t>(context.image.y_distribution.values.count);
}

ETX_SHARED_INLINE uint32_t image_sample_shared_cpu_x_count(ETX_IN(ImageSampleSharedCPUContext, context), uint32_t y_index) {
  ETX_ASSERT(y_index < context.image.x_distributions.count);
  return static_cast<uint32_t>(context.image.x_distributions[y_index].values.count);
}

ETX_SHARED_INLINE float2 image_sample_shared_cpu_image_fsize(ETX_IN(ImageSampleSharedCPUContext, context)) {
  return context.image.fsize;
}

ETX_SHARED_INLINE uint32_t image_sample_shared_cpu_sample_y(ETX_INOUT(ImageSampleSharedCPUContext, context), float rnd, ETX_OUT(float, pdf)) {
  return context.image.y_distribution.sample(rnd, pdf);
}

ETX_SHARED_INLINE uint32_t image_sample_shared_cpu_sample_x(ETX_INOUT(ImageSampleSharedCPUContext, context), uint32_t y_index, float rnd, ETX_OUT(float, pdf)) {
  ETX_ASSERT(y_index < context.image.x_distributions.count);
  return context.image.x_distributions[y_index].sample(rnd, pdf);
}

ETX_SHARED_INLINE float image_sample_shared_cpu_cdf_y(ETX_IN(ImageSampleSharedCPUContext, context), uint32_t y_index) {
  ETX_ASSERT(y_index < context.image.y_distribution.values.count);
  return context.image.y_distribution.values[y_index].cdf;
}

ETX_SHARED_INLINE float image_sample_shared_cpu_cdf_x(ETX_IN(ImageSampleSharedCPUContext, context), uint32_t y_index, uint32_t x_index) {
  ETX_ASSERT(y_index < context.image.x_distributions.count);
  ETX_ASSERT(x_index < context.image.x_distributions[y_index].values.count);
  return context.image.x_distributions[y_index].values[x_index].cdf;
}

#define ETX_IMAGE_SAMPLE_SHARED_CONTEXT_TYPE ImageSampleSharedCPUContext
#define ETX_IMAGE_SAMPLE_SHARED_Y_COUNT(context) image_sample_shared_cpu_y_count(context)
#define ETX_IMAGE_SAMPLE_SHARED_X_COUNT(context, y_index) image_sample_shared_cpu_x_count(context, y_index)
#define ETX_IMAGE_SAMPLE_SHARED_IMAGE_FSIZE(context) image_sample_shared_cpu_image_fsize(context)
#define ETX_IMAGE_SAMPLE_SHARED_SAMPLE_Y(context, rnd, pdf) image_sample_shared_cpu_sample_y(context, rnd, pdf)
#define ETX_IMAGE_SAMPLE_SHARED_SAMPLE_X(context, y_index, rnd, pdf) image_sample_shared_cpu_sample_x(context, y_index, rnd, pdf)
#define ETX_IMAGE_SAMPLE_SHARED_CDF_Y(context, y_index) image_sample_shared_cpu_cdf_y(context, y_index)
#define ETX_IMAGE_SAMPLE_SHARED_CDF_X(context, y_index, x_index) image_sample_shared_cpu_cdf_x(context, y_index, x_index)
#include <etx/render/interop/image_sample_shared.hxx>
#undef ETX_IMAGE_SAMPLE_SHARED_CDF_X
#undef ETX_IMAGE_SAMPLE_SHARED_CDF_Y
#undef ETX_IMAGE_SAMPLE_SHARED_SAMPLE_X
#undef ETX_IMAGE_SAMPLE_SHARED_SAMPLE_Y
#undef ETX_IMAGE_SAMPLE_SHARED_IMAGE_FSIZE
#undef ETX_IMAGE_SAMPLE_SHARED_X_COUNT
#undef ETX_IMAGE_SAMPLE_SHARED_Y_COUNT
#undef ETX_IMAGE_SAMPLE_SHARED_CONTEXT_TYPE

}  // namespace etx
