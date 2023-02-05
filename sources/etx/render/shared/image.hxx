#pragma once

#include <etx/render/shared/distribution.hxx>
#include <etx/render/shared/spectrum.hxx>

namespace etx {

struct Image {
  enum class Format : uint32_t {
    Undefined,
    RGBA32F,
    RGBA8,
  };

  enum : uint32_t {
    Regular = 0u,
    BuildSamplingTable = 1u << 0u,
    RepeatU = 1u << 1u,
    RepeatV = 1u << 2u,
    Linear = 1u << 3u,
    HasAlphaChannel = 1u << 4u,
    UniformSamplingTable = 1u << 5u,
    DelayLoad = 1u << 6u,
  };

  struct Gather {
    float4 p00 = {};
    float4 p01 = {};
    float4 p10 = {};
    float4 p11 = {};
    uint32_t row_0 = 0;
    uint32_t row_1 = 0;
  };

  union {
    ArrayView<float4> f32;
    ArrayView<ubyte4> u8;
  } pixels = {};

  ArrayView<Distribution> x_distributions = {};
  Distribution y_distribution = {};
  float2 fsize = {};
  uint2 isize = {};
  float normalization = 0.0f;
  uint32_t options = 0;
  Format format = Format::Undefined;
  uint32_t pad = {};

  ETX_GPU_CODE Gather gather(const float2& in_uv) const {
    float2 uv = in_uv * fsize;
    auto x0 = tex_coord_u(uv.x, fsize.x);
    auto y0 = tex_coord_v(uv.y, fsize.y);
    float dx = x0 - floorf(x0);
    float dy = y0 - floorf(y0);

    uint32_t row_0 = clamp(static_cast<uint32_t>(y0), 0u, isize.y - 1u);
    uint32_t row_1 = clamp(row_0 + 1u, 0u, isize.y - 1u);
    uint32_t col_0 = clamp(static_cast<uint32_t>(x0), 0u, isize.x - 1u);
    uint32_t col_1 = clamp(col_0 + 1u, 0u, isize.x - 1u);

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

  ETX_GPU_CODE float4 evaluate(const float2& in_uv) const {
    auto g = gather(in_uv);
    return g.p00 + g.p01 + g.p10 + g.p11;
  }

  ETX_GPU_CODE float pdf(const float2& in_uv) const {
    auto g = gather(in_uv);
    auto t = luminance(to_float3(g.p00 + g.p01)) * ((options & UniformSamplingTable) || (fsize.y == 1.0f) ? 1.0f : sinf(kPi * saturate(in_uv.y + 0.0f / fsize.y)));
    auto b = luminance(to_float3(g.p10 + g.p11)) * ((options & UniformSamplingTable) || (fsize.y == 1.0f) ? 1.0f : sinf(kPi * saturate(in_uv.y + 1.0f / fsize.y)));
    return (t + b) / normalization;
  }

  ETX_GPU_CODE float4 pixel(uint32_t i) const {
    ETX_ASSERT(format != Format::Undefined);

    if (format == Format::RGBA8)
      return to_float4(pixels.u8[i]);
    else
      return pixels.f32[i];
  }

  ETX_GPU_CODE float4 pixel(uint32_t x, uint32_t y) const {
    int32_t i = min(x + y * isize.x, isize.x * isize.y - 1u);
    return pixel(i);
  }

  ETX_GPU_CODE float3 evaluate_normal(const float2& uv, float scale) const {
    float4 value = evaluate(uv);
    return {
      scale * (value.x * 2.0f - 1.0f),
      scale * (value.y * 2.0f - 1.0f),
      scale * (value.z * 2.0f - 1.0f) + (1.0f - scale),
    };
  }

  ETX_GPU_CODE float2 sample(const float2& rnd, float& image_pdf, uint2& location) const {
    float y_pdf = 0.0f;
    location.y = y_distribution.sample(rnd.y, y_pdf);

    float x_pdf = 0.0f;
    const auto& x_distribution = x_distributions[location.y];
    location.x = x_distribution.sample(rnd.x, x_pdf);

    auto x0 = x_distribution.values[location.x];
    auto x1 = x_distribution.values[min(location.x + 1llu, x_distribution.values.count - 1)];
    float dx = (rnd.x - x0.cdf);
    if (x1.cdf - x0.cdf > 0.0f) {
      dx /= (x1.cdf - x0.cdf);
    }

    auto y0 = y_distribution.values[location.y];
    auto y1 = y_distribution.values[min(location.y + 1llu, y_distribution.values.count - 1)];
    float dy = (rnd.y - y0.cdf);
    if (y1.cdf - y0.cdf > 0.0f) {
      dy /= (y1.cdf - y0.cdf);
    }

    float2 uv = {
      (float(location.x) + dx) / fsize.x,
      (float(location.y) + dy) / fsize.y,
    };

    image_pdf = pdf(uv);
    return uv;
  }

  ETX_GPU_CODE float tex_coord_repeat(float u, float size) const {
    auto x = fmodf(u, size);
    return x < 0.0f ? (x + size) : x;
  }

  ETX_GPU_CODE float tex_coord_clamp(float u, float size) const {
    return clamp(u, 0.0f, nextafterf(size, 0.0f));
  }

  ETX_GPU_CODE float tex_coord_u(float u, float size) const {
    return (options & RepeatU) ? tex_coord_repeat(u, size) : tex_coord_clamp(u, size);
  }

  ETX_GPU_CODE float tex_coord_v(float u, float size) const {
    return (options & RepeatV) ? tex_coord_repeat(u, size) : tex_coord_clamp(u, size);
  }

  ETX_GPU_CODE float4 read(const float2& uv) const {
    auto x0 = tex_coord_u(uv.x - 0.0f, fsize.x);
    auto x1 = tex_coord_u(uv.x + 1.0f, fsize.x);
    auto y0 = tex_coord_v(uv.y - 0.0f, fsize.y);
    auto y1 = tex_coord_v(uv.y + 1.0f, fsize.y);
    float dx = x0 - floorf(x0);
    float dy = y0 - floorf(y0);
    const auto& p00 = pixel(uint32_t(x0), uint32_t(y0)) * (1.0f - dx) * (1.0f - dy);
    const auto& p01 = pixel(uint32_t(x1), uint32_t(y0)) * (dx) * (1.0f - dy);
    const auto& p10 = pixel(uint32_t(x0), uint32_t(y1)) * (1.0f - dx) * (dy);
    const auto& p11 = pixel(uint32_t(x1), uint32_t(y1)) * (dx) * (dy);
    return p00 + p01 + p10 + p11;
  }
};

}  // namespace etx