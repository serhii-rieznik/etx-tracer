#pragma once

#include "interop.hxx"

struct ViewLayer {
  enum : uint32_t {
    Result,
    Denoised,
    CurrentFrame,
    Accumulation,
    AdaptiveAccumulation,
    Albedo,
    Normals,
    Debug,

    Count,
  };
};

struct OutputView {
  enum : uint32_t {
    OutputImage,
    AlphaChannel,
    ReferenceImage,
    RelativeDifference,
    AbsoluteDifference,

    Count,
  };
};

struct ViewOptions {
  enum : uint32_t {
    Tonemapped,
    HDR,
    Normalized,

    Count,
  };
};

struct ETX_ALIGNED ViewParameters {
  float exposure;
  uint32_t view_option;
  uint32_t view_image;
  uint32_t view_layer;
};

struct ETX_ALIGNED RenderParameters {
  ViewParameters view;
  float4 dimensions;
  uint32_t sample_count;
  uint32_t sample_image_index;
  uint32_t reference_image_index;
  uint32_t pad;
};

struct ShaderConstants {
  float4 dimensions;
  float exposure;
  uint32_t image_view;
  uint32_t options;
  uint32_t sample_count;
};

#if defined(__cplusplus)
static_assert(std::is_standard_layout_v<ViewParameters>, "ViewParameters must stay standard layout for C++/HLSL interop");
static_assert(std::is_standard_layout_v<RenderParameters>, "RenderParameters must stay standard layout for C++/HLSL interop");
static_assert(sizeof(ViewParameters) == 16, "ViewParameters size changed; update shared ABI or padding");
static_assert(sizeof(RenderParameters) == 48, "RenderParameters size changed; update shared ABI or padding");
static_assert(offsetof(RenderParameters, dimensions) == 16, "RenderParameters::dimensions offset changed");
static_assert(offsetof(RenderParameters, sample_count) == 32, "RenderParameters::sample_count offset changed");
static_assert(sizeof(ShaderConstants) == 32, "ShaderConstants size changed; update shared ABI or padding");
#endif
