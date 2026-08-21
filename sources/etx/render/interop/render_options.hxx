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
  float4 viewport;
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
