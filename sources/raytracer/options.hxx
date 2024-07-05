#pragma once

#include <string>

#include <etx/render/host/film.hxx>

namespace etx {

enum class OutputView : uint32_t {
  TonemappedImage,
  AlphaChannel,
  HDRImage,
  ReferenceImage,
  RelativeDifference,
  AbsoluteDifference,

  Count,
};

enum class SaveImageMode : uint32_t {
  RGB,
  TonemappedLDR,
};

inline std::string output_view_to_string(uint32_t i) {
  switch (OutputView(i)) {
    case OutputView::TonemappedImage:
      return "Tonemapped Image";
    case OutputView::HDRImage:
      return "HDR Image";
    case OutputView::ReferenceImage:
      return "Reference Image";
    case OutputView::RelativeDifference:
      return "Relative Difference";
    case OutputView::AbsoluteDifference:
      return "Absolute Difference";
    case OutputView::AlphaChannel:
      return "Alpha Channel";
    default:
      return "???";
  }
}

struct ViewOptions {
  enum : uint32_t {
    ToneMapping = 1u << 0u,
    sRGB = 1u << 1u,
    SkipColorConversion = 1u << 2u,
  };

  OutputView view = OutputView::TonemappedImage;
  uint32_t layer = Film::Result;
  uint32_t options = ToneMapping | sRGB;
  float exposure = 1.0f;
};

}  // namespace etx
