#pragma once

#include <string>

#include <etx/render/host/film.hxx>

#include <shaders/shared/render_options.hxx>

namespace etx {

enum class SaveImageMode : uint32_t {
  RGB,
  TonemappedLDR,
};

inline std::string view_option_to_string(uint32_t i) {
  switch (i) {
    case ViewOptions::Tonemapped:
      return "Tonemapped";
    case ViewOptions::HDR:
      return "HDR";
    case ViewOptions::Normalized:
      return "Normalized";
    default:
      return "???";
  }
}

inline std::string output_view_to_string(uint32_t i) {
  switch (i) {
    case OutputView::OutputImage:
      return "Output Image";
    case OutputView::AlphaChannel:
      return "Alpha Channel";
    case OutputView::ReferenceImage:
      return "Reference Image";
    case OutputView::RelativeDifference:
      return "Relative Difference";
    case OutputView::AbsoluteDifference:
      return "Absolute Difference";
    default:
      return "???";
  }
}

}  // namespace etx
