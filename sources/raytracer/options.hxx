#pragma once

#include <string>

namespace etx {

enum class OutputView : uint32_t {
  Result,
  CameraImage,
  LightImage,
  ReferenceImage,
  RelativeDifference,
  AbsoluteDifference,
  Count,
};

inline std::string output_view_to_string(uint32_t i) {
  switch (OutputView(i)) {
    case OutputView::Result:
      return "Result Image";
    case OutputView::CameraImage:
      return "Camera Image";
    case OutputView::LightImage:
      return "Light Image";
    case OutputView::ReferenceImage:
      return "Reference Image";
    case OutputView::RelativeDifference:
      return "Relative Difference";
    case OutputView::AbsoluteDifference:
      return "Absolute Differenec";
    default:
      return "???";
  }
}

struct ViewOptions {
  enum : uint32_t {
    ToneMapping = 1u << 0u,
    sRGB = 1u << 1u,
  };

  OutputView view = OutputView::Result;
  uint32_t options = ToneMapping | sRGB;
  float exposure = 0.0f;
};

}  // namespace etx
