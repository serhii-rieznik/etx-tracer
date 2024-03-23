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

enum class SaveImageMode : uint32_t {
  RGB,
  TonemappedLDR,
  XYZ,
};

inline std::string output_view_to_string(uint32_t i) {
  switch (OutputView(i)) {
    case OutputView::Result:
      return "[1] Result Image ";
    case OutputView::CameraImage:
      return "[2] Camera Image ";
    case OutputView::LightImage:
      return "[3] Light Image ";
    case OutputView::ReferenceImage:
      return "[4] Reference Image ";
    case OutputView::RelativeDifference:
      return "[5] Relative Difference ";
    case OutputView::AbsoluteDifference:
      return "[6] Absolute Difference ";
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
  float exposure = 1.0f;
};

}  // namespace etx
