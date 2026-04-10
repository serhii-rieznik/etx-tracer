#pragma once

#include <cstdint>

namespace etx {

enum class BDPTMode : uint32_t {
  PathTracing = 0u,
  LightTracing = 1u,
  BDPTFast = 2u,
  BDPTFull = 3u,

  Count = 4u,
  Invalid = 0xffffffffu,
};

inline bool bdpt_mode_valid(BDPTMode mode) {
  return static_cast<uint32_t>(mode) < static_cast<uint32_t>(BDPTMode::Count);
}

inline const char* bdpt_mode_name(BDPTMode mode) {
  switch (mode) {
    case BDPTMode::PathTracing:
      return "PathTracing";
    case BDPTMode::LightTracing:
      return "LightTracing";
    case BDPTMode::BDPTFast:
      return "BDPTFast";
    case BDPTMode::BDPTFull:
      return "BDPTFull";
    default:
      return "Unknown";
  }
}

inline const char* bdpt_mode_display_name(BDPTMode mode) {
  switch (mode) {
    case BDPTMode::PathTracing:
      return "Path Tracing";
    case BDPTMode::LightTracing:
      return "Light Tracing";
    case BDPTMode::BDPTFast:
      return "BDPT Fast";
    case BDPTMode::BDPTFull:
      return "BDPT Full";
    default:
      return "Unknown";
  }
}

}  // namespace etx
