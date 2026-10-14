#pragma once

#include <etx/core/log.hxx>
#include <etx/rhi/rhi_types.hxx>
#include <cstdio>
#include <string>

namespace etx {
template <class... Args>
std::string format_diagnostic(const char* format, Args... args) {
  const int length = std::snprintf(nullptr, 0, format, args...);
  std::string message;
  if (length >= 0) {
    message.resize(static_cast<size_t>(length) + 1u);
    std::snprintf(message.data(), message.size(), format, args...);
    message.resize(static_cast<size_t>(length));
  }
  return message;
}

enum class SceneDiagnosticKind { None, Validation, Unsupported, Resource, File };
enum class SceneDiagnosticTarget { None, Material, Medium, Camera };

struct SceneDiagnostic {
  SceneDiagnosticKind kind = SceneDiagnosticKind::None;
  SceneDiagnosticTarget target = SceneDiagnosticTarget::None;
  uint32_t resource_index = kInvalidIndex;
  std::string resource_name;
  std::string phase;
  std::string field;
  std::string message;
  RHIResult result = RHIResult::Success;

  void set_result(RHIResult value) {
    if (message.empty() || ((value == RHIResult::DeviceLost) && (result != RHIResult::DeviceLost))) {
      if (value == RHIResult::DeviceLost)
        message.clear();
      result = value;
    }
  }

  bool rejected() const {
    return (kind == SceneDiagnosticKind::Validation) || (kind == SceneDiagnosticKind::Unsupported);
  }

  std::string text() const {
    std::string value = phase;
    if (resource_name.empty() == false) {
      if (value.empty() == false)
        value += " / ";
      value += resource_name;
    }
    if (field.empty() == false) {
      if (value.empty() == false)
        value += " / ";
      value += field;
    }
    if (value.empty() == false)
      value += ":\n";
    value += message;
    return value;
  }

  template <class... Args>
  bool fail(SceneDiagnosticKind failure_kind, const char* failure_field, const char* format, Args... args) {
    if (message.empty() == false)
      return false;
    kind = failure_kind;
    field = failure_field;
    message = format_diagnostic(format, args...);
    log::error("%s", text().c_str());
    return false;
  }
};
}  // namespace etx
