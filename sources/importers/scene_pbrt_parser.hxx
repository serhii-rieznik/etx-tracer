#pragma once

#include <cstdint>
#include <filesystem>
#include <functional>
#include <string>
#include <vector>

namespace etx {

enum class PbrtVersion : uint32_t { V3, V4 };

struct PbrtLocation {
  std::filesystem::path file;
  uint32_t line = 1u;

  std::string describe() const;
};

struct PbrtParameter {
  std::string type;
  std::string name;
  std::vector<double> numbers;
  std::vector<std::string> strings;
};

struct PbrtStatement {
  PbrtLocation location;
  std::string directive;
  std::vector<std::string> arguments;
  std::vector<double> numbers;
  std::vector<PbrtParameter> parameters;

  const PbrtParameter* find(const char* name) const;
};

using PbrtVisitor = std::function<void(const PbrtStatement&)>;

// PBRT resolves includes and asset filenames relative to the root input file.
void visit_pbrt_file(const std::filesystem::path& file, bool expand_includes, const PbrtVisitor& visitor);

}  // namespace etx
