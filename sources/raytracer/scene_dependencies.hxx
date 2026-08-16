#pragma once

#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace etx {

struct SceneDependencyInspection {
  std::vector<std::string> references = {};
  std::vector<std::string> geometry_with_external_materials = {};
  std::string error = {};
};

SceneDependencyInspection inspect_scene_dependencies(const std::filesystem::path& file_path, std::string_view relative_path);

}  // namespace etx
