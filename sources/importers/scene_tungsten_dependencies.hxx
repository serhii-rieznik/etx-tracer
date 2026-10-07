#pragma once

#include <etx/import/scene_dependencies.hxx>
#include <json.hpp>

namespace etx {

bool is_tungsten_document(const nlohmann::json& document);
SceneDependencyInspection inspect_tungsten_dependencies(const std::filesystem::path& file_path, std::string_view relative_path);

}  // namespace etx
