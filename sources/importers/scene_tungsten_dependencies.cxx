#include <etx/std.hxx>
#include <etx/core/core.hxx>
#include <etx/core/environment.hxx>
#include "scene_tungsten_dependencies.hxx"
#include <json.hpp>
#include <fstream>
#include <unordered_set>

namespace etx {
namespace {

using Json = nlohmann::json;

void add_reference(std::vector<std::string>& references, const Json& value) {
  if (value.is_string()) {
    const std::string path = value.get<std::string>();
    if ((path.empty() == false) && (std::string_view(path).starts_with("data:") == false)) {
      references.push_back(path);
    }
  }
}

void inspect_tungsten_material(const Json& material, std::vector<std::string>& references) {
  if (material.is_object() == false) {
    return;
  }
  if (material.contains("albedo"))
    add_reference(references, material["albedo"]);
  if (material.contains("alpha"))
    add_reference(references, material["alpha"]);
  if (material.contains("substrate") && material["substrate"].is_object()) {
    inspect_tungsten_material(material["substrate"], references);
  }
}

void inspect_tungsten_scene(const Json& scene, std::vector<std::string>& references) {
  if (scene.contains("bsdfs") && scene["bsdfs"].is_array()) {
    for (const Json& material : scene["bsdfs"]) {
      inspect_tungsten_material(material, references);
    }
  }
  if ((scene.contains("primitives") == false) || (scene["primitives"].is_array() == false)) {
    return;
  }
  for (const Json& primitive : scene["primitives"]) {
    if (primitive.is_object() == false) {
      continue;
    }
    const std::string type = primitive.value("type", std::string{});
    if (type == "mesh") {
      if (primitive.contains("filename"))
        add_reference(references, primitive["filename"]);
      else if (primitive.contains("file"))
        add_reference(references, primitive["file"]);
    }
    if (type == "infinite_sphere") {
      if (primitive.value("sample", true) && primitive.contains("emission")) {
        add_reference(references, primitive["emission"]);
      }
    } else {
      if (primitive.contains("emission"))
        add_reference(references, primitive["emission"]);
      if (primitive.contains("power"))
        add_reference(references, primitive["power"]);
    }
    if (primitive.contains("bsdf") && primitive["bsdf"].is_object()) {
      inspect_tungsten_material(primitive["bsdf"], references);
    }
  }
}

}  // namespace

bool is_tungsten_document(const nlohmann::json& document) {
  return document.is_object() && (document.contains("etx_document") == false) && (document.contains("asset") == false) && (document.contains("geometry") == false) &&
         document.contains("primitives") && document["primitives"].is_array() && ((document.contains("bsdfs") == false) || document["bsdfs"].is_array());
}

SceneDependencyInspection inspect_tungsten_dependencies(const std::filesystem::path& file_path, std::string_view relative_path) {
  if (_stricmp(path_to_utf8(file_path.extension()).c_str(), ".json") != 0)
    return inspect_scene_dependencies(file_path, relative_path);
  SceneDependencyInspection result;
  std::ifstream stream(file_path, std::ios::binary);
  if (stream.is_open() == false) {
    result.error = "Failed to open dependency descriptor";
    return result;
  }
  std::error_code size_error;
  const uint64_t file_size = std::filesystem::file_size(file_path, size_error);
  if (size_error) {
    result.error = "Failed to determine dependency descriptor size";
    return result;
  }
  constexpr uint64_t kMaxDependencyDocumentSize = 256ull * 1024ull * 1024ull;
  if (file_size > kMaxDependencyDocumentSize) {
    result.error = "Dependency document is too large to inspect";
    return result;
  }
  const Json json = Json::parse(stream, nullptr, false);
  if (json.is_discarded() || (json.is_object() == false)) {
    result.error = "Failed to parse scene JSON";
    return result;
  }
  if (is_tungsten_document(json)) {
    try {
      inspect_tungsten_scene(json, result.references);
    } catch (const Json::exception&) {
      result.error = "Scene JSON fields have invalid types";
    }
    std::unordered_set<std::string> unique;
    result.references.erase(std::remove_if(result.references.begin(), result.references.end(),
                              [&](const std::string& path) {
                                std::string key = path;
                                std::transform(key.begin(), key.end(), key.begin(), [](unsigned char character) {
                                  return static_cast<char>(std::tolower(character));
                                });
                                return path.empty() || (unique.insert(key).second == false);
                              }),
      result.references.end());
    return result;
  }
  return inspect_scene_dependencies(file_path, relative_path);
}

}  // namespace etx
