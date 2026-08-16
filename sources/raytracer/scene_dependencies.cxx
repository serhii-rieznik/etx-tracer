#include "scene_dependencies.hxx"

#include <json.hpp>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <unordered_set>

namespace etx {
namespace {

using Json = nlohmann::json;

constexpr uint64_t kMaxDependencyDocumentSize = 256ull * 1024ull * 1024ull;

std::string lowercase(std::string_view value) {
  std::string result(value);
  std::transform(result.begin(), result.end(), result.begin(), [](unsigned char character) {
    return static_cast<char>(std::tolower(character));
  });
  return result;
}

std::string extension(std::string_view path) {
  const size_t slash = path.find_last_of("/\\");
  const size_t dot = path.find_last_of('.');
  return ((dot == std::string_view::npos) || ((slash != std::string_view::npos) && (dot < slash))) ? std::string{} : lowercase(path.substr(dot));
}

void add_reference(std::vector<std::string>& references, const Json& value) {
  if (value.is_string()) {
    const std::string path = value.get<std::string>();
    if (!path.empty() && !std::string_view(path).starts_with("data:")) {
      references.push_back(path);
    }
  }
}

void inspect_tungsten_material(const Json& material, std::vector<std::string>& references) {
  if (!material.is_object()) {
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
  if (!scene.contains("primitives") || !scene["primitives"].is_array()) {
    return;
  }
  for (const Json& primitive : scene["primitives"]) {
    if (!primitive.is_object()) {
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

int hexadecimal_digit(char value) {
  if ((value >= '0') && (value <= '9'))
    return value - '0';
  if ((value >= 'a') && (value <= 'f'))
    return value - 'a' + 10;
  if ((value >= 'A') && (value <= 'F'))
    return value - 'A' + 10;
  return -1;
}

std::string decode_uri_path(std::string_view uri) {
  std::string result = {};
  result.reserve(uri.size());
  for (size_t index = 0u; index < uri.size(); ++index) {
    if ((uri[index] == '%') && (index + 2u < uri.size())) {
      const int high = hexadecimal_digit(uri[index + 1u]);
      const int low = hexadecimal_digit(uri[index + 2u]);
      if ((high >= 0) && (low >= 0)) {
        result.push_back(static_cast<char>((high << 4) | low));
        index += 2u;
        continue;
      }
    }
    result.push_back(uri[index]);
  }
  return result;
}

bool inspect_gltf(const Json& gltf, std::vector<std::string>& references, std::string& error) {
  if (!gltf.is_object()) {
    error = "glTF document is not an object";
    return false;
  }
  const auto inspect_uri_array = [&](const char* name) {
    if (!gltf.contains(name) || !gltf[name].is_array())
      return;
    for (const Json& item : gltf[name]) {
      if (item.is_object() && item.contains("uri") && item["uri"].is_string()) {
        const std::string uri = item["uri"].get<std::string>();
        if (!std::string_view(uri).starts_with("data:"))
          references.push_back(decode_uri_path(uri));
      }
    }
  };
  inspect_uri_array("buffers");
  inspect_uri_array("images");
  return true;
}

bool inspect_json(std::string_view contents, std::vector<std::string>& references, std::vector<std::string>& geometry_with_external_materials, std::string& error) {
  const Json json = Json::parse(contents, nullptr, false);
  if (json.is_discarded() || !json.is_object()) {
    error = "Failed to parse scene JSON";
    return false;
  }
  try {
    if (json.contains("asset") && json["asset"].is_object()) {
      return inspect_gltf(json, references, error);
    }
    const bool tungsten = json.contains("bsdfs") && (json.contains("primitives") || json.contains("renderer"));
    if (tungsten) {
      inspect_tungsten_scene(json, references);
    } else {
      if (json.contains("geometry"))
        add_reference(references, json["geometry"]);
      if (json.contains("materials"))
        add_reference(references, json["materials"]);
      if (json.contains("geometry") && json["geometry"].is_string() && json.contains("materials") && json["materials"].is_string()) {
        geometry_with_external_materials.push_back(json["geometry"].get<std::string>());
      }
    }
  } catch (const Json::exception&) {
    error = "Scene JSON fields have invalid types";
    return false;
  }
  return true;
}

std::vector<std::string> split_words(std::string value) {
  std::istringstream stream(std::move(value));
  std::vector<std::string> words = {};
  for (std::string word = {}; stream >> word;)
    words.push_back(std::move(word));
  return words;
}

void inspect_material_line(const std::string& key, const std::string& value, std::vector<std::string>& references) {
  static const std::unordered_set<std::string> direct_paths = {"shape", "volume", "image", "map_Ke", "map_Kd", "map_Ks", "map_Kt"};
  static const std::unordered_set<std::string> first_word_paths = {"map_Pr", "map_Ml", "map_Tm"};
  if (direct_paths.contains(key)) {
    if (!value.empty())
      references.push_back(value);
    return;
  }
  const std::vector<std::string> words = split_words(value);
  if (first_word_paths.contains(key) && !words.empty()) {
    references.push_back(words.front());
  }
  if ((key == "normalmap") || (key == "thinfilm") || (key == "emitter")) {
    for (size_t index = 0u; index + 1u < words.size(); ++index) {
      if (words[index] == "image")
        references.push_back(words[index + 1u]);
      if ((key == "emitter") && (words[index] == "spectrum"))
        references.push_back(words[index + 1u]);
    }
  }
  if ((key == "int_ior") || (key == "ext_ior")) {
    if (!words.empty() && ((extension(words.front()) == ".spd") || (words.front().find('/') != std::string::npos))) {
      references.push_back(words.front());
    }
  }
}

void inspect_materials(std::istream& stream, std::vector<std::string>& references) {
  for (std::string line = {}; std::getline(stream, line);) {
    const size_t first = line.find_first_not_of(" \t\r");
    if ((first == std::string::npos) || (line[first] == '#'))
      continue;
    const size_t separator = line.find_first_of(" \t", first);
    if (separator == std::string::npos)
      continue;
    const size_t value_begin = line.find_first_not_of(" \t", separator);
    if (value_begin == std::string::npos)
      continue;
    size_t value_end = line.find_last_not_of(" \t\r");
    inspect_material_line(line.substr(first, separator - first), line.substr(value_begin, value_end - value_begin + 1u), references);
  }
}

void inspect_obj(std::istream& stream, std::vector<std::string>& references) {
  for (std::string line = {}; std::getline(stream, line);) {
    const size_t first = line.find_first_not_of(" \t\r");
    if ((first == std::string::npos) || (line.compare(first, 6u, "mtllib") != 0))
      continue;
    const size_t value_begin = line.find_first_not_of(" \t", first + 6u);
    if (value_begin != std::string::npos) {
      const std::vector<std::string> words = split_words(line.substr(value_begin));
      if (!words.empty())
        references.push_back(words.front());
    }
    break;
  }
}

bool read_stream(std::istream& stream, std::string& contents) {
  std::ostringstream buffer = {};
  buffer << stream.rdbuf();
  if (stream.bad()) {
    return false;
  }
  contents = std::move(buffer).str();
  return true;
}

bool inspect_glb_file(std::ifstream& stream, uint64_t file_size, std::vector<std::string>& references, std::string& error) {
  constexpr uint32_t kGLBMagic = 0x46546c67u;
  constexpr uint32_t kGLBVersion = 2u;
  constexpr uint32_t kJSONChunk = 0x4e4f534au;
  uint32_t header[3] = {};
  stream.read(reinterpret_cast<char*>(header), sizeof(header));
  if (!stream || (header[0] != kGLBMagic) || (header[1] != kGLBVersion) || (header[2] != file_size)) {
    error = "GLB file has an invalid header";
    return false;
  }
  uint64_t offset = sizeof(header);
  while (offset < file_size) {
    if ((file_size - offset) < (2u * sizeof(uint32_t))) {
      error = "GLB file has a truncated chunk header";
      return false;
    }
    uint32_t chunk[2] = {};
    stream.read(reinterpret_cast<char*>(chunk), sizeof(chunk));
    if (!stream) {
      error = "Failed to read a GLB chunk header";
      return false;
    }
    offset += sizeof(chunk);
    if ((chunk[0] % 4u) != 0u || (static_cast<uint64_t>(chunk[0]) > (file_size - offset))) {
      error = "GLB chunk has an invalid size";
      return false;
    }
    if (chunk[1] == kJSONChunk) {
      if (chunk[0] > kMaxDependencyDocumentSize) {
        error = "GLB JSON chunk is too large to inspect";
        return false;
      }
      std::string json_contents(chunk[0], '\0');
      stream.read(json_contents.data(), static_cast<std::streamsize>(json_contents.size()));
      if (!stream) {
        error = "Failed to read the GLB JSON chunk";
        return false;
      }
      while (!json_contents.empty() && ((json_contents.back() == '\0') || std::isspace(static_cast<unsigned char>(json_contents.back())))) {
        json_contents.pop_back();
      }
      const Json json = Json::parse(json_contents, nullptr, false);
      if (json.is_discarded()) {
        error = "Failed to parse the GLB JSON chunk";
        return false;
      }
      return inspect_gltf(json, references, error);
    }
    stream.seekg(static_cast<std::streamoff>(chunk[0]), std::ios::cur);
    if (!stream) {
      error = "Failed to skip a GLB chunk";
      return false;
    }
    offset += chunk[0];
  }
  error = "GLB file does not contain a JSON chunk";
  return false;
}

}  // namespace

SceneDependencyInspection inspect_scene_dependencies(const std::filesystem::path& file_path, std::string_view relative_path) {
  SceneDependencyInspection result = {};
  const std::string ext = extension(relative_path);
  bool valid = true;
  std::ifstream stream(file_path, std::ios::binary);
  if (!stream) {
    result.error = "Failed to open dependency descriptor";
    return result;
  }
  std::error_code size_error = {};
  const uint64_t file_size = std::filesystem::file_size(file_path, size_error);
  if (size_error) {
    result.error = "Failed to determine dependency descriptor size";
    return result;
  }
  if (ext == ".obj") {
    inspect_obj(stream, result.references);
  } else if ((ext == ".mtl") || (ext == ".materials")) {
    inspect_materials(stream, result.references);
  } else if (ext == ".glb") {
    valid = inspect_glb_file(stream, file_size, result.references, result.error);
  } else if ((ext == ".json") || (ext == ".gltf")) {
    if (file_size > kMaxDependencyDocumentSize) {
      result.error = "Dependency document is too large to inspect";
      return result;
    }
    std::string contents = {};
    if (!read_stream(stream, contents)) {
      result.error = "Failed to read dependency descriptor";
      return result;
    }
    valid = inspect_json(contents, result.references, result.geometry_with_external_materials, result.error);
  }
  if (!valid)
    return result;

  std::unordered_set<std::string> unique = {};
  result.references.erase(std::remove_if(result.references.begin(), result.references.end(),
                            [&](const std::string& path) {
                              return path.empty() || !unique.insert(lowercase(path)).second;
                            }),
    result.references.end());
  return result;
}

}  // namespace etx
