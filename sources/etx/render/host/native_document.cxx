#include <etx/render/host/native_document.hxx>
#include <etx/core/core.hxx>
#include <json.hpp>
#include <fstream>

namespace etx {

bool validate_native_document(const std::filesystem::path& path, std::string& error) {
  std::string extension = path_to_utf8(path.extension());
  std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  if (extension == ".etx") {
    return true;
  }
  if (extension != ".json") {
    error = "Open requires a native ETX document. Use Import for this format.";
    return false;
  }
  std::ifstream input(path, std::ios::binary);
  const auto document = nlohmann::json::parse(input, nullptr, false);
  if (document.is_discarded() || (document.is_object() == false) || document.contains("bsdfs") || document.contains("asset")) {
    error = "The file is not a native ETX document. Use Import for foreign scenes.";
    return false;
  }
  if (document.contains("etx_document")) {
    const auto& identity = document["etx_document"];
    if ((identity.is_object() == false) || (identity.contains("format") == false) || (identity["format"].is_string() == false) || (identity["format"] != "etx") ||
        (identity.contains("version") == false) || (identity["version"].is_number_unsigned() == false) || (identity["version"] != 1u)) {
      error = "Unsupported native ETX document version.";
      return false;
    }
  }
  if (document.contains("geometry")) {
    if (document["geometry"].is_string() == false) {
      error = "Invalid native geometry reference.";
      return false;
    }
    const auto geometry = std::filesystem::u8path(document["geometry"].get<std::string>());
    std::string geometry_extension = path_to_utf8(geometry.extension());
    std::transform(geometry_extension.begin(), geometry_extension.end(), geometry_extension.begin(), [](unsigned char c) {
      return static_cast<char>(std::tolower(c));
    });
    if (geometry_extension != ".etx") {
      error = "This scene references foreign geometry. Import it to produce a native ETX document.";
      return false;
    }
  } else if ((document.contains("materials") == false) || (document["materials"].is_string() == false)) {
    error = "Native ETX documents must reference ETX geometry or native materials.";
    return false;
  }
  return true;
}

}  // namespace etx
