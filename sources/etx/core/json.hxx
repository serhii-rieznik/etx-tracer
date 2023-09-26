#pragma once

#include <json.hpp>

namespace etx {

inline nlohmann::json json_from_file(const char* filename) {
  auto fin = fopen(filename, "rb");
  if (fin == nullptr) {
    return {};
  }

  fseek(fin, 0, SEEK_END);
  auto file_size = ftell(fin);
  fseek(fin, 0, SEEK_SET);

  std::string json_string(file_size + 1, 0);
  fread(json_string.data(), 1, file_size, fin);
  fclose(fin);

  nlohmann::json result = {};
  try {
    auto parsed = nlohmann::json::parse(json_string);
    result = parsed;
  } catch (...) {
  }

  return result;
}

inline void json_to_file(const nlohmann::json& js, const char* filename) {
  auto fout = fopen(filename, "wb");
  if (fout == nullptr) {
    return;
  }
  std::string json_string = js.dump(2);
  fwrite(json_string.data(), 1, json_string.size(), fout);
  fflush(fout);
  fclose(fout);
}

inline bool json_get_string(const nlohmann::json::const_iterator& js, const std::string& id, std::string& value) {
  if ((js.key() == id) && js.value().is_string()) {
    value = js.value().get<std::string>();
    return true;
  }
  return false;
}

inline bool json_get_float(const nlohmann::json::const_iterator& js, const std::string& id, float& value) {
  if ((js.key() == id) && (js.value().is_number_float() || js.value().is_number_integer())) {
    value = js.value().get<float>();
    return true;
  }
  return false;
}

}  // namespace etx
