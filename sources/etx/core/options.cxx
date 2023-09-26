#include <etx/core/options.hxx>
#include <etx/core/json.hxx>

#include <json.hpp>

namespace etx {

void Options::save_to_file(const char* filename) {
  nlohmann::json js;
  for (const auto& val : values) {
    switch (val.cls) {
      case OptionalValue::Class::Boolean: {
        js[val.id] = val.to_bool();
        break;
      }
      case OptionalValue::Class::Float: {
        js[val.id] = val.to_float();
        break;
      }
      case OptionalValue::Class::Integer: {
        js[val.id] = val.to_integer();
        break;
      }
      case OptionalValue::Class::InfoString: {
        js[val.id] = val.name;
        break;
      }
      default:
        break;
    }
  }
  json_to_file(js, filename);
}

bool Options::load_from_file(const char* filename) {
  auto js = json_from_file(filename);

  if (js.is_object() == false) {
    log::error("Invalid options file: %s", filename);
    return false;
  }

  for (auto i = js.begin(), e = js.end(); i != e; ++i) {
    const auto& key = i.key();
    const auto& obj = i.value();
    if (obj.is_string()) {
      add(key, obj.get<std::string>());
    } else if (obj.is_boolean()) {
      add(obj.get<bool>(), key, std::string{});
    } else if (obj.is_number_float()) {
      add(obj.get<float>(), key, std::string{});
    } else if (obj.is_number_integer()) {
      add(obj.get<uint32_t>(), key, std::string{});
    } else {
      log::warning("Unhandled value in options : %s", key.c_str());
    }
  }
  return true;
}

}  // namespace etx
