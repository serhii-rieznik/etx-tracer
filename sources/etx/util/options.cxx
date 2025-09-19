#include <etx/util/options.hxx>
#include <etx/core/json.hxx>
#include <etx/core/log.hxx>
#include <json.hpp>

namespace etx {

void Option::serialize(void* wrap) const {
  nlohmann::json& js = *reinterpret_cast<nlohmann::json*>(wrap);

  js["class"] = cls;
  js["meta"] = meta;
  js["id"] = id;
  js["description"] = description;
  switch (cls) {
    case Class::Boolean: {
      js["value"] = as<Class::Boolean>().value;
      break;
    }
    case Class::Integral: {
      const auto& data = as<Class::Integral>();
      js["value"] = data.value;
      js["value.min"] = data.bounds.minimum;
      js["value.max"] = data.bounds.minimum;
      break;
    }
    case Class::Float: {
      const auto& data = as<Class::Float>();
      js["value"] = data.value;
      js["value.min"] = data.bounds.minimum;
      js["value.max"] = data.bounds.minimum;
      break;
    }
    case Class::Float3: {
      const auto& data = as<Class::Float3>();
      js["value.x"] = data.value.x;
      js["value.y"] = data.value.y;
      js["value.z"] = data.value.z;
      js["value.x.min"] = data.bounds.minimum.x;
      js["value.y.min"] = data.bounds.minimum.y;
      js["value.z.min"] = data.bounds.minimum.z;
      js["value.x.max"] = data.bounds.maximum.x;
      js["value.y.max"] = data.bounds.maximum.y;
      js["value.z.max"] = data.bounds.maximum.z;
      break;
    }
    case Class::String: {
      js["value"] = as<Class::String>().value;
      break;
    }
    default:
      break;
  }
}

void Option::deserialize(const void* wrap) {
  // clang-format off
#define ETX_OPT_GETN(name, val) if (js.contains(name)) val = js[name]
#define ETX_OPT_GET(val) ETX_OPT_GETN(#val, val)
  // clang-format on

  const nlohmann::json& js = *reinterpret_cast<const nlohmann::json*>(wrap);
  if (js.is_object() == false)
    return;

  if (js.contains("class") == false)
    return;

  cls = js.at("class").get<Class>();
  if ((cls == Class::Undefined) || (cls == Class::Any))
    return;

  ETX_OPT_GET(id);
  ETX_OPT_GET(description);
  ETX_OPT_GET(meta);

  switch (cls) {
    case Class::Boolean: {
      auto& data = init<Class::Boolean>();
      ETX_OPT_GETN("value", data.value);
      break;
    }
    case Class::Integral: {
      auto& data = init<Class::Integral>();
      ETX_OPT_GETN("value", data.value);
      ETX_OPT_GETN("value.min", data.bounds.minimum);
      ETX_OPT_GETN("value.max", data.bounds.minimum);
      break;
    }
    case Class::Float: {
      auto& data = init<Class::Float>();
      ETX_OPT_GETN("value", data.value);
      ETX_OPT_GETN("value.min", data.bounds.minimum);
      ETX_OPT_GETN("value.max", data.bounds.minimum);
      break;
    }
    case Class::Float3: {
      auto& data = init<Class::Float3>();
      ETX_OPT_GETN("value.x", data.value.x);
      ETX_OPT_GETN("value.y", data.value.y);
      ETX_OPT_GETN("value.z", data.value.z);
      ETX_OPT_GETN("value.x.min", data.bounds.minimum.x);
      ETX_OPT_GETN("value.y.min", data.bounds.minimum.y);
      ETX_OPT_GETN("value.z.min", data.bounds.minimum.z);
      ETX_OPT_GETN("value.x.max", data.bounds.maximum.x);
      ETX_OPT_GETN("value.y.max", data.bounds.maximum.y);
      ETX_OPT_GETN("value.z.max", data.bounds.maximum.z);
      break;
    }
    case Class::String: {
      auto& data = init<Class::String>();
      ETX_OPT_GETN("value", data.value);
      break;
    }
    default:
      break;
  }

#undef ETX_OPT_GET
#undef ETX_OPT_GETN
}

//*
void Options::save_to_file(const std::string& filename) {
  nlohmann::json js;
  nlohmann::json values = nlohmann::json::array();
  for (const auto& option : options) {
    nlohmann::json option_serialized;
    option.serialize(static_cast<void*>(&option_serialized));
    values.emplace_back(option_serialized);
  }
  js["values"] = values;
  json_to_file(js, filename.c_str());
}

bool Options::load_from_file(const std::string& filename) {
  auto js = json_from_file(filename.c_str());

  if (js.is_object() == false) {
    log::error("Invalid options file: %s", filename.c_str());
    return false;
  }

  if (js.contains("values") == false) {
    return true;
  }

  const auto& values = js.at("values");
  if (values.is_array() == false) {
    log::error("Invalid options file - values are incorrect: %s", filename.c_str());
    return false;
  }

  for (const auto& value : values) {
    auto& option = options.emplace_back();
    option.deserialize(static_cast<const void*>(&value));
  }

  return true;
}
// */

}  // namespace etx
