#include <etx/core/options.hxx>

#include <jansson.h>

namespace etx {

void Options::save_to_file(const char* filename) {
  json_t* js = json_object();

  for (const auto& val : values) {
    switch (val.cls) {
      case OptionalValue::Class::Boolean: {
        json_object_set_new(js, val.id.c_str(), json_boolean(val.to_bool()));
        break;
      }
      case OptionalValue::Class::Float: {
        json_object_set_new(js, val.id.c_str(), json_real(val.to_float()));
        break;
      }
      case OptionalValue::Class::InfoString: {
        json_object_set_new(js, val.id.c_str(), json_string(val.name.c_str()));
        break;
      }
      case OptionalValue::Class::Integer: {
        json_object_set_new(js, val.id.c_str(), json_integer(val.to_integer()));
        break;
      }
      default:
        break;
    }
  }
  json_dump_file(js, filename, JSON_INDENT(2) | JSON_ESCAPE_SLASH);
  json_decref(js);
}

bool Options::load_from_file(const char* filename) {
  json_error_t err = {};
  auto js = json_load_file(filename, 0, &err);
  if (js == nullptr) {
    return false;
  }

  if (json_is_object(js)) {
    const char* key = {};
    json_t* value = {};
    json_object_foreach(js, key, value) {
      switch (json_typeof(value)) {
        case JSON_STRING: {
          set(key, std::string(json_string_value(value)));
          break;
        }
        case JSON_INTEGER: {
          set(uint32_t(json_integer_value(value)), key, std::string{});
          break;
        }
        case JSON_REAL: {
          set(float(json_real_value(value)), key, std::string{});
          break;
        }
        case JSON_TRUE: {
          set(true, key, std::string{});
          break;
        }
        case JSON_FALSE: {
          set(false, key, std::string{});
          break;
        }
        default:
          break;
      }
    }
  }

  json_decref(js);
  return true;
}

}  // namespace etx
