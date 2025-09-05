#pragma once

#include <etx/core/debug.hxx>
#include <etx/render/shared/base.hxx>

#include <vector>
#include <string>

namespace etx {

struct OptionalValue {
  enum class Class : uint32_t {
    Undefined,
    Integer,
    Boolean,
    InfoString,
    Float,
    Enum,
    Float3,

    Count,
  };

  union Value {
    float3 f3;
    uint32_t integer;
    float flt;
    bool boolean;
  };

  Class cls = Class::Integer;
  std::string id = {};
  std::string name = {};
  Value min_value = {};
  Value value = {};
  Value max_value = {};
  std::string (*name_func)(uint32_t) = {};

  OptionalValue() = default;

  OptionalValue(uint32_t val, const std::string& a_id, const std::string& desc)
    : cls(Class::Integer)
    , id(a_id)
    , name(desc) {
    min_value.integer = 0;
    value.integer = val;
    max_value.integer = std::numeric_limits<uint32_t>::max();
  }

  OptionalValue(uint32_t min_val, uint32_t val, uint32_t max_val, const std::string& a_id, const std::string& desc)
    : cls(Class::Integer)
    , id(a_id)
    , name(desc) {
    min_value.integer = min_val;
    value.integer = val;
    max_value.integer = max_val;
  }

  OptionalValue(float val, const std::string& a_id, const std::string& desc)
    : cls(Class::Float)
    , id(a_id)
    , name(desc) {
    min_value.flt = -std::numeric_limits<float>::max();
    value.flt = val;
    max_value.flt = std::numeric_limits<float>::max();
  }

  OptionalValue(const float3& val, const std::string& a_id, const std::string& desc)
    : cls(Class::Float3)
    , id(a_id)
    , name(desc) {
    min_value.f3 = {-std::numeric_limits<float>::max(), -std::numeric_limits<float>::max(), -std::numeric_limits<float>::max()};
    value.f3 = val;
    max_value.f3 = {std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max()};
  }

  OptionalValue(float min_val, float val, float max_val, const std::string& a_id, const std::string& desc)
    : cls(Class::Float)
    , id(a_id)
    , name(desc) {
    min_value.flt = min_val;
    value.flt = val;
    max_value.flt = max_val;
  }

  OptionalValue(const float3& min_val, const float3& val, const float3& max_val, const std::string& a_id, const std::string& desc)
    : cls(Class::Float3)
    , id(a_id)
    , name(desc) {
    min_value.f3 = min_val;
    value.f3 = val;
    max_value.f3 = max_val;
  }

  OptionalValue(bool val, const std::string& a_id, const std::string& desc)
    : cls(Class::Boolean)
    , id(a_id)
    , name(desc) {
    value.boolean = val;
  }

  OptionalValue(const std::string& a_id, const std::string& desc)
    : cls(Class::InfoString)
    , id(a_id)
    , name(desc) {
  }

  template <class T>
  OptionalValue(T v, T end, std::string (*f)(uint32_t), const std::string& a_id, const std::string& desc)
    : cls(Class::Enum)
    , id(a_id)
    , name(desc)
    , name_func(f) {
    min_value.integer = 0;
    value.integer = uint32_t(v);
    max_value.integer = uint32_t(end) - 1u;
  }

  void set(uint32_t new_integer_value) {
    ETX_ASSERT((cls == Class::Integer) || (cls == Class::Enum));
    value.integer = clamp_to(new_integer_value, min_value.integer, max_value.integer);
  }

  void set(bool new_bool_value) {
    ETX_ASSERT(cls == Class::Boolean);
    value.boolean = new_bool_value;
  }

  void set(float new_float_value) {
    ETX_ASSERT(cls == Class::Float);
    value.flt = new_float_value;
  }

  void set(const float3& new_float3_value) {
    ETX_ASSERT(cls == Class::Float3);
    value.f3 = new_float3_value;
  }

  uint32_t to_integer() const {
    ETX_ASSERT((cls == Class::Integer) || (cls == Class::Enum) || (cls == Class::Boolean));
    return (cls == Class::Boolean) ? value.boolean : clamp_to(value.integer, min_value.integer, max_value.integer);
  }

  float to_float() const {
    ETX_ASSERT(cls == Class::Float);
    return clamp_to(value.flt, min_value.flt, max_value.flt);
  }

  bool to_bool() const {
    ETX_ASSERT(cls == Class::Boolean);
    return value.boolean;
  }

  template <class T>
  T to_enum() {
    ETX_ASSERT((cls == Class::Enum) || (cls == Class::Integer));
    return T(value.integer);
  }

  float3 to_float3() const {
    ETX_ASSERT(cls == Class::Float3);
    return {
      clamp_to(value.f3.x, min_value.f3.x, max_value.f3.x),
      clamp_to(value.f3.y, min_value.f3.y, max_value.f3.y),
      clamp_to(value.f3.z, min_value.f3.z, max_value.f3.z),
    };
  }

 private:
  template <class T>
  inline static T clamp_to(T value, T min, T max) {
    return value < min ? min : (value > max ? max : value);
  }
};

struct Options {
  std::vector<OptionalValue> values;

  template <class T>
  OptionalValue get(const std::string& id, T def) const {
    for (const auto& option : values) {
      if (option.id == id) {
        return option;
      }
    }
    return {def, id, {}};
  }

  template <>
  OptionalValue get<std::string>(const std::string& id, std::string def) const {
    for (const auto& option : values) {
      if (option.id == id) {
        return option;
      }
    }
    return {id, def};
  }

  OptionalValue& add(const OptionalValue& def) {
    for (auto& option : values) {
      if (option.id == def.id) {
        auto c = option.cls;
        option = def;
        option.cls = c;
        return option;
      }
    }
    return values.emplace_back(def);
  }

  template <class... args>
  void add(args&&... a) {
    add({std::forward<args>(a)...});
  }

  void set(const std::string& id, const std::string& value) {
    for (auto& option : values) {
      if (option.id != id)
        continue;

      ETX_ASSERT(option.cls == OptionalValue::Class::InfoString);
      option.name = value;
      return;
    }

    OptionalValue& val = values.emplace_back();
    val.cls = OptionalValue::Class::InfoString;
    val.id = id;
    val.name = value;
  }

  void set(const std::string& id, uint32_t value) {
    for (auto& option : values) {
      if (option.id != id)
        continue;

      ETX_ASSERT((option.cls == OptionalValue::Class::Integer) || (option.cls == OptionalValue::Class::Enum));
      option.value.integer = value < option.min_value.integer ? option.min_value.integer : (value > option.max_value.integer ? option.max_value.integer : value);
      return;
    }

    OptionalValue& val = values.emplace_back();
    val.cls = OptionalValue::Class::Integer;
    val.id = id;
    val.value.integer = value;
  }

  void set(const std::string& id, float value) {
    for (auto& option : values) {
      if (option.id != id)
        continue;

      ETX_ASSERT(option.cls == OptionalValue::Class::Float);
      option.value.flt = value < option.min_value.flt ? option.min_value.flt : (value > option.max_value.flt ? option.max_value.flt : value);
      return;
    }

    OptionalValue& val = values.emplace_back();
    val.cls = OptionalValue::Class::Float;
    val.id = id;
    val.value.flt = value;
  }

  bool has(const std::string& id) const {
    for (auto& option : values) {
      if (option.id == id) {
        return true;
      }
    }
    return false;
  }

  void save_to_file(const char*);
  bool load_from_file(const char*);
};

}  // namespace etx
