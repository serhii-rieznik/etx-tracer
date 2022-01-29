#pragma once

#include <etx/core/debug.hxx>

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

    Count,
  };

  Class cls = Class::Integer;
  std::string id = {};
  std::string name = {};

  union Value {
    uint32_t integer;
    float flt;
    bool boolean;
  };

  Value min_value = {};
  Value value = {};
  Value max_value = {};
  std::string (*name_func)(uint32_t) = {};

  OptionalValue() = default;

  OptionalValue(uint32_t val, const std::string& a_id, const std::string& desc)
    : cls(Class::Integer)
    , id(a_id)
    , name(desc) {
    min_value.integer = val;
    value.integer = val;
    max_value.integer = val;
  }

  OptionalValue(float val, const std::string& a_id, const std::string& desc)
    : cls(Class::Float)
    , id(a_id)
    , name(desc) {
    min_value.flt = val;
    value.flt = val;
    max_value.flt = val;
  }

  OptionalValue(uint32_t min_val, uint32_t val, uint32_t max_val, const std::string& a_id, const std::string& desc)
    : cls(Class::Integer)
    , id(a_id)
    , name(desc) {
    min_value.integer = min_val;
    value.integer = val;
    max_value.integer = max_val;
  }

  OptionalValue(float min_val, float val, float max_val, const std::string& a_id, const std::string& desc)
    : cls(Class::Float)
    , id(a_id)
    , name(desc) {
    min_value.flt = min_val;
    value.flt = val;
    max_value.flt = max_val;
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
    , name_func(f)
    , id(a_id)
    , name(desc) {
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

  uint32_t to_integer() const {
    ETX_ASSERT((cls == Class::Integer) || (cls == Class::Enum));
    return clamp_to(value.integer, min_value.integer, max_value.integer);
  }

  float to_float() const {
    ETX_ASSERT(cls == Class::Float);
    return clamp_to(value.flt, min_value.flt, max_value.flt);
  }

  bool to_bool() const {
    ETX_ASSERT(cls == Class::Boolean);
    return value.boolean;
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

  OptionalValue& set(const OptionalValue& def) {
    for (auto& option : values) {
      if (option.id == def.id) {
        option = def;
        return option;
      }
    }
    return values.emplace_back(def);
  }

  template <class... args>
  void set(args&&... a) {
    set({std::forward<args>(a)...});
  }
};

}  // namespace etx
