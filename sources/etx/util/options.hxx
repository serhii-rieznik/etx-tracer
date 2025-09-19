#pragma once

#include <etx/core/debug.hxx>
#include <etx/render/shared/base.hxx>

#include <vector>
#include <string>
#include <functional>

namespace etx {

struct Option {
  enum class Class : uint32_t {
    Undefined,
    Boolean,
    Integral,
    Float,
    Float3,
    String,
    Any,
  };

  enum Meta : uint32_t {
    RegularValue = 0,
    EnumValue = 1u << 0u,
  };

  template <Class cls>
  struct Accessor {};

  Class cls = Class::Undefined;
  uint32_t meta = 0;
  uint8_t data[40] = {};
  std::string id = {};
  std::string description = {};
  std::function<std::string(uint32_t)> name_getter = {};

  template <Class cls>
  Accessor<cls>::T& as() {
    return *reinterpret_cast<Accessor<cls>::T*>(data);
  }

  template <Class cls>
  const Accessor<cls>::T& as() const {
    return *reinterpret_cast<const Accessor<cls>::T*>(data);
  }

  template <Class cls>
  Accessor<cls>::T& init() {
    using DataType = Accessor<cls>::T;
    auto ptr = reinterpret_cast<DataType*>(data);
    ptr->~DataType();
    new (data) DataType();
    return *ptr;
  }

  template <class T>
  struct Bounds {
    T minimum = {};
    T maximum = {};
  };
  template <class T>
  struct Value {
    T value = {};
  };
  template <class T>
  struct BoundedValue : public Value<T> {
    Bounds<T> bounds = {};
  };

  using BooleanValue = Value<bool>;
  using StringValue = Value<std::string>;
  using IntegralValue = BoundedValue<int32_t>;
  using FloatValue = BoundedValue<float>;
  using Float3Value = BoundedValue<float3>;

#define EXT_OPT_BIND(Cls)       \
  template <>                   \
  struct Accessor<Class::Cls> { \
    using T = Cls##Value;       \
  };                            \
  static_assert(sizeof(Cls##Value) <= sizeof(data))

  EXT_OPT_BIND(Boolean);
  EXT_OPT_BIND(Integral);
  EXT_OPT_BIND(Float);
  EXT_OPT_BIND(Float3);
  EXT_OPT_BIND(String);

 private:
  friend struct Options;
  void serialize(void*) const;
  void deserialize(const void*);
};

struct Options {
  std::vector<Option> options;

  void save_to_file(const std::string&);
  bool load_from_file(const std::string&);

  bool has(const std::string& id, Option::Class cls) const {
    for (const auto& opt : options) {
      if ((opt.id == id) && ((opt.cls == cls) || (cls == Option::Class::Any)))
        return true;
    }
    return false;
  }

  const Option& get(const std::string& id) const {
    for (const auto& option : options) {
      if (option.id == id)
        return option;
    }

    static Option empty = {.cls = Option::Class::Undefined};
    return empty;
  }

  Option& get_mutable(const std::string& id) {
    for (auto& option : options) {
      if (option.id == id)
        return option;
    }
    ETX_FAIL_FMT("Option %s not found", id.c_str());
    static Option empty = {.cls = Option::Class::Undefined};
    return empty;
  }

  template <Option::Class Cls, class Init>
  Option& make(const std::string& id, const std::string& description, uint32_t meta, Init init) {
    using DataType = Option::Accessor<Cls>::T;

    auto& opt = has(id, Cls) ? get_mutable(id) : options.emplace_back();
    opt.id = id;
    opt.cls = Cls;
    opt.meta = meta;
    opt.description = description;
    init(opt.init<Cls>());
    return opt;
  }

  Option& set_bool(const std::string& id, const bool value, const std::string& description) {
    auto& opt = make<Option::Class::Boolean>(id, description, 0u, [&](auto& data) {
      data.value = value;
    });
    return opt;
  }

  bool get_bool(const std::string& id, const bool default_value) const {
    if (has(id, Option::Class::Boolean))
      return get(id).as<Option::Class::Boolean>().value;
    if (has(id, Option::Class::Integral))
      return get(id).as<Option::Class::Integral>().value != 0;
    return default_value;
  }

  template <class T>
  Option& set_integral(const std::string& id, const T value, const std::string& description, uint32_t meta = Option::Meta::RegularValue, const Option::Bounds<T> bounds = {}) {
    auto& opt = make<Option::Class::Integral>(id, description, meta, [&](auto& data) {
      data.value = static_cast<int64_t>(value);
      data.bounds.minimum = static_cast<int64_t>(bounds.minimum);
      data.bounds.maximum = static_cast<int64_t>(bounds.maximum);
    });
    return opt;
  }

  template <class T>
  T get_integral(const std::string& id, const T default_value) const {
    if (has(id, Option::Class::Integral))
      return static_cast<T>(get(id).as<Option::Class::Integral>().value);
    if (has(id, Option::Class::Float))
      return static_cast<T>(get(id).as<Option::Class::Float>().value);
    return default_value;
  }

  Option& set_float(const std::string& id, const float value, const std::string& description, const Option::Bounds<float> bounds = {}) {
    auto& opt = make<Option::Class::Float>(id, description, 0u, [&](auto& data) {
      data.value = value;
      data.bounds.minimum = bounds.minimum;
      data.bounds.maximum = bounds.maximum;
    });
    return opt;
  }

  float get_float(const std::string& id, const float default_value) const {
    if (has(id, Option::Class::Float))
      return static_cast<float>(get(id).as<Option::Class::Float>().value);
    if (has(id, Option::Class::Integral))
      return static_cast<float>(get(id).as<Option::Class::Integral>().value);
    return default_value;
  }

  Option& set_float3(const std::string& id, const float3& value, const std::string& description, const Option::Bounds<float3> bounds = {}) {
    auto& opt = make<Option::Class::Float3>(id, description, 0u, [&](auto& data) {
      data.value = value;
      data.bounds = bounds;
    });
    return opt;
  }

  float3 get_float3(const std::string& id, const float3& default_value) const {
    if (has(id, Option::Class::Float3))
      return get(id).as<Option::Class::Float3>().value;
    return default_value;
  }

  Option& set_string(const std::string& id, const std::string& value, const std::string& description) {
    auto& opt = make<Option::Class::String>(id, description, 0u, [&](auto& data) {
      data.value = value;
    });
    return opt;
  }

  const std::string& get_string(const std::string& id, const std::string& default_value) const {
    if (has(id, Option::Class::String))
      return get(id).as<Option::Class::String>().value;
    return default_value;
  }
};  // namespace etx

/*
struct OptionalValue {
  enum class Class : uint32_t {
    Undefined,
    Integral,
    Floating,
    String,

    Count,
  };

  template <Class cls>
  struct Container;

  template <>
  struct Container<Class::Integral> {
    int64_t value;
    int64_t min_value = 0;
    int64_t max_value = 0;
  };

  template <>
  struct Container<Class::Boolean> {
    int64_t value;
    int64_t min_value = 0;
    int64_t max_value = 0;
  };

  std::string id = {};
  std::string name = {};
  std::string (*name_func)(uint32_t) = {};

  Value min_value = {};
  Value value = {};
  Value max_value = {};
  Class cls = Class::Integer;

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

      ETX_ASSERT(option.cls == Options::Class::InfoString);
      option.name = value;
      return;
    }

    OptionalValue& val = values.emplace_back();
    val.cls = Options::Class::InfoString;
    val.id = id;
    val.name = value;
  }

  void set(const std::string& id, uint32_t value) {
    for (auto& option : values) {
      if (option.id != id)
        continue;

      ETX_ASSERT((option.cls == Options::Class::Integer) || (option.cls == Options::Class::Enum));
      option.value.integer = value < option.min_value.integer ? option.min_value.integer : (value > option.max_value.integer ? option.max_value.integer : value);
      return;
    }

    OptionalValue& val = values.emplace_back();
    val.cls = Options::Class::Integer;
    val.id = id;
    val.value.integer = value;
  }

  void set(const std::string& id, float value) {
    for (auto& option : values) {
      if (option.id != id)
        continue;

      ETX_ASSERT(option.cls == Options::Class::Float);
      option.value.flt = value < option.min_value.flt ? option.min_value.flt : (value > option.max_value.flt ? option.max_value.flt : value);
      return;
    }

    OptionalValue& val = values.emplace_back();
    val.cls = Options::Class::Float;
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

  void save_to_file(const std::string&);
  bool load_from_file(const std::string&);
};
// */

}  // namespace etx
