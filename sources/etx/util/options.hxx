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

  bool remove(const std::string& id) {
    auto it = std::find_if(options.begin(), options.end(), [&](const Option& opt) {
      return opt.id == id;
    });
    if (it != options.end()) {
      options.erase(it);
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
};

}  // namespace etx