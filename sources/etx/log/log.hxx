#pragma once

namespace etx {

struct log {
  enum class Color {
    Green,
    White,
    Yellow,
    Red,
  };

  static void output(Color color, const char* fmt, ...);

  template <class... args>
  static inline void success(const char* fmt, args... a) {
    output(Color::Green, fmt, static_cast<args&&>(a)...);
  }

  template <class... args>
  static inline void info(const char* fmt, args... a) {
    output(Color::White, fmt, static_cast<args&&>(a)...);
  }

  template <class... args>
  static inline void warning(const char* fmt, args... a) {
    output(Color::Yellow, fmt, static_cast<args&&>(a)...);
  }

  template <class... args>
  static inline void error(const char* fmt, args... a) {
    output(Color::Red, fmt, static_cast<args&&>(a)...);
  }
};

}  // namespace etx
