#pragma once

#include <stdint.h>
#include <string>
#include <vector>

namespace etx {

struct TimeMeasure {
  TimeMeasure();

  void reset();
  double lap();

  double measure() const;
  double measure_ms() const;
  uint64_t measure_exact() const;

  static float get_cpu_load();

 private:
  uint64_t _data;
};

void init_platform();

std::string open_file(const std::vector<std::string>& filters);
std::string save_file(const std::vector<std::string>& filters);

bool load_binary_file(const char* filename, std::vector<uint8_t>& data);

template <class T>
constexpr inline T align_up(T sz, T al) {
  static_assert(std::is_integral<T>::value);
  T m = al - T(1);
  return sz + m & (~m);
}

enum : uint32_t {
  kFnv1a32Prime = 16777619u,
  kFnv1a32Begin = 2166136261u,
};

constexpr inline uint32_t fnv1a32(const char* str, const uint32_t hash = kFnv1a32Begin) {
  return (str && (*str)) ? fnv1a32(str + 1, (hash ^ uint32_t(*str)) * kFnv1a32Prime) : hash;
}

}  // namespace etx
