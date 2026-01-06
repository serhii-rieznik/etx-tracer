#pragma once

#include <etx/core/platform.hxx>
#include <etx/core/debug.hxx>
#include <etx/core/log.hxx>
#include <etx/core/profiler.hxx>

#include <vector>
#include <string>
#include <cstring>

extern "C" {
uint32_t XXH32(const void* input, size_t length, uint32_t seed);
uint64_t XXH64(const void* input, size_t length, uint64_t seed);
}

namespace etx {

struct TimeMeasure {
  TimeMeasure();

  void reset();
  double lap();

  double measure() const;
  double measure_ms() const;
  uint64_t measure_exact() const;

 private:
  uint64_t _data;
};

void init_platform();

std::string open_file(const char* filters);
std::string save_file(const char* filters);

uint32_t atomic_inc(int32_t* ptr);
uint64_t atomic_inc(int64_t* ptr);
void atomic_add_float(float* ptr, float value);
int64_t atomic_add_int64(int64_t* ptr, int64_t value);

bool load_binary_file(const char* filename, std::vector<uint8_t>& data);

template <class T>
constexpr inline T align_up(T sz, T al) {
  static_assert(std::is_integral<T>::value);
  T m = al - T(1);
  return sz + m & (~m);
}

inline uint32_t xxh32(const void* ptr, uint64_t size) {
  return XXH32(ptr, size, 0);
}

inline uint32_t xxh32(const char* str) {
  return XXH32(str, str ? strlen(str) : 0, 0);
}

inline uint64_t xxh64(const void* ptr, uint64_t size) {
  return XXH64(ptr, size, 0);
}

inline uint64_t xxh64(const char* str) {
  return XXH64(str, str ? strlen(str) : 0, 0);
}

inline uint32_t etx_hash32(const uint8_t* ptr, uint64_t size) {
  return xxh32(ptr, size);
}

inline uint32_t etx_hash32(const char* str) {
  return xxh32(str);
}

inline uint64_t etx_hash64(const uint8_t* ptr, uint64_t size) {
  return xxh64(ptr, size);
}

inline uint64_t etx_hash64(const char* str) {
  return xxh64(str);
}

inline uint32_t etx_hash32_continue(const void* ptr, uint64_t size, uint32_t seed) {
  return XXH32(ptr, size, seed);
}

inline uint32_t etx_hash32_continue(const char* str, uint32_t seed) {
  return XXH32(str, strlen(str), seed);
}

inline uint64_t etx_hash64_continue(const void* ptr, uint64_t size, uint64_t seed) {
  return XXH64(ptr, size, seed);
}

inline uint64_t etx_hash64_continue(const char* str, uint64_t seed) {
  return XXH64(str, strlen(str), seed);
}

}  // namespace etx
