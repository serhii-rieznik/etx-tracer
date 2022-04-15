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

 private:
  uint64_t _data;
};

std::string open_file(const std::vector<std::string>& filters);
std::string save_file(const std::vector<std::string>& filters);

bool load_binary_file(const char* filename, std::vector<uint8_t>& data);

constexpr inline uint64_t align_up(uint64_t sz, uint64_t al) {
  uint64_t m = al - 1;
  return sz + m & (~m);
}

}  // namespace etx
