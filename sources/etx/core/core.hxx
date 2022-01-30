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
  uint64_t measure_exact() const;

 private:
  uint64_t _data;
};

std::string open_file(const std::vector<std::string>& filters);

}  // namespace etx
