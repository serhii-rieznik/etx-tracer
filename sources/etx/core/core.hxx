#pragma once

#include <stdint.h>

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

}  // namespace etx
