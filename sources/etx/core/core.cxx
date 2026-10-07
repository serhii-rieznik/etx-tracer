#include <etx/core/core.hxx>

namespace etx {

TimeMeasure::TimeMeasure() {
  reset();
}

void TimeMeasure::reset() {
  _data = std::chrono::steady_clock::now().time_since_epoch().count();
}

double TimeMeasure::measure_ms() const {
  auto exact = measure_exact();
  return double(exact) / double(std::micro::den);
}

double TimeMeasure::measure() const {
  auto exact = measure_exact();
  return double(exact) / double(std::nano::den);
}

double TimeMeasure::lap() {
  auto m = measure();
  reset();
  return m;
}

uint64_t TimeMeasure::measure_exact() const {
  return std::chrono::steady_clock::now().time_since_epoch().count() - _data;
}

bool load_binary_file(const char* filename, std::vector<uint8_t>& output) {
  FILE* f_in = fopen_utf8(filename, "rb");
  if (f_in == nullptr) {
    return false;
  }

  if (fseek(f_in, 0, SEEK_END) != 0) {
    fclose(f_in);
    return false;
  }
  const long file_size = ftell(f_in);
  if ((file_size < 0) || (fseek(f_in, 0, SEEK_SET) != 0)) {
    fclose(f_in);
    return false;
  }
  output.resize(file_size);

  const size_t bytes_read = fread(output.data(), 1, output.size(), f_in);
  if (bytes_read != output.size()) {
    fclose(f_in);
    return false;
  }

  fclose(f_in);
  return true;
}

}  // namespace etx
