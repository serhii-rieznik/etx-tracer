#include <etx/core/core.hxx>

#include <chrono>

#include <etx/core/windows.hxx>
#include <commdlg.h>

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

std::string open_file(const std::vector<std::string>& filters) {
  char name_buffer[MAX_PATH] = {};

  size_t fp = 0;
  char filter_buffer[2048] = {};
  for (const std::string& w : filters) {
    memcpy(filter_buffer + fp, w.data(), w.length());
    fp += 1 + w.length();
  }

  OPENFILENAME of = {};
  of.lStructSize = sizeof(of);
  of.hInstance = GetModuleHandle(nullptr);
  of.Flags = OFN_ENABLESIZING | OFN_EXPLORER | OFN_NOCHANGEDIR | OFN_PATHMUSTEXIST;
  of.lpstrFile = name_buffer;
  of.nMaxFile = MAX_PATH;
  of.lpstrFilter = filter_buffer;
  of.nFilterIndex = filters.empty() ? 0 : 1;
  return GetOpenFileNameA(&of) ? of.lpstrFile : "";
}

std::string save_file(const std::vector<std::string>& filters) {
  char name_buffer[MAX_PATH] = {};

  size_t fp = 0;
  char filter_buffer[2048] = {};
  for (const std::string& w : filters) {
    memcpy(filter_buffer + fp, w.data(), w.length());
    fp += 1 + w.length();
  }

  OPENFILENAME of = {};
  of.lStructSize = sizeof(of);
  of.hInstance = GetModuleHandle(nullptr);
  of.Flags = OFN_ENABLESIZING | OFN_EXPLORER | OFN_NOCHANGEDIR | OFN_OVERWRITEPROMPT;
  of.lpstrFile = name_buffer;
  of.nMaxFile = MAX_PATH;
  of.lpstrFilter = filter_buffer;
  of.nFilterIndex = filters.empty() ? 0 : 1;
  return GetSaveFileName(&of) ? of.lpstrFile : "";
}

bool load_binary_file(const char* filename, std::vector<uint8_t>& output) {
  FILE* f_in = fopen(filename, "rb");
  if (f_in == nullptr) {
    return false;
  }

  fseek(f_in, 0, SEEK_END);
  uint64_t file_size = ftell(f_in);
  output.resize(file_size);
  fseek(f_in, 0, SEEK_SET);

  uint64_t bytes_read = fread(output.data(), 1, file_size, f_in);
  if (bytes_read != file_size) {
    fclose(f_in);
    return false;
  }

  fclose(f_in);
  return true;
}

}  // namespace etx
