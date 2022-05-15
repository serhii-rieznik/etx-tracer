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

float TimeMeasure::get_cpu_load() {
  auto CalculateCPULoad = [](unsigned long long idleTicks, unsigned long long totalTicks) {
    static unsigned long long _previousTotalTicks = 0;
    static unsigned long long _previousIdleTicks = 0;

    unsigned long long totalTicksSinceLastTime = totalTicks - _previousTotalTicks;
    unsigned long long idleTicksSinceLastTime = idleTicks - _previousIdleTicks;

    float ret = 1.0f - ((totalTicksSinceLastTime > 0) ? ((float)idleTicksSinceLastTime) / totalTicksSinceLastTime : 0);

    _previousTotalTicks = totalTicks;
    _previousIdleTicks = idleTicks;
    return ret;
  };

  auto FileTimeToInt64 = [](const FILETIME& ft) {
    return (((unsigned long long)(ft.dwHighDateTime)) << 32) | ((unsigned long long)ft.dwLowDateTime);
  };

  FILETIME idleTime = {}, kernelTime = {}, userTime = {};
  return GetSystemTimes(&idleTime, &kernelTime, &userTime) ? CalculateCPULoad(FileTimeToInt64(idleTime), FileTimeToInt64(kernelTime) + FileTimeToInt64(userTime)) : -1.0f;
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
