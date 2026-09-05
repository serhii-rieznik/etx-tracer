#if (ETX_PLATFORM_APPLE)
# include <libkern/OSAtomic.h>
#endif

#include <nfd.h>

#include <filesystem>

namespace etx {

uint32_t atomic_inc(int32_t* ptr) {
#if (ETX_PLATFORM_APPLE)
  return OSAtomicIncrement32(ptr);
#elif (ETX_PLATFORM_WINDOWS)
  static_assert(sizeof(long) == sizeof(int32_t));
  return _InterlockedIncrement(reinterpret_cast<volatile long*>(ptr));
#endif
}

uint64_t atomic_inc(int64_t* ptr) {
#if (ETX_PLATFORM_APPLE)
  return OSAtomicIncrement64(ptr);
#elif (ETX_PLATFORM_WINDOWS)
  static_assert(sizeof(long long) == sizeof(int64_t));
  return _InterlockedIncrement64(reinterpret_cast<volatile long long*>(ptr));
#endif
}

int64_t atomic_add_int64(int64_t* ptr, int64_t value) {
#if (ETX_PLATFORM_APPLE)
  return OSAtomicAdd64(value, ptr);
#elif (ETX_PLATFORM_WINDOWS)
  static_assert(sizeof(long) == sizeof(int32_t));
  return _InterlockedExchangeAdd64(reinterpret_cast<volatile long long*>(ptr), value);
#endif
}

void atomic_add_float(float* ptr, float value) {
#if (ETX_PLATFORM_WINDOWS)
  volatile long* iptr = std::bit_cast<volatile long*>(ptr);
  long old_value, new_value;
  do {
    old_value = std::bit_cast<long>(*ptr);
    new_value = std::bit_cast<long>(*ptr + value);
  } while (_InterlockedCompareExchange(iptr, new_value, old_value) != old_value);
#elif (ETX_PLATFORM_APPLE)
  volatile int32_t* iptr = std::bit_cast<volatile int32_t*>(ptr);
  int32_t old_value, new_value;
  do {
    old_value = std::bit_cast<int32_t>(*ptr);
    new_value = std::bit_cast<int32_t>(*ptr + value);
  } while (!OSAtomicCompareAndSwap32(old_value, new_value, iptr));
#endif
}

namespace {

nfdwindowhandle_t native_dialog_parent(void* parent_window) {
  nfdwindowhandle_t result = {};
  if (parent_window == nullptr) {
    return result;
  }

#if defined(_WIN32)
  result.type = NFD_WINDOW_HANDLE_TYPE_WINDOWS;
#elif defined(__APPLE__)
  result.type = NFD_WINDOW_HANDLE_TYPE_COCOA;
#endif
  result.handle = parent_window;
  return result;
}

std::string copy_selected_path(nfdu8char_t* selected_path, nfdresult_t result) {
  if ((result != NFD_OKAY) || (selected_path == nullptr)) {
    return {};
  }

  std::string path = selected_path;
  NFD_FreePathU8(selected_path);
  return path;
}

}  // namespace

std::string open_file(const char* filters, void* parent_window) {
  if (NFD_Init() != NFD_OKAY) {
    return {};
  }

  const nfdu8filteritem_t filter = {"Supported files", filters};
  const nfdopendialogu8args_t args = {
    .filterList = (filters != nullptr) && (filters[0] != '\0') ? &filter : nullptr,
    .filterCount = (filters != nullptr) && (filters[0] != '\0') ? 1u : 0u,
    .defaultPath = nullptr,
    .parentWindow = native_dialog_parent(parent_window),
  };

  nfdu8char_t* selected_path = nullptr;
  const nfdresult_t result = NFD_OpenDialogU8_With(&selected_path, &args);
  std::string path = copy_selected_path(selected_path, result);
  NFD_Quit();
  return path;
}

std::string save_file(const char* filters, void* parent_window) {
  if (NFD_Init() != NFD_OKAY) {
    return {};
  }

  const nfdu8filteritem_t filter = {"Supported files", filters};
  const nfdsavedialogu8args_t args = {
    .filterList = (filters != nullptr) && (filters[0] != '\0') ? &filter : nullptr,
    .filterCount = (filters != nullptr) && (filters[0] != '\0') ? 1u : 0u,
    .defaultPath = nullptr,
    .defaultName = nullptr,
    .parentWindow = native_dialog_parent(parent_window),
  };

  nfdu8char_t* selected_path = nullptr;
  const nfdresult_t result = NFD_SaveDialogU8_With(&selected_path, &args);
  std::string path = copy_selected_path(selected_path, result);
  NFD_Quit();
  return path;
}

std::string utf8_file_name(const std::string& path) {
  const std::u8string file_name = std::filesystem::u8path(path).filename().u8string();
  if (file_name.empty()) {
    return path;
  }
  return {reinterpret_cast<const char*>(file_name.data()), file_name.size()};
}

}  // namespace etx
