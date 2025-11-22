#include <etx/core/environment.hxx>
#include <etx/core/debug.hxx>

#include <filesystem>

#include <string.h>
#include <stdio.h>

#if (ETX_PLATFORM_WINDOWS)
# include <windows.h>
#else
# include <unistd.h>
# include <limits.h>
#endif

namespace etx {

#if (ETX_PLATFORM_WINDOWS)
constexpr char kDelimiter = '\\';
#else
constexpr char kDelimiter = '/';
#endif

inline static void normalize_path(char buffer[]) {
  if ((buffer == nullptr) || (buffer[0] == 0))
    return;

  auto ptr = buffer;
  while (*ptr) {
    if ((*ptr == '\\') || (*ptr == '/')) {
      *ptr = kDelimiter;
    }
    ++ptr;
  }
}

static struct {
  char data_folder[2048] = {};
  char current_directory[2048] = {};
  Environment e;
} _env;

const char* Environment::data_folder() {
  return _env.data_folder;
}

const char* Environment::file_in_data(const char* f, char buffer[], uint64_t buffer_size) {
  snprintf(buffer, buffer_size, "%s%s", _env.data_folder, f);
  normalize_path(buffer);
  return buffer;
}

const char* Environment::file_in_data(const char* f) {
  static char buffer[2048] = {};
  return file_in_data(f, buffer, sizeof(buffer));
}

void Environment::setup(const char* executable_path) {
#if (ETX_PLATFORM_WINDOWS)
  char exe_path[MAX_PATH] = {};
  DWORD len = GetModuleFileNameA(nullptr, exe_path, MAX_PATH);
  if (len > 0 && len < MAX_PATH) {
    executable_path = exe_path;
  }
#else
  char exe_path[PATH_MAX] = {};
  ssize_t len = readlink("/proc/self/exe", exe_path, sizeof(exe_path) - 1);
  if (len > 0 && len < (ssize_t)sizeof(exe_path)) {
    exe_path[len] = '\0';
    executable_path = exe_path;
  }
#endif
  get_file_folder(executable_path, _env.data_folder, sizeof(_env.data_folder));
  snprintf(_env.current_directory, sizeof(_env.current_directory), "%s", _env.data_folder);
}

const char* Environment::current_directory() const {
  return _env.current_directory;
}

std::string Environment::to_project_relative(const std::string& path) const {
  if (path.empty()) {
    return {};
  }

  std::filesystem::path in_path(path);
  if (in_path.is_relative()) {
    auto str = in_path.generic_string();
    if (str[0] == '/')
      str = "." + str;
    else if (str[0] != '.') {
      str = "./" + str;
    }
    return str;
  }

  std::error_code ec;
  std::filesystem::path canonical = std::filesystem::weakly_canonical(in_path, ec);
  if (ec) {
    canonical = std::filesystem::absolute(in_path, ec);
  }

  std::filesystem::path root(current_directory());
  auto relative = canonical.lexically_relative(root);
  std::string relative_str = relative.generic_string();
  if (relative.empty() || (relative_str.rfind("..", 0) == 0)) {
    return canonical.generic_string();
  }

  return "./" + relative.generic_string();
}

std::string Environment::resolve_to_absolute(const std::string& path) const {
  if (path.empty()) {
    return {};
  }

  std::filesystem::path in_path(path);
  std::error_code ec;
  if (in_path.is_absolute()) {
    auto canonical = std::filesystem::weakly_canonical(in_path, ec);
    if (!ec) {
      return canonical.generic_string();
    }
    canonical = std::filesystem::absolute(in_path, ec);
    return canonical.generic_string();
  }

  std::filesystem::path root(current_directory());
  std::filesystem::path resolved = std::filesystem::weakly_canonical(root / in_path, ec);
  if (ec) {
    resolved = std::filesystem::absolute(root / in_path, ec);
  }
  return resolved.generic_string();
}

uint64_t get_file_folder(const char* file_name, char buffer[], uint64_t buffer_size) {
  uint64_t fn_len = file_name ? strlen(file_name) : 0;
  if (fn_len == 0) {
    return 0;
  }
  int len = snprintf(buffer, buffer_size, "%s", file_name);
  while ((len > 0) && (buffer[len] != '/') && (buffer[len] != '\\')) {
    --len;
  }
  buffer[len] = '/';
  ETX_ASSERT(len + 1 < buffer_size);
  buffer[1llu + len] = 0;
  normalize_path(buffer);
  return 1ll + len;
}

const char* get_file_ext(const char* file_name) {
  uint64_t fn_len = file_name ? strlen(file_name) : 0;
  while (fn_len > 0) {
    if (file_name[fn_len - 1] == '.') {
      return file_name + fn_len - 1;
    }
    --fn_len;
  }
  return "";
}

Environment& env() {
  return _env.e;
}

}  // namespace etx
