#include <etx/core/environment.hxx>
#include <etx/core/debug.hxx>
#if (ETX_PLATFORM_WINDOWS)
# include <windows.h>
#elif (ETX_PLATFORM_APPLE)
# include <mach-o/dyld.h>
# include <pwd.h>
# include <unistd.h>
#else
# include <unistd.h>
#endif

#include <filesystem>
#include <string>
#include <vector>

namespace etx {

#if (ETX_PLATFORM_WINDOWS)
constexpr char kDelimiter = '\\';
# define ETX_FSEEK     _fseeki64
# define ETX_FTELL     _ftelli64
# define ETX_FPOS_TYPE long long
#else
constexpr char kDelimiter = '/';
# define ETX_FSEEK     fseeko
# define ETX_FTELL     ftello
# define ETX_FPOS_TYPE off_t
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
  char user_data_folder[2048] = {};
  char config_folder[2048] = {};
  char cache_folder[2048] = {};
  char library_folder[2048] = {};
  char tmp_folder[2048] = {};
  char current_directory[2048] = {};
  bool bundled = false;
  Environment e;
} _env;

static void set_folder_path(char* destination, size_t destination_size, const std::filesystem::path& path) {
  std::string value = path.generic_string();
  if (!value.empty() && (value.back() != '/')) {
    value.push_back('/');
  }
  snprintf(destination, destination_size, "%s", value.c_str());
  normalize_path(destination);
}

static void ensure_tmp_directory() {
  if (_env.tmp_folder[0] == 0)
    return;

  std::error_code ec;
  std::filesystem::create_directories(_env.tmp_folder, ec);
}

static void clear_directory(const char* path) {
  if ((path == nullptr) || (path[0] == 0))
    return;

  std::error_code ec;
  std::filesystem::remove_all(path, ec);
  std::filesystem::create_directories(path, ec);
}

const char* Environment::data_folder() {
  return _env.data_folder;
}

const char* Environment::user_data_folder() {
  return _env.user_data_folder;
}

const char* Environment::config_folder() {
  return _env.config_folder;
}

const char* Environment::cache_folder() {
  return _env.cache_folder;
}

const char* Environment::library_folder() {
  return _env.library_folder;
}

bool Environment::bundled() const {
  return _env.bundled;
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

const char* Environment::file_in_user_data(const char* f, char buffer[], uint64_t buffer_size) {
  snprintf(buffer, buffer_size, "%s%s", _env.user_data_folder, f);
  normalize_path(buffer);
  return buffer;
}

const char* Environment::file_in_user_data(const char* f) {
  static char buffer[2048] = {};
  return file_in_user_data(f, buffer, sizeof(buffer));
}

const char* Environment::file_in_config(const char* f, char buffer[], uint64_t buffer_size) {
  snprintf(buffer, buffer_size, "%s%s", _env.config_folder, f);
  normalize_path(buffer);
  return buffer;
}

const char* Environment::file_in_config(const char* f) {
  static char buffer[2048] = {};
  return file_in_config(f, buffer, sizeof(buffer));
}

const char* Environment::file_in_cache(const char* f, char buffer[], uint64_t buffer_size) {
  snprintf(buffer, buffer_size, "%s%s", _env.cache_folder, f);
  normalize_path(buffer);
  return buffer;
}

const char* Environment::file_in_cache(const char* f) {
  static char buffer[2048] = {};
  return file_in_cache(f, buffer, sizeof(buffer));
}

const char* Environment::tmp_folder() {
  ensure_tmp_directory();
  return _env.tmp_folder;
}

const char* Environment::file_in_tmp(const char* f, char buffer[], uint64_t buffer_size) {
  ensure_tmp_directory();
  snprintf(buffer, buffer_size, "%s%s", _env.tmp_folder, f);
  normalize_path(buffer);
  return buffer;
}

const char* Environment::file_in_tmp(const char* f) {
  static char buffer[2048] = {};
  return file_in_tmp(f, buffer, sizeof(buffer));
}

void Environment::clear_tmp_folder() {
  ensure_tmp_directory();
  clear_directory(_env.tmp_folder);
}

void Environment::setup(const char* executable_path) {
  std::string platform_executable_path;

#if (ETX_PLATFORM_WINDOWS)
  char exe_path[MAX_PATH] = {};
  DWORD len = GetModuleFileNameA(nullptr, exe_path, MAX_PATH);
  if (len > 0 && len < MAX_PATH) {
    platform_executable_path.assign(exe_path, len);
  }
#elif (ETX_PLATFORM_APPLE)
  uint32_t path_size = PATH_MAX;
  std::vector<char> exe_path(path_size);
  if (_NSGetExecutablePath(exe_path.data(), &path_size) != 0) {
    exe_path.resize(path_size);
  }
  if (_NSGetExecutablePath(exe_path.data(), &path_size) == 0) {
    platform_executable_path.assign(exe_path.data());
  }
#else
  char exe_path[PATH_MAX] = {};
  ssize_t len = readlink("/proc/self/exe", exe_path, sizeof(exe_path) - 1);
  if (len > 0 && len < (ssize_t)sizeof(exe_path)) {
    exe_path[len] = '\0';
    platform_executable_path.assign(exe_path, static_cast<size_t>(len));
  }
#endif

  std::filesystem::path resolved_executable = platform_executable_path.empty()
    ? std::filesystem::path(executable_path ? executable_path : "")
    : std::filesystem::path(platform_executable_path);

  std::error_code ec;
  if (resolved_executable.is_relative()) {
    resolved_executable = std::filesystem::absolute(resolved_executable, ec);
  }
  if (!ec) {
    auto canonical_executable = std::filesystem::weakly_canonical(resolved_executable, ec);
    if (!ec) {
      resolved_executable = std::move(canonical_executable);
    }
  }

  std::filesystem::path executable_folder = resolved_executable.parent_path();
  if (executable_folder.empty()) {
    executable_folder = std::filesystem::current_path(ec);
  }

  std::filesystem::path data_folder = executable_folder;
  std::filesystem::path user_data_folder = executable_folder;
  std::filesystem::path config_folder = executable_folder;
  std::filesystem::path cache_folder = executable_folder / "cache";
  std::filesystem::path library_folder = executable_folder;
  std::filesystem::path tmp_folder = executable_folder / "tmp";

#if (ETX_PLATFORM_APPLE)
  const std::filesystem::path contents_folder = executable_folder.parent_path();
  const std::filesystem::path bundle_folder = contents_folder.parent_path();
  _env.bundled = (executable_folder.filename() == "MacOS") && (contents_folder.filename() == "Contents") && (bundle_folder.extension() == ".app");
  if (_env.bundled) {
    data_folder = contents_folder / "Resources";
    library_folder = contents_folder / "Frameworks";
  }

  const char* home_value = getenv("HOME");
  if ((home_value == nullptr) || (home_value[0] == '\0')) {
    if (const passwd* user = getpwuid(getuid()); user != nullptr) {
      home_value = user->pw_dir;
    }
  }
  const std::filesystem::path home_folder = ((home_value != nullptr) && (home_value[0] != '\0'))
    ? std::filesystem::path(home_value)
    : std::filesystem::temp_directory_path(ec);
  const std::filesystem::path application_support_folder = home_folder / "Library" / "Application Support" / "ETX Tracer";
  user_data_folder = application_support_folder;
  config_folder = application_support_folder;
  cache_folder = home_folder / "Library" / "Caches" / "ETX Tracer";
  tmp_folder = cache_folder / "tmp";
#else
  _env.bundled = false;
#endif

  set_folder_path(_env.data_folder, sizeof(_env.data_folder), data_folder);
  set_folder_path(_env.user_data_folder, sizeof(_env.user_data_folder), user_data_folder);
  set_folder_path(_env.config_folder, sizeof(_env.config_folder), config_folder);
  set_folder_path(_env.cache_folder, sizeof(_env.cache_folder), cache_folder);
  set_folder_path(_env.library_folder, sizeof(_env.library_folder), library_folder);
  set_folder_path(_env.tmp_folder, sizeof(_env.tmp_folder), tmp_folder);
  snprintf(_env.current_directory, sizeof(_env.current_directory), "%s", _env.data_folder);

  std::filesystem::create_directories(_env.user_data_folder, ec);
  std::filesystem::create_directories(_env.config_folder, ec);
  std::filesystem::create_directories(_env.cache_folder, ec);
  clear_tmp_folder();
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

void get_base_directory(const char* file_path, char* buffer, size_t buffer_size) {
  memset(buffer, 0, buffer_size);
  get_file_folder(file_path, buffer, buffer_size);
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

size_t get_file_size(FILE* f) {
  if (f == nullptr) {
    return 0;
  }

  if (ETX_FSEEK(f, 0, SEEK_END) != 0) {
    return 0;
  }
  ETX_FPOS_TYPE size = ETX_FTELL(f);
  if (ETX_FSEEK(f, 0, SEEK_SET) != 0) {
    return 0;
  }
  return (size > 0) ? static_cast<size_t>(size) : 0;
}

Environment& env() {
  return _env.e;
}

}  // namespace etx
