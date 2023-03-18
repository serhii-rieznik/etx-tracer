#include <etx/core/environment.hxx>
#include <etx/core/debug.hxx>

#include <string.h>
#include <stdio.h>

namespace etx {

static struct {
  char data_folder[2048] = {};
  Environment e;
} _env;

const char* Environment::data_folder() {
  return _env.data_folder;
}

const char* Environment::file_in_data(const char* f, char buffer[], uint64_t buffer_size) {
  snprintf(buffer, buffer_size, "%s%s", _env.data_folder, f);
  return buffer;
}

const char* Environment::file_in_data(const char* f) {
  static char buffer[2048] = {};
  return file_in_data(f, buffer, sizeof(buffer));
}

void Environment::setup(const char* executable_path) {
  get_file_folder(executable_path, _env.data_folder, sizeof(_env.data_folder));
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

#if defined(__MSC_VER)
  buffer[len] = '\\';
#else
  buffer[len] = '/';
#endif
  
  ETX_ASSERT(len + 1 < buffer_size);
  buffer[1llu + len] = 0;
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
