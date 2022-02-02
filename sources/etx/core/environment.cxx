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

void Environment::setup(const char* executable_path) {
  uint64_t pos = get_file_folder(executable_path, _env.data_folder, sizeof(_env.data_folder));
  snprintf(_env.data_folder + pos, sizeof(_env.data_folder) - pos, "data\\");
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
  buffer[len] = '\\';
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
