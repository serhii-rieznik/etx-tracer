#pragma once

#include <stdint.h>

namespace etx {

struct Environment {
  const char* data_folder();
  void setup(const char* executable_path);
};

uint64_t get_file_folder(const char* file_name, char buffer[], uint64_t buffer_size);
const char* get_file_ext(const char* file_name);  // returns `ext` with dot

Environment& env();

}  // namespace etx
