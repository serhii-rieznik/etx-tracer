#pragma once

#include <etx/core/debug.hxx>

#include <string>

namespace etx {

struct Environment {
  const char* data_folder();

  // thread save, uses extenal data storage
  const char* file_in_data(const char* f, char buffer[], uint64_t buffer_size);

  // not thread save, uses static data storage, use at your own risk
  const char* file_in_data(const char*);

  void setup(const char* executable_path);
  const char* current_directory() const;
  std::string to_project_relative(const std::string& path) const;
  std::string resolve_to_absolute(const std::string& path) const;
};

uint64_t get_file_folder(const char* file_name, char buffer[], uint64_t buffer_size);
const char* get_file_ext(const char* file_name);  // returns `ext` with dot

Environment& env();

}  // namespace etx
