#pragma once

#include <etx/core/debug.hxx>
namespace etx {

struct Environment {
  const char* data_folder();
  const char* user_data_folder();
  const char* cache_folder();
  const char* library_folder();
  bool bundled() const;

  // thread save, uses extenal data storage
  const char* file_in_data(const char* f, char buffer[], uint64_t buffer_size);
  const char* file_in_user_data(const char* f, char buffer[], uint64_t buffer_size);
  const char* file_in_cache(const char* f, char buffer[], uint64_t buffer_size);
  const char* file_in_tmp(const char* f, char buffer[], uint64_t buffer_size);

  // not thread save, uses static data storage, use at your own risk
  const char* file_in_data(const char*);
  const char* file_in_user_data(const char*);
  const char* file_in_cache(const char*);
  const char* file_in_tmp(const char*);

  const char* tmp_folder();
  void clear_tmp_folder();

  void setup(const char* executable_path);
  const char* current_directory() const;
  std::string to_project_relative(const std::string& path) const;
  std::string resolve_to_absolute(const std::string& path) const;
};

uint64_t get_file_folder(const char* file_name, char buffer[], uint64_t buffer_size);
void get_base_directory(const char* file_path, char* buffer, size_t buffer_size);
const char* get_file_ext(const char* file_name);  // returns `ext` with dot
size_t get_file_size(FILE* f);

Environment& env();

}  // namespace etx
