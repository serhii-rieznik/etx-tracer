#pragma once

#include <filesystem>
#include <string>

namespace etx {

bool is_pbrt_gzip_file(const std::filesystem::path& file);
bool is_pbrt_scene_file(const std::filesystem::path& file);
std::string read_pbrt_gzip(const std::filesystem::path& file);

}  // namespace etx
