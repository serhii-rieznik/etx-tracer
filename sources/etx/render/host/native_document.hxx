#pragma once

#include <filesystem>
#include <string>

namespace etx {

bool validate_native_document(const std::filesystem::path& path, std::string& error);

}  // namespace etx
