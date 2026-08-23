#pragma once

#include <etx/rhi/rhi_types.hxx>

#include <cstddef>
#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace etx {

struct ShaderPackageRequest {
  std::string source_name = {};
  std::string entry_point = {};
  RHIShaderStage stage = RHIShaderStage::Vertex;
  RHIBackend backend = RHIBackend::Vulkan;
  std::map<std::string, std::string> defines = {};
};

struct ShaderPackageBinary {
  std::vector<uint8_t> data = {};
  RHIShaderBinaryFormat format = RHIShaderBinaryFormat::SpirV;
  uint32_t local_size_x = 1u;
  uint32_t local_size_y = 1u;
  uint32_t local_size_z = 1u;
  RHIMetalShaderMetadata metal_metadata = {};
};

struct ShaderPackageBuildEntry {
  ShaderPackageRequest request = {};
  ShaderPackageBinary binary = {};
};

struct ShaderPackageBuildStatistics {
  uint32_t entry_count = 0u;
  uint64_t binary_size_bytes = 0u;
  uint64_t package_size_bytes = 0u;
};

struct ShaderPackage {
  ShaderPackage();
  ~ShaderPackage();

  ShaderPackage(const ShaderPackage&) = delete;
  ShaderPackage& operator=(const ShaderPackage&) = delete;
  ShaderPackage(ShaderPackage&&) noexcept;
  ShaderPackage& operator=(ShaderPackage&&) noexcept;

  bool load(const std::filesystem::path& path, RHIBackend backend, std::string& error_message);
  bool loaded() const;
  bool contains(const ShaderPackageRequest& request) const;
  bool read(const ShaderPackageRequest& request, ShaderPackageBinary& binary, std::string& error_message) const;
  uint32_t entry_count() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> _impl;
};

uint64_t shader_package_request_hash(const ShaderPackageRequest& request);
bool write_shader_package(const std::filesystem::path& path, std::vector<ShaderPackageBuildEntry> entries, ShaderPackageBuildStatistics& statistics, std::string& error_message);
bool compile_metal_shader_library(const void* source_data, size_t source_size, const std::filesystem::path& cache_directory, const std::string& minimum_macos_version,
  std::vector<uint8_t>& library, std::string& error_message);

}  // namespace etx
