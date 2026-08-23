#pragma once

#include <etx/rhi/rhi_types.hxx>

#include <filesystem>
#include <string>

namespace etx {

struct RaytracerShaderPackageStatistics {
  uint32_t variant_count = 0u;
  uint64_t binary_size_bytes = 0u;
  uint64_t package_size_bytes = 0u;
  double compile_time_ms = 0.0;
  double package_time_ms = 0.0;
};

bool build_raytracer_shader_package(const std::filesystem::path& output_path, RHIBackend backend, RaytracerShaderPackageStatistics& statistics, std::string& error_message);

}  // namespace etx
