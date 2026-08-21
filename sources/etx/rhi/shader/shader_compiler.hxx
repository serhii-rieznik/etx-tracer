#pragma once

#include <etx/core/core.hxx>
#include <etx/rhi/rhi_types.hxx>
namespace etx {

constexpr size_t MAX_SHADER_FILE_SIZE = 16 * 1024 * 1024;
constexpr size_t MAX_SHADER_SOURCE_SIZE = 16 * 1024 * 1024;

struct ShaderCompilationResult {
  RHIResult result = RHIResult::Success;
  std::vector<uint8_t> spirv_data;
  RHIMetalShaderMetadata metal_metadata = {};
  uint32_t local_size_x = 1u;
  uint32_t local_size_y = 1u;
  uint32_t local_size_z = 1u;
  std::string error_message;
  std::string warning_message;
  uint32_t error_line = 0;
  uint32_t error_column = 0;
};

struct ShaderReflectionInfo {
  struct ResourceBinding {
    std::string name;
    uint32_t binding_index = 0;
    uint32_t set_index = 0;
    RHIResourceType resource_type = RHIResourceType::Buffer;
    bool is_bindless = false;
  };

  std::vector<ResourceBinding> resources;

  uint32_t local_size_x = 1;
  uint32_t local_size_y = 1;
  uint32_t local_size_z = 1;

  bool supports_atomics = false;
  bool supports_subgroups = false;
};

struct ShaderCompilerStatistics {
  uint64_t compile_calls = 0;
  uint64_t requested_entry_points = 0;
  uint64_t compiled_entry_points = 0;
  uint64_t preprocessed_memory_cache_hits = 0;
  uint64_t preprocessed_disk_cache_hits = 0;
  uint64_t shader_memory_cache_hits = 0;
  uint64_t shader_disk_cache_hits = 0;
  uint64_t preprocess_invocations = 0;
  uint64_t dxc_compile_invocations = 0;
  uint64_t spirv_to_msl_translations = 0;
  uint64_t cache_writes = 0;
  double total_wall_time_ms = 0.0;
  double preprocess_time_ms = 0.0;
  double dxc_compile_time_ms = 0.0;
  double spirv_to_msl_time_ms = 0.0;
  double cache_read_time_ms = 0.0;
  double cache_write_time_ms = 0.0;
};

struct ShaderCompiler {
  // Deleted constructors - enforce singleton usage
  ShaderCompiler(const ShaderCompiler&) = delete;
  ShaderCompiler& operator=(const ShaderCompiler&) = delete;
  ShaderCompiler(ShaderCompiler&&) = delete;
  ShaderCompiler& operator=(ShaderCompiler&&) = delete;

  // Singleton access
  static ShaderCompiler& instance();
  void shutdown();

  RHIResult initialize();
  bool is_initialized() const;

  struct ShaderEntryPoint {
    std::string entry_point;
    RHIShaderStage stage;
  };

  struct MultiShaderCompilationResult {
    RHIResult result = RHIResult::Success;
    std::vector<uint8_t> shared_blob;
    std::vector<RHIShaderBinary> binaries;
    std::string error_message;
  };

  MultiShaderCompilationResult compile(const std::string& hlsl_source, const std::string& source_name, const std::vector<ShaderEntryPoint>& entry_points,
    const std::unordered_map<std::string, std::string>& defines = {}, RHIBackend backend = RHIBackend::Vulkan);

  MultiShaderCompilationResult compile(const std::string& filename, const std::vector<ShaderEntryPoint>& entry_points,
    const std::unordered_map<std::string, std::string>& defines = {}, RHIBackend backend = RHIBackend::Vulkan);

  std::string read_file_content(const std::string& file_path, std::string& error_message);

  ShaderCompilerStatistics statistics() const;
  void reset_statistics();
  void log_statistics(const char* label = nullptr) const;

  void parse_dxc_error(const std::string& error_message, ShaderCompilationResult& result);
  static std::string get_error_description(RHIResult result);

 private:
  ShaderCompiler();
  ~ShaderCompiler();

  struct Impl;                  // Forward declaration - PIMPL idiom
  std::unique_ptr<Impl> _impl;  // PIMPL pointer for instance members
};

}  // namespace etx
