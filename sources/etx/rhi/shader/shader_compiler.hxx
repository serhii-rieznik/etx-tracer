#pragma once

#include <etx/core/core.hxx>
#include <etx/rhi/rhi_types.hxx>
namespace etx {

constexpr size_t MAX_SHADER_FILE_SIZE = 10 * 1024 * 1024;
constexpr size_t MAX_SHADER_SOURCE_SIZE = 5 * 1024 * 1024;

struct ShaderCompilationResult {
  RHIResult result = RHIResult::Success;
  std::vector<uint8_t> spirv_data;
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

struct ShaderCompiler {
  // Deleted constructors - enforce singleton usage
  ShaderCompiler(const ShaderCompiler&) = delete;
  ShaderCompiler& operator=(const ShaderCompiler&) = delete;
  ShaderCompiler(ShaderCompiler&&) = delete;
  ShaderCompiler& operator=(ShaderCompiler&&) = delete;

  // Singleton access
  static ShaderCompiler& instance();
  static void shutdown();

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
    const std::unordered_map<std::string, std::string>& defines = {});

  MultiShaderCompilationResult compile(const std::string& filename, const std::vector<ShaderEntryPoint>& entry_points,
    const std::unordered_map<std::string, std::string>& defines = {});

  std::string read_file_content(const std::string& file_path, std::string& error_message);

  void parse_dxc_error(const std::string& error_message, ShaderCompilationResult& result);
  static std::string get_error_description(RHIResult result);

 private:
  ShaderCompiler();
  ~ShaderCompiler();

  struct Impl;                  // Forward declaration - PIMPL idiom
  std::unique_ptr<Impl> _impl;  // PIMPL pointer for instance members
};

}  // namespace etx
