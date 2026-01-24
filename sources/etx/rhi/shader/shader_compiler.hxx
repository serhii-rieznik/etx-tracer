#pragma once

#include <etx/core/core.hxx>
#include <etx/rhi/rhi_types.hxx>

#include <string>
#include <vector>
#include <unordered_map>
#include <map>
#include <functional>
#include <mutex>

#if !defined(WIN32_LEAN_AND_MEAN)
# define WIN32_LEAN_AND_MEAN
#endif

#include <windows.h>

#include <objbase.h>
#include <combaseapi.h>

#include <dxc/dxcapi.h>
#include <wrl.h>
#include <wrl.h>

namespace etx {

constexpr size_t MAX_SHADER_FILE_SIZE = 10 * 1024 * 1024;
constexpr size_t MAX_SHADER_SOURCE_SIZE = 5 * 1024 * 1024;

class CustomIncludeHandler : public IDxcIncludeHandler {
 public:
  CustomIncludeHandler(Microsoft::WRL::ComPtr<IDxcUtils> dxc_utils, const std::vector<std::string>& include_paths);
  ~CustomIncludeHandler();

  HRESULT STDMETHODCALLTYPE QueryInterface(REFIID iid, void** ppvObject) override;
  ULONG STDMETHODCALLTYPE AddRef() override;
  ULONG STDMETHODCALLTYPE Release() override;

  HRESULT STDMETHODCALLTYPE LoadSource(LPCWSTR pFilename, IDxcBlob** ppIncludeSource) override;

 private:
  Microsoft::WRL::ComPtr<IDxcUtils> _dxc_utils;
  std::vector<std::string> _include_paths;
  std::unordered_map<std::string, Microsoft::WRL::ComPtr<IDxcBlobEncoding>> _include_cache;
  std::mutex _include_cache_mutex;
  std::atomic<ULONG> _ref_count = 1;

  std::string find_include_file(const std::string& filename);
  std::string wstring_to_string(const std::wstring& wstr);
  std::wstring string_to_wstring(const std::string& str);
};

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

class ShaderCompiler {
 public:
  ShaderCompiler();
  ~ShaderCompiler();

  RHIResult initialize();
  bool is_initialized() const {
    return _dxc_utils != nullptr && _dxc_compiler != nullptr && _custom_include_handler != nullptr;
  }

  static RHIResult initialize_global();
  static void shutdown_global();
  static ShaderCompiler* get_global_instance();
  static bool is_global_initialized();

 private:
  static std::unique_ptr<ShaderCompiler> global_instance;
  static std::mutex global_init_mutex;
  static std::atomic<bool> global_initialized;
  static Microsoft::WRL::ComPtr<IDxcUtils> global_dxc_utils;
  static Microsoft::WRL::ComPtr<IDxcCompiler3> global_dxc_compiler;
  static CustomIncludeHandler* global_custom_include_handler;
  static HMODULE global_dxc_dll;
  static std::atomic<bool> global_com_initialized;
  static std::mutex global_dll_mutex;
  static DxcCreateInstanceProc global_dxc_create_instance;

 public:
  void add_include_path(const std::string& path);
  void clear_include_paths();

  struct ShaderVariantKey {
    std::string source_name;
    std::string entry_point;
    RHIShaderStage stage;
    std::map<std::string, std::string> defines;

    bool operator==(const ShaderVariantKey& other) const {
      return source_name == other.source_name && entry_point == other.entry_point && stage == other.stage && defines == other.defines;
    }
  };

  struct ShaderVariantKeyHash {
    std::size_t operator()(const ShaderVariantKey& key) const {
      std::size_t h = 0;
      h = etx_hash64_continue(key.source_name.data(), key.source_name.size(), h);
      h = etx_hash64_continue(key.entry_point.data(), key.entry_point.size(), h);
      h = etx_hash64_continue(&key.stage, sizeof(key.stage), h);
      for (const auto& define : key.defines) {
        h = etx_hash64_continue(define.first.data(), define.first.size(), h);
        h = etx_hash64_continue(define.second.data(), define.second.size(), h);
      }
      return h;
    }
  };

  ShaderCompilationResult get_or_compile_shader_variant(const std::string& hlsl_source, const std::string& entry_point, RHIShaderStage stage,
    const std::string& source_name = "shader.hlsl", const std::unordered_map<std::string, std::string>& defines = {});
  void clear_shader_cache();

  ShaderCompilationResult load_and_compile_shader_from_file(const std::string& file_path, const std::string& entry_point, RHIShaderStage stage,
    const std::unordered_map<std::string, std::string>& defines = {});
  std::string read_file_content(const std::string& file_path, std::string& error_message);

  ShaderCompilationResult compile_hlsl_to_spirv(const std::string& hlsl_source, const std::string& entry_point, RHIShaderStage stage,
    const std::string& source_name = "shader.hlsl", const std::unordered_map<std::string, std::string>& defines = {});

  ShaderCompilationResult compile_hlsl_with_multiple_entry_points(const std::string& hlsl_source, const std::vector<std::string>& entry_points, RHIShaderStage stage,
    const std::string& source_name = "shader.hlsl", const std::unordered_map<std::string, std::string>& defines = {});

  RHIResult reflect_spirv(const std::vector<uint8_t>& spirv_data, ShaderReflectionInfo& reflection_info);

  std::string generate_bindless_hlsl_wrapper(const std::string& user_shader_code, const ShaderReflectionInfo& reflection_info);

  void parse_dxc_error(const std::string& error_message, ShaderCompilationResult& result);
  static std::string get_error_description(RHIResult result);

 private:
  using DxcCreateInstanceProc = HRESULT(__stdcall*)(REFCLSID rclsid, REFIID riid, LPVOID* ppv);

  static RHIResult load_dxc_dll_global();
  static void unload_dxc_dll_global();
  static RHIResult initialize_dxc_interfaces_global();

  RHIResult initialize_global_instance();

  std::vector<std::string> _include_paths;
  std::unordered_map<ShaderVariantKey, ShaderCompilationResult, ShaderVariantKeyHash> _shader_cache;
  std::mutex _cache_mutex;

  Microsoft::WRL::ComPtr<IDxcUtils> _dxc_utils;
  Microsoft::WRL::ComPtr<IDxcCompiler3> _dxc_compiler;
  CustomIncludeHandler* _custom_include_handler = nullptr;

  RHIResult load_dxc_dll();
  void unload_dxc_dll();
  RHIResult initialize_dxc_interfaces();

  std::vector<std::wstring> build_dxc_arguments(const std::string& entry_point, RHIShaderStage stage, const std::unordered_map<std::string, std::string>& defines,
    bool for_preprocessing = false);
};

}  // namespace etx
