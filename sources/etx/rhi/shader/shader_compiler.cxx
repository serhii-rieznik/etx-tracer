#include <etx/rhi/shader/shader_compiler.hxx>
#include <etx/rhi/shader/dxc_com_ptr.hxx>

#include <etx/core/log.hxx>
#include <etx/core/platform.hxx>
#include <etx/core/environment.hxx>
#include <spirv_msl.hpp>
#include <array>
#include <codecvt>
#include <cstring>
#include <cstdio>
#include <fstream>
#include <locale>
#include <regex>
#include <sstream>
#include <string_view>
#include <vector>

#if (ETX_PLATFORM_WINDOWS)
# if !defined(WIN32_LEAN_AND_MEAN)
#  define WIN32_LEAN_AND_MEAN
# endif
# include <windows.h>
# include <unknwn.h>
# include <objidl.h>
# include <oleauto.h>
# include <combaseapi.h>
#else
# include <dlfcn.h>
#endif

#include <dxc/dxcapi.h>

namespace etx {

// Forward declarations for global helper functions
using DxcCreateInstanceProc = HRESULT(__stdcall*)(REFCLSID rclsid, REFIID riid, LPVOID* ppv);
RHIResult load_dxc_dll_global();
void unload_dxc_dll_global();
RHIResult initialize_dxc_interfaces_global();

namespace {

constexpr bool kEnableShaderDebugInfo = false;

#if (ETX_PLATFORM_WINDOWS)
using DxcLibraryHandle = HMODULE;
#else
using DxcLibraryHandle = void*;
#endif

DxcLibraryHandle load_dxc_library(const char* path) {
#if (ETX_PLATFORM_WINDOWS)
  return LoadLibraryA(path);
#else
  return dlopen(path, RTLD_NOW | RTLD_LOCAL);
#endif
}

void* load_dxc_symbol(DxcLibraryHandle library, const char* symbol_name) {
#if (ETX_PLATFORM_WINDOWS)
  return reinterpret_cast<void*>(GetProcAddress(library, symbol_name));
#else
  return dlsym(library, symbol_name);
#endif
}

void unload_dxc_library(DxcLibraryHandle library) {
#if (ETX_PLATFORM_WINDOWS)
  FreeLibrary(library);
#else
  dlclose(library);
#endif
}

std::wstring utf8_to_wstring(const std::string& value) {
  if (value.empty()) {
    return {};
  }

#if (ETX_PLATFORM_WINDOWS)
  int size_needed = MultiByteToWideChar(CP_UTF8, 0, value.c_str(), static_cast<int>(value.size()), nullptr, 0);
  std::wstring wstr(size_needed, 0);
  MultiByteToWideChar(CP_UTF8, 0, value.c_str(), static_cast<int>(value.size()), wstr.data(), size_needed);
  return wstr;
#else
  try {
    std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
    return converter.from_bytes(value);
  } catch (const std::exception&) {
    std::wstring fallback;
    fallback.reserve(value.size());
    for (unsigned char ch : value) {
      fallback.push_back(static_cast<wchar_t>(ch));
    }
    return fallback;
  }
#endif
}

std::string wstring_to_utf8(const std::wstring& value) {
  if (value.empty()) {
    return {};
  }

#if (ETX_PLATFORM_WINDOWS)
  int size_needed = WideCharToMultiByte(CP_UTF8, 0, value.c_str(), static_cast<int>(value.size()), nullptr, 0, nullptr, nullptr);
  std::string str(size_needed, 0);
  WideCharToMultiByte(CP_UTF8, 0, value.c_str(), static_cast<int>(value.size()), str.data(), size_needed, nullptr, nullptr);
  return str;
#else
  try {
    std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
    return converter.to_bytes(value);
  } catch (const std::exception&) {
    std::string fallback;
    fallback.reserve(value.size());
    for (wchar_t ch : value) {
      fallback.push_back((ch >= 0 && ch <= 0x7F) ? static_cast<char>(ch) : '?');
    }
    return fallback;
  }
#endif
}

constexpr std::array<std::string_view, 6> default_shader_search_paths = {
  "./sources/etx/render",
  "../sources/etx/render",
  "../../sources/etx/render",
  "./sources",
  "../sources",
  "../../sources",
};

void append_unique_existing_directory(std::vector<std::string>& directories, const std::filesystem::path& input_path) {
  if (input_path.empty()) {
    return;
  }

  std::error_code ec;
  const std::filesystem::path normalized_path = input_path.lexically_normal();
  if ((std::filesystem::exists(normalized_path, ec) == false) || (ec.value() != 0)) {
    return;
  }

  const std::filesystem::path absolute_path = std::filesystem::absolute(normalized_path, ec);
  const std::string directory = (ec.value() == 0) ? absolute_path.string() : normalized_path.string();
  if (directory.empty()) {
    return;
  }

  for (const auto& existing_directory : directories) {
    if (existing_directory == directory) {
      return;
    }
  }

  directories.push_back(directory);
}

std::vector<std::string> build_shader_include_directories(const std::string& source_name) {
  std::vector<std::string> include_directories = {};

  if (source_name.empty() == false) {
    std::filesystem::path source_path(source_name);
    if (source_path.has_parent_path()) {
      std::filesystem::path root = source_path.parent_path();
      for (uint32_t depth = 0; (depth < 8u) && (root.empty() == false); ++depth) {
        append_unique_existing_directory(include_directories, root);

        const std::filesystem::path parent = root.parent_path();
        if (parent.empty() || (parent == root)) {
          break;
        }
        root = parent;
      }
    }
  }

  for (const auto path : default_shader_search_paths) {
    append_unique_existing_directory(include_directories, std::filesystem::path(std::string(path)));
  }

  return include_directories;
}

bool translate_spirv_to_msl(const std::vector<uint8_t>& spirv_data, std::string& out_msl, std::string& error_message) {
  if ((spirv_data.empty()) || ((spirv_data.size() % sizeof(uint32_t)) != 0u)) {
    error_message = "SPIR-V payload is empty or not aligned to 32-bit words.";
    return false;
  }

  std::vector<uint32_t> spirv_words(spirv_data.size() / sizeof(uint32_t));
  std::memcpy(spirv_words.data(), spirv_data.data(), spirv_data.size());

  try {
    spirv_cross::CompilerMSL compiler(std::move(spirv_words));
    spirv_cross::CompilerMSL::Options options = compiler.get_msl_options();
    options.platform = spirv_cross::CompilerMSL::Options::macOS;
    options.msl_version = spirv_cross::CompilerMSL::Options::make_msl_version(3, 0);
    options.argument_buffers = true;
    // The Metal backend relies on runtime-sized bindless descriptor arrays, which require tier 2.
    options.argument_buffers_tier = spirv_cross::CompilerMSL::Options::ArgumentBuffersTier::Tier2;
    compiler.set_msl_options(options);
    compiler.add_discrete_descriptor_set(0u);

    out_msl = compiler.compile();
  } catch (const spirv_cross::CompilerError& error) {
    error_message = error.what();
    return false;
  } catch (const std::exception& error) {
    error_message = error.what();
    return false;
  }

  if (out_msl.empty()) {
    error_message = "Embedded SPIRV-Cross produced empty MSL output.";
    return false;
  }

  return true;
}

bool translated_msl_has_unsafe_overlapping_bindless(const std::vector<uint8_t>& spirv_data, std::string& out_error_message) {
  std::string msl_source = {};
  if (translate_spirv_to_msl(spirv_data, msl_source, out_error_message) == false) {
    return false;
  }

  if ((msl_source.find("ETX_METAL_UNSUPPORTED_OVERLAPPING_BINDLESS") != std::string::npos) ||
      (msl_source.find("Overlapping binding:") != std::string::npos)) {
    out_error_message =
      "Shader translation produced overlapping bindless descriptor layouts that are unsafe on macOS GPU backends.";
    return true;
  }

  out_error_message.clear();
  return false;
}

std::string resolve_shader_file_path(const std::string& filename) {
  if (filename.empty()) {
    return {};
  }

  std::error_code ec;
  std::filesystem::path input_path(filename);
  if (std::filesystem::exists(input_path, ec) && (ec.value() == 0)) {
    auto abs_path = std::filesystem::absolute(input_path, ec);
    return ec.value() == 0 ? abs_path.string() : input_path.string();
  }

  for (const auto path : default_shader_search_paths) {
    if (path.empty()) {
      continue;
    }

    std::filesystem::path candidate = std::filesystem::path(std::string(path)) / filename;
    ec = {};
    if (std::filesystem::exists(candidate, ec) && (ec.value() == 0)) {
      auto abs_path = std::filesystem::absolute(candidate, ec);
      return ec.value() == 0 ? abs_path.string() : candidate.string();
    }
  }

  return {};
}

bool contains_include_directive(const std::string& source) {
  size_t pos = 0;
  while (pos < source.size()) {
    size_t line_end = source.find('\n', pos);
    if (line_end == std::string::npos) {
      line_end = source.size();
    }

    size_t i = pos;
    while (i < line_end && (source[i] == ' ' || source[i] == '\t' || source[i] == '\r')) {
      ++i;
    }

    if (i < line_end && source[i] == '#') {
      ++i;
      while (i < line_end && (source[i] == ' ' || source[i] == '\t')) {
        ++i;
      }

      static constexpr std::string_view include_kw = "include";
      if ((line_end - i) >= include_kw.size() && source.compare(i, include_kw.size(), include_kw.data()) == 0) {
        return true;
      }
    }

    pos = line_end + 1;
  }

  return false;
}

void extract_compute_local_size(const std::string& source, const std::string& entry_point, RHIShaderStage stage, uint32_t& out_x, uint32_t& out_y, uint32_t& out_z) {
  out_x = 1;
  out_y = 1;
  out_z = 1;

  if (stage != RHIShaderStage::Compute) {
    return;
  }

  const std::string escaped_entry = std::regex_replace(entry_point, std::regex(R"([.^$|()\\[\]{}*+?])"), R"(\\$&)");
  const std::regex entry_regex("\\[\\s*numthreads\\s*\\(\\s*(\\d+)\\s*,\\s*(\\d+)\\s*,\\s*(\\d+)\\s*\\)\\s*\\][^\\n\\r]*?[A-Za-z_][A-Za-z0-9_<>\\s]*\\b" + escaped_entry +
                                 "\\s*\\(",
    std::regex::ECMAScript);
  std::smatch match = {};
  if (std::regex_search(source, match, entry_regex) && (match.size() == 4u)) {
    out_x = static_cast<uint32_t>(std::stoul(match[1].str()));
    out_y = static_cast<uint32_t>(std::stoul(match[2].str()));
    out_z = static_cast<uint32_t>(std::stoul(match[3].str()));
  }
}

}  // namespace

class CustomIncludeHandler : public IDxcIncludeHandler {
 public:
  CustomIncludeHandler(DxcComPtr<IDxcUtils> dxc_utils, std::string shader_directory);
  ~CustomIncludeHandler();

  HRESULT STDMETHODCALLTYPE QueryInterface(REFIID iid, void** ppvObject) override;
  ULONG STDMETHODCALLTYPE AddRef() override;
  ULONG STDMETHODCALLTYPE Release() override;

  HRESULT STDMETHODCALLTYPE LoadSource(LPCWSTR pFilename, IDxcBlob** ppIncludeSource) override;

 private:
  DxcComPtr<IDxcUtils> _dxc_utils;
  std::string _shader_directory;
  std::unordered_map<std::string, DxcComPtr<IDxcBlobEncoding>> _include_cache;
  std::mutex _include_cache_mutex;
  std::atomic<ULONG> _ref_count = 1;

  std::string find_include_file(const std::string& filename);
  std::string wstring_to_string(const std::wstring& wstr);
};

struct ShaderVariantKey {
  std::string source_name;
  std::string entry_point;
  RHIShaderStage stage;
  RHIBackend backend = RHIBackend::Vulkan;
  std::map<std::string, std::string> defines;
  uint64_t source_hash = 0;

  bool operator==(const ShaderVariantKey& other) const {
    return source_name == other.source_name && entry_point == other.entry_point && stage == other.stage && backend == other.backend && defines == other.defines &&
           source_hash == other.source_hash;
  }
};

struct ShaderVariantKeyHash {
  std::size_t operator()(const ShaderVariantKey& key) const {
    std::size_t h = 0;
    h = etx_hash64_continue(key.source_name.data(), key.source_name.size(), h);
    h = etx_hash64_continue(key.entry_point.data(), key.entry_point.size(), h);
    h = etx_hash64_continue(&key.stage, sizeof(key.stage), h);
    h = etx_hash64_continue(&key.backend, sizeof(key.backend), h);
    for (const auto& define : key.defines) {
      h = etx_hash64_continue(define.first.data(), define.first.size(), h);
      h = etx_hash64_continue(define.second.data(), define.second.size(), h);
    }
    h = etx_hash64_continue(&key.source_hash, sizeof(key.source_hash), h);
    return h;
  }
};

struct ShaderCompiler::Impl {
  std::unordered_map<ShaderVariantKey, ShaderCompilationResult, ShaderVariantKeyHash> shader_cache;
  std::mutex cache_mutex;

  DxcComPtr<IDxcUtils> dxc_utils;
  DxcComPtr<IDxcCompiler3> dxc_compiler;

  // Helper method
  std::vector<std::wstring> build_dxc_arguments(const std::string& entry_point, RHIShaderStage stage, const std::unordered_map<std::string, std::string>& defines,
    const std::vector<std::string>& include_directories, bool for_preprocessing = false);
};

// File-scope global variables for DXC
#if ETX_PLATFORM_APPLE
std::mutex& global_init_mutex = *new std::mutex();
DxcComPtr<IDxcUtils>& global_dxc_utils = *new DxcComPtr<IDxcUtils>();
DxcComPtr<IDxcCompiler3>& global_dxc_compiler = *new DxcComPtr<IDxcCompiler3>();
DxcLibraryHandle& global_dxc_dll = *new DxcLibraryHandle(nullptr);
std::mutex& global_dll_mutex = *new std::mutex();
DxcCreateInstanceProc& global_dxc_create_instance = *new DxcCreateInstanceProc(nullptr);
#else
std::mutex global_init_mutex;
DxcComPtr<IDxcUtils> global_dxc_utils;
DxcComPtr<IDxcCompiler3> global_dxc_compiler;
DxcLibraryHandle global_dxc_dll = nullptr;
#if (ETX_PLATFORM_WINDOWS)
std::atomic<bool> global_com_initialized{false};
#endif
std::mutex global_dll_mutex;
DxcCreateInstanceProc global_dxc_create_instance = nullptr;
#endif

// Singleton implementation - Meyer's singleton with thread-safe initialization
ShaderCompiler& ShaderCompiler::instance() {
  static ShaderCompiler singleton;
  static std::once_flag init_flag;
  static bool init_failed = false;

  std::call_once(init_flag, []() {
    // Initialize global DXC resources
    RHIResult dll_result = load_dxc_dll_global();
    if (dll_result != RHIResult::Success) {
      log::error("Failed to load DXC DLL");
      init_failed = true;
      return;
    }

#if (ETX_PLATFORM_WINDOWS)
    // DXC uses COM on Windows only.
    HRESULT hr = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);
    if (FAILED(hr) && (hr != RPC_E_CHANGED_MODE)) {
      log::error("Failed to initialize COM: 0x%08X", static_cast<uint32_t>(hr));
      unload_dxc_dll_global();
      init_failed = true;
      return;
    }

    if ((hr != RPC_E_CHANGED_MODE)) {
      global_com_initialized.store(true, std::memory_order_release);
    }
#endif

    // Initialize DXC interfaces
    RHIResult init_result = initialize_dxc_interfaces_global();
    if (init_result != RHIResult::Success) {
      log::error("Failed to initialize DXC interfaces");
#if (ETX_PLATFORM_WINDOWS)
      if (global_com_initialized.load(std::memory_order_acquire)) {
        CoUninitialize();
        global_com_initialized.store(false, std::memory_order_release);
      }
#endif
      unload_dxc_dll_global();
      init_failed = true;
      return;
    }

    // Initialize the singleton instance
    RHIResult instance_result = singleton.initialize();
    if (instance_result != RHIResult::Success) {
      log::error("Failed to initialize shader compiler instance");
      init_failed = true;
    }
  });

  if (init_failed) {
    log::error("ShaderCompiler initialization failed");
  }

  return singleton;
}

void ShaderCompiler::shutdown() {
#if ETX_PLATFORM_APPLE
  if (_impl != nullptr) {
    std::lock_guard<std::mutex> cache_lock(_impl->cache_mutex);
    _impl->shader_cache.clear();
  }
  return;
#endif

  std::lock_guard<std::mutex> dll_lock(global_dll_mutex);

  if (_impl != nullptr) {
    std::lock_guard<std::mutex> cache_lock(_impl->cache_mutex);
    _impl->shader_cache.clear();
    _impl->dxc_utils.Reset();
    _impl->dxc_compiler.Reset();
  }

  global_dxc_utils.Reset();
  global_dxc_compiler.Reset();

#if (ETX_PLATFORM_WINDOWS)
  if (global_com_initialized.load(std::memory_order_acquire)) {
    CoUninitialize();
    global_com_initialized.store(false, std::memory_order_release);
  }
#endif
  if (global_dxc_dll) {
#if ETX_PLATFORM_APPLE
    // DXC keeps thread-local state alive through process teardown on macOS.
    // Keeping the dylib loaded avoids exit-time crashes in libdxcompiler.
#else
    unload_dxc_library(global_dxc_dll);
    global_dxc_dll = nullptr;
#endif
  }
  if (global_dxc_dll == nullptr) {
    global_dxc_create_instance = nullptr;
  }
}

ShaderCompiler::ShaderCompiler()
  : _impl(std::make_unique<Impl>()) {
}

ShaderCompiler::~ShaderCompiler() {
  shutdown();
  if (_impl != nullptr) {
    _impl->shader_cache.clear();
  }
}

RHIResult ShaderCompiler::initialize() {
  if (is_initialized()) {
    return RHIResult::Success;
  }

  _impl->dxc_utils = global_dxc_utils;
  _impl->dxc_compiler = global_dxc_compiler;

  return RHIResult::Success;
}

bool ShaderCompiler::is_initialized() const {
  return _impl && _impl->dxc_utils && _impl->dxc_compiler;
}

// File-loading overload
ShaderCompiler::MultiShaderCompilationResult ShaderCompiler::compile(const std::string& filename, const std::vector<ShaderEntryPoint>& entry_points,
  const std::unordered_map<std::string, std::string>& defines, RHIBackend backend) {
  std::string source_path = resolve_shader_file_path(filename);
  if (source_path.empty()) {
    source_path = filename;
  }

  std::string error;
  std::string source = read_file_content(source_path, error);

  if (error.empty() == false) {
    std::string details = "Failed to read shader file '" + filename + "'";
    if (source_path != filename) {
      details += " (resolved to '" + source_path + "')";
    }

    MultiShaderCompilationResult result = {
      .result = RHIResult::ValidationError,
      .error_message = details + ": " + error,
    };
    return result;
  }

  return compile(source, source_path, entry_points, defines, backend);
}

ShaderCompiler::MultiShaderCompilationResult ShaderCompiler::compile(const std::string& hlsl_source, const std::string& source_name,
  const std::vector<ShaderEntryPoint>& entry_points, const std::unordered_map<std::string, std::string>& defines, RHIBackend backend) {
  MultiShaderCompilationResult result = {};

  if (hlsl_source.empty()) {
    result.result = RHIResult::InvalidArgument;
    result.error_message = "HLSL source code is empty";
    return result;
  }

  if (entry_points.empty()) {
    result.result = RHIResult::InvalidArgument;
    result.error_message = "No entry points provided";
    return result;
  }

  if (hlsl_source.size() > MAX_SHADER_SOURCE_SIZE) {
    result.result = RHIResult::InvalidArgument;
    result.error_message = "HLSL source exceeds max size: " + std::to_string(hlsl_source.size()) + " bytes (max: " + std::to_string(MAX_SHADER_SOURCE_SIZE) + " bytes)";
    return result;
  }

  if (is_initialized() == false) {
    result.result = RHIResult::InvalidArgument;
    result.error_message = "Shader compiler not initialized";
    return result;
  }

  std::string shader_directory;
  if (source_name.empty() == false) {
    std::filesystem::path source_path(source_name);
    if (source_path.has_parent_path()) {
      shader_directory = source_path.parent_path().string();
    }
  }

  auto include_handler =
    std::unique_ptr<CustomIncludeHandler, void (*)(CustomIncludeHandler*)>(new CustomIncludeHandler(_impl->dxc_utils, shader_directory), [](CustomIncludeHandler* p) {
      if (p != nullptr) {
        p->Release();
      }
    });
  const std::vector<std::string> include_directories = build_shader_include_directories(source_name);

  if (include_handler == nullptr) {
    result.result = RHIResult::OutOfMemory;
    result.error_message = "Failed to create include handler";
    return result;
  }

  // Preprocess once
  std::string preprocessed_source;
  {
    DxcComPtr<IDxcBlobEncoding> source_blob;
    HRESULT hr = _impl->dxc_utils->CreateBlob(hlsl_source.data(), static_cast<uint32_t>(hlsl_source.size()), DXC_CP_UTF8, source_blob.ReleaseAndGetAddressOf());
    if (FAILED(hr)) {
      result.result = RHIResult::ValidationError;
      result.error_message = "Failed to create DXC source blob";
      return result;
    }

    DxcBuffer dxc_buffer = {
      .Ptr = source_blob->GetBufferPointer(),
      .Size = source_blob->GetBufferSize(),
      .Encoding = DXC_CP_UTF8,
    };

    // Preprocess arguments (entry point doesn't matter for preprocessing usually, but we need one)
    auto preprocess_args = _impl->build_dxc_arguments(entry_points[0].entry_point, entry_points[0].stage, defines, include_directories, true);
    std::vector<const wchar_t*> preprocess_args_ptr;
    for (const auto& arg : preprocess_args) {
      preprocess_args_ptr.push_back(arg.c_str());
    }

    DxcComPtr<IDxcResult> preprocess_result;
    preprocess_result.Reset();
    hr = _impl->dxc_compiler->Compile(&dxc_buffer, preprocess_args_ptr.data(), static_cast<uint32_t>(preprocess_args_ptr.size()), include_handler.get(),
      IID_PPV_ARGS(preprocess_result.GetAddressOf()));
    if (FAILED(hr)) {
      result.result = RHIResult::ValidationError;
      result.error_message = "Preprocessing compilation failed";
      return result;
    }

    HRESULT preprocess_status = S_OK;
    hr = preprocess_result->GetStatus(&preprocess_status);
    if (FAILED(hr)) {
      result.result = RHIResult::ValidationError;
      result.error_message = "Failed to get preprocessing status";
      return result;
    }

    std::string preprocess_diagnostics;
    {
      DxcComPtr<IDxcBlobUtf8> preprocess_errors;
      preprocess_errors.Reset();
      hr = preprocess_result->GetOutput(DXC_OUT_ERRORS, IID_PPV_ARGS(preprocess_errors.GetAddressOf()), nullptr);
      if (SUCCEEDED(hr) && preprocess_errors && preprocess_errors->GetStringLength() > 0) {
        preprocess_diagnostics.assign(preprocess_errors->GetStringPointer(), preprocess_errors->GetStringLength());
      }
    }

    if (preprocess_diagnostics.empty() == false) {
      log::warning("Shader preprocessing diagnostics [%s]:\n%s", source_name.empty() ? "<memory>" : source_name.c_str(), preprocess_diagnostics.c_str());
    }

    if (FAILED(preprocess_status) && (preprocess_diagnostics.empty() == false)) {
      result.result = RHIResult::ValidationError;
      result.error_message = preprocess_diagnostics;
      return result;
    }

    DxcComPtr<IDxcBlobUtf8> canonical_hlsl_blob;
    canonical_hlsl_blob.Reset();
    hr = preprocess_result->GetOutput(DXC_OUT_HLSL, IID_PPV_ARGS(canonical_hlsl_blob.GetAddressOf()), nullptr);
    if (FAILED(hr)) {
      result.result = RHIResult::ValidationError;
      result.error_message = "Failed to get preprocessed HLSL";
      return result;
    }

    if (!canonical_hlsl_blob || canonical_hlsl_blob->GetStringLength() == 0) {
      result.result = RHIResult::ValidationError;
      result.error_message = "Failed to get preprocessed HLSL";
      return result;
    }
    preprocessed_source.assign(canonical_hlsl_blob->GetStringPointer(), canonical_hlsl_blob->GetStringLength());

    if (preprocessed_source.size() > MAX_SHADER_SOURCE_SIZE) {
      result.result = RHIResult::ValidationError;
      result.error_message =
        "Preprocessed HLSL exceeds max size: " + std::to_string(preprocessed_source.size()) + " bytes (max: " + std::to_string(MAX_SHADER_SOURCE_SIZE) + " bytes)";
      return result;
    }
  }

  if (contains_include_directive(preprocessed_source)) {
    log::warning("Preprocessed shader source still contains #include directives [%s]", source_name.empty() ? "<memory>" : source_name.c_str());
  }

  uint64_t source_hash = etx_hash64(preprocessed_source.data(), preprocessed_source.size());
  std::map<std::string, std::string> ordered_defines(defines.begin(), defines.end());

  result.binaries.resize(entry_points.size());

  for (uint32_t i = 0, e = entry_points.size(); i < e; ++i) {
    const auto& ep = entry_points[i];
    ShaderVariantKey key{source_name, ep.entry_point, ep.stage, backend, ordered_defines, source_hash};
    uint32_t local_size_x = 1;
    uint32_t local_size_y = 1;
    uint32_t local_size_z = 1;
    extract_compute_local_size(preprocessed_source, ep.entry_point, ep.stage, local_size_x, local_size_y, local_size_z);

    // Check cache
    std::vector<uint8_t> cached_spirv;
    {
      std::lock_guard<std::mutex> lock(_impl->cache_mutex);
      auto it = _impl->shader_cache.find(key);
      if (it != _impl->shader_cache.end() && it->second.result == RHIResult::Success) {
        cached_spirv = it->second.spirv_data;
      }
    }

    if (cached_spirv.empty() == false) {
      result.binaries[i].spirv_data = nullptr;  // Fix up pointers later
      result.binaries[i].spirv_size = cached_spirv.size();
      result.binaries[i].stage = ep.stage;
      result.binaries[i].backend = backend;
      result.binaries[i].format = (backend == RHIBackend::Metal) ? RHIShaderBinaryFormat::MetalSource : RHIShaderBinaryFormat::SpirV;
      result.binaries[i].entry_point = ep.entry_point;
      result.binaries[i].local_size_x = local_size_x;
      result.binaries[i].local_size_y = local_size_y;
      result.binaries[i].local_size_z = local_size_z;
      result.shared_blob.insert(result.shared_blob.end(), cached_spirv.begin(), cached_spirv.end());
      continue;
    }

    std::vector<std::wstring> arguments = _impl->build_dxc_arguments(ep.entry_point, ep.stage, defines, include_directories, false);
    if (arguments.empty()) {
      result.result = RHIResult::InvalidArgument;
      result.error_message = "Unsupported shader stage for DXC compilation";
      return result;
    }

    std::vector<const wchar_t*> arguments_ptr;
    arguments_ptr.reserve(arguments.size());
    for (const auto& arg : arguments) {
      arguments_ptr.push_back(arg.c_str());
    }

    DxcBuffer source_buffer = {
      .Ptr = preprocessed_source.data(),
      .Size = preprocessed_source.size(),
      .Encoding = DXC_CP_UTF8,
    };

    DxcComPtr<IDxcResult> compile_result = {};
    HRESULT hr = _impl->dxc_compiler->Compile(&source_buffer, arguments_ptr.data(), (uint32_t)arguments_ptr.size(), nullptr, IID_PPV_ARGS(compile_result.GetAddressOf()));

    ShaderCompilationResult entry_result = {
      .result = RHIResult::Success,
    };

    if (FAILED(hr) || !compile_result) {
      entry_result.result = RHIResult::ValidationError;
      entry_result.error_message = "DXC Compile failed";
    } else {
      HRESULT compile_status = S_OK;
      if (FAILED(compile_result->GetStatus(&compile_status))) {
        entry_result.result = RHIResult::ValidationError;
        entry_result.error_message = "Failed to get DXC compile status";
      } else if (FAILED(compile_status)) {
        entry_result.result = RHIResult::ValidationError;
      }
      DxcComPtr<IDxcBlobUtf8> errors;
      compile_result->GetOutput(DXC_OUT_ERRORS, IID_PPV_ARGS(errors.GetAddressOf()), nullptr);
      if (errors && errors->GetStringLength() > 0) {
        entry_result.error_message = errors->GetStringPointer();
        if (entry_result.result == RHIResult::Success) {
          // Treat warnings as non-fatal if result was success, but we log them/store them?
          // For now, if we have a failure status, we assume errors contains the error.
        }
      }
    }

    if (entry_result.result != RHIResult::Success) {
      result.result = entry_result.result;
      result.error_message = entry_result.error_message;
      return result;
    }

    DxcComPtr<IDxcBlob> shader_obj;
    HRESULT object_hr = compile_result->GetOutput(DXC_OUT_OBJECT, IID_PPV_ARGS(shader_obj.GetAddressOf()), nullptr);
    if (FAILED(object_hr) || !shader_obj || shader_obj->GetBufferPointer() == nullptr || shader_obj->GetBufferSize() == 0) {
      result.result = RHIResult::ValidationError;
      result.error_message = entry_result.error_message.empty() ? "DXC did not produce shader object output" : entry_result.error_message;
      return result;
    }

    const uint8_t* ptr = static_cast<const uint8_t*>(shader_obj->GetBufferPointer());
    size_t size = shader_obj->GetBufferSize();
    std::vector<uint8_t> final_binary = {};
    RHIShaderBinaryFormat binary_format = RHIShaderBinaryFormat::SpirV;

    std::vector<uint8_t> spirv_binary = {};
    if ((backend == RHIBackend::Metal)
#if ETX_PLATFORM_APPLE
        || (backend == RHIBackend::Vulkan)
#endif
    ) {
      spirv_binary.assign(ptr, ptr + size);
    }

#if ETX_PLATFORM_APPLE
    if (backend == RHIBackend::Vulkan) {
      std::string preflight_error = {};
      if (translated_msl_has_unsafe_overlapping_bindless(spirv_binary, preflight_error)) {
        result.result = RHIResult::ValidationError;
        result.error_message = preflight_error;
        return result;
      }
      if (preflight_error.empty() == false) {
        log::warning("Skipping Vulkan-on-macOS shader translation safety preflight for '%s': %s", ep.entry_point.c_str(), preflight_error.c_str());
      }
    }
#endif

    if (backend == RHIBackend::Metal) {
      std::string msl_source = {};
      std::string translation_error = {};
      if (translate_spirv_to_msl(spirv_binary, msl_source, translation_error) == false) {
        result.result = RHIResult::ValidationError;
        result.error_message = translation_error;
        return result;
      }

      final_binary.assign(msl_source.begin(), msl_source.end());
      binary_format = RHIShaderBinaryFormat::MetalSource;
      ptr = final_binary.data();
      size = final_binary.size();
    }

    result.binaries[i].spirv_data = nullptr;  // Fix up pointers later or store offsets
    result.binaries[i].spirv_size = size;
    result.binaries[i].stage = ep.stage;
    result.binaries[i].backend = backend;
    result.binaries[i].format = binary_format;
    result.binaries[i].entry_point = ep.entry_point;
    result.binaries[i].local_size_x = local_size_x;
    result.binaries[i].local_size_y = local_size_y;
    result.binaries[i].local_size_z = local_size_z;

    result.shared_blob.insert(result.shared_blob.end(), ptr, ptr + size);
    // Update cache (we need to convert back to ShaderCompilationResult for cache compatibility)
    ShaderCompilationResult partial_res;
    partial_res.result = RHIResult::Success;
    partial_res.spirv_data.assign(ptr, ptr + size);
    {
      std::lock_guard<std::mutex> lock(_impl->cache_mutex);
      _impl->shader_cache[key] = partial_res;
    }
  }

  // Fix up pointers
  size_t current_offset = 0;
  for (auto& binary : result.binaries) {
    binary.spirv_data = result.shared_blob.data() + current_offset;
    current_offset += binary.spirv_size;
  }
  return result;
}

// Temporary placeholder to satisfy linker if needed, or we just remove declaration from header too

RHIResult initialize_dxc_interfaces_global() {
  HRESULT hr;

  global_dxc_utils.Reset();
  hr = global_dxc_create_instance(CLSID_DxcUtils, IID_PPV_ARGS(global_dxc_utils.GetAddressOf()));
  if (FAILED(hr)) {
    log::error("Failed to create DXC utils: 0x%08X", static_cast<uint32_t>(hr));
    return RHIResult::ValidationError;
  }

  global_dxc_compiler.Reset();
  hr = global_dxc_create_instance(CLSID_DxcCompiler, IID_PPV_ARGS(global_dxc_compiler.GetAddressOf()));
  if (FAILED(hr)) {
    log::error("Failed to create DXC compiler: 0x%08X", static_cast<uint32_t>(hr));
    return RHIResult::ValidationError;
  }

  return RHIResult::Success;
}

void ShaderCompiler::parse_dxc_error(const std::string& error_message, ShaderCompilationResult& result) {
  size_t colon_pos = error_message.find(':');
  if (colon_pos == std::string::npos) {
    result.error_line = 0;
    result.error_column = 0;
    return;
  }

  std::string location_part = error_message.substr(0, colon_pos);
  std::string message_part = error_message.substr(colon_pos + 1);

  size_t message_start = message_part.find_first_not_of(" \t");
  if (message_start != std::string::npos) {
    result.error_message = message_part.substr(message_start);
  } else {
    result.error_message = message_part;
  }

  size_t open_paren = location_part.find('(');
  size_t close_paren = location_part.find(')', open_paren);

  if (open_paren == std::string::npos || close_paren == std::string::npos) {
    result.error_line = 0;
    result.error_column = 0;
    return;
  }

  std::string paren_content = location_part.substr(open_paren + 1, close_paren - open_paren - 1);

  size_t comma_pos = paren_content.find(',');
  if (comma_pos != std::string::npos) {
    try {
      result.error_line = static_cast<uint32_t>(std::stoi(paren_content.substr(0, comma_pos)));
      result.error_column = static_cast<uint32_t>(std::stoi(paren_content.substr(comma_pos + 1)));
    } catch (const std::exception&) {
      result.error_line = 0;
      result.error_column = 0;
    }
  } else {
    try {
      result.error_line = static_cast<uint32_t>(std::stoi(paren_content));
      result.error_column = 0;
    } catch (const std::exception&) {
      result.error_line = 0;
      result.error_column = 0;
    }
  }
}

std::string ShaderCompiler::get_error_description(RHIResult result) {
  switch (result) {
    case RHIResult::Success:
      return "Success";
    case RHIResult::InvalidArgument:
      return "Invalid argument";
    case RHIResult::NotImplemented:
      return "Not implemented";
    case RHIResult::OutOfMemory:
      return "Out of memory";
    case RHIResult::ValidationError:
      return "Validation error";
    case RHIResult::NotReady:
      return "Not ready";
    default:
      return "Unknown error";
  }
}

RHIResult load_dxc_dll_global() {
  std::lock_guard<std::mutex> lock(global_dll_mutex);

  if (global_dxc_dll != nullptr) {
    return RHIResult::Success;
  }

  auto append_with_separator = [](std::vector<std::string>& paths, std::string path, char separator) {
    if (path.empty()) {
      return;
    }
    if (path.back() != '\\' && path.back() != '/') {
      path.push_back(separator);
    }
    paths.push_back(std::move(path));
  };

#if (ETX_PLATFORM_WINDOWS)
  constexpr char path_list_separator = ';';
  constexpr char path_separator = '\\';
  const char* library_names[] = {"dxcompiler.dll", "dxc.dll", "dxil.dll"};
  std::vector<std::string> search_paths = {
    "",
    ".\\",
    "bin\\",
    ".\\bin\\",
    "..\\bin\\",
  };
  const std::vector<std::string> system_paths = {
    "C:\\Program Files\\Microsoft DirectX Shader Compiler\\",
    "C:\\Program Files (x86)\\Microsoft DirectX Shader Compiler\\",
    "C:\\Windows\\System32\\",
    "C:\\Windows\\SysWOW64\\",
  };
#elif (ETX_PLATFORM_APPLE)
  constexpr char path_separator = '/';
  const char* library_names[] = {"libdxcompiler.dylib"};
  std::vector<std::string> search_paths = {
    "",
    "./",
    "bin/",
    "./bin/",
    "../bin/",
  };
  const std::vector<std::string> system_paths = {
    "/opt/homebrew/lib/",
    "/usr/local/lib/",
    "/usr/lib/",
    "/opt/dxc/lib/",
  };
#else
  constexpr char path_separator = '/';
  const char* library_names[] = {"libdxcompiler.so"};
  std::vector<std::string> search_paths = {
    "",
    "./",
    "bin/",
    "./bin/",
    "../bin/",
  };
  const std::vector<std::string> system_paths = {
    "/usr/local/lib/",
    "/usr/lib/",
    "/opt/dxc/lib/",
  };
#endif

  const char* data_folder = env().data_folder();
  if (data_folder && data_folder[0] != '\0') {
    append_with_separator(search_paths, std::string(data_folder), path_separator);
    append_with_separator(search_paths, std::string(data_folder) + "bin", path_separator);
  }

  search_paths.insert(search_paths.end(), system_paths.begin(), system_paths.end());

  for (const char* library_name : library_names) {
    global_dxc_dll = load_dxc_library(library_name);
    if (global_dxc_dll) {
      break;
    }

    for (const std::string& search_path : search_paths) {
      if (search_path.empty()) {
        continue;
      }

      std::string full_path = search_path + library_name;
      global_dxc_dll = load_dxc_library(full_path.c_str());
      if (global_dxc_dll) {
        break;
      }
    }

    if (global_dxc_dll) {
      break;
    }
  }

  if (global_dxc_dll == nullptr) {
    log::error("Failed to load DXC runtime library.");
    log::error("DirectX Shader Compiler was not found in known runtime search paths.");
    log::error("Set DXC_PATH to a valid DXC installation prefix if needed.");
    log::error("If DXC is missing, try:");
    log::error("1. Reconfigure CMake so FindDXC resolves include/library paths");
    log::error("2. Check that DXC runtime library is available near the executable");
    log::error("3. Set DXC_PATH to the DXC install folder");

    return RHIResult::NotImplemented;
  }

  global_dxc_create_instance = reinterpret_cast<DxcCreateInstanceProc>(load_dxc_symbol(global_dxc_dll, "DxcCreateInstance"));
  if (global_dxc_create_instance == nullptr) {
    log::error("Failed to resolve DxcCreateInstance in loaded DXC library.");
    if (global_dxc_dll) {
      unload_dxc_library(global_dxc_dll);
      global_dxc_dll = nullptr;
    }
    global_dxc_create_instance = nullptr;
    return RHIResult::NotImplemented;
  }

  return RHIResult::Success;
}

void unload_dxc_dll_global() {
  std::lock_guard<std::mutex> lock(global_dll_mutex);
  if (global_dxc_dll) {
#if ETX_PLATFORM_APPLE
    return;
#else
    unload_dxc_library(global_dxc_dll);
    global_dxc_dll = nullptr;
    global_dxc_create_instance = nullptr;
#endif
  }
}

std::vector<std::wstring> ShaderCompiler::Impl::build_dxc_arguments(const std::string& entry_point, RHIShaderStage stage,
  const std::unordered_map<std::string, std::string>& defines, const std::vector<std::string>& include_directories, bool for_preprocessing) {
  std::vector<std::wstring> arguments;
  uint32_t optimization_level = 3u;
  auto opt_it = defines.find("ETX_DXC_OPT_LEVEL");
  if (opt_it != defines.end()) {
    const char opt_char = opt_it->second.empty() ? '3' : opt_it->second[0];
    if ((opt_char >= '0') && (opt_char <= '3')) {
      optimization_level = static_cast<uint32_t>(opt_char - '0');
    }
  }

  std::wstring profile;
  switch (stage) {
    case RHIShaderStage::Vertex:
      profile = L"vs_6_6";
      break;
    case RHIShaderStage::Fragment:
      profile = L"ps_6_6";
      break;
    case RHIShaderStage::Compute:
      profile = L"cs_6_6";
      break;
    default:
      log::error("Unsupported shader stage for DXC compilation");
      return arguments;
  }

  if (for_preprocessing == false) {
    std::wstring entry_wstr = utf8_to_wstring(entry_point);

    arguments.emplace_back(L"-T");
    arguments.emplace_back(profile.c_str());
    arguments.emplace_back(L"-E");
    arguments.emplace_back(entry_wstr.c_str());
    arguments.emplace_back(L"-spirv");
    arguments.emplace_back(L"-fvk-use-dx-layout");
    arguments.emplace_back(L"-fvk-use-dx-position-w");
    arguments.emplace_back(L"-fspv-target-env=vulkan1.3");
    arguments.emplace_back(L"-fspv-extension=SPV_EXT_descriptor_indexing");
    arguments.emplace_back(L"-fspv-extension=SPV_KHR_ray_query");
    arguments.emplace_back(L"-enable-16bit-types");
    arguments.emplace_back((optimization_level == 0u) ? L"-O0" : ((optimization_level == 1u) ? L"-O1" : ((optimization_level == 2u) ? L"-O2" : L"-O3")));
    if constexpr (kEnableShaderDebugInfo) {
      arguments.emplace_back(L"-Zi");
      arguments.emplace_back(L"-Qembed_debug");
    }
  } else {
    arguments.emplace_back(L"-P");
  }

  for (const auto& [key, value] : defines) {
    if (key == "ETX_DXC_OPT_LEVEL") {
      continue;
    }
    std::string define_str = key;
    if (!value.empty()) {
      define_str += "=" + value;
    }
    std::wstring define_wstr = utf8_to_wstring(define_str);
    arguments.push_back(L"-D");
    arguments.push_back(define_wstr);
  }

  static const std::vector<std::pair<std::string, std::string>> default_defines = {{"VULKAN", "1"}, {"SPIRV", "1"}, {"BINDLESS", "1"}};

  for (const auto& [key, value] : default_defines) {
    std::string define_str = key + "=" + value;
    std::wstring define_wstr = utf8_to_wstring(define_str);
    arguments.push_back(L"-D");
    arguments.push_back(define_wstr);
  }

  for (const auto& include_directory : include_directories) {
    if (include_directory.empty()) {
      continue;
    }

    const std::wstring include_directory_wstr = utf8_to_wstring(include_directory);
    if (include_directory_wstr.empty() == false) {
      arguments.push_back(L"-I");
      arguments.push_back(include_directory_wstr);
    }
  }

  return arguments;
}

CustomIncludeHandler::CustomIncludeHandler(DxcComPtr<IDxcUtils> dxc_utils, std::string shader_directory)
  : _dxc_utils(dxc_utils) {
  if (shader_directory.empty() == false) {
    std::error_code ec;
    auto absolute_path = std::filesystem::absolute(std::filesystem::path(shader_directory), ec);
    _shader_directory = (ec.value() == 0) ? absolute_path.string() : shader_directory;
  }
}

CustomIncludeHandler::~CustomIncludeHandler() {
  _include_cache.clear();
}

HRESULT STDMETHODCALLTYPE CustomIncludeHandler::QueryInterface(REFIID iid, void** ppvObject) {
  if (ppvObject == nullptr) {
    return E_INVALIDARG;
  }

  if (iid == __uuidof(IUnknown) || iid == __uuidof(IDxcIncludeHandler)) {
    *ppvObject = static_cast<IDxcIncludeHandler*>(this);
    AddRef();
    return S_OK;
  }

  *ppvObject = nullptr;
  return E_NOINTERFACE;
}

ULONG STDMETHODCALLTYPE CustomIncludeHandler::AddRef() {
  return _ref_count.fetch_add(1, std::memory_order_relaxed) + 1;
}

ULONG STDMETHODCALLTYPE CustomIncludeHandler::Release() {
  ULONG ref_count = _ref_count.fetch_sub(1, std::memory_order_acq_rel) - 1;
  if (ref_count == 0) {
    delete this;
  }
  return ref_count;
}

HRESULT STDMETHODCALLTYPE CustomIncludeHandler::LoadSource(LPCWSTR pFilename, IDxcBlob** ppIncludeSource) {
  if (pFilename == nullptr || ppIncludeSource == nullptr) {
    return E_INVALIDARG;
  }

  std::string filename = wstring_to_string(pFilename);
  std::string full_path = find_include_file(filename);

  if (full_path.empty()) {
    return HRESULT_FROM_WIN32(ERROR_FILE_NOT_FOUND);
  }

  std::error_code ec;
  auto file_size = std::filesystem::file_size(full_path, ec);
  if (ec || file_size > MAX_SHADER_FILE_SIZE) {
    log::error("Include file too large or invalid: %s (%zu bytes)", full_path.c_str(), static_cast<size_t>(file_size));
    return E_FAIL;
  }

  {
    std::lock_guard<std::mutex> lock(_include_cache_mutex);
    auto it = _include_cache.find(full_path);
    if (it != _include_cache.end()) {
      *ppIncludeSource = it->second.Get();
      (*ppIncludeSource)->AddRef();
      return S_OK;
    }
  }

  auto file = fopen(full_path.c_str(), "rb");
  if (file == nullptr) {
    log::error("Failed to open include file: %s", full_path.c_str());
    return E_FAIL;
  }

  std::string content(static_cast<size_t>(file_size), '\0');
  if (content.empty() == false) {
    size_t read_size = fread(content.data(), 1, content.size(), file);
    if (read_size != content.size()) {
      fclose(file);
      log::error("Failed to read include file: %s", full_path.c_str());
      return E_FAIL;
    }
  }
  fclose(file);

  if (content.size() != file_size) {
    log::error("Include file size mismatch: %s", full_path.c_str());
    return E_FAIL;
  }

  DxcComPtr<IDxcBlobEncoding> blob;
  HRESULT hr = _dxc_utils->CreateBlob(content.data(), static_cast<uint32_t>(content.size()), DXC_CP_UTF8, blob.ReleaseAndGetAddressOf());

  if (FAILED(hr)) {
    log::error("Failed to create include blob for: %s", full_path.c_str());
    return hr;
  }

  {
    std::lock_guard<std::mutex> lock(_include_cache_mutex);
    _include_cache[full_path] = blob;
  }

  *ppIncludeSource = blob.Detach();
  return S_OK;
}

std::string CustomIncludeHandler::find_include_file(const std::string& filename) {
  auto absolute_if_exists = [](const std::filesystem::path& path) -> std::string {
    std::error_code ec;
    if (std::filesystem::exists(path, ec) == false || ec.value() != 0) {
      return {};
    }
    auto absolute_path = std::filesystem::absolute(path, ec);
    return (ec.value() == 0) ? absolute_path.string() : path.string();
  };

  auto contains_parent_reference = [](const std::filesystem::path& path) -> bool {
    for (const auto& part : path) {
      if (part == "..") {
        return true;
      }
    }
    return false;
  };

  auto resolve_relative_to_root = [&](const std::filesystem::path& root, std::filesystem::path include_path) -> std::string {
    if (root.empty()) {
      return {};
    }

    include_path = include_path.lexically_normal();
    if (include_path.empty() == false) {
      auto candidate = root / include_path;
      if (auto resolved = absolute_if_exists(candidate); resolved.empty() == false) {
        return resolved;
      }
    }

    return {};
  };

  std::filesystem::path include_path = std::filesystem::path(filename).lexically_normal();
  if (contains_parent_reference(include_path)) {
    return {};
  }

  if (include_path.is_absolute()) {
    return absolute_if_exists(include_path);
  }

  if (_shader_directory.empty() == false) {
    std::filesystem::path root = std::filesystem::path(_shader_directory);
    for (uint32_t depth = 0; depth < 8 && root.empty() == false; ++depth) {
      if (auto resolved = resolve_relative_to_root(root, include_path); resolved.empty() == false) {
        return resolved;
      }

      std::filesystem::path parent = root.parent_path();
      if (parent.empty() || parent == root) {
        break;
      }
      root = std::move(parent);
    }
  }

  for (const auto root : default_shader_search_paths) {
    if (auto resolved = resolve_relative_to_root(std::filesystem::path(std::string(root)), include_path); resolved.empty() == false) {
      return resolved;
    }
  }

  return {};
}

std::string CustomIncludeHandler::wstring_to_string(const std::wstring& wstr) {
  return wstring_to_utf8(wstr);
}

std::string ShaderCompiler::read_file_content(const std::string& file_path, std::string& error_message) {
  if (file_path.empty()) {
    error_message = "File path is empty";
    return "";
  }

  std::error_code ec;
  uintmax_t file_size = std::filesystem::file_size(file_path, ec);
  if (ec) {
    error_message = "Failed to query file size: " + file_path;
    return "";
  }

  if (file_size > MAX_SHADER_FILE_SIZE) {
    error_message = "File too large: " + std::to_string(file_size) + " bytes (max: " + std::to_string(MAX_SHADER_FILE_SIZE) + " bytes)";
    return "";
  }

  FILE* file = nullptr;
#if defined(_MSC_VER)
  if (fopen_s(&file, file_path.c_str(), "rb") != 0 || file == nullptr) {
#else
  file = fopen(file_path.c_str(), "rb");
  if (file == nullptr) {
#endif
    error_message = "Failed to open file: " + file_path;
    return "";
  }

  std::string content(static_cast<size_t>(file_size), '\0');
  if (content.empty() == false) {
    size_t read_size = fread(content.data(), 1, content.size(), file);
    if (read_size != content.size()) {
      fclose(file);
      error_message = "Failed to read file: " + file_path;
      return "";
    }
  }

  if (ferror(file)) {
    fclose(file);
    error_message = "Failed to read file: " + file_path;
    return "";
  }
  fclose(file);

  error_message.clear();
  return content;
}

}  // namespace etx
