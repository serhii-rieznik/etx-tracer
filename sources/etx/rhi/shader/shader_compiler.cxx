#include <etx/rhi/shader/shader_compiler.hxx>

#include <etx/core/log.hxx>
#include <array>
#include <string_view>

#if !defined(WIN32_LEAN_AND_MEAN)
# define WIN32_LEAN_AND_MEAN
#endif

#include <windows.h>
#include <comdef.h>

#include <dxc/dxcapi.h>
#include <wrl.h>
using namespace Microsoft::WRL;

namespace etx {

// Forward declarations for global helper functions
using DxcCreateInstanceProc = HRESULT(__stdcall*)(REFCLSID rclsid, REFIID riid, LPVOID* ppv);
RHIResult load_dxc_dll_global();
void unload_dxc_dll_global();
RHIResult initialize_dxc_interfaces_global();

namespace {

constexpr std::array<std::string_view, 6> default_shader_search_paths = {
  "./sources/etx/render",
  "../sources/etx/render",
  "../../sources/etx/render",
  "./sources",
  "../sources",
  "../../sources",
};

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

bool should_suppress_preprocess_diagnostics(const std::string& diagnostics) {
  static constexpr std::string_view angled_include_advisory = "with <angled> include; use \"quotes\" instead";
  return diagnostics.find(angled_include_advisory) != std::string::npos;
}
}  // namespace

class CustomIncludeHandler : public IDxcIncludeHandler {
 public:
  CustomIncludeHandler(Microsoft::WRL::ComPtr<IDxcUtils> dxc_utils, std::string shader_directory);
  ~CustomIncludeHandler();

  HRESULT STDMETHODCALLTYPE QueryInterface(REFIID iid, void** ppvObject) override;
  ULONG STDMETHODCALLTYPE AddRef() override;
  ULONG STDMETHODCALLTYPE Release() override;

  HRESULT STDMETHODCALLTYPE LoadSource(LPCWSTR pFilename, IDxcBlob** ppIncludeSource) override;

 private:
  Microsoft::WRL::ComPtr<IDxcUtils> _dxc_utils;
  std::string _shader_directory;
  std::unordered_map<std::string, Microsoft::WRL::ComPtr<IDxcBlobEncoding>> _include_cache;
  std::mutex _include_cache_mutex;
  std::atomic<ULONG> _ref_count = 1;

  std::string find_include_file(const std::string& filename);
  std::string wstring_to_string(const std::wstring& wstr);
};

struct ShaderVariantKey {
  std::string source_name;
  std::string entry_point;
  RHIShaderStage stage;
  std::map<std::string, std::string> defines;
  uint64_t source_hash = 0;

  bool operator==(const ShaderVariantKey& other) const {
    return source_name == other.source_name && entry_point == other.entry_point && stage == other.stage && defines == other.defines && source_hash == other.source_hash;
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
    h = etx_hash64_continue(&key.source_hash, sizeof(key.source_hash), h);
    return h;
  }
};

struct ShaderCompiler::Impl {
  std::unordered_map<ShaderVariantKey, ShaderCompilationResult, ShaderVariantKeyHash> shader_cache;
  std::mutex cache_mutex;

  ComPtr<IDxcUtils> dxc_utils;
  ComPtr<IDxcCompiler3> dxc_compiler;

  // Helper method
  std::vector<std::wstring> build_dxc_arguments(const std::string& entry_point, RHIShaderStage stage, const std::unordered_map<std::string, std::string>& defines,
    bool for_preprocessing = false);
};

// File-scope global variables for DXC
std::mutex global_init_mutex;
Microsoft::WRL::ComPtr<IDxcUtils> global_dxc_utils;
Microsoft::WRL::ComPtr<IDxcCompiler3> global_dxc_compiler;
HMODULE global_dxc_dll = nullptr;
std::atomic<bool> global_com_initialized{false};
std::mutex global_dll_mutex;
DxcCreateInstanceProc global_dxc_create_instance = nullptr;

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

    // Initialize COM
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

    // Initialize DXC interfaces
    RHIResult init_result = initialize_dxc_interfaces_global();
    if (init_result != RHIResult::Success) {
      log::error("Failed to initialize DXC interfaces");
      if (global_com_initialized.load(std::memory_order_acquire)) {
        CoUninitialize();
        global_com_initialized.store(false, std::memory_order_release);
      }
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
  std::lock_guard<std::mutex> dll_lock(global_dll_mutex);

  global_dxc_utils.Reset();
  global_dxc_compiler.Reset();

  if (global_com_initialized.load(std::memory_order_acquire)) {
    CoUninitialize();
    global_com_initialized.store(false, std::memory_order_release);
  }

  if (global_dxc_dll) {
    FreeLibrary(global_dxc_dll);
    global_dxc_dll = nullptr;
    global_dxc_create_instance = nullptr;
  }
}

ShaderCompiler::ShaderCompiler()
  : _impl(std::make_unique<Impl>()) {
}

ShaderCompiler::~ShaderCompiler() {
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
  const std::unordered_map<std::string, std::string>& defines) {
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

  return compile(source, source_path, entry_points, defines);
}

ShaderCompiler::MultiShaderCompilationResult ShaderCompiler::compile(const std::string& hlsl_source, const std::string& source_name,
  const std::vector<ShaderEntryPoint>& entry_points, const std::unordered_map<std::string, std::string>& defines) {
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

  if (include_handler == nullptr) {
    result.result = RHIResult::OutOfMemory;
    result.error_message = "Failed to create include handler";
    return result;
  }

  // Preprocess once
  std::string preprocessed_source;
  {
    ComPtr<IDxcBlobEncoding> source_blob;
    HRESULT hr = _impl->dxc_utils->CreateBlob(hlsl_source.data(), static_cast<uint32_t>(hlsl_source.size()), CP_UTF8, &source_blob);
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
    auto preprocess_args = _impl->build_dxc_arguments(entry_points[0].entry_point, entry_points[0].stage, defines, true);
    std::vector<const wchar_t*> preprocess_args_ptr;
    for (const auto& arg : preprocess_args) {
      preprocess_args_ptr.push_back(arg.c_str());
    }

    ComPtr<IDxcResult> preprocess_result;
    hr = _impl->dxc_compiler->Compile(&dxc_buffer, preprocess_args_ptr.data(), static_cast<uint32_t>(preprocess_args_ptr.size()), include_handler.get(),
      IID_PPV_ARGS(&preprocess_result));
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
      ComPtr<IDxcBlobUtf8> preprocess_errors;
      hr = preprocess_result->GetOutput(DXC_OUT_ERRORS, IID_PPV_ARGS(&preprocess_errors), nullptr);
      if (SUCCEEDED(hr) && preprocess_errors && preprocess_errors->GetStringLength() > 0) {
        preprocess_diagnostics.assign(preprocess_errors->GetStringPointer(), preprocess_errors->GetStringLength());
      }
    }

    const bool suppress_preprocess_diagnostics = should_suppress_preprocess_diagnostics(preprocess_diagnostics);

    if (preprocess_diagnostics.empty() == false && (suppress_preprocess_diagnostics == false)) {
      log::warning("Shader preprocessing diagnostics [%s]:\n%s", source_name.empty() ? "<memory>" : source_name.c_str(), preprocess_diagnostics.c_str());
    }

    if (FAILED(preprocess_status) && (suppress_preprocess_diagnostics == false)) {
      log::warning("Shader preprocessing status is failed for [%s], attempting compilation of produced HLSL anyway", source_name.empty() ? "<memory>" : source_name.c_str());
    }

    ComPtr<IDxcBlobUtf8> canonical_hlsl_blob;
    hr = preprocess_result->GetOutput(DXC_OUT_HLSL, IID_PPV_ARGS(&canonical_hlsl_blob), nullptr);
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
  }

  if (contains_include_directive(preprocessed_source)) {
    log::warning("Preprocessed shader source still contains #include directives [%s]", source_name.empty() ? "<memory>" : source_name.c_str());
  }

  uint64_t source_hash = etx_hash64(preprocessed_source.data(), preprocessed_source.size());
  std::map<std::string, std::string> ordered_defines(defines.begin(), defines.end());

  result.binaries.resize(entry_points.size());

  for (uint32_t i = 0, e = entry_points.size(); i < e; ++i) {
    const auto& ep = entry_points[i];
    ShaderVariantKey key{source_name, ep.entry_point, ep.stage, ordered_defines, source_hash};

    // Check cache
    {
      std::lock_guard<std::mutex> lock(_impl->cache_mutex);
      auto it = _impl->shader_cache.find(key);
      if (it != _impl->shader_cache.end() && it->second.result == RHIResult::Success) {
        // Cache hit - copy data to shared Blob (we will optimize shared blob usage later or doing it differently)
        // For now, let's just use the logic as requested: return array of compiled spir-v
        // But the user asked for potentially shared buffer.
        // Let's postpone packing into one buffer for simplicity and correctness first, or just append to the vector.
        // Wait, if we use cache, we get individual blobs.
        // Let's proceed with individual compilations and append to shared blob.
      }
    }

    // Prepare for compilation
    ComPtr<IDxcBlobEncoding> preprocessed_blob;
    _impl->dxc_utils->CreateBlob(preprocessed_source.data(), static_cast<uint32_t>(preprocessed_source.size()), DXC_CP_UTF8, &preprocessed_blob);
    DxcBuffer compile_buffer = {preprocessed_blob->GetBufferPointer(), preprocessed_blob->GetBufferSize(), DXC_CP_UTF8};

    std::vector<std::wstring> arguments = _impl->build_dxc_arguments(ep.entry_point, ep.stage, defines, false);
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

    Microsoft::WRL::ComPtr<IDxcResult> compile_result = {};
    HRESULT hr = _impl->dxc_compiler->Compile(&source_buffer, arguments_ptr.data(), (uint32_t)arguments_ptr.size(), nullptr, IID_PPV_ARGS(compile_result.GetAddressOf()));

    ShaderCompilationResult entry_result = {
      .result = RHIResult::Success,
    };

    if (FAILED(hr) || !compile_result) {
      entry_result.result = RHIResult::ValidationError;
      entry_result.error_message = "DXC Compile failed";
    } else {
      compile_result->GetStatus(&hr);
      if (FAILED(hr)) {
        entry_result.result = RHIResult::ValidationError;
      }
      Microsoft::WRL::ComPtr<IDxcBlobUtf8> errors;
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

    Microsoft::WRL::ComPtr<IDxcBlob> shader_obj;
    compile_result->GetOutput(DXC_OUT_OBJECT, IID_PPV_ARGS(shader_obj.GetAddressOf()), nullptr);

    if (shader_obj) {
      const uint8_t* ptr = static_cast<const uint8_t*>(shader_obj->GetBufferPointer());
      size_t size = shader_obj->GetBufferSize();

      result.binaries[i].spirv_data = nullptr;  // Fix up pointers later or store offsets
      result.binaries[i].spirv_size = size;
      result.binaries[i].stage = ep.stage;
      result.binaries[i].entry_point = ep.entry_point;

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

  hr = global_dxc_create_instance(CLSID_DxcUtils, IID_PPV_ARGS(global_dxc_utils.GetAddressOf()));
  if (FAILED(hr)) {
    log::error("Failed to create DXC utils: 0x%08X", static_cast<uint32_t>(hr));
    return RHIResult::ValidationError;
  }

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
    default:
      return "Unknown error";
  }
}

RHIResult load_dxc_dll_global() {
  std::lock_guard<std::mutex> lock(global_dll_mutex);

  if (global_dxc_dll != nullptr) {
    return RHIResult::Success;
  }

  const char* dll_names[] = {"dxcompiler.dll", "dxc.dll", "dxil.dll"};

  std::vector<std::string> search_paths = {
    "",
    "bin\\",
    "..\\bin\\",
    ".\\bin\\",
  };

  std::vector<std::string> system_paths = {
    "C:\\Program Files\\Microsoft DirectX Shader Compiler\\",
    "C:\\Program Files (x86)\\Microsoft DirectX Shader Compiler\\",
    "C:\\Windows\\System32\\",
    "C:\\Windows\\SysWOW64\\",
  };
  search_paths.insert(search_paths.end(), system_paths.begin(), system_paths.end());

  const char* path_env = getenv("PATH");
  if (path_env) {
    std::string path_str(path_env);
    size_t pos = 0;
    while ((pos = path_str.find(';', pos)) != std::string::npos) {
      search_paths.push_back(path_str.substr(0, pos + 1));
      pos++;
    }
  }

  const char* dxc_path_env = getenv("DXC_PATH");
  if (dxc_path_env) {
    search_paths.push_back(std::string(dxc_path_env) + "\\");
  }

  for (const char* dll_name : dll_names) {
    global_dxc_dll = LoadLibraryA(dll_name);
    if (global_dxc_dll) {
      break;
    }

    std::string bin_path = std::string("bin\\") + dll_name;
    global_dxc_dll = LoadLibraryA(bin_path.c_str());
    if (global_dxc_dll) {
      break;
    }

    for (const std::string& search_path : search_paths) {
      if (search_path.empty() || search_path == "bin\\")
        continue;

      std::string full_path = search_path + dll_name;
      global_dxc_dll = LoadLibraryA(full_path.c_str());
      if (global_dxc_dll) {
        break;
      }
    }

    if (global_dxc_dll)
      break;
  }

  if (global_dxc_dll == nullptr) {
    log::error("Failed to load DXC DLL. DirectX Shader Compiler was not found.");
    log::error("DXC should have been installed automatically during the build process.");
    log::error("Please ensure the build completed successfully and DXC was copied to the bin folder.");
    log::error("");
    log::error("If DXC is missing, try:");
    log::error("1. Rebuild the project (CMake will download and install DXC)");
    log::error("2. Check that bin/dxcompiler.dll exists");
    log::error("3. Manually download from: https://github.com/microsoft/DirectXShaderCompiler/releases");

    return RHIResult::NotImplemented;
  }

  global_dxc_create_instance = reinterpret_cast<DxcCreateInstanceProc>(GetProcAddress(global_dxc_dll, "DxcCreateInstance"));
  if (global_dxc_create_instance == nullptr) {
    log::error("Failed to get DxcCreateInstance function from DXC DLL");
    log::error("The loaded DLL may not be a valid DXC installation.");
    unload_dxc_dll_global();
    return RHIResult::NotImplemented;
  }

  return RHIResult::Success;
}

void unload_dxc_dll_global() {
  std::lock_guard<std::mutex> lock(global_dll_mutex);
  if (global_dxc_dll) {
    FreeLibrary(global_dxc_dll);
    global_dxc_dll = nullptr;
    global_dxc_create_instance = nullptr;
  }
}

std::vector<std::wstring> ShaderCompiler::Impl::build_dxc_arguments(const std::string& entry_point, RHIShaderStage stage,
  const std::unordered_map<std::string, std::string>& defines, bool for_preprocessing) {
  std::vector<std::wstring> arguments;

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
    std::wstring entry_wstr(entry_point.begin(), entry_point.end());

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
    arguments.emplace_back(L"-O3");
    arguments.emplace_back(L"-Zi");
    arguments.emplace_back(L"-Qembed_debug");
  } else {
    arguments.emplace_back(L"-P");
  }

  auto string_to_wstring = [](const std::string& str) -> std::wstring {
    if (str.empty())
      return L"";
    int size_needed = MultiByteToWideChar(CP_UTF8, 0, str.c_str(), static_cast<int>(str.size()), nullptr, 0);
    std::wstring wstr(size_needed, 0);
    MultiByteToWideChar(CP_UTF8, 0, str.c_str(), static_cast<int>(str.size()), &wstr[0], size_needed);
    return wstr;
  };

  for (const auto& [key, value] : defines) {
    std::string define_str = key;
    if (!value.empty()) {
      define_str += "=" + value;
    }
    std::wstring define_wstr = string_to_wstring(define_str);
    arguments.push_back(L"-D");
    arguments.push_back(define_wstr.c_str());
  }

  static const std::vector<std::pair<std::string, std::string>> default_defines = {{"VULKAN", "1"}, {"SPIRV", "1"}, {"BINDLESS", "1"}};

  for (const auto& [key, value] : default_defines) {
    std::string define_str = key + "=" + value;
    std::wstring define_wstr = string_to_wstring(define_str);
    arguments.push_back(L"-D");
    arguments.push_back(define_wstr.c_str());
  }

  return arguments;
}

CustomIncludeHandler::CustomIncludeHandler(Microsoft::WRL::ComPtr<IDxcUtils> dxc_utils, std::string shader_directory)
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
    log::error("Include file not found: %s", filename.c_str());
    return E_FAIL;
  }

  std::error_code ec;
  auto file_size = std::filesystem::file_size(full_path, ec);
  if (ec || file_size > MAX_SHADER_FILE_SIZE) {
    log::error("Include file too large or invalid: %s (%zu bytes)", full_path.c_str(), file_size);
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

  ComPtr<IDxcBlobEncoding> blob;
  HRESULT hr = _dxc_utils->CreateBlob(content.data(), static_cast<uint32_t>(content.size()), DXC_CP_UTF8, &blob);

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

  auto trim_first_component = [](const std::filesystem::path& path) -> std::filesystem::path {
    auto it = path.begin();
    if (it == path.end()) {
      return {};
    }
    ++it;

    std::filesystem::path trimmed;
    for (; it != path.end(); ++it) {
      trimmed /= *it;
    }
    return trimmed;
  };

  auto resolve_relative_to_root = [&](const std::filesystem::path& root, std::filesystem::path include_path) -> std::string {
    if (root.empty()) {
      return {};
    }

    include_path = include_path.lexically_normal();
    for (;;) {
      if (include_path.empty() == false) {
        auto candidate = root / include_path;
        if (auto resolved = absolute_if_exists(candidate); resolved.empty() == false) {
          return resolved;
        }
      }

      auto trimmed = trim_first_component(include_path);
      if (trimmed.empty() || trimmed == include_path) {
        break;
      }
      include_path = std::move(trimmed);
    }

    return {};
  };

  if (auto resolved = absolute_if_exists(std::filesystem::path(filename)); resolved.empty() == false) {
    return resolved;
  }

  std::filesystem::path include_path = std::filesystem::path(filename).lexically_normal();
  if (include_path.is_absolute()) {
    return {};
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
  if (wstr.empty())
    return "";

  int size_needed = WideCharToMultiByte(CP_UTF8, 0, wstr.c_str(), static_cast<int>(wstr.size()), nullptr, 0, nullptr, nullptr);
  std::string str(size_needed, 0);
  WideCharToMultiByte(CP_UTF8, 0, wstr.c_str(), static_cast<int>(wstr.size()), &str[0], size_needed, nullptr, nullptr);
  return str;
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
