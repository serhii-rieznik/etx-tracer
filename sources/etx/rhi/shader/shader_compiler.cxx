#include <etx/rhi/shader/shader_compiler.hxx>

#include <etx/core/log.hxx>

#if !defined(WIN32_LEAN_AND_MEAN)
# define WIN32_LEAN_AND_MEAN
#endif

#include <windows.h>
#include <comdef.h>

#include <dxc/dxcapi.h>
#include <wrl.h>

#include <fstream>
#include <sstream>
#include <algorithm>
#include <filesystem>
#include <atomic>

using namespace Microsoft::WRL;

namespace etx {

std::unique_ptr<ShaderCompiler> ShaderCompiler::global_instance;
std::mutex ShaderCompiler::global_init_mutex;
std::atomic<bool> ShaderCompiler::global_initialized{false};

Microsoft::WRL::ComPtr<IDxcUtils> ShaderCompiler::global_dxc_utils;
Microsoft::WRL::ComPtr<IDxcCompiler3> ShaderCompiler::global_dxc_compiler;
CustomIncludeHandler* ShaderCompiler::global_custom_include_handler = nullptr;
HMODULE ShaderCompiler::global_dxc_dll = nullptr;
std::atomic<bool> ShaderCompiler::global_com_initialized{false};
std::mutex ShaderCompiler::global_dll_mutex;
ShaderCompiler::DxcCreateInstanceProc ShaderCompiler::global_dxc_create_instance = nullptr;

RHIResult ShaderCompiler::initialize_global() {
  std::lock_guard<std::mutex> lock(global_init_mutex);
  if (global_initialized.load(std::memory_order_acquire)) {
    return RHIResult::Success;
  }

  global_instance = std::make_unique<ShaderCompiler>();

  RHIResult result = global_instance->initialize_global_instance();
  if (result != RHIResult::Success) {
    global_instance.reset();
    return result;
  }

  global_initialized.store(true, std::memory_order_release);
  return RHIResult::Success;
}

void ShaderCompiler::shutdown_global() {
  std::lock_guard<std::mutex> lock(global_init_mutex);
  if (global_initialized.load(std::memory_order_acquire)) {
    global_instance.reset();
    global_initialized.store(false, std::memory_order_release);

    {
      std::lock_guard<std::mutex> dll_lock(global_dll_mutex);

      if (global_custom_include_handler != nullptr) {
        global_custom_include_handler->Release();
        global_custom_include_handler = nullptr;
      }

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
  }
}

ShaderCompiler* ShaderCompiler::get_global_instance() {
  initialize_global();

  if (global_initialized.load(std::memory_order_acquire) == false) {
    log::error("Global shader compiler not initialized - call initialize_global() first");
    return nullptr;
  }

  if (global_instance->is_initialized() == false) {
    RHIResult result = global_instance->initialize();
    if (result != RHIResult::Success) {
      log::error("Failed to initialize global shader compiler instance");
      return nullptr;
    }
  }

  return global_instance.get();
}

bool ShaderCompiler::is_global_initialized() {
  return global_initialized.load(std::memory_order_acquire);
}

ShaderCompiler::ShaderCompiler() {
}

ShaderCompiler::~ShaderCompiler() {
  _shader_cache.clear();
  _include_paths.clear();
}

RHIResult ShaderCompiler::initialize() {
  if (is_initialized()) {
    return RHIResult::Success;
  }

  if (!is_global_initialized()) {
    log::error("Global shader compiler not initialized - call initialize_global() first");
    return RHIResult::InvalidArgument;
  }

  _dxc_utils = global_dxc_utils;
  _dxc_compiler = global_dxc_compiler;
  _custom_include_handler = global_custom_include_handler;

  if (_custom_include_handler) {
    _custom_include_handler->AddRef();
  }

  return RHIResult::Success;
}

void ShaderCompiler::add_include_path(const std::string& path) {
  if (path.empty()) {
    log::warning("Attempted to add empty include path");
    return;
  }

  if (path.length() > 1024) {
    log::warning("Include path too long: %zu characters", path.length());
    return;
  }

  _include_paths.push_back(path);
}

void ShaderCompiler::clear_include_paths() {
  _include_paths.clear();
}

ShaderCompilationResult ShaderCompiler::get_or_compile_shader_variant(const std::string& hlsl_source, const std::string& entry_point, RHIShaderStage stage,
  const std::string& source_name, const std::unordered_map<std::string, std::string>& defines) {
  ShaderCompilationResult result;

  if (hlsl_source.empty()) {
    result.result = RHIResult::InvalidArgument;
    result.error_message = "HLSL source code is empty";
    log::error("Shader compilation failed: %s", result.error_message.c_str());
    return result;
  }

  if (hlsl_source.size() > MAX_SHADER_SOURCE_SIZE) {
    result.result = RHIResult::InvalidArgument;
    result.error_message = "HLSL source code too large: " + std::to_string(hlsl_source.size()) + " bytes (max: " + std::to_string(MAX_SHADER_SOURCE_SIZE) + " bytes)";
    log::error("Shader compilation failed: %s", result.error_message.c_str());
    return result;
  }

  if (entry_point.empty()) {
    result.result = RHIResult::InvalidArgument;
    result.error_message = "Entry point is empty";
    log::error("Shader compilation failed: %s", result.error_message.c_str());
    return result;
  }

  if (source_name.empty()) {
    result.result = RHIResult::InvalidArgument;
    result.error_message = "Source name is empty";
    log::error("Shader compilation failed: %s", result.error_message.c_str());
    return result;
  }

  if (stage != RHIShaderStage::Vertex && stage != RHIShaderStage::Fragment && stage != RHIShaderStage::Compute) {
    result.result = RHIResult::InvalidArgument;
    result.error_message = "Unsupported shader stage";
    log::error("Shader compilation failed: %s", result.error_message.c_str());
    return result;
  }

  for (const auto& define : defines) {
    if (define.first.empty()) {
      result.result = RHIResult::InvalidArgument;
      result.error_message = "Empty define key found";
      log::error("Shader compilation failed: %s", result.error_message.c_str());
      return result;
    }
    if (define.first.length() > 256 || define.second.length() > 1024) {
      result.result = RHIResult::InvalidArgument;
      result.error_message = "Define key or value too long";
      log::error("Shader compilation failed: %s", result.error_message.c_str());
      return result;
    }
  }

  if (!is_initialized()) {
    result.result = RHIResult::InvalidArgument;
    result.error_message = "Shader compiler not initialized";
    log::error("Shader compilation failed: %s", result.error_message.c_str());
    return result;
  }

  std::map<std::string, std::string> ordered_defines(defines.begin(), defines.end());

  ShaderVariantKey key{source_name, entry_point, stage, ordered_defines};

  {
    std::lock_guard<std::mutex> lock(_cache_mutex);
    auto it = _shader_cache.find(key);
    if (it != _shader_cache.end()) {
      return it->second;
    }
  }

  result = compile_hlsl_to_spirv(hlsl_source, entry_point, stage, source_name, defines);

  if (result.result == RHIResult::Success) {
    std::lock_guard<std::mutex> lock(_cache_mutex);
    _shader_cache[key] = result;
  }

  return result;
}

void ShaderCompiler::clear_shader_cache() {
  std::lock_guard<std::mutex> lock(_cache_mutex);
  _shader_cache.clear();
}

ShaderCompilationResult ShaderCompiler::compile_hlsl_with_multiple_entry_points(const std::string& hlsl_source, const std::vector<std::string>& entry_points, RHIShaderStage stage,
  const std::string& source_name, const std::unordered_map<std::string, std::string>& defines) {
  ShaderCompilationResult result;

  if (entry_points.empty()) {
    result.result = RHIResult::InvalidArgument;
    result.error_message = "No entry points specified";
    return result;
  }

  return compile_hlsl_to_spirv(hlsl_source, entry_points[0], stage, source_name, defines);
}

ShaderCompilationResult ShaderCompiler::load_and_compile_shader_from_file(const std::string& file_path, const std::string& entry_point, RHIShaderStage stage,
  const std::unordered_map<std::string, std::string>& defines) {
  ShaderCompilationResult result;

  if (file_path.empty()) {
    result.result = RHIResult::InvalidArgument;
    result.error_message = "File path is empty";
    log::error("Shader file loading failed: %s", result.error_message.c_str());
    return result;
  }

  if (entry_point.empty()) {
    result.result = RHIResult::InvalidArgument;
    result.error_message = "Entry point is empty";
    log::error("Shader file loading failed: %s", result.error_message.c_str());
    return result;
  }

  if (stage != RHIShaderStage::Vertex && stage != RHIShaderStage::Fragment && stage != RHIShaderStage::Compute) {
    result.result = RHIResult::InvalidArgument;
    result.error_message = "Unsupported shader stage";
    log::error("Shader file loading failed: %s", result.error_message.c_str());
    return result;
  }

  for (const auto& define : defines) {
    if (define.first.empty()) {
      result.result = RHIResult::InvalidArgument;
      result.error_message = "Empty define key found";
      log::error("Shader file loading failed: %s", result.error_message.c_str());
      return result;
    }
    if (define.first.length() > 256 || define.second.length() > 1024) {
      result.result = RHIResult::InvalidArgument;
      result.error_message = "Define key or value too long";
      log::error("Shader file loading failed: %s", result.error_message.c_str());
      return result;
    }
  }

  if (!is_initialized()) {
    result.result = RHIResult::InvalidArgument;
    result.error_message = "Shader compiler not initialized";
    log::error("Shader file loading failed: %s", result.error_message.c_str());
    return result;
  }

  if (!std::filesystem::exists(file_path)) {
    result.result = RHIResult::InvalidArgument;
    result.error_message = "Shader file not found: " + file_path;
    log::error("Shader file not found: %s", file_path.c_str());
    return result;
  }

  std::error_code ec;
  auto file_size = std::filesystem::file_size(file_path, ec);
  if (ec) {
    result.result = RHIResult::InvalidArgument;
    result.error_message = "Failed to get file size: " + file_path;
    log::error("Failed to get file size for: %s", file_path.c_str());
    return result;
  }

  if (file_size > MAX_SHADER_FILE_SIZE) {
    result.result = RHIResult::InvalidArgument;
    result.error_message = "Shader file too large: " + std::to_string(file_size) + " bytes (max: " + std::to_string(MAX_SHADER_FILE_SIZE) + " bytes)";
    log::error("Shader file too large: %s (%zu bytes)", file_path.c_str(), file_size);
    return result;
  }

  std::ifstream file(file_path, std::ios::binary);
  if (!file.is_open()) {
    result.result = RHIResult::InvalidArgument;
    result.error_message = "Failed to open shader file: " + file_path;
    return result;
  }

  std::string hlsl_source((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
  file.close();

  if (hlsl_source.empty()) {
    result.result = RHIResult::InvalidArgument;
    result.error_message = "Shader file is empty: " + file_path;
    return result;
  }

  if (hlsl_source.size() != file_size) {
    result.result = RHIResult::InvalidArgument;
    result.error_message = "File size mismatch during reading: expected " + std::to_string(file_size) + " bytes, got " + std::to_string(hlsl_source.size()) + " bytes";
    log::error("File size mismatch for: %s", file_path.c_str());
    return result;
  }

  std::filesystem::path shader_path(file_path);
  std::string shader_dir = shader_path.parent_path().string();

  std::vector<std::string> original_include_paths = _include_paths;
  if (!shader_dir.empty()) {
    _include_paths.insert(_include_paths.begin(), shader_dir);
  }

  std::string filename = shader_path.filename().string();
  result = get_or_compile_shader_variant(hlsl_source, entry_point, stage, filename, defines);

  _include_paths = original_include_paths;

  return result;
}

RHIResult ShaderCompiler::initialize_dxc_interfaces_global() {
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

  std::vector<std::string> include_paths = {"./shaders", "./bin/shaders"};
  global_custom_include_handler = new CustomIncludeHandler(global_dxc_utils, include_paths);
  if (!global_custom_include_handler) {
    log::error("Failed to create global custom include handler");
    return RHIResult::OutOfMemory;
  }

  return RHIResult::Success;
}

RHIResult ShaderCompiler::initialize_global_instance() {
  RHIResult dll_result = load_dxc_dll_global();
  if (dll_result != RHIResult::Success) {
    log::error("Failed to load DXC DLL");
    return dll_result;
  }

  HRESULT hr = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);
  if (FAILED(hr) && hr != RPC_E_CHANGED_MODE) {
    log::error("Failed to initialize COM: 0x%08X", static_cast<uint32_t>(hr));
    unload_dxc_dll_global();
    return RHIResult::InvalidArgument;
  }
  global_com_initialized.store(true, std::memory_order_release);

  RHIResult init_result = initialize_dxc_interfaces_global();
  if (init_result != RHIResult::Success) {
    unload_dxc_dll_global();
    return init_result;
  }

  return RHIResult::Success;
}

ShaderCompilationResult ShaderCompiler::compile_hlsl_to_spirv(const std::string& hlsl_source, const std::string& entry_point, RHIShaderStage stage, const std::string& source_name,
  const std::unordered_map<std::string, std::string>& defines) {
  ShaderCompilationResult result;

  if (hlsl_source.empty()) {
    result.result = RHIResult::InvalidArgument;
    result.error_message = "HLSL source code is empty";
    return result;
  }

  if (entry_point.empty()) {
    result.result = RHIResult::InvalidArgument;
    result.error_message = "Entry point is empty";
    return result;
  }

  if (stage != RHIShaderStage::Vertex && stage != RHIShaderStage::Fragment && stage != RHIShaderStage::Compute) {
    result.result = RHIResult::InvalidArgument;
    result.error_message = "Unsupported shader stage";
    return result;
  }

  if (!is_initialized()) {
    result.result = RHIResult::InvalidArgument;
    result.error_message = "Shader compiler not initialized";
    return result;
  }

  ComPtr<IDxcBlobEncoding> source_blob;
  HRESULT hr = _dxc_utils->CreateBlob(hlsl_source.data(), static_cast<uint32_t>(hlsl_source.size()), CP_UTF8, &source_blob);
  if (FAILED(hr)) {
    result.result = RHIResult::ValidationError;
    result.error_message = "Failed to create DXC source blob";
    return result;
  }

  ComPtr<IDxcBlobEncoding> source_name_blob;
  std::wstring source_name_wstr(source_name.begin(), source_name.end());
  hr = _dxc_utils->CreateBlob(source_name_wstr.data(), static_cast<uint32_t>(source_name_wstr.size() * sizeof(wchar_t)), DXC_CP_UTF16, &source_name_blob);
  if (FAILED(hr)) {
    result.result = RHIResult::ValidationError;
    result.error_message = "Failed to create DXC source name blob";
    return result;
  }

  DxcBuffer dxc_buffer = {
    .Ptr = source_blob->GetBufferPointer(),
    .Size = source_blob->GetBufferSize(),
    .Encoding = DXC_CP_UTF8,
  };

  auto preprocess_args = build_dxc_arguments(entry_point, stage, defines, true);
  std::vector<const wchar_t*> preprocess_args_ptr;
  for (const auto& arg : preprocess_args) {
    preprocess_args_ptr.push_back(arg.c_str());
  }

  ComPtr<IDxcResult> preprocess_result;
  hr =
    _dxc_compiler->Compile(&dxc_buffer, preprocess_args_ptr.data(), static_cast<uint32_t>(preprocess_args_ptr.size()), _custom_include_handler, IID_PPV_ARGS(&preprocess_result));
  if (FAILED(hr)) {
    result.result = RHIResult::ValidationError;
    result.error_message = "Preprocessing compilation failed";
    return result;
  }

  ComPtr<IDxcBlobUtf8> preprocess_errors;
  hr = preprocess_result->GetOutput(DXC_OUT_ERRORS, IID_PPV_ARGS(&preprocess_errors), nullptr);
  if (FAILED(hr)) {
    result.result = RHIResult::ValidationError;
    result.error_message = "Failed to get preprocessing errors";
    return result;
  }
  if (preprocess_errors && preprocess_errors->GetStringLength() > 0) {
    result.result = RHIResult::ValidationError;
    result.error_message = std::string(preprocess_errors->GetStringPointer(), preprocess_errors->GetStringLength());
    log::error("Preprocessing failed: %s", result.error_message.c_str());
    return result;
  }

  ComPtr<IDxcBlobUtf8> canonical_hlsl;
  hr = preprocess_result->GetOutput(DXC_OUT_HLSL, IID_PPV_ARGS(&canonical_hlsl), nullptr);
  if (FAILED(hr)) {
    result.result = RHIResult::ValidationError;
    result.error_message = "Failed to get preprocessed HLSL";
    return result;
  }

  if (!canonical_hlsl || canonical_hlsl->GetStringLength() == 0) {
    result.result = RHIResult::ValidationError;
    result.error_message = "Failed to get preprocessed HLSL";
    return result;
  }

  std::string preprocessed_source(canonical_hlsl->GetStringPointer(), canonical_hlsl->GetStringLength());

  ComPtr<IDxcBlobEncoding> preprocessed_blob;
  hr = _dxc_utils->CreateBlob(preprocessed_source.data(), static_cast<uint32_t>(preprocessed_source.size()), DXC_CP_UTF8, &preprocessed_blob);
  if (FAILED(hr)) {
    result.result = RHIResult::ValidationError;
    result.error_message = "Failed to create preprocessed source blob";
    return result;
  }

  DxcBuffer compile_buffer = {
    .Ptr = preprocessed_blob->GetBufferPointer(),
    .Size = preprocessed_blob->GetBufferSize(),
    .Encoding = DXC_CP_UTF8,
  };

  auto compile_args = build_dxc_arguments(entry_point, stage, defines, false);
  std::vector<const wchar_t*> compile_args_ptr;
  for (const auto& arg : compile_args) {
    compile_args_ptr.push_back(arg.c_str());
  }

  ComPtr<IDxcResult> compile_result;
  hr = _dxc_compiler->Compile(&compile_buffer, compile_args_ptr.data(), static_cast<uint32_t>(compile_args_ptr.size()), _custom_include_handler, IID_PPV_ARGS(&compile_result));
  if (FAILED(hr)) {
    result.result = RHIResult::ValidationError;
    result.error_message = "SPIR-V compilation failed";
    return result;
  }

  ComPtr<IDxcBlobUtf8> compile_errors;
  hr = compile_result->GetOutput(DXC_OUT_ERRORS, IID_PPV_ARGS(&compile_errors), nullptr);
  if (FAILED(hr)) {
    result.result = RHIResult::ValidationError;
    result.error_message = "Failed to get compilation errors";
    return result;
  }
  if (compile_errors && compile_errors->GetStringLength() > 0) {
    result.result = RHIResult::ValidationError;
    result.error_message = std::string(compile_errors->GetStringPointer(), compile_errors->GetStringLength());

    parse_dxc_error(result.error_message, result);

    log::error("Compilation failed: %s", result.error_message.c_str());
    return result;
  }

  ComPtr<IDxcBlobUtf8> warnings;
  hr = compile_result->GetOutput(DXC_OUT_ERRORS, IID_PPV_ARGS(&warnings), nullptr);
  if (FAILED(hr)) {
    result.result = RHIResult::ValidationError;
    result.error_message = "Failed to get compilation warnings";
    return result;
  }
  if (warnings && warnings->GetStringLength() > 0) {
    result.warning_message = std::string(warnings->GetStringPointer(), warnings->GetStringLength());
    log::warning("Compilation warnings: %s", result.warning_message.c_str());
  }

  ComPtr<IDxcBlob> spirv_blob;
  hr = compile_result->GetOutput(DXC_OUT_OBJECT, IID_PPV_ARGS(&spirv_blob), nullptr);
  if (FAILED(hr)) {
    result.result = RHIResult::ValidationError;
    result.error_message = "Failed to get SPIR-V output";
    return result;
  }

  if (!spirv_blob) {
    result.result = RHIResult::ValidationError;
    result.error_message = "Failed to get SPIR-V output from compilation";
    return result;
  }

  const uint8_t* spirv_data = reinterpret_cast<const uint8_t*>(spirv_blob->GetBufferPointer());
  size_t spirv_size = spirv_blob->GetBufferSize();
  result.spirv_data.assign(spirv_data, spirv_data + spirv_size);

  result.result = RHIResult::Success;

  return result;
}

RHIResult ShaderCompiler::reflect_spirv(const std::vector<uint8_t>& spirv_data, ShaderReflectionInfo& reflection_info) {
  log::warning("SPIR-V reflection not yet implemented");

  reflection_info.resources.clear();
  reflection_info.local_size_x = 1;
  reflection_info.local_size_y = 1;
  reflection_info.local_size_z = 1;
  reflection_info.supports_atomics = false;
  reflection_info.supports_subgroups = false;

  return RHIResult::Success;
}

std::string ShaderCompiler::generate_bindless_hlsl_wrapper(const std::string& user_shader_code, const ShaderReflectionInfo& reflection_info) {
  log::warning("Bindless HLSL wrapper generation not yet implemented");
  return user_shader_code;
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

RHIResult ShaderCompiler::load_dxc_dll_global() {
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

  std::vector<std::string> system_paths = {"C:\\Program Files\\Microsoft DirectX Shader Compiler\\", "C:\\Program Files (x86)\\Microsoft DirectX Shader Compiler\\",
    "C:\\Windows\\System32\\", "C:\\Windows\\SysWOW64\\"};
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

  if (!global_dxc_dll) {
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

  if (!global_dxc_create_instance) {
    log::error("Failed to get DxcCreateInstance function from DXC DLL");
    log::error("The loaded DLL may not be a valid DXC installation.");
    unload_dxc_dll_global();
    return RHIResult::NotImplemented;
  }

  return RHIResult::Success;
}

void ShaderCompiler::unload_dxc_dll_global() {
  std::lock_guard<std::mutex> lock(global_dll_mutex);
  if (global_dxc_dll) {
    FreeLibrary(global_dxc_dll);
    global_dxc_dll = nullptr;
    global_dxc_create_instance = nullptr;
  }
}

void ShaderCompiler::unload_dxc_dll() {
}

std::vector<std::wstring> ShaderCompiler::build_dxc_arguments(const std::string& entry_point, RHIShaderStage stage, const std::unordered_map<std::string, std::string>& defines,
  bool for_preprocessing) {
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
    arguments.emplace_back(L"-fspv-target-env=vulkan1.2");
    arguments.emplace_back(L"-fspv-extension=SPV_EXT_descriptor_indexing");
    arguments.emplace_back(L"-fspv-extension=SPV_KHR_ray_query");
    arguments.emplace_back(L"-enable-16bit-types");
    arguments.emplace_back(L"-O3");

    constexpr const wchar_t* shift_args[] = {L"-fvk-s-shift", L"-fvk-t-shift", L"-fvk-b-shift", L"-fvk-u-shift"};
    constexpr const wchar_t* offset_args[] = {L"0", L"0", L"0", L"0"};

    for (uint32_t i = 0u; i < std::size(shift_args); i++) {
      arguments.emplace_back(shift_args[i]);
      arguments.emplace_back(offset_args[i]);
      arguments.emplace_back(L"all");
    }
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

CustomIncludeHandler::CustomIncludeHandler(Microsoft::WRL::ComPtr<IDxcUtils> dxc_utils, const std::vector<std::string>& include_paths)
  : _dxc_utils(dxc_utils)
  , _include_paths(include_paths) {
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

  std::ifstream file(full_path, std::ios::binary);
  if (!file.is_open()) {
    log::error("Failed to open include file: %s", full_path.c_str());
    return E_FAIL;
  }

  std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
  file.close();

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
  if (std::filesystem::exists(filename)) {
    return std::filesystem::absolute(filename).string();
  }

  for (const auto& path : _include_paths) {
    std::filesystem::path include_path = path;
    include_path /= filename;

    if (std::filesystem::exists(include_path)) {
      return std::filesystem::absolute(include_path).string();
    }
  }

  return "";
}

std::string CustomIncludeHandler::wstring_to_string(const std::wstring& wstr) {
  if (wstr.empty())
    return "";

  int size_needed = WideCharToMultiByte(CP_UTF8, 0, wstr.c_str(), static_cast<int>(wstr.size()), nullptr, 0, nullptr, nullptr);
  std::string str(size_needed, 0);
  WideCharToMultiByte(CP_UTF8, 0, wstr.c_str(), static_cast<int>(wstr.size()), &str[0], size_needed, nullptr, nullptr);
  return str;
}

std::wstring CustomIncludeHandler::string_to_wstring(const std::string& str) {
  if (str.empty())
    return L"";

  int size_needed = MultiByteToWideChar(CP_UTF8, 0, str.c_str(), static_cast<int>(str.size()), nullptr, 0);
  std::wstring wstr(size_needed, 0);
  MultiByteToWideChar(CP_UTF8, 0, str.c_str(), static_cast<int>(str.size()), &wstr[0], size_needed);
  return wstr;
}

std::string ShaderCompiler::read_file_content(const std::string& file_path, std::string& error_message) {
  if (file_path.empty()) {
    error_message = "File path is empty";
    return "";
  }

  std::ifstream file(file_path, std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    error_message = "Failed to open file: " + file_path;
    return "";
  }

  std::streamsize size = file.tellg();

  if (size > static_cast<std::streamsize>(MAX_SHADER_FILE_SIZE)) {
    error_message = "File too large: " + std::to_string(size) + " bytes (max: " + std::to_string(MAX_SHADER_FILE_SIZE) + " bytes)";
    return "";
  }

  if (size < 0) {
    error_message = "Invalid file size: " + file_path;
    return "";
  }

  file.seekg(0, std::ios::beg);

  std::string content(size, '\0');
  if (!file.read(&content[0], size)) {
    error_message = "Failed to read file: " + file_path;
    return "";
  }

  error_message.clear();
  return content;
}

}  // namespace etx
