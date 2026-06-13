#include <etx/rhi/shader/shader_compiler.hxx>
#include <etx/rhi/shader/dxc_com_ptr.hxx>

#include <etx/core/log.hxx>
#include <etx/core/platform.hxx>
#include <etx/core/environment.hxx>
#include <spirv_msl.hpp>
#include <atomic>
#include <array>
#include <chrono>
#include <codecvt>
#include <cstring>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <locale>
#include <mutex>
#include <regex>
#include <sstream>
#include <string_view>
#include <thread>
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
constexpr uint32_t kShaderCacheVersion = 3u;
constexpr std::string_view kShaderVariantCacheMagic = "ETXSHV1";
constexpr std::string_view kPreprocessedShaderCacheMagic = "ETXSHP1";

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

int64_t filesystem_timestamp_ticks(const std::filesystem::file_time_type& time_point) {
  return static_cast<int64_t>(time_point.time_since_epoch().count());
}

std::string format_hash_hex(uint64_t value) {
  char buffer[32] = {};
  snprintf(buffer, sizeof(buffer), "%016llx", static_cast<unsigned long long>(value));
  return std::string(buffer);
}

template <typename T>
bool write_pod(std::ofstream& stream, const T& value) {
  stream.write(reinterpret_cast<const char*>(&value), static_cast<std::streamsize>(sizeof(T)));
  return stream.good();
}

template <typename T>
bool read_pod(std::ifstream& stream, T& value) {
  stream.read(reinterpret_cast<char*>(&value), static_cast<std::streamsize>(sizeof(T)));
  return stream.good();
}

bool write_string(std::ofstream& stream, const std::string& value) {
  const uint32_t size = static_cast<uint32_t>(value.size());
  return write_pod(stream, size) && (size == 0u || (stream.write(value.data(), static_cast<std::streamsize>(size)), stream.good()));
}

bool read_string(std::ifstream& stream, std::string& value) {
  uint32_t size = 0u;
  if (read_pod(stream, size) == false) {
    return false;
  }

  value.resize(size);
  if (size == 0u) {
    return true;
  }

  stream.read(value.data(), static_cast<std::streamsize>(size));
  return stream.good();
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
  const std::regex entry_regex("\\[\\s*numthreads\\s*\\(\\s*(\\d+)\\s*,\\s*(\\d+)\\s*,\\s*(\\d+)\\s*\\)\\s*\\]\\s*[A-Za-z_][A-Za-z0-9_<>]*\\s+" + escaped_entry + "\\s*\\(",
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

  std::vector<std::string> dependencies() const;

 private:
  void record_dependency(const std::string& full_path);

  DxcComPtr<IDxcUtils> _dxc_utils;
  std::string _shader_directory;
  std::unordered_map<std::string, DxcComPtr<IDxcBlobEncoding>> _include_cache;
  std::mutex _include_cache_mutex;
  std::vector<std::string> _dependencies;
  mutable std::mutex _dependency_mutex;
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

struct PreprocessedShaderKey {
  std::string source_name;
  std::map<std::string, std::string> defines;
  uint64_t source_hash = 0u;

  bool operator==(const PreprocessedShaderKey& other) const {
    return source_name == other.source_name && defines == other.defines && source_hash == other.source_hash;
  }
};

struct PreprocessedShaderKeyHash {
  std::size_t operator()(const PreprocessedShaderKey& key) const {
    std::size_t h = 0;
    h = etx_hash64_continue(key.source_name.data(), key.source_name.size(), h);
    for (const auto& define : key.defines) {
      h = etx_hash64_continue(define.first.data(), define.first.size(), h);
      h = etx_hash64_continue(define.second.data(), define.second.size(), h);
    }
    h = etx_hash64_continue(&key.source_hash, sizeof(key.source_hash), h);
    return h;
  }
};

struct FileDependencyInfo {
  std::string path;
  uint64_t file_size = 0u;
  int64_t timestamp_ticks = 0;
};

struct PreprocessedShaderValue {
  std::string source;
  uint64_t source_hash = 0u;
  std::vector<FileDependencyInfo> dependencies;
};

struct ShaderVariantDiskCacheHeader {
  char magic[8] = {};
  uint32_t version = kShaderCacheVersion;
  uint32_t stage = 0u;
  uint32_t backend = 0u;
  uint32_t format = 0u;
  uint64_t key_hash = 0u;
  uint64_t data_size = 0u;
};

struct PreprocessedShaderDiskCacheHeader {
  char magic[8] = {};
  uint32_t version = kShaderCacheVersion;
  uint64_t key_hash = 0u;
  uint64_t source_hash = 0u;
  uint32_t dependency_count = 0u;
  uint64_t source_size = 0u;
};

uint64_t shader_variant_cache_hash(const ShaderVariantKey& key) {
  return static_cast<uint64_t>(ShaderVariantKeyHash{}(key));
}

uint64_t preprocessed_shader_cache_hash(const PreprocessedShaderKey& key) {
  return static_cast<uint64_t>(PreprocessedShaderKeyHash{}(key));
}

std::filesystem::path shader_cache_root_directory() {
  std::filesystem::path root(env().data_folder());
  root /= "cache";
  root /= "shaders";
  root /= ("v" + std::to_string(kShaderCacheVersion));
  return root;
}

std::filesystem::path shader_variant_cache_directory() {
  return shader_cache_root_directory() / "variants";
}

std::filesystem::path shader_preprocessed_cache_directory() {
  return shader_cache_root_directory() / "preprocessed";
}

std::filesystem::path shader_variant_cache_file_path(uint64_t key_hash) {
  return shader_variant_cache_directory() / (format_hash_hex(key_hash) + ".bin");
}

std::filesystem::path shader_preprocessed_cache_file_path(uint64_t key_hash) {
  return shader_preprocessed_cache_directory() / (format_hash_hex(key_hash) + ".bin");
}

bool query_file_dependency_info(const std::string& path, FileDependencyInfo& out_info) {
  std::error_code ec;
  if (path.empty()) {
    return false;
  }

  const std::filesystem::path fs_path(path);
  if (std::filesystem::exists(fs_path, ec) == false || ec.value() != 0) {
    return false;
  }

  const uintmax_t file_size = std::filesystem::file_size(fs_path, ec);
  if (ec.value() != 0) {
    return false;
  }

  const auto timestamp = std::filesystem::last_write_time(fs_path, ec);
  if (ec.value() != 0) {
    return false;
  }

  out_info.path = path;
  out_info.file_size = static_cast<uint64_t>(file_size);
  out_info.timestamp_ticks = filesystem_timestamp_ticks(timestamp);
  return true;
}

bool dependencies_are_current(const std::vector<FileDependencyInfo>& dependencies) {
  for (const auto& dependency : dependencies) {
    FileDependencyInfo current = {};
    if (query_file_dependency_info(dependency.path, current) == false) {
      return false;
    }
    if ((current.file_size != dependency.file_size) || (current.timestamp_ticks != dependency.timestamp_ticks)) {
      return false;
    }
  }

  return true;
}

bool ensure_cache_directory_exists(const std::filesystem::path& directory) {
  std::error_code ec;
  std::filesystem::create_directories(directory, ec);
  return ec.value() == 0;
}

bool load_shader_variant_from_disk(uint64_t key_hash, RHIShaderStage stage, RHIBackend backend, RHIShaderBinaryFormat format, std::vector<uint8_t>& out_binary) {
  std::ifstream stream(shader_variant_cache_file_path(key_hash), std::ios::binary);
  if (stream.is_open() == false) {
    return false;
  }

  ShaderVariantDiskCacheHeader header = {};
  if (read_pod(stream, header) == false) {
    return false;
  }

  if ((std::string_view(header.magic, kShaderVariantCacheMagic.size()) != kShaderVariantCacheMagic) || (header.version != kShaderCacheVersion) || (header.key_hash != key_hash) ||
      (header.stage != static_cast<uint32_t>(stage)) || (header.backend != static_cast<uint32_t>(backend)) || (header.format != static_cast<uint32_t>(format))) {
    return false;
  }

  out_binary.resize(static_cast<size_t>(header.data_size));
  if (header.data_size > 0u) {
    stream.read(reinterpret_cast<char*>(out_binary.data()), static_cast<std::streamsize>(header.data_size));
  }

  return stream.good() || stream.eof();
}

bool store_shader_variant_to_disk(uint64_t key_hash, RHIShaderStage stage, RHIBackend backend, RHIShaderBinaryFormat format, const std::vector<uint8_t>& binary) {
  if (ensure_cache_directory_exists(shader_variant_cache_directory()) == false) {
    return false;
  }

  std::ofstream stream(shader_variant_cache_file_path(key_hash), std::ios::binary | std::ios::trunc);
  if (stream.is_open() == false) {
    return false;
  }

  ShaderVariantDiskCacheHeader header = {};
  std::memcpy(header.magic, kShaderVariantCacheMagic.data(), kShaderVariantCacheMagic.size());
  header.version = kShaderCacheVersion;
  header.stage = static_cast<uint32_t>(stage);
  header.backend = static_cast<uint32_t>(backend);
  header.format = static_cast<uint32_t>(format);
  header.key_hash = key_hash;
  header.data_size = static_cast<uint64_t>(binary.size());

  if (write_pod(stream, header) == false) {
    return false;
  }

  if (binary.empty() == false) {
    stream.write(reinterpret_cast<const char*>(binary.data()), static_cast<std::streamsize>(binary.size()));
  }

  return stream.good();
}

bool load_preprocessed_shader_from_disk(uint64_t key_hash, PreprocessedShaderValue& out_value) {
  std::ifstream stream(shader_preprocessed_cache_file_path(key_hash), std::ios::binary);
  if (stream.is_open() == false) {
    return false;
  }

  PreprocessedShaderDiskCacheHeader header = {};
  if (read_pod(stream, header) == false) {
    return false;
  }

  if ((std::string_view(header.magic, kPreprocessedShaderCacheMagic.size()) != kPreprocessedShaderCacheMagic) || (header.version != kShaderCacheVersion) ||
      (header.key_hash != key_hash)) {
    return false;
  }

  out_value.dependencies.resize(header.dependency_count);
  for (auto& dependency : out_value.dependencies) {
    if (read_string(stream, dependency.path) == false || read_pod(stream, dependency.file_size) == false || read_pod(stream, dependency.timestamp_ticks) == false) {
      return false;
    }
  }

  out_value.source.resize(static_cast<size_t>(header.source_size));
  if (header.source_size > 0u) {
    stream.read(out_value.source.data(), static_cast<std::streamsize>(header.source_size));
    if (stream.good() == false && stream.eof() == false) {
      return false;
    }
  }
  out_value.source_hash = header.source_hash;

  if (dependencies_are_current(out_value.dependencies) == false) {
    out_value = {};
    return false;
  }

  return true;
}

bool store_preprocessed_shader_to_disk(uint64_t key_hash, const PreprocessedShaderValue& value) {
  if (ensure_cache_directory_exists(shader_preprocessed_cache_directory()) == false) {
    return false;
  }

  std::ofstream stream(shader_preprocessed_cache_file_path(key_hash), std::ios::binary | std::ios::trunc);
  if (stream.is_open() == false) {
    return false;
  }

  PreprocessedShaderDiskCacheHeader header = {};
  std::memcpy(header.magic, kPreprocessedShaderCacheMagic.data(), kPreprocessedShaderCacheMagic.size());
  header.version = kShaderCacheVersion;
  header.key_hash = key_hash;
  header.source_hash = value.source_hash;
  header.dependency_count = static_cast<uint32_t>(value.dependencies.size());
  header.source_size = static_cast<uint64_t>(value.source.size());

  if (write_pod(stream, header) == false) {
    return false;
  }

  for (const auto& dependency : value.dependencies) {
    if (write_string(stream, dependency.path) == false || write_pod(stream, dependency.file_size) == false || write_pod(stream, dependency.timestamp_ticks) == false) {
      return false;
    }
  }

  if (value.source.empty() == false) {
    stream.write(value.source.data(), static_cast<std::streamsize>(value.source.size()));
  }

  return stream.good();
}

struct ShaderCompiler::Impl {
  std::unordered_map<ShaderVariantKey, ShaderCompilationResult, ShaderVariantKeyHash> shader_cache;
  std::mutex cache_mutex;
  std::unordered_map<PreprocessedShaderKey, PreprocessedShaderValue, PreprocessedShaderKeyHash> preprocessed_cache;
  std::mutex preprocessed_cache_mutex;
  ShaderCompilerStatistics statistics = {};
  std::mutex statistics_mutex;

  DxcComPtr<IDxcUtils> dxc_utils;
  DxcComPtr<IDxcCompiler3> dxc_compiler;

  // Helper method
  std::vector<std::wstring> build_dxc_arguments(const std::string& entry_point, RHIShaderStage stage, const std::unordered_map<std::string, std::string>& defines,
    const std::vector<std::string>& include_directories, bool for_preprocessing = false);

  void accumulate_statistics(const ShaderCompilerStatistics& delta) {
    std::lock_guard<std::mutex> lock(statistics_mutex);
    statistics.compile_calls += delta.compile_calls;
    statistics.requested_entry_points += delta.requested_entry_points;
    statistics.compiled_entry_points += delta.compiled_entry_points;
    statistics.preprocessed_memory_cache_hits += delta.preprocessed_memory_cache_hits;
    statistics.preprocessed_disk_cache_hits += delta.preprocessed_disk_cache_hits;
    statistics.shader_memory_cache_hits += delta.shader_memory_cache_hits;
    statistics.shader_disk_cache_hits += delta.shader_disk_cache_hits;
    statistics.preprocess_invocations += delta.preprocess_invocations;
    statistics.dxc_compile_invocations += delta.dxc_compile_invocations;
    statistics.spirv_to_msl_translations += delta.spirv_to_msl_translations;
    statistics.cache_writes += delta.cache_writes;
    statistics.total_wall_time_ms += delta.total_wall_time_ms;
    statistics.preprocess_time_ms += delta.preprocess_time_ms;
    statistics.dxc_compile_time_ms += delta.dxc_compile_time_ms;
    statistics.spirv_to_msl_time_ms += delta.spirv_to_msl_time_ms;
    statistics.cache_read_time_ms += delta.cache_read_time_ms;
    statistics.cache_write_time_ms += delta.cache_write_time_ms;
  }
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
std::atomic<uint32_t> global_dxc_thread_context_count{0u};

// Singleton implementation - Meyer's singleton with thread-safe initialization
struct ThreadLocalDxcContext {
#if (ETX_PLATFORM_WINDOWS)
  bool com_initialized = false;
#endif
  DxcComPtr<IDxcUtils> dxc_utils = {};
  DxcComPtr<IDxcCompiler3> dxc_compiler = {};
  bool valid = false;

  ThreadLocalDxcContext() {
    HRESULT hr = S_OK;
#if (ETX_PLATFORM_WINDOWS)
    hr = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    if (SUCCEEDED(hr)) {
      com_initialized = true;
    } else if (hr != RPC_E_CHANGED_MODE) {
      log::error("Failed to initialize COM for shader compilation thread: 0x%08X", static_cast<uint32_t>(hr));
      return;
    }
#endif

    if (global_dxc_create_instance == nullptr) {
      return;
    }

    hr = global_dxc_create_instance(CLSID_DxcUtils, IID_PPV_ARGS(dxc_utils.GetAddressOf()));
    if (FAILED(hr)) {
      log::error("Failed to create thread-local DXC utils: 0x%08X", static_cast<uint32_t>(hr));
      return;
    }

    hr = global_dxc_create_instance(CLSID_DxcCompiler, IID_PPV_ARGS(dxc_compiler.GetAddressOf()));
    if (FAILED(hr)) {
      log::error("Failed to create thread-local DXC compiler: 0x%08X", static_cast<uint32_t>(hr));
      dxc_utils.Reset();
      return;
    }

    valid = true;
  }

  ~ThreadLocalDxcContext() {
    dxc_compiler.Reset();
    dxc_utils.Reset();
#if (ETX_PLATFORM_WINDOWS)
    if (com_initialized) {
      CoUninitialize();
    }
#endif
  }
};

struct ThreadLocalDxcContextHolder {
  ThreadLocalDxcContext* context = nullptr;

  ThreadLocalDxcContext& get() {
    if (context == nullptr) {
      context = new ThreadLocalDxcContext();
      global_dxc_thread_context_count.fetch_add(1u, std::memory_order_acq_rel);
    }

    return *context;
  }

  void reset() {
    if (context == nullptr) {
      return;
    }

    delete context;
    context = nullptr;
    global_dxc_thread_context_count.fetch_sub(1u, std::memory_order_acq_rel);
  }

  ~ThreadLocalDxcContextHolder() {
    reset();
  }
};

ThreadLocalDxcContextHolder& thread_local_dxc_context_holder() {
  thread_local ThreadLocalDxcContextHolder holder = {};
  return holder;
}

ThreadLocalDxcContext& thread_local_dxc_context() {
  return thread_local_dxc_context_holder().get();
}

void reset_thread_local_dxc_context() {
  thread_local_dxc_context_holder().reset();
}

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
    {
      std::lock_guard<std::mutex> cache_lock(_impl->cache_mutex);
      _impl->shader_cache.clear();
    }
    {
      std::lock_guard<std::mutex> preprocess_lock(_impl->preprocessed_cache_mutex);
      _impl->preprocessed_cache.clear();
    }
    {
      std::lock_guard<std::mutex> stats_lock(_impl->statistics_mutex);
      _impl->statistics = {};
    }
  }
  return;
#endif

  reset_thread_local_dxc_context();

  std::lock_guard<std::mutex> dll_lock(global_dll_mutex);

  if (_impl != nullptr) {
    {
      std::lock_guard<std::mutex> cache_lock(_impl->cache_mutex);
      _impl->shader_cache.clear();
    }
    {
      std::lock_guard<std::mutex> preprocess_lock(_impl->preprocessed_cache_mutex);
      _impl->preprocessed_cache.clear();
    }
    {
      std::lock_guard<std::mutex> stats_lock(_impl->statistics_mutex);
      _impl->statistics = {};
    }
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
    const uint32_t remaining_thread_contexts = global_dxc_thread_context_count.load(std::memory_order_acquire);
    if (remaining_thread_contexts == 0u) {
      unload_dxc_library(global_dxc_dll);
      global_dxc_dll = nullptr;
    } else {
      log::warning("Skipping DXC runtime unload because %u thread-local DXC context(s) are still alive", remaining_thread_contexts);
    }
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

ShaderCompilerStatistics ShaderCompiler::statistics() const {
  if (_impl == nullptr) {
    return {};
  }

  std::lock_guard<std::mutex> lock(_impl->statistics_mutex);
  return _impl->statistics;
}

void ShaderCompiler::reset_statistics() {
  if (_impl == nullptr) {
    return;
  }

  std::lock_guard<std::mutex> lock(_impl->statistics_mutex);
  _impl->statistics = {};
}

void ShaderCompiler::log_statistics(const char* label) const {
  const ShaderCompilerStatistics stats = statistics();
  const char* stats_label = (label != nullptr && label[0] != 0) ? label : "global";
  log::info(
    "Shader compiler stats [%s]: calls=%llu entries=%llu compiled=%llu preprocess(mem=%llu,disk=%llu,dxc=%llu) variants(mem=%llu,disk=%llu,dxc=%llu) msl=%llu writes=%llu "
    "time_ms(total=%.2f preprocess=%.2f dxc=%.2f msl=%.2f cache_read=%.2f cache_write=%.2f)",
    stats_label, static_cast<unsigned long long>(stats.compile_calls), static_cast<unsigned long long>(stats.requested_entry_points),
    static_cast<unsigned long long>(stats.compiled_entry_points), static_cast<unsigned long long>(stats.preprocessed_memory_cache_hits),
    static_cast<unsigned long long>(stats.preprocessed_disk_cache_hits), static_cast<unsigned long long>(stats.preprocess_invocations),
    static_cast<unsigned long long>(stats.shader_memory_cache_hits), static_cast<unsigned long long>(stats.shader_disk_cache_hits),
    static_cast<unsigned long long>(stats.dxc_compile_invocations), static_cast<unsigned long long>(stats.spirv_to_msl_translations),
    static_cast<unsigned long long>(stats.cache_writes), stats.total_wall_time_ms, stats.preprocess_time_ms, stats.dxc_compile_time_ms, stats.spirv_to_msl_time_ms,
    stats.cache_read_time_ms, stats.cache_write_time_ms);
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
  ShaderCompilerStatistics local_stats = {};
  local_stats.compile_calls = 1u;
  local_stats.requested_entry_points = static_cast<uint64_t>(entry_points.size());
  const auto total_begin = std::chrono::steady_clock::now();
  auto finalize_result = [&](MultiShaderCompilationResult&& final_result) {
    const auto total_end = std::chrono::steady_clock::now();
    local_stats.total_wall_time_ms += std::chrono::duration<double, std::milli>(total_end - total_begin).count();
    _impl->accumulate_statistics(local_stats);
    return std::move(final_result);
  };

  if (hlsl_source.empty()) {
    result.result = RHIResult::InvalidArgument;
    result.error_message = "HLSL source code is empty";
    return finalize_result(std::move(result));
  }

  if (entry_points.empty()) {
    result.result = RHIResult::InvalidArgument;
    result.error_message = "No entry points provided";
    return finalize_result(std::move(result));
  }

  if (hlsl_source.size() > MAX_SHADER_SOURCE_SIZE) {
    result.result = RHIResult::InvalidArgument;
    result.error_message = "HLSL source exceeds max size: " + std::to_string(hlsl_source.size()) + " bytes (max: " + std::to_string(MAX_SHADER_SOURCE_SIZE) + " bytes)";
    return finalize_result(std::move(result));
  }

  if (is_initialized() == false) {
    result.result = RHIResult::InvalidArgument;
    result.error_message = "Shader compiler not initialized";
    return finalize_result(std::move(result));
  }

  auto& dxc_context = thread_local_dxc_context();
  if (dxc_context.valid == false) {
    result.result = RHIResult::ValidationError;
    result.error_message = "Thread-local DXC initialization failed";
    return finalize_result(std::move(result));
  }

  std::string shader_directory;
  if (source_name.empty() == false) {
    std::filesystem::path source_path(source_name);
    if (source_path.has_parent_path()) {
      shader_directory = source_path.parent_path().string();
    }
  }

  auto include_handler =
    std::unique_ptr<CustomIncludeHandler, void (*)(CustomIncludeHandler*)>(new CustomIncludeHandler(dxc_context.dxc_utils, shader_directory), [](CustomIncludeHandler* p) {
      if (p != nullptr) {
        p->Release();
      }
    });
  const std::vector<std::string> include_directories = build_shader_include_directories(source_name);

  if (include_handler == nullptr) {
    result.result = RHIResult::OutOfMemory;
    result.error_message = "Failed to create include handler";
    return finalize_result(std::move(result));
  }

  std::map<std::string, std::string> ordered_defines(defines.begin(), defines.end());
  const uint64_t source_input_hash = etx_hash64(hlsl_source.data(), hlsl_source.size());
  const PreprocessedShaderKey preprocessed_key = {source_name, ordered_defines, source_input_hash};
  const uint64_t preprocessed_key_hash = preprocessed_shader_cache_hash(preprocessed_key);

  PreprocessedShaderValue preprocessed_value = {};
  bool have_preprocessed_source = false;
  {
    std::lock_guard<std::mutex> lock(_impl->preprocessed_cache_mutex);
    auto it = _impl->preprocessed_cache.find(preprocessed_key);
    if (it != _impl->preprocessed_cache.end()) {
      preprocessed_value = it->second;
      have_preprocessed_source = true;
      local_stats.preprocessed_memory_cache_hits += 1u;
    }
  }

  if (have_preprocessed_source == false) {
    const auto disk_read_begin = std::chrono::steady_clock::now();
    if (load_preprocessed_shader_from_disk(preprocessed_key_hash, preprocessed_value)) {
      const auto disk_read_end = std::chrono::steady_clock::now();
      local_stats.cache_read_time_ms += std::chrono::duration<double, std::milli>(disk_read_end - disk_read_begin).count();
      local_stats.preprocessed_disk_cache_hits += 1u;
      have_preprocessed_source = true;
      std::lock_guard<std::mutex> lock(_impl->preprocessed_cache_mutex);
      _impl->preprocessed_cache[preprocessed_key] = preprocessed_value;
    }
  }

  if (have_preprocessed_source == false) {
    const auto preprocess_begin = std::chrono::steady_clock::now();
    local_stats.preprocess_invocations += 1u;

    DxcComPtr<IDxcBlobEncoding> source_blob;
    HRESULT hr = dxc_context.dxc_utils->CreateBlob(hlsl_source.data(), static_cast<uint32_t>(hlsl_source.size()), DXC_CP_UTF8, source_blob.ReleaseAndGetAddressOf());
    if (FAILED(hr)) {
      result.result = RHIResult::ValidationError;
      result.error_message = "Failed to create DXC source blob";
      return finalize_result(std::move(result));
    }

    DxcBuffer dxc_buffer = {
      .Ptr = source_blob->GetBufferPointer(),
      .Size = source_blob->GetBufferSize(),
      .Encoding = DXC_CP_UTF8,
    };

    auto preprocess_args = _impl->build_dxc_arguments(entry_points[0].entry_point, entry_points[0].stage, defines, include_directories, true);
    std::vector<const wchar_t*> preprocess_args_ptr;
    for (const auto& arg : preprocess_args) {
      preprocess_args_ptr.push_back(arg.c_str());
    }

    DxcComPtr<IDxcResult> preprocess_result;
    preprocess_result.Reset();
    hr = dxc_context.dxc_compiler->Compile(&dxc_buffer, preprocess_args_ptr.data(), static_cast<uint32_t>(preprocess_args_ptr.size()), include_handler.get(),
      IID_PPV_ARGS(preprocess_result.GetAddressOf()));
    if (FAILED(hr)) {
      result.result = RHIResult::ValidationError;
      result.error_message = "Preprocessing compilation failed";
      return finalize_result(std::move(result));
    }

    HRESULT preprocess_status = S_OK;
    hr = preprocess_result->GetStatus(&preprocess_status);
    if (FAILED(hr)) {
      result.result = RHIResult::ValidationError;
      result.error_message = "Failed to get preprocessing status";
      return finalize_result(std::move(result));
    }

    std::string preprocess_diagnostics = {};
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
      return finalize_result(std::move(result));
    }

    DxcComPtr<IDxcBlobUtf8> canonical_hlsl_blob;
    canonical_hlsl_blob.Reset();
    hr = preprocess_result->GetOutput(DXC_OUT_HLSL, IID_PPV_ARGS(canonical_hlsl_blob.GetAddressOf()), nullptr);
    if ((FAILED(hr)) || (canonical_hlsl_blob.Get() == nullptr) || (canonical_hlsl_blob->GetStringLength() == 0)) {
      result.result = RHIResult::ValidationError;
      result.error_message = "Failed to get preprocessed HLSL";
      return finalize_result(std::move(result));
    }

    preprocessed_value.source.assign(canonical_hlsl_blob->GetStringPointer(), canonical_hlsl_blob->GetStringLength());
    if (preprocessed_value.source.size() > MAX_SHADER_SOURCE_SIZE) {
      result.result = RHIResult::ValidationError;
      result.error_message = "Preprocessed HLSL exceeds max size: " + std::to_string(preprocessed_value.source.size()) + " bytes (max: " +
                             std::to_string(MAX_SHADER_SOURCE_SIZE) + " bytes)";
      return finalize_result(std::move(result));
    }

    preprocessed_value.source_hash = etx_hash64(preprocessed_value.source.data(), preprocessed_value.source.size());

    std::vector<std::string> dependency_paths = include_handler->dependencies();
    if (source_name.empty() == false) {
      dependency_paths.insert(dependency_paths.begin(), source_name);
    }

    for (const auto& dependency_path : dependency_paths) {
      FileDependencyInfo dependency = {};
      if (query_file_dependency_info(dependency_path, dependency)) {
        bool duplicate = false;
        for (const auto& existing : preprocessed_value.dependencies) {
          if (existing.path == dependency.path) {
            duplicate = true;
            break;
          }
        }
        if (duplicate == false) {
          preprocessed_value.dependencies.push_back(std::move(dependency));
        }
      }
    }

    const auto preprocess_end = std::chrono::steady_clock::now();
    local_stats.preprocess_time_ms += std::chrono::duration<double, std::milli>(preprocess_end - preprocess_begin).count();

    {
      std::lock_guard<std::mutex> lock(_impl->preprocessed_cache_mutex);
      _impl->preprocessed_cache[preprocessed_key] = preprocessed_value;
    }

    if (source_name.empty() == false) {
      const auto disk_write_begin = std::chrono::steady_clock::now();
      if (store_preprocessed_shader_to_disk(preprocessed_key_hash, preprocessed_value)) {
        const auto disk_write_end = std::chrono::steady_clock::now();
        local_stats.cache_write_time_ms += std::chrono::duration<double, std::milli>(disk_write_end - disk_write_begin).count();
        local_stats.cache_writes += 1u;
      }
    }
  }

  const std::string& preprocessed_source = preprocessed_value.source;

  if (contains_include_directive(preprocessed_source)) {
    log::warning("Preprocessed shader source still contains #include directives [%s]", source_name.empty() ? "<memory>" : source_name.c_str());
  }

  const uint64_t source_hash = preprocessed_value.source_hash;

  result.binaries.resize(entry_points.size());

  for (uint32_t i = 0, e = entry_points.size(); i < e; ++i) {
    const auto& ep = entry_points[i];
    ShaderVariantKey key{source_name, ep.entry_point, ep.stage, backend, ordered_defines, source_hash};
    const uint64_t key_hash = shader_variant_cache_hash(key);
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
        local_stats.shader_memory_cache_hits += 1u;
      }
    }

    RHIShaderBinaryFormat binary_format = (backend == RHIBackend::Metal) ? RHIShaderBinaryFormat::MetalSource : RHIShaderBinaryFormat::SpirV;

    if (cached_spirv.empty()) {
      const auto disk_read_begin = std::chrono::steady_clock::now();
      if (load_shader_variant_from_disk(key_hash, ep.stage, backend, binary_format, cached_spirv)) {
        const auto disk_read_end = std::chrono::steady_clock::now();
        local_stats.cache_read_time_ms += std::chrono::duration<double, std::milli>(disk_read_end - disk_read_begin).count();
        local_stats.shader_disk_cache_hits += 1u;

        ShaderCompilationResult partial_res = {};
        partial_res.result = RHIResult::Success;
        partial_res.spirv_data = cached_spirv;
        std::lock_guard<std::mutex> lock(_impl->cache_mutex);
        _impl->shader_cache[key] = std::move(partial_res);
      }
    }

    if (cached_spirv.empty() == false) {
      result.binaries[i].spirv_data = nullptr;  // Fix up pointers later
      result.binaries[i].spirv_size = cached_spirv.size();
      result.binaries[i].stage = ep.stage;
      result.binaries[i].backend = backend;
      result.binaries[i].format = binary_format;
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
      return finalize_result(std::move(result));
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
    const auto compile_begin = std::chrono::steady_clock::now();
    HRESULT hr = dxc_context.dxc_compiler->Compile(&source_buffer, arguments_ptr.data(), (uint32_t)arguments_ptr.size(), nullptr, IID_PPV_ARGS(compile_result.GetAddressOf()));
    const auto compile_end = std::chrono::steady_clock::now();
    local_stats.dxc_compile_time_ms += std::chrono::duration<double, std::milli>(compile_end - compile_begin).count();
    local_stats.dxc_compile_invocations += 1u;
    local_stats.compiled_entry_points += 1u;

    ShaderCompilationResult entry_result = {
      .result = RHIResult::Success,
    };

    if ((FAILED(hr)) || (compile_result.Get() == nullptr)) {
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
      return finalize_result(std::move(result));
    }

    DxcComPtr<IDxcBlob> shader_obj;
    HRESULT object_hr = compile_result->GetOutput(DXC_OUT_OBJECT, IID_PPV_ARGS(shader_obj.GetAddressOf()), nullptr);
    if ((FAILED(object_hr)) || (shader_obj.Get() == nullptr) || (shader_obj->GetBufferPointer() == nullptr) || (shader_obj->GetBufferSize() == 0)) {
      result.result = RHIResult::ValidationError;
      result.error_message = entry_result.error_message.empty() ? "DXC did not produce shader object output" : entry_result.error_message;
      return finalize_result(std::move(result));
    }

    const uint8_t* ptr = static_cast<const uint8_t*>(shader_obj->GetBufferPointer());
    size_t size = shader_obj->GetBufferSize();
    std::vector<uint8_t> final_binary = {};

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
        return finalize_result(std::move(result));
      }
      if (preflight_error.empty() == false) {
        log::warning("Skipping Vulkan-on-macOS shader translation safety preflight for '%s': %s", ep.entry_point.c_str(), preflight_error.c_str());
      }
    }
#endif

    if (backend == RHIBackend::Metal) {
      std::string msl_source = {};
      std::string translation_error = {};
      const auto translate_begin = std::chrono::steady_clock::now();
      if (translate_spirv_to_msl(spirv_binary, msl_source, translation_error) == false) {
        result.result = RHIResult::ValidationError;
        result.error_message = translation_error;
        return finalize_result(std::move(result));
      }
      const auto translate_end = std::chrono::steady_clock::now();
      local_stats.spirv_to_msl_time_ms += std::chrono::duration<double, std::milli>(translate_end - translate_begin).count();
      local_stats.spirv_to_msl_translations += 1u;

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

    const auto disk_write_begin = std::chrono::steady_clock::now();
    if (store_shader_variant_to_disk(key_hash, ep.stage, backend, binary_format, partial_res.spirv_data)) {
      const auto disk_write_end = std::chrono::steady_clock::now();
      local_stats.cache_write_time_ms += std::chrono::duration<double, std::milli>(disk_write_end - disk_write_begin).count();
      local_stats.cache_writes += 1u;
    }
  }

  // Fix up pointers
  size_t current_offset = 0;
  for (auto& binary : result.binaries) {
    binary.spirv_data = result.shared_blob.data() + current_offset;
    current_offset += binary.spirv_size;
  }
  return finalize_result(std::move(result));
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
    const auto spirv_opt_config_it = defines.find("ETX_DXC_SPIRV_OPT_CONFIG");

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
    if ((spirv_opt_config_it != defines.end()) && (spirv_opt_config_it->second.empty() == false)) {
      arguments.emplace_back(L"-enable-16bit-types");
      arguments.emplace_back(L"-Oconfig=" + utf8_to_wstring(spirv_opt_config_it->second));
    } else {
      arguments.emplace_back(L"-enable-16bit-types");
      arguments.emplace_back((optimization_level == 0u) ? L"-O0" : ((optimization_level == 1u) ? L"-O1" : ((optimization_level == 2u) ? L"-O2" : L"-O3")));
    }
    if constexpr (kEnableShaderDebugInfo) {
      arguments.emplace_back(L"-Zi");
      arguments.emplace_back(L"-Qembed_debug");
    }
  } else {
    arguments.emplace_back(L"-P");
  }

  for (const auto& [key, value] : defines) {
    if ((key == "ETX_DXC_OPT_LEVEL") || (key == "ETX_DXC_SPIRV_OPT_CONFIG")) {
      continue;
    }
    std::string define_str = key;
    if (value.empty() == false) {
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

  record_dependency(full_path);

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

void CustomIncludeHandler::record_dependency(const std::string& full_path) {
  std::lock_guard<std::mutex> lock(_dependency_mutex);
  for (const auto& dependency : _dependencies) {
    if (dependency == full_path) {
      return;
    }
  }
  _dependencies.push_back(full_path);
}

std::vector<std::string> CustomIncludeHandler::dependencies() const {
  std::lock_guard<std::mutex> lock(_dependency_mutex);
  return _dependencies;
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
