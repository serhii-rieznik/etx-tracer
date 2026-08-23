#include <etx/rhi/shader/shader_package.hxx>

#include <etx/core/core.hxx>
#include <etx/core/platform.hxx>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <mutex>
#include <system_error>
#include <type_traits>

#if ETX_PLATFORM_WINDOWS
# if !defined(WIN32_LEAN_AND_MEAN)
#  define WIN32_LEAN_AND_MEAN
# endif
# include <Windows.h>
#else
# include <unistd.h>
# if ETX_PLATFORM_APPLE
#  include <sys/wait.h>
# endif
#endif

namespace etx {
namespace {

constexpr std::array<char, 8> kShaderPackageMagic = {'E', 'T', 'X', 'S', 'H', 'P', 'K', '1'};
constexpr uint32_t kShaderPackageVersion = 1u;
constexpr uint64_t kMaximumPackagedShaderSize = 64ull * 1024ull * 1024ull;

struct ShaderPackageHeader {
  char magic[8] = {};
  uint32_t version = kShaderPackageVersion;
  uint32_t backend = 0u;
  uint32_t entry_count = 0u;
  uint32_t reserved = 0u;
  uint64_t records_offset = 0u;
  uint64_t data_offset = 0u;
  uint64_t index_hash = 0u;
};

struct ShaderPackageRecord {
  uint64_t request_hash = 0u;
  uint64_t content_hash = 0u;
  uint64_t data_offset = 0u;
  uint64_t data_size = 0u;
  uint32_t stage = 0u;
  uint32_t format = 0u;
  uint32_t local_size_x = 1u;
  uint32_t local_size_y = 1u;
  uint32_t local_size_z = 1u;
  uint32_t metal_metadata_valid = 0u;
  uint32_t metal_bindless_buffer_indices[kRHIMetalBindlessBindingCount] = {};
  uint32_t metal_bindless_binding_access[kRHIMetalBindlessBindingCount] = {};
  uint32_t metal_push_constants_buffer_index = kRHIInvalidMetalBufferIndex;
  uint32_t reserved = 0u;
};

static_assert(std::is_trivially_copyable_v<ShaderPackageHeader>);
static_assert(std::is_trivially_copyable_v<ShaderPackageRecord>);

std::string normalize_shader_source_name(const std::string& source_name) {
  std::string result = source_name;
  std::replace(result.begin(), result.end(), '\\', '/');
  while (result.rfind("./", 0u) == 0u) {
    result.erase(0u, 2u);
  }
  return result;
}

uint64_t append_hash_string(uint64_t hash, const std::string& value) {
  const uint64_t size = static_cast<uint64_t>(value.size());
  hash = etx_hash64_continue(&size, sizeof(size), hash);
  return etx_hash64_continue(value.data(), value.size(), hash);
}

bool shader_package_requests_equal(const ShaderPackageRequest& lhs, const ShaderPackageRequest& rhs) {
  return (normalize_shader_source_name(lhs.source_name) == normalize_shader_source_name(rhs.source_name)) && (lhs.entry_point == rhs.entry_point) && (lhs.stage == rhs.stage) &&
         (lhs.backend == rhs.backend) && (lhs.defines == rhs.defines);
}

uint64_t align_up(uint64_t value, uint64_t alignment) {
  return (value + alignment - 1u) & ~(alignment - 1u);
}

bool read_file_range(const std::filesystem::path& path, uint64_t offset, void* data, uint64_t size) {
  std::ifstream stream(path, std::ios::binary);
  if (stream.is_open() == false) {
    return false;
  }
  stream.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
  if (stream.good() == false) {
    return false;
  }
  stream.read(static_cast<char*>(data), static_cast<std::streamsize>(size));
  return stream.gcount() == static_cast<std::streamsize>(size);
}

bool write_padding(std::ofstream& stream, uint64_t size) {
  static constexpr std::array<char, 16> zeroes = {};
  uint64_t remaining = size;
  while (remaining > 0u) {
    const uint64_t block_size = std::min<uint64_t>(remaining, zeroes.size());
    stream.write(zeroes.data(), static_cast<std::streamsize>(block_size));
    remaining -= block_size;
  }
  return stream.good();
}

#if ETX_PLATFORM_APPLE
std::string shell_quote_shader_package(const std::string& value) {
  std::string result = "'";
  for (const char character : value) {
    if (character == '\'') {
      result += "'\\''";
    } else {
      result.push_back(character);
    }
  }
  result.push_back('\'');
  return result;
}

bool run_shader_package_command(const std::string& command, std::string& output, int& exit_code) {
  FILE* pipe = popen(command.c_str(), "r");
  if (pipe == nullptr) {
    output = "Failed to launch compiler command.";
    exit_code = -1;
    return false;
  }

  output.clear();
  std::array<char, 4096> buffer = {};
  while (true) {
    const size_t bytes_read = fread(buffer.data(), 1u, buffer.size(), pipe);
    if (bytes_read > 0u) {
      output.append(buffer.data(), bytes_read);
    }
    if (bytes_read < buffer.size()) {
      break;
    }
  }

  const int status = pclose(pipe);
# if defined(WIFEXITED) && defined(WEXITSTATUS)
  exit_code = WIFEXITED(status) ? WEXITSTATUS(status) : status;
# else
  exit_code = status;
# endif
  return exit_code == 0;
}
#endif

uint64_t shader_package_process_id() {
#if ETX_PLATFORM_WINDOWS
  return static_cast<uint64_t>(GetCurrentProcessId());
#else
  return static_cast<uint64_t>(getpid());
#endif
}

std::filesystem::path make_temporary_path(const std::filesystem::path& path) {
  static std::atomic<uint64_t> next_file_id = 0u;
  std::filesystem::path temporary_path = path;
  temporary_path += ".tmp-" + std::to_string(shader_package_process_id()) + "-" + std::to_string(next_file_id.fetch_add(1u));
  return temporary_path;
}

bool publish_file_atomically(const std::filesystem::path& temporary_path, const std::filesystem::path& destination_path, std::string& error_message) {
#if ETX_PLATFORM_WINDOWS
  if (MoveFileExW(temporary_path.c_str(), destination_path.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) == 0) {
    error_message = std::system_category().message(static_cast<int>(GetLastError()));
    return false;
  }
  return true;
#else
  std::error_code ec = {};
  std::filesystem::rename(temporary_path, destination_path, ec);
  if (ec.value() != 0) {
    error_message = ec.message();
    return false;
  }
  return true;
#endif
}

bool valid_spirv_binary(const std::vector<uint8_t>& binary) {
  static constexpr std::array<uint8_t, 4> magic = {0x03u, 0x02u, 0x23u, 0x07u};
  return (binary.size() >= 20u) && ((binary.size() % sizeof(uint32_t)) == 0u) && (std::memcmp(binary.data(), magic.data(), magic.size()) == 0);
}

bool valid_metal_library(const std::vector<uint8_t>& library) {
  static constexpr std::array<uint8_t, 4> magic = {'M', 'T', 'L', 'B'};
  return (library.size() >= magic.size()) && (std::memcmp(library.data(), magic.data(), magic.size()) == 0);
}

#if ETX_PLATFORM_APPLE
uint64_t metal_compiler_fingerprint() {
  static std::once_flag fingerprint_once = {};
  static uint64_t fingerprint = 0u;
  std::call_once(fingerprint_once, []() {
    std::string output = {};
    int exit_code = 0;
    run_shader_package_command("xcrun metal --version 2>&1", output, exit_code);
    fingerprint = append_hash_string(fingerprint, output);
    run_shader_package_command("xcrun metallib --version 2>&1", output, exit_code);
    fingerprint = append_hash_string(fingerprint, output);
  });
  return fingerprint;
}
#endif

bool read_complete_file(const std::filesystem::path& path, std::vector<uint8_t>& data) {
  std::error_code ec = {};
  const uintmax_t file_size = std::filesystem::file_size(path, ec);
  if ((ec.value() != 0) || (file_size > kMaximumPackagedShaderSize)) {
    return false;
  }
  data.resize(static_cast<size_t>(file_size));
  return read_file_range(path, 0u, data.data(), data.size());
}

}  // namespace

struct ShaderPackage::Impl {
  std::filesystem::path path = {};
  RHIBackend backend = RHIBackend::Vulkan;
  std::vector<ShaderPackageRecord> records = {};
};

ShaderPackage::ShaderPackage()
  : _impl(std::make_unique<Impl>()) {
}

ShaderPackage::~ShaderPackage() = default;
ShaderPackage::ShaderPackage(ShaderPackage&&) noexcept = default;
ShaderPackage& ShaderPackage::operator=(ShaderPackage&&) noexcept = default;

uint64_t shader_package_request_hash(const ShaderPackageRequest& request) {
  uint64_t hash = 0x4554585348504b31ull;
  hash = append_hash_string(hash, normalize_shader_source_name(request.source_name));
  hash = append_hash_string(hash, request.entry_point);
  const uint32_t stage = static_cast<uint32_t>(request.stage);
  const uint32_t backend = static_cast<uint32_t>(request.backend);
  hash = etx_hash64_continue(&stage, sizeof(stage), hash);
  hash = etx_hash64_continue(&backend, sizeof(backend), hash);
  const uint64_t define_count = static_cast<uint64_t>(request.defines.size());
  hash = etx_hash64_continue(&define_count, sizeof(define_count), hash);
  for (const auto& [name, value] : request.defines) {
    hash = append_hash_string(hash, name);
    hash = append_hash_string(hash, value);
  }
  return hash;
}

bool ShaderPackage::load(const std::filesystem::path& path, RHIBackend backend, std::string& error_message) {
  _impl->path.clear();
  _impl->records.clear();

  std::error_code ec = {};
  const uintmax_t package_size = std::filesystem::file_size(path, ec);
  if (ec.value() != 0) {
    error_message = "Failed to query shader package '" + path.string() + "': " + ec.message();
    return false;
  }

  ShaderPackageHeader header = {};
  if (read_file_range(path, 0u, &header, sizeof(header)) == false) {
    error_message = "Failed to read shader package header.";
    return false;
  }
  if ((std::memcmp(header.magic, kShaderPackageMagic.data(), kShaderPackageMagic.size()) != 0) || (header.version != kShaderPackageVersion)) {
    error_message = "Shader package format is unsupported.";
    return false;
  }
  if (header.backend != static_cast<uint32_t>(backend)) {
    error_message = "Shader package backend does not match the active renderer backend.";
    return false;
  }
  if (header.entry_count == 0u) {
    error_message = "Shader package is empty.";
    return false;
  }

  const uint64_t records_size = static_cast<uint64_t>(header.entry_count) * sizeof(ShaderPackageRecord);
  if ((header.records_offset < sizeof(ShaderPackageHeader)) || (records_size > package_size) || (header.records_offset > (package_size - records_size)) ||
      (header.data_offset < (header.records_offset + records_size)) || (header.data_offset > package_size)) {
    error_message = "Shader package index range is invalid.";
    return false;
  }

  _impl->records.resize(header.entry_count);
  if (read_file_range(path, header.records_offset, _impl->records.data(), records_size) == false) {
    error_message = "Failed to read shader package index.";
    _impl->records.clear();
    return false;
  }
  if (etx_hash64(_impl->records.data(), static_cast<size_t>(records_size)) != header.index_hash) {
    error_message = "Shader package index integrity check failed.";
    _impl->records.clear();
    return false;
  }

  uint64_t previous_hash = 0u;
  for (size_t record_index = 0u; record_index < _impl->records.size(); ++record_index) {
    const ShaderPackageRecord& record = _impl->records[record_index];
    if (((record_index > 0u) && (record.request_hash <= previous_hash)) || (record.data_size == 0u) || (record.data_size > kMaximumPackagedShaderSize) ||
        (record.data_offset < header.data_offset) || (record.data_offset > package_size) || (record.data_size > (package_size - record.data_offset)) ||
        (record.stage > static_cast<uint32_t>(RHIShaderStage::Compute)) || (record.format > static_cast<uint32_t>(RHIShaderBinaryFormat::MetalLibrary)) ||
        (record.local_size_x == 0u) || (record.local_size_y == 0u) || (record.local_size_z == 0u)) {
      error_message = "Shader package contains an invalid record.";
      _impl->records.clear();
      return false;
    }
    if ((backend == RHIBackend::Vulkan) && (record.format != static_cast<uint32_t>(RHIShaderBinaryFormat::SpirV))) {
      error_message = "Vulkan shader package contains a non-SPIR-V record.";
      _impl->records.clear();
      return false;
    }
    if ((backend == RHIBackend::Metal) && (record.format != static_cast<uint32_t>(RHIShaderBinaryFormat::MetalLibrary))) {
      error_message = "Metal shader package contains a non-metallib record.";
      _impl->records.clear();
      return false;
    }
    if (((backend == RHIBackend::Vulkan) && (record.metal_metadata_valid != 0u)) || ((backend == RHIBackend::Metal) && (record.metal_metadata_valid != 1u))) {
      error_message = "Shader package contains invalid backend metadata.";
      _impl->records.clear();
      return false;
    }
    if (record.metal_metadata_valid != 0u) {
      for (const uint32_t binding_access : record.metal_bindless_binding_access) {
        if (binding_access > static_cast<uint32_t>(RHIMetalBindingAccess::ReadWrite)) {
          error_message = "Shader package contains invalid Metal binding metadata.";
          _impl->records.clear();
          return false;
        }
      }
    }
    previous_hash = record.request_hash;
  }

  _impl->path = path;
  _impl->backend = backend;
  error_message.clear();
  return true;
}

bool ShaderPackage::loaded() const {
  return (_impl != nullptr) && (_impl->path.empty() == false) && (_impl->records.empty() == false);
}

bool ShaderPackage::contains(const ShaderPackageRequest& request) const {
  if (loaded() == false) {
    return false;
  }
  const uint64_t request_hash = shader_package_request_hash(request);
  const auto iterator = std::lower_bound(_impl->records.begin(), _impl->records.end(), request_hash, [](const ShaderPackageRecord& record, uint64_t hash) {
    return record.request_hash < hash;
  });
  return (iterator != _impl->records.end()) && (iterator->request_hash == request_hash);
}

bool ShaderPackage::read(const ShaderPackageRequest& request, ShaderPackageBinary& binary, std::string& error_message) const {
  if (loaded() == false) {
    error_message = "Shader package is not loaded.";
    return false;
  }
  const uint64_t request_hash = shader_package_request_hash(request);
  const auto iterator = std::lower_bound(_impl->records.begin(), _impl->records.end(), request_hash, [](const ShaderPackageRecord& record, uint64_t hash) {
    return record.request_hash < hash;
  });
  if ((iterator == _impl->records.end()) || (iterator->request_hash != request_hash)) {
    error_message = "Shader variant is absent from the package.";
    return false;
  }

  binary = {};
  binary.data.resize(static_cast<size_t>(iterator->data_size));
  if (read_file_range(_impl->path, iterator->data_offset, binary.data.data(), iterator->data_size) == false) {
    error_message = "Failed to read packaged shader data.";
    binary = {};
    return false;
  }
  if (etx_hash64(binary.data.data(), binary.data.size()) != iterator->content_hash) {
    error_message = "Packaged shader integrity check failed.";
    binary = {};
    return false;
  }

  const RHIShaderBinaryFormat format = static_cast<RHIShaderBinaryFormat>(iterator->format);
  if (((format == RHIShaderBinaryFormat::SpirV) && (valid_spirv_binary(binary.data) == false)) ||
      ((format == RHIShaderBinaryFormat::MetalLibrary) && (valid_metal_library(binary.data) == false))) {
    error_message = "Packaged shader binary is invalid.";
    binary = {};
    return false;
  }

  binary.format = format;
  binary.local_size_x = iterator->local_size_x;
  binary.local_size_y = iterator->local_size_y;
  binary.local_size_z = iterator->local_size_z;
  if (iterator->metal_metadata_valid != 0u) {
    for (uint32_t binding_index = 0u; binding_index < kRHIMetalBindlessBindingCount; ++binding_index) {
      if (iterator->metal_bindless_binding_access[binding_index] > static_cast<uint32_t>(RHIMetalBindingAccess::ReadWrite)) {
        error_message = "Packaged Metal shader metadata is invalid.";
        binary = {};
        return false;
      }
      binary.metal_metadata.bindless_buffer_indices[binding_index] = iterator->metal_bindless_buffer_indices[binding_index];
      binary.metal_metadata.bindless_binding_access[binding_index] = static_cast<RHIMetalBindingAccess>(iterator->metal_bindless_binding_access[binding_index]);
    }
    binary.metal_metadata.push_constants_buffer_index = iterator->metal_push_constants_buffer_index;
    binary.metal_metadata.valid = true;
  }
  error_message.clear();
  return true;
}

uint32_t ShaderPackage::entry_count() const {
  return (_impl != nullptr) ? static_cast<uint32_t>(_impl->records.size()) : 0u;
}

bool write_shader_package(const std::filesystem::path& path, std::vector<ShaderPackageBuildEntry> entries, ShaderPackageBuildStatistics& statistics, std::string& error_message) {
  statistics = {};
  if (entries.empty()) {
    error_message = "Cannot write an empty shader package.";
    return false;
  }
  if (entries.size() > std::numeric_limits<uint32_t>::max()) {
    error_message = "Shader package contains too many entries.";
    return false;
  }

  const RHIBackend backend = entries.front().request.backend;
  if ((backend != RHIBackend::Vulkan) && (backend != RHIBackend::Metal)) {
    error_message = "Shader package backend is unsupported.";
    return false;
  }
  std::sort(entries.begin(), entries.end(), [](const ShaderPackageBuildEntry& lhs, const ShaderPackageBuildEntry& rhs) {
    return shader_package_request_hash(lhs.request) < shader_package_request_hash(rhs.request);
  });

  std::vector<ShaderPackageRecord> records;
  records.resize(entries.size());
  const uint64_t records_offset = sizeof(ShaderPackageHeader);
  const uint64_t records_size = static_cast<uint64_t>(records.size()) * sizeof(ShaderPackageRecord);
  uint64_t data_offset = align_up(records_offset + records_size, 16u);
  uint64_t previous_request_hash = 0u;

  for (size_t entry_index = 0u; entry_index < entries.size(); ++entry_index) {
    const ShaderPackageBuildEntry& entry = entries[entry_index];
    const uint64_t request_hash = shader_package_request_hash(entry.request);
    if (entry.request.backend != backend) {
      error_message = "Shader package entries use mixed backends.";
      return false;
    }
    if ((entry_index > 0u) && (request_hash == previous_request_hash)) {
      if (shader_package_requests_equal(entry.request, entries[entry_index - 1u].request)) {
        error_message = "Shader package contains a duplicate request.";
      } else {
        error_message = "Shader package request hash collision detected.";
      }
      return false;
    }
    if (entry.binary.data.empty() || (entry.binary.data.size() > kMaximumPackagedShaderSize)) {
      error_message = "Shader package entry has an invalid binary size.";
      return false;
    }
    if ((entry.request.stage > RHIShaderStage::Compute) || (entry.binary.local_size_x == 0u) || (entry.binary.local_size_y == 0u) || (entry.binary.local_size_z == 0u)) {
      error_message = "Shader package entry metadata is invalid.";
      return false;
    }
    if (((backend == RHIBackend::Vulkan) && (entry.binary.format != RHIShaderBinaryFormat::SpirV)) ||
        ((backend == RHIBackend::Metal) && ((entry.binary.format != RHIShaderBinaryFormat::MetalLibrary) || (entry.binary.metal_metadata.valid == false)))) {
      error_message = "Shader package entry format does not match its backend.";
      return false;
    }
    if (((backend == RHIBackend::Vulkan) && ((entry.binary.metal_metadata.valid) || (valid_spirv_binary(entry.binary.data) == false))) ||
        ((backend == RHIBackend::Metal) && (valid_metal_library(entry.binary.data) == false))) {
      error_message = "Shader package entry binary is invalid.";
      return false;
    }
    if (entry.binary.metal_metadata.valid) {
      for (const RHIMetalBindingAccess binding_access : entry.binary.metal_metadata.bindless_binding_access) {
        if (binding_access > RHIMetalBindingAccess::ReadWrite) {
          error_message = "Shader package entry contains invalid Metal binding metadata.";
          return false;
        }
      }
    }

    ShaderPackageRecord& record = records[entry_index];
    record.request_hash = request_hash;
    record.content_hash = etx_hash64(entry.binary.data.data(), entry.binary.data.size());
    record.data_offset = data_offset;
    record.data_size = static_cast<uint64_t>(entry.binary.data.size());
    record.stage = static_cast<uint32_t>(entry.request.stage);
    record.format = static_cast<uint32_t>(entry.binary.format);
    record.local_size_x = entry.binary.local_size_x;
    record.local_size_y = entry.binary.local_size_y;
    record.local_size_z = entry.binary.local_size_z;
    record.metal_metadata_valid = entry.binary.metal_metadata.valid ? 1u : 0u;
    for (uint32_t binding_index = 0u; binding_index < kRHIMetalBindlessBindingCount; ++binding_index) {
      record.metal_bindless_buffer_indices[binding_index] = entry.binary.metal_metadata.bindless_buffer_indices[binding_index];
      record.metal_bindless_binding_access[binding_index] = static_cast<uint32_t>(entry.binary.metal_metadata.bindless_binding_access[binding_index]);
    }
    record.metal_push_constants_buffer_index = entry.binary.metal_metadata.push_constants_buffer_index;
    data_offset = align_up(data_offset + record.data_size, 16u);
    previous_request_hash = request_hash;
    statistics.binary_size_bytes += record.data_size;
  }

  std::error_code ec = {};
  if (path.parent_path().empty() == false) {
    std::filesystem::create_directories(path.parent_path(), ec);
    if (ec.value() != 0) {
      error_message = "Failed to create shader package directory: " + ec.message();
      return false;
    }
  }
  const std::filesystem::path temporary_path = make_temporary_path(path);
  std::ofstream stream(temporary_path, std::ios::binary | std::ios::trunc);
  if (stream.is_open() == false) {
    error_message = "Failed to create shader package file.";
    return false;
  }

  ShaderPackageHeader header = {};
  std::memcpy(header.magic, kShaderPackageMagic.data(), kShaderPackageMagic.size());
  header.backend = static_cast<uint32_t>(backend);
  header.entry_count = static_cast<uint32_t>(records.size());
  header.records_offset = records_offset;
  header.data_offset = align_up(records_offset + records_size, 16u);
  header.index_hash = etx_hash64(records.data(), static_cast<size_t>(records_size));
  stream.write(reinterpret_cast<const char*>(&header), sizeof(header));
  stream.write(reinterpret_cast<const char*>(records.data()), static_cast<std::streamsize>(records_size));
  if (write_padding(stream, header.data_offset - (records_offset + records_size)) == false) {
    error_message = "Failed to write shader package index.";
    stream.close();
    std::filesystem::remove(temporary_path, ec);
    return false;
  }

  uint64_t current_offset = header.data_offset;
  for (size_t entry_index = 0u; entry_index < entries.size(); ++entry_index) {
    const auto& data = entries[entry_index].binary.data;
    stream.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
    current_offset += data.size();
    const uint64_t next_offset = align_up(current_offset, 16u);
    if (write_padding(stream, next_offset - current_offset) == false) {
      break;
    }
    current_offset = next_offset;
  }
  stream.close();
  const bool write_succeeded = stream.good();
  if (write_succeeded == false) {
    error_message = "Failed to write shader package data.";
    std::filesystem::remove(temporary_path, ec);
    return false;
  }

  std::string publish_error = {};
  if (publish_file_atomically(temporary_path, path, publish_error) == false) {
    error_message = "Failed to publish shader package: " + publish_error;
    std::filesystem::remove(temporary_path, ec);
    return false;
  }

  statistics.entry_count = static_cast<uint32_t>(entries.size());
  statistics.package_size_bytes = current_offset;
  error_message.clear();
  return true;
}

bool compile_metal_shader_library(const void* source_data, size_t source_size, const std::filesystem::path& cache_directory, const std::string& minimum_macos_version,
  std::vector<uint8_t>& library, std::string& error_message) {
#if ETX_PLATFORM_APPLE
  library.clear();
  error_message.clear();
  if ((source_data == nullptr) || (source_size == 0u)) {
    error_message = "Metal shader source is empty.";
    return false;
  }

  const uint64_t compiler_fingerprint = metal_compiler_fingerprint();
  uint64_t source_hash = etx_hash64_continue(&compiler_fingerprint, sizeof(compiler_fingerprint), etx_hash64(source_data, source_size));
  source_hash = append_hash_string(source_hash, minimum_macos_version);
  std::array<char, 17> hash_text = {};
  snprintf(hash_text.data(), hash_text.size(), "%016llx", static_cast<unsigned long long>(source_hash));
  std::error_code ec = {};
  std::filesystem::create_directories(cache_directory, ec);
  if (ec.value() != 0) {
    error_message = "Failed to create the Metal library build cache: " + ec.message();
    return false;
  }
  const std::filesystem::path cached_library_path = cache_directory / (std::string(hash_text.data()) + ".metallib");
  if (std::filesystem::exists(cached_library_path, ec) && (ec.value() == 0) && read_complete_file(cached_library_path, library) && valid_metal_library(library)) {
    return true;
  }
  library.clear();

  const std::filesystem::path temporary_root = std::filesystem::temp_directory_path(ec) / "etx-shader-packager";
  std::filesystem::create_directories(temporary_root, ec);
  if (ec.value() != 0) {
    error_message = "Failed to create the Metal compiler temporary directory: " + ec.message();
    return false;
  }

  static std::atomic<uint64_t> next_file_id = 0u;
  const std::string file_id = std::to_string(shader_package_process_id()) + "-" + std::to_string(next_file_id.fetch_add(1u));
  const std::filesystem::path source_path = temporary_root / (file_id + ".metal");
  const std::filesystem::path air_path = temporary_root / (file_id + ".air");
  const std::filesystem::path library_path = temporary_root / (file_id + ".metallib");

  {
    std::ofstream source_stream(source_path, std::ios::binary | std::ios::trunc);
    if (source_stream.is_open() == false) {
      error_message = "Failed to create temporary Metal source file.";
      return false;
    }
    source_stream.write(static_cast<const char*>(source_data), static_cast<std::streamsize>(source_size));
    if (source_stream.good() == false) {
      error_message = "Failed to write temporary Metal source file.";
      source_stream.close();
      std::filesystem::remove(source_path, ec);
      return false;
    }
  }

  int exit_code = 0;
  std::string command_output = {};
  std::string metal_command = "xcrun metal -c";
  if (minimum_macos_version.empty() == false) {
    metal_command += " -mmacosx-version-min=" + shell_quote_shader_package(minimum_macos_version);
  }
  metal_command += " " + shell_quote_shader_package(source_path.string()) + " -o " + shell_quote_shader_package(air_path.string()) + " 2>&1";
  if (run_shader_package_command(metal_command, command_output, exit_code) == false) {
    error_message = "Metal compiler failed (" + std::to_string(exit_code) + "): " + command_output;
  } else {
    const std::string metallib_command = "xcrun metallib " + shell_quote_shader_package(air_path.string()) + " -o " + shell_quote_shader_package(library_path.string()) + " 2>&1";
    if (run_shader_package_command(metallib_command, command_output, exit_code) == false) {
      error_message = "Metal library linker failed (" + std::to_string(exit_code) + "): " + command_output;
    } else if ((read_complete_file(library_path, library) == false) || (valid_metal_library(library) == false)) {
      error_message = "Metal compiler produced an invalid library.";
    } else {
      const std::filesystem::path temporary_cache_path = make_temporary_path(cached_library_path);
      std::filesystem::copy_file(library_path, temporary_cache_path, std::filesystem::copy_options::none, ec);
      std::string publish_error = {};
      if ((ec.value() != 0) || (publish_file_atomically(temporary_cache_path, cached_library_path, publish_error) == false)) {
        error_message = "Failed to update the Metal library build cache: " + ((ec.value() != 0) ? ec.message() : publish_error);
        std::filesystem::remove(temporary_cache_path, ec);
        library.clear();
      }
    }
  }

  std::filesystem::remove(source_path, ec);
  std::filesystem::remove(air_path, ec);
  std::filesystem::remove(library_path, ec);
  return error_message.empty() && (library.empty() == false);
#else
  static_cast<void>(source_data);
  static_cast<void>(source_size);
  static_cast<void>(cache_directory);
  static_cast<void>(minimum_macos_version);
  library.clear();
  error_message = "Metal libraries can only be built on macOS.";
  return false;
#endif
}

}  // namespace etx
