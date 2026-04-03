#include "gpu_renderer.hxx"
#include <interop/gpu_abi_constants.hxx>
#include <interop/gpu_rt_shared.hxx>
#include <interop/gpu_wavefront_abi.hxx>
#include <interop/material.hxx>
#include <interop/sampler_policy.hxx>
#include <etx/core/profiler.hxx>
#include <etx/rhi/rhi.hxx>
#include <etx/rhi/shader/shader_compiler.hxx>
#include <etx/render/host/gpu_asset_descriptor.hxx>
#include <etx/render/host/emitter_packing.hxx>
#include <etx/render/host/scene_representation.hxx>
#include <etx/render/shared/density_grid.hxx>
#include <bluenoise.hxx>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>
#include "gpu_renderer_abi_static_asserts.hxx"

namespace etx {
namespace {
constexpr uint32_t kInvalidDescriptorIndex = ~0u;

constexpr uint32_t kBlueNoiseTileSize = kSamplerBlueNoiseTileSize;
constexpr uint32_t kBlueNoiseSampleCount = kSamplerBlueNoiseSampleCount;
constexpr uint32_t kBlueNoiseDimensionCount = kSamplerBlueNoiseDimensionCount;
constexpr uint32_t kWavefrontRollingHistoryBounces = 2u;

enum class GPUPathMode : uint32_t {
  PathTracing = 0u,
  LightTracing = 1u,
  BDPTFast = 2u,
};

GPUPathMode gpu_path_mode_from_scene_strategies(const SceneData& scene_data) {
  const uint32_t strategy_flags = scene_data.options.strategy_flags;
  const bool direct_hit_enabled = (strategy_flags & Scene::Strategy::DirectHit) != 0u;
  const bool connect_to_light_enabled = (strategy_flags & Scene::Strategy::ConnectToLight) != 0u;
  const bool connect_to_camera_enabled = (strategy_flags & Scene::Strategy::ConnectToCamera) != 0u;

  if ((connect_to_camera_enabled == false) && (direct_hit_enabled || connect_to_light_enabled)) {
    return GPUPathMode::PathTracing;
  }

  if ((connect_to_camera_enabled) && (direct_hit_enabled == false) && (connect_to_light_enabled == false)) {
    return GPUPathMode::LightTracing;
  }

  return GPUPathMode::BDPTFast;
}

GPUPathMode gpu_path_mode_from_scene(const SceneRepresentation& scene) {
  const auto& integrator_data = scene.integrator_data();

  if (integrator_data.selected == Integrator::Type::PathTracing) {
    return GPUPathMode::PathTracing;
  }

  if (integrator_data.selected == Integrator::Type::Bidirectional) {
    auto settings_it = integrator_data.settings.find(Integrator::Type::Bidirectional);
    uint32_t bidirectional_mode = 2u;
    if (settings_it != integrator_data.settings.end()) {
      bidirectional_mode = settings_it->second.get_integral("bdpt-mode", bidirectional_mode);
    }

    switch (bidirectional_mode) {
      case 0u:
        return GPUPathMode::PathTracing;
      case 1u:
        return GPUPathMode::LightTracing;
      case 2u:
        return GPUPathMode::BDPTFast;
      default:
        return GPUPathMode::BDPTFast;
    }
  }

  return gpu_path_mode_from_scene_strategies(scene.data());
}

const char* gpu_path_mode_to_string(GPUPathMode path_mode) {
  switch (path_mode) {
    case GPUPathMode::PathTracing:
      return "PathTracing";
    case GPUPathMode::LightTracing:
      return "LightTracing";
    case GPUPathMode::BDPTFast:
      return "BDPTFast";
    default:
      return "Unknown";
  }
}

constexpr uint32_t material_compile_bit(uint32_t material_class) {
  return (material_class < 32u) ? (1u << material_class) : 0u;
}

bool material_compile_mask_has(uint32_t mask, uint32_t material_class) {
  const uint32_t bit = material_compile_bit(material_class);
  return (bit != 0u) && ((mask & bit) != 0u);
}

uint32_t build_material_compile_mask(const SceneData& scene_data) {
  uint32_t result = 0u;
  for (const auto& material : scene_data.materials) {
    result |= material_compile_bit(material.cls);
  }
  return result;
}

uint32_t normalize_blue_noise_target_samples(uint32_t value) {
  uint32_t result = value;
  if (result == 0u) {
    result = 1u;
  }
  if (result > kBlueNoiseSampleCount) {
    result = kBlueNoiseSampleCount;
  }

  result -= 1u;
  result |= (result >> 1u);
  result |= (result >> 2u);
  result |= (result >> 4u);
  result |= (result >> 8u);
  result |= (result >> 16u);
  result += 1u;
  return result;
}

uint32_t blue_noise_table_index(uint32_t x, uint32_t y, uint32_t sample_index, uint32_t dimension) {
  return (((sample_index * kBlueNoiseDimensionCount) + dimension) * kBlueNoiseTileSize + y) * kBlueNoiseTileSize + x;
}

bool compile_timed_compute_stage(RHIDevice& device, ShaderCompiler& compiler, const char* source_file, const char* entry_point, const char* optimization_level,
  const char* bsdf_kind, RHIPipeline& pipeline) {
  pipeline = {};

  std::unordered_map<std::string, std::string> defines = {};
  if (optimization_level != nullptr) {
    defines["ETX_DXC_OPT_LEVEL"] = optimization_level;
  }
  defines["ETX_STAGE_ENTRY"] = entry_point;
  if (bsdf_kind != nullptr) {
    defines["ETX_BSDF_KIND"] = bsdf_kind;
  }

  log::info("Compiling %s...", entry_point);
  const auto compile_begin = std::chrono::steady_clock::now();
  ShaderCompiler::MultiShaderCompilationResult result = {};
  {
    ETX_PROFILER_NAMED_SCOPE("gpu_rt_compile_compute_shader");
    result = compiler.compile(source_file, {{entry_point, RHIShaderStage::Compute}}, defines, device.backend());
  }
  const auto compile_end = std::chrono::steady_clock::now();
  const auto compile_ms = std::chrono::duration_cast<std::chrono::milliseconds>(compile_end - compile_begin).count();

  if ((result.result != RHIResult::Success) || (result.binaries.size() != 1u)) {
    log::error("Failed to compile GPU RT shader stage '%s' after %llims: %s", entry_point, static_cast<long long>(compile_ms), result.error_message.c_str());
    return false;
  }

  log::info("Compiled %s - %llims", entry_point, static_cast<long long>(compile_ms));

  RHIComputePipelineDesc desc = device.make_compute_pipeline_desc(result.binaries[0]);
  auto pipeline_result = device.create_compute_pipeline(desc);
  if ((pipeline_result.result != RHIResult::Success) || (pipeline_result.handle.valid() == false)) {
    log::error("Failed to create GPU RT wavefront pipeline '%s' (%u)", entry_point, static_cast<uint32_t>(pipeline_result.result));
    return false;
  }

  pipeline = pipeline_result.handle;
  return true;
}

bool build_blue_noise_table_data(uint32_t target_samples, std::vector<float>& table_data) {
  target_samples = normalize_blue_noise_target_samples(target_samples);

  const uint64_t table_size_u64 = static_cast<uint64_t>(kBlueNoiseTileSize) * static_cast<uint64_t>(kBlueNoiseTileSize) * static_cast<uint64_t>(kBlueNoiseSampleCount) *
                                  static_cast<uint64_t>(kBlueNoiseDimensionCount);
  if (table_size_u64 > std::numeric_limits<size_t>::max()) {
    return false;
  }

  const size_t table_size = static_cast<size_t>(table_size_u64);
  table_data.assign(table_size, 0.0f);

  for (uint32_t sample_index = 0u; sample_index < kBlueNoiseSampleCount; ++sample_index) {
    for (uint32_t y = 0u; y < kBlueNoiseTileSize; ++y) {
      for (uint32_t x = 0u; x < kBlueNoiseTileSize; ++x) {
        const ::BNSampler sampler = ::BNSampler(x, y, target_samples, sample_index);
        for (uint32_t dimension = 0u; dimension < kBlueNoiseDimensionCount; ++dimension) {
          const float sample_value = sampler.get(dimension);
          const uint32_t index = blue_noise_table_index(x, y, sample_index, dimension);
          table_data[index] = sample_value;
        }
      }
    }
  }

  return true;
}

GPUScene make_invalid_gpu_scene() {
  return {
    .vertex_positions = kInvalidDescriptorIndex,
    .vertex_normals = kInvalidDescriptorIndex,
    .vertex_tangents = kInvalidDescriptorIndex,
    .vertex_bitangents = kInvalidDescriptorIndex,
    .vertex_texcoords = kInvalidDescriptorIndex,
    .triangles = kInvalidDescriptorIndex,
    .meshes = kInvalidDescriptorIndex,
    .emitter_profiles = kInvalidDescriptorIndex,
    .emitter_instances = kInvalidDescriptorIndex,
    .scene_globals = kInvalidDescriptorIndex,
    .materials = kInvalidDescriptorIndex,
    .spectrums = kInvalidDescriptorIndex,
    .images = kInvalidDescriptorIndex,
    .mediums = kInvalidDescriptorIndex,
    .emitters_distribution = kInvalidDescriptorIndex,
    .scene_options = kInvalidDescriptorIndex,
  };
}

void destroy_linear_scene_buffer(RHIDevice& device, RHIBindlessHandle& buffer, uint64_t& buffer_size, uint32_t& descriptor_index) {
  ETX_PROFILER_NAMED_SCOPE("gpu_rt_destroy_linear_scene_buffer");
  device.destroy_buffer(buffer);
  buffer = {};
  buffer_size = 0;
  descriptor_index = kInvalidDescriptorIndex;
}

bool ensure_storage_buffer(RHIDevice& device, uint64_t required_size, RHIBufferUsage usage, RHIBindlessHandle& buffer, uint64_t& buffer_size, uint32_t& descriptor_index,
  const char* buffer_name) {
  if (required_size == 0u) {
    destroy_linear_scene_buffer(device, buffer, buffer_size, descriptor_index);
    return true;
  }

  if (buffer.valid() && (buffer_size == required_size)) {
    descriptor_index = get_bindless_descriptor_index(buffer);
    return true;
  }

  RHIBufferDesc desc = {};
  desc.size = required_size;
  desc.usage = usage;

  auto create_result = device.create_buffer(desc);
  if ((create_result.result != RHIResult::Success) || (create_result.handle.valid() == false)) {
    log::error("GPU RT: failed to create '%s' storage buffer (%u)", (buffer_name != nullptr) ? buffer_name : "unknown", static_cast<uint32_t>(create_result.result));
    return false;
  }

  std::vector<uint8_t> zero_data(static_cast<size_t>(required_size), uint8_t(0u));
  const RHIResult upload_result = device.update_buffer(create_result.handle, zero_data.data(), required_size);
  if (upload_result != RHIResult::Success) {
    log::error("GPU RT: failed to initialize '%s' storage buffer (%u)", (buffer_name != nullptr) ? buffer_name : "unknown", static_cast<uint32_t>(upload_result));
    device.destroy_buffer(create_result.handle);
    return false;
  }

  if (buffer.valid()) {
    const RHIResult destroy_result = device.destroy_buffer(buffer);
    if (destroy_result != RHIResult::Success) {
      log::warning("GPU RT: failed to destroy previous '%s' storage buffer (%u)", (buffer_name != nullptr) ? buffer_name : "unknown", static_cast<uint32_t>(destroy_result));
    }
  }

  buffer = create_result.handle;
  buffer_size = required_size;
  descriptor_index = get_bindless_descriptor_index(buffer);
  return true;
}

bool ensure_host_visible_buffer(RHIDevice& device, uint64_t required_size, RHIBufferUsage usage, RHIBindlessHandle& buffer, uint64_t& buffer_size, uint32_t& descriptor_index,
  const char* buffer_name) {
  if (required_size == 0u) {
    destroy_linear_scene_buffer(device, buffer, buffer_size, descriptor_index);
    return true;
  }

  if (buffer.valid() && (buffer_size == required_size)) {
    descriptor_index = get_bindless_descriptor_index(buffer);
    return true;
  }

  RHIBufferDesc desc = {};
  desc.size = required_size;
  desc.usage = usage;
  desc.host_visible = true;

  auto create_result = device.create_buffer(desc);
  if ((create_result.result != RHIResult::Success) || (create_result.handle.valid() == false)) {
    log::error("GPU RT: failed to create '%s' host-visible buffer (%u)", (buffer_name != nullptr) ? buffer_name : "unknown", static_cast<uint32_t>(create_result.result));
    return false;
  }

  if (buffer.valid()) {
    const RHIResult destroy_result = device.destroy_buffer(buffer);
    if (destroy_result != RHIResult::Success) {
      log::error("GPU RT: failed to destroy previous '%s' host-visible buffer (%u)", (buffer_name != nullptr) ? buffer_name : "unknown", static_cast<uint32_t>(destroy_result));
      device.destroy_buffer(create_result.handle);
      return false;
    }
  }

  buffer = create_result.handle;
  buffer_size = required_size;
  descriptor_index = get_bindless_descriptor_index(buffer);
  return true;
}

template <typename T>
bool upload_or_update_linear_scene_buffer(RHIDevice& device, const T* data, size_t count, RHIBufferUsage usage, RHIBindlessHandle& buffer, uint64_t& buffer_size,
  uint32_t& descriptor_index, const char* buffer_name) {
  ETX_PROFILER_NAMED_SCOPE("gpu_rt_upload_or_update_linear_scene_buffer");

  if ((data == nullptr) || (count == 0u)) {
    destroy_linear_scene_buffer(device, buffer, buffer_size, descriptor_index);
    return true;
  }

  const uint64_t required_size = static_cast<uint64_t>(count) * sizeof(T);
  if (buffer.valid() && (buffer_size == required_size)) {
    const RHIResult update_result = device.update_buffer(buffer, data, required_size);
    if (update_result != RHIResult::Success) {
      log::error("GPU RT: failed to update '%s' buffer (%u)", (buffer_name != nullptr) ? buffer_name : "unknown", static_cast<uint32_t>(update_result));
      return false;
    }
    descriptor_index = get_bindless_descriptor_index(buffer);
    return true;
  }

  RHIBufferDesc desc = {};
  desc.size = required_size;
  desc.usage = usage;

  auto result = device.create_buffer(desc);
  if ((result.result != RHIResult::Success) || (result.handle.valid() == false)) {
    log::error("GPU RT: failed to create '%s' buffer (%u)", (buffer_name != nullptr) ? buffer_name : "unknown", static_cast<uint32_t>(result.result));
    descriptor_index = buffer.valid() ? get_bindless_descriptor_index(buffer) : kInvalidDescriptorIndex;
    return false;
  }

  const RHIResult update_result = device.update_buffer(result.handle, data, required_size);
  if (update_result != RHIResult::Success) {
    log::error("GPU RT: failed to upload '%s' buffer (%u)", (buffer_name != nullptr) ? buffer_name : "unknown", static_cast<uint32_t>(update_result));
    const RHIResult destroy_result = device.destroy_buffer(result.handle);
    if (destroy_result != RHIResult::Success) {
      log::error("GPU RT: failed to cleanup temporary '%s' buffer (%u)", (buffer_name != nullptr) ? buffer_name : "unknown", static_cast<uint32_t>(destroy_result));
    }
    descriptor_index = buffer.valid() ? get_bindless_descriptor_index(buffer) : kInvalidDescriptorIndex;
    return false;
  }

  if (buffer.valid()) {
    const RHIResult destroy_result = device.destroy_buffer(buffer);
    if (destroy_result != RHIResult::Success) {
      log::error("GPU RT: failed to destroy previous '%s' buffer (%u)", (buffer_name != nullptr) ? buffer_name : "unknown", static_cast<uint32_t>(destroy_result));
      const RHIResult cleanup_result = device.destroy_buffer(result.handle);
      if (cleanup_result != RHIResult::Success) {
        log::warning("GPU RT: failed to cleanup replacement '%s' buffer (%u)", (buffer_name != nullptr) ? buffer_name : "unknown", static_cast<uint32_t>(cleanup_result));
      }
      descriptor_index = get_bindless_descriptor_index(buffer);
      return false;
    }
  }

  buffer = result.handle;
  buffer_size = required_size;
  descriptor_index = get_bindless_descriptor_index(buffer);
  return true;
}

uint64_t align_up_u64(uint64_t value, uint64_t alignment) {
  const uint64_t a = (alignment == 0u) ? 1u : alignment;
  return ((value + a - 1u) / a) * a;
}

uint32_t append_aligned_bytes(std::vector<uint8_t>& blob, const void* data, uint64_t byte_size, uint64_t alignment = 16u) {
  if ((data == nullptr) || (byte_size == 0u)) {
    return kInvalidIndex;
  }

  const uint64_t aligned_offset = align_up_u64(static_cast<uint64_t>(blob.size()), alignment);
  if (aligned_offset > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
    log::error("GPU RT: packed blob offset overflow (%llu)", aligned_offset);
    return kInvalidIndex;
  }
  if (aligned_offset > blob.size()) {
    blob.resize(static_cast<size_t>(aligned_offset), 0u);
  }

  if (aligned_offset > (std::numeric_limits<uint64_t>::max() - byte_size)) {
    log::error("GPU RT: packed blob size overflow (offset=%llu, size=%llu)", aligned_offset, byte_size);
    return kInvalidIndex;
  }

  const uint64_t end_offset = aligned_offset + byte_size;
  ETX_ASSERT(end_offset >= aligned_offset);
  if (end_offset > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
    log::error("GPU RT: packed blob exceeds 32-bit offset range (%llu)", end_offset);
    return kInvalidIndex;
  }

  const size_t old_size = blob.size();
  blob.resize(static_cast<size_t>(end_offset), 0u);
  std::memcpy(blob.data() + old_size, data, static_cast<size_t>(byte_size));
  return static_cast<uint32_t>(aligned_offset);
}

template <typename T>
uint32_t append_aligned_array(std::vector<uint8_t>& blob, const T* data, size_t count, uint64_t alignment = alignof(T)) {
  if (count > (std::numeric_limits<uint64_t>::max() / sizeof(T))) {
    log::error("GPU RT: packed array size overflow (count=%llu, stride=%llu)", static_cast<uint64_t>(count), static_cast<uint64_t>(sizeof(T)));
    return kInvalidIndex;
  }
  return append_aligned_bytes(blob, data, static_cast<uint64_t>(count) * sizeof(T), alignment);
}

constexpr uint64_t kSceneBlobChunkSizeBytes = 512ull * 1024ull * 1024ull;

struct ChunkedPayloadLocation {
  uint32_t chunk_index = kInvalidIndex;
  uint32_t offset = kInvalidIndex;
};

struct ChunkedBlobPayloadBuilder {
  uint64_t chunk_size = kSceneBlobChunkSizeBytes;
  std::vector<uint8_t> payload_blob = {};
  std::vector<RHIChunkedBufferRange> chunk_ranges = {};
  std::vector<uint64_t> chunk_capacities = {};

  bool append(const void* data, uint64_t byte_size, uint64_t alignment, ChunkedPayloadLocation& location) {
    location = {};
    if ((data == nullptr) || (byte_size == 0u)) {
      return true;
    }

    uint64_t required_chunk_capacity = chunk_size;
    if (required_chunk_capacity < byte_size) {
      required_chunk_capacity = byte_size;
    }

    if (required_chunk_capacity > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
      log::error("GPU RT: chunk capacity exceeds 32-bit local offset range (%llu)", required_chunk_capacity);
      return false;
    }

    uint32_t target_chunk_index = kInvalidIndex;
    uint64_t aligned_offset = 0u;
    uint64_t chunk_start_offset = 0u;

    if (chunk_ranges.empty() == false) {
      const uint32_t last_chunk_index = static_cast<uint32_t>(chunk_ranges.size() - 1u);
      const uint64_t last_chunk_capacity = chunk_capacities[last_chunk_index];
      const auto& last_chunk_range = chunk_ranges[last_chunk_index];
      const uint64_t last_chunk_size = last_chunk_range.size;
      const uint64_t candidate_offset = align_up_u64(last_chunk_size, alignment);
      if ((candidate_offset <= last_chunk_capacity) && ((last_chunk_capacity - candidate_offset) >= byte_size)) {
        target_chunk_index = last_chunk_index;
        aligned_offset = candidate_offset;
        chunk_start_offset = last_chunk_range.offset;
      }
    }

    if (target_chunk_index == kInvalidIndex) {
      chunk_start_offset = static_cast<uint64_t>(payload_blob.size());
      chunk_ranges.push_back({.offset = chunk_start_offset, .size = 0u});
      chunk_capacities.push_back(required_chunk_capacity);

      target_chunk_index = static_cast<uint32_t>(chunk_ranges.size() - 1u);
      aligned_offset = 0u;
    }

    if (aligned_offset > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
      log::error("GPU RT: chunk local offset exceeds 32-bit range (%llu)", aligned_offset);
      return false;
    }

    auto& chunk_range = chunk_ranges[target_chunk_index];
    if (aligned_offset > chunk_range.size) {
      const uint64_t aligned_write_offset = chunk_start_offset + aligned_offset;
      if (aligned_write_offset > static_cast<uint64_t>(payload_blob.size())) {
        payload_blob.resize(static_cast<size_t>(aligned_write_offset), 0u);
      }
    }

    if (aligned_offset > (std::numeric_limits<uint64_t>::max() - byte_size)) {
      log::error("GPU RT: chunk payload range overflow (offset=%llu size=%llu)", aligned_offset, byte_size);
      return false;
    }

    const uint64_t end_offset = aligned_offset + byte_size;
    if (end_offset > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
      log::error("GPU RT: chunk payload exceeds 32-bit local range (%llu)", end_offset);
      return false;
    }

    const uint64_t chunk_capacity = chunk_capacities[target_chunk_index];
    if (end_offset > chunk_capacity) {
      log::error("GPU RT: chunk payload exceeds chunk capacity (end=%llu capacity=%llu)", end_offset, chunk_capacity);
      return false;
    }

    const uint64_t global_dst_offset = chunk_start_offset + aligned_offset;
    const uint64_t global_end_offset = chunk_start_offset + end_offset;
    if (global_end_offset > static_cast<uint64_t>(payload_blob.size())) {
      payload_blob.resize(static_cast<size_t>(global_end_offset), 0u);
    }
    std::memcpy(payload_blob.data() + static_cast<size_t>(global_dst_offset), data, static_cast<size_t>(byte_size));
    chunk_range.size = end_offset;

    location.chunk_index = target_chunk_index;
    location.offset = static_cast<uint32_t>(aligned_offset);
    return true;
  }
};

using PackedChunkedBlobBuildResult = RHIChunkedBufferUploadData;

PackedChunkedBlobBuildResult build_packed_images_blob(const SceneData& scene_data) {
  ETX_PROFILER_SCOPE();

  PackedChunkedBlobBuildResult result = {};
  GPUImageBlobHeader header = {};
  const uint64_t image_count_u64 = scene_data.images.array_size();
  if (image_count_u64 > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
    log::error("GPU RT: image count exceeds 32-bit ABI limit (%llu)", image_count_u64);
    result.success = false;
    return result;
  }

  header.image_count = static_cast<uint32_t>(image_count_u64);
  result.metadata = std::vector<uint8_t>(sizeof(GPUImageBlobHeader), 0u);
  std::vector<::Image> packed_images(header.image_count);
  ChunkedBlobPayloadBuilder payload_builder = {};

  const auto* images = scene_data.images.as_array();
  for (uint32_t i = 0u; i < header.image_count; ++i) {
    const auto& src = images[i];
    auto& dst = packed_images[i];

    PackedPayloadLocation pixel_payload = {};
    PackedPayloadLocation x_distribution_payload = {};
    PackedPayloadLocation y_distribution_payload = {};

    if (src.data.valid()) {
      const void* ptr = scene_data.buffer_pool.map(src.data);
      ChunkedPayloadLocation payload_location = {};
      const bool append_success = payload_builder.append(ptr, src.data.byte_size, 16u, payload_location);
      if (append_success == false) {
        log::error("GPU RT: failed to pack image payload for index %u", i);
        result.success = false;
        return result;
      }

      pixel_payload.offset = payload_location.offset;
      pixel_payload.chunk_index = payload_location.chunk_index;
      if ((src.data.byte_size > 0u) && (pixel_payload.offset == kInvalidIndex)) {
        log::error("GPU RT: failed to pack image payload for index %u", i);
        result.success = false;
        return result;
      }
    }

    if (src.x_distributions_storage.valid()) {
      const void* ptr = scene_data.buffer_pool.map(src.x_distributions_storage);
      ChunkedPayloadLocation payload_location = {};
      const bool append_success = payload_builder.append(ptr, src.x_distributions_storage.byte_size, alignof(Distribution::Entry), payload_location);
      if (append_success == false) {
        log::error("GPU RT: failed to pack image x-distribution payload for index %u", i);
        result.success = false;
        return result;
      }

      x_distribution_payload.offset = payload_location.offset;
      x_distribution_payload.chunk_index = payload_location.chunk_index;
      if ((src.x_distributions_storage.byte_size > 0u) && (x_distribution_payload.offset == kInvalidIndex)) {
        log::error("GPU RT: failed to pack image x-distribution payload for index %u", i);
        result.success = false;
        return result;
      }
    }

    if (src.y_distribution_storage.valid()) {
      const void* ptr = scene_data.buffer_pool.map(src.y_distribution_storage);
      ChunkedPayloadLocation payload_location = {};
      const bool append_success = payload_builder.append(ptr, src.y_distribution_storage.byte_size, alignof(Distribution::Entry), payload_location);
      if (append_success == false) {
        log::error("GPU RT: failed to pack image y-distribution payload for index %u", i);
        result.success = false;
        return result;
      }

      y_distribution_payload.offset = payload_location.offset;
      y_distribution_payload.chunk_index = payload_location.chunk_index;
      if ((src.y_distribution_storage.byte_size > 0u) && (y_distribution_payload.offset == kInvalidIndex)) {
        log::error("GPU RT: failed to pack image y-distribution payload for index %u", i);
        result.success = false;
        return result;
      }
    }

    dst = make_gpu_image_descriptor(src, pixel_payload, x_distribution_payload, y_distribution_payload);
  }

  header.images_offset = append_aligned_array(result.metadata, packed_images.data(), packed_images.size(), alignof(::Image));
  if ((packed_images.empty() == false) && (header.images_offset == kInvalidIndex)) {
    log::error("GPU RT: failed to pack image descriptors");
    result.success = false;
    return result;
  }

  if (payload_builder.chunk_ranges.size() > static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
    log::error("GPU RT: image chunk count exceeds 32-bit ABI limit (%llu)", static_cast<uint64_t>(payload_builder.chunk_ranges.size()));
    result.success = false;
    return result;
  }

  header.data_chunk_count = static_cast<uint32_t>(payload_builder.chunk_ranges.size());
  if (header.data_chunk_count > 0u) {
    std::vector<uint32_t> placeholder_chunk_indices(header.data_chunk_count, kInvalidDescriptorIndex);
    header.data_chunk_indices_offset = append_aligned_array(result.metadata, placeholder_chunk_indices.data(), placeholder_chunk_indices.size(), alignof(uint32_t));
    if (header.data_chunk_indices_offset == kInvalidIndex) {
      log::error("GPU RT: failed to pack image chunk descriptors table");
      result.success = false;
      return result;
    }
  } else {
    header.data_chunk_indices_offset = kInvalidIndex;
  }

  result.chunk_indices_offset = header.data_chunk_indices_offset;
  result.payload_data = std::move(payload_builder.payload_blob);
  result.payload_chunk_ranges = std::move(payload_builder.chunk_ranges);
  std::memcpy(result.metadata.data(), &header, sizeof(header));
  return result;
}

PackedChunkedBlobBuildResult build_packed_mediums_blob(const SceneData& scene_data) {
  ETX_PROFILER_SCOPE();

  PackedChunkedBlobBuildResult result = {};
  GPUMediumBlobHeader header = {};
  const uint64_t medium_count_u64 = scene_data.mediums.array_size();
  if (medium_count_u64 > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
    log::error("GPU RT: medium count exceeds 32-bit ABI limit (%llu)", medium_count_u64);
    result.success = false;
    return result;
  }

  header.medium_count = static_cast<uint32_t>(medium_count_u64);
  result.metadata = std::vector<uint8_t>(sizeof(GPUMediumBlobHeader), 0u);
  std::vector<::Medium> packed_mediums(header.medium_count);
  ChunkedBlobPayloadBuilder payload_builder = {};

  const auto* mediums = scene_data.mediums.as_array();
  for (uint32_t i = 0u; i < header.medium_count; ++i) {
    const auto& src = mediums[i];
    auto& dst = packed_mediums[i];

    PackedPayloadLocation density_payload = {};

    if (src.density_data.valid()) {
      const void* ptr = scene_data.buffer_pool.map(src.density_data);
      ChunkedPayloadLocation payload_location = {};
      const bool append_success = payload_builder.append(ptr, src.density_data.byte_size, alignof(float), payload_location);
      if (append_success == false) {
        log::error("GPU RT: failed to pack medium density payload for index %u", i);
        result.success = false;
        return result;
      }

      density_payload.offset = payload_location.offset;
      density_payload.chunk_index = payload_location.chunk_index;
      if ((src.density_data.byte_size > 0u) && (density_payload.offset == kInvalidIndex)) {
        log::error("GPU RT: failed to pack medium density payload for index %u", i);
        result.success = false;
        return result;
      }
    }

    dst = make_gpu_medium_descriptor(src, density_payload);
  }

  header.mediums_offset = append_aligned_array(result.metadata, packed_mediums.data(), packed_mediums.size(), alignof(::Medium));
  if ((packed_mediums.empty() == false) && (header.mediums_offset == kInvalidIndex)) {
    log::error("GPU RT: failed to pack medium descriptors");
    result.success = false;
    return result;
  }

  if (payload_builder.chunk_ranges.size() > static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
    log::error("GPU RT: medium chunk count exceeds 32-bit ABI limit (%llu)", static_cast<uint64_t>(payload_builder.chunk_ranges.size()));
    result.success = false;
    return result;
  }

  header.data_chunk_count = static_cast<uint32_t>(payload_builder.chunk_ranges.size());
  if (header.data_chunk_count > 0u) {
    std::vector<uint32_t> placeholder_chunk_indices(header.data_chunk_count, kInvalidDescriptorIndex);
    header.data_chunk_indices_offset = append_aligned_array(result.metadata, placeholder_chunk_indices.data(), placeholder_chunk_indices.size(), alignof(uint32_t));
    if (header.data_chunk_indices_offset == kInvalidIndex) {
      log::error("GPU RT: failed to pack medium chunk descriptors table");
      result.success = false;
      return result;
    }
  } else {
    header.data_chunk_indices_offset = kInvalidIndex;
  }

  result.chunk_indices_offset = header.data_chunk_indices_offset;
  result.payload_data = std::move(payload_builder.payload_blob);
  result.payload_chunk_ranges = std::move(payload_builder.chunk_ranges);
  std::memcpy(result.metadata.data(), &header, sizeof(header));
  return result;
}

GPUSceneGlobals build_scene_globals(const SceneData& scene_data, const PackedEmitterData& packed_emitters) {
  ETX_PROFILER_SCOPE();

  BoundingBox bbox = {};
  {
    ETX_PROFILER_NAMED_SCOPE("gpu_rt_compute_scene_bounds");
    bbox = scene_data.compute_bounding_volumes();
  }

  const float3 sphere_center = 0.5f * (bbox.p_min + bbox.p_max);
  const float sphere_radius = length(bbox.p_max - sphere_center);

  GPUSceneGlobals globals = {};
  globals.vertex_count = static_cast<uint32_t>(scene_data.vertices.pos.size());
  globals.triangle_count = static_cast<uint32_t>(scene_data.triangles.size());
  globals.mesh_count = static_cast<uint32_t>(scene_data.meshes.size());
  globals.emitter_profile_count = static_cast<uint32_t>(packed_emitters.emitter_profiles.size());
  globals.emitter_instance_count = static_cast<uint32_t>(packed_emitters.emitter_instances.size());
  globals.active_emitter_count = static_cast<uint32_t>(packed_emitters.active_emitter_indices.size());

  globals.bounding_sphere_center = sphere_center;
  globals.bounding_sphere_radius = sphere_radius;
  globals.bounding_box_min = bbox.p_min;
  globals.bounding_box_max = bbox.p_max;

  globals.default_black_spectrum = scene_data.defaults.black_spectrum;
  globals.default_white_spectrum = scene_data.defaults.white_spectrum;
  globals.default_rayleigh_spectrum = scene_data.defaults.rayleigh_spectrum;
  globals.default_mie_spectrum = scene_data.defaults.mie_spectrum;
  globals.default_ozone_spectrum = scene_data.defaults.ozone_spectrum;
  globals.default_subsurface_scatter_material = scene_data.defaults.subsurface_scatter_material;
  globals.default_subsurface_exit_material = scene_data.defaults.subsurface_exit_material;
  globals.default_missing_material = scene_data.defaults.missing_material;
  globals.default_dielectric_eta = scene_data.defaults.dielectric_eta;
  globals.default_conductor_eta = scene_data.defaults.conductor_eta;
  globals.default_conductor_k = scene_data.defaults.conductor_k;
  globals.pixel_filter_image_index = scene_data.pixel_filter.image_index;
  globals.pixel_filter_radius = scene_data.pixel_filter.radius;

  globals.environment_emitter_count = packed_emitters.environment_emitters.count;
  for (uint32_t i = 0u; i < packed_emitters.environment_emitters.count; ++i) {
    globals.environment_emitters[i] = packed_emitters.environment_emitters.emitters[i];
  }

  return globals;
}

GPUSceneOptions build_scene_options(const SceneRepresentation& scene) {
  const SceneData& scene_data = scene.data();
  GPUSceneOptions options = {};
  options.min_path_length = scene_data.options.min_path_length;
  options.max_path_length = scene_data.options.max_path_length;
  options.samples = scene_data.options.samples;
  options.random_path_termination = scene_data.options.random_path_termination;
  options.random_seed = scene_data.options.random_seed;
  options.noise_threshold = scene_data.options.noise_threshold;
  options.radiance_clamp = scene_data.options.radiance_clamp;
  options.strategy_flags = scene_data.options.strategy_flags;
  options.light_sampling = static_cast<uint32_t>(scene_data.options.light_sampling);

  uint32_t properties_flags = 0u;
  for (uint32_t i = 0u; i < Scene::Properties::Count; ++i) {
    if (scene_data.options.properties[i]) {
      properties_flags |= (1u << i);
    }
  }
  options.properties_flags = properties_flags;
  options.path_mode = static_cast<uint32_t>(gpu_path_mode_from_scene(scene));
  return options;
}
}  // namespace

GPURaytracingRenderer::GPURaytracingRenderer(TaskScheduler& s)
  : Renderer(s) {
  _gpu_scene = make_invalid_gpu_scene();
}

GPURaytracingRenderer::~GPURaytracingRenderer() {
}

bool GPURaytracingRenderer::set_render_window(const uint2& origin, const uint2& size, const uint2& full_size) {
  if ((size.x == 0u) || (size.y == 0u)) {
    return false;
  }

  if ((full_size.x == 0u) || (full_size.y == 0u)) {
    return false;
  }

  if ((origin.x >= full_size.x) || (origin.y >= full_size.y) || (size.x > (full_size.x - origin.x)) || (size.y > (full_size.y - origin.y))) {
    return false;
  }

  _render_window_origin = origin;
  _render_window_size = size;
  return true;
}

void GPURaytracingRenderer::reset_render_window() {
  _render_window_origin = {};
  _render_window_size = {};
}

void GPURaytracingRenderer::reset_runtime_failure() {
  _runtime_failed = false;
  _runtime_failure_reason.clear();
}

void GPURaytracingRenderer::set_runtime_failure(std::string message) {
  if (_runtime_failed == false) {
    _runtime_failure_reason = std::move(message);
  }
  _runtime_failed = true;
}

void GPURaytracingRenderer::init(RHIContext& ctx, SceneRepresentation& scene) {
  ETX_PROFILER_SCOPE();

  Renderer::init(ctx, scene);

  reset_runtime_failure();
  _path_mode = static_cast<uint32_t>(gpu_path_mode_from_scene(scene));
  _material_compile_mask = build_material_compile_mask(scene.data());
  _initialized = true;
}

void GPURaytracingRenderer::create_pipelines(RHIContext& ctx) {
  ETX_PROFILER_SCOPE();

  auto& device = ctx.device();
  _compile_filter_matched = false;

  for (auto& pipeline : _pipelines) {
    if (pipeline.valid()) {
      device.destroy_pipeline(pipeline);
      pipeline = {};
    }
  }

  struct WavefrontStage {
    PipelineStage stage = PipelineStage::PrepareSample;
    const char* source_file = nullptr;
    const char* entry_point = nullptr;
    const char* optimization_level = nullptr;
    const char* bsdf_kind = nullptr;
  };

  const WavefrontStage stages[] = {
    {PipelineStage::PrepareSample, "shaders/gpu_rt_wavefront_prepare.hlsl", "wavefront_prepare_sample_main", nullptr, nullptr},
    {PipelineStage::InitCameraPath0, "shaders/gpu_rt_wavefront_init_camera.hlsl", "wavefront_init_camera_path_0_main", nullptr, nullptr},
    {PipelineStage::InitLightPath0, "shaders/gpu_rt_wavefront_init_light.hlsl", "wavefront_init_light_path_0_main", nullptr, nullptr},
    {PipelineStage::TraceCamera, "shaders/gpu_rt_wavefront_trace_camera.hlsl", "wavefront_trace_camera_main", nullptr, nullptr},
    {PipelineStage::CameraSurfaceClassify, "shaders/gpu_rt_wavefront_surface_camera.hlsl", "wavefront_camera_surface_classify_main", nullptr, nullptr},
    {PipelineStage::CameraDirectLightSample, "shaders/gpu_rt_wavefront_direct_light_sample.hlsl", "wavefront_camera_direct_light_sample_main", "0", nullptr},
    {PipelineStage::CameraDirectLightPrepareDiffuse, "shaders/gpu_rt_wavefront_direct_light_prepare_variant.hlsl", "wavefront_camera_direct_light_prepare_diffuse_main", "0", "1"},
    {PipelineStage::CameraDirectLightPreparePlastic, "shaders/gpu_rt_wavefront_direct_light_prepare_variant.hlsl", "wavefront_camera_direct_light_prepare_plastic_main", "0", "2"},
    {PipelineStage::CameraDirectLightPrepareConductor, "shaders/gpu_rt_wavefront_direct_light_prepare_variant.hlsl", "wavefront_camera_direct_light_prepare_conductor_main", "0",
      "3"},
    {PipelineStage::CameraDirectLightPrepareDielectricEval, "shaders/gpu_rt_wavefront_direct_light_prepare_dielectric_eval.hlsl",
      "wavefront_camera_direct_light_prepare_dielectric_eval_main", "0", nullptr},
    {PipelineStage::CameraDirectLightPrepareDielectricPdf, "shaders/gpu_rt_wavefront_direct_light_prepare_dielectric_pdf.hlsl",
      "wavefront_camera_direct_light_prepare_dielectric_pdf_main", "0", nullptr},
    {PipelineStage::CameraDirectLightShadow, "shaders/gpu_rt_wavefront_shadow.hlsl", "wavefront_camera_direct_light_shadow_main", nullptr, nullptr},
    {PipelineStage::CameraDirectLightAccumulate, "shaders/gpu_rt_wavefront_direct_light.hlsl", "wavefront_camera_direct_light_accumulate_main", nullptr, nullptr},
    {PipelineStage::CameraDirectHitAccumulate, "shaders/gpu_rt_wavefront_direct_hit.hlsl", "wavefront_camera_direct_hit_accumulate_main", nullptr, nullptr},
    {PipelineStage::CameraConnectLightPrepareDiffuse, "shaders/gpu_rt_wavefront_connect_light_prepare_variant.hlsl", "wavefront_camera_connect_light_prepare_diffuse_main", "0",
      "1"},
    {PipelineStage::CameraConnectLightPreparePlastic, "shaders/gpu_rt_wavefront_connect_light_prepare_variant.hlsl", "wavefront_camera_connect_light_prepare_plastic_main", "0",
      "2"},
    {PipelineStage::CameraConnectLightPrepareConductor, "shaders/gpu_rt_wavefront_connect_light_prepare_variant.hlsl", "wavefront_camera_connect_light_prepare_conductor_main", "0",
      "3"},
    {PipelineStage::CameraConnectLightPrepareDielectric, "shaders/gpu_rt_wavefront_connect_light_prepare_variant.hlsl", "wavefront_camera_connect_light_prepare_dielectric_main",
      "0", "4"},
    {PipelineStage::CameraConnectLightShadow, "shaders/gpu_rt_wavefront_shadow.hlsl", "wavefront_camera_connect_light_shadow_main", nullptr, nullptr},
    {PipelineStage::CameraConnectLightAccumulate, "shaders/gpu_rt_wavefront_connect_light_path.hlsl", "wavefront_camera_connect_light_accumulate_main", nullptr, nullptr},
    {PipelineStage::CameraContinuePrepareDiffuse, "shaders/gpu_rt_wavefront_surface_continue_prepare_camera_variant.hlsl", "wavefront_camera_continue_prepare_diffuse_main", "0",
      "1"},
    {PipelineStage::CameraContinuePreparePlastic, "shaders/gpu_rt_wavefront_surface_continue_prepare_camera_variant.hlsl", "wavefront_camera_continue_prepare_plastic_main", "0",
      "2"},
    {PipelineStage::CameraContinuePrepareConductor, "shaders/gpu_rt_wavefront_surface_continue_prepare_camera_variant.hlsl", "wavefront_camera_continue_prepare_conductor_main",
      "0", "3"},
    {PipelineStage::CameraContinuePrepareDielectric, "shaders/gpu_rt_wavefront_surface_continue_prepare_camera_variant.hlsl", "wavefront_camera_continue_prepare_dielectric_main",
      "0", "4"},
    {PipelineStage::CameraContinuePrepareThinfilm, "shaders/gpu_rt_wavefront_surface_continue_prepare_camera_variant.hlsl", "wavefront_camera_continue_prepare_thinfilm_main", "0",
      "5"},
    {PipelineStage::CameraContinueFinalize, "shaders/gpu_rt_wavefront_surface_camera.hlsl", "wavefront_camera_continue_finalize_main", nullptr, nullptr},
    {PipelineStage::TraceLight, "shaders/gpu_rt_wavefront_trace_light.hlsl", "wavefront_trace_light_main", nullptr, nullptr},
    {PipelineStage::LightSurfaceClassify, "shaders/gpu_rt_wavefront_surface_light.hlsl", "wavefront_light_surface_classify_main", nullptr, nullptr},
    {PipelineStage::LightContinuePrepareDiffuse, "shaders/gpu_rt_wavefront_surface_continue_prepare_light_variant.hlsl", "wavefront_light_continue_prepare_diffuse_main", "0", "1"},
    {PipelineStage::LightContinuePreparePlastic, "shaders/gpu_rt_wavefront_surface_continue_prepare_light_variant.hlsl", "wavefront_light_continue_prepare_plastic_main", "0", "2"},
    {PipelineStage::LightContinuePrepareConductor, "shaders/gpu_rt_wavefront_surface_continue_prepare_light_variant.hlsl", "wavefront_light_continue_prepare_conductor_main", "0",
      "3"},
    {PipelineStage::LightContinuePrepareDielectric, "shaders/gpu_rt_wavefront_surface_continue_prepare_light_variant.hlsl", "wavefront_light_continue_prepare_dielectric_main", "0",
      "4"},
    {PipelineStage::LightContinuePrepareThinfilm, "shaders/gpu_rt_wavefront_surface_continue_prepare_light_variant.hlsl", "wavefront_light_continue_prepare_thinfilm_main", "0",
      "5"},
    {PipelineStage::LightConnectCameraPrepareDiffuse, "shaders/gpu_rt_wavefront_connect_camera_prepare_variant.hlsl", "wavefront_light_connect_camera_prepare_diffuse_main", "0",
      "1"},
    {PipelineStage::LightConnectCameraPreparePlastic, "shaders/gpu_rt_wavefront_connect_camera_prepare_variant.hlsl", "wavefront_light_connect_camera_prepare_plastic_main", "0",
      "2"},
    {PipelineStage::LightConnectCameraPrepareConductor, "shaders/gpu_rt_wavefront_connect_camera_prepare_variant.hlsl", "wavefront_light_connect_camera_prepare_conductor_main",
      "0", "3"},
    {PipelineStage::LightConnectCameraPrepareDielectric, "shaders/gpu_rt_wavefront_connect_camera_prepare_variant.hlsl", "wavefront_light_connect_camera_prepare_dielectric_main",
      "0", "4"},
    {PipelineStage::LightConnectCameraShadow, "shaders/gpu_rt_wavefront_shadow.hlsl", "wavefront_light_connect_camera_shadow_main", nullptr, nullptr},
    {PipelineStage::LightConnectCameraAccumulate, "shaders/gpu_rt_wavefront_connect_camera.hlsl", "wavefront_light_connect_camera_accumulate_main", nullptr, nullptr},
    {PipelineStage::LightContinueFinalize, "shaders/gpu_rt_wavefront_surface_light.hlsl", "wavefront_light_continue_finalize_main", nullptr, nullptr},
    {PipelineStage::SwapQueues, "shaders/gpu_rt_wavefront_prepare.hlsl", "wavefront_swap_queues_main", nullptr, nullptr},
    {PipelineStage::FinalizeSample, "shaders/gpu_rt_wavefront_prepare.hlsl", "wavefront_finalize_sample_main", nullptr, nullptr},
  };

  const GPUPathMode path_mode = static_cast<GPUPathMode>(_path_mode);
  const bool enable_camera_path = path_mode != GPUPathMode::LightTracing;
  const bool enable_light_path = path_mode != GPUPathMode::PathTracing;
  const bool enable_direct_light = enable_camera_path;
  const bool enable_connect_to_camera = enable_light_path;
  const bool has_diffuse = material_compile_mask_has(_material_compile_mask, MaterialClass::Diffuse);
  const bool has_plastic = material_compile_mask_has(_material_compile_mask, MaterialClass::Plastic);
  const bool has_conductor = material_compile_mask_has(_material_compile_mask, MaterialClass::Conductor);
  const bool has_dielectric = material_compile_mask_has(_material_compile_mask, MaterialClass::Dielectric);
  const bool has_thinfilm = material_compile_mask_has(_material_compile_mask, MaterialClass::Thinfilm);
  const auto stage_enabled_for_scene = [&](PipelineStage stage) {
    switch (stage) {
      case PipelineStage::InitCameraPath0:
      case PipelineStage::TraceCamera:
      case PipelineStage::CameraSurfaceClassify:
      case PipelineStage::CameraContinuePrepareDiffuse:
        return enable_camera_path && has_diffuse;
      case PipelineStage::CameraContinuePreparePlastic:
        return enable_camera_path && has_plastic;
      case PipelineStage::CameraContinuePrepareConductor:
        return enable_camera_path && has_conductor;
      case PipelineStage::CameraContinuePrepareDielectric:
        return enable_camera_path && has_dielectric;
      case PipelineStage::CameraContinuePrepareThinfilm:
        return enable_camera_path && has_thinfilm;
      case PipelineStage::CameraContinueFinalize:
        return enable_camera_path;
      case PipelineStage::CameraDirectLightSample:
      case PipelineStage::CameraDirectLightShadow:
      case PipelineStage::CameraDirectLightAccumulate:
      case PipelineStage::CameraDirectHitAccumulate:
        return enable_direct_light;
      case PipelineStage::CameraDirectLightPrepareDiffuse:
        return enable_direct_light && has_diffuse;
      case PipelineStage::CameraDirectLightPreparePlastic:
        return enable_direct_light && has_plastic;
      case PipelineStage::CameraDirectLightPrepareConductor:
        return enable_direct_light && has_conductor;
      case PipelineStage::CameraDirectLightPrepareDielectricEval:
      case PipelineStage::CameraDirectLightPrepareDielectricPdf:
        return false;
      case PipelineStage::CameraConnectLightPrepareDiffuse:
      case PipelineStage::CameraConnectLightPreparePlastic:
      case PipelineStage::CameraConnectLightPrepareConductor:
      case PipelineStage::CameraConnectLightPrepareDielectric:
      case PipelineStage::CameraConnectLightShadow:
      case PipelineStage::CameraConnectLightAccumulate:
        return false;
      case PipelineStage::InitLightPath0:
      case PipelineStage::TraceLight:
      case PipelineStage::LightSurfaceClassify:
      case PipelineStage::LightContinuePrepareDiffuse:
        return enable_light_path && has_diffuse;
      case PipelineStage::LightContinuePreparePlastic:
        return enable_light_path && has_plastic;
      case PipelineStage::LightContinuePrepareConductor:
        return enable_light_path && has_conductor;
      case PipelineStage::LightContinuePrepareDielectric:
        return enable_light_path && has_dielectric;
      case PipelineStage::LightContinuePrepareThinfilm:
        return enable_light_path && has_thinfilm;
      case PipelineStage::LightContinueFinalize:
        return enable_light_path;
      case PipelineStage::LightConnectCameraShadow:
      case PipelineStage::LightConnectCameraAccumulate:
        return enable_connect_to_camera;
      case PipelineStage::LightConnectCameraPrepareDiffuse:
        return enable_connect_to_camera && has_diffuse;
      case PipelineStage::LightConnectCameraPreparePlastic:
        return enable_connect_to_camera && has_plastic;
      case PipelineStage::LightConnectCameraPrepareConductor:
        return enable_connect_to_camera && has_conductor;
      case PipelineStage::LightConnectCameraPrepareDielectric:
        return enable_connect_to_camera && has_dielectric;
      case PipelineStage::PrepareSample:
      case PipelineStage::SwapQueues:
      case PipelineStage::FinalizeSample:
        return true;
      default:
        return true;
    }
  };

  auto& compiler = ShaderCompiler::instance();
  for (const auto& stage_info : stages) {
    if ((_compile_stage_filter.empty() == false) && (_compile_stage_filter != stage_info.entry_point)) {
      continue;
    }
    if ((_compile_stage_filter.empty()) && (stage_enabled_for_scene(stage_info.stage) == false)) {
      continue;
    }

    _compile_filter_matched = true;
    RHIPipeline pipeline = {};
    if (compile_timed_compute_stage(device, compiler, stage_info.source_file, stage_info.entry_point, stage_info.optimization_level, stage_info.bsdf_kind, pipeline) == false) {
      set_runtime_failure("GPU shader pipeline compilation failed at '" + std::string(stage_info.entry_point) + "'");
      for (auto& pipeline : _pipelines) {
        if (pipeline.valid()) {
          device.destroy_pipeline(pipeline);
          pipeline = {};
        }
      }
      return;
    }

    _pipelines[static_cast<uint32_t>(stage_info.stage)] = pipeline;
  }

  if ((_compile_stage_filter.empty() == false) && (_compile_filter_matched == false)) {
    log::error("GPU RT: compile stage filter '%s' did not match any pipeline entry point", _compile_stage_filter.c_str());
  }
}

void GPURaytracingRenderer::reload_shaders(RHIContext& ctx) {
  ETX_PROFILER_SCOPE();
  reset_runtime_failure();
  create_pipelines(ctx);
}

bool GPURaytracingRenderer::pipelines_valid() const {
  if (_compile_stage_filter.empty() == false) {
    if (_compile_filter_matched == false) {
      return false;
    }

    for (const auto& pipeline : _pipelines) {
      if (pipeline.valid()) {
        return true;
      }
    }

    return false;
  }

  return _pipelines[static_cast<uint32_t>(PipelineStage::PrepareSample)].valid();
}

void GPURaytracingRenderer::set_compile_stage_filter(const std::string& value) {
  _compile_stage_filter = value;
}

void GPURaytracingRenderer::destroy_scene_buffers(RHIContext& ctx) {
  ETX_PROFILER_SCOPE();

  auto& device = ctx.device();
  destroy_linear_scene_buffer(device, _vertex_normals_buffer, _vertex_normals_buffer_size, _gpu_scene.vertex_normals);
  destroy_linear_scene_buffer(device, _vertex_tangents_buffer, _vertex_tangents_buffer_size, _gpu_scene.vertex_tangents);
  destroy_linear_scene_buffer(device, _vertex_bitangents_buffer, _vertex_bitangents_buffer_size, _gpu_scene.vertex_bitangents);
  destroy_linear_scene_buffer(device, _vertex_texcoords_buffer, _vertex_texcoords_buffer_size, _gpu_scene.vertex_texcoords);
  destroy_linear_scene_buffer(device, _triangles_buffer, _triangles_buffer_size, _gpu_scene.triangles);
  destroy_linear_scene_buffer(device, _meshes_buffer, _meshes_buffer_size, _gpu_scene.meshes);
  destroy_linear_scene_buffer(device, _emitter_profiles_buffer, _emitter_profiles_buffer_size, _gpu_scene.emitter_profiles);
  destroy_linear_scene_buffer(device, _emitter_instances_buffer, _emitter_instances_buffer_size, _gpu_scene.emitter_instances);
  destroy_linear_scene_buffer(device, _materials_buffer, _materials_buffer_size, _gpu_scene.materials);
  destroy_linear_scene_buffer(device, _spectrums_buffer, _spectrums_buffer_size, _gpu_scene.spectrums);
  device.destroy_chunked_buffer(_images_blob_state);
  _gpu_scene.images = kInvalidDescriptorIndex;
  device.destroy_chunked_buffer(_mediums_blob_state);
  _gpu_scene.mediums = kInvalidDescriptorIndex;
  destroy_linear_scene_buffer(device, _scene_globals_buffer, _scene_globals_buffer_size, _gpu_scene.scene_globals);
  destroy_linear_scene_buffer(device, _scene_options_buffer, _scene_options_buffer_size, _gpu_scene.scene_options);
  destroy_linear_scene_buffer(device, _emitters_distribution_buffer, _emitters_distribution_buffer_size, _gpu_scene.emitters_distribution);
}

void GPURaytracingRenderer::destroy_wavefront_buffers(RHIContext& ctx) {
  ETX_PROFILER_SCOPE();

  auto& device = ctx.device();
  destroy_linear_scene_buffer(device, _wavefront_resources_buffer, _wavefront_resources_buffer_size, _wavefront_resources_buffer_descriptor_index);
  destroy_linear_scene_buffer(device, _camera_state_buffer, _camera_state_buffer_size, _camera_state_buffer_descriptor_index);
  destroy_linear_scene_buffer(device, _light_state_buffer, _light_state_buffer_size, _light_state_buffer_descriptor_index);
  destroy_linear_scene_buffer(device, _camera_hit_buffer, _camera_hit_buffer_size, _camera_hit_buffer_descriptor_index);
  destroy_linear_scene_buffer(device, _light_hit_buffer, _light_hit_buffer_size, _light_hit_buffer_descriptor_index);
  destroy_linear_scene_buffer(device, _camera_queue_a_buffer, _camera_queue_a_buffer_size, _camera_queue_a_buffer_descriptor_index);
  destroy_linear_scene_buffer(device, _camera_queue_b_buffer, _camera_queue_b_buffer_size, _camera_queue_b_buffer_descriptor_index);
  destroy_linear_scene_buffer(device, _light_queue_a_buffer, _light_queue_a_buffer_size, _light_queue_a_buffer_descriptor_index);
  destroy_linear_scene_buffer(device, _light_queue_b_buffer, _light_queue_b_buffer_size, _light_queue_b_buffer_descriptor_index);
  destroy_linear_scene_buffer(device, _camera_queue_count_readback_buffer, _camera_queue_count_readback_buffer_size, _camera_queue_count_readback_buffer_descriptor_index);
  destroy_linear_scene_buffer(device, _light_queue_count_readback_buffer, _light_queue_count_readback_buffer_size, _light_queue_count_readback_buffer_descriptor_index);
  destroy_linear_scene_buffer(device, _camera_vertex_buffer, _camera_vertex_buffer_size, _camera_vertex_buffer_descriptor_index);
  destroy_linear_scene_buffer(device, _light_vertex_buffer, _light_vertex_buffer_size, _light_vertex_buffer_descriptor_index);
  destroy_linear_scene_buffer(device, _film_buffer, _film_buffer_size, _film_buffer_descriptor_index);
  destroy_linear_scene_buffer(device, _path_meta_buffer, _path_meta_buffer_size, _path_meta_buffer_descriptor_index);
  destroy_linear_scene_buffer(device, _direct_light_sample_buffer, _direct_light_sample_buffer_size, _direct_light_sample_buffer_descriptor_index);
  destroy_linear_scene_buffer(device, _direct_light_task_buffer, _direct_light_task_buffer_size, _direct_light_task_buffer_descriptor_index);
  destroy_linear_scene_buffer(device, _direct_light_result_buffer, _direct_light_result_buffer_size, _direct_light_result_buffer_descriptor_index);
  destroy_linear_scene_buffer(device, _connect_light_task_buffer, _connect_light_task_buffer_size, _connect_light_task_buffer_descriptor_index);
  destroy_linear_scene_buffer(device, _connect_light_result_buffer, _connect_light_result_buffer_size, _connect_light_result_buffer_descriptor_index);
  destroy_linear_scene_buffer(device, _connect_camera_task_buffer, _connect_camera_task_buffer_size, _connect_camera_task_buffer_descriptor_index);
  destroy_linear_scene_buffer(device, _connect_camera_result_buffer, _connect_camera_result_buffer_size, _connect_camera_result_buffer_descriptor_index);
  _wavefront_path_capacity = 0u;
  _wavefront_vertex_capacity = 0u;
}

bool GPURaytracingRenderer::ensure_wavefront_buffers(RHIContext& ctx, const SceneRepresentation& scene) {
  ETX_PROFILER_SCOPE();

  const uint2 film_size = scene.camera().film_size;
  if ((film_size.x == 0u) || (film_size.y == 0u)) {
    return false;
  }

  const uint64_t pixel_count_u64 = static_cast<uint64_t>(film_size.x) * static_cast<uint64_t>(film_size.y);
  if (pixel_count_u64 > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
    log::error("GPU RT: wavefront path capacity overflow");
    return false;
  }

  const uint32_t path_capacity = static_cast<uint32_t>(pixel_count_u64);
  const uint32_t scene_max_path_length = std::max(1u, scene.data().options.max_path_length);
  const uint32_t camera_history_bounces = kWavefrontRollingHistoryBounces;
  const uint32_t light_history_bounces = kWavefrontRollingHistoryBounces;
  const uint32_t stored_history_bounces = kWavefrontRollingHistoryBounces;
  const uint32_t max_path_length = scene_max_path_length;
  const uint64_t vertex_capacity_u64 = static_cast<uint64_t>(path_capacity) * static_cast<uint64_t>(stored_history_bounces + 1u);
  if (vertex_capacity_u64 > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
    log::error("GPU RT: wavefront vertex capacity overflow");
    return false;
  }
  const uint32_t vertex_capacity = static_cast<uint32_t>(vertex_capacity_u64);

  const uint64_t queue_buffer_size = kGPUWavefrontQueueHeaderSize + static_cast<uint64_t>(path_capacity) * sizeof(uint32_t);
  const uint64_t path_state_buffer_size = static_cast<uint64_t>(path_capacity) * kGPUWavefrontPathStateStride;
  const uint64_t hit_buffer_size = static_cast<uint64_t>(path_capacity) * kGPUWavefrontHitStride;
  const uint64_t vertex_buffer_size = static_cast<uint64_t>(vertex_capacity) * kGPUWavefrontPathVertexStride;
  const uint64_t film_buffer_size = static_cast<uint64_t>(path_capacity) * sizeof(float4);
  const uint64_t path_meta_buffer_size = static_cast<uint64_t>(path_capacity) * kGPUWavefrontPathMetaStride;
  const uint64_t direct_light_sample_buffer_size = static_cast<uint64_t>(path_capacity) * kGPUWavefrontDirectLightSampleStride;
  const uint64_t direct_light_task_buffer_size = static_cast<uint64_t>(path_capacity) * kGPUWavefrontDirectLightTaskStride;
  const uint64_t direct_light_result_buffer_size = static_cast<uint64_t>(path_capacity) * kGPUWavefrontDirectLightResultStride;
  const uint64_t connect_light_task_buffer_size = static_cast<uint64_t>(vertex_capacity) * kGPUWavefrontConnectLightTaskStride;
  const uint64_t connect_light_result_buffer_size = static_cast<uint64_t>(vertex_capacity) * kGPUWavefrontConnectLightResultStride;
  const uint64_t connect_camera_task_buffer_size = static_cast<uint64_t>(path_capacity) * kGPUWavefrontConnectCameraTaskStride;
  const uint64_t connect_camera_result_buffer_size = static_cast<uint64_t>(path_capacity) * kGPUWavefrontConnectCameraResultStride;
  const RHIBufferUsage wavefront_usage = RHIBufferUsage::Storage | RHIBufferUsage::TransferDst;
  const RHIBufferUsage queue_buffer_usage = RHIBufferUsage::Storage | RHIBufferUsage::TransferDst | RHIBufferUsage::TransferSrc;
  const RHIBufferUsage queue_readback_usage = RHIBufferUsage::TransferDst;

  auto& device = ctx.device();
  if (ensure_storage_buffer(device, path_state_buffer_size, wavefront_usage, _camera_state_buffer, _camera_state_buffer_size, _camera_state_buffer_descriptor_index,
        "wavefront_camera_state") == false) {
    return false;
  }
  if (ensure_storage_buffer(device, path_state_buffer_size, wavefront_usage, _light_state_buffer, _light_state_buffer_size, _light_state_buffer_descriptor_index,
        "wavefront_light_state") == false) {
    return false;
  }
  if (ensure_storage_buffer(device, hit_buffer_size, wavefront_usage, _camera_hit_buffer, _camera_hit_buffer_size, _camera_hit_buffer_descriptor_index, "wavefront_camera_hit") ==
      false) {
    return false;
  }
  if (ensure_storage_buffer(device, hit_buffer_size, wavefront_usage, _light_hit_buffer, _light_hit_buffer_size, _light_hit_buffer_descriptor_index, "wavefront_light_hit") ==
      false) {
    return false;
  }
  if (ensure_storage_buffer(device, queue_buffer_size, queue_buffer_usage, _camera_queue_a_buffer, _camera_queue_a_buffer_size, _camera_queue_a_buffer_descriptor_index,
        "wavefront_camera_queue_a") == false) {
    return false;
  }
  if (ensure_storage_buffer(device, queue_buffer_size, queue_buffer_usage, _camera_queue_b_buffer, _camera_queue_b_buffer_size, _camera_queue_b_buffer_descriptor_index,
        "wavefront_camera_queue_b") == false) {
    return false;
  }
  if (ensure_storage_buffer(device, queue_buffer_size, queue_buffer_usage, _light_queue_a_buffer, _light_queue_a_buffer_size, _light_queue_a_buffer_descriptor_index,
        "wavefront_light_queue_a") == false) {
    return false;
  }
  if (ensure_storage_buffer(device, queue_buffer_size, queue_buffer_usage, _light_queue_b_buffer, _light_queue_b_buffer_size, _light_queue_b_buffer_descriptor_index,
        "wavefront_light_queue_b") == false) {
    return false;
  }
  if (ensure_host_visible_buffer(device, kGPUWavefrontQueueHeaderSize, queue_readback_usage, _camera_queue_count_readback_buffer, _camera_queue_count_readback_buffer_size,
        _camera_queue_count_readback_buffer_descriptor_index, "wavefront_camera_queue_count_readback") == false) {
    return false;
  }
  if (ensure_host_visible_buffer(device, kGPUWavefrontQueueHeaderSize, queue_readback_usage, _light_queue_count_readback_buffer, _light_queue_count_readback_buffer_size,
        _light_queue_count_readback_buffer_descriptor_index, "wavefront_light_queue_count_readback") == false) {
    return false;
  }
  if (ensure_storage_buffer(device, vertex_buffer_size, wavefront_usage, _camera_vertex_buffer, _camera_vertex_buffer_size, _camera_vertex_buffer_descriptor_index,
        "wavefront_camera_vertex") == false) {
    return false;
  }
  if (ensure_storage_buffer(device, vertex_buffer_size, wavefront_usage, _light_vertex_buffer, _light_vertex_buffer_size, _light_vertex_buffer_descriptor_index,
        "wavefront_light_vertex") == false) {
    return false;
  }
  if (ensure_storage_buffer(device, film_buffer_size, wavefront_usage, _film_buffer, _film_buffer_size, _film_buffer_descriptor_index, "wavefront_film") == false) {
    return false;
  }
  if (ensure_storage_buffer(device, path_meta_buffer_size, wavefront_usage, _path_meta_buffer, _path_meta_buffer_size, _path_meta_buffer_descriptor_index, "wavefront_path_meta") ==
      false) {
    return false;
  }
  if (ensure_storage_buffer(device, direct_light_sample_buffer_size, wavefront_usage, _direct_light_sample_buffer, _direct_light_sample_buffer_size,
        _direct_light_sample_buffer_descriptor_index, "wavefront_direct_light_sample") == false) {
    return false;
  }
  if (ensure_storage_buffer(device, direct_light_task_buffer_size, wavefront_usage, _direct_light_task_buffer, _direct_light_task_buffer_size,
        _direct_light_task_buffer_descriptor_index, "wavefront_direct_light_task") == false) {
    return false;
  }
  if (ensure_storage_buffer(device, direct_light_result_buffer_size, wavefront_usage, _direct_light_result_buffer, _direct_light_result_buffer_size,
        _direct_light_result_buffer_descriptor_index, "wavefront_direct_light_result") == false) {
    return false;
  }
  if (ensure_storage_buffer(device, connect_light_task_buffer_size, wavefront_usage, _connect_light_task_buffer, _connect_light_task_buffer_size,
        _connect_light_task_buffer_descriptor_index, "wavefront_connect_light_task") == false) {
    return false;
  }
  if (ensure_storage_buffer(device, connect_light_result_buffer_size, wavefront_usage, _connect_light_result_buffer, _connect_light_result_buffer_size,
        _connect_light_result_buffer_descriptor_index, "wavefront_connect_light_result") == false) {
    return false;
  }
  if (ensure_storage_buffer(device, connect_camera_task_buffer_size, wavefront_usage, _connect_camera_task_buffer, _connect_camera_task_buffer_size,
        _connect_camera_task_buffer_descriptor_index, "wavefront_connect_camera_task") == false) {
    return false;
  }
  if (ensure_storage_buffer(device, connect_camera_result_buffer_size, wavefront_usage, _connect_camera_result_buffer, _connect_camera_result_buffer_size,
        _connect_camera_result_buffer_descriptor_index, "wavefront_connect_camera_result") == false) {
    return false;
  }

  GPUWavefrontResources resources = {};
  resources.camera_state_buffer = _camera_state_buffer_descriptor_index;
  resources.light_state_buffer = _light_state_buffer_descriptor_index;
  resources.camera_hit_buffer = _camera_hit_buffer_descriptor_index;
  resources.light_hit_buffer = _light_hit_buffer_descriptor_index;
  resources.camera_queue_a_buffer = _camera_queue_a_buffer_descriptor_index;
  resources.camera_queue_b_buffer = _camera_queue_b_buffer_descriptor_index;
  resources.light_queue_a_buffer = _light_queue_a_buffer_descriptor_index;
  resources.light_queue_b_buffer = _light_queue_b_buffer_descriptor_index;
  resources.camera_vertex_buffer = _camera_vertex_buffer_descriptor_index;
  resources.light_vertex_buffer = _light_vertex_buffer_descriptor_index;
  resources.film_buffer = _film_buffer_descriptor_index;
  resources.path_meta_buffer = _path_meta_buffer_descriptor_index;
  resources.direct_light_sample_buffer = _direct_light_sample_buffer_descriptor_index;
  resources.direct_light_task_buffer = _direct_light_task_buffer_descriptor_index;
  resources.direct_light_result_buffer = _direct_light_result_buffer_descriptor_index;
  resources.connect_light_task_buffer = _connect_light_task_buffer_descriptor_index;
  resources.connect_light_result_buffer = _connect_light_result_buffer_descriptor_index;
  resources.connect_camera_task_buffer = _connect_camera_task_buffer_descriptor_index;
  resources.connect_camera_result_buffer = _connect_camera_result_buffer_descriptor_index;
  resources.path_capacity = path_capacity;
  resources.max_path_length = max_path_length;
  resources.vertex_capacity = vertex_capacity;
  resources.fixed_max_bounces = (camera_history_bounces << 16u) | light_history_bounces;

  if (upload_or_update_linear_scene_buffer(device, &resources, size_t(1), wavefront_usage, _wavefront_resources_buffer, _wavefront_resources_buffer_size,
        _wavefront_resources_buffer_descriptor_index, "wavefront_resources") == false) {
    return false;
  }

  _wavefront_path_capacity = path_capacity;
  _wavefront_vertex_capacity = vertex_capacity;
  return true;
}

void GPURaytracingRenderer::destroy_blue_noise_buffer(RHIContext& ctx) {
  ETX_PROFILER_SCOPE();

  auto& device = ctx.device();
  destroy_linear_scene_buffer(device, _blue_noise_buffer, _blue_noise_buffer_size, _blue_noise_buffer_descriptor_index);
  _blue_noise_target_samples = 0u;
}

bool GPURaytracingRenderer::update_blue_noise_buffer(RHIContext& ctx, const SceneRepresentation& scene) {
  ETX_PROFILER_SCOPE();

  const bool blue_noise_enabled = scene.data().options.properties[Scene::Properties::BlueNoise];
  if (blue_noise_enabled == false) {
    if (_blue_noise_buffer.valid()) {
      destroy_blue_noise_buffer(ctx);
    } else {
      _blue_noise_target_samples = 0u;
    }
    return true;
  }

  const uint32_t target_samples = normalize_blue_noise_target_samples(scene.data().options.samples);
  const bool needs_upload = (_blue_noise_buffer.valid() == false) || (_blue_noise_target_samples != target_samples);
  if (needs_upload == false) {
    return true;
  }

  std::vector<float> table_data = {};
  if (build_blue_noise_table_data(target_samples, table_data) == false) {
    log::error("GPU RT: failed to build blue noise table");
    return false;
  }

  auto& device = ctx.device();
  const RHIBufferUsage blue_noise_usage = RHIBufferUsage::Storage | RHIBufferUsage::TransferDst;
  const bool upload_success = upload_or_update_linear_scene_buffer(device, table_data.data(), table_data.size(), blue_noise_usage, _blue_noise_buffer, _blue_noise_buffer_size,
    _blue_noise_buffer_descriptor_index, "blue_noise");
  if (upload_success == false) {
    log::error("GPU RT: failed to upload blue noise table");
    return false;
  }

  _blue_noise_target_samples = target_samples;
  return true;
}

void GPURaytracingRenderer::destroy_acceleration_structures(RHIContext& ctx) {
  ETX_PROFILER_SCOPE();
  auto& device = ctx.device();

  if (_tlas.valid()) {
    device.destroy_acceleration_structure(_tlas);
    _tlas = {};
  }

  for (auto blas : _blas) {
    device.destroy_acceleration_structure(blas);
  }
  _blas.clear();

  for (auto buf : _blas_buffers) {
    device.destroy_buffer(buf);
  }
  _blas_buffers.clear();
  _vertex_positions_buffer = {};
  _gpu_scene.vertex_positions = kInvalidDescriptorIndex;
}

void GPURaytracingRenderer::render(RHIContext& ctx, SceneRepresentation& scene, const FrameData& frame_data) {
  ETX_PROFILER_SCOPE();

  Renderer::update_camera(scene, frame_data.dt);

  if (_initialized == false)
    return;

  const uint32_t new_path_mode = static_cast<uint32_t>(gpu_path_mode_from_scene(scene));
  const uint32_t new_material_compile_mask = build_material_compile_mask(scene.data());
  const bool path_mode_changed = (_path_mode != new_path_mode);
  const bool material_compile_mask_changed = (_material_compile_mask != new_material_compile_mask);
  if (path_mode_changed || material_compile_mask_changed || (_pipelines[static_cast<uint32_t>(PipelineStage::PrepareSample)].valid() == false)) {
    _path_mode = new_path_mode;
    _material_compile_mask = new_material_compile_mask;
    create_pipelines(ctx);
  }

  if (_pipelines[static_cast<uint32_t>(PipelineStage::PrepareSample)].valid() == false)
    return;

  auto& device = ctx.device();

  const bool scene_check_requested = consume_scene_update_request();

  SceneHashes new_hashes = _current_scene_hashes;
  UpdateFlags changes = {};
  bool scene_changed = false;
  if (scene_check_requested) {
    ETX_PROFILER_NAMED_SCOPE("gpu_rt_scene_hashes_and_changes");
    new_hashes = scene.data().compute_hashes();
    changes = new_hashes.compare(_current_scene_hashes);
    scene_changed = changes.any();
  }

  const auto& camera = scene.camera();
  const uint64_t new_camera_hash = xxh64(&camera, sizeof(camera));
  const bool camera_changed = (new_camera_hash != _current_camera_hash);
  const bool restart_accumulation = scene_check_requested || path_mode_changed || material_compile_mask_changed || camera_changed;

  const bool geometry_structure_changed = changes[UpdateFlags::AnyGeometryStructure];
  bool needs_full_rebuild = scene_check_requested && geometry_structure_changed;
  const bool needs_scene_data_reupload = scene_check_requested && scene_changed && (geometry_structure_changed == false);
  bool scene_data_update_success = true;

  if (needs_scene_data_reupload && (_vertex_positions_buffer.valid() == false)) {
    needs_full_rebuild = true;
  }

  if (restart_accumulation || scene_changed) {
    _frame_index = 0u;
    _sample_index = 0u;
  }

  if (needs_full_rebuild) {
    ETX_PROFILER_NAMED_SCOPE("gpu_rt_full_rebuild_resources");
    destroy_wavefront_buffers(ctx);
    destroy_scene_buffers(ctx);
    destroy_acceleration_structures(ctx);
  }

  if (_tlas.valid() == false) {
    ETX_PROFILER_NAMED_SCOPE("gpu_rt_build_acceleration_structures");
    scene_data_update_success = build_acceleration_structures(ctx, scene);
    if (scene_data_update_success == false) {
      set_runtime_failure("GPU acceleration-structure build failed");
    }
  } else if (needs_scene_data_reupload) {
    ETX_PROFILER_NAMED_SCOPE("gpu_rt_partial_scene_update");
    const bool update_success = update_scene_data_partial(ctx, scene, changes);
    if (update_success == false) {
      log::error("GPU RT: partial scene update failed");
      scene_data_update_success = false;
    }
  }

  if (_tlas.valid() == false) {
    return;
  }

  if (ensure_wavefront_buffers(ctx, scene) == false) {
    log::error("GPU RT: failed to allocate wavefront buffers");
    return;
  }

  if (scene_changed && (scene_data_update_success == false)) {
    log::error("GPU RT: scene data update failed, skipping frame to retry with current hashes");
    request_scene_update();
    return;
  }

  if (scene_check_requested && scene_data_update_success) {
    _current_scene_hashes = new_hashes;
  }
  _current_camera_hash = new_camera_hash;

  if (path_mode_changed) {
    ETX_PROFILER_NAMED_SCOPE("gpu_rt_runtime_update_scene_options");
    const RHIBufferUsage scene_buffer_usage = RHIBufferUsage::Storage | RHIBufferUsage::TransferDst;
    const GPUSceneOptions options = build_scene_options(scene);
    const bool options_upload_success = upload_or_update_linear_scene_buffer(device, &options, size_t(1), scene_buffer_usage, _scene_options_buffer, _scene_options_buffer_size,
      _gpu_scene.scene_options, "scene_options");
    if (options_upload_success == false) {
      log::error("GPU RT: failed to upload scene options buffer for path mode change");
      return;
    }
  }

  {
    ETX_PROFILER_NAMED_SCOPE("gpu_rt_upload_camera");
    const RHIBufferUsage camera_buffer_usage = RHIBufferUsage::Storage | RHIBufferUsage::TransferDst;
    const bool camera_upload_success =
      upload_or_update_linear_scene_buffer(device, &camera, size_t(1), camera_buffer_usage, _camera_buffer, _camera_buffer_size, _camera_buffer_descriptor_index, "camera");
    if (camera_upload_success == false) {
      log::error("GPU RT: failed to upload camera buffer");
      return;
    }
  }

  if (update_blue_noise_buffer(ctx, scene) == false) {
    log::warning("GPU RT: blue noise buffer is unavailable, falling back to white noise");
  }

  const uint2 full_dim = scene.camera().film_size;
  const bool has_render_window = (_render_window_size.x > 0u) && (_render_window_size.y > 0u);
  const uint2 render_dim = has_render_window ? _render_window_size : full_dim;
  if (_output_dimensions.x != full_dim.x || _output_dimensions.y != full_dim.y) {
    ETX_PROFILER_NAMED_SCOPE("gpu_rt_recreate_output_texture");
    if (_output_texture.valid()) {
      device.destroy_texture(_output_texture);
    }
    RHITextureDesc desc = {};
    desc.width = full_dim.x;
    desc.height = full_dim.y;
    desc.format = RHITextureFormat::R32G32B32A32_FLOAT;
    desc.usage = RHITextureUsage::Storage | RHITextureUsage::Sampled | RHITextureUsage::TransferSrc;
    auto output_texture_result = device.create_texture(desc);
    if ((output_texture_result.result != RHIResult::Success) || (output_texture_result.handle.valid() == false)) {
      log::error("GPU RT: failed to create output texture (%u)", static_cast<uint32_t>(output_texture_result.result));
      return;
    }
    _output_texture = output_texture_result.handle;
    _output_dimensions = full_dim;
    _output_texture_state = RHIResourceState::Undefined;
  }

  if (_sample_index >= std::max(1u, scene.data().options.samples)) {
    return;
  }

  GPURTConstants constants = {
    .camera_buffer_index = _camera_buffer_descriptor_index,
    .as_index = get_bindless_descriptor_index(_tlas),
    .output_image_index = get_bindless_descriptor_index(_output_texture),
    .frame_index = _frame_index,
    .sample_index = _sample_index,
    .blue_noise_buffer_index = _blue_noise_buffer_descriptor_index,
    .wavefront_buffer_index = _wavefront_resources_buffer_descriptor_index,
    .path_iteration = 0u,
    .render_window_origin_x = _render_window_origin.x,
    .render_window_origin_y = _render_window_origin.y,
    .render_window_width = render_dim.x,
    .render_window_height = render_dim.y,
    .scene = _gpu_scene,
  };

  {
    ETX_PROFILER_NAMED_SCOPE("gpu_rt_dispatch_and_submit");
    const auto dispatch_stage = [&](RHICommandBuffer cmd, PipelineStage stage, const RHIDispatchDesc& dispatch, uint32_t path_iteration) {
      GPURTConstants stage_constants = constants;
      stage_constants.path_iteration = path_iteration;
      ctx.cmd_set_pipeline(cmd, _pipelines[static_cast<uint32_t>(stage)]);
      ctx.cmd_push_constants(cmd, &stage_constants, sizeof(stage_constants));
      ctx.cmd_dispatch(cmd, dispatch);
    };

    const auto barrier_wavefront_buffers = [&](RHICommandBuffer cmd) {
      const RHIBindlessHandle buffers[] = {
        _wavefront_resources_buffer,
        _camera_state_buffer,
        _light_state_buffer,
        _camera_hit_buffer,
        _light_hit_buffer,
        _camera_queue_a_buffer,
        _camera_queue_b_buffer,
        _light_queue_a_buffer,
        _light_queue_b_buffer,
        _camera_vertex_buffer,
        _light_vertex_buffer,
        _film_buffer,
        _path_meta_buffer,
        _direct_light_sample_buffer,
        _direct_light_task_buffer,
        _direct_light_result_buffer,
        _connect_light_task_buffer,
        _connect_light_result_buffer,
        _connect_camera_task_buffer,
        _connect_camera_result_buffer,
      };

      for (const auto& buffer : buffers) {
        if (buffer.valid()) {
          ctx.cmd_buffer_barrier(cmd, buffer, RHIResourceState::General, RHIResourceState::General);
        }
      }
    };
    const RHIDispatchDesc film_dispatch = {
      .group_count_x = (render_dim.x + 7u) / 8u,
      .group_count_y = (render_dim.y + 7u) / 8u,
      .group_count_z = 1u,
    };
    const RHIDispatchDesc full_film_dispatch = {
      .group_count_x = (full_dim.x + 7u) / 8u,
      .group_count_y = (full_dim.y + 7u) / 8u,
      .group_count_z = 1u,
    };
    const RHIDispatchDesc scalar_dispatch = {
      .group_count_x = 1u,
      .group_count_y = 1u,
      .group_count_z = 1u,
    };
    const uint32_t scene_max_path_length = std::max(1u, scene.data().options.max_path_length);
    const GPUPathMode path_mode = static_cast<GPUPathMode>(_path_mode);
    const uint32_t max_path_length = scene_max_path_length;
    constexpr uint32_t kQueueCountReadbackSubmitLag = 4u;
    std::vector<RHICommandBuffer> submitted_commands = {};
    submitted_commands.reserve(static_cast<size_t>(kQueueCountReadbackSubmitLag) + 1u);
    const auto record_and_submit = [&](const auto& record_commands) {
      RHICommandBuffer cmd = ctx.get_command_buffer();
      ctx.command_buffer_begin(cmd);
      record_commands(cmd);
      ctx.command_buffer_end(cmd);
      ctx.submit_command_buffer({cmd});
      submitted_commands.push_back(cmd);
    };
    const bool enable_camera_path = path_mode != GPUPathMode::LightTracing;
    const bool enable_light_path = path_mode != GPUPathMode::PathTracing;
    const bool enable_direct_light = enable_camera_path;
    const bool enable_connect_to_camera = enable_light_path;
    const bool has_diffuse = material_compile_mask_has(_material_compile_mask, MaterialClass::Diffuse);
    const bool has_plastic = material_compile_mask_has(_material_compile_mask, MaterialClass::Plastic);
    const bool has_conductor = material_compile_mask_has(_material_compile_mask, MaterialClass::Conductor);
    const bool has_dielectric = material_compile_mask_has(_material_compile_mask, MaterialClass::Dielectric);
    const bool has_thinfilm = material_compile_mask_has(_material_compile_mask, MaterialClass::Thinfilm);
    const uint32_t render_pixel_count = render_dim.x * render_dim.y;
    const bool use_full_frame_light_init = enable_light_path && has_render_window;
    const uint32_t light_init_pixel_count = use_full_frame_light_init ? (full_dim.x * full_dim.y) : render_pixel_count;
    uint32_t current_camera_queue_count = enable_camera_path ? render_pixel_count : 0u;
    uint32_t current_light_queue_count = enable_light_path ? light_init_pixel_count : 0u;
    RHIResourceState camera_queue_count_readback_state = RHIResourceState::Undefined;
    RHIResourceState light_queue_count_readback_state = RHIResourceState::Undefined;
    if (_sample_index == 0u) {
      log::info("GPU path mode: %s", gpu_path_mode_to_string(path_mode));
    }

    record_and_submit([&](RHICommandBuffer cmd) {
      ctx.cmd_texture_barrier(cmd, _output_texture, _output_texture_state, RHIResourceState::General);
      barrier_wavefront_buffers(cmd);
      dispatch_stage(cmd, PipelineStage::PrepareSample, film_dispatch, 0u);
      barrier_wavefront_buffers(cmd);
      if (enable_camera_path) {
        dispatch_stage(cmd, PipelineStage::InitCameraPath0, film_dispatch, 0u);
        barrier_wavefront_buffers(cmd);
      }
      if (enable_light_path) {
        if (use_full_frame_light_init) {
          GPURTConstants light_init_constants = constants;
          light_init_constants.render_window_origin_x = 0u;
          light_init_constants.render_window_origin_y = 0u;
          light_init_constants.render_window_width = full_dim.x;
          light_init_constants.render_window_height = full_dim.y;
          ctx.cmd_set_pipeline(cmd, _pipelines[static_cast<uint32_t>(PipelineStage::InitLightPath0)]);
          ctx.cmd_push_constants(cmd, &light_init_constants, sizeof(light_init_constants));
          ctx.cmd_dispatch(cmd, full_film_dispatch);
        } else {
          dispatch_stage(cmd, PipelineStage::InitLightPath0, film_dispatch, 0u);
        }
        barrier_wavefront_buffers(cmd);
      }
    });

    for (uint32_t path_iteration = 0u; path_iteration < max_path_length;) {
      if ((current_camera_queue_count == 0u) && (current_light_queue_count == 0u)) {
        break;
      }

      const RHIDispatchDesc camera_queue_dispatch = {
        .group_count_x = (current_camera_queue_count + 63u) / 64u,
        .group_count_y = 1u,
        .group_count_z = 1u,
      };
      const RHIDispatchDesc light_queue_dispatch = {
        .group_count_x = (current_light_queue_count + 63u) / 64u,
        .group_count_y = 1u,
        .group_count_z = 1u,
      };
      const uint32_t remaining_iterations = max_path_length - path_iteration;
      const uint32_t submit_batch_size = std::min(kQueueCountReadbackSubmitLag, remaining_iterations);
      const bool needs_next_batch = remaining_iterations > submit_batch_size;

      for (uint32_t batch_iteration_index = 0u; batch_iteration_index < submit_batch_size; ++batch_iteration_index) {
        const uint32_t batch_path_iteration = path_iteration + batch_iteration_index;
        const bool copy_queue_counts = needs_next_batch && ((batch_iteration_index + 1u) == submit_batch_size);
        const RHIBindlessHandle next_camera_queue_buffer = ((batch_path_iteration & 1u) == 0u) ? _camera_queue_b_buffer : _camera_queue_a_buffer;
        const RHIBindlessHandle next_light_queue_buffer = ((batch_path_iteration & 1u) == 0u) ? _light_queue_b_buffer : _light_queue_a_buffer;

        constexpr bool enable_dielectric_direct_light_runtime = false;
        record_and_submit([&](RHICommandBuffer cmd) {
          barrier_wavefront_buffers(cmd);
          if (current_camera_queue_count > 0u) {
            dispatch_stage(cmd, PipelineStage::TraceCamera, camera_queue_dispatch, batch_path_iteration);
            barrier_wavefront_buffers(cmd);
            dispatch_stage(cmd, PipelineStage::CameraSurfaceClassify, camera_queue_dispatch, batch_path_iteration);
            barrier_wavefront_buffers(cmd);
            if (has_diffuse) {
              dispatch_stage(cmd, PipelineStage::CameraContinuePrepareDiffuse, camera_queue_dispatch, batch_path_iteration);
              barrier_wavefront_buffers(cmd);
            }
            if (has_plastic) {
              dispatch_stage(cmd, PipelineStage::CameraContinuePreparePlastic, camera_queue_dispatch, batch_path_iteration);
              barrier_wavefront_buffers(cmd);
            }
            if (has_conductor) {
              dispatch_stage(cmd, PipelineStage::CameraContinuePrepareConductor, camera_queue_dispatch, batch_path_iteration);
              barrier_wavefront_buffers(cmd);
            }
            if (has_dielectric) {
              dispatch_stage(cmd, PipelineStage::CameraContinuePrepareDielectric, camera_queue_dispatch, batch_path_iteration);
              barrier_wavefront_buffers(cmd);
            }
            if (has_thinfilm) {
              dispatch_stage(cmd, PipelineStage::CameraContinuePrepareThinfilm, camera_queue_dispatch, batch_path_iteration);
              barrier_wavefront_buffers(cmd);
            }
            if (enable_direct_light) {
              dispatch_stage(cmd, PipelineStage::CameraDirectLightSample, camera_queue_dispatch, batch_path_iteration);
              barrier_wavefront_buffers(cmd);
              if (has_diffuse) {
                dispatch_stage(cmd, PipelineStage::CameraDirectLightPrepareDiffuse, camera_queue_dispatch, batch_path_iteration);
                barrier_wavefront_buffers(cmd);
              }
              if (has_plastic) {
                dispatch_stage(cmd, PipelineStage::CameraDirectLightPreparePlastic, camera_queue_dispatch, batch_path_iteration);
                barrier_wavefront_buffers(cmd);
              }
              if (has_conductor) {
                dispatch_stage(cmd, PipelineStage::CameraDirectLightPrepareConductor, camera_queue_dispatch, batch_path_iteration);
                barrier_wavefront_buffers(cmd);
              }
              if (enable_dielectric_direct_light_runtime) {
                if (has_dielectric) {
                  dispatch_stage(cmd, PipelineStage::CameraDirectLightPrepareDielectricEval, camera_queue_dispatch, batch_path_iteration);
                  barrier_wavefront_buffers(cmd);
                  dispatch_stage(cmd, PipelineStage::CameraDirectLightPrepareDielectricPdf, camera_queue_dispatch, batch_path_iteration);
                  barrier_wavefront_buffers(cmd);
                }
              }
              dispatch_stage(cmd, PipelineStage::CameraDirectLightShadow, camera_queue_dispatch, batch_path_iteration);
              barrier_wavefront_buffers(cmd);
              dispatch_stage(cmd, PipelineStage::CameraDirectLightAccumulate, camera_queue_dispatch, batch_path_iteration);
              barrier_wavefront_buffers(cmd);
              dispatch_stage(cmd, PipelineStage::CameraDirectHitAccumulate, camera_queue_dispatch, batch_path_iteration);
              barrier_wavefront_buffers(cmd);
            }
            dispatch_stage(cmd, PipelineStage::CameraContinueFinalize, camera_queue_dispatch, batch_path_iteration);
            barrier_wavefront_buffers(cmd);
          }
          if (current_light_queue_count > 0u) {
            dispatch_stage(cmd, PipelineStage::TraceLight, light_queue_dispatch, batch_path_iteration);
            barrier_wavefront_buffers(cmd);
            dispatch_stage(cmd, PipelineStage::LightSurfaceClassify, light_queue_dispatch, batch_path_iteration);
            barrier_wavefront_buffers(cmd);
            if (has_diffuse) {
              dispatch_stage(cmd, PipelineStage::LightContinuePrepareDiffuse, light_queue_dispatch, batch_path_iteration);
              barrier_wavefront_buffers(cmd);
            }
            if (has_plastic) {
              dispatch_stage(cmd, PipelineStage::LightContinuePreparePlastic, light_queue_dispatch, batch_path_iteration);
              barrier_wavefront_buffers(cmd);
            }
            if (has_conductor) {
              dispatch_stage(cmd, PipelineStage::LightContinuePrepareConductor, light_queue_dispatch, batch_path_iteration);
              barrier_wavefront_buffers(cmd);
            }
            if (has_dielectric) {
              dispatch_stage(cmd, PipelineStage::LightContinuePrepareDielectric, light_queue_dispatch, batch_path_iteration);
              barrier_wavefront_buffers(cmd);
            }
            if (has_thinfilm) {
              dispatch_stage(cmd, PipelineStage::LightContinuePrepareThinfilm, light_queue_dispatch, batch_path_iteration);
              barrier_wavefront_buffers(cmd);
            }
            if (enable_connect_to_camera) {
              if (has_diffuse) {
                dispatch_stage(cmd, PipelineStage::LightConnectCameraPrepareDiffuse, light_queue_dispatch, batch_path_iteration);
                barrier_wavefront_buffers(cmd);
              }
              if (has_plastic) {
                dispatch_stage(cmd, PipelineStage::LightConnectCameraPreparePlastic, light_queue_dispatch, batch_path_iteration);
                barrier_wavefront_buffers(cmd);
              }
              if (has_conductor) {
                dispatch_stage(cmd, PipelineStage::LightConnectCameraPrepareConductor, light_queue_dispatch, batch_path_iteration);
                barrier_wavefront_buffers(cmd);
              }
              if (has_dielectric) {
                dispatch_stage(cmd, PipelineStage::LightConnectCameraPrepareDielectric, light_queue_dispatch, batch_path_iteration);
                barrier_wavefront_buffers(cmd);
              }
              dispatch_stage(cmd, PipelineStage::LightConnectCameraShadow, light_queue_dispatch, batch_path_iteration);
              barrier_wavefront_buffers(cmd);
              dispatch_stage(cmd, PipelineStage::LightConnectCameraAccumulate, light_queue_dispatch, batch_path_iteration);
              barrier_wavefront_buffers(cmd);
            }
            dispatch_stage(cmd, PipelineStage::LightContinueFinalize, light_queue_dispatch, batch_path_iteration);
            barrier_wavefront_buffers(cmd);
          }
          dispatch_stage(cmd, PipelineStage::SwapQueues, scalar_dispatch, batch_path_iteration);
          barrier_wavefront_buffers(cmd);

          if (copy_queue_counts) {
            if ((enable_camera_path) && (current_camera_queue_count > 0u)) {
              ctx.cmd_buffer_barrier(cmd, next_camera_queue_buffer, RHIResourceState::General, RHIResourceState::TransferSrc);
              ctx.cmd_buffer_barrier(cmd, _camera_queue_count_readback_buffer, camera_queue_count_readback_state, RHIResourceState::TransferDst);
              ctx.cmd_copy_buffer(cmd, next_camera_queue_buffer, _camera_queue_count_readback_buffer, kGPUWavefrontQueueHeaderSize);
              ctx.cmd_buffer_barrier(cmd, next_camera_queue_buffer, RHIResourceState::TransferSrc, RHIResourceState::General);
              camera_queue_count_readback_state = RHIResourceState::TransferDst;
            }
            if ((enable_light_path) && (current_light_queue_count > 0u)) {
              ctx.cmd_buffer_barrier(cmd, next_light_queue_buffer, RHIResourceState::General, RHIResourceState::TransferSrc);
              ctx.cmd_buffer_barrier(cmd, _light_queue_count_readback_buffer, light_queue_count_readback_state, RHIResourceState::TransferDst);
              ctx.cmd_copy_buffer(cmd, next_light_queue_buffer, _light_queue_count_readback_buffer, kGPUWavefrontQueueHeaderSize);
              ctx.cmd_buffer_barrier(cmd, next_light_queue_buffer, RHIResourceState::TransferSrc, RHIResourceState::General);
              light_queue_count_readback_state = RHIResourceState::TransferDst;
            }
          }
        });
      }

      if (needs_next_batch) {
        const RHIResult batch_wait_result = ctx.wait_idle();
        if (batch_wait_result != RHIResult::Success) {
          log::warning("GPU RT: wait_idle failed after queue-count batch submit (%u)", static_cast<uint32_t>(batch_wait_result));
        }
        for (const auto cmd : submitted_commands) {
          ctx.destroy_command_buffer(cmd);
        }
        submitted_commands.clear();

        if ((enable_camera_path) && (current_camera_queue_count > 0u)) {
          GPUWavefrontQueueHeader queue_header = {};
          const RHIResult read_result = device.read_buffer(_camera_queue_count_readback_buffer, &queue_header, static_cast<uint64_t>(sizeof(queue_header)));
          if (read_result != RHIResult::Success) {
            log::warning("GPU RT: failed to read camera queue count (%u)", static_cast<uint32_t>(read_result));
            current_camera_queue_count = 0u;
          } else {
            current_camera_queue_count = queue_header.count;
          }
        } else {
          current_camera_queue_count = 0u;
        }

        if ((enable_light_path) && (current_light_queue_count > 0u)) {
          GPUWavefrontQueueHeader queue_header = {};
          const RHIResult read_result = device.read_buffer(_light_queue_count_readback_buffer, &queue_header, static_cast<uint64_t>(sizeof(queue_header)));
          if (read_result != RHIResult::Success) {
            log::warning("GPU RT: failed to read light queue count (%u)", static_cast<uint32_t>(read_result));
            current_light_queue_count = 0u;
          } else {
            current_light_queue_count = queue_header.count;
          }
        } else {
          current_light_queue_count = 0u;
        }
      }

      path_iteration += submit_batch_size;
    }

    record_and_submit([&](RHICommandBuffer cmd) {
      barrier_wavefront_buffers(cmd);
      dispatch_stage(cmd, PipelineStage::FinalizeSample, film_dispatch, max_path_length);
      ctx.cmd_texture_barrier(cmd, _output_texture, RHIResourceState::General, RHIResourceState::ShaderReadOnly);
    });

    const RHIResult finalize_wait_result = ctx.wait_idle();
    if (finalize_wait_result != RHIResult::Success) {
      log::warning("GPU RT: wait_idle failed after finalize sample submit (%u)", static_cast<uint32_t>(finalize_wait_result));
    }
    for (const auto cmd : submitted_commands) {
      ctx.destroy_command_buffer(cmd);
    }
    submitted_commands.clear();
  }

  _output_texture_state = RHIResourceState::ShaderReadOnly;
  _frame_index += 1u;
  _sample_index += 1u;
}

void GPURaytracingRenderer::cleanup(RHIContext& ctx) {
  ETX_PROFILER_SCOPE();

  auto& device = ctx.device();
  const RHIResult wait_result = ctx.wait_idle();
  if (wait_result != RHIResult::Success) {
    log::warning("GPU RT: wait_idle failed during cleanup (%u)", static_cast<uint32_t>(wait_result));
  }

  for (auto& pipeline : _pipelines) {
    if (pipeline.valid()) {
      device.destroy_pipeline(pipeline);
      pipeline = {};
    }
  }

  destroy_wavefront_buffers(ctx);
  destroy_scene_buffers(ctx);
  destroy_blue_noise_buffer(ctx);
  destroy_acceleration_structures(ctx);
  destroy_linear_scene_buffer(device, _camera_buffer, _camera_buffer_size, _camera_buffer_descriptor_index);

  if (_output_texture.valid()) {
    device.destroy_texture(_output_texture);
    _output_texture = {};
  }
  _output_texture_state = RHIResourceState::Undefined;

  _initialized = false;
  _current_scene_hashes = {};
  _current_camera_hash = 0;
  _frame_index = 0u;
  _sample_index = 0u;
  _path_mode = 0u;
  _material_compile_mask = 0u;
  _render_window_origin = {};
  _render_window_size = {};
  reset_runtime_failure();
  request_scene_update();
}

void GPURaytracingRenderer::on_camera_changed(SceneRepresentation& scene) {
  ETX_PROFILER_SCOPE();
}

void GPURaytracingRenderer::on_scene_changed(SceneRepresentation& scene) {
  ETX_PROFILER_SCOPE();
  Renderer::on_scene_changed(scene);
}

bool GPURaytracingRenderer::build_acceleration_structures(RHIContext& ctx, SceneRepresentation& scene) {
  ETX_PROFILER_SCOPE();

  auto& device = ctx.device();
  const auto& s = scene.data();

  if (s.vertices.pos.empty() || s.triangles.empty()) {
    log::warning("GPU RT: cannot build acceleration structures for an empty scene");
    return false;
  }

  std::vector<RHIBindlessHandle> new_blas = {};
  std::vector<RHIBindlessHandle> new_blas_buffers = {};
  RHIBindlessHandle new_vertex_positions_buffer = {};
  RHIBindlessHandle new_tlas = {};

  auto cleanup_failed_build = [&]() {
    for (auto as_handle : new_blas) {
      device.destroy_acceleration_structure(as_handle);
    }
    new_blas.clear();

    if (new_tlas.valid()) {
      device.destroy_acceleration_structure(new_tlas);
      new_tlas = {};
    }

    for (auto buffer_handle : new_blas_buffers) {
      device.destroy_buffer(buffer_handle);
    }
    new_blas_buffers.clear();
    new_vertex_positions_buffer = {};
  };

  std::vector<uint32_t> indices;
  RHIBufferDesc vb_desc = {};
  auto vb_res = RHICreateBindlessResult{};
  RHIBufferDesc ib_desc = {};
  auto ib_res = RHICreateBindlessResult{};

  {
    ETX_PROFILER_NAMED_SCOPE("gpu_rt_build_as_upload_geometry_buffers");

    // Vertex Buffer
    vb_desc.size = s.vertices.pos.size() * sizeof(float3);
    vb_desc.usage = RHIBufferUsage::Vertex | RHIBufferUsage::AccelerationStructureBuild | RHIBufferUsage::ShaderDeviceAddress | RHIBufferUsage::TransferDst;
    vb_res = device.create_buffer(vb_desc);
    if ((vb_res.result != RHIResult::Success) || (vb_res.handle.valid() == false)) {
      log::error("GPU RT: failed to create vertex buffer for BLAS (%u)", static_cast<uint32_t>(vb_res.result));
      cleanup_failed_build();
      return false;
    }
    const RHIResult vb_update_result = device.update_buffer(vb_res.handle, s.vertices.pos.data(), vb_desc.size);
    if (vb_update_result != RHIResult::Success) {
      log::error("GPU RT: failed to upload vertex buffer for BLAS (%u)", static_cast<uint32_t>(vb_update_result));
      device.destroy_buffer(vb_res.handle);
      cleanup_failed_build();
      return false;
    }
    new_blas_buffers.push_back(vb_res.handle);
    new_vertex_positions_buffer = vb_res.handle;

    // Index Buffer (Repack from Triangle to uint32 stream)
    indices.reserve(s.triangles.size() * 3);
    for (const auto& tri : s.triangles) {
      indices.push_back(tri.i[0]);
      indices.push_back(tri.i[1]);
      indices.push_back(tri.i[2]);
    }

    ib_desc.size = indices.size() * sizeof(uint32_t);
    ib_desc.usage = RHIBufferUsage::Index | RHIBufferUsage::AccelerationStructureBuild | RHIBufferUsage::ShaderDeviceAddress | RHIBufferUsage::TransferDst;
    ib_res = device.create_buffer(ib_desc);
    if ((ib_res.result != RHIResult::Success) || (ib_res.handle.valid() == false)) {
      log::error("GPU RT: failed to create index buffer for BLAS (%u)", static_cast<uint32_t>(ib_res.result));
      cleanup_failed_build();
      return false;
    }
    const RHIResult ib_update_result = device.update_buffer(ib_res.handle, indices.data(), ib_desc.size);
    if (ib_update_result != RHIResult::Success) {
      log::error("GPU RT: failed to upload index buffer for BLAS (%u)", static_cast<uint32_t>(ib_update_result));
      device.destroy_buffer(ib_res.handle);
      cleanup_failed_build();
      return false;
    }
    new_blas_buffers.push_back(ib_res.handle);
  }

  RHIAccelerationStructureGeometry geometry = {};
  geometry.is_opaque = true;
  geometry.triangles.vertex_buffer = vb_res.handle;
  geometry.triangles.vertex_count = static_cast<uint32_t>(s.vertices.pos.size());
  geometry.triangles.vertex_stride = sizeof(float3);
  geometry.triangles.vertex_format = RHIVertexFormat::Float3;
  geometry.triangles.index_buffer = ib_res.handle;
  geometry.triangles.index_count = static_cast<uint32_t>(indices.size());
  geometry.triangles.index_type = RHIIndexType::UInt32;

  RHIAccelerationStructureDesc blas_desc = {};
  blas_desc.type = RHIAccelerationStructureType::BottomLevel;
  blas_desc.geometry_count = 1;
  blas_desc.geometries = &geometry;

  auto blas_result = device.create_acceleration_structure(blas_desc);
  if ((blas_result.result != RHIResult::Success) || (blas_result.handle.valid() == false)) {
    log::error("GPU RT: failed to create BLAS (%u)", static_cast<uint32_t>(blas_result.result));
    cleanup_failed_build();
    return false;
  }
  new_blas.push_back(blas_result.handle);

  RHIAccelerationStructureBuildDesc build_desc = {};
  build_desc.as_handle = blas_result.handle;
  build_desc.type = RHIAccelerationStructureType::BottomLevel;
  build_desc.geometry_count = 1;
  build_desc.geometries = &geometry;

  // 2. Create TLAS
  // Create Instance Buffer
  RHIAccelerationStructureInstance instance = {};
  instance.transform[0] = 1.0f;
  instance.transform[5] = 1.0f;
  instance.transform[10] = 1.0f;
  instance.instance_custom_index = 0;
  instance.mask = 0xFF;
  instance.instance_shader_binding_table_record_offset = 0;
  instance.flags = 0;  // VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR etc.
  instance.acceleration_structure_reference = device.get_acceleration_structure_device_address(blas_result.handle);

  RHIBufferDesc inst_buf_desc = {};
  inst_buf_desc.size = sizeof(RHIAccelerationStructureInstance);
  inst_buf_desc.usage = RHIBufferUsage::ShaderDeviceAddress | RHIBufferUsage::AccelerationStructureBuild | RHIBufferUsage::TransferDst;
  auto inst_res = device.create_buffer(inst_buf_desc);
  if ((inst_res.result != RHIResult::Success) || (inst_res.handle.valid() == false)) {
    log::error("GPU RT: failed to create TLAS instance buffer (%u)", static_cast<uint32_t>(inst_res.result));
    cleanup_failed_build();
    return false;
  }
  const RHIResult inst_update_result = device.update_buffer(inst_res.handle, &instance, sizeof(instance));
  if (inst_update_result != RHIResult::Success) {
    log::error("GPU RT: failed to upload TLAS instance buffer (%u)", static_cast<uint32_t>(inst_update_result));
    device.destroy_buffer(inst_res.handle);
    cleanup_failed_build();
    return false;
  }
  new_blas_buffers.push_back(inst_res.handle);

  RHIAccelerationStructureDesc tlas_desc = {};
  tlas_desc.type = RHIAccelerationStructureType::TopLevel;
  tlas_desc.instance_count = 1;

  auto tlas_result = device.create_acceleration_structure(tlas_desc);
  if ((tlas_result.result != RHIResult::Success) || (tlas_result.handle.valid() == false)) {
    log::error("GPU RT: failed to create TLAS (%u)", static_cast<uint32_t>(tlas_result.result));
    cleanup_failed_build();
    return false;
  }
  new_tlas = tlas_result.handle;

  const uint64_t blas_scratch_size = device.get_acceleration_structure_build_scratch_size(blas_result.handle);
  if (blas_scratch_size == 0u) {
    log::error("GPU RT: failed to query BLAS scratch size");
    cleanup_failed_build();
    return false;
  }

  const uint64_t tlas_scratch_size = device.get_acceleration_structure_build_scratch_size(new_tlas);
  if (tlas_scratch_size == 0u) {
    log::error("GPU RT: failed to query TLAS scratch size");
    cleanup_failed_build();
    return false;
  }

  const uint64_t scratch_size = (blas_scratch_size > tlas_scratch_size) ? blas_scratch_size : tlas_scratch_size;
  RHIBufferDesc scratch_desc = {};
  scratch_desc.size = scratch_size;
  scratch_desc.usage = RHIBufferUsage::Storage | RHIBufferUsage::ShaderDeviceAddress;
  auto scratch_res = device.create_buffer(scratch_desc);
  if ((scratch_res.result != RHIResult::Success) || (scratch_res.handle.valid() == false)) {
    log::error("GPU RT: failed to create AS scratch buffer (%u)", static_cast<uint32_t>(scratch_res.result));
    cleanup_failed_build();
    return false;
  }
  new_blas_buffers.push_back(scratch_res.handle);

  RHIAccelerationStructureBuildDesc tlas_build_desc = {};
  tlas_build_desc.as_handle = new_tlas;
  tlas_build_desc.type = RHIAccelerationStructureType::TopLevel;
  tlas_build_desc.instance_count = 1;
  tlas_build_desc.instance_buffer = inst_res.handle;

  auto cmd = ctx.get_command_buffer();
  if (cmd.valid() == false) {
    log::error("GPU RT: failed to get command buffer for AS build");
    cleanup_failed_build();
    return false;
  }
  {
    ETX_PROFILER_NAMED_SCOPE("gpu_rt_build_blas");
    ctx.command_buffer_begin(cmd);
    ctx.cmd_build_acceleration_structure(cmd, build_desc, scratch_res.handle, 0);
    ctx.cmd_buffer_barrier(cmd, scratch_res.handle, RHIResourceState::AccelerationStructure, RHIResourceState::AccelerationStructure);
  }

  {
    ETX_PROFILER_NAMED_SCOPE("gpu_rt_build_tlas");
    ctx.cmd_build_acceleration_structure(cmd, tlas_build_desc, scratch_res.handle, 0);
    ctx.command_buffer_end(cmd);
    ctx.submit_command_buffer({cmd});
  }

  const RHIResult as_wait_result = ctx.wait_idle();
  if (as_wait_result != RHIResult::Success) {
    log::error("GPU RT: failed to wait for AS build completion (%u)", static_cast<uint32_t>(as_wait_result));
    ctx.destroy_command_buffer(cmd);
    cleanup_failed_build();
    return false;
  }
  ctx.destroy_command_buffer(cmd);

  _vertex_positions_buffer = new_vertex_positions_buffer;
  _blas = std::move(new_blas);
  _blas_buffers = std::move(new_blas_buffers);
  _tlas = new_tlas;
  _gpu_scene.vertex_positions = get_bindless_descriptor_index(_vertex_positions_buffer);

  ETX_PROFILER_NAMED_SCOPE("gpu_rt_upload_scene_after_as_build");
  const bool upload_success = upload_scene_data(ctx, scene, _vertex_positions_buffer);
  if (upload_success == false) {
    log::error("GPU RT: failed to upload scene data after AS build");
  }
  return upload_success;
}

bool GPURaytracingRenderer::upload_scene_data(RHIContext& ctx, SceneRepresentation& scene, RHIBindlessHandle vertex_positions_buffer) {
  ETX_PROFILER_SCOPE();

  auto& device = ctx.device();
  const auto& data = scene.data();
  _gpu_scene = make_invalid_gpu_scene();

  if (vertex_positions_buffer.valid()) {
    _gpu_scene.vertex_positions = get_bindless_descriptor_index(vertex_positions_buffer);
  }

  const RHIBufferUsage scene_buffer_usage = RHIBufferUsage::Storage | RHIBufferUsage::TransferDst;
  PackedEmitterData packed_emitters = {};
  {
    ETX_PROFILER_NAMED_SCOPE("gpu_rt_build_packed_emitters_full_upload");
    packed_emitters = build_packed_emitters(data);
  }

  bool upload_success = true;
  {
    ETX_PROFILER_NAMED_SCOPE("gpu_rt_upload_vertex_and_geometry_buffers");
    upload_success = upload_or_update_linear_scene_buffer(device, data.vertices.nrm.data(), data.vertices.nrm.size(), scene_buffer_usage, _vertex_normals_buffer,
                       _vertex_normals_buffer_size, _gpu_scene.vertex_normals, "vertex_normals") &&
                     upload_success;
    upload_success = upload_or_update_linear_scene_buffer(device, data.vertices.tan.data(), data.vertices.tan.size(), scene_buffer_usage, _vertex_tangents_buffer,
                       _vertex_tangents_buffer_size, _gpu_scene.vertex_tangents, "vertex_tangents") &&
                     upload_success;
    upload_success = upload_or_update_linear_scene_buffer(device, data.vertices.btn.data(), data.vertices.btn.size(), scene_buffer_usage, _vertex_bitangents_buffer,
                       _vertex_bitangents_buffer_size, _gpu_scene.vertex_bitangents, "vertex_bitangents") &&
                     upload_success;
    upload_success = upload_or_update_linear_scene_buffer(device, data.vertices.tex.data(), data.vertices.tex.size(), scene_buffer_usage, _vertex_texcoords_buffer,
                       _vertex_texcoords_buffer_size, _gpu_scene.vertex_texcoords, "vertex_texcoords") &&
                     upload_success;
    upload_success = upload_or_update_linear_scene_buffer(device, packed_emitters.triangles.data(), packed_emitters.triangles.size(), scene_buffer_usage, _triangles_buffer,
                       _triangles_buffer_size, _gpu_scene.triangles, "triangles") &&
                     upload_success;
    upload_success =
      upload_or_update_linear_scene_buffer(device, data.meshes.data(), data.meshes.size(), scene_buffer_usage, _meshes_buffer, _meshes_buffer_size, _gpu_scene.meshes, "meshes") &&
      upload_success;
  }

  {
    ETX_PROFILER_NAMED_SCOPE("gpu_rt_upload_material_and_emitter_buffers");
    upload_success = upload_or_update_linear_scene_buffer(device, packed_emitters.emitter_profiles.data(), packed_emitters.emitter_profiles.size(), scene_buffer_usage,
                       _emitter_profiles_buffer, _emitter_profiles_buffer_size, _gpu_scene.emitter_profiles, "emitter_profiles") &&
                     upload_success;
    upload_success = upload_or_update_linear_scene_buffer(device, packed_emitters.emitter_instances.data(), packed_emitters.emitter_instances.size(), scene_buffer_usage,
                       _emitter_instances_buffer, _emitter_instances_buffer_size, _gpu_scene.emitter_instances, "emitter_instances") &&
                     upload_success;
    upload_success = upload_or_update_linear_scene_buffer(device, data.materials.data(), data.materials.size(), scene_buffer_usage, _materials_buffer, _materials_buffer_size,
                       _gpu_scene.materials, "materials") &&
                     upload_success;
    upload_success = upload_or_update_linear_scene_buffer(device, data.spectrum_values.data(), data.spectrum_values.size(), scene_buffer_usage, _spectrums_buffer,
                       _spectrums_buffer_size, _gpu_scene.spectrums, "spectrums") &&
                     upload_success;
  }

  {
    ETX_PROFILER_NAMED_SCOPE("gpu_rt_upload_images");
    auto images_blob = build_packed_images_blob(data);
    if (images_blob.success) {
      const bool image_upload_success = device.upload_or_update_chunked_buffer(images_blob, scene_buffer_usage, _images_blob_state, "images_blob");
      if (image_upload_success) {
        _gpu_scene.images = _images_blob_state.metadata_descriptor_index;
      }
      upload_success = image_upload_success && upload_success;
    } else {
      log::error("GPU RT: failed to build packed image blob");
      upload_success = false;
    }
  }

  {
    ETX_PROFILER_NAMED_SCOPE("gpu_rt_upload_mediums");
    auto mediums_blob = build_packed_mediums_blob(data);
    if (mediums_blob.success) {
      const bool medium_upload_success = device.upload_or_update_chunked_buffer(mediums_blob, scene_buffer_usage, _mediums_blob_state, "mediums_blob");
      if (medium_upload_success) {
        _gpu_scene.mediums = _mediums_blob_state.metadata_descriptor_index;
      }
      upload_success = medium_upload_success && upload_success;
    } else {
      log::error("GPU RT: failed to build packed medium blob");
      upload_success = false;
    }
  }

  {
    ETX_PROFILER_NAMED_SCOPE("gpu_rt_upload_scene_globals");
    GPUSceneGlobals globals = build_scene_globals(data, packed_emitters);
    upload_success = upload_or_update_linear_scene_buffer(device, &globals, size_t(1), scene_buffer_usage, _scene_globals_buffer, _scene_globals_buffer_size,
                       _gpu_scene.scene_globals, "scene_globals") &&
                     upload_success;
  }

  {
    ETX_PROFILER_NAMED_SCOPE("gpu_rt_upload_scene_options");
    GPUSceneOptions options = build_scene_options(scene);
    upload_success = upload_or_update_linear_scene_buffer(device, &options, size_t(1), scene_buffer_usage, _scene_options_buffer, _scene_options_buffer_size,
                       _gpu_scene.scene_options, "scene_options") &&
                     upload_success;
  }

  {
    ETX_PROFILER_NAMED_SCOPE("gpu_rt_upload_emitters_distribution");
    auto emitters_distribution = build_packed_emitter_distribution(packed_emitters);
    upload_success = upload_or_update_linear_scene_buffer(device, emitters_distribution.data(), emitters_distribution.size(), scene_buffer_usage, _emitters_distribution_buffer,
                       _emitters_distribution_buffer_size, _gpu_scene.emitters_distribution, "emitters_distribution") &&
                     upload_success;
  }

  return upload_success;
}

bool GPURaytracingRenderer::update_scene_data_partial(RHIContext& ctx, SceneRepresentation& scene, const UpdateFlags& changes) {
  ETX_PROFILER_SCOPE();

  auto& device = ctx.device();
  const auto& data = scene.data();
  const RHIBufferUsage scene_buffer_usage = RHIBufferUsage::Storage | RHIBufferUsage::TransferDst;
  bool upload_success = true;

  {
    ETX_PROFILER_NAMED_SCOPE("gpu_rt_partial_direct_buffer_updates");
    if (changes[UpdateFlags::VerticesNrm]) {
      upload_success = upload_or_update_linear_scene_buffer(device, data.vertices.nrm.data(), data.vertices.nrm.size(), scene_buffer_usage, _vertex_normals_buffer,
                         _vertex_normals_buffer_size, _gpu_scene.vertex_normals, "vertex_normals") &&
                       upload_success;
    }
    if (changes[UpdateFlags::VerticesTan]) {
      upload_success = upload_or_update_linear_scene_buffer(device, data.vertices.tan.data(), data.vertices.tan.size(), scene_buffer_usage, _vertex_tangents_buffer,
                         _vertex_tangents_buffer_size, _gpu_scene.vertex_tangents, "vertex_tangents") &&
                       upload_success;
    }
    if (changes[UpdateFlags::VerticesBtn]) {
      upload_success = upload_or_update_linear_scene_buffer(device, data.vertices.btn.data(), data.vertices.btn.size(), scene_buffer_usage, _vertex_bitangents_buffer,
                         _vertex_bitangents_buffer_size, _gpu_scene.vertex_bitangents, "vertex_bitangents") &&
                       upload_success;
    }
    if (changes[UpdateFlags::VerticesTex]) {
      upload_success = upload_or_update_linear_scene_buffer(device, data.vertices.tex.data(), data.vertices.tex.size(), scene_buffer_usage, _vertex_texcoords_buffer,
                         _vertex_texcoords_buffer_size, _gpu_scene.vertex_texcoords, "vertex_texcoords") &&
                       upload_success;
    }
    if (changes[UpdateFlags::Meshes]) {
      upload_success = upload_or_update_linear_scene_buffer(device, data.meshes.data(), data.meshes.size(), scene_buffer_usage, _meshes_buffer, _meshes_buffer_size,
                         _gpu_scene.meshes, "meshes") &&
                       upload_success;
    }
    if (changes[UpdateFlags::Materials]) {
      upload_success = upload_or_update_linear_scene_buffer(device, data.materials.data(), data.materials.size(), scene_buffer_usage, _materials_buffer, _materials_buffer_size,
                         _gpu_scene.materials, "materials") &&
                       upload_success;
    }
    if (changes[UpdateFlags::Spectra]) {
      upload_success = upload_or_update_linear_scene_buffer(device, data.spectrum_values.data(), data.spectrum_values.size(), scene_buffer_usage, _spectrums_buffer,
                         _spectrums_buffer_size, _gpu_scene.spectrums, "spectrums") &&
                       upload_success;
    }
    if (changes[UpdateFlags::Images]) {
      auto images_blob = build_packed_images_blob(data);
      if (images_blob.success) {
        const bool image_upload_success = device.upload_or_update_chunked_buffer(images_blob, scene_buffer_usage, _images_blob_state, "images_blob");
        if (image_upload_success) {
          _gpu_scene.images = _images_blob_state.metadata_descriptor_index;
        }
        upload_success = image_upload_success && upload_success;
      } else {
        log::error("GPU RT: failed to build packed image blob");
        upload_success = false;
      }
    }
    if (changes[UpdateFlags::Mediums]) {
      auto mediums_blob = build_packed_mediums_blob(data);
      if (mediums_blob.success) {
        const bool medium_upload_success = device.upload_or_update_chunked_buffer(mediums_blob, scene_buffer_usage, _mediums_blob_state, "mediums_blob");
        if (medium_upload_success) {
          _gpu_scene.mediums = _mediums_blob_state.metadata_descriptor_index;
        }
        upload_success = medium_upload_success && upload_success;
      } else {
        log::error("GPU RT: failed to build packed medium blob");
        upload_success = false;
      }
    }
  }

  const bool packed_emitters_changed = changes[UpdateFlags::Triangles] || changes[UpdateFlags::Emitters] || changes[UpdateFlags::Materials] || changes[UpdateFlags::Spectra];
  const bool scene_globals_changed = changes[UpdateFlags::Triangles] || changes[UpdateFlags::Meshes] || changes[UpdateFlags::Emitters] || changes[UpdateFlags::Defaults] ||
                                     changes[UpdateFlags::Images] || changes[UpdateFlags::PixelFilter];

  PackedEmitterData packed_emitters = {};
  if (packed_emitters_changed || scene_globals_changed) {
    ETX_PROFILER_NAMED_SCOPE("gpu_rt_partial_build_packed_emitters");
    packed_emitters = build_packed_emitters(data);
  }

  {
    ETX_PROFILER_NAMED_SCOPE("gpu_rt_partial_dependent_buffer_updates");
    if (changes[UpdateFlags::Triangles] || changes[UpdateFlags::Emitters]) {
      upload_success = upload_or_update_linear_scene_buffer(device, packed_emitters.triangles.data(), packed_emitters.triangles.size(), scene_buffer_usage, _triangles_buffer,
                         _triangles_buffer_size, _gpu_scene.triangles, "triangles") &&
                       upload_success;
    }

    if (packed_emitters_changed) {
      upload_success = upload_or_update_linear_scene_buffer(device, packed_emitters.emitter_profiles.data(), packed_emitters.emitter_profiles.size(), scene_buffer_usage,
                         _emitter_profiles_buffer, _emitter_profiles_buffer_size, _gpu_scene.emitter_profiles, "emitter_profiles") &&
                       upload_success;

      upload_success = upload_or_update_linear_scene_buffer(device, packed_emitters.emitter_instances.data(), packed_emitters.emitter_instances.size(), scene_buffer_usage,
                         _emitter_instances_buffer, _emitter_instances_buffer_size, _gpu_scene.emitter_instances, "emitter_instances") &&
                       upload_success;

      auto emitters_distribution = build_packed_emitter_distribution(packed_emitters);
      upload_success = upload_or_update_linear_scene_buffer(device, emitters_distribution.data(), emitters_distribution.size(), scene_buffer_usage, _emitters_distribution_buffer,
                         _emitters_distribution_buffer_size, _gpu_scene.emitters_distribution, "emitters_distribution") &&
                       upload_success;
    }
  }

  if (scene_globals_changed) {
    ETX_PROFILER_NAMED_SCOPE("gpu_rt_partial_update_scene_globals");
    GPUSceneGlobals globals = build_scene_globals(data, packed_emitters);
    upload_success = upload_or_update_linear_scene_buffer(device, &globals, size_t(1), scene_buffer_usage, _scene_globals_buffer, _scene_globals_buffer_size,
                       _gpu_scene.scene_globals, "scene_globals") &&
                     upload_success;
  }

  if (changes[UpdateFlags::Options]) {
    ETX_PROFILER_NAMED_SCOPE("gpu_rt_partial_update_scene_options");
    GPUSceneOptions options = build_scene_options(scene);
    upload_success = upload_or_update_linear_scene_buffer(device, &options, size_t(1), scene_buffer_usage, _scene_options_buffer, _scene_options_buffer_size,
                       _gpu_scene.scene_options, "scene_options") &&
                     upload_success;
  }

  return upload_success;
}

}  // namespace etx
