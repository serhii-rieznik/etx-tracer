#include "gpu_renderer.hxx"
#include <interop/gpu_abi_constants.hxx>
#include <interop/gpu_rt_shared.hxx>
#include <interop/sampler_policy.hxx>
#include <etx/core/profiler.hxx>
#include <etx/rhi/rhi.hxx>
#include <etx/rhi/shader/shader_compiler.hxx>
#include <etx/render/host/gpu_asset_descriptor.hxx>
#include <etx/render/host/emitter_packing.hxx>
#include <etx/render/host/scene_representation.hxx>
#include <etx/render/shared/density_grid.hxx>
#include <bluenoise.hxx>
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

bool build_blue_noise_table_data(uint32_t target_samples, std::vector<uint8_t>& table_data) {
  target_samples = normalize_blue_noise_target_samples(target_samples);

  const uint64_t table_size_u64 = static_cast<uint64_t>(kBlueNoiseTileSize) * static_cast<uint64_t>(kBlueNoiseTileSize) * static_cast<uint64_t>(kBlueNoiseSampleCount) *
                                  static_cast<uint64_t>(kBlueNoiseDimensionCount);
  if (table_size_u64 > std::numeric_limits<size_t>::max()) {
    return false;
  }

  const size_t table_size = static_cast<size_t>(table_size_u64);
  table_data.assign(table_size, uint8_t(0u));

  for (uint32_t sample_index = 0u; sample_index < kBlueNoiseSampleCount; ++sample_index) {
    for (uint32_t y = 0u; y < kBlueNoiseTileSize; ++y) {
      for (uint32_t x = 0u; x < kBlueNoiseTileSize; ++x) {
        const ::BNSampler sampler = ::BNSampler(x, y, target_samples, sample_index);
        for (uint32_t dimension = 0u; dimension < kBlueNoiseDimensionCount; ++dimension) {
          const float sample_value = sampler.get(dimension);
          uint32_t quantized = static_cast<uint32_t>(sample_value * static_cast<float>(kBlueNoiseSampleCount));
          if (quantized > 255u) {
            quantized = 255u;
          }
          const uint32_t index = blue_noise_table_index(x, y, sample_index, dimension);
          table_data[index] = static_cast<uint8_t>(quantized);
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

  globals.environment_emitter_count = packed_emitters.environment_emitters.count;
  for (uint32_t i = 0u; i < packed_emitters.environment_emitters.count; ++i) {
    globals.environment_emitters[i] = packed_emitters.environment_emitters.emitters[i];
  }

  return globals;
}

GPUSceneOptions build_scene_options(const SceneData& scene_data) {
  GPUSceneOptions options = {};
  options.min_path_length = scene_data.options.min_path_length;
  options.max_path_length = scene_data.options.max_path_length;
  options.samples = scene_data.options.samples;
  options.random_path_termination = scene_data.options.random_path_termination;
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
  return options;
}
}  // namespace

GPURaytracingRenderer::GPURaytracingRenderer(TaskScheduler& s)
  : Renderer(s) {
  _gpu_scene = make_invalid_gpu_scene();
}

GPURaytracingRenderer::~GPURaytracingRenderer() {
}

void GPURaytracingRenderer::init(RHIContext& ctx, SceneRepresentation& scene) {
  ETX_PROFILER_SCOPE();

  Renderer::init(ctx, scene);

  create_pipelines(ctx);
  _initialized = true;
}

void GPURaytracingRenderer::create_pipelines(RHIContext& ctx) {
  ETX_PROFILER_SCOPE();

  auto& device = ctx.device();

  if (_pipeline.valid()) {
    device.destroy_pipeline(_pipeline);
    _pipeline = {};
  }

  auto& compiler = ShaderCompiler::instance();

  ShaderCompiler::MultiShaderCompilationResult result = {};
  {
    ETX_PROFILER_NAMED_SCOPE("gpu_rt_compile_compute_shader");
    result = compiler.compile("shaders/gpu_rt.hlsl", {{"compute_main", RHIShaderStage::Compute}});
  }

  if (result.result != RHIResult::Success) {
    log::error("Failed to compile GPU RT shader: %s", result.error_message.c_str());
    return;
  }

  RHIComputePipelineDesc desc = device.make_compute_pipeline_desc(result.binaries[0]);
  _pipeline = device.create_compute_pipeline(desc).handle;
}

void GPURaytracingRenderer::reload_shaders(RHIContext& ctx) {
  ETX_PROFILER_SCOPE();
  create_pipelines(ctx);
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

  std::vector<uint8_t> table_data = {};
  if (build_blue_noise_table_data(target_samples, table_data) == false) {
    log::error("GPU RT: failed to build blue noise table");
    return false;
  }

  auto& device = ctx.device();
  const RHIBufferUsage blue_noise_usage = RHIBufferUsage::Storage | RHIBufferUsage::TransferDst;
  const bool upload_success = upload_or_update_linear_scene_buffer(device, table_data.data(), table_data.size(), blue_noise_usage, _blue_noise_buffer,
    _blue_noise_buffer_size, _blue_noise_buffer_descriptor_index, "blue_noise");
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

  if ((_initialized == false) || (_pipeline.valid() == false))
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

  const bool geometry_structure_changed = changes[UpdateFlags::AnyGeometryStructure];
  bool needs_full_rebuild = scene_check_requested && geometry_structure_changed;
  const bool needs_scene_data_reupload = scene_check_requested && scene_changed && (geometry_structure_changed == false);
  bool scene_data_update_success = true;

  if (needs_scene_data_reupload && (_vertex_positions_buffer.valid() == false)) {
    needs_full_rebuild = true;
  }

  if (scene_changed || camera_changed) {
    _frame_index = 0u;
    _sample_index = 0u;
  }

  if (needs_full_rebuild) {
    ETX_PROFILER_NAMED_SCOPE("gpu_rt_full_rebuild_resources");
    destroy_scene_buffers(ctx);
    destroy_acceleration_structures(ctx);
  }

  if (_tlas.valid() == false) {
    ETX_PROFILER_NAMED_SCOPE("gpu_rt_build_acceleration_structures");
    scene_data_update_success = build_acceleration_structures(ctx, scene);
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

  if (scene_changed && (scene_data_update_success == false)) {
    log::error("GPU RT: scene data update failed, skipping frame to retry with current hashes");
    request_scene_update();
    return;
  }

  if (scene_check_requested && scene_data_update_success) {
    _current_scene_hashes = new_hashes;
  }
  _current_camera_hash = new_camera_hash;

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

  uint2 current_dim = scene.camera().film_size;
  if (_output_dimensions.x != current_dim.x || _output_dimensions.y != current_dim.y) {
    ETX_PROFILER_NAMED_SCOPE("gpu_rt_recreate_output_texture");
    if (_output_texture.valid()) {
      device.destroy_texture(_output_texture);
    }
    RHITextureDesc desc = {};
    desc.width = current_dim.x;
    desc.height = current_dim.y;
    desc.format = RHITextureFormat::R32G32B32A32_FLOAT;
    desc.usage = RHITextureUsage::Storage | RHITextureUsage::Sampled | RHITextureUsage::TransferSrc;
    auto output_texture_result = device.create_texture(desc);
    if ((output_texture_result.result != RHIResult::Success) || (output_texture_result.handle.valid() == false)) {
      log::error("GPU RT: failed to create output texture (%u)", static_cast<uint32_t>(output_texture_result.result));
      return;
    }
    _output_texture = output_texture_result.handle;
    _output_dimensions = current_dim;
  }

  if (_sample_index > 0u) {
    return;
  }

  GPURTConstants constants = {
    .camera_buffer_index = _camera_buffer_descriptor_index,
    .as_index = get_bindless_descriptor_index(_tlas),
    .output_image_index = get_bindless_descriptor_index(_output_texture),
    .frame_index = _frame_index,
    .sample_index = _sample_index,
    .blue_noise_buffer_index = _blue_noise_buffer_descriptor_index,
    .scene = _gpu_scene,
  };

  {
    ETX_PROFILER_NAMED_SCOPE("gpu_rt_dispatch_and_submit");
    auto cmd = ctx.get_command_buffer();
    ctx.command_buffer_begin(cmd);
    ctx.cmd_texture_barrier(cmd, _output_texture, RHIResourceState::Undefined, RHIResourceState::General);
    ctx.cmd_set_pipeline(cmd, _pipeline);
    ctx.cmd_push_constants(cmd, &constants, sizeof(constants));
    ctx.cmd_dispatch(cmd, {(current_dim.x + 7u) / 8u, (current_dim.y + 7u) / 8u, 1u});
    ctx.cmd_texture_barrier(cmd, _output_texture, RHIResourceState::General, RHIResourceState::ShaderReadOnly);
    ctx.command_buffer_end(cmd);
    ctx.submit_command_buffer({cmd});
  }

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

  if (_pipeline.value != 0) {
    device.destroy_pipeline(_pipeline);
    _pipeline = {};
  }

  destroy_scene_buffers(ctx);
  destroy_blue_noise_buffer(ctx);
  destroy_acceleration_structures(ctx);
  destroy_linear_scene_buffer(device, _camera_buffer, _camera_buffer_size, _camera_buffer_descriptor_index);

  if (_output_texture.valid()) {
    device.destroy_texture(_output_texture);
    _output_texture = {};
  }

  _initialized = false;
  _current_scene_hashes = {};
  _current_camera_hash = 0;
  _frame_index = 0u;
  _sample_index = 0u;
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
    GPUSceneOptions options = build_scene_options(data);
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
  const bool scene_globals_changed = changes[UpdateFlags::Triangles] || changes[UpdateFlags::Meshes] || changes[UpdateFlags::Emitters] || changes[UpdateFlags::Defaults];

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
    GPUSceneOptions options = build_scene_options(data);
    upload_success = upload_or_update_linear_scene_buffer(device, &options, size_t(1), scene_buffer_usage, _scene_options_buffer, _scene_options_buffer_size,
                       _gpu_scene.scene_options, "scene_options") &&
                     upload_success;
  }

  return upload_success;
}

}  // namespace etx
