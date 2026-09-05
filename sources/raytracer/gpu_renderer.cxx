#include "gpu_renderer.hxx"
#include "shader_packager.hxx"
#include <interop/gpu_abi_constants.hxx>
#include <interop/gpu_rt_shared.hxx>
#include <interop/gpu_upbp_abi.hxx>
#include <interop/gpu_wavefront_abi.hxx>
#include <interop/material.hxx>
#include <interop/sampler_policy.hxx>
#include <etx/core/profiler.hxx>
#include <etx/core/environment.hxx>
#include <etx/rhi/rhi.hxx>
#include <etx/rhi/shader/shader_compiler.hxx>
#include <etx/rhi/shader/shader_package.hxx>
#include <etx/render/host/gpu_asset_descriptor.hxx>
#include <etx/render/host/emitter_packing.hxx>
#include <etx/render/host/scene_representation.hxx>
#include <etx/render/host/tasks.hxx>
#include <etx/render/shared/density_grid.hxx>
#include <etx/rt/integrators/upbp_iteration.hxx>
#include <etx/rt/integrators/upbp_options.hxx>
#include <etx/rt/shared/bdpt_mode.hxx>
#include <etx/rt/shared/vcm_radius.hxx>
#include <bluenoise.hxx>
#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iterator>
#include <limits>
#include <map>
#include <mutex>
#include <set>
#include <thread>
#include <type_traits>
#include "gpu_renderer_abi_static_asserts.hxx"

namespace etx {

namespace {
constexpr uint32_t kInvalidDescriptorIndex = ~0u;

double elapsed_ms(const std::chrono::steady_clock::time_point& begin, const std::chrono::steady_clock::time_point& end) {
  return std::chrono::duration<double, std::milli>(end - begin).count();
}

constexpr uint32_t kBlueNoiseTileSize = kSamplerBlueNoiseTileSize;
constexpr uint32_t kBlueNoiseSampleCount = kSamplerBlueNoiseSampleCount;
constexpr uint32_t kBlueNoiseDimensionCount = kSamplerBlueNoiseDimensionCount;
constexpr uint32_t kWavefrontRollingHistoryBounces = 1u;
constexpr uint32_t kWavefrontLightHistoryBounces = 3u;
constexpr uint32_t kWavefrontFastLightHistoryBounces = 1u;
constexpr uint64_t kWavefrontMaxAddressableBufferSize = static_cast<uint64_t>(std::numeric_limits<uint32_t>::max());
constexpr uint32_t kWavefrontInitialLightHistoryBounces = 16u;
constexpr uint32_t kWavefrontLightHistoryShrinkSampleCount = 8u;
constexpr uint32_t kWavefrontAutoMaximumSteps = 1024u;
constexpr uint32_t kWavefrontAutoAdjustmentDivisor = 4u;
constexpr double kWavefrontFinalAutoTargetMs = 12.0;
constexpr double kWavefrontFinalAutoLowerDeadZoneMs = 8.0;
constexpr double kWavefrontFinalAutoUpperDeadZoneMs = 16.0;
constexpr double kWavefrontPreviewAutoTargetMs = 4.0;
constexpr double kWavefrontPreviewAutoLowerDeadZoneMs = 2.0;
constexpr double kWavefrontPreviewAutoUpperDeadZoneMs = 6.0;
constexpr double kWavefrontAutoSmoothingFactor = 0.25;
constexpr uint64_t kWavefrontLightVertexCounterSize = sizeof(uint32_t);
constexpr uint32_t kWavefrontCoarseQueueReadbackInterval = 16u;
constexpr uint32_t kNonVulkanPipelinePublishBatchSize = 2u;
constexpr uint32_t kVulkanPipelineMaxWorkerCount = 6u;
constexpr uint64_t kVulkanPipelineWorkerMemoryReserve = 4ull * 1024ull * 1024ull * 1024ull;
constexpr uint64_t kWavefrontFallbackMemoryBudget = 512ull * 1024ull * 1024ull;
constexpr uint32_t kUPBPInitialTrackingEventsPerInterval = 2u;
constexpr uint32_t kUPBPDensityQueryDispatchChunkSize = 65535u;
constexpr uint32_t kDensityQueryGroupDispatchChunkSize = 4096u;
constexpr uint32_t kUPBPDensityLinearDispatchChunkSize = kUPBPDensityQueryDispatchChunkSize * 64u;
constexpr uint32_t kUPBPBP2DTargetPartitionInstances = 16384u;
constexpr uint32_t kUPBPBeamGridMaximumResolution = 32u;
constexpr uint32_t kUPBPBeamGridMaximumShardCount = 4096u;
constexpr uint64_t kUPBPBeamGridScratchBudget = 256ull * 1024ull * 1024ull;
constexpr uint32_t kUPBPDensityTechniqueMask = static_cast<uint32_t>(UPBPTechnique::Surface) | static_cast<uint32_t>(UPBPTechnique::PP3D) |
                                               static_cast<uint32_t>(UPBPTechnique::PB2D) | static_cast<uint32_t>(UPBPTechnique::BP2D) | static_cast<uint32_t>(UPBPTechnique::BB1D);
constexpr uint64_t kWavefrontMemoryReserveDivisor = 8ull;
constexpr uint32_t kMaterialCompileConnectibleConductor = 1u << 30u;
constexpr uint32_t kMaterialCompileConnectibleDielectric = 1u << 31u;

static_assert(kGPUWavefrontDirectLightTaskStride == kGPUWavefrontConnectCameraTaskStride);
static_assert(kGPUWavefrontDirectLightResultStride == kGPUWavefrontConnectCameraResultStride);
static_assert(kMaximumPathLength <= kGPUWavefrontLightPathVertexPathLengthMask);
static_assert((GPUWavefrontVertexFlags::Subsurface & ~kGPUWavefrontLightPathVertexFlagsValueMask) == 0u);
static_assert((GPUWavefrontSubsurfaceFlags::InlineMedium & ~kGPUWavefrontLightPathVertexInlineMediumFlagsValueMask) == 0u);

uint32_t divide_round_up(uint32_t value, uint32_t divisor);

uint32_t upbp_partition_size(uint32_t total_count, uint32_t partition_index, uint32_t partition_count) {
  const uint32_t base_count = total_count / partition_count;
  return base_count + static_cast<uint32_t>(partition_index < (total_count % partition_count));
}

uint32_t upbp_partition_offset(uint32_t total_count, uint32_t partition_index, uint32_t partition_count) {
  const uint32_t base_count = total_count / partition_count;
  return partition_index * base_count + std::min(partition_index, total_count % partition_count);
}

uint64_t upbp_initial_resident_storage_bytes(uint32_t resident_path_capacity, uint32_t maximum_path_length, uint32_t maximum_boundary_count, uint32_t technique_mask,
  uint32_t maximum_bb1d_light_path_count) {
  (void)maximum_bb1d_light_path_count;
  const uint64_t path_count = resident_path_capacity;
  const uint64_t vertex_count = path_count * (static_cast<uint64_t>(maximum_path_length) + 1u);
  const uint64_t segment_count = path_count * maximum_path_length;
  const uint64_t interval_count = path_count * (static_cast<uint64_t>(maximum_path_length) + maximum_boundary_count);
  const bool collect_points =
    (technique_mask & (static_cast<uint32_t>(UPBPTechnique::Surface) | static_cast<uint32_t>(UPBPTechnique::PP3D) | static_cast<uint32_t>(UPBPTechnique::PB2D))) != 0u;
  const bool collect_beams = (technique_mask & (static_cast<uint32_t>(UPBPTechnique::BP2D) | static_cast<uint32_t>(UPBPTechnique::BB1D))) != 0u;
  const bool track_camera_events = (technique_mask & (static_cast<uint32_t>(UPBPTechnique::PB2D) | static_cast<uint32_t>(UPBPTechnique::BB1D))) != 0u;
  const uint64_t light_event_count = collect_beams ? interval_count * kUPBPInitialTrackingEventsPerInterval : 0u;
  const uint64_t camera_event_count = track_camera_events ? interval_count * kUPBPInitialTrackingEventsPerInterval : 0u;
  const uint64_t event_count = std::max(light_event_count, camera_event_count);
  const uint64_t point_count = collect_points ? segment_count : 0u;
  const uint64_t beam_count = collect_beams ? segment_count : 0u;
  const uint64_t counter_bytes = static_cast<uint64_t>(GPUUPBPCounterIndex::Count) * sizeof(uint32_t);

  return vertex_count * (kGPUUPBPVertexStride + kGPUUPBPBPTVertexStride) + segment_count * kGPUUPBPSegmentStride + interval_count * kGPUUPBPIntervalStride +
         std::max<uint64_t>(1u, event_count) * kGPUUPBPTrackingEventStride + std::max<uint64_t>(1u, point_count) * kGPUUPBPPointStride +
         std::max<uint64_t>(1u, beam_count) * kGPUUPBPBeamStride + path_count * (kGPUUPBPPathStateStride + kGPUUPBPBPTPathStateStride) + kGPUUPBPResourcesStride +
         2u * counter_bytes;
}

bool upbp_resident_storage_addressable(uint32_t resident_path_capacity, uint32_t maximum_path_length, uint32_t maximum_boundary_count, uint32_t technique_mask,
  uint32_t maximum_bb1d_light_path_count) {
  (void)maximum_bb1d_light_path_count;
  const uint64_t path_count = resident_path_capacity;
  const uint64_t vertex_count = path_count * (static_cast<uint64_t>(maximum_path_length) + 1u);
  const uint64_t segment_count = path_count * maximum_path_length;
  const uint64_t interval_count = path_count * (static_cast<uint64_t>(maximum_path_length) + maximum_boundary_count);
  const bool collect_points =
    (technique_mask & (static_cast<uint32_t>(UPBPTechnique::Surface) | static_cast<uint32_t>(UPBPTechnique::PP3D) | static_cast<uint32_t>(UPBPTechnique::PB2D))) != 0u;
  const bool collect_beams = (technique_mask & (static_cast<uint32_t>(UPBPTechnique::BP2D) | static_cast<uint32_t>(UPBPTechnique::BB1D))) != 0u;
  const bool track_camera_events = (technique_mask & (static_cast<uint32_t>(UPBPTechnique::PB2D) | static_cast<uint32_t>(UPBPTechnique::BB1D))) != 0u;
  const uint64_t light_event_count = collect_beams ? interval_count * kUPBPInitialTrackingEventsPerInterval : 0u;
  const uint64_t camera_event_count = track_camera_events ? interval_count * kUPBPInitialTrackingEventsPerInterval : 0u;
  const uint64_t event_count = std::max(light_event_count, camera_event_count);
  const uint64_t point_count = collect_points ? path_count * maximum_path_length : 0u;
  const uint64_t beam_count = collect_beams ? path_count * maximum_path_length : 0u;
  const uint64_t path_state_count = path_count;
  const auto addressable = [](uint64_t count, uint32_t stride) {
    return (count <= std::numeric_limits<uint32_t>::max()) && ((count * stride) <= kWavefrontMaxAddressableBufferSize);
  };

  return addressable(vertex_count, kGPUUPBPVertexStride) && addressable(segment_count, kGPUUPBPSegmentStride) && addressable(interval_count, kGPUUPBPIntervalStride) &&
         addressable(std::max<uint64_t>(1u, event_count), kGPUUPBPTrackingEventStride) && addressable(std::max<uint64_t>(1u, point_count), kGPUUPBPPointStride) &&
         addressable(std::max<uint64_t>(1u, beam_count), kGPUUPBPBeamStride) && addressable(path_state_count, kGPUUPBPPathStateStride) &&
         addressable(vertex_count, kGPUUPBPBPTVertexStride) && addressable(path_state_count, kGPUUPBPBPTPathStateStride);
}

uint64_t upbp_density_cache_reserve_bytes(uint64_t working_set_budget, uint32_t global_path_count, uint32_t maximum_path_length, uint32_t technique_mask,
  uint32_t maximum_bb1d_light_path_count) {
  const bool collect_points =
    (technique_mask & (static_cast<uint32_t>(UPBPTechnique::Surface) | static_cast<uint32_t>(UPBPTechnique::PP3D) | static_cast<uint32_t>(UPBPTechnique::PB2D))) != 0u;
  const bool collect_beams = (technique_mask & (static_cast<uint32_t>(UPBPTechnique::BP2D) | static_cast<uint32_t>(UPBPTechnique::BB1D))) != 0u;
  const bool collect_bp2d_beams = (technique_mask & static_cast<uint32_t>(UPBPTechnique::BP2D)) != 0u;
  const bool collect_bb1d_beams = (technique_mask & static_cast<uint32_t>(UPBPTechnique::BB1D)) != 0u;
  const uint64_t maximum_record_count = static_cast<uint64_t>(global_path_count) * maximum_path_length;
  const uint64_t maximum_selected_path_count = maximum_bb1d_light_path_count > 0u ? std::min<uint64_t>(global_path_count, maximum_bb1d_light_path_count) : global_path_count;
  const uint64_t point_bytes = collect_points ? maximum_record_count * (kGPUUPBPDensityPointStride + kGPUUPBPAABBStride) : 0u;
  const uint64_t beam_bytes = collect_beams ? maximum_record_count * kGPUUPBPDensityBeamStride : 0u;
  const uint64_t event_bytes = collect_beams ? maximum_record_count * kUPBPInitialTrackingEventsPerInterval * kGPUUPBPTrackingEventStride : 0u;
  const uint64_t maximum_selected_beam_count = maximum_selected_path_count * maximum_path_length;
  const uint64_t beam_reference_bytes =
    collect_bp2d_beams ? maximum_record_count * kGPUUPBPBeamReferenceStride : (collect_bb1d_beams ? maximum_selected_beam_count * kGPUUPBPBeamReferenceStride : 0u);
  const uint64_t bb1d_instance_bytes = collect_bb1d_beams ? maximum_selected_beam_count * sizeof(RHIAccelerationStructureInstance) : 0u;
  const uint64_t compact_input_bytes = point_bytes + beam_bytes + event_bytes + beam_reference_bytes + bb1d_instance_bytes;
  const uint64_t cache_with_acceleration_reserve = compact_input_bytes + compact_input_bytes / 2u;
  return std::min(working_set_budget / 2u, cache_with_acceleration_reserve);
}

uint32_t upbp_resident_path_capacity(uint64_t working_set_budget, uint32_t maximum_capacity, uint32_t maximum_path_length, uint32_t maximum_boundary_count, uint32_t technique_mask,
  uint32_t maximum_bb1d_light_path_count, uint64_t co_resident_bytes_per_path) {
  uint32_t first = 1u;
  uint32_t last = maximum_capacity;
  uint32_t result = 0u;
  while (first <= last) {
    const uint32_t candidate = first + (last - first) / 2u;
    const uint64_t co_resident_bytes = static_cast<uint64_t>(candidate) * co_resident_bytes_per_path;
    const uint64_t upbp_bytes = upbp_initial_resident_storage_bytes(candidate, maximum_path_length, maximum_boundary_count, technique_mask, maximum_bb1d_light_path_count);
    if (upbp_resident_storage_addressable(candidate, maximum_path_length, maximum_boundary_count, technique_mask, maximum_bb1d_light_path_count) &&
        (co_resident_bytes <= working_set_budget) && (upbp_bytes <= (working_set_budget - co_resident_bytes))) {
      result = candidate;
      first = candidate + 1u;
    } else {
      last = candidate - 1u;
    }
  }
  return result;
}

uint64_t upbp_phase_wavefront_storage_bytes(uint32_t resident_path_capacity, bool connect_to_light, bool connect_to_camera, bool has_subsurface_material) {
  const uint64_t camera_bytes_per_path = kGPUWavefrontPathStateStride + kGPUWavefrontHitStride + 2ull * sizeof(uint32_t) +
                                         static_cast<uint64_t>(kWavefrontRollingHistoryBounces + 1u) * kGPUWavefrontPathVertexStride +
                                         (has_subsurface_material ? kGPUWavefrontSubsurfaceStateStride : 0u);
  uint64_t light_bytes_per_path = kGPUWavefrontPathStateStride + kGPUWavefrontHitStride + 2ull * sizeof(uint32_t) +
                                  static_cast<uint64_t>(kWavefrontLightHistoryBounces + 1u) * kGPUWavefrontLightPathVertexStride +
                                  (has_subsurface_material ? kGPUWavefrontSubsurfaceStateStride : 0u);
  if (connect_to_camera && (connect_to_light == false)) {
    light_bytes_per_path += kGPUWavefrontConnectCameraTaskStride + kGPUWavefrontConnectCameraResultStride;
  }
  return static_cast<uint64_t>(resident_path_capacity) * std::max(camera_bytes_per_path, light_bytes_per_path);
}

uint64_t upbp_camera_resident_storage_bytes(uint32_t resident_path_capacity, uint32_t maximum_path_length, uint32_t maximum_boundary_count, uint32_t technique_mask,
  bool connect_to_light, bool connect_to_camera, bool has_subsurface_material) {
  const uint64_t path_count = resident_path_capacity;
  const uint64_t vertex_count = path_count * (static_cast<uint64_t>(maximum_path_length) + 1u);
  const uint64_t segment_count = path_count * maximum_path_length;
  const uint64_t interval_count = path_count * (static_cast<uint64_t>(maximum_path_length) + maximum_boundary_count);
  const bool collect_points =
    (technique_mask & (static_cast<uint32_t>(UPBPTechnique::Surface) | static_cast<uint32_t>(UPBPTechnique::PP3D) | static_cast<uint32_t>(UPBPTechnique::PB2D))) != 0u;
  const bool collect_beams = (technique_mask & (static_cast<uint32_t>(UPBPTechnique::BP2D) | static_cast<uint32_t>(UPBPTechnique::BB1D))) != 0u;
  const bool track_camera_events = (technique_mask & (static_cast<uint32_t>(UPBPTechnique::PB2D) | static_cast<uint32_t>(UPBPTechnique::BB1D))) != 0u;
  const uint64_t event_count = track_camera_events ? interval_count * kUPBPInitialTrackingEventsPerInterval : 0u;
  const uint64_t point_count = collect_points ? segment_count : 0u;
  const uint64_t beam_count = collect_beams ? segment_count : 0u;
  return vertex_count * kGPUUPBPVertexStride + segment_count * kGPUUPBPSegmentStride + interval_count * kGPUUPBPIntervalStride +
         std::max<uint64_t>(1u, event_count) * kGPUUPBPTrackingEventStride + std::max<uint64_t>(1u, point_count) * kGPUUPBPPointStride +
         std::max<uint64_t>(1u, beam_count) * kGPUUPBPBeamStride + path_count * kGPUUPBPPathStateStride +
         upbp_phase_wavefront_storage_bytes(resident_path_capacity, connect_to_light, connect_to_camera, has_subsurface_material);
}

uint32_t upbp_camera_resident_path_capacity(uint64_t working_set_budget, uint32_t maximum_capacity, uint32_t maximum_path_length, uint32_t maximum_boundary_count,
  uint32_t technique_mask, bool connect_to_light, bool connect_to_camera, bool has_subsurface_material) {
  uint32_t first = 1u;
  uint32_t last = maximum_capacity;
  uint32_t result = 0u;
  while (first <= last) {
    const uint32_t candidate = first + (last - first) / 2u;
    const uint64_t required_bytes =
      upbp_camera_resident_storage_bytes(candidate, maximum_path_length, maximum_boundary_count, technique_mask, connect_to_light, connect_to_camera, has_subsurface_material);
    if (upbp_resident_storage_addressable(candidate, maximum_path_length, maximum_boundary_count, technique_mask, 0u) && (required_bytes <= working_set_budget)) {
      result = candidate;
      first = candidate + 1u;
    } else {
      last = candidate - 1u;
    }
  }
  return result;
}

struct UPBPLightBatch {
  uint32_t offset = 0u;
  uint32_t count = 0u;
};

uint32_t upbp_light_batch_count(uint32_t global_path_count, uint32_t resident_light_capacity, uint32_t matching_offset, uint32_t matching_count) {
  ETX_ASSERT((resident_light_capacity > 0u) && (matching_count > 0u) && (matching_count <= resident_light_capacity) && (matching_offset <= global_path_count) &&
             (matching_count <= (global_path_count - matching_offset)));
  const uint32_t prefix_batch_count = divide_round_up(matching_offset, resident_light_capacity);
  const uint32_t suffix_path_count = global_path_count - matching_offset - matching_count;
  const uint32_t suffix_batch_count = divide_round_up(suffix_path_count, resident_light_capacity);
  return 1u + prefix_batch_count + suffix_batch_count;
}

UPBPLightBatch upbp_light_batch_for_iteration(uint32_t iteration, uint32_t global_path_count, uint32_t resident_light_capacity, uint32_t matching_offset, uint32_t matching_count) {
  ETX_ASSERT(iteration < upbp_light_batch_count(global_path_count, resident_light_capacity, matching_offset, matching_count));
  if (iteration == 0u) {
    return {.offset = matching_offset, .count = matching_count};
  }

  const uint32_t prefix_batch_count = divide_round_up(matching_offset, resident_light_capacity);
  const uint32_t sequential_index = iteration - 1u;
  if (sequential_index < prefix_batch_count) {
    const uint32_t offset = sequential_index * resident_light_capacity;
    return {.offset = offset, .count = std::min(resident_light_capacity, matching_offset - offset)};
  }

  const uint32_t suffix_begin = matching_offset + matching_count;
  const uint32_t suffix_index = sequential_index - prefix_batch_count;
  const uint32_t offset = suffix_begin + suffix_index * resident_light_capacity;
  return {.offset = offset, .count = std::min(resident_light_capacity, global_path_count - offset)};
}

enum class GPUIntegratorMode : uint32_t {
  PathTracing = 0u,
  LightTracing = 1u,
  BDPTFast = 2u,
  BDPTFull = 3u,
  VCM = 4u,
  UPBP = 5u,
};

enum class GPUSpectralMode : uint32_t {
  RGB = 1u,
  Spectral = 2u,
};

GPUSpectralMode gpu_spectral_mode(const SceneData& scene_data) {
  if (scene_data.options.properties[Scene::Properties::Spectral]) {
    return GPUSpectralMode::Spectral;
  }
  return GPUSpectralMode::RGB;
}

struct GPUIntegratorFeatures {
  enum : uint32_t {
    CameraPath = 1u << 0u,
    LightPath = 1u << 1u,
    DirectHit = 1u << 2u,
    ConnectToLight = 1u << 3u,
    ConnectToCamera = 1u << 4u,
    ConnectVertices = 1u << 5u,
    MergeVertices = 1u << 6u,
    VCMMis = 1u << 7u,
    UPBPMis = 1u << 8u,
    UPBPSurface = 1u << 9u,
    UPBPPP3D = 1u << 10u,
    UPBPPB2D = 1u << 11u,
    UPBPBP2D = 1u << 12u,
    UPBPBB1D = 1u << 13u,
  };
};

struct GPUIntegratorSelection {
  GPUIntegratorMode mode = GPUIntegratorMode::PathTracing;
  uint32_t features = 0u;
  Integrator::Type integrator_type = Integrator::Type::Invalid;
  BDPTMode requested_bdpt_mode = BDPTMode::Invalid;
  bool supported = true;
  std::string unsupported_reason = {};
};

struct WavefrontStage {
  GPURaytracingRenderer::PipelineStage stage = GPURaytracingRenderer::PipelineStage::PrepareSample;
  const char* source_file = nullptr;
  const char* entry_point = nullptr;
  const char* optimization_level = nullptr;
  const char* bsdf_kind = nullptr;
  bool uses_stage_entry_define = false;
};

constexpr WavefrontStage kWavefrontStages[] = {
  {GPURaytracingRenderer::PipelineStage::PrepareSample, "shaders/gpu_rt_wavefront_prepare.hlsl", "wavefront_prepare_sample_main", nullptr, nullptr},
  {GPURaytracingRenderer::PipelineStage::PrepareSpectralValues, "shaders/gpu_rt_wavefront_prepare.hlsl", "wavefront_prepare_spectral_values_main", "3", nullptr},
  {GPURaytracingRenderer::PipelineStage::InitCameraPath0, "shaders/gpu_rt_wavefront_init_camera.hlsl", "wavefront_init_camera_path_0_main", nullptr, nullptr},
  {GPURaytracingRenderer::PipelineStage::InitLightPath0, "shaders/gpu_rt_wavefront_init_light.hlsl", "wavefront_init_light_path_0_main", nullptr, nullptr},
  {GPURaytracingRenderer::PipelineStage::TraceCamera, "shaders/gpu_rt_wavefront_trace_camera.hlsl", "wavefront_trace_camera_main", nullptr, nullptr},
  {GPURaytracingRenderer::PipelineStage::CameraSurfaceClassify, "shaders/gpu_rt_wavefront_surface_camera.hlsl", "wavefront_camera_surface_classify_main", nullptr, nullptr},
  {GPURaytracingRenderer::PipelineStage::CameraDirectLightSample, "shaders/gpu_rt_wavefront_direct_light_sample.hlsl", "wavefront_camera_direct_light_sample_main", "3", nullptr},
  {GPURaytracingRenderer::PipelineStage::CameraDirectLightPrepareDiffuse, "shaders/gpu_rt_wavefront_direct_light_prepare_variant.hlsl",
    "wavefront_camera_direct_light_prepare_diffuse_main", "3", "1", true},
  {GPURaytracingRenderer::PipelineStage::CameraDirectLightPreparePlastic, "shaders/gpu_rt_wavefront_direct_light_prepare_variant.hlsl",
    "wavefront_camera_direct_light_prepare_plastic_main", "3", "2", true},
  {GPURaytracingRenderer::PipelineStage::CameraDirectLightPrepareConductor, "shaders/gpu_rt_wavefront_direct_light_prepare_variant.hlsl",
    "wavefront_camera_direct_light_prepare_conductor_main", "3", "3", true},
  {GPURaytracingRenderer::PipelineStage::CameraDirectLightPrepareDielectric, "shaders/gpu_rt_wavefront_direct_light_prepare_variant.hlsl",
    "wavefront_camera_direct_light_prepare_dielectric_main", "3", "4", true},
  {GPURaytracingRenderer::PipelineStage::CameraDirectLightShadow, "shaders/gpu_rt_wavefront_shadow.hlsl", "wavefront_camera_direct_light_shadow_main", nullptr, nullptr},
  {GPURaytracingRenderer::PipelineStage::CameraDirectLightAccumulate, "shaders/gpu_rt_wavefront_direct_light.hlsl", "wavefront_camera_direct_light_accumulate_main", nullptr,
    nullptr},
  {GPURaytracingRenderer::PipelineStage::CameraDirectHitAccumulate, "shaders/gpu_rt_wavefront_direct_hit.hlsl", "wavefront_camera_direct_hit_accumulate_main", nullptr, nullptr},
  {GPURaytracingRenderer::PipelineStage::CameraConnectLightPrepareDiffuse, "shaders/gpu_rt_wavefront_connect_light_prepare_variant.hlsl",
    "wavefront_camera_connect_light_prepare_diffuse_main", "3", "1", true},
  {GPURaytracingRenderer::PipelineStage::CameraConnectLightPreparePlastic, "shaders/gpu_rt_wavefront_connect_light_prepare_variant.hlsl",
    "wavefront_camera_connect_light_prepare_plastic_main", "3", "2", true},
  {GPURaytracingRenderer::PipelineStage::CameraConnectLightPrepareConductor, "shaders/gpu_rt_wavefront_connect_light_prepare_variant.hlsl",
    "wavefront_camera_connect_light_prepare_conductor_main", "3", "3", true},
  {GPURaytracingRenderer::PipelineStage::CameraConnectLightPrepareDielectric, "shaders/gpu_rt_wavefront_connect_light_prepare_variant.hlsl",
    "wavefront_camera_connect_light_prepare_dielectric_main", "3", "4", true},
  {GPURaytracingRenderer::PipelineStage::CameraConnectLightResolveDiffuse, "shaders/gpu_rt_wavefront_connect_light_resolve_variant.hlsl",
    "wavefront_camera_connect_light_resolve_diffuse_main", "3", "1", true},
  {GPURaytracingRenderer::PipelineStage::CameraConnectLightResolvePlastic, "shaders/gpu_rt_wavefront_connect_light_resolve_variant.hlsl",
    "wavefront_camera_connect_light_resolve_plastic_main", "3", "2", true},
  {GPURaytracingRenderer::PipelineStage::CameraConnectLightResolveConductor, "shaders/gpu_rt_wavefront_connect_light_resolve_variant.hlsl",
    "wavefront_camera_connect_light_resolve_conductor_main", "3", "3", true},
  {GPURaytracingRenderer::PipelineStage::CameraConnectLightResolveDielectric, "shaders/gpu_rt_wavefront_connect_light_resolve_variant.hlsl",
    "wavefront_camera_connect_light_resolve_dielectric_main", "3", "4", true},
  {GPURaytracingRenderer::PipelineStage::CameraConnectLightShadow, "shaders/gpu_rt_wavefront_shadow.hlsl", "wavefront_camera_connect_light_shadow_main", nullptr, nullptr},
  {GPURaytracingRenderer::PipelineStage::CameraContinuePrepareDiffuse, "shaders/gpu_rt_wavefront_surface_continue_prepare_camera_variant.hlsl",
    "wavefront_camera_continue_prepare_diffuse_main", "3", "1", true},
  {GPURaytracingRenderer::PipelineStage::CameraContinuePreparePlastic, "shaders/gpu_rt_wavefront_surface_continue_prepare_camera_variant.hlsl",
    "wavefront_camera_continue_prepare_plastic_main", "3", "2", true},
  {GPURaytracingRenderer::PipelineStage::CameraContinuePrepareConductor, "shaders/gpu_rt_wavefront_surface_continue_prepare_camera_variant.hlsl",
    "wavefront_camera_continue_prepare_conductor_main", "3", "3", true},
  {GPURaytracingRenderer::PipelineStage::CameraContinuePrepareDielectric, "shaders/gpu_rt_wavefront_surface_continue_prepare_camera_variant.hlsl",
    "wavefront_camera_continue_prepare_dielectric_main", "3", "4", true},
  {GPURaytracingRenderer::PipelineStage::CameraContinuePrepareThinfilm, "shaders/gpu_rt_wavefront_surface_continue_prepare_camera_variant.hlsl",
    "wavefront_camera_continue_prepare_thinfilm_main", "3", "5", true},
  {GPURaytracingRenderer::PipelineStage::CameraContinueFinalize, "shaders/gpu_rt_wavefront_surface_camera.hlsl", "wavefront_camera_continue_finalize_main", nullptr, nullptr},
  {GPURaytracingRenderer::PipelineStage::TraceLight, "shaders/gpu_rt_wavefront_trace_light.hlsl", "wavefront_trace_light_main", nullptr, nullptr},
  {GPURaytracingRenderer::PipelineStage::LightSurfaceClassify, "shaders/gpu_rt_wavefront_surface_light.hlsl", "wavefront_light_surface_classify_main", nullptr, nullptr},
  {GPURaytracingRenderer::PipelineStage::LightContinuePrepareDiffuse, "shaders/gpu_rt_wavefront_surface_continue_prepare_light_variant.hlsl",
    "wavefront_light_continue_prepare_diffuse_main", "3", "1", true},
  {GPURaytracingRenderer::PipelineStage::LightContinuePreparePlastic, "shaders/gpu_rt_wavefront_surface_continue_prepare_light_variant.hlsl",
    "wavefront_light_continue_prepare_plastic_main", "3", "2", true},
  {GPURaytracingRenderer::PipelineStage::LightContinuePrepareConductor, "shaders/gpu_rt_wavefront_surface_continue_prepare_light_variant.hlsl",
    "wavefront_light_continue_prepare_conductor_main", "3", "3", true},
  {GPURaytracingRenderer::PipelineStage::LightContinuePrepareDielectric, "shaders/gpu_rt_wavefront_surface_continue_prepare_light_variant.hlsl",
    "wavefront_light_continue_prepare_dielectric_main", "3", "4", true},
  {GPURaytracingRenderer::PipelineStage::LightContinuePrepareThinfilm, "shaders/gpu_rt_wavefront_surface_continue_prepare_light_variant.hlsl",
    "wavefront_light_continue_prepare_thinfilm_main", "3", "5", true},
  {GPURaytracingRenderer::PipelineStage::LightConnectCameraClear, "shaders/gpu_rt_wavefront_connect_camera.hlsl", "wavefront_light_connect_camera_prepare_main", nullptr, nullptr},
  {GPURaytracingRenderer::PipelineStage::LightConnectCameraPrepareDiffuse, "shaders/gpu_rt_wavefront_connect_camera_prepare_variant.hlsl",
    "wavefront_light_connect_camera_prepare_diffuse_main", "3", "1", true},
  {GPURaytracingRenderer::PipelineStage::LightConnectCameraPreparePlastic, "shaders/gpu_rt_wavefront_connect_camera_prepare_variant.hlsl",
    "wavefront_light_connect_camera_prepare_plastic_main", "3", "2", true},
  {GPURaytracingRenderer::PipelineStage::LightConnectCameraPrepareConductor, "shaders/gpu_rt_wavefront_connect_camera_prepare_variant.hlsl",
    "wavefront_light_connect_camera_prepare_conductor_main", "3", "3", true},
  {GPURaytracingRenderer::PipelineStage::LightConnectCameraPrepareDielectric, "shaders/gpu_rt_wavefront_connect_camera_prepare_variant.hlsl",
    "wavefront_light_connect_camera_prepare_dielectric_main", "3", "4", true},
  {GPURaytracingRenderer::PipelineStage::LightConnectCameraShadow, "shaders/gpu_rt_wavefront_shadow.hlsl", "wavefront_light_connect_camera_shadow_main", nullptr, nullptr},
  {GPURaytracingRenderer::PipelineStage::LightConnectCameraAccumulate, "shaders/gpu_rt_wavefront_connect_camera.hlsl", "wavefront_light_connect_camera_accumulate_main", nullptr,
    nullptr},
  {GPURaytracingRenderer::PipelineStage::LightContinueFinalize, "shaders/gpu_rt_wavefront_surface_light.hlsl", "wavefront_light_continue_finalize_main", nullptr, nullptr},
  {GPURaytracingRenderer::PipelineStage::SwapQueues, "shaders/gpu_rt_wavefront_prepare.hlsl", "wavefront_swap_queues_main", nullptr, nullptr},
  {GPURaytracingRenderer::PipelineStage::FinalizeSample, "shaders/gpu_rt_wavefront_prepare.hlsl", "wavefront_finalize_sample_main", nullptr, nullptr},
  {GPURaytracingRenderer::PipelineStage::UPBPClear, "shaders/gpu_rt_wavefront_upbp_prepare.hlsl", "wavefront_upbp_clear_main", "3", nullptr},
  {GPURaytracingRenderer::PipelineStage::BuildDispatchArgs, "shaders/gpu_rt_wavefront_dispatch_args.hlsl", "wavefront_build_dispatch_args_main", nullptr, nullptr},
  {GPURaytracingRenderer::PipelineStage::VCMGridClear, "shaders/gpu_rt_wavefront_vcm_grid.hlsl", "wavefront_vcm_grid_clear_main", "3", nullptr},
  {GPURaytracingRenderer::PipelineStage::VCMGridBuild, "shaders/gpu_rt_wavefront_vcm_grid.hlsl", "wavefront_vcm_grid_build_main", "3", nullptr},
  {GPURaytracingRenderer::PipelineStage::VCMGridPrefix, "shaders/gpu_rt_wavefront_vcm_grid.hlsl", "wavefront_vcm_grid_prefix_main", "3", nullptr},
  {GPURaytracingRenderer::PipelineStage::VCMGridScatter, "shaders/gpu_rt_wavefront_vcm_grid.hlsl", "wavefront_vcm_grid_scatter_main", "3", nullptr},
  {GPURaytracingRenderer::PipelineStage::CameraConnectLightClassify, "shaders/gpu_rt_wavefront_connect_light_prepare_variant.hlsl", "wavefront_connect_light_classify_main", "3",
    "1"},
  {GPURaytracingRenderer::PipelineStage::CameraConnectLightCompact, "shaders/gpu_rt_wavefront_connect_light_prepare_variant.hlsl", "wavefront_connect_light_compact_main", "3",
    "1"},
  {GPURaytracingRenderer::PipelineStage::VCMMergeDiffuse, "shaders/gpu_rt_wavefront_vcm_merge_variant.hlsl", "wavefront_vcm_merge_diffuse_main", "3", "1", true},
  {GPURaytracingRenderer::PipelineStage::VCMMergePlastic, "shaders/gpu_rt_wavefront_vcm_merge_variant.hlsl", "wavefront_vcm_merge_plastic_main", "3", "2", true},
  {GPURaytracingRenderer::PipelineStage::VCMMergeConductor, "shaders/gpu_rt_wavefront_vcm_merge_variant.hlsl", "wavefront_vcm_merge_conductor_main", "3", "3", true},
  {GPURaytracingRenderer::PipelineStage::VCMMergeDielectric, "shaders/gpu_rt_wavefront_vcm_merge_variant.hlsl", "wavefront_vcm_merge_dielectric_main", "3", "4", true},
  {GPURaytracingRenderer::PipelineStage::UPBPDensityCompact, "shaders/gpu_rt_wavefront_upbp_density.hlsl", "wavefront_upbp_density_compact_main", "3", nullptr},
  {GPURaytracingRenderer::PipelineStage::UPBPPP3D, "shaders/gpu_rt_wavefront_upbp_density.hlsl", "wavefront_upbp_pp3d_main", "3", nullptr},
  {GPURaytracingRenderer::PipelineStage::UPBPPB2D, "shaders/gpu_rt_wavefront_upbp_density.hlsl", "wavefront_upbp_pb2d_main", "3", nullptr},
  {GPURaytracingRenderer::PipelineStage::UPBPBP2D, "shaders/gpu_rt_wavefront_upbp_density.hlsl", "wavefront_upbp_bp2d_main", "3", nullptr},
  {GPURaytracingRenderer::PipelineStage::UPBPBB1D, "shaders/gpu_rt_wavefront_upbp_density.hlsl", "wavefront_upbp_bb1d_main", "3", nullptr},
  {GPURaytracingRenderer::PipelineStage::UPBPDirectHit, "shaders/gpu_rt_wavefront_upbp_density.hlsl", "wavefront_upbp_direct_hit_main", "3", nullptr},
  {GPURaytracingRenderer::PipelineStage::UPBPValidate, "shaders/gpu_rt_wavefront_upbp_prepare.hlsl", "wavefront_upbp_validate_main", "3", nullptr},
  {GPURaytracingRenderer::PipelineStage::UPBPBeamInstances, "shaders/gpu_rt_wavefront_upbp_density.hlsl", "wavefront_upbp_beam_instances_main", "3", nullptr},
  {GPURaytracingRenderer::PipelineStage::UPBPBeamGridBuild, "shaders/gpu_rt_wavefront_upbp_density.hlsl", "wavefront_upbp_beam_grid_build_main", "3", nullptr},
};

struct WavefrontWindow {
  uint2 origin = {};
  uint2 size = {};
};

uint32_t divide_round_up(uint32_t value, uint32_t divisor) {
  return (value / divisor) + (((value % divisor) == 0u) ? 0u : 1u);
}

uint32_t wavefront_vcm_grid_head_count(uint32_t vertex_capacity) {
  uint32_t value = std::max(1u, vertex_capacity) - 1u;
  value |= value >> 1u;
  value |= value >> 2u;
  value |= value >> 4u;
  value |= value >> 8u;
  value |= value >> 16u;
  return value + 1u;
}

uint64_t wavefront_vcm_grid_storage_size(uint32_t cell_count) {
  uint64_t words = 3ull * cell_count;
  while (cell_count > 64u) {
    cell_count = divide_round_up(cell_count, 64u);
    words += cell_count;
  }
  return words * sizeof(uint32_t);
}

uint64_t material_dispatch_args_offset(bool from_camera, uint32_t material_queue_index) {
  const uint32_t path_queue_offset = from_camera ? 0u : kGPUWavefrontMaterialQueueCountPerPathType;
  return kGPUWavefrontMaterialDispatchArgsOffset + static_cast<uint64_t>(path_queue_offset + material_queue_index) * kGPUWavefrontDispatchArgsStride;
}

uint64_t shadow_dispatch_args_offset(uint32_t shadow_queue_index) {
  return kGPUWavefrontShadowDispatchArgsOffset + static_cast<uint64_t>(shadow_queue_index) * kGPUWavefrontDispatchArgsStride;
}

uint64_t gpu_device_local_allocated_bytes(const RHIMemoryStats& memory_stats) {
  const uint64_t tracked_allocated_bytes = std::min(memory_stats.gpu_allocated_bytes, memory_stats.gpu_device_local_budget_bytes);
  return std::max(std::min(memory_stats.gpu_device_local_allocated_bytes, memory_stats.gpu_device_local_budget_bytes), tracked_allocated_bytes);
}

uint64_t gpu_device_local_available_bytes(const RHIMemoryStats& memory_stats) {
  if (memory_stats.gpu_device_local_budget_bytes == 0ull) {
    return kWavefrontFallbackMemoryBudget;
  }

  return memory_stats.gpu_device_local_budget_bytes - gpu_device_local_allocated_bytes(memory_stats);
}

uint64_t gpu_resident_working_set_budget_bytes(const RHIMemoryStats& memory_stats) {
  if (memory_stats.gpu_device_local_budget_bytes == 0ull) {
    return kWavefrontFallbackMemoryBudget;
  }

  const uint64_t available_bytes = gpu_device_local_available_bytes(memory_stats);
  const uint64_t reserve_bytes = memory_stats.gpu_device_local_budget_bytes / kWavefrontMemoryReserveDivisor;
  const uint64_t budget_bytes = (available_bytes > reserve_bytes) ? (available_bytes - reserve_bytes) : 1ull;
  return budget_bytes;
}

uint64_t wavefront_tile_budget_bytes(const RHIMemoryStats& memory_stats) {
  const uint64_t budget_bytes = gpu_resident_working_set_budget_bytes(memory_stats);
  return std::min(budget_bytes, kWavefrontMaxAddressableBufferSize);
}

uint32_t wavefront_initial_light_history_bounces(uint32_t max_path_length) {
  return std::min(std::max(1u, max_path_length), kWavefrontInitialLightHistoryBounces);
}

uint64_t wavefront_tile_bytes_per_path(uint32_t integrator_features, bool has_subsurface_material, uint32_t light_history_capacity_bounces, bool retain_complete_light_history,
  bool compact_connections) {
  const bool enable_camera_path = (integrator_features & GPUIntegratorFeatures::CameraPath) != 0u;
  const bool enable_light_path = (integrator_features & GPUIntegratorFeatures::LightPath) != 0u;
  const bool enable_connect_to_light = (integrator_features & GPUIntegratorFeatures::ConnectToLight) != 0u;
  const bool enable_connect_to_camera = (integrator_features & GPUIntegratorFeatures::ConnectToCamera) != 0u;
  const bool enable_connect_vertices = (integrator_features & GPUIntegratorFeatures::ConnectVertices) != 0u;
  const bool enable_merge_vertices = (integrator_features & GPUIntegratorFeatures::MergeVertices) != 0u;
  const bool store_complete_light_history = retain_complete_light_history && (enable_connect_vertices || enable_merge_vertices);
  const uint32_t camera_history_bounces = enable_camera_path ? kWavefrontRollingHistoryBounces : 0u;
  const uint32_t light_history_bounces = enable_light_path ? (store_complete_light_history ? light_history_capacity_bounces : kWavefrontLightHistoryBounces) : 0u;

  uint64_t result = kGPUWavefrontPathMetaStride + static_cast<uint64_t>(kGPUWavefrontMaterialQueueCount + 1u + kGPUWavefrontConnectDispatchArgsCount + 1u) * sizeof(uint32_t);
  if (enable_camera_path) {
    result += kGPUWavefrontPathStateStride;
    result += kGPUWavefrontHitStride;
    result += 2ull * sizeof(uint32_t);
    result += static_cast<uint64_t>(camera_history_bounces + 1u) * kGPUWavefrontPathVertexStride;
    if (has_subsurface_material) {
      result += kGPUWavefrontSubsurfaceStateStride;
    }
  }
  if (enable_light_path) {
    result += kGPUWavefrontPathStateStride;
    result += kGPUWavefrontHitStride;
    result += 2ull * sizeof(uint32_t);
    result += static_cast<uint64_t>(light_history_bounces + 1u) * kGPUWavefrontLightPathVertexStride;
    if (has_subsurface_material) {
      result += kGPUWavefrontSubsurfaceStateStride;
    }
  }
  if (enable_connect_to_light || enable_connect_to_camera) {
    uint64_t shadow_work_stride = enable_connect_to_camera ? kGPUWavefrontConnectCameraTaskStride : 0u;
    uint64_t shadow_result_stride = enable_connect_to_camera ? kGPUWavefrontConnectCameraResultStride : 0u;
    if (enable_connect_to_light) {
      shadow_work_stride = std::max<uint64_t>(shadow_work_stride, std::max(kGPUWavefrontDirectLightSampleStride, kGPUWavefrontDirectLightTaskStride));
      shadow_result_stride = std::max<uint64_t>(shadow_result_stride, kGPUWavefrontDirectLightResultStride);
    }
    result += shadow_work_stride + shadow_result_stride;
  }
  if (enable_connect_vertices) {
    result += static_cast<uint64_t>(kGPUWavefrontConnectDispatchArgsCount) * kGPUWavefrontConnectLightTaskStride + 2ull * sizeof(uint32_t);
    if (compact_connections) {
      result += static_cast<uint64_t>(kGPUWavefrontConnectDispatchArgsCount) * 2ull * sizeof(uint32_t);
    }
  }
  if (enable_merge_vertices) {
    // Three power-of-two cell arrays, scan levels, and photon indices fit within eight words per vertex.
    result += static_cast<uint64_t>(light_history_bounces + 1u) * 8ull * sizeof(uint32_t);
  }
  return std::max<uint64_t>(1ull, result);
}

uint32_t wavefront_tile_max_pixels(uint64_t tile_budget_bytes, uint64_t tile_bytes_per_path, uint32_t requested_pixel_count) {
  const uint64_t max_pixels_by_budget = tile_budget_bytes / tile_bytes_per_path;
  if (static_cast<uint64_t>(requested_pixel_count) <= max_pixels_by_budget) {
    return std::max(1u, requested_pixel_count);
  }
  return static_cast<uint32_t>(std::max<uint64_t>(1ull, max_pixels_by_budget));
}

uint32_t wavefront_tile_width(uint2 base_size, uint32_t max_tile_pixels) {
  return std::min(base_size.x, max_tile_pixels);
}

uint32_t wavefront_tile_count(uint2 base_size, uint32_t max_tile_pixels) {
  const uint32_t tile_width = wavefront_tile_width(base_size, max_tile_pixels);
  const uint32_t tile_height = std::max(1u, std::min(base_size.y, max_tile_pixels / tile_width));
  return divide_round_up(base_size.x, tile_width) * divide_round_up(base_size.y, tile_height);
}

uint32_t wavefront_tile_path_capacity(uint2 base_size, uint32_t max_tile_pixels) {
  const uint32_t tile_width = wavefront_tile_width(base_size, max_tile_pixels);
  const uint32_t tile_height = std::max(1u, std::min(base_size.y, max_tile_pixels / tile_width));
  return tile_width * tile_height;
}

WavefrontWindow wavefront_tile_window(uint2 base_origin, uint2 base_size, uint32_t max_tile_pixels, uint32_t tile_index) {
  const uint32_t tile_width = wavefront_tile_width(base_size, max_tile_pixels);
  const uint32_t tile_height = std::max(1u, std::min(base_size.y, max_tile_pixels / tile_width));
  const uint32_t tile_count_x = divide_round_up(base_size.x, tile_width);
  const uint32_t tile_x = tile_index % tile_count_x;
  const uint32_t tile_y = tile_index / tile_count_x;
  const uint32_t local_x = tile_x * tile_width;
  const uint32_t local_y = tile_y * tile_height;

  WavefrontWindow result = {};
  result.origin = {base_origin.x + local_x, base_origin.y + local_y};
  result.size = {std::min(tile_width, base_size.x - local_x), std::min(tile_height, base_size.y - local_y)};
  return result;
}

uint32_t wavefront_tile_path_offset(uint2 base_origin, uint2 base_size, uint32_t max_tile_pixels, uint32_t tile_index) {
  uint64_t result = 0u;
  for (uint32_t index = 0u; index < tile_index; ++index) {
    const WavefrontWindow window = wavefront_tile_window(base_origin, base_size, max_tile_pixels, index);
    result += static_cast<uint64_t>(window.size.x) * window.size.y;
  }
  ETX_ASSERT(result <= std::numeric_limits<uint32_t>::max());
  return static_cast<uint32_t>(result);
}

uint32_t gpu_integrator_features_from_scene_strategies(const SceneData& scene_data, bool enable_camera_path, bool enable_light_path) {
  const uint32_t strategy_flags = scene_data.options.strategy_flags;
  uint32_t result = 0u;

  if (enable_camera_path) {
    result |= GPUIntegratorFeatures::CameraPath;
    if ((strategy_flags & Scene::Strategy::DirectHit) != 0u) {
      result |= GPUIntegratorFeatures::DirectHit;
    }
    if ((strategy_flags & Scene::Strategy::ConnectToLight) != 0u) {
      result |= GPUIntegratorFeatures::ConnectToLight;
    }
  }

  if (enable_light_path) {
    result |= GPUIntegratorFeatures::LightPath;
    if ((strategy_flags & Scene::Strategy::ConnectToCamera) != 0u) {
      result |= GPUIntegratorFeatures::ConnectToCamera;
    }
  }

  if ((enable_camera_path) && (enable_light_path) && ((strategy_flags & Scene::Strategy::ConnectVertices) != 0u)) {
    result |= GPUIntegratorFeatures::ConnectVertices;
  }

  if ((enable_camera_path) && (enable_light_path) && ((strategy_flags & Scene::Strategy::MergeVertices) != 0u)) {
    result |= GPUIntegratorFeatures::MergeVertices | GPUIntegratorFeatures::VCMMis;
  }

  return result;
}

GPUIntegratorMode gpu_integrator_mode_from_scene_strategies(const SceneData& scene_data) {
  const uint32_t strategy_flags = scene_data.options.strategy_flags;
  const bool direct_hit_enabled = (strategy_flags & Scene::Strategy::DirectHit) != 0u;
  const bool connect_to_light_enabled = (strategy_flags & Scene::Strategy::ConnectToLight) != 0u;
  const bool connect_to_camera_enabled = (strategy_flags & Scene::Strategy::ConnectToCamera) != 0u;

  if ((connect_to_camera_enabled == false) && (direct_hit_enabled || connect_to_light_enabled)) {
    return GPUIntegratorMode::PathTracing;
  }

  if ((connect_to_camera_enabled) && (direct_hit_enabled == false) && (connect_to_light_enabled == false)) {
    return GPUIntegratorMode::LightTracing;
  }

  return GPUIntegratorMode::BDPTFast;
}

const char* integrator_type_to_display_name(Integrator::Type type) {
  switch (type) {
    case Integrator::Type::Debug:
      return "Debug";
    case Integrator::Type::PathTracing:
      return "Path Tracing";
    case Integrator::Type::Bidirectional:
      return "Bidirectional";
    case Integrator::Type::VCM:
      return "VCM";
    case Integrator::Type::UPBP:
      return "UPBP";
    default:
      return "Unknown";
  }
}

GPUIntegratorMode gpu_integrator_mode_from_bdpt_mode(BDPTMode mode) {
  switch (mode) {
    case BDPTMode::PathTracing:
      return GPUIntegratorMode::PathTracing;
    case BDPTMode::LightTracing:
      return GPUIntegratorMode::LightTracing;
    case BDPTMode::BDPTFast:
      return GPUIntegratorMode::BDPTFast;
    case BDPTMode::BDPTFull:
      return GPUIntegratorMode::BDPTFull;
    default:
      return GPUIntegratorMode::BDPTFast;
  }
}

GPUIntegratorSelection gpu_integrator_selection_from_scene(const SceneRepresentation& scene) {
  GPUIntegratorSelection result = {};
  const auto& integrator_data = scene.integrator_data();
  result.integrator_type = integrator_data.selected;

  if (integrator_data.selected == Integrator::Type::PathTracing) {
    result.requested_bdpt_mode = BDPTMode::PathTracing;
    result.mode = GPUIntegratorMode::PathTracing;
    result.features = gpu_integrator_features_from_scene_strategies(scene.data(), true, false);
    return result;
  }

  if (integrator_data.selected == Integrator::Type::Bidirectional) {
    auto settings_it = integrator_data.settings.find(Integrator::Type::Bidirectional);
    BDPTMode bidirectional_mode = BDPTMode::BDPTFast;
    if (settings_it != integrator_data.settings.end()) {
      bidirectional_mode = settings_it->second.get_integral("bdpt-mode", bidirectional_mode);
    }

    result.requested_bdpt_mode = bidirectional_mode;
    result.mode = gpu_integrator_mode_from_bdpt_mode(bidirectional_mode);

    if (bdpt_mode_valid(bidirectional_mode) == false) {
      result.supported = false;
      return result;
    }

    const bool enable_camera_path = bidirectional_mode != BDPTMode::LightTracing;
    const bool enable_light_path = bidirectional_mode != BDPTMode::PathTracing;
    result.features = gpu_integrator_features_from_scene_strategies(scene.data(), enable_camera_path, enable_light_path);
    if (bidirectional_mode == BDPTMode::PathTracing) {
      result.features &= ~(GPUIntegratorFeatures::LightPath | GPUIntegratorFeatures::ConnectToCamera | GPUIntegratorFeatures::ConnectVertices |
                           GPUIntegratorFeatures::MergeVertices | GPUIntegratorFeatures::VCMMis);
    } else if (bidirectional_mode == BDPTMode::LightTracing) {
      result.features &= ~(GPUIntegratorFeatures::CameraPath | GPUIntegratorFeatures::DirectHit | GPUIntegratorFeatures::ConnectToLight | GPUIntegratorFeatures::ConnectVertices |
                           GPUIntegratorFeatures::MergeVertices | GPUIntegratorFeatures::VCMMis);
    } else if (bidirectional_mode == BDPTMode::BDPTFast) {
      result.features &= ~(GPUIntegratorFeatures::ConnectVertices | GPUIntegratorFeatures::MergeVertices | GPUIntegratorFeatures::VCMMis);
    } else if (bidirectional_mode == BDPTMode::BDPTFull) {
      result.features &= ~(GPUIntegratorFeatures::MergeVertices | GPUIntegratorFeatures::VCMMis);
    }
    return result;
  }

  if (integrator_data.selected == Integrator::Type::VCM) {
    result.mode = GPUIntegratorMode::VCM;
    result.features = gpu_integrator_features_from_scene_strategies(scene.data(), true, true);
    result.features |= GPUIntegratorFeatures::VCMMis;
    auto settings_it = integrator_data.settings.find(Integrator::Type::VCM);
    if ((settings_it != integrator_data.settings.end()) && (settings_it->second.get_bool("vcm-merging", true) == false)) {
      result.features &= ~GPUIntegratorFeatures::MergeVertices;
    }
    return result;
  }

  if (integrator_data.selected == Integrator::Type::UPBP) {
    result.mode = GPUIntegratorMode::UPBP;
    result.features = gpu_integrator_features_from_scene_strategies(scene.data(), true, true);
    result.features |= GPUIntegratorFeatures::UPBPMis;
    result.features &= ~GPUIntegratorFeatures::VCMMis;

    UPBPOptions options = {};
    const auto settings_it = integrator_data.settings.find(Integrator::Type::UPBP);
    if (settings_it != integrator_data.settings.end()) {
      options.load(settings_it->second);
    }
    if (upbp_options_valid(options, result.unsupported_reason) == false) {
      result.supported = false;
      return result;
    }

    const bool merge_vertices_enabled = (scene.data().options.strategy_flags & Scene::Strategy::MergeVertices) != 0u;
    const uint32_t effective_technique_mask = upbp_effective_technique_mask(options, merge_vertices_enabled);
    if (effective_technique_mask == 0u) {
      result.supported = false;
      result.unsupported_reason = "UPBP has no enabled techniques after applying the scene strategy controls";
      return result;
    }
    const auto technique_enabled = [effective_technique_mask](const UPBPTechnique technique) {
      return (effective_technique_mask & static_cast<uint32_t>(technique)) != 0u;
    };
    const bool multiple_importance_sampling = scene.data().options.properties[Scene::Properties::MultipleImportanceSampling];
    if ((multiple_importance_sampling == false) && (effective_technique_mask != 0u) && ((effective_technique_mask & (effective_technique_mask - 1u)) != 0u)) {
      result.supported = false;
      result.unsupported_reason = "UPBP requires multiple importance sampling when more than one technique is enabled";
      return result;
    }
    const uint32_t required_bpt_strategies = Scene::Strategy::DirectHit | Scene::Strategy::ConnectToLight | Scene::Strategy::ConnectToCamera | Scene::Strategy::ConnectVertices;
    const bool all_bpt_strategies_enabled = (scene.data().options.strategy_flags & required_bpt_strategies) == required_bpt_strategies;
    if (technique_enabled(UPBPTechnique::BPT) && multiple_importance_sampling && (all_bpt_strategies_enabled == false)) {
      result.supported = false;
      result.unsupported_reason = (effective_technique_mask & kUPBPDensityTechniqueMask) != 0u
                                    ? "UPBP cross-technique MIS requires all BPT endpoint strategies when BPT and density techniques are enabled together"
                                    : "GPU UPBP does not support partial BPT endpoint strategies with multiple importance sampling";
      return result;
    }

    if (technique_enabled(UPBPTechnique::BPT) == false) {
      result.features &=
        ~(GPUIntegratorFeatures::DirectHit | GPUIntegratorFeatures::ConnectToLight | GPUIntegratorFeatures::ConnectToCamera | GPUIntegratorFeatures::ConnectVertices);
    }
    if (technique_enabled(UPBPTechnique::Surface)) {
      result.features |= GPUIntegratorFeatures::UPBPSurface;
    }
    if (technique_enabled(UPBPTechnique::PP3D)) {
      result.features |= GPUIntegratorFeatures::UPBPPP3D;
    }
    if (technique_enabled(UPBPTechnique::PB2D)) {
      result.features |= GPUIntegratorFeatures::UPBPPB2D;
    }
    if (technique_enabled(UPBPTechnique::BP2D)) {
      result.features |= GPUIntegratorFeatures::UPBPBP2D;
    }
    if (technique_enabled(UPBPTechnique::BB1D)) {
      result.features |= GPUIntegratorFeatures::UPBPBB1D;
    }
    const uint32_t density_features =
      GPUIntegratorFeatures::UPBPSurface | GPUIntegratorFeatures::UPBPPP3D | GPUIntegratorFeatures::UPBPPB2D | GPUIntegratorFeatures::UPBPBP2D | GPUIntegratorFeatures::UPBPBB1D;
    if ((result.features & density_features) == 0u) {
      result.features &= ~GPUIntegratorFeatures::MergeVertices;
    }
    return result;
  }

  if ((integrator_data.selected != Integrator::Type::Invalid) && (integrator_data.selected != Integrator::Type::PathTracing)) {
    result.supported = false;
    result.mode = gpu_integrator_mode_from_scene_strategies(scene.data());
    result.features = gpu_integrator_features_from_scene_strategies(scene.data(), result.mode != GPUIntegratorMode::LightTracing, result.mode != GPUIntegratorMode::PathTracing);
    if (result.mode == GPUIntegratorMode::BDPTFast) {
      result.features &= ~(GPUIntegratorFeatures::ConnectVertices | GPUIntegratorFeatures::MergeVertices | GPUIntegratorFeatures::VCMMis);
    }
    return result;
  }

  result.mode = gpu_integrator_mode_from_scene_strategies(scene.data());
  result.features = gpu_integrator_features_from_scene_strategies(scene.data(), result.mode != GPUIntegratorMode::LightTracing, result.mode != GPUIntegratorMode::PathTracing);
  if (result.mode == GPUIntegratorMode::BDPTFast) {
    result.features &= ~(GPUIntegratorFeatures::ConnectVertices | GPUIntegratorFeatures::MergeVertices | GPUIntegratorFeatures::VCMMis);
  }
  return result;
}

struct GPUVCMIterationParameters {
  float radius = 0.0f;
  float vm_weight = 0.0f;
  float vc_weight = 0.0f;
  float vm_normalization = 0.0f;
  uint32_t kernel = 1u;
};

GPUVCMIterationParameters gpu_vcm_iteration_parameters(const SceneRepresentation& scene, float bounding_sphere_radius, uint32_t sample_index, const uint2& render_dimensions,
  uint32_t light_path_count, bool merging_enabled) {
  GPUVCMIterationParameters result = {};
  float initial_radius = 0.0f;
  auto settings_it = scene.integrator_data().settings.find(Integrator::Type::VCM);
  if (settings_it != scene.integrator_data().settings.end()) {
    initial_radius = settings_it->second.get_float("vcm-initial_radius", initial_radius);
    result.kernel = settings_it->second.get_integral("vcm-kernel", result.kernel);
  }

  result.radius = vcm_iteration_radius(initial_radius, bounding_sphere_radius, std::max(render_dimensions.x, render_dimensions.y), sample_index);
  const float eta = kPi * result.radius * result.radius * static_cast<float>(std::max(1u, light_path_count));
  result.vc_weight = 1.0f / eta;
  result.vm_weight = merging_enabled ? eta : 0.0f;
  result.vm_normalization = 1.0f / eta;
  return result;
}

GPUIntegratorMode gpu_integrator_mode_from_scene(const SceneRepresentation& scene) {
  return gpu_integrator_selection_from_scene(scene).mode;
}

std::string gpu_integrator_selection_error_message(const GPUIntegratorSelection& selection) {
  if (selection.unsupported_reason.empty() == false) {
    return std::string("GPU RT cannot use the requested '") + integrator_type_to_display_name(selection.integrator_type) + "' settings: " + selection.unsupported_reason + ".";
  }
  if (selection.integrator_type == Integrator::Type::Bidirectional) {
    if (bdpt_mode_valid(selection.requested_bdpt_mode) == false) {
      return "GPU RT does not support the requested bidirectional mode value.";
    }
  }

  return std::string("GPU RT does not support the '") + integrator_type_to_display_name(selection.integrator_type) + "' integrator.";
}

const char* gpu_integrator_mode_to_string(GPUIntegratorMode mode) {
  switch (mode) {
    case GPUIntegratorMode::PathTracing:
      return "PathTracing";
    case GPUIntegratorMode::LightTracing:
      return "LightTracing";
    case GPUIntegratorMode::BDPTFast:
      return "BDPTFast";
    case GPUIntegratorMode::BDPTFull:
      return "BDPTFull";
    case GPUIntegratorMode::VCM:
      return "VCM";
    case GPUIntegratorMode::UPBP:
      return "UPBP";
    default:
      return "Unknown";
  }
}

const char* pipeline_stage_to_string(GPURaytracingRenderer::PipelineStage stage) {
  switch (stage) {
    case GPURaytracingRenderer::PipelineStage::PrepareSample:
      return "PrepareSample";
    case GPURaytracingRenderer::PipelineStage::PrepareSpectralValues:
      return "PrepareSpectralValues";
    case GPURaytracingRenderer::PipelineStage::InitCameraPath0:
      return "InitCameraPath0";
    case GPURaytracingRenderer::PipelineStage::InitLightPath0:
      return "InitLightPath0";
    case GPURaytracingRenderer::PipelineStage::TraceCamera:
      return "TraceCamera";
    case GPURaytracingRenderer::PipelineStage::CameraSurfaceClassify:
      return "CameraSurfaceClassify";
    case GPURaytracingRenderer::PipelineStage::CameraDirectLightSample:
      return "CameraDirectLightSample";
    case GPURaytracingRenderer::PipelineStage::CameraDirectLightPrepareDiffuse:
      return "CameraDirectLightPrepareDiffuse";
    case GPURaytracingRenderer::PipelineStage::CameraDirectLightPreparePlastic:
      return "CameraDirectLightPreparePlastic";
    case GPURaytracingRenderer::PipelineStage::CameraDirectLightPrepareConductor:
      return "CameraDirectLightPrepareConductor";
    case GPURaytracingRenderer::PipelineStage::CameraDirectLightPrepareDielectric:
      return "CameraDirectLightPrepareDielectric";
    case GPURaytracingRenderer::PipelineStage::CameraDirectLightShadow:
      return "CameraDirectLightShadow";
    case GPURaytracingRenderer::PipelineStage::CameraDirectLightAccumulate:
      return "CameraDirectLightAccumulate";
    case GPURaytracingRenderer::PipelineStage::CameraDirectHitAccumulate:
      return "CameraDirectHitAccumulate";
    case GPURaytracingRenderer::PipelineStage::CameraConnectLightPrepareDiffuse:
      return "CameraConnectLightPrepareDiffuse";
    case GPURaytracingRenderer::PipelineStage::CameraConnectLightPreparePlastic:
      return "CameraConnectLightPreparePlastic";
    case GPURaytracingRenderer::PipelineStage::CameraConnectLightPrepareConductor:
      return "CameraConnectLightPrepareConductor";
    case GPURaytracingRenderer::PipelineStage::CameraConnectLightPrepareDielectric:
      return "CameraConnectLightPrepareDielectric";
    case GPURaytracingRenderer::PipelineStage::CameraConnectLightResolveDiffuse:
      return "CameraConnectLightResolveDiffuse";
    case GPURaytracingRenderer::PipelineStage::CameraConnectLightResolvePlastic:
      return "CameraConnectLightResolvePlastic";
    case GPURaytracingRenderer::PipelineStage::CameraConnectLightResolveConductor:
      return "CameraConnectLightResolveConductor";
    case GPURaytracingRenderer::PipelineStage::CameraConnectLightResolveDielectric:
      return "CameraConnectLightResolveDielectric";
    case GPURaytracingRenderer::PipelineStage::CameraConnectLightShadow:
      return "CameraConnectLightShadow";
    case GPURaytracingRenderer::PipelineStage::CameraContinuePrepareDiffuse:
      return "CameraContinuePrepareDiffuse";
    case GPURaytracingRenderer::PipelineStage::CameraContinuePreparePlastic:
      return "CameraContinuePreparePlastic";
    case GPURaytracingRenderer::PipelineStage::CameraContinuePrepareConductor:
      return "CameraContinuePrepareConductor";
    case GPURaytracingRenderer::PipelineStage::CameraContinuePrepareDielectric:
      return "CameraContinuePrepareDielectric";
    case GPURaytracingRenderer::PipelineStage::CameraContinuePrepareThinfilm:
      return "CameraContinuePrepareThinfilm";
    case GPURaytracingRenderer::PipelineStage::CameraContinueFinalize:
      return "CameraContinueFinalize";
    case GPURaytracingRenderer::PipelineStage::TraceLight:
      return "TraceLight";
    case GPURaytracingRenderer::PipelineStage::LightSurfaceClassify:
      return "LightSurfaceClassify";
    case GPURaytracingRenderer::PipelineStage::LightConnectCameraPrepareDiffuse:
      return "LightConnectCameraPrepareDiffuse";
    case GPURaytracingRenderer::PipelineStage::LightConnectCameraPreparePlastic:
      return "LightConnectCameraPreparePlastic";
    case GPURaytracingRenderer::PipelineStage::LightConnectCameraPrepareConductor:
      return "LightConnectCameraPrepareConductor";
    case GPURaytracingRenderer::PipelineStage::LightConnectCameraPrepareDielectric:
      return "LightConnectCameraPrepareDielectric";
    case GPURaytracingRenderer::PipelineStage::LightConnectCameraShadow:
      return "LightConnectCameraShadow";
    case GPURaytracingRenderer::PipelineStage::LightConnectCameraAccumulate:
      return "LightConnectCameraAccumulate";
    case GPURaytracingRenderer::PipelineStage::LightContinuePrepareDiffuse:
      return "LightContinuePrepareDiffuse";
    case GPURaytracingRenderer::PipelineStage::LightContinuePreparePlastic:
      return "LightContinuePreparePlastic";
    case GPURaytracingRenderer::PipelineStage::LightContinuePrepareConductor:
      return "LightContinuePrepareConductor";
    case GPURaytracingRenderer::PipelineStage::LightContinuePrepareDielectric:
      return "LightContinuePrepareDielectric";
    case GPURaytracingRenderer::PipelineStage::LightContinuePrepareThinfilm:
      return "LightContinuePrepareThinfilm";
    case GPURaytracingRenderer::PipelineStage::LightContinueFinalize:
      return "LightContinueFinalize";
    case GPURaytracingRenderer::PipelineStage::LightConnectCameraClear:
      return "LightConnectCameraClear";
    case GPURaytracingRenderer::PipelineStage::SwapQueues:
      return "SwapQueues";
    case GPURaytracingRenderer::PipelineStage::FinalizeSample:
      return "FinalizeSample";
    case GPURaytracingRenderer::PipelineStage::UPBPClear:
      return "UPBPClear";
    case GPURaytracingRenderer::PipelineStage::BuildDispatchArgs:
      return "BuildDispatchArgs";
    case GPURaytracingRenderer::PipelineStage::VCMGridClear:
      return "VCMGridClear";
    case GPURaytracingRenderer::PipelineStage::VCMGridBuild:
      return "VCMGridBuild";
    case GPURaytracingRenderer::PipelineStage::VCMGridPrefix:
      return "VCMGridPrefix";
    case GPURaytracingRenderer::PipelineStage::VCMGridScatter:
      return "VCMGridScatter";
    case GPURaytracingRenderer::PipelineStage::CameraConnectLightClassify:
      return "CameraConnectLightClassify";
    case GPURaytracingRenderer::PipelineStage::CameraConnectLightCompact:
      return "CameraConnectLightCompact";
    case GPURaytracingRenderer::PipelineStage::VCMMergeDiffuse:
      return "VCMMergeDiffuse";
    case GPURaytracingRenderer::PipelineStage::VCMMergePlastic:
      return "VCMMergePlastic";
    case GPURaytracingRenderer::PipelineStage::VCMMergeConductor:
      return "VCMMergeConductor";
    case GPURaytracingRenderer::PipelineStage::VCMMergeDielectric:
      return "VCMMergeDielectric";
    case GPURaytracingRenderer::PipelineStage::UPBPDensityCompact:
      return "UPBPDensityCompact";
    case GPURaytracingRenderer::PipelineStage::UPBPBeamInstances:
      return "UPBPBeamInstances";
    case GPURaytracingRenderer::PipelineStage::UPBPBeamGridBuild:
      return "UPBPBeamGridBuild";
    case GPURaytracingRenderer::PipelineStage::UPBPPP3D:
      return "UPBPPP3D";
    case GPURaytracingRenderer::PipelineStage::UPBPPB2D:
      return "UPBPPB2D";
    case GPURaytracingRenderer::PipelineStage::UPBPBP2D:
      return "UPBPBP2D";
    case GPURaytracingRenderer::PipelineStage::UPBPBB1D:
      return "UPBPBB1D";
    case GPURaytracingRenderer::PipelineStage::UPBPDirectHit:
      return "UPBPDirectHit";
    case GPURaytracingRenderer::PipelineStage::UPBPValidate:
      return "UPBPValidate";
    case GPURaytracingRenderer::PipelineStage::Count:
      return "Count";
    default:
      return "Unknown";
  }
}

const char* upbp_beam_grid_failure_to_string(uint32_t failure) {
  switch (failure) {
    case GPUUPBPBeamGridBuildFailure::None:
      return "none";
    case GPUUPBPBeamGridBuildFailure::InvalidInput:
      return "invalid input";
    case GPUUPBPBeamGridBuildFailure::EntryCountOverflow:
      return "entry-count overflow";
    case GPUUPBPBeamGridBuildFailure::OutputCapacity:
      return "output capacity exceeded";
    case GPUUPBPBeamGridBuildFailure::InvalidIndex:
      return "invalid beam index";
    case GPUUPBPBeamGridBuildFailure::DuplicateIndex:
      return "duplicate beam index";
    case GPUUPBPBeamGridBuildFailure::InvalidOrder:
      return "out-of-order beam index";
    case GPUUPBPBeamGridBuildFailure::CountScatterMismatch:
      return "count/scatter mismatch";
    default:
      return "multiple or unknown failures";
  }
}

const char* bsdf_kind_to_string(const std::string& bsdf_kind) {
  if (bsdf_kind == "1") {
    return "Diffuse";
  }
  if (bsdf_kind == "2") {
    return "Plastic";
  }
  if (bsdf_kind == "3") {
    return "Conductor";
  }
  if (bsdf_kind == "4") {
    return "Dielectric";
  }
  if (bsdf_kind == "5") {
    return "Thinfilm";
  }
  return bsdf_kind.empty() ? "-" : bsdf_kind.c_str();
}

constexpr uint32_t material_compile_bit(uint32_t material_class) {
  return (material_class < 32u) ? (1u << material_class) : 0u;
}

bool material_compile_mask_has(uint32_t mask, uint32_t material_class) {
  const uint32_t bit = material_compile_bit(material_class);
  return (bit != 0u) && ((mask & bit) != 0u);
}

bool material_compile_mask_has_various_continue(uint32_t mask) {
  return (material_compile_mask_has(mask, MaterialClass::Diffuse)) || (material_compile_mask_has(mask, MaterialClass::Translucent)) ||
         (material_compile_mask_has(mask, MaterialClass::Mirror)) || (material_compile_mask_has(mask, MaterialClass::Boundary)) ||
         (material_compile_mask_has(mask, MaterialClass::Velvet)) || (material_compile_mask_has(mask, MaterialClass::Void)) ||
         (material_compile_mask_has(mask, MaterialClass::DiffractionGrating));
}

bool material_compile_mask_has_various_connect(uint32_t mask) {
  return (material_compile_mask_has_various_continue(mask)) || (material_compile_mask_has(mask, MaterialClass::Thinfilm));
}

bool material_compile_mask_has_conductor_stage(uint32_t mask) {
  return material_compile_mask_has(mask, MaterialClass::Conductor);
}

bool material_compile_mask_has_connectible_conductor(uint32_t mask) {
  return (mask & kMaterialCompileConnectibleConductor) != 0u;
}

bool material_compile_mask_has_connectible_dielectric(uint32_t mask) {
  return (mask & kMaterialCompileConnectibleDielectric) != 0u;
}

uint32_t material_compile_mask_connection_queue_count(uint32_t mask) {
  return static_cast<uint32_t>(material_compile_mask_has_various_connect(mask)) + static_cast<uint32_t>(material_compile_mask_has(mask, MaterialClass::Plastic)) +
         static_cast<uint32_t>(material_compile_mask_has_connectible_conductor(mask)) + static_cast<uint32_t>(material_compile_mask_has_connectible_dielectric(mask));
}

uint32_t material_compile_mask_work_queue_count(uint32_t mask) {
  return static_cast<uint32_t>(material_compile_mask_has_various_continue(mask)) + static_cast<uint32_t>(material_compile_mask_has(mask, MaterialClass::Plastic)) +
         static_cast<uint32_t>(material_compile_mask_has_conductor_stage(mask)) + static_cast<uint32_t>(material_compile_mask_has(mask, MaterialClass::Dielectric)) +
         static_cast<uint32_t>(material_compile_mask_has(mask, MaterialClass::Thinfilm));
}

bool wavefront_stage_source_is(const WavefrontStage& stage, const char* source_file) {
  return (stage.source_file != nullptr) && (std::strcmp(stage.source_file, source_file) == 0);
}

struct WavefrontStageCompileOptions {
  bool path_tracing_only = false;
  bool work_queues = false;
  bool thinfilm = false;
  bool velvet = false;
  bool upbp = false;
};

WavefrontStageCompileOptions wavefront_stage_compile_options(const WavefrontStage& stage, GPUIntegratorMode mode, uint32_t material_compile_mask) {
  const bool diffuse_variant = (stage.bsdf_kind != nullptr) && (std::strcmp(stage.bsdf_kind, "1") == 0);
  const bool dielectric_variant = (stage.bsdf_kind != nullptr) && (std::strcmp(stage.bsdf_kind, "4") == 0);
  const bool surface_continue_variant = wavefront_stage_source_is(stage, "shaders/gpu_rt_wavefront_surface_continue_prepare_camera_variant.hlsl") ||
                                        wavefront_stage_source_is(stage, "shaders/gpu_rt_wavefront_surface_continue_prepare_light_variant.hlsl");
  const bool direct_light_variant = wavefront_stage_source_is(stage, "shaders/gpu_rt_wavefront_direct_light_prepare_variant.hlsl");
  const bool connect_light_prepare_variant = wavefront_stage_source_is(stage, "shaders/gpu_rt_wavefront_connect_light_prepare_variant.hlsl");
  const bool connect_light_resolve_variant = wavefront_stage_source_is(stage, "shaders/gpu_rt_wavefront_connect_light_resolve_variant.hlsl");
  const bool connect_camera_variant = wavefront_stage_source_is(stage, "shaders/gpu_rt_wavefront_connect_camera_prepare_variant.hlsl");
  const bool vcm_merge_variant = wavefront_stage_source_is(stage, "shaders/gpu_rt_wavefront_vcm_merge_variant.hlsl");
  const bool surface_classify =
    wavefront_stage_source_is(stage, "shaders/gpu_rt_wavefront_surface_camera.hlsl") || wavefront_stage_source_is(stage, "shaders/gpu_rt_wavefront_surface_light.hlsl");

  WavefrontStageCompileOptions result = {};
  result.path_tracing_only = (mode == GPUIntegratorMode::PathTracing) && (surface_continue_variant || direct_light_variant);
  result.work_queues = (material_compile_mask_work_queue_count(material_compile_mask) > 1u) &&
                       (surface_classify || surface_continue_variant || direct_light_variant || connect_camera_variant || vcm_merge_variant);
  if (connect_light_prepare_variant || connect_light_resolve_variant) {
    result.work_queues = material_compile_mask_connection_queue_count(material_compile_mask) > 1u;
  }
  const bool thinfilm_material = material_compile_mask_has(material_compile_mask, MaterialClass::Thinfilm);
  const bool upbp_surface_thinfilm = (mode == GPUIntegratorMode::UPBP) && vcm_merge_variant && dielectric_variant;
  const bool other_thinfilm_stage = diffuse_variant && (direct_light_variant || connect_light_prepare_variant || connect_light_resolve_variant || connect_camera_variant ||
                                                         (vcm_merge_variant && (mode != GPUIntegratorMode::UPBP)));
  result.thinfilm = thinfilm_material && (upbp_surface_thinfilm || other_thinfilm_stage);
  result.velvet =
    diffuse_variant && material_compile_mask_has(material_compile_mask, MaterialClass::Velvet) &&
    (surface_continue_variant || direct_light_variant || connect_light_prepare_variant || connect_light_resolve_variant || connect_camera_variant || vcm_merge_variant);
  result.upbp = mode == GPUIntegratorMode::UPBP;
  return result;
}

uint64_t wavefront_stage_variant_key(const WavefrontStage& stage, GPUIntegratorMode mode, uint32_t material_compile_mask, uint32_t spectral_mode) {
  const WavefrontStageCompileOptions options = wavefront_stage_compile_options(stage, mode, material_compile_mask);
  uint64_t result = spectral_mode;
  result |= options.path_tracing_only ? (1ull << 8u) : 0ull;
  result |= options.work_queues ? (1ull << 9u) : 0ull;
  result |= options.thinfilm ? (1ull << 10u) : 0ull;
  result |= options.velvet ? (1ull << 11u) : 0ull;
  result |= options.upbp ? (1ull << 12u) : 0ull;
  return result;
}

std::unordered_map<std::string, std::string> wavefront_stage_defines(const WavefrontStage& stage, GPUIntegratorMode mode, uint32_t material_compile_mask, uint32_t spectral_mode) {
  std::unordered_map<std::string, std::string> result = {};
  if (stage.optimization_level != nullptr) {
    result["ETX_DXC_OPT_LEVEL"] = stage.optimization_level;
  }
  if (stage.bsdf_kind != nullptr) {
    result["ETX_BSDF_KIND"] = stage.bsdf_kind;
  }
  result["ETX_SPECTRAL_MODE"] = std::to_string(spectral_mode);

  const WavefrontStageCompileOptions options = wavefront_stage_compile_options(stage, mode, material_compile_mask);
  if (options.path_tracing_only) {
    result["ETX_WAVEFRONT_PATH_TRACING_ONLY"] = "1";
  }
  if (options.work_queues) {
    result["ETX_ENABLE_WORK_QUEUES"] = "1";
  }
  if (options.thinfilm) {
    result["ETX_ENABLE_THINFILM_STAGE"] = "1";
  }
  if (options.velvet) {
    result["ETX_ENABLE_VELVET_STAGE"] = "1";
  }
  if (options.upbp) {
    result["ETX_UPBP"] = "1";
  }
  if (stage.uses_stage_entry_define) {
    result["ETX_STAGE_ENTRY"] = stage.entry_point;
  }
  return result;
}

bool gpu_integrator_feature_enabled(uint32_t features, uint32_t feature) {
  return (features & feature) != 0u;
}

bool wavefront_stage_enabled(GPURaytracingRenderer::PipelineStage stage, GPUIntegratorMode mode, uint32_t features, uint32_t material_compile_mask, uint32_t spectral_mode) {
  const bool enable_camera_path = gpu_integrator_feature_enabled(features, GPUIntegratorFeatures::CameraPath);
  const bool enable_light_path = gpu_integrator_feature_enabled(features, GPUIntegratorFeatures::LightPath);
  const bool enable_direct_hit = gpu_integrator_feature_enabled(features, GPUIntegratorFeatures::DirectHit);
  const bool enable_connect_to_light = gpu_integrator_feature_enabled(features, GPUIntegratorFeatures::ConnectToLight);
  const bool enable_connect_to_camera = gpu_integrator_feature_enabled(features, GPUIntegratorFeatures::ConnectToCamera);
  const bool enable_connect_vertices = gpu_integrator_feature_enabled(features, GPUIntegratorFeatures::ConnectVertices);
  const bool enable_merge_vertices = gpu_integrator_feature_enabled(features, GPUIntegratorFeatures::MergeVertices);
  const bool enable_upbp_surface = gpu_integrator_feature_enabled(features, GPUIntegratorFeatures::UPBPSurface);
  const bool enable_upbp_pp3d = gpu_integrator_feature_enabled(features, GPUIntegratorFeatures::UPBPPP3D);
  const bool enable_upbp_pb2d = gpu_integrator_feature_enabled(features, GPUIntegratorFeatures::UPBPPB2D);
  const bool enable_upbp_bp2d = gpu_integrator_feature_enabled(features, GPUIntegratorFeatures::UPBPBP2D);
  const bool enable_upbp_bb1d = gpu_integrator_feature_enabled(features, GPUIntegratorFeatures::UPBPBB1D);
  const bool has_various_continue = material_compile_mask_has_various_continue(material_compile_mask);
  const bool has_various_connect = material_compile_mask_has_various_connect(material_compile_mask);
  const bool has_plastic = material_compile_mask_has(material_compile_mask, MaterialClass::Plastic);
  const bool has_conductor = material_compile_mask_has_conductor_stage(material_compile_mask);
  const bool has_dielectric = material_compile_mask_has(material_compile_mask, MaterialClass::Dielectric);
  const bool has_connectible_conductor = material_compile_mask_has_connectible_conductor(material_compile_mask);
  const bool has_connectible_dielectric = material_compile_mask_has_connectible_dielectric(material_compile_mask);
  const bool has_thinfilm = material_compile_mask_has(material_compile_mask, MaterialClass::Thinfilm);

  switch (stage) {
    case GPURaytracingRenderer::PipelineStage::InitCameraPath0:
    case GPURaytracingRenderer::PipelineStage::TraceCamera:
    case GPURaytracingRenderer::PipelineStage::CameraSurfaceClassify:
      return enable_camera_path;
    case GPURaytracingRenderer::PipelineStage::CameraContinuePrepareDiffuse:
      return enable_camera_path && has_various_continue;
    case GPURaytracingRenderer::PipelineStage::CameraContinuePreparePlastic:
      return enable_camera_path && has_plastic;
    case GPURaytracingRenderer::PipelineStage::CameraContinuePrepareConductor:
      return enable_camera_path && has_conductor;
    case GPURaytracingRenderer::PipelineStage::CameraContinuePrepareDielectric:
      return enable_camera_path && has_dielectric;
    case GPURaytracingRenderer::PipelineStage::CameraContinuePrepareThinfilm:
      return enable_camera_path && has_thinfilm;
    case GPURaytracingRenderer::PipelineStage::CameraContinueFinalize:
      return enable_camera_path;
    case GPURaytracingRenderer::PipelineStage::CameraDirectLightSample:
    case GPURaytracingRenderer::PipelineStage::CameraDirectLightShadow:
    case GPURaytracingRenderer::PipelineStage::CameraDirectLightAccumulate:
      return enable_connect_to_light;
    case GPURaytracingRenderer::PipelineStage::CameraDirectHitAccumulate:
      return enable_direct_hit;
    case GPURaytracingRenderer::PipelineStage::CameraDirectLightPrepareDiffuse:
      return enable_connect_to_light && has_various_connect;
    case GPURaytracingRenderer::PipelineStage::CameraDirectLightPreparePlastic:
      return enable_connect_to_light && has_plastic;
    case GPURaytracingRenderer::PipelineStage::CameraDirectLightPrepareConductor:
      return enable_connect_to_light && has_connectible_conductor;
    case GPURaytracingRenderer::PipelineStage::CameraDirectLightPrepareDielectric:
      return enable_connect_to_light && has_connectible_dielectric;
    case GPURaytracingRenderer::PipelineStage::CameraConnectLightPrepareDiffuse:
      return enable_connect_vertices && has_various_connect;
    case GPURaytracingRenderer::PipelineStage::CameraConnectLightPreparePlastic:
      return enable_connect_vertices && has_plastic;
    case GPURaytracingRenderer::PipelineStage::CameraConnectLightPrepareConductor:
      return enable_connect_vertices && has_connectible_conductor;
    case GPURaytracingRenderer::PipelineStage::CameraConnectLightPrepareDielectric:
      return enable_connect_vertices && has_connectible_dielectric;
    case GPURaytracingRenderer::PipelineStage::CameraConnectLightResolveDiffuse:
      return enable_connect_vertices && has_various_connect;
    case GPURaytracingRenderer::PipelineStage::CameraConnectLightResolvePlastic:
      return enable_connect_vertices && has_plastic;
    case GPURaytracingRenderer::PipelineStage::CameraConnectLightResolveConductor:
      return enable_connect_vertices && has_connectible_conductor;
    case GPURaytracingRenderer::PipelineStage::CameraConnectLightResolveDielectric:
      return enable_connect_vertices && has_connectible_dielectric;
    case GPURaytracingRenderer::PipelineStage::CameraConnectLightShadow:
      return enable_connect_vertices;
    case GPURaytracingRenderer::PipelineStage::CameraConnectLightClassify:
    case GPURaytracingRenderer::PipelineStage::CameraConnectLightCompact:
      return enable_connect_vertices && (material_compile_mask_connection_queue_count(material_compile_mask) > 1u);
    case GPURaytracingRenderer::PipelineStage::InitLightPath0:
    case GPURaytracingRenderer::PipelineStage::TraceLight:
    case GPURaytracingRenderer::PipelineStage::LightSurfaceClassify:
      return enable_light_path;
    case GPURaytracingRenderer::PipelineStage::LightContinuePrepareDiffuse:
      return enable_light_path && has_various_continue;
    case GPURaytracingRenderer::PipelineStage::LightContinuePreparePlastic:
      return enable_light_path && has_plastic;
    case GPURaytracingRenderer::PipelineStage::LightContinuePrepareConductor:
      return enable_light_path && has_conductor;
    case GPURaytracingRenderer::PipelineStage::LightContinuePrepareDielectric:
      return enable_light_path && has_dielectric;
    case GPURaytracingRenderer::PipelineStage::LightContinuePrepareThinfilm:
      return enable_light_path && has_thinfilm;
    case GPURaytracingRenderer::PipelineStage::LightContinueFinalize:
      return enable_light_path;
    case GPURaytracingRenderer::PipelineStage::LightConnectCameraClear:
      return enable_connect_to_camera;
    case GPURaytracingRenderer::PipelineStage::LightConnectCameraShadow:
    case GPURaytracingRenderer::PipelineStage::LightConnectCameraAccumulate:
      return enable_connect_to_camera;
    case GPURaytracingRenderer::PipelineStage::LightConnectCameraPrepareDiffuse:
      return enable_connect_to_camera && has_various_connect;
    case GPURaytracingRenderer::PipelineStage::LightConnectCameraPreparePlastic:
      return enable_connect_to_camera && has_plastic;
    case GPURaytracingRenderer::PipelineStage::LightConnectCameraPrepareConductor:
      return enable_connect_to_camera && has_connectible_conductor;
    case GPURaytracingRenderer::PipelineStage::LightConnectCameraPrepareDielectric:
      return enable_connect_to_camera && has_connectible_dielectric;
    case GPURaytracingRenderer::PipelineStage::VCMGridClear:
    case GPURaytracingRenderer::PipelineStage::VCMGridBuild:
    case GPURaytracingRenderer::PipelineStage::VCMGridPrefix:
    case GPURaytracingRenderer::PipelineStage::VCMGridScatter:
      return (mode == GPUIntegratorMode::VCM) && enable_merge_vertices;
    case GPURaytracingRenderer::PipelineStage::VCMMergeDiffuse:
      return (((mode == GPUIntegratorMode::VCM) && enable_merge_vertices) && has_various_connect) ||
             (((mode == GPUIntegratorMode::UPBP) && enable_upbp_surface) && has_various_continue);
    case GPURaytracingRenderer::PipelineStage::VCMMergePlastic:
      return (((mode == GPUIntegratorMode::VCM) && enable_merge_vertices) || ((mode == GPUIntegratorMode::UPBP) && enable_upbp_surface)) && has_plastic;
    case GPURaytracingRenderer::PipelineStage::VCMMergeConductor:
      return (((mode == GPUIntegratorMode::VCM) && enable_merge_vertices) || ((mode == GPUIntegratorMode::UPBP) && enable_upbp_surface)) && has_connectible_conductor;
    case GPURaytracingRenderer::PipelineStage::VCMMergeDielectric:
      return (((mode == GPUIntegratorMode::VCM) && enable_merge_vertices) && has_connectible_dielectric) ||
             (((mode == GPUIntegratorMode::UPBP) && enable_upbp_surface) && (has_connectible_dielectric || has_thinfilm));
    case GPURaytracingRenderer::PipelineStage::UPBPClear:
    case GPURaytracingRenderer::PipelineStage::UPBPValidate:
      return mode == GPUIntegratorMode::UPBP;
    case GPURaytracingRenderer::PipelineStage::UPBPDensityCompact:
      return mode == GPUIntegratorMode::UPBP;
    case GPURaytracingRenderer::PipelineStage::UPBPBeamInstances:
      return (mode == GPUIntegratorMode::UPBP) && (enable_upbp_bp2d || enable_upbp_bb1d);
    case GPURaytracingRenderer::PipelineStage::UPBPBeamGridBuild:
      return (mode == GPUIntegratorMode::UPBP) && (enable_upbp_bp2d || enable_upbp_bb1d);
    case GPURaytracingRenderer::PipelineStage::UPBPPP3D:
      return (mode == GPUIntegratorMode::UPBP) && enable_upbp_pp3d;
    case GPURaytracingRenderer::PipelineStage::UPBPPB2D:
      return (mode == GPUIntegratorMode::UPBP) && enable_upbp_pb2d;
    case GPURaytracingRenderer::PipelineStage::UPBPBP2D:
      return (mode == GPUIntegratorMode::UPBP) && enable_upbp_bp2d;
    case GPURaytracingRenderer::PipelineStage::UPBPBB1D:
      return (mode == GPUIntegratorMode::UPBP) && enable_upbp_bb1d;
    case GPURaytracingRenderer::PipelineStage::UPBPDirectHit:
      return mode == GPUIntegratorMode::UPBP;
    case GPURaytracingRenderer::PipelineStage::PrepareSpectralValues:
      return ((mode == GPUIntegratorMode::VCM) || (mode == GPUIntegratorMode::UPBP)) && (spectral_mode == static_cast<uint32_t>(GPUSpectralMode::Spectral));
    case GPURaytracingRenderer::PipelineStage::PrepareSample:
    case GPURaytracingRenderer::PipelineStage::SwapQueues:
    case GPURaytracingRenderer::PipelineStage::FinalizeSample:
    case GPURaytracingRenderer::PipelineStage::BuildDispatchArgs:
      return true;
    default:
      return true;
  }
}

uint32_t build_material_compile_mask(const SceneData& scene_data) {
  uint32_t result = 0u;
  for (const auto& material : scene_data.materials) {
    result |= material_compile_bit(material.cls);
    const float maximum_roughness = std::max(material.roughness.value.x, material.roughness.value.y);
    if ((material.cls == MaterialClass::Conductor) && (maximum_roughness > kDeltaAlphaTreshold)) {
      result |= kMaterialCompileConnectibleConductor;
    }
    if ((material.cls == MaterialClass::Dielectric) && (maximum_roughness > kDeltaAlphaTreshold)) {
      result |= kMaterialCompileConnectibleDielectric;
    }
  }
  return result;
}

bool gpu_material_compile_mask_supported(uint32_t mask) {
  return material_compile_mask_has(mask, MaterialClass::OpenPBR) == false;
}

const char* gpu_material_compile_mask_error_message(uint32_t mask) {
  if (material_compile_mask_has(mask, MaterialClass::OpenPBR)) {
    return "GPU RT does not support OpenPBR materials yet. Use the CPU renderer for scenes containing OpenPBR materials.";
  }
  return "GPU RT does not support one or more scene material classes.";
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

bool build_blue_noise_table_data(uint32_t target_samples, std::vector<uint8_t>& table_data) {
  return ::build_blue_noise_gpu_data(normalize_blue_noise_target_samples(target_samples), table_data);
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
    .instances = kInvalidDescriptorIndex,
    .emitter_profiles = kInvalidDescriptorIndex,
    .emitter_instances = kInvalidDescriptorIndex,
    .scene_globals = kInvalidDescriptorIndex,
    .materials = kInvalidDescriptorIndex,
    .spectrums = kInvalidDescriptorIndex,
    .images = kInvalidDescriptorIndex,
    .mediums = kInvalidDescriptorIndex,
    .emitters_distribution = kInvalidDescriptorIndex,
    .scene_options = kInvalidDescriptorIndex,
    .energy_compensation_interfaces = kInvalidDescriptorIndex,
    .spectral_values = kInvalidDescriptorIndex,
    .spectrum_count = 0u,
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
    const RHIMemoryStats memory_stats = device.get_memory_statistics();
    log::error("GPU RT: failed to create '%s' storage buffer (%u): requested=%llu bytes (%.2fMB), previous=%llu bytes (%.2fMB), device-local=%.2f/%.2fMB",
      (buffer_name != nullptr) ? buffer_name : "unknown", static_cast<uint32_t>(create_result.result), static_cast<unsigned long long>(required_size),
      static_cast<double>(required_size) / (1024.0 * 1024.0), static_cast<unsigned long long>(buffer_size), static_cast<double>(buffer_size) / (1024.0 * 1024.0),
      static_cast<double>(memory_stats.gpu_device_local_allocated_bytes) / (1024.0 * 1024.0), static_cast<double>(memory_stats.gpu_device_local_budget_bytes) / (1024.0 * 1024.0));
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

bool ensure_storage_buffer_capacity(RHIDevice& device, uint64_t required_size, RHIBufferUsage usage, RHIBindlessHandle& buffer, uint64_t& buffer_size, uint32_t& descriptor_index,
  const char* buffer_name) {
  if (buffer.valid() && (buffer_size >= required_size)) {
    descriptor_index = get_bindless_descriptor_index(buffer);
    return true;
  }

  return ensure_storage_buffer(device, required_size, usage, buffer, buffer_size, descriptor_index, buffer_name);
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

bool ensure_host_visible_buffer_capacity(RHIDevice& device, uint64_t required_size, RHIBufferUsage usage, RHIBindlessHandle& buffer, uint64_t& buffer_size,
  uint32_t& descriptor_index, const char* buffer_name) {
  if (buffer.valid() && (buffer_size >= required_size)) {
    descriptor_index = get_bindless_descriptor_index(buffer);
    return true;
  }

  return ensure_host_visible_buffer(device, required_size, usage, buffer, buffer_size, descriptor_index, buffer_name);
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

GPUSceneGlobals build_scene_globals(const SceneData& scene_data, const Camera& camera, const PackedEmitterData& packed_emitters, const BoundingBox& transport_bounds) {
  ETX_PROFILER_SCOPE();

  BoundingBox bbox = {};
  {
    ETX_PROFILER_NAMED_SCOPE("gpu_rt_compute_scene_bounds");
    bbox = scene_data.compute_bounding_volumes();
  }

  const SceneBoundingSphere transport_sphere = compute_transport_bounding_sphere(transport_bounds, camera);

  GPUSceneGlobals globals = {};
  globals.vertex_count = static_cast<uint32_t>(scene_data.vertices.pos.size());
  globals.triangle_count = static_cast<uint32_t>(scene_data.triangles.size());
  globals.mesh_count = static_cast<uint32_t>(scene_data.meshes.size());
  globals.emitter_profile_count = static_cast<uint32_t>(packed_emitters.emitter_profiles.size());
  globals.emitter_instance_count = static_cast<uint32_t>(packed_emitters.emitter_instances.size());
  globals.active_emitter_count = static_cast<uint32_t>(packed_emitters.active_emitter_indices.size());

  globals.bounding_sphere_center = transport_sphere.center;
  globals.bounding_sphere_radius = transport_sphere.radius;
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
  options.path_mode = static_cast<uint32_t>(gpu_integrator_mode_from_scene(scene));
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
  _wavefront_vcm_light_vertex_count = 0u;
  _wavefront_tile_index = 0u;
  _wavefront_tile_max_pixels = 0u;
  _wavefront_tile_count = 1u;
  _wavefront_tile_path_capacity = 0u;
  _upbp_light_path_capacity = 0u;
  _wavefront_tile_base_origin = {};
  _wavefront_tile_base_size = {};
  _wavefront_tile_plan_valid = false;
  return true;
}

void GPURaytracingRenderer::set_wavefront_steps_per_render(uint32_t value) {
  _wavefront_auto_tuning_enabled = false;
  _wavefront_steps_per_render = std::clamp(value, 1u, kWavefrontAutoMaximumSteps);
  _final_wavefront_schedule.steps_per_render = _wavefront_steps_per_render;
  _preview_wavefront_schedule.steps_per_render = _wavefront_steps_per_render;
}

void GPURaytracingRenderer::set_wavefront_auto_tuning(bool value) {
  _wavefront_auto_tuning_enabled = value;
  reset_wavefront_auto_tuning();
}

void GPURaytracingRenderer::set_batch_coarse_progress(bool value) {
  _batch_coarse_progress = value;
}

void GPURaytracingRenderer::reset_render_window() {
  _render_window_origin = {};
  _render_window_size = {};
  _wavefront_vcm_light_vertex_count = 0u;
  _wavefront_tile_index = 0u;
  _wavefront_tile_max_pixels = 0u;
  _wavefront_tile_count = 1u;
  _wavefront_tile_path_capacity = 0u;
  _upbp_light_path_capacity = 0u;
  _wavefront_tile_base_origin = {};
  _wavefront_tile_base_size = {};
  _wavefront_tile_plan_valid = false;
}

void GPURaytracingRenderer::reset_runtime_failure() {
  _runtime_failed = false;
  _runtime_failure_reason.clear();
}

void GPURaytracingRenderer::set_runtime_failure(std::string message) {
  const bool report_failure = (_runtime_failed == false) || (_runtime_failure_reason != message);
  if (_runtime_failed == false) {
    preserve_render_statistics();
    _run_state = RunState::Stopped;
  }
  if (report_failure) {
    _runtime_failure_reason = std::move(message);
    log::error("%s", _runtime_failure_reason.c_str());
  }
  _runtime_failed = true;
  set_preparation_failed(_runtime_failure_reason, "Failed");
}

void GPURaytracingRenderer::init(RHIContext& ctx, SceneRepresentation& scene) {
  ETX_PROFILER_SCOPE();

  if (_initialized == false) {
    Renderer::init(ctx, scene);
  }

  _cleanup_wait_succeeded = false;
  reset_runtime_failure();
  _backend = ctx.device().backend();
  _use_compute_upbp_beam_grid = ctx.capabilities().ray_traversal_class == RHIRayTraversalClass::Compute;
  _kernel_timing_stats.supported = ctx.supports_timestamps() && (ctx.timestamp_query_capacity() >= 2u);
  _kernel_timing_stats.enabled = _kernel_timing_enabled;
  const GPUIntegratorSelection integrator_selection = gpu_integrator_selection_from_scene(scene);
  _integrator_mode = static_cast<uint32_t>(integrator_selection.mode);
  _integrator_features = integrator_selection.features;
  _material_compile_mask = build_material_compile_mask(scene.data());
  _spectral_mode = static_cast<uint32_t>(gpu_spectral_mode(scene.data()));
  _initialized = true;
  _scene_valid = scene.valid();
  _run_state = RunState::Stopped;

  log::info("GPU UPBP beam index: %s", _use_compute_upbp_beam_grid ? "compute grid" : "acceleration structure");

  if (integrator_selection.supported == false) {
    set_runtime_failure(gpu_integrator_selection_error_message(integrator_selection));
    return;
  }
  if (gpu_material_compile_mask_supported(_material_compile_mask) == false) {
    set_runtime_failure(gpu_material_compile_mask_error_message(_material_compile_mask));
    return;
  }

  set_preparation_ready();
}

void GPURaytracingRenderer::reload_shaders(RHIContext& ctx, SceneRepresentation& scene) {
  ETX_PROFILER_SCOPE();
  (void)ctx;
  request_pipeline_preparation(scene, "reload", true);
}

void GPURaytracingRenderer::set_compile_stage_filter(const std::string& value) {
  _compile_stage_filter = value;
}

void GPURaytracingRenderer::set_kernel_timing_enabled(bool value) {
  if (_kernel_timing_enabled == value) {
    return;
  }

  _kernel_timing_enabled = value;
  reset_kernel_timings();
  _preserved_kernel_timing_stats = _kernel_timing_stats;
}

void GPURaytracingRenderer::reset_kernel_timings() {
  for (auto& timing : _kernel_timing_accumulators) {
    timing = {};
  }
  _kernel_timing_stats.kernels.clear();
  _kernel_timing_stats.dropped_dispatch_count = 0u;
  _kernel_timing_stats.captured_sample_count = 0u;
  _kernel_timing_stats.total_ms = 0.0;
  _kernel_timing_stats.capture_elapsed_ms = 0.0;
  _kernel_timing_stats.enabled = _kernel_timing_enabled;
  _kernel_timing_started_at = {};
}

void GPURaytracingRenderer::update_kernel_timing_stats() {
  _kernel_timing_stats.kernels.clear();
  _kernel_timing_stats.total_ms = 0.0;
  for (const auto& timing : _kernel_timing_accumulators) {
    _kernel_timing_stats.total_ms += timing.total_ms;
  }
  _kernel_timing_stats.captured_sample_count = _kernel_timing_accumulators[static_cast<uint32_t>(PipelineStage::FinalizeSample)].dispatch_count;
  if (_kernel_timing_started_at != std::chrono::steady_clock::time_point{}) {
    _kernel_timing_stats.capture_elapsed_ms = elapsed_ms(_kernel_timing_started_at, std::chrono::steady_clock::now());
  }

  const uint32_t stage_count = static_cast<uint32_t>(PipelineStage::Count);
  _kernel_timing_stats.kernels.reserve(stage_count);
  for (uint32_t stage_index = 0u; stage_index < stage_count; ++stage_index) {
    const KernelTimingAccumulator& timing = _kernel_timing_accumulators[stage_index];
    if (timing.dispatch_count == 0u) {
      continue;
    }

    const double percentage = (_kernel_timing_stats.total_ms > 0.0) ? ((timing.total_ms * 100.0) / _kernel_timing_stats.total_ms) : 0.0;
    _kernel_timing_stats.kernels.push_back({
      .name = pipeline_stage_to_string(static_cast<PipelineStage>(stage_index)),
      .dispatch_count = timing.dispatch_count,
      .total_ms = timing.total_ms,
      .average_ms = timing.total_ms / static_cast<double>(timing.dispatch_count),
      .percentage = percentage,
    });
  }

  std::sort(_kernel_timing_stats.kernels.begin(), _kernel_timing_stats.kernels.end(), [](const RendererKernelTiming& lhs, const RendererKernelTiming& rhs) {
    return lhs.total_ms > rhs.total_ms;
  });
}

RendererPreparationStatus GPURaytracingRenderer::preparation_status() const {
  RendererPreparationStatus result = {
    .state = _preparation_state,
    .phase = _preparation_phase,
    .message = _preparation_message,
    .completed_steps = 0u,
    .total_steps = 0u,
  };

  const auto active = _publish_preparation ? _publish_preparation : _active_preparation;
  if ((result.state == RendererPreparationState::Preparing) && active) {
    {
      std::lock_guard lock(active->progress_mutex);
      result.total_steps = active->total_steps;
      result.steps = active->pipeline_progress;
      result.completed_steps = active->completed_compile_groups.load() + active->completed_pipelines.load();
    }
    result.worker_count = active->compile_worker_count.load() + (_pipeline_publish_task ? _pipeline_publish_task->worker_count : 0u);
    result.completed_steps = std::min(result.completed_steps, result.total_steps);
    result.elapsed_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - _preparation_started_at).count();
    result.cancelable = _preparation_canceled == false;
  }

  return result;
}

RendererStatus GPURaytracingRenderer::status() const {
  RendererStatus result = {
    .mode = RendererMode::GPURaytracing,
  };
  result.output_stale = display_texture().valid() && (_sample_index == 0u);

  const bool failed = (_preparation_state == RendererPreparationState::Failed) || _runtime_failed;
  if (failed) {
    result.state = RendererStatusState::Failed;
    result.message = _runtime_failed ? _runtime_failure_reason : _preparation_message;
  }

  if ((failed == false) && (_preparation_state == RendererPreparationState::Preparing)) {
    result.state = RendererStatusState::Preparing;
    result.progress_kind = RendererProgressKind::Steps;
    const auto active = _publish_preparation ? _publish_preparation : _active_preparation;
    if (active) {
      std::lock_guard lock(active->progress_mutex);
      result.total_units = active->total_steps;
      result.completed_units = _publish_preparation ? active->total_compile_groups + active->completed_pipelines.load() : active->completed_compile_groups.load();
      result.completed_units = std::min(result.completed_units, result.total_units);
    }
    result.elapsed_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - _preparation_started_at).count();
    result.elapsed_available = true;
    return result;
  }

  if ((_initialized == false) || (_scene_valid == false)) {
    return result;
  }

  if (failed == false) {
    switch (_run_state) {
      case RunState::Running:
        result.state = RendererStatusState::Running;
        break;
      case RunState::Finishing:
        result.state = RendererStatusState::Finishing;
        break;
      case RunState::Completed:
        result.state = RendererStatusState::Completed;
        break;
      default:
        result.state = RendererStatusState::Idle;
        break;
    }
  }

  result.progress_kind = RendererProgressKind::Samples;
  result.completed_units = _sample_index;
  result.total_units = _last_target_samples;
  const bool upbp_status_visible =
    (static_cast<GPUIntegratorMode>(_integrator_mode) == GPUIntegratorMode::UPBP) &&
    ((result.state == RendererStatusState::Running) || (result.state == RendererStatusState::Finishing) || (result.state == RendererStatusState::Failed)) &&
    _upbp.vertex_buffer.handle.valid();
  if (upbp_status_visible) {
    RendererUPBPStatus& upbp = result.upbp;
    if (_wavefront_render_step == WavefrontRenderStep::FinalizeSample) {
      upbp.phase = RendererUPBPPhase::Finalize;
    } else if (_upbp.camera_phase_started) {
      upbp.phase = (_wavefront_render_step == WavefrontRenderStep::UPBPEvaluateLightBatch) ? RendererUPBPPhase::CameraEvaluation : RendererUPBPPhase::CameraPaths;
    } else if (_wavefront_render_step == WavefrontRenderStep::UPBPEvaluateLightBatch) {
      const bool final_density_batch = (_upbp.density_cache_ready == false) && ((_upbp.light_batch_iteration + 1u) >= _upbp.light_batch_count_total);
      upbp.phase = final_density_batch ? RendererUPBPPhase::DensityIndex : RendererUPBPPhase::LightCompaction;
    } else {
      upbp.phase = RendererUPBPPhase::LightPaths;
    }

    upbp.total_light_batches = _upbp.light_batch_count_total;
    upbp.current_light_batch = upbp.total_light_batches > 0u ? std::min(_upbp.light_batch_iteration + 1u, upbp.total_light_batches) : 0u;
    upbp.total_camera_batches = _upbp.camera_batch_count_total;
    upbp.current_camera_batch = upbp.total_camera_batches > 0u ? std::min(_upbp.camera_batch_index + 1u, upbp.total_camera_batches) : 0u;
    upbp.active_path_count = _upbp.camera_phase_started ? _upbp.camera_batch_count : _upbp.light_batch_count;
    upbp.resident_path_count = std::max(_upbp.resident_light_path_capacity, _upbp.resident_camera_path_capacity);
    upbp.global_path_count = _upbp.global_path_count;
    upbp.density_batch_count = _upbp.density_batch_count;
    upbp.density_cache_ready = _upbp.density_cache_ready;
    const uint32_t density_batch_count = std::min<uint32_t>(_upbp.density_batch_count, static_cast<uint32_t>(_upbp.density_batches.size()));
    for (uint32_t batch_index = 0u; batch_index < density_batch_count; ++batch_index) {
      const UPBPDensityBatchResources& batch = _upbp.density_batches[batch_index];
      upbp.tracking_event_count += batch.event_count;
      upbp.tracking_event_bytes += static_cast<uint64_t>(batch.event_count) * kGPUUPBPTrackingEventStride;
    }

    if (_upbp.density_cache_ready) {
      upbp.surface_point_count = _upbp.density_surface_point_count;
      upbp.medium_point_count = _upbp.density_medium_point_count;
      upbp.bp2d_beam_count = _upbp.density_beam_count;
      upbp.bb1d_beam_count = _upbp.density_bb1d_beam_count;
    } else {
      for (uint32_t batch_index = 0u; batch_index < density_batch_count; ++batch_index) {
        const UPBPDensityBatchResources& batch = _upbp.density_batches[batch_index];
        upbp.surface_point_count += batch.surface_point_count;
        upbp.medium_point_count += batch.medium_point_count;
        upbp.bp2d_beam_count += batch.beam_count;
        upbp.bb1d_beam_count += batch.selected_beam_count;
      }
    }
    upbp.bp2d_partition_count = static_cast<uint32_t>(_upbp.density_bp2d_beam_tlas.size());
    for (const RHIBindlessHandle partition : _upbp.density_bb1d_beam_tlas) {
      upbp.bb1d_partition_count += partition.valid() ? 1u : 0u;
    }
    upbp.gpu_memory_used_bytes = _last_memory_stats.gpu_device_local_allocated_bytes;
    upbp.gpu_memory_budget_bytes = _last_memory_stats.gpu_device_local_budget_bytes;

    if ((upbp.phase == RendererUPBPPhase::CameraPaths) || (upbp.phase == RendererUPBPPhase::CameraEvaluation)) {
      result.path_phase = RendererPathPhase::Camera;
      result.completed_path_count = std::min<uint64_t>(_upbp.camera_batch_offset, _upbp.global_path_count);
      result.total_path_count = _upbp.global_path_count;
    } else if (upbp.phase != RendererUPBPPhase::Finalize) {
      result.path_phase = RendererPathPhase::Light;
      const uint64_t completed_batch_count = _upbp.light_batch_iteration;
      result.completed_path_count = std::min<uint64_t>(completed_batch_count * _upbp.resident_light_path_capacity, _upbp.global_path_count);
      if (_wavefront_render_step == WavefrontRenderStep::UPBPEvaluateLightBatch) {
        result.completed_path_count = std::min<uint64_t>(result.completed_path_count + _upbp.light_batch_count, _upbp.global_path_count);
      }
      result.total_path_count = (_upbp.light_batch_count_total > 1u) ? _upbp.global_path_count : _upbp.light_batch_count;
    }
  }
  result.elapsed_seconds = _preserved_timing_stats_valid ? _preserved_render_elapsed_seconds : _last_render_elapsed_seconds;
  if (_render_timing_active) {
    const auto now = std::chrono::steady_clock::now();
    result.elapsed_seconds = std::chrono::duration<double>(now - _render_started_at).count();
  }
  result.elapsed_available = _render_timing_active || (result.elapsed_seconds > 0.0);
  const double tile_count = static_cast<double>(std::max(1u, _wavefront_tile_count));
  const double tiled_sample_progress = (_wavefront_tile_plan_valid || (_wavefront_tile_count > 1u)) ? (static_cast<double>(_wavefront_tile_index) / tile_count) : 0.0;
  const double completed_sample_count = static_cast<double>(_sample_index) + tiled_sample_progress;
  if (result.state != RendererStatusState::Failed) {
    if ((result.elapsed_seconds > 0.0) && (_last_target_samples > 0u) && (completed_sample_count > 0.0)) {
      const double sample_rate = completed_sample_count / result.elapsed_seconds;
      if (sample_rate > 0.0) {
        const double remaining_samples = std::max(0.0, static_cast<double>(_last_target_samples) - completed_sample_count);
        result.remaining_seconds = remaining_samples / sample_rate;
        result.remaining_available = true;
      }
    } else if ((_last_target_samples > 0u) && (_sample_index >= _last_target_samples)) {
      result.remaining_seconds = 0.0;
      result.remaining_available = true;
    }
  }
  return result;
}

RendererMemoryStats GPURaytracingRenderer::memory_stats() const {
  RendererMemoryStats result = {};
  const auto add_entry = [&result](const char* category, const char* name, RendererMemoryLocation location, uint64_t bytes, uint32_t allocation_count) {
    if (bytes == 0u) {
      return;
    }
    result.entries.push_back({category, name, location, bytes, allocation_count});
  };
  const auto chunked_buffer_bytes = [](const RHIChunkedBufferState& state) {
    uint64_t result_bytes = state.metadata_buffer_size;
    for (const uint64_t chunk_size : state.chunk_buffer_sizes) {
      result_bytes += chunk_size;
    }
    return result_bytes;
  };

  const uint64_t geometry_bytes = _vertex_positions_buffer_size + _vertex_normals_buffer_size + _vertex_tangents_buffer_size + _vertex_bitangents_buffer_size +
                                  _vertex_texcoords_buffer_size + _triangles_buffer_size + _meshes_buffer_size + _instances_buffer_size;
  const uint64_t shading_bytes = _materials_buffer_size + _spectrums_buffer_size + _spectral_values_buffer_size + _energy_compensation_interfaces_buffer_size +
                                 _emitter_profiles_buffer_size + _emitter_instances_buffer_size + _emitters_distribution_buffer_size;
  const uint64_t scene_constants_bytes = _scene_globals_buffer_size + _scene_options_buffer_size + _camera_buffer_size + _wavefront_resources_buffer_size;
  const uint64_t path_state_bytes = _camera_state_buffer_size + _light_state_buffer_size + _camera_hit_buffer_size + _light_hit_buffer_size;
  const uint64_t queue_bytes = _camera_queue_a_buffer_size + _camera_queue_b_buffer_size + _light_queue_a_buffer_size + _light_queue_b_buffer_size + _material_queue_buffer_size +
                               _shadow_queue_buffer_size + _wavefront_dispatch_args_buffer_size;
  const uint64_t direct_light_bytes = _direct_light_sample_buffer_size + _direct_light_task_buffer_size + _direct_light_result_buffer_size;
  const uint64_t connection_bytes = _connect_light_task_buffer_size + _connect_camera_task_buffer_size + _connect_camera_result_buffer_size;
  const uint64_t subsurface_bytes = _camera_subsurface_state_buffer_size + _light_subsurface_state_buffer_size;
  const uint64_t readback_bytes = _camera_queue_count_readback_buffer_size + _light_queue_count_readback_buffer_size + _light_vertex_counter_readback_buffer_size;
  const uint64_t output_bytes = static_cast<uint64_t>(_output_dimensions.x) * static_cast<uint64_t>(_output_dimensions.y) * sizeof(float4);
  const UPBPBuffer* upbp_resident_buffers[] = {&_upbp.resources_buffer, &_upbp.vertex_buffer, &_upbp.segment_buffer, &_upbp.interval_buffer, &_upbp.event_buffer,
    &_upbp.point_buffer, &_upbp.beam_buffer, &_upbp.counter_buffer, &_upbp.path_state_buffer};
  uint64_t upbp_resident_bytes = 0u;
  uint32_t upbp_resident_allocation_count = 0u;
  for (const UPBPBuffer* buffer : upbp_resident_buffers) {
    upbp_resident_bytes += buffer->size;
    upbp_resident_allocation_count += buffer->handle.valid() ? 1u : 0u;
  }
  uint64_t upbp_density_record_bytes = 0u;
  uint32_t upbp_density_record_allocation_count = 0u;
  for (const UPBPDensityBatchResources& batch : _upbp.density_batches) {
    const UPBPBuffer* batch_buffers[] = {&batch.surface_point_buffer, &batch.surface_point_aabb_buffer, &batch.medium_point_buffer, &batch.medium_point_aabb_buffer,
      &batch.beam_buffer, &batch.event_buffer};
    for (const UPBPBuffer* buffer : batch_buffers) {
      upbp_density_record_bytes += buffer->size;
      upbp_density_record_allocation_count += buffer->handle.valid() ? 1u : 0u;
    }
  }
  const UPBPBuffer* upbp_density_index_buffers[] = {&_upbp.density_batch_buffer, &_upbp.density_surface_point_buffer, &_upbp.density_surface_point_aabb_buffer,
    &_upbp.density_medium_point_buffer, &_upbp.density_medium_point_aabb_buffer, &_upbp.density_beam_buffer, &_upbp.density_surface_point_instance_buffer,
    &_upbp.density_medium_point_instance_buffer, &_upbp.density_bp2d_beam_instance_buffer, &_upbp.density_bb1d_beam_instance_buffer, &_upbp.density_bb1d_beam_buffer,
    &_upbp.density_beam_reference_buffer, &_upbp.density_beam_unit_aabb_buffer, &_upbp.density_as_scratch_buffer};
  uint64_t upbp_density_index_bytes = 0u;
  uint32_t upbp_density_index_allocation_count = 0u;
  for (const UPBPBuffer* buffer : upbp_density_index_buffers) {
    upbp_density_index_bytes += buffer->size;
    upbp_density_index_allocation_count += buffer->handle.valid() ? 1u : 0u;
  }
  const uint64_t upbp_bpt_device_bytes = _upbp.bpt_light_vertex_buffer.size + _upbp.bpt_light_path_state_buffer.size;
  const uint32_t upbp_bpt_device_allocation_count = (_upbp.bpt_light_vertex_buffer.handle.valid() ? 1u : 0u) + (_upbp.bpt_light_path_state_buffer.handle.valid() ? 1u : 0u);
  const uint64_t upbp_readback_bytes = _upbp.counter_readback_buffer.size;
  const uint32_t upbp_readback_allocation_count = _upbp.counter_readback_buffer.handle.valid() ? 1u : 0u;

  add_entry("Scene", "Geometry and attributes", RendererMemoryLocation::GPUDevice, geometry_bytes, 7u);
  add_entry("Scene", "Materials, spectra, and emitters", RendererMemoryLocation::GPUDevice, shading_bytes, 6u);
  add_entry("Scene", "Image payload", RendererMemoryLocation::GPUDevice, chunked_buffer_bytes(_images_blob_state),
    static_cast<uint32_t>(_images_blob_state.chunk_buffers.size()) + (_images_blob_state.metadata_buffer.valid() ? 1u : 0u));
  add_entry("Scene", "Medium payload", RendererMemoryLocation::GPUDevice, chunked_buffer_bytes(_mediums_blob_state),
    static_cast<uint32_t>(_mediums_blob_state.chunk_buffers.size()) + (_mediums_blob_state.metadata_buffer.valid() ? 1u : 0u));
  add_entry("Renderer", "Scene and camera constants", RendererMemoryLocation::GPUDevice, scene_constants_bytes, 4u);
  add_entry("Renderer", "Blue-noise samples", RendererMemoryLocation::GPUDevice, _blue_noise_buffer_size, 1u);
  add_entry("Renderer", "Output image", RendererMemoryLocation::GPUDevice, output_bytes, _output_texture.valid() ? 1u : 0u);
  add_entry("Wavefront", "Path states and intersections", RendererMemoryLocation::GPUDevice, path_state_bytes, 4u);
  add_entry("Wavefront", "Queues and indirect arguments", RendererMemoryLocation::GPUDevice, queue_bytes, 7u);
  add_entry("Wavefront", "Camera vertex history", RendererMemoryLocation::GPUDevice, _camera_vertex_buffer_size, 1u);
  add_entry("Wavefront", "Compact light vertex history", RendererMemoryLocation::GPUDevice, _light_vertex_buffer_size, 1u);
  add_entry("Wavefront", "VCM grid ranges", RendererMemoryLocation::GPUDevice, _vcm_grid_heads_buffer_size, _vcm_grid_heads_buffer.valid() ? 1u : 0u);
  add_entry("Wavefront", "VCM grid indices", RendererMemoryLocation::GPUDevice, _vcm_grid_next_buffer_size, _vcm_grid_next_buffer.valid() ? 1u : 0u);
  add_entry("Wavefront", "Fast light endpoint metadata", RendererMemoryLocation::GPUDevice, _fast_light_endpoint_buffer_size, _fast_light_endpoint_buffer.valid() ? 1u : 0u);
  add_entry("Wavefront", "Film and path metadata", RendererMemoryLocation::GPUDevice, _film_buffer_size + _path_meta_buffer_size, 2u);
  const uint32_t direct_light_allocation_count = (_direct_light_sample_buffer.valid() ? 1u : 0u) + (_direct_light_result_buffer.valid() ? 1u : 0u);
  const uint32_t connection_allocation_count =
    (_connect_light_task_buffer.valid() ? 1u : 0u) + (_connect_camera_task_buffer.valid() ? 1u : 0u) + (_connect_camera_result_buffer.valid() ? 1u : 0u);
  add_entry("Wavefront", "Direct-light work", RendererMemoryLocation::GPUDevice, direct_light_bytes, direct_light_allocation_count);
  add_entry("Wavefront", "Vertex-connection work", RendererMemoryLocation::GPUDevice, connection_bytes, connection_allocation_count);
  add_entry("Wavefront", "Subsurface state", RendererMemoryLocation::GPUDevice, subsurface_bytes, 2u);
  add_entry("Wavefront", "Light-history counter", RendererMemoryLocation::GPUDevice, _light_vertex_counter_buffer_size, 1u);
  add_entry("Readback", "Queue and history counters", RendererMemoryLocation::GPUHostVisible, readback_bytes, 3u);
  add_entry("UPBP", "Resident path storage", RendererMemoryLocation::GPUDevice, upbp_resident_bytes, upbp_resident_allocation_count);
  add_entry("UPBP", "Persistent density records and bounds", RendererMemoryLocation::GPUDevice, upbp_density_record_bytes, upbp_density_record_allocation_count);
  add_entry("UPBP", "Density index buffers and build scratch", RendererMemoryLocation::GPUDevice, upbp_density_index_bytes, upbp_density_index_allocation_count);
  add_entry("UPBP", "Resident BPT light history", RendererMemoryLocation::GPUDevice, upbp_bpt_device_bytes, upbp_bpt_device_allocation_count);
  add_entry("UPBP", "Counter readback", RendererMemoryLocation::GPUHostVisible, upbp_readback_bytes, upbp_readback_allocation_count);

  result.wavefront_path_capacity = _wavefront_path_capacity;
  result.light_vertex_capacity = _wavefront_resources.light_vertex_capacity;
  result.light_vertex_count = std::min(_wavefront_light_vertex_reserved_count, result.light_vertex_capacity);
  result.tile_index = _wavefront_tile_index;
  result.tile_count = _wavefront_tile_count;
  result.max_path_length = _wavefront_resources.max_path_length;
  result.max_observed_camera_path_length = _wavefront_max_observed_camera_path_length;
  result.max_observed_light_path_length = _wavefront_max_observed_light_path_length;
  return result;
}

RendererControlState GPURaytracingRenderer::control_state() const {
  RendererControlState result = {};

  const bool preparation_ready = _preparation_state == RendererPreparationState::Ready;
  const bool render_ready = _initialized && _scene_valid && (_runtime_failed == false) && preparation_ready && pipelines_valid();
  const bool render_active = (_run_state == RunState::Running) || (_run_state == RunState::Finishing);
  result.can_run = render_ready && ((_run_state == RunState::Stopped) || (_run_state == RunState::Completed));
  result.can_finish = render_ready && (_run_state == RunState::Running);
  result.can_stop = render_active || (_preparation_state == RendererPreparationState::Preparing);
  result.can_restart = render_ready && (_run_state != RunState::Stopped);
  return result;
}

bool GPURaytracingRenderer::is_running() const {
  return (_run_state == RunState::Running) || (_run_state == RunState::Finishing);
}

void GPURaytracingRenderer::invalidate_output() {
  _display_output_valid = false;
}

void GPURaytracingRenderer::start() {
  start_render(true, true);
}

void GPURaytracingRenderer::start_render(bool invalidate_display_output, bool request_full_scene_update) {
  if ((_initialized == false) || (_scene_valid == false) || _runtime_failed) {
    return;
  }

  _preparation_canceled = false;
  _preserved_timing_stats_valid = false;
  reset_render_progress(preview_pixel_size() == 0u);
  if (invalidate_display_output) {
    invalidate_output();
  }
  _run_state = RunState::Running;
  if (request_full_scene_update) {
    request_scene_update();
  }
}

void GPURaytracingRenderer::reset_render_timing() {
  _render_started_at = {};
  _last_render_elapsed_seconds = 0.0;
  _render_timing_active = false;
}

void GPURaytracingRenderer::reset_render_progress(bool reset_wavefront_schedule) {
  if (_run_state != RunState::Stopped) {
    _preserved_timing_stats_valid = false;
  }
  _frame_index = 0u;
  _sample_index = 0u;
  reset_render_timing();
  reset_kernel_timings();
  _wavefront_render_step = WavefrontRenderStep::InitSample;
  _wavefront_path_iteration = 0u;
  _wavefront_hard_iteration_cap = 0u;
  _wavefront_camera_queue_count = 0u;
  _wavefront_light_queue_count = 0u;
  _wavefront_light_max_path_length = 0u;
  _wavefront_max_observed_camera_path_length = 0u;
  _wavefront_max_observed_light_path_length = 0u;
  _wavefront_connect_light_vertex_length = 0u;
  _wavefront_connect_light_history_bounces = 0u;
  _wavefront_light_vertex_sample_peak_count = 0u;
  _wavefront_light_history_underuse_sample_count = 0u;
  _wavefront_light_history_underuse_peak_count = 0u;
  _wavefront_vcm_light_vertex_count = 0u;
  _wavefront_tile_index = 0u;
  _wavefront_tile_max_pixels = 0u;
  _wavefront_tile_count = 1u;
  _wavefront_tile_path_capacity = 0u;
  _upbp_light_path_capacity = 0u;
  _wavefront_tile_base_origin = {};
  _wavefront_tile_base_size = {};
  _wavefront_tile_plan_valid = false;
  _wavefront_camera_phase_initialized = false;
  if (static_cast<GPUIntegratorMode>(_integrator_mode) == GPUIntegratorMode::UPBP) {
    _upbp.render_reset_pending = _upbp.render_reset_pending || (_upbp.sample_index != ~0u) || _upbp.density_cache_ready || _upbp.camera_phase_started;
    _upbp.light_batch_iteration = 0u;
    _upbp.light_batch_index = 0u;
    _upbp.light_batch_offset = _upbp.camera_batch_offset;
    _upbp.light_batch_count = _upbp.camera_batch_count;
    _upbp.camera_phase_started = false;
  }
  if (reset_wavefront_schedule) {
    reset_active_wavefront_auto_tuning();
  }
}

void GPURaytracingRenderer::restart_render_after_change() {
  if ((_runtime_failed) || (_run_state == RunState::Stopped)) {
    return;
  }

  reset_render_progress(preview_pixel_size() == 0u);
  if ((_run_state == RunState::Completed) || (_run_state == RunState::Finishing)) {
    _run_state = RunState::Running;
  }
}

void GPURaytracingRenderer::reset_wavefront_auto_tuning() {
  if (_wavefront_auto_tuning_enabled) {
    _final_wavefront_schedule = {};
    _preview_wavefront_schedule = {};
    load_active_wavefront_schedule();
  } else {
    _wavefront_last_batch_ms = 0.0;
    _wavefront_smoothed_ms_per_step = 0.0;
    store_active_wavefront_schedule();
  }
}

void GPURaytracingRenderer::reset_active_wavefront_auto_tuning() {
  if (_wavefront_auto_tuning_enabled) {
    WavefrontSchedule& schedule = _preview_wavefront_schedule_active ? _preview_wavefront_schedule : _final_wavefront_schedule;
    schedule = {};
    load_active_wavefront_schedule();
  } else {
    _wavefront_last_batch_ms = 0.0;
    _wavefront_smoothed_ms_per_step = 0.0;
    store_active_wavefront_schedule();
  }
}

void GPURaytracingRenderer::update_wavefront_auto_tuning(uint32_t executed_steps, double elapsed_ms, bool budget_consumed, bool measurement_valid) {
  if ((_wavefront_auto_tuning_enabled == false) || (measurement_valid == false) || (executed_steps == 0u) || (elapsed_ms <= 0.0)) {
    return;
  }

  _wavefront_last_batch_ms = elapsed_ms;
  const double target_ms = _preview_wavefront_schedule_active ? kWavefrontPreviewAutoTargetMs : kWavefrontFinalAutoTargetMs;
  const double lower_dead_zone_ms = _preview_wavefront_schedule_active ? kWavefrontPreviewAutoLowerDeadZoneMs : kWavefrontFinalAutoLowerDeadZoneMs;
  const double upper_dead_zone_ms = _preview_wavefront_schedule_active ? kWavefrontPreviewAutoUpperDeadZoneMs : kWavefrontFinalAutoUpperDeadZoneMs;
  if ((budget_consumed == false) && (elapsed_ms <= upper_dead_zone_ms)) {
    store_active_wavefront_schedule();
    return;
  }

  const double measured_ms_per_step = elapsed_ms / static_cast<double>(executed_steps);
  if (_wavefront_smoothed_ms_per_step == 0.0) {
    _wavefront_smoothed_ms_per_step = measured_ms_per_step;
  } else {
    _wavefront_smoothed_ms_per_step += kWavefrontAutoSmoothingFactor * (measured_ms_per_step - _wavefront_smoothed_ms_per_step);
  }

  const uint32_t estimated_steps = static_cast<uint32_t>(std::clamp(std::floor(target_ms / _wavefront_smoothed_ms_per_step), 1.0, static_cast<double>(kWavefrontAutoMaximumSteps)));
  const uint32_t adjustment_limit = std::max(1u, _wavefront_steps_per_render / kWavefrontAutoAdjustmentDivisor);
  if (elapsed_ms > upper_dead_zone_ms) {
    if (_wavefront_steps_per_render > 1u) {
      const uint32_t minimum_steps = std::max(1u, _wavefront_steps_per_render - adjustment_limit);
      _wavefront_steps_per_render = std::min(_wavefront_steps_per_render - 1u, std::max(estimated_steps, minimum_steps));
    }
    store_active_wavefront_schedule();
    return;
  }

  if ((budget_consumed == false) || (elapsed_ms >= lower_dead_zone_ms)) {
    store_active_wavefront_schedule();
    return;
  }

  const uint32_t maximum_steps = std::min(kWavefrontAutoMaximumSteps, _wavefront_steps_per_render + adjustment_limit);
  _wavefront_steps_per_render = std::max(_wavefront_steps_per_render, std::min(estimated_steps, maximum_steps));
  store_active_wavefront_schedule();
}

void GPURaytracingRenderer::on_preview_mode_changed(bool active) {
  if (_preview_wavefront_schedule_active == active) {
    return;
  }

  store_active_wavefront_schedule();
  if (active && (_preview_wavefront_schedule.smoothed_ms_per_step <= 0.0) && (_final_wavefront_schedule.smoothed_ms_per_step > 0.0)) {
    _preview_wavefront_schedule.smoothed_ms_per_step = _final_wavefront_schedule.smoothed_ms_per_step;
    _preview_wavefront_schedule.steps_per_render = static_cast<uint32_t>(
      std::clamp(std::floor(kWavefrontPreviewAutoTargetMs / _preview_wavefront_schedule.smoothed_ms_per_step), 1.0, static_cast<double>(kWavefrontAutoMaximumSteps)));
  }
  _preview_wavefront_schedule_active = active;
  load_active_wavefront_schedule();
}

void GPURaytracingRenderer::store_active_wavefront_schedule() {
  WavefrontSchedule& schedule = _preview_wavefront_schedule_active ? _preview_wavefront_schedule : _final_wavefront_schedule;
  schedule.steps_per_render = _wavefront_steps_per_render;
  schedule.last_batch_ms = _wavefront_last_batch_ms;
  schedule.smoothed_ms_per_step = _wavefront_smoothed_ms_per_step;
}

void GPURaytracingRenderer::load_active_wavefront_schedule() {
  const WavefrontSchedule& schedule = _preview_wavefront_schedule_active ? _preview_wavefront_schedule : _final_wavefront_schedule;
  _wavefront_steps_per_render = schedule.steps_per_render;
  _wavefront_last_batch_ms = schedule.last_batch_ms;
  _wavefront_smoothed_ms_per_step = schedule.smoothed_ms_per_step;
}

void GPURaytracingRenderer::stop_render_timing() {
  if (_render_timing_active) {
    _last_render_elapsed_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - _render_started_at).count();
    _render_timing_active = false;
  }
}

void GPURaytracingRenderer::preserve_render_statistics() {
  stop_render_timing();
  if (_kernel_timing_enabled) {
    update_kernel_timing_stats();
  }

  const bool live_statistics_available = (_last_render_elapsed_seconds > 0.0) || (_kernel_timing_stats.kernels.empty() == false);
  if (live_statistics_available == false) {
    return;
  }

  _preserved_render_elapsed_seconds = _last_render_elapsed_seconds;
  _preserved_kernel_timing_stats = _kernel_timing_stats;
  _preserved_timing_stats_valid = true;
}

void GPURaytracingRenderer::set_preparation_state(RendererPreparationState state, const char* phase, const std::string& message, uint32_t completed_steps, uint32_t total_steps) {
  _preparation_state = state;
  _preparation_phase = phase ? phase : "";
  _preparation_message = message;
  (void)completed_steps;
  (void)total_steps;
}

void GPURaytracingRenderer::set_preparation_failed(const std::string& message, const char* phase) {
  set_preparation_state(RendererPreparationState::Failed, phase, message);
}

void GPURaytracingRenderer::set_preparation_ready(const char* message) {
  set_preparation_state(RendererPreparationState::Ready, "Ready", message ? std::string(message) : std::string{});
}

void GPURaytracingRenderer::destroy_pipelines(RHIDevice& device) {
  for (uint32_t stage_index = 0u; stage_index < static_cast<uint32_t>(PipelineStage::Count); ++stage_index) {
    auto& pipeline = _pipelines[stage_index];
    if (pipeline.valid()) {
      device.destroy_pipeline(pipeline);
      pipeline = {};
    }
    _pipeline_variant_keys[stage_index] = 0u;
  }
}

void GPURaytracingRenderer::release_inflight_preparation_tasks(bool wait) {
  for (size_t i = 0u; i < _inflight_preparation_tasks.size();) {
    auto& task = _inflight_preparation_tasks[i];
    const bool completed = (wait || scheduler.completed(task.handle));
    if (completed == false) {
      ++i;
      continue;
    }

    if (wait) {
      scheduler.wait_and_release(task.handle);
    } else {
      scheduler.release(task.handle);
    }
    _inflight_preparation_tasks.erase(_inflight_preparation_tasks.begin() + static_cast<ptrdiff_t>(i));
  }
}

void GPURaytracingRenderer::compile_pipeline_preparation(std::shared_ptr<PendingPipelinePreparation> result) {
  if (result == nullptr) {
    return;
  }

  result->compile_started_at = std::chrono::steady_clock::now();
  log::info("GPU RT preparation queued: generation=%u integrator=%s features=0x%08x material_mask=0x%08x spectral_mode=%u filter=%s", result->generation,
    gpu_integrator_mode_to_string(static_cast<GPUIntegratorMode>(result->integrator_mode)), result->integrator_features, result->material_compile_mask, result->spectral_mode,
    result->compile_stage_filter.empty() ? "<all>" : result->compile_stage_filter.c_str());
  log::info("GPU RT preparation background compile started: generation=%u", result->generation);

  auto& compiler = ShaderCompiler::instance();
  struct StageCompileGroup {
    std::string source_file = {};
    std::string optimization_level = {};
    std::string spirv_opt_config = {};
    std::string bsdf_kind = {};
    std::string stage_entry_define = {};
    std::vector<const WavefrontStage*> stages = {};
    std::vector<uint32_t> progress_indices = {};
    ShaderCompiler::MultiShaderCompilationResult compilation = {};
    double compile_time_ms = 0.0;
  };

  std::vector<StageCompileGroup> compile_groups = {};
  auto find_or_add_group = [&](const WavefrontStage& stage_info) -> StageCompileGroup& {
    const std::string source_file = stage_info.source_file ? stage_info.source_file : "";
    const std::string optimization_level = stage_info.optimization_level ? stage_info.optimization_level : "";
    const std::string spirv_opt_config = {};
    const std::string bsdf_kind = stage_info.bsdf_kind ? stage_info.bsdf_kind : "";
    const std::string stage_entry_define = stage_info.uses_stage_entry_define ? std::string(stage_info.entry_point ? stage_info.entry_point : "") : "";
    for (auto& group : compile_groups) {
      if ((group.source_file == source_file) && (group.optimization_level == optimization_level) && (group.spirv_opt_config == spirv_opt_config) &&
          (group.bsdf_kind == bsdf_kind) && (group.stage_entry_define == stage_entry_define)) {
        return group;
      }
    }

    compile_groups.push_back({
      .source_file = source_file,
      .optimization_level = optimization_level,
      .spirv_opt_config = spirv_opt_config,
      .bsdf_kind = bsdf_kind,
      .stage_entry_define = stage_entry_define,
    });
    return compile_groups.back();
  };

  for (const auto& stage_info : kWavefrontStages) {
    if (result->requested_stages.test(static_cast<size_t>(stage_info.stage)) == false) {
      continue;
    }
    if ((result->compile_stage_filter.empty() == false) && (result->compile_stage_filter != stage_info.entry_point)) {
      continue;
    }
    if (result->compile_stage_filter.empty() && (wavefront_stage_enabled(stage_info.stage, static_cast<GPUIntegratorMode>(result->integrator_mode), result->integrator_features,
                                                   result->material_compile_mask, result->spectral_mode) == false)) {
      continue;
    }

    result->compile_filter_matched = true;
    find_or_add_group(stage_info).stages.push_back(&stage_info);
    result->total_pipelines += 1u;
  }

  std::vector<RendererPreparationStepStatus> initial_progress = {};
  initial_progress.reserve(result->total_pipelines);
  for (auto& group : compile_groups) {
    group.progress_indices.reserve(group.stages.size());
    for (const WavefrontStage* stage_info : group.stages) {
      group.progress_indices.push_back(static_cast<uint32_t>(initial_progress.size()));
      std::string detail = stage_info->entry_point;
      if (group.bsdf_kind.empty() == false) {
        detail += " | ";
        detail += bsdf_kind_to_string(group.bsdf_kind);
      }
      initial_progress.push_back({
        .name = pipeline_stage_to_string(stage_info->stage),
        .detail = std::move(detail),
        .state = RendererPreparationStepState::QueuedForShaderCompilation,
      });
    }
  }
  {
    std::lock_guard lock(result->progress_mutex);
    result->total_compile_groups = static_cast<uint32_t>(compile_groups.size());
    result->total_steps = result->total_compile_groups + result->total_pipelines;
    result->pipeline_progress = std::move(initial_progress);
  }

  if ((result->compile_stage_filter.empty() == false) && (result->compile_filter_matched == false)) {
    result->error_message = "GPU RT compile stage filter '" + result->compile_stage_filter + "' did not match any pipeline entry point";
    log::error("%s", result->error_message.c_str());
    result->compile_finished_at = std::chrono::steady_clock::now();
    result->compilation_complete.store(true, std::memory_order_release);
    result->progress_condition.notify_all();
    return;
  }

  result->compiled_stages.resize(result->total_pipelines);
  result->initialization_complete.store(true, std::memory_order_release);
  result->progress_condition.notify_all();

  if (compile_groups.empty()) {
    result->success = true;
    result->compile_finished_at = std::chrono::steady_clock::now();
    result->compilation_complete.store(true, std::memory_order_release);
    result->progress_condition.notify_all();
    log::info("GPU RT preparation background compile finished: generation=%u groups=0 stages=0 wall=0.00ms", result->generation);
    return;
  }

  compiler.reset_statistics();
  const auto compile_begin = std::chrono::steady_clock::now();
  const uint32_t worker_count = std::min<uint32_t>(static_cast<uint32_t>(compile_groups.size()), std::max(1u, scheduler.max_thread_count()));
  result->compile_worker_count.store(worker_count);

  auto compile_group = [&](StageCompileGroup& group) {
    {
      std::lock_guard lock(result->progress_mutex);
      for (const uint32_t progress_index : group.progress_indices) {
        result->pipeline_progress[progress_index].state = RendererPreparationStepState::CompilingSpirV;
      }
    }

    std::vector<ShaderCompiler::ShaderEntryPoint> entry_points = {};
    entry_points.reserve(group.stages.size());
    for (const WavefrontStage* stage_info : group.stages) {
      entry_points.push_back({stage_info->entry_point, RHIShaderStage::Compute});
    }

    ETX_CRITICAL(group.stages.empty() == false);
    std::unordered_map<std::string, std::string> defines =
      wavefront_stage_defines(*group.stages.front(), static_cast<GPUIntegratorMode>(result->integrator_mode), result->material_compile_mask, result->spectral_mode);

    const auto group_compile_begin = std::chrono::steady_clock::now();
    group.compilation = compiler.compile(group.source_file, entry_points, defines, _backend);
    const auto group_compile_end = std::chrono::steady_clock::now();
    group.compile_time_ms = elapsed_ms(group_compile_begin, group_compile_end);
    const bool compile_succeeded = (group.compilation.result == RHIResult::Success) && (group.compilation.binaries.size() == group.stages.size());

    std::vector<CompiledStageBinary> compiled_stages = {};
    if (compile_succeeded) {
      compiled_stages.reserve(group.stages.size());
      for (size_t binary_index = 0u; binary_index < group.stages.size(); ++binary_index) {
        const WavefrontStage& stage_info = *group.stages[binary_index];
        CompiledStageBinary compiled_stage = {
          .stage = stage_info.stage,
          .variant_key = wavefront_stage_variant_key(stage_info, static_cast<GPUIntegratorMode>(result->integrator_mode), result->material_compile_mask, result->spectral_mode),
          .entry_point = stage_info.entry_point ? stage_info.entry_point : "",
          .source_file = stage_info.source_file ? stage_info.source_file : "",
          .optimization_level = group.optimization_level,
          .bsdf_kind = stage_info.bsdf_kind ? stage_info.bsdf_kind : "",
          .uses_stage_entry_define = stage_info.uses_stage_entry_define,
          .blob = {},
          .binary = group.compilation.binaries[binary_index],
        };
        if ((compiled_stage.binary.spirv_data != nullptr) && (compiled_stage.binary.spirv_size > 0u)) {
          const auto* binary_begin = compiled_stage.binary.spirv_data;
          compiled_stage.blob.assign(binary_begin, binary_begin + compiled_stage.binary.spirv_size);
          compiled_stage.binary.spirv_data = compiled_stage.blob.data();
          compiled_stage.binary.spirv_size = compiled_stage.blob.size();
        }
        compiled_stages.push_back(std::move(compiled_stage));
      }
    }

    {
      std::lock_guard lock(result->progress_mutex);
      for (size_t stage_index = 0u; stage_index < group.progress_indices.size(); ++stage_index) {
        const uint32_t progress_index = group.progress_indices[stage_index];
        auto& progress = result->pipeline_progress[progress_index];
        progress.state = compile_succeeded ? RendererPreparationStepState::QueuedForDriver : RendererPreparationStepState::Failed;
        if (compile_succeeded) {
          progress.spirv_size_bytes = group.compilation.binaries[stage_index].spirv_size;
          result->compiled_stages[progress_index] = std::move(compiled_stages[stage_index]);
          result->ready_pipeline_indices.push_back(progress_index);
        }
      }
    }
    result->progress_condition.notify_all();
    result->completed_compile_groups.fetch_add(1u);
  };

  if (worker_count <= 1u) {
    for (auto& group : compile_groups) {
      compile_group(group);
    }
  } else {
    scheduler.execute(uint64_t(compile_groups.size()), [&](uint32_t begin, uint32_t end, uint32_t) {
      for (uint32_t group_index = begin; group_index < end; ++group_index) {
        compile_group(compile_groups[group_index]);
      }
    });
  }
  result->compile_worker_count.store(0u);

  const auto compile_end = std::chrono::steady_clock::now();
  const double compile_wall_time_ms = elapsed_ms(compile_begin, compile_end);

  for (const auto& group : compile_groups) {
    if ((group.compilation.result != RHIResult::Success) || (group.compilation.binaries.size() != group.stages.size())) {
      const char* failing_stage = group.stages.empty() ? "<unknown>" : group.stages.front()->entry_point;
      result->error_message = "GPU shader pipeline compilation failed at '" + std::string(failing_stage) + "'";
      log::error("Failed to compile GPU RT shader group rooted at '%s' after %.2fms: %s", failing_stage, group.compile_time_ms, group.compilation.error_message.c_str());
      compiler.log_statistics("GPU RT wavefront");
      result->compile_finished_at = std::chrono::steady_clock::now();
      result->compilation_complete.store(true, std::memory_order_release);
      result->progress_condition.notify_all();
      return;
    }
  }

  log::info("GPU RT preparation background compile finished: generation=%u stages=%u groups=%u workers=%u wall=%.2fms", result->generation, result->total_pipelines,
    result->total_compile_groups, worker_count, compile_wall_time_ms);
  const char* shader_action = compiler.runtime_compilation_allowed() ? "Compiled" : "Loaded";
  for (const auto& group : compile_groups) {
    if (group.stages.size() == 1u) {
      log::info("%s %s - %.2fms", shader_action, group.stages[0]->entry_point, group.compile_time_ms);
    } else {
      log::info("%s %zu stages from %s - %.2fms", shader_action, group.stages.size(), group.source_file.c_str(), group.compile_time_ms);
    }
  }
  compiler.log_statistics("GPU RT wavefront");

  result->success = true;
  result->compile_finished_at = std::chrono::steady_clock::now();
  result->compilation_complete.store(true, std::memory_order_release);
  result->progress_condition.notify_all();
}

void GPURaytracingRenderer::request_pipeline_preparation(const SceneRepresentation& scene, const char* reason, bool force_reload) {
  if (_initialized == false) {
    return;
  }

  const GPUIntegratorSelection integrator_selection = gpu_integrator_selection_from_scene(scene);
  if (integrator_selection.supported == false) {
    set_runtime_failure(gpu_integrator_selection_error_message(integrator_selection));
    return;
  }

  const uint32_t material_compile_mask = build_material_compile_mask(scene.data());
  if (gpu_material_compile_mask_supported(material_compile_mask) == false) {
    set_runtime_failure(gpu_material_compile_mask_error_message(material_compile_mask));
    return;
  }

  const uint32_t integrator_mode = static_cast<uint32_t>(integrator_selection.mode);
  const uint32_t spectral_mode = static_cast<uint32_t>(gpu_spectral_mode(scene.data()));
  std::bitset<static_cast<size_t>(PipelineStage::Count)> requested_stages = {};
  uint32_t requested_stage_count = 0u;
  for (const auto& stage_info : kWavefrontStages) {
    const bool filter_matches = (_compile_stage_filter.empty() == false) && (_compile_stage_filter == stage_info.entry_point);
    const bool stage_enabled = wavefront_stage_enabled(stage_info.stage, integrator_selection.mode, integrator_selection.features, material_compile_mask, spectral_mode);
    if (((_compile_stage_filter.empty() == false) && (filter_matches == false)) || (_compile_stage_filter.empty() && (stage_enabled == false))) {
      continue;
    }

    const uint32_t stage_index = static_cast<uint32_t>(stage_info.stage);
    const uint64_t variant_key = wavefront_stage_variant_key(stage_info, integrator_selection.mode, material_compile_mask, spectral_mode);
    const bool stage_requires_preparation = force_reload || filter_matches || (_pipelines[stage_index].valid() == false) || (_pipeline_variant_keys[stage_index] != variant_key);
    if (stage_requires_preparation) {
      requested_stages.set(stage_index);
      requested_stage_count += 1u;
    }
  }

  if ((_compile_stage_filter.empty() == false) && (requested_stage_count == 0u)) {
    set_runtime_failure("GPU RT compile stage filter '" + _compile_stage_filter + "' did not match any pipeline entry point");
    return;
  }

  if (_pipeline_publish_task) {
    _preparation_generation += 1u;
    _preparation_canceled = false;
    set_preparation_state(RendererPreparationState::Preparing, "Waiting for pipeline compiler",
      reason ? (std::string("Reload queued (") + reason + ")") : std::string("Reload queued"));
    return;
  }

  reset_runtime_failure();
  _preparation_canceled = false;
  reset_wavefront_auto_tuning();
  reset_render_progress(true);
  if ((_integrator_mode != integrator_mode) || (_integrator_features != integrator_selection.features)) {
    _scene_options_upload_pending = true;
  }
  _integrator_mode = integrator_mode;
  _integrator_features = integrator_selection.features;
  _material_compile_mask = material_compile_mask;
  _spectral_mode = spectral_mode;
  _preparation_generation += 1u;

  if (requested_stage_count == 0u) {
    _active_preparation.reset();
    _publish_preparation.reset();
    _published_pipeline_count = 0u;
    _publish_pipeline_index = 0u;
    _pipeline_publish_logged = false;
    set_preparation_ready("Existing pipelines match the scene configuration");
    log::info("GPU RT preparation reused existing pipelines: generation=%u integrator=%s features=0x%08x material_mask=0x%08x spectral_mode=%u", _preparation_generation,
      gpu_integrator_mode_to_string(integrator_selection.mode), _integrator_features, _material_compile_mask, _spectral_mode);
    return;
  }

  _active_preparation = std::make_shared<PendingPipelinePreparation>();
  _active_preparation->generation = _preparation_generation;
  _active_preparation->integrator_mode = _integrator_mode;
  _active_preparation->integrator_features = _integrator_features;
  _active_preparation->material_compile_mask = _material_compile_mask;
  _active_preparation->spectral_mode = _spectral_mode;
  _active_preparation->requested_stages = requested_stages;
  _active_preparation->compile_stage_filter = _compile_stage_filter;
  _active_preparation->queued_at = std::chrono::steady_clock::now();
  _publish_preparation.reset();
  _published_pipeline_count = 0u;
  _publish_pipeline_index = 0u;
  _pipeline_publish_logged = false;
  _preparation_started_at = _active_preparation->queued_at;
  const char* preparation_phase = ShaderCompiler::instance().runtime_compilation_allowed() ? "Compiling shaders" : "Loading shader package";
  set_preparation_state(RendererPreparationState::Preparing, preparation_phase, reason ? (std::string("Queued (") + reason + ")") : std::string("Queued"));
  _inflight_preparation_tasks.push_back({
    .handle = scheduler.schedule(1u,
      [this, result = _active_preparation](uint32_t, uint32_t, uint32_t) {
        compile_pipeline_preparation(result);
      }),
    .result = _active_preparation,
  });
}

void GPURaytracingRenderer::poll_preparation_tasks(RHIContext& ctx, bool wait_for_active) {
  auto& device = ctx.device();
  auto begin_available_publish = [&]() {
    if ((_publish_preparation == nullptr) && _active_preparation && (_active_preparation->generation == _preparation_generation) &&
        _active_preparation->initialization_complete.load(std::memory_order_acquire)) {
      begin_pipeline_publish(_active_preparation);
    }
  };

  begin_available_publish();
  for (size_t i = 0u; i < _inflight_preparation_tasks.size();) {
    auto& task = _inflight_preparation_tasks[i];
    const bool is_active_generation = task.result && (task.result->generation == _preparation_generation);
    const bool completed = wait_for_active ? (is_active_generation ? true : scheduler.completed(task.handle)) : scheduler.completed(task.handle);
    if (completed == false) {
      ++i;
      continue;
    }

    if (wait_for_active && is_active_generation) {
      scheduler.wait_and_release(task.handle);
    } else {
      scheduler.release(task.handle);
    }

    auto result = task.result;
    _inflight_preparation_tasks.erase(_inflight_preparation_tasks.begin() + static_cast<ptrdiff_t>(i));
    if (result == nullptr) {
      continue;
    }

    if (result->generation != _preparation_generation) {
      log::info("GPU RT preparation discarded stale generation=%u current_generation=%u", result->generation, _preparation_generation);
      continue;
    }

    _active_preparation = result;
    if (result->success == false) {
      set_runtime_failure(result->error_message);
      set_preparation_failed(result->error_message, "Compile failed");
      log::error("GPU RT preparation failed: generation=%u phase=compile message=%s", result->generation, result->error_message.c_str());
      _publish_preparation.reset();
      continue;
    }

    if (_publish_preparation != result) {
      begin_pipeline_publish(result);
    }
  }
  begin_available_publish();
}

bool GPURaytracingRenderer::begin_pipeline_publish(std::shared_ptr<PendingPipelinePreparation> result) {
  if (result == nullptr) {
    return false;
  }

  bool expected_publish_started = false;
  if (result->publish_started.compare_exchange_strong(expected_publish_started, true) == false) {
    return false;
  }

  _publish_preparation = std::move(result);
  _publish_pipeline_index = 0u;
  _published_pipeline_count = 0u;
  _pipeline_publish_logged = false;
  _pipeline_publish_started_at = std::chrono::steady_clock::now();
  _compile_filter_matched = _publish_preparation->compile_filter_matched;
  _publish_preparation->publish_timings.clear();
  _publish_preparation->publish_timings.reserve(_publish_preparation->compiled_stages.size());
  _publish_preparation->completed_pipelines.store(0u);
  set_preparation_state(RendererPreparationState::Preparing, "Creating pipelines");
  return true;
}

bool GPURaytracingRenderer::finish_pipeline_publish_batch(RHIDevice& device, bool wait) {
  if (_pipeline_publish_task == nullptr) {
    return false;
  }

  const bool scheduled = _pipeline_publish_task->handle.data != Task::InvalidHandle;
  if (scheduled) {
    if ((wait == false) && (scheduler.completed(_pipeline_publish_task->handle) == false)) {
      return false;
    }
    if (wait) {
      scheduler.wait_and_release(_pipeline_publish_task->handle);
    } else {
      scheduler.release(_pipeline_publish_task->handle);
    }
  }

  const auto task = std::move(_pipeline_publish_task);
  const bool current_generation = _publish_preparation && (task->preparation == _publish_preparation) && (task->preparation->generation == _preparation_generation);
  if ((current_generation == false) || _preparation_canceled) {
    for (const auto& pipeline_result : task->results) {
      if (pipeline_result.handle.valid()) {
        device.destroy_pipeline(pipeline_result.handle);
      }
    }
    if (_preparation_canceled && current_generation) {
      _publish_preparation.reset();
      set_preparation_failed("Preparation canceled", "Canceled");
    } else if (task->preparation == _publish_preparation) {
      _publish_preparation.reset();
      set_preparation_failed("Starting the queued pipeline reload", "Retrying");
    }
    return true;
  }

  if (task->results.size() != task->pipeline_count) {
    for (const auto& pipeline_result : task->results) {
      if (pipeline_result.handle.valid()) {
        device.destroy_pipeline(pipeline_result.handle);
      }
    }
    set_runtime_failure("GPU pipeline batch creation returned an invalid result count");
    _publish_preparation.reset();
    return true;
  }

  bool batch_success = true;
  for (uint32_t batch_index = 0u; batch_index < task->pipeline_count; ++batch_index) {
    const uint32_t pipeline_index = task->pipeline_indices[batch_index];
    const auto& stage = _publish_preparation->compiled_stages[pipeline_index];
    const auto& pipeline_result = task->results[batch_index];
    _publish_preparation->publish_timings.push_back({
      .stage = stage.stage,
      .entry_point = stage.entry_point,
      .source_file = stage.source_file,
      .optimization_level = stage.optimization_level,
      .bsdf_kind = stage.bsdf_kind,
      .uses_stage_entry_define = stage.uses_stage_entry_define,
      .spirv_size_bytes = stage.binary.spirv_size,
      .elapsed_ms = pipeline_result.elapsed_ms,
    });
    log::info("GPU RT pipeline create: generation=%u index=%u/%u stage=%s entry=%s source=%s material=%s opt=%s entry_define=%s spirv=%lluB cache=%s time=%.2fms",
      _publish_preparation->generation, pipeline_index + 1u, _publish_preparation->total_pipelines, pipeline_stage_to_string(stage.stage), stage.entry_point.c_str(),
      stage.source_file.empty() ? "-" : stage.source_file.c_str(), bsdf_kind_to_string(stage.bsdf_kind), stage.optimization_level.empty() ? "-" : stage.optimization_level.c_str(),
      stage.uses_stage_entry_define ? "yes" : "no", static_cast<unsigned long long>(stage.binary.spirv_size), pipeline_result.cache_hit ? "hit" : "compiled",
      pipeline_result.elapsed_ms);
    if ((pipeline_result.result != RHIResult::Success) || (pipeline_result.handle.valid() == false)) {
      batch_success = false;
    }
  }

  if (batch_success == false) {
    for (const auto& pipeline_result : task->results) {
      if (pipeline_result.handle.valid()) {
        device.destroy_pipeline(pipeline_result.handle);
      }
    }
    const std::string message = "GPU pipeline batch creation failed";
    set_runtime_failure(message);
    set_preparation_failed(message, "Pipeline creation failed");
    const uint32_t first_pipeline = task->pipeline_indices.empty() ? 0u : task->pipeline_indices.front();
    log::error("GPU RT preparation failed: generation=%u phase=pipeline batch_start=%u", _publish_preparation->generation, first_pipeline);
    _publish_preparation.reset();
    return true;
  }

  for (uint32_t batch_index = 0u; batch_index < task->pipeline_count; ++batch_index) {
    const uint32_t pipeline_index = task->pipeline_indices[batch_index];
    const auto& stage = _publish_preparation->compiled_stages[pipeline_index];
    const uint32_t stage_index = static_cast<uint32_t>(stage.stage);
    if (_pipelines[stage_index].valid()) {
      device.destroy_pipeline(_pipelines[stage_index]);
    }
    _pipelines[stage_index] = task->results[batch_index].handle;
    _pipeline_variant_keys[stage_index] = stage.variant_key;
  }
  _publish_pipeline_index += task->pipeline_count;
  _published_pipeline_count += task->pipeline_count;
  log::info("GPU RT pipeline batch: generation=%u pipelines=%u workers=%u wall=%.2fms", _publish_preparation->generation, task->pipeline_count, task->worker_count,
    elapsed_ms(task->started_at, task->finished_at));
  return true;
}

bool GPURaytracingRenderer::advance_pipeline_publish(RHIContext& ctx, uint32_t max_pipelines, bool wait_for_batch) {
  if (_publish_preparation == nullptr) {
    return false;
  }

  auto& device = ctx.device();
  if (_pipeline_publish_task) {
    if (finish_pipeline_publish_batch(device, wait_for_batch) == false) {
      return false;
    }
    if (_publish_preparation == nullptr) {
      return true;
    }
  }

  const bool compilation_complete = _publish_preparation->compilation_complete.load(std::memory_order_acquire);
  if (compilation_complete && _publish_preparation->success && (_published_pipeline_count >= _publish_preparation->total_pipelines)) {
    const auto ready_at = std::chrono::steady_clock::now();
    std::vector<PipelinePublishTiming> slowest_pipelines = _publish_preparation->publish_timings;
    double driver_work_ms = 0.0;
    for (const auto& timing : slowest_pipelines) {
      driver_work_ms += timing.elapsed_ms;
    }
    std::sort(slowest_pipelines.begin(), slowest_pipelines.end(), [](const auto& lhs, const auto& rhs) {
      return lhs.elapsed_ms > rhs.elapsed_ms;
    });
    const size_t slow_pipeline_count = std::min<size_t>(5u, slowest_pipelines.size());
    for (size_t i = 0; i < slow_pipeline_count; ++i) {
      const auto& timing = slowest_pipelines[i];
      log::info("GPU RT pipeline slowest[%zu]: generation=%u stage=%s entry=%s source=%s material=%s opt=%s entry_define=%s spirv=%lluB time=%.2fms", i + 1u,
        _publish_preparation->generation, pipeline_stage_to_string(timing.stage), timing.entry_point.c_str(), timing.source_file.empty() ? "-" : timing.source_file.c_str(),
        bsdf_kind_to_string(timing.bsdf_kind), timing.optimization_level.empty() ? "-" : timing.optimization_level.c_str(), timing.uses_stage_entry_define ? "yes" : "no",
        static_cast<unsigned long long>(timing.spirv_size_bytes), timing.elapsed_ms);
    }
    const double compile_ms = elapsed_ms(_publish_preparation->compile_started_at, _publish_preparation->compile_finished_at);
    const double pipeline_span_ms = elapsed_ms(_pipeline_publish_started_at, ready_at);
    const double total_ms = elapsed_ms(_preparation_started_at, ready_at);
    const double overlapped_work_ms = std::max(0.0, compile_ms + driver_work_ms - total_ms);
    log::info("GPU RT preparation ready: generation=%u compile=%.2fms driver_work=%.2fms overlapped_work=%.2fms pipeline_span=%.2fms total=%.2fms",
      _publish_preparation->generation, compile_ms, driver_work_ms, overlapped_work_ms, pipeline_span_ms, total_ms);
    device.persist_pipeline_cache();
    set_preparation_ready();
    _publish_preparation.reset();
    return true;
  }

  if (_pipeline_publish_logged == false) {
    _pipeline_publish_logged = true;
    log::info("GPU RT preparation pipeline creation started: generation=%u stages=%u", _publish_preparation->generation, _publish_preparation->total_pipelines);
  }

  std::vector<uint32_t> pipeline_indices = {};
  {
    std::unique_lock lock(_publish_preparation->progress_mutex);
    if (wait_for_batch && _publish_preparation->ready_pipeline_indices.empty() && (_publish_preparation->compilation_complete.load(std::memory_order_acquire) == false)) {
      _publish_preparation->progress_condition.wait(lock, [&]() {
        return (_publish_preparation->ready_pipeline_indices.empty() == false) || _publish_preparation->compilation_complete.load(std::memory_order_acquire) ||
               (_publish_preparation->generation != _preparation_generation) || _preparation_canceled;
      });
    }

    if (_publish_preparation->ready_pipeline_indices.empty()) {
      return false;
    }

    const uint32_t available_pipeline_count = static_cast<uint32_t>(_publish_preparation->ready_pipeline_indices.size());
    const uint32_t requested_batch_pipeline_count = std::min(available_pipeline_count, std::max(1u, max_pipelines));
    uint32_t batch_pipeline_count = requested_batch_pipeline_count;
    if (device.backend() == RHIBackend::Vulkan) {
      batch_pipeline_count = std::min(requested_batch_pipeline_count, kVulkanPipelineMaxWorkerCount);
    } else if (device.backend() == RHIBackend::Metal) {
      // Metal creates this batch serially, and large batches can stall its synchronous compiler service.
      batch_pipeline_count = 1u;
    }

    pipeline_indices.assign(_publish_preparation->ready_pipeline_indices.begin(), _publish_preparation->ready_pipeline_indices.begin() + batch_pipeline_count);
    _publish_preparation->ready_pipeline_indices.erase(_publish_preparation->ready_pipeline_indices.begin(),
      _publish_preparation->ready_pipeline_indices.begin() + batch_pipeline_count);
  }

  const uint32_t batch_pipeline_count = static_cast<uint32_t>(pipeline_indices.size());
  uint32_t pipeline_creation_worker_count = 1u;
  if (device.backend() == RHIBackend::Vulkan) {
    const uint32_t cpu_worker_limit = std::max(1u, (scheduler.max_thread_count() + 1u) / 2u);
    const RHIMemoryStats memory_stats = device.get_memory_statistics();
    uint32_t worker_limit = std::min(kVulkanPipelineMaxWorkerCount, cpu_worker_limit);
    if (memory_stats.cpu_system_available_bytes > 0u) {
      const uint64_t memory_worker_limit = std::max(1ull, memory_stats.cpu_system_available_bytes / kVulkanPipelineWorkerMemoryReserve);
      worker_limit = static_cast<uint32_t>(std::min<uint64_t>(worker_limit, memory_worker_limit));
    }
    pipeline_creation_worker_count = std::min(batch_pipeline_count, worker_limit);
  }
  std::vector<RHIComputePipelineDesc> pipeline_descs = {};
  pipeline_descs.reserve(batch_pipeline_count);
  for (const uint32_t pipeline_index : pipeline_indices) {
    const auto& stage = _publish_preparation->compiled_stages[pipeline_index];
    pipeline_descs.push_back(device.make_compute_pipeline_desc(stage.binary));
  }

  const auto publish_task = std::make_shared<InflightPipelinePublishTask>();
  publish_task->preparation = _publish_preparation;
  publish_task->pipeline_indices = pipeline_indices;
  publish_task->pipeline_count = batch_pipeline_count;
  publish_task->worker_count = pipeline_creation_worker_count;
  publish_task->started_at = std::chrono::steady_clock::now();
  const auto progress_callback = [preparation = _publish_preparation, pipeline_indices](uint32_t batch_index, const RHICreatePipelineBatchEntry& pipeline_result) {
    {
      std::lock_guard lock(preparation->progress_mutex);
      auto& progress = preparation->pipeline_progress[pipeline_indices[batch_index]];
      switch (pipeline_result.state) {
        case RHIPipelineBatchProgressState::Queued:
          progress.state = RendererPreparationStepState::QueuedForDriver;
          break;
        case RHIPipelineBatchProgressState::CheckingCache:
          progress.state = RendererPreparationStepState::CheckingCache;
          break;
        case RHIPipelineBatchProgressState::DriverCompiling:
          progress.state = RendererPreparationStepState::DriverCompiling;
          break;
        case RHIPipelineBatchProgressState::Complete:
          progress.state = (pipeline_result.result == RHIResult::Success) ? RendererPreparationStepState::Complete : RendererPreparationStepState::Failed;
          progress.elapsed_ms = pipeline_result.elapsed_ms;
          progress.cache_hit = pipeline_result.cache_hit;
          break;
      }
    }
    if (pipeline_result.state == RHIPipelineBatchProgressState::Complete) {
      preparation->completed_pipelines.fetch_add(1u);
    }
  };

  _pipeline_publish_task = publish_task;
  if (device.backend() == RHIBackend::Vulkan) {
    RHIDevice* const publish_device = &device;
    publish_task->handle = scheduler.schedule(1u, [publish_device, publish_task, pipeline_descs = std::move(pipeline_descs), progress_callback](uint32_t, uint32_t, uint32_t) {
      publish_task->results = publish_device->create_compute_pipelines(pipeline_descs, publish_task->worker_count, progress_callback);
      publish_task->finished_at = std::chrono::steady_clock::now();
    });
    return true;
  }

  publish_task->results = device.create_compute_pipelines(pipeline_descs, pipeline_creation_worker_count, progress_callback);
  publish_task->finished_at = std::chrono::steady_clock::now();
  finish_pipeline_publish_batch(device, true);
  return true;
}

bool GPURaytracingRenderer::create_pipelines_sync(RHIContext& ctx, SceneRepresentation& scene, const char* reason) {
  request_pipeline_preparation(scene, reason, true);
  return finish_preparation(ctx, scene);
}

bool GPURaytracingRenderer::finish_preparation(RHIContext& ctx, SceneRepresentation& scene) {
  (void)scene;
  while (_preparation_state == RendererPreparationState::Preparing) {
    poll_preparation_tasks(ctx, false);
    if (_publish_preparation) {
      advance_pipeline_publish(ctx, std::max(1u, _publish_preparation->total_pipelines), true);
    } else if (_active_preparation) {
      std::unique_lock lock(_active_preparation->progress_mutex);
      _active_preparation->progress_condition.wait(lock, [&]() {
        return _active_preparation->initialization_complete.load(std::memory_order_acquire) || _active_preparation->compilation_complete.load(std::memory_order_acquire) ||
               (_active_preparation->generation != _preparation_generation) || _preparation_canceled;
      });
    }
  }

  return pipelines_valid();
}

void GPURaytracingRenderer::poll_preparation(RHIContext& ctx) {
  poll_preparation_tasks(ctx, false);
  if (_pipeline_publish_task) {
    finish_pipeline_publish_batch(ctx.device(), false);
  }
}

bool GPURaytracingRenderer::pipelines_valid() const {
  if (_preparation_state != RendererPreparationState::Ready) {
    return false;
  }

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

  for (const auto& stage_info : kWavefrontStages) {
    if (wavefront_stage_enabled(stage_info.stage, static_cast<GPUIntegratorMode>(_integrator_mode), _integrator_features, _material_compile_mask, _spectral_mode) == false) {
      continue;
    }
    const uint32_t stage_index = static_cast<uint32_t>(stage_info.stage);
    const uint64_t variant_key = wavefront_stage_variant_key(stage_info, static_cast<GPUIntegratorMode>(_integrator_mode), _material_compile_mask, _spectral_mode);
    if ((_pipelines[stage_index].valid() == false) || (_pipeline_variant_keys[stage_index] != variant_key)) {
      return false;
    }
  }

  return true;
}

void GPURaytracingRenderer::cancel_preparation() {
  if (_preparation_state != RendererPreparationState::Preparing) {
    return;
  }

  if (_pipeline_publish_task) {
    _preparation_canceled = true;
    if (_active_preparation) {
      _active_preparation->progress_condition.notify_all();
    }
    set_preparation_state(RendererPreparationState::Preparing, "Canceling pipelines", "Waiting for the active driver compilation to finish");
    return;
  }

  const auto canceled_preparation = _active_preparation;
  _preparation_generation += 1u;
  _active_preparation.reset();
  _publish_preparation.reset();
  _published_pipeline_count = 0u;
  _publish_pipeline_index = 0u;
  _preparation_canceled = true;
  if (canceled_preparation) {
    canceled_preparation->progress_condition.notify_all();
  }
  log::info("GPU RT preparation canceled: generation=%u", _preparation_generation - 1u);
  set_preparation_failed("Preparation canceled", "Canceled");
}

void GPURaytracingRenderer::stop() {
  cancel_preparation();
  stop_rendering();
}

void GPURaytracingRenderer::stop_rendering() {
  preserve_render_statistics();
  _wavefront_render_step = WavefrontRenderStep::InitSample;
  _wavefront_path_iteration = 0u;
  _wavefront_camera_queue_count = 0u;
  _wavefront_light_queue_count = 0u;
  _wavefront_connect_light_vertex_length = 0u;
  _wavefront_connect_light_history_bounces = 0u;
  _wavefront_vcm_light_vertex_count = 0u;
  _wavefront_camera_phase_initialized = false;
  _run_state = RunState::Stopped;
}

void GPURaytracingRenderer::finish() {
  if (_run_state == RunState::Running) {
    _run_state = RunState::Finishing;
  }
}

void GPURaytracingRenderer::set_sample_limit(uint32_t sample_limit) {
  _sample_limit = sample_limit;
  _last_target_samples = std::max(1u, sample_limit);
}

void GPURaytracingRenderer::restart() {
  start_render(false, false);
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
  destroy_linear_scene_buffer(device, _instances_buffer, _instances_buffer_size, _gpu_scene.instances);
  destroy_linear_scene_buffer(device, _emitter_profiles_buffer, _emitter_profiles_buffer_size, _gpu_scene.emitter_profiles);
  destroy_linear_scene_buffer(device, _emitter_instances_buffer, _emitter_instances_buffer_size, _gpu_scene.emitter_instances);
  destroy_linear_scene_buffer(device, _materials_buffer, _materials_buffer_size, _gpu_scene.materials);
  destroy_linear_scene_buffer(device, _spectrums_buffer, _spectrums_buffer_size, _gpu_scene.spectrums);
  destroy_linear_scene_buffer(device, _energy_compensation_interfaces_buffer, _energy_compensation_interfaces_buffer_size, _gpu_scene.energy_compensation_interfaces);
  device.destroy_chunked_buffer(_images_blob_state);
  _gpu_scene.images = kInvalidDescriptorIndex;
  device.destroy_chunked_buffer(_mediums_blob_state);
  _gpu_scene.mediums = kInvalidDescriptorIndex;
  destroy_linear_scene_buffer(device, _scene_globals_buffer, _scene_globals_buffer_size, _gpu_scene.scene_globals);
  _host_scene_globals = {};
  _host_transport_bounds = {};
  _scene_bounding_sphere_radius = 0.0f;
  destroy_linear_scene_buffer(device, _scene_options_buffer, _scene_options_buffer_size, _gpu_scene.scene_options);
  destroy_linear_scene_buffer(device, _emitters_distribution_buffer, _emitters_distribution_buffer_size, _gpu_scene.emitters_distribution);
}

void GPURaytracingRenderer::destroy_wavefront_buffers(RHIContext& ctx) {
  ETX_PROFILER_SCOPE();

  auto& device = ctx.device();
  destroy_upbp_buffers(device);
  destroy_linear_scene_buffer(device, _wavefront_resources_buffer, _wavefront_resources_buffer_size, _wavefront_resources_buffer_descriptor_index);
  destroy_linear_scene_buffer(device, _camera_state_buffer, _camera_state_buffer_size, _camera_state_buffer_descriptor_index);
  destroy_linear_scene_buffer(device, _light_state_buffer, _light_state_buffer_size, _light_state_buffer_descriptor_index);
  destroy_linear_scene_buffer(device, _camera_hit_buffer, _camera_hit_buffer_size, _camera_hit_buffer_descriptor_index);
  destroy_linear_scene_buffer(device, _light_hit_buffer, _light_hit_buffer_size, _light_hit_buffer_descriptor_index);
  destroy_linear_scene_buffer(device, _camera_queue_a_buffer, _camera_queue_a_buffer_size, _camera_queue_a_buffer_descriptor_index);
  destroy_linear_scene_buffer(device, _camera_queue_b_buffer, _camera_queue_b_buffer_size, _camera_queue_b_buffer_descriptor_index);
  destroy_linear_scene_buffer(device, _light_queue_a_buffer, _light_queue_a_buffer_size, _light_queue_a_buffer_descriptor_index);
  destroy_linear_scene_buffer(device, _light_queue_b_buffer, _light_queue_b_buffer_size, _light_queue_b_buffer_descriptor_index);
  destroy_linear_scene_buffer(device, _material_queue_buffer, _material_queue_buffer_size, _material_queue_buffer_descriptor_index);
  destroy_linear_scene_buffer(device, _shadow_queue_buffer, _shadow_queue_buffer_size, _shadow_queue_buffer_descriptor_index);
  destroy_linear_scene_buffer(device, _wavefront_dispatch_args_buffer, _wavefront_dispatch_args_buffer_size, _wavefront_dispatch_args_buffer_descriptor_index);
  destroy_linear_scene_buffer(device, _camera_queue_count_readback_buffer, _camera_queue_count_readback_buffer_size, _camera_queue_count_readback_buffer_descriptor_index);
  destroy_linear_scene_buffer(device, _light_queue_count_readback_buffer, _light_queue_count_readback_buffer_size, _light_queue_count_readback_buffer_descriptor_index);
  destroy_linear_scene_buffer(device, _camera_vertex_buffer, _camera_vertex_buffer_size, _camera_vertex_buffer_descriptor_index);
  destroy_linear_scene_buffer(device, _light_vertex_buffer, _light_vertex_buffer_size, _light_vertex_buffer_descriptor_index);
  destroy_linear_scene_buffer(device, _fast_light_endpoint_buffer, _fast_light_endpoint_buffer_size, _fast_light_endpoint_buffer_descriptor_index);
  destroy_linear_scene_buffer(device, _light_vertex_counter_buffer, _light_vertex_counter_buffer_size, _light_vertex_counter_buffer_descriptor_index);
  destroy_linear_scene_buffer(device, _light_vertex_counter_readback_buffer, _light_vertex_counter_readback_buffer_size, _light_vertex_counter_readback_buffer_descriptor_index);
  destroy_linear_scene_buffer(device, _vcm_grid_heads_buffer, _vcm_grid_heads_buffer_size, _vcm_grid_heads_buffer_descriptor_index);
  destroy_linear_scene_buffer(device, _vcm_grid_next_buffer, _vcm_grid_next_buffer_size, _vcm_grid_next_buffer_descriptor_index);
  destroy_linear_scene_buffer(device, _film_buffer, _film_buffer_size, _film_buffer_descriptor_index);
  destroy_linear_scene_buffer(device, _path_meta_buffer, _path_meta_buffer_size, _path_meta_buffer_descriptor_index);
  destroy_linear_scene_buffer(device, _direct_light_sample_buffer, _direct_light_sample_buffer_size, _direct_light_sample_buffer_descriptor_index);
  destroy_linear_scene_buffer(device, _direct_light_task_buffer, _direct_light_task_buffer_size, _direct_light_task_buffer_descriptor_index);
  destroy_linear_scene_buffer(device, _direct_light_result_buffer, _direct_light_result_buffer_size, _direct_light_result_buffer_descriptor_index);
  destroy_linear_scene_buffer(device, _connect_light_task_buffer, _connect_light_task_buffer_size, _connect_light_task_buffer_descriptor_index);
  destroy_linear_scene_buffer(device, _connect_camera_task_buffer, _connect_camera_task_buffer_size, _connect_camera_task_buffer_descriptor_index);
  destroy_linear_scene_buffer(device, _connect_camera_result_buffer, _connect_camera_result_buffer_size, _connect_camera_result_buffer_descriptor_index);
  destroy_linear_scene_buffer(device, _camera_subsurface_state_buffer, _camera_subsurface_state_buffer_size, _camera_subsurface_state_buffer_descriptor_index);
  destroy_linear_scene_buffer(device, _light_subsurface_state_buffer, _light_subsurface_state_buffer_size, _light_subsurface_state_buffer_descriptor_index);
  _wavefront_path_capacity = 0u;
  _wavefront_vertex_capacity = 0u;
  _wavefront_allocated_integrator_mode = ~0u;
  _wavefront_allocated_integrator_features = 0u;
  _wavefront_render_step = WavefrontRenderStep::InitSample;
  _wavefront_path_iteration = 0u;
  _wavefront_hard_iteration_cap = 0u;
  _wavefront_camera_queue_count = 0u;
  _wavefront_light_queue_count = 0u;
  _wavefront_light_max_path_length = 0u;
  _wavefront_max_observed_camera_path_length = 0u;
  _wavefront_max_observed_light_path_length = 0u;
  _wavefront_connect_light_vertex_length = 0u;
  _wavefront_connect_light_history_bounces = 0u;
  _wavefront_light_history_capacity_bounces = 0u;
  _wavefront_light_vertex_reserved_count = 0u;
  _wavefront_light_vertex_sample_peak_count = 0u;
  _wavefront_light_history_underuse_sample_count = 0u;
  _wavefront_light_history_underuse_peak_count = 0u;
  _wavefront_vcm_light_vertex_count = 0u;
  _wavefront_resources = {};
  _wavefront_tile_index = 0u;
  _wavefront_tile_max_pixels = 0u;
  _wavefront_tile_count = 1u;
  _wavefront_tile_path_capacity = 0u;
  _upbp_light_path_capacity = 0u;
  _wavefront_tile_base_origin = {};
  _wavefront_tile_base_size = {};
  _wavefront_tile_plan_valid = false;
  _wavefront_camera_phase_initialized = false;
  _wavefront_dispatch_args_buffer_state = RHIResourceState::Undefined;
  _camera_queue_count_readback_state = RHIResourceState::Undefined;
  _light_queue_count_readback_state = RHIResourceState::Undefined;
  _light_vertex_counter_readback_state = RHIResourceState::Undefined;
}

void GPURaytracingRenderer::destroy_upbp_density_cache(RHIDevice& device, bool release_beam_grid_storage) {
  auto destroy = [&device](UPBPBuffer& buffer) {
    destroy_linear_scene_buffer(device, buffer.handle, buffer.size, buffer.descriptor_index);
  };
  auto destroy_grid = [&destroy](UPBPBeamGridResources& grid) {
    destroy(grid.metadata_buffer);
    destroy(grid.cell_offsets_buffer);
    destroy(grid.beam_indices_buffer);
    grid.beam_index_count = 0u;
    grid.metadata_state = RHIResourceState::Undefined;
    grid.cell_offsets_state = RHIResourceState::Undefined;
    grid.beam_indices_state = RHIResourceState::Undefined;
  };
  if (_upbp.density_surface_point_tlas.valid()) {
    device.destroy_acceleration_structure(_upbp.density_surface_point_tlas);
    _upbp.density_surface_point_tlas = {};
  }
  if (_upbp.density_medium_point_tlas.valid()) {
    device.destroy_acceleration_structure(_upbp.density_medium_point_tlas);
    _upbp.density_medium_point_tlas = {};
  }
  for (RHIBindlessHandle& tlas : _upbp.density_bp2d_beam_tlas) {
    if (tlas.valid()) {
      device.destroy_acceleration_structure(tlas);
      tlas = {};
    }
  }
  _upbp.density_bp2d_beam_tlas.clear();
  for (RHIBindlessHandle& tlas : _upbp.density_bb1d_beam_tlas) {
    if (tlas.valid()) {
      device.destroy_acceleration_structure(tlas);
      tlas = {};
    }
  }
  for (RHIBindlessHandle& blas : _upbp.density_surface_point_blas) {
    if (blas.valid()) {
      device.destroy_acceleration_structure(blas);
      blas = {};
    }
  }
  if (_upbp.density_medium_point_blas.valid()) {
    device.destroy_acceleration_structure(_upbp.density_medium_point_blas);
    _upbp.density_medium_point_blas = {};
  }
  _upbp.density_surface_point_tlas_capacity = 0u;
  _upbp.density_medium_point_tlas_capacity = 0u;
  _upbp.density_surface_point_blas_capacities = {};
  _upbp.density_medium_point_blas_capacity = 0u;
  _upbp.density_bp2d_beam_tlas_capacities.clear();
  _upbp.density_bb1d_beam_tlas_capacities = {};
  reset_upbp_density_cache();
  for (UPBPDensityBatchResources& batch : _upbp.density_batches) {
    destroy(batch.surface_point_buffer);
    destroy(batch.surface_point_aabb_buffer);
    destroy(batch.medium_point_buffer);
    destroy(batch.medium_point_aabb_buffer);
    destroy(batch.beam_buffer);
    destroy(batch.event_buffer);
  }
  _upbp.density_batches.clear();
  if (_upbp.density_beam_unit_blas.valid()) {
    device.destroy_acceleration_structure(_upbp.density_beam_unit_blas);
    _upbp.density_beam_unit_blas = {};
  }
  destroy(_upbp.density_batch_buffer);
  destroy(_upbp.density_surface_point_buffer);
  destroy(_upbp.density_surface_point_aabb_buffer);
  destroy(_upbp.density_medium_point_buffer);
  destroy(_upbp.density_medium_point_aabb_buffer);
  destroy(_upbp.density_beam_buffer);
  destroy(_upbp.density_surface_point_instance_buffer);
  destroy(_upbp.density_medium_point_instance_buffer);
  destroy(_upbp.density_bp2d_beam_instance_buffer);
  destroy(_upbp.density_bb1d_beam_instance_buffer);
  destroy(_upbp.density_bb1d_beam_buffer);
  destroy(_upbp.density_beam_reference_buffer);
  _upbp.density_bp2d_beam_grid.beam_index_count = 0u;
  _upbp.density_bb1d_beam_grid.beam_index_count = 0u;
  if (release_beam_grid_storage) {
    destroy(_upbp.density_beam_grid_metadata_readback_buffer);
    destroy(_upbp.density_beam_grid_scratch_buffer);
    destroy_grid(_upbp.density_bp2d_beam_grid);
    destroy_grid(_upbp.density_bb1d_beam_grid);
    _upbp.density_beam_grid_metadata_readback_state = RHIResourceState::Undefined;
    _upbp.density_beam_grid_scratch_state = RHIResourceState::Undefined;
  }
  destroy(_upbp.density_beam_unit_aabb_buffer);
  destroy(_upbp.density_as_scratch_buffer);
}

void GPURaytracingRenderer::reset_upbp_density_cache() {
  _upbp.density_batch_count = 0u;
  _upbp.resources.density_batch_buffer = kInvalidDescriptorIndex;
  _upbp.resources.density_batch_count = 0u;
  _upbp.resources.density_output_beam_instance_buffer = kInvalidDescriptorIndex;
  _upbp.resources.density_output_beam_reference_buffer = kInvalidDescriptorIndex;
  _upbp.resources.density_output_beam_instance_capacity = 0u;
  _upbp.resources.density_output_surface_point_buffer = kInvalidDescriptorIndex;
  _upbp.resources.density_output_surface_point_capacity = 0u;
  _upbp.resources.density_output_medium_point_buffer = kInvalidDescriptorIndex;
  _upbp.resources.density_output_medium_point_capacity = 0u;
  _upbp.resources.density_output_beam_buffer = kInvalidDescriptorIndex;
  _upbp.resources.density_output_beam_capacity = 0u;
  _upbp.density_surface_point_count = 0u;
  _upbp.density_medium_point_count = 0u;
  _upbp.density_beam_count = 0u;
  _upbp.density_bb1d_beam_count = 0u;
  _upbp.resources.density_output_bb1d_beam_instance_buffer = kInvalidDescriptorIndex;
  _upbp.resources.density_output_bb1d_beam_instance_capacity = 0u;
  _upbp.resources.density_beam_acceleration_structure_reference_low = 0u;
  _upbp.resources.density_beam_acceleration_structure_reference_high = 0u;
  _upbp.resources.point_acceleration_structure = kInvalidDescriptorIndex;
  _upbp.resources.medium_point_acceleration_structure = kInvalidDescriptorIndex;
  _upbp.resources.density_output_event_buffer = kInvalidDescriptorIndex;
  _upbp.resources.beam_acceleration_structure = kInvalidDescriptorIndex;
  _upbp.resources.bp2d_beam_grid = {};
  _upbp.resources.bb1d_beam_grid = {};
  for (uint32_t& descriptor_index : _upbp.resources.bp2d_beam_acceleration_structures) {
    descriptor_index = kInvalidDescriptorIndex;
  }
  for (uint32_t& descriptor_index : _upbp.resources.bb1d_partition_acceleration_structures) {
    descriptor_index = kInvalidDescriptorIndex;
  }
  _upbp.resources.bb1d_beam_buffer = kInvalidDescriptorIndex;
  _upbp.resources.beam_reference_buffer = kInvalidDescriptorIndex;
  _upbp.resources.beam_index_mode = _use_compute_upbp_beam_grid ? GPUUPBPBeamIndexMode::ComputeGrid : GPUUPBPBeamIndexMode::AccelerationStructure;
  _upbp.density_cache_ready = false;
}

void GPURaytracingRenderer::bind_upbp_density_cache_resources() {
  const bool ready = _upbp.density_cache_ready;
  _upbp.resources.beam_index_mode = _use_compute_upbp_beam_grid ? GPUUPBPBeamIndexMode::ComputeGrid : GPUUPBPBeamIndexMode::AccelerationStructure;
  _upbp.resources.density_output_beam_instance_buffer = kInvalidDescriptorIndex;
  _upbp.resources.density_output_beam_reference_buffer = kInvalidDescriptorIndex;
  _upbp.resources.density_output_beam_instance_capacity = 0u;
  _upbp.resources.density_beam_acceleration_structure_reference_low = 0u;
  _upbp.resources.density_beam_acceleration_structure_reference_high = 0u;
  _upbp.resources.density_batch_buffer = ready ? _upbp.density_batch_buffer.descriptor_index : kInvalidDescriptorIndex;
  _upbp.resources.density_batch_count = ready ? _upbp.density_batch_count : 0u;
  _upbp.resources.density_output_surface_point_buffer =
    ready && _upbp.density_surface_point_buffer.handle.valid() ? _upbp.density_surface_point_buffer.descriptor_index : kInvalidDescriptorIndex;
  _upbp.resources.density_output_surface_point_capacity = ready ? _upbp.density_surface_point_count : 0u;
  _upbp.resources.density_output_medium_point_buffer =
    ready && _upbp.density_medium_point_buffer.handle.valid() ? _upbp.density_medium_point_buffer.descriptor_index : kInvalidDescriptorIndex;
  _upbp.resources.density_output_medium_point_capacity = ready ? _upbp.density_medium_point_count : 0u;
  _upbp.resources.density_output_beam_buffer = ready && _upbp.density_beam_buffer.handle.valid() ? _upbp.density_beam_buffer.descriptor_index : kInvalidDescriptorIndex;
  _upbp.resources.density_output_beam_capacity = ready ? _upbp.density_beam_count : 0u;
  _upbp.resources.density_output_event_buffer = kInvalidDescriptorIndex;
  _upbp.resources.density_output_bb1d_beam_instance_buffer = kInvalidDescriptorIndex;
  _upbp.resources.density_output_bb1d_beam_instance_capacity = 0u;
  _upbp.resources.point_acceleration_structure = ready && (_upbp.density_surface_point_count > 0u) && _upbp.density_surface_point_tlas.valid()
                                                   ? get_bindless_descriptor_index(_upbp.density_surface_point_tlas)
                                                   : kInvalidDescriptorIndex;
  _upbp.resources.medium_point_acceleration_structure = ready && (_upbp.density_medium_point_count > 0u) && _upbp.density_medium_point_tlas.valid()
                                                          ? get_bindless_descriptor_index(_upbp.density_medium_point_tlas)
                                                          : kInvalidDescriptorIndex;
  for (uint32_t partition_index = 0u; partition_index < kGPUUPBPBP2DPartitionCount; ++partition_index) {
    const RHIBindlessHandle partition_tlas = partition_index < _upbp.density_bp2d_beam_tlas.size() ? _upbp.density_bp2d_beam_tlas[partition_index] : RHIBindlessHandle{};
    _upbp.resources.bp2d_beam_acceleration_structures[partition_index] = ready && partition_tlas.valid() ? get_bindless_descriptor_index(partition_tlas) : kInvalidDescriptorIndex;
  }
  _upbp.resources.bb1d_beam_buffer =
    ready && (_upbp.density_bb1d_beam_count > 0u) && _upbp.density_bb1d_beam_buffer.handle.valid() ? _upbp.density_bb1d_beam_buffer.descriptor_index : kInvalidDescriptorIndex;
  _upbp.resources.beam_reference_buffer = ready && ((_upbp.density_beam_count > 0u) || (_upbp.density_bb1d_beam_count > 0u)) && _upbp.density_beam_reference_buffer.handle.valid()
                                            ? _upbp.density_beam_reference_buffer.descriptor_index
                                            : kInvalidDescriptorIndex;
  const auto bind_beam_grid = [ready](const UPBPBeamGridResources& grid, uint32_t beam_count) {
    GPUUPBPBeamGridResources result = {};
    if (ready && (beam_count > 0u) && (grid.beam_index_count > 0u) && grid.metadata_buffer.handle.valid() && grid.cell_offsets_buffer.handle.valid() &&
        grid.beam_indices_buffer.handle.valid()) {
      result.metadata_buffer = grid.metadata_buffer.descriptor_index;
      result.cell_offsets_buffer = grid.cell_offsets_buffer.descriptor_index;
      result.beam_indices_buffer = grid.beam_indices_buffer.descriptor_index;
      result.beam_count = beam_count;
      result.beam_index_count = grid.beam_index_count;
    }
    return result;
  };
  _upbp.resources.bp2d_beam_grid = bind_beam_grid(_upbp.density_bp2d_beam_grid, _upbp.density_beam_count);
  _upbp.resources.bb1d_beam_grid = bind_beam_grid(_upbp.density_bb1d_beam_grid, _upbp.density_bb1d_beam_count);
  const uint32_t first_bb1d_partition_count = upbp_partition_size(_upbp.density_bb1d_beam_count, 0u, kGPUUPBPBB1DPartitionCount);
  _upbp.resources.beam_acceleration_structure = ready && (first_bb1d_partition_count > 0u) && _upbp.density_bb1d_beam_tlas[0].valid()
                                                  ? get_bindless_descriptor_index(_upbp.density_bb1d_beam_tlas[0])
                                                  : kInvalidDescriptorIndex;
  for (uint32_t partition_index = 1u; partition_index < kGPUUPBPBB1DPartitionCount; ++partition_index) {
    const RHIBindlessHandle partition_tlas = _upbp.density_bb1d_beam_tlas[partition_index];
    const uint32_t partition_count = upbp_partition_size(_upbp.density_bb1d_beam_count, partition_index, kGPUUPBPBB1DPartitionCount);
    _upbp.resources.bb1d_partition_acceleration_structures[partition_index - 1u] =
      ready && (partition_count > 0u) && partition_tlas.valid() ? get_bindless_descriptor_index(partition_tlas) : kInvalidDescriptorIndex;
  }
}

void GPURaytracingRenderer::destroy_upbp_buffers(RHIDevice& device) {
  auto destroy = [&device](UPBPBuffer& buffer) {
    destroy_linear_scene_buffer(device, buffer.handle, buffer.size, buffer.descriptor_index);
  };
  destroy_upbp_density_cache(device, true);
  destroy(_upbp.resources_buffer);
  destroy(_upbp.vertex_buffer);
  destroy(_upbp.segment_buffer);
  destroy(_upbp.interval_buffer);
  destroy(_upbp.event_buffer);
  destroy(_upbp.point_buffer);
  destroy(_upbp.beam_buffer);
  destroy(_upbp.counter_buffer);
  destroy(_upbp.counter_readback_buffer);
  destroy(_upbp.path_state_buffer);
  destroy(_upbp.bpt_light_vertex_buffer);
  destroy(_upbp.bpt_light_path_state_buffer);
  _upbp = {};
}

uint64_t GPURaytracingRenderer::destroy_upbp_resident_path_buffers(RHIDevice& device) {
  uint64_t released_bytes = 0u;
  const auto destroy = [&device, &released_bytes](UPBPBuffer& buffer) {
    released_bytes += buffer.size;
    destroy_linear_scene_buffer(device, buffer.handle, buffer.size, buffer.descriptor_index);
  };
  destroy(_upbp.vertex_buffer);
  destroy(_upbp.segment_buffer);
  destroy(_upbp.interval_buffer);
  destroy(_upbp.event_buffer);
  destroy(_upbp.point_buffer);
  destroy(_upbp.beam_buffer);
  destroy(_upbp.path_state_buffer);
  return released_bytes;
}

uint64_t GPURaytracingRenderer::destroy_upbp_completed_camera_wavefront_buffers(RHIDevice& device) {
  uint64_t released_bytes = 0u;
  const auto destroy = [&device, &released_bytes](RHIBindlessHandle& buffer, uint64_t& size, uint32_t& descriptor_index) {
    released_bytes += size;
    destroy_linear_scene_buffer(device, buffer, size, descriptor_index);
  };
  destroy(_camera_state_buffer, _camera_state_buffer_size, _camera_state_buffer_descriptor_index);
  destroy(_camera_hit_buffer, _camera_hit_buffer_size, _camera_hit_buffer_descriptor_index);
  destroy(_camera_queue_a_buffer, _camera_queue_a_buffer_size, _camera_queue_a_buffer_descriptor_index);
  destroy(_camera_queue_b_buffer, _camera_queue_b_buffer_size, _camera_queue_b_buffer_descriptor_index);
  destroy(_camera_queue_count_readback_buffer, _camera_queue_count_readback_buffer_size, _camera_queue_count_readback_buffer_descriptor_index);
  destroy(_camera_vertex_buffer, _camera_vertex_buffer_size, _camera_vertex_buffer_descriptor_index);
  destroy(_camera_subsurface_state_buffer, _camera_subsurface_state_buffer_size, _camera_subsurface_state_buffer_descriptor_index);

  _wavefront_resources.camera_state_buffer = kInvalidDescriptorIndex;
  _wavefront_resources.camera_hit_buffer = kInvalidDescriptorIndex;
  _wavefront_resources.camera_queue_a_buffer = kInvalidDescriptorIndex;
  _wavefront_resources.camera_queue_b_buffer = kInvalidDescriptorIndex;
  _wavefront_resources.camera_vertex_buffer = kInvalidDescriptorIndex;
  _wavefront_resources.camera_subsurface_state_buffer = kInvalidDescriptorIndex;
  _wavefront_resources.camera_vertex_capacity = 0u;
  _wavefront_resources.camera_fixed_max_bounces = 0u;
  _camera_queue_count_readback_state = RHIResourceState::Undefined;
  _wavefront_camera_phase_initialized = false;
  return released_bytes;
}

uint64_t GPURaytracingRenderer::destroy_upbp_completed_light_wavefront_buffers(RHIDevice& device) {
  uint64_t released_bytes = 0u;
  const auto destroy = [&device, &released_bytes](RHIBindlessHandle& buffer, uint64_t& size, uint32_t& descriptor_index) {
    released_bytes += size;
    destroy_linear_scene_buffer(device, buffer, size, descriptor_index);
  };
  destroy(_light_state_buffer, _light_state_buffer_size, _light_state_buffer_descriptor_index);
  destroy(_light_hit_buffer, _light_hit_buffer_size, _light_hit_buffer_descriptor_index);
  destroy(_light_queue_a_buffer, _light_queue_a_buffer_size, _light_queue_a_buffer_descriptor_index);
  destroy(_light_queue_b_buffer, _light_queue_b_buffer_size, _light_queue_b_buffer_descriptor_index);
  destroy(_light_queue_count_readback_buffer, _light_queue_count_readback_buffer_size, _light_queue_count_readback_buffer_descriptor_index);
  destroy(_light_vertex_buffer, _light_vertex_buffer_size, _light_vertex_buffer_descriptor_index);
  destroy(_fast_light_endpoint_buffer, _fast_light_endpoint_buffer_size, _fast_light_endpoint_buffer_descriptor_index);
  destroy(_light_vertex_counter_buffer, _light_vertex_counter_buffer_size, _light_vertex_counter_buffer_descriptor_index);
  destroy(_light_vertex_counter_readback_buffer, _light_vertex_counter_readback_buffer_size, _light_vertex_counter_readback_buffer_descriptor_index);
  destroy(_light_subsurface_state_buffer, _light_subsurface_state_buffer_size, _light_subsurface_state_buffer_descriptor_index);
  destroy(_connect_camera_task_buffer, _connect_camera_task_buffer_size, _connect_camera_task_buffer_descriptor_index);
  destroy(_connect_camera_result_buffer, _connect_camera_result_buffer_size, _connect_camera_result_buffer_descriptor_index);

  _wavefront_resources.light_state_buffer = kInvalidDescriptorIndex;
  _wavefront_resources.light_hit_buffer = kInvalidDescriptorIndex;
  _wavefront_resources.light_queue_a_buffer = kInvalidDescriptorIndex;
  _wavefront_resources.light_queue_b_buffer = kInvalidDescriptorIndex;
  _wavefront_resources.light_vertex_buffer = kInvalidDescriptorIndex;
  _wavefront_resources.light_subsurface_state_buffer = kInvalidDescriptorIndex;
  _wavefront_resources.light_vertex_counter_buffer = kInvalidDescriptorIndex;
  _wavefront_resources.fast_light_endpoint_buffer = kInvalidDescriptorIndex;
  _wavefront_resources.connect_camera_task_buffer = kInvalidDescriptorIndex;
  _wavefront_resources.connect_camera_result_buffer = kInvalidDescriptorIndex;
  _wavefront_resources.light_vertex_capacity = 0u;
  _wavefront_resources.light_fixed_max_bounces = 0u;
  _light_queue_count_readback_state = RHIResourceState::Undefined;
  _light_vertex_counter_readback_state = RHIResourceState::Undefined;
  _wavefront_light_history_capacity_bounces = 0u;
  return released_bytes;
}

bool GPURaytracingRenderer::ensure_upbp_buffers(RHIContext& ctx, const SceneRepresentation& scene, uint32_t global_path_count, uint32_t resident_light_capacity,
  uint32_t resident_camera_capacity, uint32_t camera_batch_index, uint32_t camera_batch_offset, uint32_t camera_batch_count) {
  if ((global_path_count == 0u) || (resident_light_capacity == 0u) || (resident_camera_capacity == 0u)) {
    return false;
  }

  UPBPOptions options = {};
  const auto settings = scene.integrator_data().settings.find(Integrator::Type::UPBP);
  if (settings != scene.integrator_data().settings.end()) {
    options.load(settings->second);
  }
  std::string validation_reason = {};
  if (upbp_options_valid(options, validation_reason) == false) {
    log::error("GPU UPBP: %s", validation_reason.c_str());
    return false;
  }

  const uint32_t maximum_path_length = std::max(1u, scene.data().options.max_path_length);
  const bool merge_vertices_enabled = (scene.data().options.strategy_flags & Scene::Strategy::MergeVertices) != 0u;
  const uint32_t storage_technique_mask = upbp_effective_technique_mask(options, merge_vertices_enabled);
  if (storage_technique_mask == 0u) {
    log::error("GPU UPBP: no techniques remain enabled after applying the scene strategy controls");
    return false;
  }
  resident_light_capacity = std::min(global_path_count, resident_light_capacity);
  resident_camera_capacity = std::min(global_path_count, resident_camera_capacity);
  if ((resident_light_capacity > _wavefront_path_capacity) || (resident_camera_capacity > _wavefront_path_capacity)) {
    log::error("GPU UPBP: resident path capacity exceeds the allocated wavefront capacity (light=%u camera=%u wavefront=%u)", resident_light_capacity, resident_camera_capacity,
      _wavefront_path_capacity);
    return false;
  }
  if ((camera_batch_count == 0u) || (camera_batch_count > resident_camera_capacity) || ((static_cast<uint64_t>(camera_batch_offset) + camera_batch_count) > global_path_count)) {
    log::error("GPU UPBP: camera batch [%u, %u) is outside the resident or global path population", camera_batch_offset, camera_batch_offset + camera_batch_count);
    return false;
  }
  auto& device = ctx.device();
  const bool sample_changed = _upbp.sample_index != _sample_index;
  const bool global_path_population_changed = _upbp.global_path_count != global_path_count;
  const bool storage_layout_changed =
    _upbp.vertex_buffer.handle.valid() &&
    ((_upbp.maximum_path_length != maximum_path_length) || (_upbp.maximum_boundary_count != options.maximum_boundary_count) || (_upbp.technique_mask != storage_technique_mask));
  const bool density_cache_invalid = (sample_changed && (_upbp.sample_index != ~0u)) || (global_path_population_changed && (_upbp.global_path_count > 0u));
  if (density_cache_invalid || storage_layout_changed) {
    const RHIResult reclaim_result = ctx.wait_idle();
    if (reclaim_result != RHIResult::Success) {
      log::error("GPU UPBP: failed to reclaim the previous resident layout (%u)", static_cast<uint32_t>(reclaim_result));
      return false;
    }
    if (storage_layout_changed) {
      destroy_upbp_buffers(device);
    } else {
      destroy_upbp_density_cache(device, false);
    }
  }

  const uint64_t light_vertex_capacity = static_cast<uint64_t>(resident_light_capacity) * (static_cast<uint64_t>(maximum_path_length) + 1u);
  const uint64_t camera_vertex_capacity = static_cast<uint64_t>(resident_camera_capacity) * (static_cast<uint64_t>(maximum_path_length) + 1u);
  const uint64_t light_segment_capacity = static_cast<uint64_t>(resident_light_capacity) * maximum_path_length;
  const uint64_t camera_segment_capacity = static_cast<uint64_t>(resident_camera_capacity) * maximum_path_length;
  const uint64_t light_interval_capacity = static_cast<uint64_t>(resident_light_capacity) * (maximum_path_length + options.maximum_boundary_count);
  const uint64_t camera_interval_capacity = static_cast<uint64_t>(resident_camera_capacity) * (maximum_path_length + options.maximum_boundary_count);
  const bool collect_points =
    (storage_technique_mask & (static_cast<uint32_t>(UPBPTechnique::Surface) | static_cast<uint32_t>(UPBPTechnique::PP3D) | static_cast<uint32_t>(UPBPTechnique::PB2D))) != 0u;
  const bool collect_beams = (storage_technique_mask & (static_cast<uint32_t>(UPBPTechnique::BP2D) | static_cast<uint32_t>(UPBPTechnique::BB1D))) != 0u;
  const bool track_camera_events = (storage_technique_mask & (static_cast<uint32_t>(UPBPTechnique::PB2D) | static_cast<uint32_t>(UPBPTechnique::BB1D))) != 0u;
  const uint64_t light_event_capacity = collect_beams ? light_interval_capacity * kUPBPInitialTrackingEventsPerInterval : 0u;
  const uint64_t camera_event_capacity = track_camera_events ? camera_interval_capacity * kUPBPInitialTrackingEventsPerInterval : 0u;
  const uint64_t vertex_capacity = std::max(light_vertex_capacity, camera_vertex_capacity);
  const uint64_t segment_capacity = std::max(light_segment_capacity, camera_segment_capacity);
  const uint64_t interval_capacity = std::max(light_interval_capacity, camera_interval_capacity);
  const uint64_t event_capacity = std::max(light_event_capacity, camera_event_capacity);
  const uint64_t point_capacity = collect_points ? static_cast<uint64_t>(resident_light_capacity) * maximum_path_length : 0u;
  const uint64_t beam_capacity = collect_beams ? static_cast<uint64_t>(resident_light_capacity) * maximum_path_length : 0u;
  const uint64_t path_state_capacity = std::max<uint64_t>(resident_light_capacity, resident_camera_capacity);

  const struct Capacity {
    const char* name;
    uint64_t count;
    uint32_t stride;
  } capacities[] = {
    {"vertex", vertex_capacity, kGPUUPBPVertexStride},
    {"segment", segment_capacity, kGPUUPBPSegmentStride},
    {"interval", interval_capacity, kGPUUPBPIntervalStride},
    {"event", event_capacity, kGPUUPBPTrackingEventStride},
    {"point", point_capacity, kGPUUPBPPointStride},
    {"beam", beam_capacity, kGPUUPBPBeamStride},
    {"path state", path_state_capacity, kGPUUPBPPathStateStride},
  };
  for (const Capacity& capacity : capacities) {
    if ((capacity.count > std::numeric_limits<uint32_t>::max()) || ((capacity.count * capacity.stride) > kWavefrontMaxAddressableBufferSize)) {
      log::error("GPU UPBP: %s storage exceeds the shader-addressable range", capacity.name);
      return false;
    }
  }

  const RHIBufferUsage storage_usage = RHIBufferUsage::Storage | RHIBufferUsage::TransferDst | RHIBufferUsage::TransferSrc;
  const auto ensure = [&device, storage_usage](UPBPBuffer& buffer, uint64_t size, const char* name) {
    return ensure_storage_buffer_capacity(device, size, storage_usage, buffer.handle, buffer.size, buffer.descriptor_index, name);
  };
  if ((ensure(_upbp.vertex_buffer, vertex_capacity * kGPUUPBPVertexStride, "upbp_vertex") == false) ||
      (ensure(_upbp.segment_buffer, segment_capacity * kGPUUPBPSegmentStride, "upbp_segment") == false) ||
      (ensure(_upbp.interval_buffer, interval_capacity * kGPUUPBPIntervalStride, "upbp_interval") == false) ||
      (ensure(_upbp.event_buffer, std::max<uint64_t>(1u, event_capacity) * kGPUUPBPTrackingEventStride, "upbp_event") == false) ||
      (ensure(_upbp.point_buffer, std::max<uint64_t>(1u, point_capacity) * kGPUUPBPPointStride, "upbp_point") == false) ||
      (ensure(_upbp.beam_buffer, std::max<uint64_t>(1u, beam_capacity) * kGPUUPBPBeamStride, "upbp_beam") == false) ||
      (ensure(_upbp.counter_buffer, static_cast<uint64_t>(GPUUPBPCounterIndex::Count) * sizeof(uint32_t), "upbp_counters") == false) ||
      (ensure(_upbp.path_state_buffer, path_state_capacity * kGPUUPBPPathStateStride, "upbp_path_state") == false) ||
      (ensure(_upbp.resources_buffer, kGPUUPBPResourcesStride, "upbp_resources") == false)) {
    return false;
  }
  const uint64_t counter_size = static_cast<uint64_t>(GPUUPBPCounterIndex::Count) * sizeof(uint32_t);
  if (ensure_host_visible_buffer(device, counter_size, RHIBufferUsage::TransferDst, _upbp.counter_readback_buffer.handle, _upbp.counter_readback_buffer.size,
        _upbp.counter_readback_buffer.descriptor_index, "upbp_counter_readback") == false) {
    return false;
  }
  const bool reset_batches = (_upbp.global_path_count != global_path_count) || sample_changed || (_upbp.resident_light_path_capacity != resident_light_capacity) ||
                             (_upbp.resident_camera_path_capacity != resident_camera_capacity);
  const bool reset_camera_batch =
    reset_batches || (_upbp.camera_batch_index != camera_batch_index) || (_upbp.camera_batch_offset != camera_batch_offset) || (_upbp.camera_batch_count != camera_batch_count);
  _upbp.resident_light_path_capacity = resident_light_capacity;
  _upbp.resident_camera_path_capacity = resident_camera_capacity;
  _upbp.maximum_path_length = maximum_path_length;
  _upbp.maximum_boundary_count = options.maximum_boundary_count;
  _upbp.technique_mask = storage_technique_mask;
  _upbp.global_path_count = global_path_count;
  _upbp.sample_index = _sample_index;
  const bool density_techniques_enabled = (storage_technique_mask & kUPBPDensityTechniqueMask) != 0u;
  const bool light_splats_enabled = gpu_integrator_feature_enabled(_integrator_features, GPUIntegratorFeatures::ConnectToCamera);
  const bool trace_global_light_population = (density_techniques_enabled && (_upbp.density_cache_ready == false)) || (light_splats_enabled && (camera_batch_index == 0u));
  _upbp.light_batch_count_total = trace_global_light_population ? upbp_light_batch_count(global_path_count, resident_light_capacity, camera_batch_offset, camera_batch_count) : 1u;
  _upbp.camera_batch_count_total = divide_round_up(global_path_count, resident_camera_capacity);
  if (reset_camera_batch) {
    _upbp.camera_batch_index = camera_batch_index;
    _upbp.camera_batch_offset = camera_batch_offset;
    _upbp.camera_batch_count = camera_batch_count;
    _upbp.light_batch_iteration = 0u;
    _upbp.light_batch_index = 0u;
    _upbp.light_batch_offset = camera_batch_offset;
    _upbp.light_batch_count = camera_batch_count;
    _upbp.camera_phase_started = false;
  }
  _upbp.resources = {};
  _upbp.resources.vertex_buffer = _upbp.vertex_buffer.descriptor_index;
  _upbp.resources.segment_buffer = _upbp.segment_buffer.descriptor_index;
  _upbp.resources.interval_buffer = _upbp.interval_buffer.descriptor_index;
  _upbp.resources.event_buffer = _upbp.event_buffer.descriptor_index;
  _upbp.resources.point_buffer = _upbp.point_buffer.descriptor_index;
  _upbp.resources.beam_buffer = _upbp.beam_buffer.descriptor_index;
  bind_upbp_density_cache_resources();
  _upbp.resources.bpt_light_vertex_buffer = _upbp.bpt_light_vertex_buffer.handle.valid() ? _upbp.bpt_light_vertex_buffer.descriptor_index : kInvalidDescriptorIndex;
  _upbp.resources.bpt_light_path_state_buffer = _upbp.bpt_light_path_state_buffer.handle.valid() ? _upbp.bpt_light_path_state_buffer.descriptor_index : kInvalidDescriptorIndex;
  _upbp.resources.counter_buffer = _upbp.counter_buffer.descriptor_index;
  _upbp.resources.path_state_buffer = _upbp.path_state_buffer.descriptor_index;
  _upbp.resources.light_vertex_capacity = static_cast<uint32_t>(light_vertex_capacity);
  _upbp.resources.camera_vertex_capacity = static_cast<uint32_t>(camera_vertex_capacity);
  _upbp.resources.light_segment_capacity = static_cast<uint32_t>(light_segment_capacity);
  _upbp.resources.camera_segment_capacity = static_cast<uint32_t>(camera_segment_capacity);
  _upbp.resources.light_interval_capacity = static_cast<uint32_t>(light_interval_capacity);
  _upbp.resources.camera_interval_capacity = static_cast<uint32_t>(camera_interval_capacity);
  _upbp.resources.light_event_capacity = static_cast<uint32_t>(light_event_capacity);
  _upbp.resources.camera_event_capacity = static_cast<uint32_t>(camera_event_capacity);
  _upbp.resources.point_capacity = static_cast<uint32_t>(point_capacity);
  _upbp.resources.beam_capacity = static_cast<uint32_t>(beam_capacity);
  _upbp.resources.light_path_state_capacity = resident_light_capacity;
  _upbp.resources.camera_path_state_capacity = resident_camera_capacity;
  _upbp.counter_readback_state = RHIResourceState::Undefined;
  return update_upbp_iteration_resources(device, scene, global_path_count);
}

bool GPURaytracingRenderer::update_upbp_iteration_resources(RHIDevice& device, const SceneRepresentation& scene, uint32_t global_path_count) {
  UPBPOptions options = {};
  const auto settings = scene.integrator_data().settings.find(Integrator::Type::UPBP);
  if (settings != scene.integrator_data().settings.end()) {
    options.load(settings->second);
  }
  const SceneData& scene_data = scene.data();
  const SpectralQuery spect =
    scene_data.options.properties[Scene::Properties::Spectral] ? SpectralQuery::progressive_sample(_sample_index, scene_data.options.random_seed) : SpectralQuery::sample();
  const UPBPIterationParameters parameters = upbp_iteration_parameters(options, _scene_bounding_sphere_radius, scene.camera().film_size, spect,
    (scene_data.options.strategy_flags & Scene::Strategy::MergeVertices) != 0u, global_path_count, _sample_index);
  GPUUPBPIteration& iteration = _upbp.resources.iteration;
  iteration = {};
  iteration.technique_mask = parameters.mis.enabled_techniques;
  iteration.kernel = static_cast<uint32_t>(options.kernel);
  iteration.sample_index = _sample_index;
  iteration.global_camera_path_count = global_path_count;
  iteration.global_light_path_count = global_path_count;
  iteration.bb1d_light_path_count = static_cast<uint32_t>(parameters.bb1d_light_subpath_count);
  iteration.light_batch_offset = _upbp.light_batch_offset;
  iteration.light_batch_count = _upbp.light_batch_count;
  iteration.camera_batch_offset = _upbp.camera_batch_offset;
  iteration.camera_batch_count = _upbp.camera_batch_count;
  iteration.maximum_null_events_per_interval = options.maximum_null_events_per_interval;
  iteration.surface_radius = static_cast<float>(parameters.surface_radius);
  iteration.pp3d_radius = static_cast<float>(parameters.pp3d_radius);
  iteration.pb2d_radius = static_cast<float>(parameters.pb2d_radius);
  iteration.bp2d_radius = static_cast<float>(parameters.bp2d_radius);
  iteration.bb1d_radius = static_cast<float>(parameters.bb1d_radius);
  iteration.beam_selection_probability = options.beam_selection_probability;
  iteration.bpt_sample_count = static_cast<float>(parameters.bpt_sample_count);
  iteration.maximum_boundary_count = options.maximum_boundary_count;
  for (uint32_t index = 0u; index < 6u; ++index) {
    iteration.technique_factors[index] = static_cast<float>(parameters.mis.technique_factors[index]);
  }
  if ((_upbp.light_batch_iteration == 0u) || _upbp.camera_phase_started) {
    iteration.flags |= GPUUPBPIterationFlags::EvaluateCameraIndependentTerms;
  }
  if (_upbp.density_cache_ready == false) {
    iteration.flags |= GPUUPBPIterationFlags::CollectLightDensityRecords;
  }
  if (scene_data.options.properties[Scene::Properties::MultipleImportanceSampling]) {
    iteration.flags |= GPUUPBPIterationFlags::MultipleImportanceSampling;
  }
  const RHIResult result = device.update_buffer(_upbp.resources_buffer.handle, &_upbp.resources, sizeof(_upbp.resources));
  if (result != RHIResult::Success) {
    log::error("GPU UPBP: failed to update iteration resources (%u)", static_cast<uint32_t>(result));
    return false;
  }
  return true;
}

bool GPURaytracingRenderer::ensure_wavefront_buffers(RHIContext& ctx, const SceneRepresentation& scene, uint32_t path_capacity, uint32_t active_path_capacity,
  bool allow_light_history_shrink) {
  ETX_PROFILER_SCOPE();

  const uint2 film_size = scaled_render_dimensions(scene.camera().film_size);
  if ((film_size.x == 0u) || (film_size.y == 0u)) {
    return false;
  }

  const uint64_t film_pixel_count_u64 = static_cast<uint64_t>(film_size.x) * static_cast<uint64_t>(film_size.y);
  if (film_pixel_count_u64 > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
    log::error("GPU RT: wavefront film capacity overflow");
    return false;
  }

  if (path_capacity == 0u) {
    log::error("GPU RT: wavefront path capacity is zero");
    return false;
  }

  const bool upbp_mode = static_cast<GPUIntegratorMode>(_integrator_mode) == GPUIntegratorMode::UPBP;
  const bool enable_camera_path = gpu_integrator_feature_enabled(_integrator_features, GPUIntegratorFeatures::CameraPath);
  const bool enable_light_path = gpu_integrator_feature_enabled(_integrator_features, GPUIntegratorFeatures::LightPath);
  const bool enable_connect_to_light = gpu_integrator_feature_enabled(_integrator_features, GPUIntegratorFeatures::ConnectToLight);
  const bool enable_connect_to_camera = gpu_integrator_feature_enabled(_integrator_features, GPUIntegratorFeatures::ConnectToCamera);
  const bool enable_connect_vertices = gpu_integrator_feature_enabled(_integrator_features, GPUIntegratorFeatures::ConnectVertices);
  const bool enable_merge_vertices = gpu_integrator_feature_enabled(_integrator_features, GPUIntegratorFeatures::MergeVertices) && (upbp_mode == false);
  const bool upbp_camera_only_phase = upbp_mode && (_upbp.sample_index == _sample_index) && (_upbp.camera_batch_index == _wavefront_tile_index) && _upbp.camera_phase_started;
  const bool allocate_camera_path = enable_camera_path && ((upbp_mode == false) || upbp_camera_only_phase);
  const bool allocate_light_path = enable_light_path && (upbp_camera_only_phase == false);
  const bool allocate_connect_to_camera = enable_connect_to_camera && (upbp_camera_only_phase == false);
  const bool store_complete_light_history = (enable_connect_vertices || enable_merge_vertices) && (upbp_mode == false);
  bool scene_has_subsurface_material = false;
  for (const auto& material : scene.data().materials) {
    if (material.subsurface_cls != SubsurfaceMaterial::Disabled) {
      scene_has_subsurface_material = true;
      break;
    }
  }
  const bool enable_subsurface_state_buffers = scene_has_subsurface_material;
  const uint32_t scene_max_path_length = std::max(1u, scene.data().options.max_path_length);
  const uint32_t camera_history_bounces = allocate_camera_path ? kWavefrontRollingHistoryBounces : 0u;
  const uint32_t initial_light_history_bounces = wavefront_initial_light_history_bounces(scene_max_path_length);
  const uint32_t retained_light_history_bounces = std::min(scene_max_path_length, std::max(initial_light_history_bounces, _wavefront_light_history_capacity_bounces));
  const bool use_fast_light_endpoints = allocate_light_path && (static_cast<GPUIntegratorMode>(_integrator_mode) == GPUIntegratorMode::BDPTFast);
  const uint32_t rolling_light_history_bounces = use_fast_light_endpoints ? kWavefrontFastLightHistoryBounces : kWavefrontLightHistoryBounces;
  const uint32_t light_history_bounces = allocate_light_path ? (store_complete_light_history ? retained_light_history_bounces : rolling_light_history_bounces) : 0u;
  const bool compact_light_history = allocate_light_path && store_complete_light_history;
  const uint32_t wavefront_hard_iteration_cap = scene_max_path_length;
  const uint64_t camera_vertex_capacity_u64 = allocate_camera_path ? static_cast<uint64_t>(path_capacity) * static_cast<uint64_t>(camera_history_bounces + 1u) : 0u;
  if (camera_vertex_capacity_u64 > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
    log::error("GPU RT: wavefront camera vertex capacity overflow");
    return false;
  }
  uint64_t light_vertex_capacity_u64 = allocate_light_path ? static_cast<uint64_t>(path_capacity) * static_cast<uint64_t>(light_history_bounces + 1u) : 0u;
  if (compact_light_history && (allow_light_history_shrink == false)) {
    light_vertex_capacity_u64 = std::max(light_vertex_capacity_u64, static_cast<uint64_t>(_wavefront_resources.light_vertex_capacity));
  }
  if (light_vertex_capacity_u64 > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
    log::error("GPU RT: wavefront light vertex capacity overflow");
    return false;
  }
  const uint32_t camera_vertex_capacity = static_cast<uint32_t>(camera_vertex_capacity_u64);
  const uint32_t light_vertex_capacity = static_cast<uint32_t>(light_vertex_capacity_u64);

  const uint64_t queue_buffer_size = kGPUWavefrontQueueHeaderSize + static_cast<uint64_t>(path_capacity) * sizeof(uint32_t);
  const uint64_t material_queue_buffer_size = queue_buffer_size * kGPUWavefrontMaterialQueueCount;
  const uint64_t shadow_queue_buffer_size = (kGPUWavefrontQueueHeaderSize + static_cast<uint64_t>(path_capacity) * sizeof(uint32_t)) +
                                            (kGPUWavefrontQueueHeaderSize + static_cast<uint64_t>(path_capacity) * kGPUWavefrontConnectDispatchArgsCount * sizeof(uint32_t)) +
                                            (kGPUWavefrontQueueHeaderSize + static_cast<uint64_t>(path_capacity) * sizeof(uint32_t));
  const uint32_t heavy_continuation_chunk_count = 1u + ((path_capacity - 1u) / kGPUWavefrontHeavyContinuationChunkSize);
  const uint64_t dispatch_args_buffer_size =
    kGPUWavefrontFixedDispatchArgsBufferSize + static_cast<uint64_t>(heavy_continuation_chunk_count) * 2u * kGPUWavefrontDispatchArgsStride;
  const uint64_t path_state_buffer_size = static_cast<uint64_t>(path_capacity) * kGPUWavefrontPathStateStride;
  const uint64_t hit_buffer_size = static_cast<uint64_t>(path_capacity) * kGPUWavefrontHitStride;
  const uint64_t camera_vertex_buffer_size = static_cast<uint64_t>(camera_vertex_capacity) * kGPUWavefrontPathVertexStride;
  const uint64_t light_vertex_buffer_size = static_cast<uint64_t>(light_vertex_capacity) * kGPUWavefrontLightPathVertexStride;
  const uint32_t vcm_grid_head_count = enable_merge_vertices ? wavefront_vcm_grid_head_count(light_vertex_capacity) : 0u;
  const uint64_t vcm_grid_heads_buffer_size = wavefront_vcm_grid_storage_size(vcm_grid_head_count);
  const uint64_t vcm_grid_next_buffer_size = enable_merge_vertices ? static_cast<uint64_t>(light_vertex_capacity) * sizeof(uint32_t) : 0u;
  const uint64_t fast_light_endpoint_buffer_size = use_fast_light_endpoints ? static_cast<uint64_t>(path_capacity) * kGPUWavefrontFastLightEndpointStride : 0u;
  const uint64_t film_buffer_size = film_pixel_count_u64 * sizeof(float4);
  const uint64_t path_meta_buffer_size = static_cast<uint64_t>(path_capacity) * kGPUWavefrontPathMetaStride;
  const uint64_t direct_light_sample_buffer_size = static_cast<uint64_t>(path_capacity) * kGPUWavefrontDirectLightSampleStride;
  const uint64_t direct_light_task_buffer_size = static_cast<uint64_t>(path_capacity) * kGPUWavefrontDirectLightTaskStride;
  const uint64_t direct_light_work_buffer_size = std::max(direct_light_sample_buffer_size, direct_light_task_buffer_size);
  const uint64_t direct_light_result_buffer_size = static_cast<uint64_t>(path_capacity) * kGPUWavefrontDirectLightResultStride;
  const uint64_t connect_light_task_count = static_cast<uint64_t>(path_capacity) * static_cast<uint64_t>(kGPUWavefrontConnectDispatchArgsCount);
  const uint64_t connect_queue_size =
    (material_compile_mask_connection_queue_count(_material_compile_mask) > 1u) ? (kGPUWavefrontConnectQueueHeaderSize + connect_light_task_count * 2ull * sizeof(uint32_t)) : 0u;
  const uint64_t connect_light_task_buffer_size =
    connect_light_task_count * kGPUWavefrontConnectLightTaskStride + static_cast<uint64_t>(path_capacity) * 2ull * sizeof(uint32_t) + connect_queue_size;
  const uint64_t connect_camera_task_buffer_size = static_cast<uint64_t>(path_capacity) * kGPUWavefrontConnectCameraTaskStride;
  const uint64_t connect_camera_result_buffer_size = static_cast<uint64_t>(path_capacity) * kGPUWavefrontConnectCameraResultStride;
  const uint64_t subsurface_state_buffer_size = static_cast<uint64_t>(path_capacity) * kGPUWavefrontSubsurfaceStateStride;
  const RHIBufferUsage wavefront_usage = RHIBufferUsage::Storage | RHIBufferUsage::TransferDst;
  const RHIBufferUsage light_vertex_usage = wavefront_usage | RHIBufferUsage::TransferSrc;
  const RHIBufferUsage queue_buffer_usage = RHIBufferUsage::Storage | RHIBufferUsage::TransferDst | RHIBufferUsage::TransferSrc;
  const RHIBufferUsage dispatch_args_buffer_usage = RHIBufferUsage::Storage | RHIBufferUsage::Indirect;
  const RHIBufferUsage queue_readback_usage = RHIBufferUsage::TransferDst;

  const auto validate_wavefront_buffer_size = [](const char* buffer_name, uint64_t size) {
    if (size > kWavefrontMaxAddressableBufferSize) {
      log::error("GPU RT: '%s' wavefront buffer exceeds 32-bit shader byte-address range (%llu bytes)", buffer_name, size);
      return false;
    }
    return true;
  };

  bool wavefront_buffer_sizes_valid =
    (validate_wavefront_buffer_size("wavefront_film", film_buffer_size) && validate_wavefront_buffer_size("wavefront_path_meta", path_meta_buffer_size) &&
      validate_wavefront_buffer_size("wavefront_material_queue", material_queue_buffer_size) && validate_wavefront_buffer_size("wavefront_shadow_queue", shadow_queue_buffer_size));
  if (allocate_camera_path) {
    wavefront_buffer_sizes_valid = wavefront_buffer_sizes_valid && validate_wavefront_buffer_size("wavefront_camera_state", path_state_buffer_size) &&
                                   validate_wavefront_buffer_size("wavefront_camera_hit", hit_buffer_size) &&
                                   validate_wavefront_buffer_size("wavefront_camera_queue", queue_buffer_size) &&
                                   validate_wavefront_buffer_size("wavefront_camera_vertex", camera_vertex_buffer_size);
  }
  if (allocate_light_path) {
    wavefront_buffer_sizes_valid = wavefront_buffer_sizes_valid && validate_wavefront_buffer_size("wavefront_light_state", path_state_buffer_size) &&
                                   validate_wavefront_buffer_size("wavefront_light_hit", hit_buffer_size) &&
                                   validate_wavefront_buffer_size("wavefront_light_queue", queue_buffer_size) &&
                                   validate_wavefront_buffer_size("wavefront_light_vertex", light_vertex_buffer_size);
  }
  if (enable_merge_vertices) {
    wavefront_buffer_sizes_valid = wavefront_buffer_sizes_valid && validate_wavefront_buffer_size("wavefront_vcm_grid_heads", vcm_grid_heads_buffer_size) &&
                                   validate_wavefront_buffer_size("wavefront_vcm_grid_next", vcm_grid_next_buffer_size);
  }
  if (use_fast_light_endpoints) {
    wavefront_buffer_sizes_valid = wavefront_buffer_sizes_valid && validate_wavefront_buffer_size("wavefront_fast_light_endpoint", fast_light_endpoint_buffer_size);
  }
  if (enable_connect_to_light) {
    wavefront_buffer_sizes_valid = wavefront_buffer_sizes_valid && validate_wavefront_buffer_size("wavefront_direct_light_sample", direct_light_sample_buffer_size) &&
                                   validate_wavefront_buffer_size("wavefront_direct_light_task", direct_light_task_buffer_size) &&
                                   validate_wavefront_buffer_size("wavefront_direct_light_result", direct_light_result_buffer_size);
  }
  if (enable_connect_vertices) {
    wavefront_buffer_sizes_valid = wavefront_buffer_sizes_valid && validate_wavefront_buffer_size("wavefront_connect_light_task", connect_light_task_buffer_size);
  }
  if (allocate_connect_to_camera) {
    wavefront_buffer_sizes_valid = wavefront_buffer_sizes_valid && validate_wavefront_buffer_size("wavefront_connect_camera_task", connect_camera_task_buffer_size) &&
                                   validate_wavefront_buffer_size("wavefront_connect_camera_result", connect_camera_result_buffer_size);
  }
  if (enable_subsurface_state_buffers) {
    wavefront_buffer_sizes_valid = wavefront_buffer_sizes_valid && validate_wavefront_buffer_size("wavefront_subsurface_state", subsurface_state_buffer_size);
  }
  if (wavefront_buffer_sizes_valid == false) {
    return false;
  }

  auto& device = ctx.device();
  if (compact_light_history) {
    if (ensure_storage_buffer(device, kWavefrontLightVertexCounterSize, queue_buffer_usage, _light_vertex_counter_buffer, _light_vertex_counter_buffer_size,
          _light_vertex_counter_buffer_descriptor_index, "wavefront_light_vertex_counter") == false) {
      return false;
    }
    const bool recreate_counter_readback =
      (_light_vertex_counter_readback_buffer.valid() == false) || (_light_vertex_counter_readback_buffer_size != kWavefrontLightVertexCounterSize);
    if (ensure_host_visible_buffer(device, kWavefrontLightVertexCounterSize, queue_readback_usage, _light_vertex_counter_readback_buffer,
          _light_vertex_counter_readback_buffer_size, _light_vertex_counter_readback_buffer_descriptor_index, "wavefront_light_vertex_counter_readback") == false) {
      return false;
    }
    if (recreate_counter_readback) {
      _light_vertex_counter_readback_state = RHIResourceState::Undefined;
    }
  } else {
    destroy_linear_scene_buffer(device, _light_vertex_counter_buffer, _light_vertex_counter_buffer_size, _light_vertex_counter_buffer_descriptor_index);
    destroy_linear_scene_buffer(device, _light_vertex_counter_readback_buffer, _light_vertex_counter_readback_buffer_size, _light_vertex_counter_readback_buffer_descriptor_index);
    _light_vertex_counter_readback_state = RHIResourceState::Undefined;
  }
  if (allocate_camera_path) {
    if (ensure_storage_buffer_capacity(device, path_state_buffer_size, wavefront_usage, _camera_state_buffer, _camera_state_buffer_size, _camera_state_buffer_descriptor_index,
          "wavefront_camera_state") == false) {
      return false;
    }
  } else {
    destroy_linear_scene_buffer(device, _camera_state_buffer, _camera_state_buffer_size, _camera_state_buffer_descriptor_index);
  }
  if (allocate_light_path) {
    if (ensure_storage_buffer_capacity(device, path_state_buffer_size, wavefront_usage, _light_state_buffer, _light_state_buffer_size, _light_state_buffer_descriptor_index,
          "wavefront_light_state") == false) {
      return false;
    }
  } else {
    destroy_linear_scene_buffer(device, _light_state_buffer, _light_state_buffer_size, _light_state_buffer_descriptor_index);
  }
  if (allocate_camera_path) {
    if (ensure_storage_buffer_capacity(device, hit_buffer_size, wavefront_usage, _camera_hit_buffer, _camera_hit_buffer_size, _camera_hit_buffer_descriptor_index,
          "wavefront_camera_hit") == false) {
      return false;
    }
  } else {
    destroy_linear_scene_buffer(device, _camera_hit_buffer, _camera_hit_buffer_size, _camera_hit_buffer_descriptor_index);
  }
  if (allocate_light_path) {
    if (ensure_storage_buffer_capacity(device, hit_buffer_size, wavefront_usage, _light_hit_buffer, _light_hit_buffer_size, _light_hit_buffer_descriptor_index,
          "wavefront_light_hit") == false) {
      return false;
    }
  } else {
    destroy_linear_scene_buffer(device, _light_hit_buffer, _light_hit_buffer_size, _light_hit_buffer_descriptor_index);
  }
  if (allocate_camera_path) {
    if (ensure_storage_buffer_capacity(device, queue_buffer_size, queue_buffer_usage, _camera_queue_a_buffer, _camera_queue_a_buffer_size, _camera_queue_a_buffer_descriptor_index,
          "wavefront_camera_queue_a") == false) {
      return false;
    }
  } else {
    destroy_linear_scene_buffer(device, _camera_queue_a_buffer, _camera_queue_a_buffer_size, _camera_queue_a_buffer_descriptor_index);
  }
  if (allocate_camera_path) {
    if (ensure_storage_buffer_capacity(device, queue_buffer_size, queue_buffer_usage, _camera_queue_b_buffer, _camera_queue_b_buffer_size, _camera_queue_b_buffer_descriptor_index,
          "wavefront_camera_queue_b") == false) {
      return false;
    }
  } else {
    destroy_linear_scene_buffer(device, _camera_queue_b_buffer, _camera_queue_b_buffer_size, _camera_queue_b_buffer_descriptor_index);
  }
  if (allocate_light_path) {
    if (ensure_storage_buffer_capacity(device, queue_buffer_size, queue_buffer_usage, _light_queue_a_buffer, _light_queue_a_buffer_size, _light_queue_a_buffer_descriptor_index,
          "wavefront_light_queue_a") == false) {
      return false;
    }
  } else {
    destroy_linear_scene_buffer(device, _light_queue_a_buffer, _light_queue_a_buffer_size, _light_queue_a_buffer_descriptor_index);
  }
  if (allocate_light_path) {
    if (ensure_storage_buffer_capacity(device, queue_buffer_size, queue_buffer_usage, _light_queue_b_buffer, _light_queue_b_buffer_size, _light_queue_b_buffer_descriptor_index,
          "wavefront_light_queue_b") == false) {
      return false;
    }
  } else {
    destroy_linear_scene_buffer(device, _light_queue_b_buffer, _light_queue_b_buffer_size, _light_queue_b_buffer_descriptor_index);
  }
  if (ensure_storage_buffer_capacity(device, material_queue_buffer_size, wavefront_usage, _material_queue_buffer, _material_queue_buffer_size,
        _material_queue_buffer_descriptor_index, "wavefront_material_queue") == false) {
    return false;
  }
  if (ensure_storage_buffer_capacity(device, shadow_queue_buffer_size, wavefront_usage, _shadow_queue_buffer, _shadow_queue_buffer_size, _shadow_queue_buffer_descriptor_index,
        "wavefront_shadow_queue") == false) {
    return false;
  }
  const bool recreate_dispatch_args_buffer = (_wavefront_dispatch_args_buffer.valid() == false) || (_wavefront_dispatch_args_buffer_size < dispatch_args_buffer_size);
  if (ensure_storage_buffer_capacity(device, dispatch_args_buffer_size, dispatch_args_buffer_usage, _wavefront_dispatch_args_buffer, _wavefront_dispatch_args_buffer_size,
        _wavefront_dispatch_args_buffer_descriptor_index, "wavefront_dispatch_args") == false) {
    return false;
  }
  if (recreate_dispatch_args_buffer) {
    _wavefront_dispatch_args_buffer_state = RHIResourceState::Undefined;
  }
  if (allocate_camera_path) {
    const bool recreate_readback_buffer = (_camera_queue_count_readback_buffer.valid() == false) || (_camera_queue_count_readback_buffer_size != kGPUWavefrontQueueHeaderSize);
    if (ensure_host_visible_buffer(device, kGPUWavefrontQueueHeaderSize, queue_readback_usage, _camera_queue_count_readback_buffer, _camera_queue_count_readback_buffer_size,
          _camera_queue_count_readback_buffer_descriptor_index, "wavefront_camera_queue_count_readback") == false) {
      return false;
    }
    if (recreate_readback_buffer) {
      _camera_queue_count_readback_state = RHIResourceState::Undefined;
    }
  } else {
    destroy_linear_scene_buffer(device, _camera_queue_count_readback_buffer, _camera_queue_count_readback_buffer_size, _camera_queue_count_readback_buffer_descriptor_index);
    _camera_queue_count_readback_state = RHIResourceState::Undefined;
  }
  if (allocate_light_path) {
    const bool recreate_readback_buffer = (_light_queue_count_readback_buffer.valid() == false) || (_light_queue_count_readback_buffer_size != kGPUWavefrontQueueHeaderSize);
    if (ensure_host_visible_buffer(device, kGPUWavefrontQueueHeaderSize, queue_readback_usage, _light_queue_count_readback_buffer, _light_queue_count_readback_buffer_size,
          _light_queue_count_readback_buffer_descriptor_index, "wavefront_light_queue_count_readback") == false) {
      return false;
    }
    if (recreate_readback_buffer) {
      _light_queue_count_readback_state = RHIResourceState::Undefined;
    }
  } else {
    destroy_linear_scene_buffer(device, _light_queue_count_readback_buffer, _light_queue_count_readback_buffer_size, _light_queue_count_readback_buffer_descriptor_index);
    _light_queue_count_readback_state = RHIResourceState::Undefined;
  }
  if (allocate_camera_path) {
    if (ensure_storage_buffer_capacity(device, camera_vertex_buffer_size, wavefront_usage, _camera_vertex_buffer, _camera_vertex_buffer_size,
          _camera_vertex_buffer_descriptor_index, "wavefront_camera_vertex") == false) {
      return false;
    }
  } else {
    destroy_linear_scene_buffer(device, _camera_vertex_buffer, _camera_vertex_buffer_size, _camera_vertex_buffer_descriptor_index);
  }
  if (allocate_light_path) {
    if (ensure_storage_buffer_capacity(device, light_vertex_buffer_size, light_vertex_usage, _light_vertex_buffer, _light_vertex_buffer_size, _light_vertex_buffer_descriptor_index,
          "wavefront_light_vertex") == false) {
      return false;
    }
  } else {
    destroy_linear_scene_buffer(device, _light_vertex_buffer, _light_vertex_buffer_size, _light_vertex_buffer_descriptor_index);
  }
  if (enable_merge_vertices) {
    if (ensure_storage_buffer_capacity(device, vcm_grid_heads_buffer_size, wavefront_usage, _vcm_grid_heads_buffer, _vcm_grid_heads_buffer_size,
          _vcm_grid_heads_buffer_descriptor_index, "wavefront_vcm_grid_heads") == false ||
        ensure_storage_buffer_capacity(device, vcm_grid_next_buffer_size, wavefront_usage, _vcm_grid_next_buffer, _vcm_grid_next_buffer_size,
          _vcm_grid_next_buffer_descriptor_index, "wavefront_vcm_grid_next") == false) {
      return false;
    }
  } else {
    destroy_linear_scene_buffer(device, _vcm_grid_heads_buffer, _vcm_grid_heads_buffer_size, _vcm_grid_heads_buffer_descriptor_index);
    destroy_linear_scene_buffer(device, _vcm_grid_next_buffer, _vcm_grid_next_buffer_size, _vcm_grid_next_buffer_descriptor_index);
  }
  if (use_fast_light_endpoints) {
    if (ensure_storage_buffer_capacity(device, fast_light_endpoint_buffer_size, wavefront_usage, _fast_light_endpoint_buffer, _fast_light_endpoint_buffer_size,
          _fast_light_endpoint_buffer_descriptor_index, "wavefront_fast_light_endpoint") == false) {
      return false;
    }
  } else {
    destroy_linear_scene_buffer(device, _fast_light_endpoint_buffer, _fast_light_endpoint_buffer_size, _fast_light_endpoint_buffer_descriptor_index);
  }
  if (ensure_storage_buffer_capacity(device, film_buffer_size, wavefront_usage, _film_buffer, _film_buffer_size, _film_buffer_descriptor_index, "wavefront_film") == false) {
    return false;
  }
  if (ensure_storage_buffer_capacity(device, path_meta_buffer_size, wavefront_usage, _path_meta_buffer, _path_meta_buffer_size, _path_meta_buffer_descriptor_index,
        "wavefront_path_meta") == false) {
    return false;
  }
  if (enable_connect_to_light) {
    if (ensure_storage_buffer_capacity(device, direct_light_work_buffer_size, wavefront_usage, _direct_light_sample_buffer, _direct_light_sample_buffer_size,
          _direct_light_sample_buffer_descriptor_index, "wavefront_direct_light_work") == false) {
      return false;
    }
    destroy_linear_scene_buffer(device, _direct_light_task_buffer, _direct_light_task_buffer_size, _direct_light_task_buffer_descriptor_index);
    _direct_light_task_buffer_descriptor_index = _direct_light_sample_buffer_descriptor_index;
  } else {
    destroy_linear_scene_buffer(device, _direct_light_sample_buffer, _direct_light_sample_buffer_size, _direct_light_sample_buffer_descriptor_index);
    destroy_linear_scene_buffer(device, _direct_light_task_buffer, _direct_light_task_buffer_size, _direct_light_task_buffer_descriptor_index);
  }
  if (enable_connect_to_light) {
    if (ensure_storage_buffer_capacity(device, direct_light_result_buffer_size, wavefront_usage, _direct_light_result_buffer, _direct_light_result_buffer_size,
          _direct_light_result_buffer_descriptor_index, "wavefront_direct_light_result") == false) {
      return false;
    }
  } else {
    destroy_linear_scene_buffer(device, _direct_light_result_buffer, _direct_light_result_buffer_size, _direct_light_result_buffer_descriptor_index);
  }
  if (enable_connect_vertices) {
    if (ensure_storage_buffer_capacity(device, connect_light_task_buffer_size, wavefront_usage, _connect_light_task_buffer, _connect_light_task_buffer_size,
          _connect_light_task_buffer_descriptor_index, "wavefront_connect_light_task") == false) {
      return false;
    }
  } else {
    destroy_linear_scene_buffer(device, _connect_light_task_buffer, _connect_light_task_buffer_size, _connect_light_task_buffer_descriptor_index);
  }
  if (allocate_connect_to_camera) {
    if (enable_connect_to_light) {
      destroy_linear_scene_buffer(device, _connect_camera_task_buffer, _connect_camera_task_buffer_size, _connect_camera_task_buffer_descriptor_index);
      destroy_linear_scene_buffer(device, _connect_camera_result_buffer, _connect_camera_result_buffer_size, _connect_camera_result_buffer_descriptor_index);
      _connect_camera_task_buffer_descriptor_index = _direct_light_sample_buffer_descriptor_index;
      _connect_camera_result_buffer_descriptor_index = _direct_light_result_buffer_descriptor_index;
    } else {
      if (ensure_storage_buffer_capacity(device, connect_camera_task_buffer_size, wavefront_usage, _connect_camera_task_buffer, _connect_camera_task_buffer_size,
            _connect_camera_task_buffer_descriptor_index, "wavefront_connect_camera_task") == false) {
        return false;
      }
      if (ensure_storage_buffer_capacity(device, connect_camera_result_buffer_size, wavefront_usage, _connect_camera_result_buffer, _connect_camera_result_buffer_size,
            _connect_camera_result_buffer_descriptor_index, "wavefront_connect_camera_result") == false) {
        return false;
      }
    }
  } else {
    destroy_linear_scene_buffer(device, _connect_camera_task_buffer, _connect_camera_task_buffer_size, _connect_camera_task_buffer_descriptor_index);
    destroy_linear_scene_buffer(device, _connect_camera_result_buffer, _connect_camera_result_buffer_size, _connect_camera_result_buffer_descriptor_index);
  }
  if (allocate_camera_path && enable_subsurface_state_buffers) {
    if (ensure_storage_buffer_capacity(device, subsurface_state_buffer_size, wavefront_usage, _camera_subsurface_state_buffer, _camera_subsurface_state_buffer_size,
          _camera_subsurface_state_buffer_descriptor_index, "wavefront_camera_subsurface_state") == false) {
      return false;
    }
  } else {
    destroy_linear_scene_buffer(device, _camera_subsurface_state_buffer, _camera_subsurface_state_buffer_size, _camera_subsurface_state_buffer_descriptor_index);
  }
  if (allocate_light_path && enable_subsurface_state_buffers) {
    if (ensure_storage_buffer_capacity(device, subsurface_state_buffer_size, wavefront_usage, _light_subsurface_state_buffer, _light_subsurface_state_buffer_size,
          _light_subsurface_state_buffer_descriptor_index, "wavefront_light_subsurface_state") == false) {
      return false;
    }
  } else {
    destroy_linear_scene_buffer(device, _light_subsurface_state_buffer, _light_subsurface_state_buffer_size, _light_subsurface_state_buffer_descriptor_index);
  }

  _wavefront_resources = {};
  GPUWavefrontResources& resources = _wavefront_resources;
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
  resources.connect_camera_task_buffer = _connect_camera_task_buffer_descriptor_index;
  resources.connect_camera_result_buffer = _connect_camera_result_buffer_descriptor_index;
  resources.camera_subsurface_state_buffer = _camera_subsurface_state_buffer_descriptor_index;
  resources.light_subsurface_state_buffer = _light_subsurface_state_buffer_descriptor_index;
  resources.path_capacity = active_path_capacity;
  resources.max_path_length = wavefront_hard_iteration_cap;
  resources.camera_vertex_capacity = camera_vertex_capacity;
  resources.light_vertex_capacity = light_vertex_capacity;
  resources.camera_fixed_max_bounces = camera_history_bounces;
  resources.light_fixed_max_bounces = light_history_bounces;
  resources.dispatch_args_buffer = _wavefront_dispatch_args_buffer_descriptor_index;
  resources.material_queue_buffer = _material_queue_buffer_descriptor_index;
  resources.shadow_queue_buffer = _shadow_queue_buffer_descriptor_index;
  resources.light_vertex_counter_buffer = compact_light_history ? _light_vertex_counter_buffer_descriptor_index : kInvalidDescriptorIndex;
  resources.fast_light_endpoint_buffer = use_fast_light_endpoints ? _fast_light_endpoint_buffer_descriptor_index : kInvalidDescriptorIndex;
  resources.vcm_grid_heads_buffer = enable_merge_vertices ? _vcm_grid_heads_buffer_descriptor_index : kInvalidDescriptorIndex;
  resources.vcm_grid_next_buffer = enable_merge_vertices ? _vcm_grid_next_buffer_descriptor_index : kInvalidDescriptorIndex;

  if (upload_or_update_linear_scene_buffer(device, &resources, size_t(1), wavefront_usage, _wavefront_resources_buffer, _wavefront_resources_buffer_size,
        _wavefront_resources_buffer_descriptor_index, "wavefront_resources") == false) {
    return false;
  }

  _wavefront_path_capacity = path_capacity;
  _wavefront_vertex_capacity = std::max(camera_vertex_capacity, light_vertex_capacity);
  _wavefront_allocated_integrator_mode = _integrator_mode;
  _wavefront_allocated_integrator_features = _integrator_features;
  _wavefront_light_history_capacity_bounces = allocate_light_path ? light_history_bounces : 0u;
  return true;
}

bool GPURaytracingRenderer::ensure_light_vertex_capacity(RHIContext& ctx, uint32_t required_vertex_capacity) {
  if (required_vertex_capacity <= _wavefront_resources.light_vertex_capacity) {
    return true;
  }
  if ((_wavefront_path_capacity == 0u) || (_light_vertex_buffer.valid() == false) || (_wavefront_resources_buffer.valid() == false)) {
    return false;
  }

  const uint64_t max_vertex_capacity =
    std::min<uint64_t>(static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()), kWavefrontMaxAddressableBufferSize / kGPUWavefrontLightPathVertexStride);
  if (static_cast<uint64_t>(required_vertex_capacity) > max_vertex_capacity) {
    log::error("GPU RT: compact light history requires more than the shader-addressable capacity (required=%u maximum=%llu)", required_vertex_capacity,
      static_cast<unsigned long long>(max_vertex_capacity));
    return false;
  }

  const uint64_t doubled_capacity = static_cast<uint64_t>(_wavefront_resources.light_vertex_capacity) * 2ull;
  const uint64_t new_vertex_capacity = std::min(max_vertex_capacity, std::max<uint64_t>(required_vertex_capacity, doubled_capacity));
  const uint64_t new_buffer_size = new_vertex_capacity * kGPUWavefrontLightPathVertexStride;
  const uint64_t allocated_vertex_capacity = _light_vertex_buffer_size / kGPUWavefrontLightPathVertexStride;
  const bool reuse_light_vertex_buffer = new_vertex_capacity <= allocated_vertex_capacity;

  auto& device = ctx.device();
  RHIBindlessHandle new_light_vertex_buffer = _light_vertex_buffer;
  if (reuse_light_vertex_buffer == false) {
    RHIBufferDesc desc = {};
    desc.size = new_buffer_size;
    desc.usage = RHIBufferUsage::Storage | RHIBufferUsage::TransferSrc | RHIBufferUsage::TransferDst;
    const RHICreateBindlessResult create_result = device.create_buffer(desc);
    if ((create_result.result != RHIResult::Success) || (create_result.handle.valid() == false)) {
      const RHIMemoryStats memory_stats = device.get_memory_statistics();
      log::error("GPU RT: failed to grow light history buffer (%u): requested=%llu bytes (%.2fMB), previous=%llu bytes (%.2fMB), device-local=%.2f/%.2fMB",
        static_cast<uint32_t>(create_result.result), static_cast<unsigned long long>(new_buffer_size), static_cast<double>(new_buffer_size) / (1024.0 * 1024.0),
        static_cast<unsigned long long>(_light_vertex_buffer_size), static_cast<double>(_light_vertex_buffer_size) / (1024.0 * 1024.0),
        static_cast<double>(memory_stats.gpu_device_local_allocated_bytes) / (1024.0 * 1024.0),
        static_cast<double>(memory_stats.gpu_device_local_budget_bytes) / (1024.0 * 1024.0));
      return false;
    }
    new_light_vertex_buffer = create_result.handle;

    RHICommandBuffer cmd = ctx.get_command_buffer();
    if (cmd.valid() == false) {
      device.destroy_buffer(new_light_vertex_buffer);
      return false;
    }
    ctx.command_buffer_begin(cmd);
    ctx.cmd_buffer_barrier(cmd, _light_vertex_buffer, RHIResourceState::General, RHIResourceState::TransferSrc);
    ctx.cmd_buffer_barrier(cmd, new_light_vertex_buffer, RHIResourceState::Undefined, RHIResourceState::TransferDst);
    ctx.cmd_copy_buffer(cmd, _light_vertex_buffer, new_light_vertex_buffer, _light_vertex_buffer_size);
    ctx.cmd_buffer_barrier(cmd, _light_vertex_buffer, RHIResourceState::TransferSrc, RHIResourceState::General);
    ctx.cmd_buffer_barrier(cmd, new_light_vertex_buffer, RHIResourceState::TransferDst, RHIResourceState::General);
    ctx.command_buffer_end(cmd);
    ctx.submit_command_buffer({cmd});
    const RHIResult copy_result = ctx.wait_for_command_buffer(cmd);
    ctx.destroy_command_buffer(cmd);
    if (copy_result != RHIResult::Success) {
      device.destroy_buffer(new_light_vertex_buffer);
      log::error("GPU RT: failed to copy the grown light history buffer (%u)", static_cast<uint32_t>(copy_result));
      return false;
    }
  }

  GPUWavefrontResources updated_resources = _wavefront_resources;
  updated_resources.light_vertex_buffer = get_bindless_descriptor_index(new_light_vertex_buffer);
  updated_resources.light_vertex_capacity = static_cast<uint32_t>(new_vertex_capacity);
  updated_resources.light_fixed_max_bounces = static_cast<uint32_t>(new_vertex_capacity / static_cast<uint64_t>(_wavefront_path_capacity)) - 1u;
  if (gpu_integrator_feature_enabled(_integrator_features, GPUIntegratorFeatures::MergeVertices)) {
    const uint64_t grid_heads_size = wavefront_vcm_grid_storage_size(wavefront_vcm_grid_head_count(updated_resources.light_vertex_capacity));
    const uint64_t grid_next_size = new_vertex_capacity * sizeof(uint32_t);
    const RHIBufferUsage grid_usage = RHIBufferUsage::Storage | RHIBufferUsage::TransferDst;
    if (ensure_storage_buffer_capacity(device, grid_heads_size, grid_usage, _vcm_grid_heads_buffer, _vcm_grid_heads_buffer_size, _vcm_grid_heads_buffer_descriptor_index,
          "wavefront_vcm_grid_heads") == false ||
        ensure_storage_buffer_capacity(device, grid_next_size, grid_usage, _vcm_grid_next_buffer, _vcm_grid_next_buffer_size, _vcm_grid_next_buffer_descriptor_index,
          "wavefront_vcm_grid_next") == false) {
      if (reuse_light_vertex_buffer == false) {
        device.destroy_buffer(new_light_vertex_buffer);
      }
      return false;
    }
    updated_resources.vcm_grid_heads_buffer = _vcm_grid_heads_buffer_descriptor_index;
    updated_resources.vcm_grid_next_buffer = _vcm_grid_next_buffer_descriptor_index;
  }
  const RHIResult resource_update_result = device.update_buffer(_wavefront_resources_buffer, &updated_resources, sizeof(updated_resources));
  if (resource_update_result != RHIResult::Success) {
    if (reuse_light_vertex_buffer == false) {
      device.destroy_buffer(new_light_vertex_buffer);
    }
    log::error("GPU RT: failed to update resources after light history growth (%u)", static_cast<uint32_t>(resource_update_result));
    return false;
  }

  const RHIBindlessHandle old_buffer = reuse_light_vertex_buffer ? RHIBindlessHandle{} : _light_vertex_buffer;
  _light_vertex_buffer = new_light_vertex_buffer;
  if (reuse_light_vertex_buffer == false) {
    _light_vertex_buffer_size = new_buffer_size;
  }
  _light_vertex_buffer_descriptor_index = updated_resources.light_vertex_buffer;
  _wavefront_resources = updated_resources;
  _wavefront_vertex_capacity = std::max(_wavefront_vertex_capacity, updated_resources.light_vertex_capacity);
  _wavefront_light_history_capacity_bounces = updated_resources.light_fixed_max_bounces;
  if (old_buffer.valid()) {
    device.destroy_buffer(old_buffer);
  }
  log::info("GPU RT: grew compact light history capacity to %u vertices (%.2fMB allocated%s)", updated_resources.light_vertex_capacity,
    static_cast<double>(_light_vertex_buffer_size) / (1024.0 * 1024.0), reuse_light_vertex_buffer ? ", reused" : "");
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

  if (scene.data().options.properties[Scene::Properties::BlueNoise] == false) {
    if (_blue_noise_buffer.valid()) {
      destroy_blue_noise_buffer(ctx);
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
  _tlas_instance_buffer = {};
  _as_scratch_buffer = {};
  _tlas_instance_count = 0u;
  _vertex_positions_buffer = {};
  _vertex_positions_buffer_size = 0u;
  _gpu_scene.vertex_positions = kInvalidDescriptorIndex;
}

void GPURaytracingRenderer::render(RHIContext& ctx, SceneRepresentation& scene, const FrameData& frame_data) {
  ETX_PROFILER_SCOPE();
  double pipeline_refresh_ms = 0.0;
  double scene_hash_ms = 0.0;
  double full_rebuild_destroy_ms = 0.0;
  double build_as_ms = 0.0;
  double partial_scene_update_ms = 0.0;
  double wavefront_buffer_ms = 0.0;
  double scene_options_upload_ms = 0.0;
  double camera_upload_ms = 0.0;
  double blue_noise_update_ms = 0.0;
  double output_texture_ms = 0.0;
  double dispatch_submit_ms = 0.0;

  if (_initialized == false)
    return;

  _scene_valid = scene.valid();
  if (_scene_valid == false)
    return;

  auto& device = ctx.device();
  const auto release_wavefront_storage = [&](const char* failure_message) {
    destroy_wavefront_buffers(ctx);
    const RHIResult release_result = ctx.wait_idle();
    if (release_result != RHIResult::Success) {
      set_runtime_failure(failure_message);
      return false;
    }
    _last_memory_stats = device.get_memory_statistics();
    return true;
  };

  poll_preparation_tasks(ctx);
  if (_publish_preparation) {
    const uint32_t pipeline_publish_batch_size = (ctx.device().backend() == RHIBackend::Vulkan) ? std::max(1u, scheduler.max_thread_count()) : kNonVulkanPipelinePublishBatchSize;
    advance_pipeline_publish(ctx, pipeline_publish_batch_size, false);
  }
  if (_pipeline_publish_task) {
    return;
  }

  const GPUIntegratorSelection integrator_selection = gpu_integrator_selection_from_scene(scene);
  const uint32_t new_integrator_mode = static_cast<uint32_t>(integrator_selection.mode);
  const uint32_t new_integrator_features = integrator_selection.features;
  const uint32_t new_material_compile_mask = build_material_compile_mask(scene.data());
  const uint32_t new_spectral_mode = static_cast<uint32_t>(gpu_spectral_mode(scene.data()));
  _last_target_samples = std::max(1u, (_sample_limit > 0u) ? _sample_limit : scene.data().options.samples);
  const bool integrator_mode_changed = (_integrator_mode != new_integrator_mode);
  const bool integrator_features_changed = (_integrator_features != new_integrator_features);
  const bool material_compile_mask_changed = (_material_compile_mask != new_material_compile_mask);
  const bool spectral_mode_changed = (_spectral_mode != new_spectral_mode);
  const bool pipeline_configuration_changed = integrator_mode_changed || integrator_features_changed || material_compile_mask_changed || spectral_mode_changed;
  const bool missing_pipelines = (_preparation_state == RendererPreparationState::Ready) && (pipelines_valid() == false);
  const bool failed_preparation_can_retry = (_preparation_state == RendererPreparationState::Failed) && (_preparation_canceled == false) && (_runtime_failed == false);
  const bool material_configuration_supported = gpu_material_compile_mask_supported(new_material_compile_mask);
  const bool should_request_prepare =
    integrator_selection.supported && material_configuration_supported && (pipeline_configuration_changed || missing_pipelines || failed_preparation_can_retry);
  const bool allocated_integrator_will_change =
    integrator_selection.supported && material_configuration_supported && _wavefront_resources_buffer.valid() &&
    ((_wavefront_allocated_integrator_mode != new_integrator_mode) || (_wavefront_allocated_integrator_features != new_integrator_features));
  if (allocated_integrator_will_change) {
    if (release_wavefront_storage("GPU RT failed to release resources before preparing the integrator change") == false) {
      return;
    }
  }
  if (should_request_prepare) {
    const auto pipeline_refresh_begin = std::chrono::steady_clock::now();
    request_pipeline_preparation(scene, pipeline_configuration_changed ? "scene pipeline change" : "missing pipelines", false);
    const auto pipeline_refresh_end = std::chrono::steady_clock::now();
    pipeline_refresh_ms = elapsed_ms(pipeline_refresh_begin, pipeline_refresh_end);
  }

  if (integrator_selection.supported == false) {
    set_runtime_failure(gpu_integrator_selection_error_message(integrator_selection));
    return;
  }
  if (material_configuration_supported == false) {
    set_runtime_failure(gpu_material_compile_mask_error_message(new_material_compile_mask));
    return;
  }
  if (_runtime_failed) {
    return;
  }

  if (pipelines_valid() == false) {
    return;
  }
  if (scene.energy_compensation_interface_preparation_status().state == EnergyCompensationPreparationState::Preparing) {
    return;
  }

  _last_memory_stats = device.get_memory_statistics();

  const SceneUpdateScope scene_update_scope = consume_scene_update_request();
  SceneHashes new_hashes = {};
  UpdateFlags changes = {};
  bool scene_changed = false;
  bool scene_update_required = false;
  {
    ETX_PROFILER_NAMED_SCOPE("gpu_rt_scene_hashes_and_changes");
    const auto scene_hash_begin = std::chrono::steady_clock::now();
    const bool camera_only_update = scene_update_scope == SceneUpdateScope::Camera;
    const bool transform_only_update = scene_update_scope == SceneUpdateScope::Transforms;
    const bool scoped_transform_update = camera_only_update || transform_only_update;
    new_hashes = _current_scene_hashes;
    if (scoped_transform_update) {
      new_hashes.transforms_hash = scene.data().compute_transforms_hash();
      if (transform_only_update) {
        new_hashes.instance_transforms_hash = scene.data().compute_instance_transforms_hash();
        changes = new_hashes.compare(_current_scene_hashes);
      }
    } else {
      new_hashes = scene.data().compute_hashes();
      changes = new_hashes.compare(_current_scene_hashes);
    }
    const bool full_update_requested = scene_update_scope == SceneUpdateScope::Full;
    if (scoped_transform_update == false) {
      bool dependencies_updated = false;
      if (scene.synchronize_render_dependencies(changes, full_update_requested, dependencies_updated) == false) {
        set_runtime_failure("Failed to synchronize derived scene state before GPU render commit");
        request_scene_update();
        return;
      }

      if (dependencies_updated) {
        new_hashes = scene.data().compute_hashes();
        changes = new_hashes.compare(_current_scene_hashes);
      }
      bool refresh_hashes = false;
      if (full_update_requested || changes[UpdateFlags::Images]) {
        scene.data().images.load_images(scheduler);
        refresh_hashes = true;
      }
      if (full_update_requested || changes[UpdateFlags::AnyMaterials]) {
        if (scene.ensure_energy_compensation_interfaces() == false) {
          set_runtime_failure("Failed to ensure BSDF energy-compensation interfaces before GPU render commit");
          request_scene_update();
          return;
        }
        refresh_hashes = true;
      }
      if (refresh_hashes) {
        new_hashes = scene.data().compute_hashes();
        changes = new_hashes.compare(_current_scene_hashes);
      }
    }
    scene_changed = changes.any();
    scene_update_required = scene_changed || (scene_update_scope != SceneUpdateScope::None);
    const auto scene_hash_end = std::chrono::steady_clock::now();
    scene_hash_ms = elapsed_ms(scene_hash_begin, scene_hash_end);
  }

  const Camera& scene_camera = scene.camera();
  Camera camera = scene_camera;
  const uint2 render_film_dimensions = scaled_render_dimensions(scene_camera.film_size);
  build_camera(camera, scene_camera.position, scene_camera.direction, scene_camera.up, render_film_dimensions, get_camera_fov(scene_camera));
  const uint64_t new_integrator_data_revision = scene.integrator_data_revision();
  const bool integrator_settings_changed = _integrator_data_revision_initialized && (new_integrator_data_revision != _current_integrator_data_revision);
  const uint64_t new_camera_hash = xxh64(&camera, sizeof(camera));
  const bool camera_changed = (new_camera_hash != _current_camera_hash);
  const bool upbp_mode = static_cast<GPUIntegratorMode>(_integrator_mode) == GPUIntegratorMode::UPBP;
  const bool wavefront_sample_in_progress_before_changes = (_wavefront_render_step != WavefrontRenderStep::InitSample) || (_wavefront_tile_index != 0u);
  const bool configuration_changed =
    integrator_settings_changed || integrator_mode_changed || integrator_features_changed || material_compile_mask_changed || spectral_mode_changed;
  const bool transform_change_requested = (scene_update_scope == SceneUpdateScope::Transforms) && changes[UpdateFlags::Transforms];
  const bool defer_preview_transform_change =
    (preview_pixel_size() > 0u) && transform_change_requested && wavefront_sample_in_progress_before_changes && (configuration_changed == false);
  if (defer_preview_transform_change) {
    request_scene_transform_update();
    new_hashes = _current_scene_hashes;
    changes = {};
    scene_changed = false;
    scene_update_required = false;
  }
  const bool non_camera_change = scene_changed || configuration_changed;
  const bool defer_preview_camera_change = (preview_pixel_size() > 0u) && camera_changed && wavefront_sample_in_progress_before_changes && (non_camera_change == false);
  const bool apply_camera_change = camera_changed && (defer_preview_camera_change == false);
  const bool restart_accumulation = non_camera_change || apply_camera_change;
  if (restart_accumulation) {
    restart_render_after_change();
  }

  const bool wavefront_sample_in_progress = (_wavefront_render_step != WavefrontRenderStep::InitSample) || (_wavefront_tile_index != 0u);

  const bool geometry_structure_changed = changes[UpdateFlags::AnyGeometryStructure];
  bool needs_full_rebuild = scene_changed && geometry_structure_changed;
  const bool needs_scene_data_reupload = scene_changed && (geometry_structure_changed == false);
  bool scene_data_update_success = true;

  if (needs_scene_data_reupload && (_vertex_positions_buffer.valid() == false)) {
    needs_full_rebuild = true;
  }

  if (needs_full_rebuild) {
    ETX_PROFILER_NAMED_SCOPE("gpu_rt_full_rebuild_resources");
    const auto full_rebuild_destroy_begin = std::chrono::steady_clock::now();
    destroy_scene_buffers(ctx);
    destroy_acceleration_structures(ctx);
    if (release_wavefront_storage("GPU RT failed to release resources for the scene rebuild") == false) {
      return;
    }
    const auto full_rebuild_destroy_end = std::chrono::steady_clock::now();
    full_rebuild_destroy_ms = elapsed_ms(full_rebuild_destroy_begin, full_rebuild_destroy_end);
  }

  if (_tlas.valid() == false) {
    ETX_PROFILER_NAMED_SCOPE("gpu_rt_build_acceleration_structures");
    const auto build_as_begin = std::chrono::steady_clock::now();
    scene_data_update_success = build_acceleration_structures(ctx, scene);
    const auto build_as_end = std::chrono::steady_clock::now();
    build_as_ms = elapsed_ms(build_as_begin, build_as_end);
    if (scene_data_update_success == false) {
      set_runtime_failure("GPU acceleration-structure build failed");
    }
  } else if (needs_scene_data_reupload) {
    ETX_PROFILER_NAMED_SCOPE("gpu_rt_partial_scene_update");
    const auto partial_scene_update_begin = std::chrono::steady_clock::now();
    bool update_success = true;
    bool acceleration_structures_rebuilt = false;
    if (changes[UpdateFlags::InstanceTransforms]) {
      update_success = refit_top_level_acceleration_structure(ctx, scene.data());
      if (update_success == false) {
        destroy_scene_buffers(ctx);
        destroy_acceleration_structures(ctx);
        update_success = build_acceleration_structures(ctx, scene);
        acceleration_structures_rebuilt = update_success;
      }
    }
    if (update_success && (acceleration_structures_rebuilt == false)) {
      update_success = update_scene_data_partial(ctx, scene, changes);
    }
    const auto partial_scene_update_end = std::chrono::steady_clock::now();
    partial_scene_update_ms = elapsed_ms(partial_scene_update_begin, partial_scene_update_end);
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

  bool invalidate_upbp_density_cache = false;
  if (scene_update_required && scene_data_update_success) {
    invalidate_upbp_density_cache = scene_changed && upbp_mode && (_upbp.sample_index != ~0u);
    _current_scene_hashes = new_hashes;
  }
  invalidate_upbp_density_cache = invalidate_upbp_density_cache || (integrator_settings_changed && upbp_mode && (_upbp.sample_index != ~0u));
  _current_integrator_data_revision = new_integrator_data_revision;
  _integrator_data_revision_initialized = true;
  if (apply_camera_change && _scene_globals_buffer.valid()) {
    const SceneBoundingSphere transport_sphere = compute_transport_bounding_sphere(_host_transport_bounds, camera);
    const bool transport_sphere_changed =
      (_host_scene_globals.bounding_sphere_center.x != transport_sphere.center.x) || (_host_scene_globals.bounding_sphere_center.y != transport_sphere.center.y) ||
      (_host_scene_globals.bounding_sphere_center.z != transport_sphere.center.z) || (_host_scene_globals.bounding_sphere_radius != transport_sphere.radius);
    if (transport_sphere_changed) {
      _host_scene_globals.bounding_sphere_center = transport_sphere.center;
      _host_scene_globals.bounding_sphere_radius = transport_sphere.radius;
      const RHIBufferUsage scene_buffer_usage = RHIBufferUsage::Storage | RHIBufferUsage::TransferDst;
      if (upload_or_update_linear_scene_buffer(device, &_host_scene_globals, size_t(1), scene_buffer_usage, _scene_globals_buffer, _scene_globals_buffer_size,
            _gpu_scene.scene_globals, "scene_globals") == false) {
        set_runtime_failure("GPU RT failed to update the transport bounds for the active camera");
        return;
      }
      invalidate_upbp_density_cache = invalidate_upbp_density_cache || (upbp_mode && (_upbp.sample_index != ~0u));
    }
  }
  if (invalidate_upbp_density_cache) {
    const RHIResult reclaim_result = ctx.wait_idle();
    if (reclaim_result != RHIResult::Success) {
      set_runtime_failure("GPU UPBP failed to reclaim invalidated density-cache storage (" + std::to_string(static_cast<uint32_t>(reclaim_result)) + ")");
      return;
    }
    destroy_upbp_density_cache(device, true);
    _upbp.sample_index = ~0u;
    _upbp.render_reset_pending = false;
    _last_memory_stats = device.get_memory_statistics();
  }
  if (apply_camera_change) {
    _current_camera_hash = new_camera_hash;
  }

  if (_scene_options_upload_pending) {
    ETX_PROFILER_NAMED_SCOPE("gpu_rt_runtime_update_scene_options");
    const auto scene_options_upload_begin = std::chrono::steady_clock::now();
    const RHIBufferUsage scene_buffer_usage = RHIBufferUsage::Storage | RHIBufferUsage::TransferDst;
    const GPUSceneOptions options = build_scene_options(scene);
    const bool options_upload_success = upload_or_update_linear_scene_buffer(device, &options, size_t(1), scene_buffer_usage, _scene_options_buffer, _scene_options_buffer_size,
      _gpu_scene.scene_options, "scene_options");
    const auto scene_options_upload_end = std::chrono::steady_clock::now();
    scene_options_upload_ms = elapsed_ms(scene_options_upload_begin, scene_options_upload_end);
    if (options_upload_success == false) {
      set_runtime_failure("GPU RT failed to upload scene options for the selected integrator");
      return;
    }
    _scene_options_upload_pending = false;
  }

  const uint2 full_dim = camera.film_size;
  uint32_t frame_camera_buffer_descriptor_index = _camera_buffer_descriptor_index;
  if (wavefront_sample_in_progress == false) {
    ETX_PROFILER_NAMED_SCOPE("gpu_rt_upload_camera");
    const auto camera_upload_begin = std::chrono::steady_clock::now();
    const RHIBufferUsage camera_buffer_usage = RHIBufferUsage::Storage | RHIBufferUsage::TransferDst;
    const bool camera_upload_success =
      upload_or_update_linear_scene_buffer(device, &camera, size_t(1), camera_buffer_usage, _camera_buffer, _camera_buffer_size, _camera_buffer_descriptor_index, "camera");
    const auto camera_upload_end = std::chrono::steady_clock::now();
    camera_upload_ms = elapsed_ms(camera_upload_begin, camera_upload_end);
    if (camera_upload_success == false) {
      set_runtime_failure("GPU RT failed to upload the camera");
      return;
    }
  }

  if (wavefront_sample_in_progress == false) {
    const auto blue_noise_update_begin = std::chrono::steady_clock::now();
    if (update_blue_noise_buffer(ctx, scene) == false) {
      log::warning("GPU RT: blue noise buffer is unavailable, falling back to white noise");
    }
    const auto blue_noise_update_end = std::chrono::steady_clock::now();
    blue_noise_update_ms = elapsed_ms(blue_noise_update_begin, blue_noise_update_end);
  }

  const bool has_render_window = (_render_window_size.x > 0u) && (_render_window_size.y > 0u);
  const uint32_t render_pixel_block_size = render_pixel_size();
  const uint2 base_render_origin = has_render_window ? uint2{_render_window_origin.x / render_pixel_block_size, _render_window_origin.y / render_pixel_block_size} : uint2{};
  const uint2 render_window_end = has_render_window ? uint2{
    (_render_window_origin.x + _render_window_size.x + render_pixel_block_size - 1u) / render_pixel_block_size,
    (_render_window_origin.y + _render_window_size.y + render_pixel_block_size - 1u) / render_pixel_block_size,
  } : full_dim;
  const uint2 base_render_dim = has_render_window ? uint2{render_window_end.x - base_render_origin.x, render_window_end.y - base_render_origin.y} : full_dim;
  const uint32_t scene_max_path_length_for_tiling = std::max(1u, scene.data().options.max_path_length);
  const bool use_complete_light_history = gpu_integrator_feature_enabled(_integrator_features, GPUIntegratorFeatures::ConnectVertices) ||
                                          gpu_integrator_feature_enabled(_integrator_features, GPUIntegratorFeatures::MergeVertices);
  const bool vcm_mode = static_cast<GPUIntegratorMode>(_integrator_mode) == GPUIntegratorMode::VCM;
  const bool allocated_integrator_changed =
    _wavefront_resources_buffer.valid() && ((_wavefront_allocated_integrator_mode != _integrator_mode) || (_wavefront_allocated_integrator_features != _integrator_features));
  if (allocated_integrator_changed) {
    if (release_wavefront_storage("GPU RT failed to release resources for the integrator change") == false) {
      return;
    }
  }
  // VCM's light population, normalization, and spatial grid are iteration-global.
  const bool use_wavefront_tiling = (vcm_mode == false) && (use_complete_light_history || upbp_mode);
  const uint64_t base_render_pixel_count_u64 = static_cast<uint64_t>(base_render_dim.x) * static_cast<uint64_t>(base_render_dim.y);
  if (base_render_pixel_count_u64 > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
    set_runtime_failure("GPU RT render window exceeds the addressable path capacity");
    return;
  }
  const uint32_t base_render_pixel_count = static_cast<uint32_t>(base_render_pixel_count_u64);
  if (base_render_pixel_count == 0u) {
    set_runtime_failure("GPU RT render window is empty");
    return;
  }
  uint32_t tile_max_pixels = base_render_pixel_count;
  uint32_t wavefront_tile_count_value = 1u;
  uint32_t wavefront_buffer_path_capacity = base_render_pixel_count;
  uint2 active_base_origin = base_render_origin;
  uint2 active_base_dim = base_render_dim;
  if (use_wavefront_tiling) {
    const bool should_rebuild_tile_plan = _wavefront_tile_plan_valid == false;
    if (should_rebuild_tile_plan) {
      bool scene_has_subsurface_material = false;
      for (const auto& material : scene.data().materials) {
        if (material.subsurface_cls != SubsurfaceMaterial::Disabled) {
          scene_has_subsurface_material = true;
          break;
        }
      }
      const uint32_t initial_light_history_bounces = wavefront_initial_light_history_bounces(scene_max_path_length_for_tiling);
      const uint32_t tile_integrator_features = upbp_mode ? (_integrator_features & ~static_cast<uint32_t>(GPUIntegratorFeatures::MergeVertices)) : _integrator_features;
      const bool compact_connections =
        ((tile_integrator_features & GPUIntegratorFeatures::ConnectVertices) != 0u) && (material_compile_mask_connection_queue_count(_material_compile_mask) > 1u);
      const uint64_t connection_queue_fixed_bytes = compact_connections ? kGPUWavefrontConnectQueueHeaderSize : 0u;
      const uint64_t tile_bytes_per_path =
        wavefront_tile_bytes_per_path(tile_integrator_features, scene_has_subsurface_material, initial_light_history_bounces, upbp_mode == false, compact_connections);
      if (upbp_mode) {
        UPBPOptions options = {};
        const auto settings = scene.integrator_data().settings.find(Integrator::Type::UPBP);
        if (settings != scene.integrator_data().settings.end()) {
          options.load(settings->second);
        }
        std::string validation_reason = {};
        if (upbp_options_valid(options, validation_reason) == false) {
          set_runtime_failure("GPU UPBP: " + validation_reason);
          return;
        }
        const bool merge_vertices_enabled = (scene.data().options.strategy_flags & Scene::Strategy::MergeVertices) != 0u;
        const uint32_t storage_technique_mask = upbp_effective_technique_mask(options, merge_vertices_enabled);
        const RHIMemoryStats planning_memory_stats = device.get_memory_statistics();
        const uint64_t working_set_budget = gpu_resident_working_set_budget_bytes(planning_memory_stats);
        const uint64_t film_pixel_count = static_cast<uint64_t>(full_dim.x) * full_dim.y;
        const uint64_t required_film_bytes = film_pixel_count * sizeof(float4);
        const uint2 output_dim = scene_camera.film_size;
        const uint64_t required_output_bytes = static_cast<uint64_t>(output_dim.x) * output_dim.y * sizeof(float4);
        const uint64_t additional_fixed_bytes = ((_film_buffer_size >= required_film_bytes) ? 0u : required_film_bytes) +
                                                (((_output_dimensions.x == output_dim.x) && (_output_dimensions.y == output_dim.y)) ? 0u : required_output_bytes) +
                                                connection_queue_fixed_bytes;
        const uint64_t available_path_and_cache_budget = (working_set_budget > additional_fixed_bytes) ? (working_set_budget - additional_fixed_bytes) : 0u;
        const uint64_t density_cache_reserve = upbp_density_cache_reserve_bytes(available_path_and_cache_budget, base_render_pixel_count, scene_max_path_length_for_tiling,
          storage_technique_mask, options.maximum_bb1d_light_path_count);
        const uint64_t path_working_set_budget = available_path_and_cache_budget - density_cache_reserve;
        const uint32_t resident_path_capacity = upbp_resident_path_capacity(path_working_set_budget, base_render_pixel_count, scene_max_path_length_for_tiling,
          options.maximum_boundary_count, storage_technique_mask, options.maximum_bb1d_light_path_count, tile_bytes_per_path);
        if (resident_path_capacity == 0u) {
          set_runtime_failure("GPU UPBP available device-local memory cannot hold one resident camera/light path pair");
          return;
        }
        const uint32_t realized_resident_path_capacity = wavefront_tile_path_capacity(base_render_dim, resident_path_capacity);
        const bool capacity_changed = _wavefront_tile_path_capacity != realized_resident_path_capacity;
        _wavefront_tile_max_pixels = resident_path_capacity;
        if (capacity_changed) {
          const uint64_t resident_bytes = upbp_initial_resident_storage_bytes(realized_resident_path_capacity, scene_max_path_length_for_tiling, options.maximum_boundary_count,
                                            storage_technique_mask, options.maximum_bb1d_light_path_count) +
                                          static_cast<uint64_t>(realized_resident_path_capacity) * tile_bytes_per_path + additional_fixed_bytes;
          log::info(
            "GPU UPBP selected %u resident camera/light paths for %u global paths; transient working set %.1f MiB, compact density reserve %.1f MiB, hardware budget %.1f MiB; "
            "device-local usage %.1f/%.1f MiB",
            realized_resident_path_capacity, base_render_pixel_count, static_cast<double>(resident_bytes) / (1024.0 * 1024.0),
            static_cast<double>(density_cache_reserve) / (1024.0 * 1024.0), static_cast<double>(working_set_budget) / (1024.0 * 1024.0),
            static_cast<double>(planning_memory_stats.gpu_device_local_allocated_bytes) / (1024.0 * 1024.0),
            static_cast<double>(planning_memory_stats.gpu_device_local_budget_bytes) / (1024.0 * 1024.0));
        }
      } else {
        const uint64_t tile_budget = wavefront_tile_budget_bytes(_last_memory_stats);
        const uint64_t path_budget = (tile_budget > connection_queue_fixed_bytes) ? (tile_budget - connection_queue_fixed_bytes) : 0u;
        _wavefront_tile_max_pixels = wavefront_tile_max_pixels(path_budget, tile_bytes_per_path, base_render_pixel_count);
      }
      _wavefront_tile_count = wavefront_tile_count(base_render_dim, _wavefront_tile_max_pixels);
      _wavefront_tile_path_capacity = wavefront_tile_path_capacity(base_render_dim, _wavefront_tile_max_pixels);
      if (upbp_mode) {
        _upbp_light_path_capacity = _wavefront_tile_path_capacity;
      }
      _wavefront_tile_base_origin = base_render_origin;
      _wavefront_tile_base_size = base_render_dim;
      _wavefront_tile_plan_valid = true;
    }
    tile_max_pixels = _wavefront_tile_max_pixels;
    wavefront_tile_count_value = _wavefront_tile_count;
    wavefront_buffer_path_capacity = _wavefront_tile_path_capacity;
    active_base_origin = _wavefront_tile_base_origin;
    active_base_dim = _wavefront_tile_base_size;
    if (_wavefront_tile_index >= wavefront_tile_count_value) {
      _wavefront_tile_index = 0u;
    }
  } else {
    _wavefront_tile_plan_valid = false;
  }
  WavefrontWindow active_window = use_wavefront_tiling ? wavefront_tile_window(active_base_origin, active_base_dim, tile_max_pixels, _wavefront_tile_index)
                                                       : WavefrontWindow{base_render_origin, base_render_dim};
  uint32_t active_path_offset = use_wavefront_tiling ? wavefront_tile_path_offset(active_base_origin, active_base_dim, tile_max_pixels, _wavefront_tile_index) : 0u;
  uint2 render_dim = active_window.size;
  uint32_t wavefront_path_capacity = render_dim.x * render_dim.y;
  const uint32_t upbp_global_path_count = base_render_pixel_count;
  const uint2 output_dim = scene_camera.film_size;
  if ((_output_dimensions.x != output_dim.x) || (_output_dimensions.y != output_dim.y)) {
    ETX_PROFILER_NAMED_SCOPE("gpu_rt_recreate_output_texture");
    const auto output_texture_begin = std::chrono::steady_clock::now();
    RHITextureDesc desc = {};
    desc.width = output_dim.x;
    desc.height = output_dim.y;
    desc.format = RHITextureFormat::R32G32B32A32_FLOAT;
    desc.usage = RHITextureUsage::Storage | RHITextureUsage::Sampled | RHITextureUsage::TransferSrc;
    const RHICreateBindlessResult output_texture_result = device.create_texture(desc);
    if ((output_texture_result.result != RHIResult::Success) || (output_texture_result.handle.valid() == false)) {
      if (output_texture_result.handle.valid()) {
        device.destroy_texture(output_texture_result.handle);
      }
      set_runtime_failure("GPU RT failed to create its output texture (" + std::to_string(static_cast<uint32_t>(output_texture_result.result)) + ")");
      return;
    }
    const RHIResult replacement_wait_result = ctx.wait_idle();
    if (replacement_wait_result != RHIResult::Success) {
      device.destroy_texture(output_texture_result.handle);
      set_runtime_failure("GPU RT failed to synchronize before replacing its output texture (" + std::to_string(static_cast<uint32_t>(replacement_wait_result)) + ")");
      return;
    }
    const RHITexture previous_output_texture = _output_texture;
    _output_texture = output_texture_result.handle;
    _output_dimensions = output_dim;
    _output_texture_state = RHIResourceState::Undefined;
    _display_output_valid = false;
    if (frame_data.cmd.valid()) {
      ctx.cmd_texture_barrier(frame_data.cmd, _output_texture, _output_texture_state, RHIResourceState::ShaderReadOnly);
      _output_texture_state = RHIResourceState::ShaderReadOnly;
    }
    _wavefront_render_step = WavefrontRenderStep::InitSample;
    _wavefront_path_iteration = 0u;
    _wavefront_hard_iteration_cap = 0u;
    _wavefront_camera_queue_count = 0u;
    _wavefront_light_queue_count = 0u;
    _wavefront_connect_light_vertex_length = 0u;
    _wavefront_connect_light_history_bounces = 0u;
    _wavefront_vcm_light_vertex_count = 0u;
    _wavefront_tile_index = 0u;
    _wavefront_tile_max_pixels = 0u;
    _wavefront_tile_count = 1u;
    _wavefront_tile_path_capacity = 0u;
    _upbp_light_path_capacity = 0u;
    _wavefront_tile_base_origin = {};
    _wavefront_tile_base_size = {};
    _wavefront_tile_plan_valid = false;
    _wavefront_camera_phase_initialized = false;
    if (previous_output_texture.valid()) {
      const RHIResult destroy_result = device.destroy_texture(previous_output_texture);
      if (destroy_result != RHIResult::Success) {
        log::warning("GPU RT: failed to destroy the previous output texture (%u)", static_cast<uint32_t>(destroy_result));
      }
    }
    const auto output_texture_end = std::chrono::steady_clock::now();
    output_texture_ms = elapsed_ms(output_texture_begin, output_texture_end);
    return;
  }

  const RHITexture render_output_texture = _output_texture;
  RHIResourceState& render_output_texture_state = _output_texture_state;

  const bool use_spectral_values = (vcm_mode || upbp_mode) && (_spectral_mode == static_cast<uint32_t>(GPUSpectralMode::Spectral));
  if (use_spectral_values) {
    const uint32_t spectrum_count = static_cast<uint32_t>(scene.data().spectrum_values.size());
    const uint64_t spectral_values_size = static_cast<uint64_t>(kGPUSpectralValuesDataOffset) + static_cast<uint64_t>(spectrum_count) * sizeof(float);
    if (ensure_storage_buffer_capacity(device, spectral_values_size, RHIBufferUsage::Storage, _spectral_values_buffer, _spectral_values_buffer_size,
          _spectral_values_buffer_descriptor_index, "wavefront_spectral_values") == false) {
      set_runtime_failure("GPU RT failed to allocate the spectral values cache");
      return;
    }
    _gpu_scene.spectral_values = _spectral_values_buffer_descriptor_index;
    _gpu_scene.spectrum_count = spectrum_count;
  } else {
    _gpu_scene.spectral_values = kInvalidDescriptorIndex;
    _gpu_scene.spectrum_count = 0u;
  }

  GPURTConstants constants = {
    .camera_buffer_index = frame_camera_buffer_descriptor_index,
    .as_index = get_bindless_descriptor_index(_tlas),
    .output_image_index = get_bindless_descriptor_index(_output_texture),
    .frame_index = _frame_index,
    .sample_index = _sample_index,
    .blue_noise_buffer_index = scene.data().options.properties[Scene::Properties::BlueNoise] ? _blue_noise_buffer_descriptor_index : kInvalidDescriptorIndex,
    .wavefront_buffer_index = _wavefront_resources_buffer_descriptor_index,
    .path_iteration = 0u,
    .connect_light_vertex_length = 0u,
    .render_window_origin_x = active_window.origin.x,
    .render_window_origin_y = active_window.origin.y,
    .render_window_width = render_dim.x,
    .render_window_height = render_dim.y,
    .dispatch_item_offset = 0u,
    .dispatch_item_count = 0u,
    .work_queue_index = kInvalidIndex,
    .vcm_light_vertex_count = _wavefront_vcm_light_vertex_count,
    .output_pixel_size = render_pixel_block_size,
    .scene = _gpu_scene,
  };
  RHIDispatchDesc film_dispatch = {
    .group_count_x = (render_dim.x + 7u) / 8u,
    .group_count_y = (render_dim.y + 7u) / 8u,
    .group_count_z = 1u,
  };

  if ((_run_state == RunState::Stopped) || (_run_state == RunState::Completed)) {
    return;
  }

  if (_sample_index >= _last_target_samples) {
    preserve_render_statistics();
    _run_state = RunState::Completed;
    return;
  }

  if (_render_timing_active == false) {
    _preserved_timing_stats_valid = false;
    _render_started_at = std::chrono::steady_clock::now();
    _last_render_elapsed_seconds = 0.0;
    _render_timing_active = true;
  }

  uint64_t released_camera_wavefront_bytes = 0u;
  const bool prepare_upbp_light_phase = upbp_mode && _upbp.camera_phase_started && ((_upbp.sample_index != _sample_index) || (_upbp.camera_batch_index != _wavefront_tile_index));
  const bool upbp_density_cache_stale = upbp_mode && (_upbp.sample_index != ~0u) && (_upbp.sample_index != _sample_index);
  const bool reclaim_upbp_phase_storage = prepare_upbp_light_phase || upbp_density_cache_stale || (upbp_mode && _upbp.render_reset_pending);
  if (reclaim_upbp_phase_storage) {
    const RHIResult reclaim_result = ctx.wait_idle();
    if (reclaim_result != RHIResult::Success) {
      set_runtime_failure("GPU UPBP failed to reclaim completed phase storage (" + std::to_string(static_cast<uint32_t>(reclaim_result)) + ")");
      return;
    }
    if (prepare_upbp_light_phase) {
      released_camera_wavefront_bytes = destroy_upbp_completed_camera_wavefront_buffers(device);
    }
    if (upbp_density_cache_stale || _upbp.render_reset_pending) {
      destroy_upbp_density_cache(device, false);
      _upbp.sample_index = ~0u;
    }
    _upbp.camera_phase_started = false;
    _upbp.render_reset_pending = false;
    if (released_camera_wavefront_bytes > 0u) {
      log::info("GPU UPBP released %.2f MiB of completed camera-wavefront storage before light tracing", static_cast<double>(released_camera_wavefront_bytes) / (1024.0 * 1024.0));
    }
    _last_memory_stats = device.get_memory_statistics();
  }

  const auto wavefront_buffer_begin = std::chrono::steady_clock::now();
  const uint32_t wavefront_resource_path_capacity = upbp_mode ? wavefront_buffer_path_capacity : wavefront_path_capacity;
  if (ensure_wavefront_buffers(ctx, scene, wavefront_buffer_path_capacity, wavefront_resource_path_capacity, false) == false) {
    if (vcm_mode) {
      set_runtime_failure("GPU VCM requires whole-frame wavefront buffers; allocation failed for the current resolution and path length");
    } else {
      set_runtime_failure("GPU RT failed to allocate wavefront buffers for the selected integrator");
    }
    return;
  }
  if (upbp_mode) {
    const bool rebuild_density_cache = (_upbp.sample_index != _sample_index) || (_upbp.density_cache_ready == false);
    const uint32_t resident_camera_capacity = wavefront_buffer_path_capacity;
    const uint32_t resident_light_capacity = rebuild_density_cache ? std::max(resident_camera_capacity, _upbp_light_path_capacity) : resident_camera_capacity;
    if (ensure_upbp_buffers(ctx, scene, upbp_global_path_count, resident_light_capacity, resident_camera_capacity, _wavefront_tile_index, active_path_offset,
          wavefront_path_capacity) == false) {
      set_runtime_failure("GPU UPBP failed to allocate its resident path batch");
      return;
    }
    _wavefront_resources.upbp_resources_buffer = _upbp.resources_buffer.descriptor_index;
    const RHIResult resources_update = device.update_buffer(_wavefront_resources_buffer, &_wavefront_resources, sizeof(_wavefront_resources));
    if (resources_update != RHIResult::Success) {
      set_runtime_failure("GPU UPBP failed to bind its path resources");
      return;
    }
  }
  const auto wavefront_buffer_end = std::chrono::steady_clock::now();
  wavefront_buffer_ms = elapsed_ms(wavefront_buffer_begin, wavefront_buffer_end);
  constants.wavefront_buffer_index = _wavefront_resources_buffer_descriptor_index;
  const bool vcm_merging_enabled = gpu_integrator_feature_enabled(_integrator_features, GPUIntegratorFeatures::MergeVertices);
  if (static_cast<GPUIntegratorMode>(_integrator_mode) == GPUIntegratorMode::VCM) {
    const GPUVCMIterationParameters vcm =
      gpu_vcm_iteration_parameters(scene, _scene_bounding_sphere_radius, _sample_index, base_render_dim, wavefront_path_capacity, vcm_merging_enabled);
    constants.vcm_radius = vcm.radius;
    constants.vcm_vm_weight = vcm.vm_weight;
    constants.vcm_vc_weight = vcm.vc_weight;
    constants.vcm_vm_normalization = vcm.vm_normalization;
    constants.vcm_grid_mask = vcm_merging_enabled ? (wavefront_vcm_grid_head_count(_wavefront_resources.light_vertex_capacity) - 1u) : 0u;
    constants.vcm_kernel = vcm.kernel;
  }

  bool completed_sample = false;
  {
    ETX_PROFILER_NAMED_SCOPE("gpu_rt_dispatch_and_submit");
    const auto dispatch_submit_begin = std::chrono::steady_clock::now();
    struct KernelTimestampSpan {
      PipelineStage stage = PipelineStage::PrepareSample;
      uint32_t begin_query = 0u;
      uint32_t end_query = 0u;
    };
    struct SubmittedCommand {
      RHICommandBuffer command_buffer = {};
      std::vector<KernelTimestampSpan> timestamp_spans = {};
      uint32_t timestamp_query_count = 0u;
    };

    const bool capture_kernel_timings = _kernel_timing_enabled && _kernel_timing_stats.supported;
    if (capture_kernel_timings && (_kernel_timing_started_at == std::chrono::steady_clock::time_point{})) {
      _kernel_timing_started_at = std::chrono::steady_clock::now();
    }
    const uint32_t timestamp_query_capacity = capture_kernel_timings ? ctx.timestamp_query_capacity() : 0u;
    const double timestamp_tick_to_ms = ctx.timestamp_period_ns() * 1.0e-6;
    SubmittedCommand* active_submitted_command = nullptr;
    const auto begin_kernel_timing = [&](RHICommandBuffer cmd, PipelineStage stage) {
      if ((capture_kernel_timings == false) || (active_submitted_command == nullptr)) {
        return ~0u;
      }
      if (active_submitted_command->timestamp_query_count > (timestamp_query_capacity - 2u)) {
        _kernel_timing_stats.dropped_dispatch_count += 1u;
        return ~0u;
      }

      const uint32_t begin_query = active_submitted_command->timestamp_query_count;
      const uint32_t end_query = begin_query + 1u;
      active_submitted_command->timestamp_query_count += 2u;
      active_submitted_command->timestamp_spans.push_back({
        .stage = stage,
        .begin_query = begin_query,
        .end_query = end_query,
      });
      ctx.cmd_begin_timestamp_scope(cmd, begin_query, end_query, RHITimestampStage::ComputeShader);
      return end_query;
    };
    const auto end_kernel_timing = [&](RHICommandBuffer cmd, uint32_t end_query) {
      if (end_query != ~0u) {
        ctx.cmd_end_timestamp_scope(cmd, end_query, RHITimestampStage::ComputeShader);
      }
    };
    const auto dispatch_stage_with_connect_light_length = [&](RHICommandBuffer cmd, PipelineStage stage, uint64_t argument_buffer_offset, uint32_t path_iteration,
                                                            uint32_t connect_light_vertex_length, uint32_t connect_light_vertex_count, bool reset_light_cursor,
                                                            bool initialize_connect_light_batch, uint32_t connect_light_cursor_slot) {
      GPURTConstants stage_constants = constants;
      stage_constants.path_iteration = path_iteration;
      stage_constants.connect_light_vertex_length = connect_light_vertex_length;
      stage_constants.dispatch_item_count = connect_light_vertex_count;
      stage_constants.dispatch_item_offset = (reset_light_cursor ? 1u : 0u) | (initialize_connect_light_batch ? 2u : 0u) | ((connect_light_cursor_slot & 1u) << 3u);
      ctx.cmd_set_pipeline(cmd, _pipelines[static_cast<uint32_t>(stage)]);
      ctx.cmd_push_constants(cmd, &stage_constants, sizeof(stage_constants));
      const uint32_t timing_end_query = begin_kernel_timing(cmd, stage);
      ctx.cmd_dispatch_indirect(cmd, _wavefront_dispatch_args_buffer, argument_buffer_offset);
      end_kernel_timing(cmd, timing_end_query);
    };
    const auto dispatch_stage = [&](RHICommandBuffer cmd, PipelineStage stage, const RHIDispatchDesc& dispatch, uint32_t path_iteration) {
      GPURTConstants stage_constants = constants;
      stage_constants.path_iteration = path_iteration;
      ctx.cmd_set_pipeline(cmd, _pipelines[static_cast<uint32_t>(stage)]);
      ctx.cmd_push_constants(cmd, &stage_constants, sizeof(stage_constants));
      const uint32_t timing_end_query = begin_kernel_timing(cmd, stage);
      ctx.cmd_dispatch(cmd, dispatch);
      end_kernel_timing(cmd, timing_end_query);
    };
    const auto dispatch_stage_mode = [&](RHICommandBuffer cmd, PipelineStage stage, const RHIDispatchDesc& dispatch, uint32_t path_iteration, uint32_t mode) {
      GPURTConstants stage_constants = constants;
      stage_constants.path_iteration = path_iteration;
      stage_constants.dispatch_item_offset = mode;
      ctx.cmd_set_pipeline(cmd, _pipelines[static_cast<uint32_t>(stage)]);
      ctx.cmd_push_constants(cmd, &stage_constants, sizeof(stage_constants));
      const uint32_t timing_end_query = begin_kernel_timing(cmd, stage);
      ctx.cmd_dispatch(cmd, dispatch);
      end_kernel_timing(cmd, timing_end_query);
    };
    const auto dispatch_prepare_sample = [&](RHICommandBuffer cmd) {
      GPURTConstants stage_constants = constants;
      RHIDispatchDesc dispatch = {
        .group_count_x = 1u,
        .group_count_y = 1u,
        .group_count_z = 1u,
      };
      if (use_wavefront_tiling) {
        const bool clear_full_render_window = (_sample_index == 0u) && (_wavefront_tile_index == 0u);
        if (clear_full_render_window) {
          stage_constants.render_window_origin_x = base_render_origin.x;
          stage_constants.render_window_origin_y = base_render_origin.y;
          stage_constants.render_window_width = base_render_dim.x;
          stage_constants.render_window_height = base_render_dim.y;
          dispatch = {
            .group_count_x = (base_render_dim.x + 7u) / 8u,
            .group_count_y = (base_render_dim.y + 7u) / 8u,
            .group_count_z = 1u,
          };
        } else {
          stage_constants.dispatch_item_offset = 1u;
        }
      } else {
        dispatch = film_dispatch;
      }
      ctx.cmd_set_pipeline(cmd, _pipelines[static_cast<uint32_t>(PipelineStage::PrepareSample)]);
      ctx.cmd_push_constants(cmd, &stage_constants, sizeof(stage_constants));
      const uint32_t timing_end_query = begin_kernel_timing(cmd, PipelineStage::PrepareSample);
      ctx.cmd_dispatch(cmd, dispatch);
      end_kernel_timing(cmd, timing_end_query);
    };
    const auto dispatch_stage_range = [&](RHICommandBuffer cmd, PipelineStage stage, uint32_t item_offset, uint32_t item_count, uint32_t path_iteration,
                                        uint32_t work_queue_index) {
      GPURTConstants stage_constants = constants;
      stage_constants.path_iteration = path_iteration;
      stage_constants.dispatch_item_offset = item_offset;
      stage_constants.dispatch_item_count = item_count;
      stage_constants.work_queue_index = work_queue_index;
      ctx.cmd_set_pipeline(cmd, _pipelines[static_cast<uint32_t>(stage)]);
      ctx.cmd_push_constants(cmd, &stage_constants, sizeof(stage_constants));
      const uint32_t timing_end_query = begin_kernel_timing(cmd, stage);
      ctx.cmd_dispatch(cmd, {.group_count_x = divide_round_up(item_count, 64u), .group_count_y = 1u, .group_count_z = 1u});
      end_kernel_timing(cmd, timing_end_query);
    };
    const auto dispatch_stage_query_groups = [&](RHICommandBuffer cmd, PipelineStage stage, uint32_t item_offset, uint32_t item_count, uint32_t path_iteration,
                                               uint32_t work_queue_index) {
      GPURTConstants stage_constants = constants;
      stage_constants.path_iteration = path_iteration;
      stage_constants.dispatch_item_offset = item_offset;
      stage_constants.dispatch_item_count = item_count;
      stage_constants.work_queue_index = work_queue_index;
      ctx.cmd_set_pipeline(cmd, _pipelines[static_cast<uint32_t>(stage)]);
      ctx.cmd_push_constants(cmd, &stage_constants, sizeof(stage_constants));
      const uint32_t timing_end_query = begin_kernel_timing(cmd, stage);
      ctx.cmd_dispatch(cmd, {.group_count_x = item_count, .group_count_y = 1u, .group_count_z = 1u});
      end_kernel_timing(cmd, timing_end_query);
    };
    const auto dispatch_upbp_light_init = [&](RHICommandBuffer cmd) {
      const uint32_t group_count_x = divide_round_up(_upbp.light_batch_count, 64u);
      GPURTConstants stage_constants = constants;
      stage_constants.render_window_width = group_count_x * 8u;
      stage_constants.render_window_height = 8u;
      stage_constants.dispatch_item_count = _upbp.light_batch_count;
      ctx.cmd_set_pipeline(cmd, _pipelines[static_cast<uint32_t>(PipelineStage::InitLightPath0)]);
      ctx.cmd_push_constants(cmd, &stage_constants, sizeof(stage_constants));
      const uint32_t timing_end_query = begin_kernel_timing(cmd, PipelineStage::InitLightPath0);
      ctx.cmd_dispatch(cmd, {.group_count_x = group_count_x, .group_count_y = 1u, .group_count_z = 1u});
      end_kernel_timing(cmd, timing_end_query);
    };
    const auto dispatch_stage_indirect = [&](RHICommandBuffer cmd, PipelineStage stage, uint64_t argument_buffer_offset, uint32_t path_iteration) {
      GPURTConstants stage_constants = constants;
      stage_constants.path_iteration = path_iteration;
      ctx.cmd_set_pipeline(cmd, _pipelines[static_cast<uint32_t>(stage)]);
      ctx.cmd_push_constants(cmd, &stage_constants, sizeof(stage_constants));
      const uint32_t timing_end_query = begin_kernel_timing(cmd, stage);
      ctx.cmd_dispatch_indirect(cmd, _wavefront_dispatch_args_buffer, argument_buffer_offset);
      end_kernel_timing(cmd, timing_end_query);
    };
    const auto dispatch_stage_work_queue_indirect = [&](RHICommandBuffer cmd, PipelineStage stage, uint64_t argument_buffer_offset, uint32_t path_iteration,
                                                      uint32_t work_queue_index) {
      GPURTConstants stage_constants = constants;
      stage_constants.path_iteration = path_iteration;
      stage_constants.work_queue_index = work_queue_index;
      ctx.cmd_set_pipeline(cmd, _pipelines[static_cast<uint32_t>(stage)]);
      ctx.cmd_push_constants(cmd, &stage_constants, sizeof(stage_constants));
      const uint32_t timing_end_query = begin_kernel_timing(cmd, stage);
      ctx.cmd_dispatch_indirect(cmd, _wavefront_dispatch_args_buffer, argument_buffer_offset);
      end_kernel_timing(cmd, timing_end_query);
    };
    const auto dispatch_stage_window = [&](RHICommandBuffer cmd, PipelineStage stage, uint64_t argument_buffer_offset, uint32_t item_offset, uint32_t item_count,
                                         uint32_t path_iteration) {
      ctx.cmd_set_pipeline(cmd, _pipelines[static_cast<uint32_t>(stage)]);
      GPURTConstants stage_constants = constants;
      stage_constants.path_iteration = path_iteration;
      stage_constants.dispatch_item_offset = item_offset;
      stage_constants.dispatch_item_count = item_count;
      stage_constants.work_queue_index = kGPUWavefrontMaterialQueueDielectric;
      ctx.cmd_push_constants(cmd, &stage_constants, sizeof(stage_constants));
      const uint32_t timing_end_query = begin_kernel_timing(cmd, stage);
      ctx.cmd_dispatch_indirect(cmd, _wavefront_dispatch_args_buffer, argument_buffer_offset);
      end_kernel_timing(cmd, timing_end_query);
    };
    const auto barrier_wavefront_buffers = [&](RHICommandBuffer cmd) {
      ctx.cmd_compute_barrier(cmd);
    };

    const RHIDispatchDesc scalar_dispatch = {
      .group_count_x = 1u,
      .group_count_y = 1u,
      .group_count_z = 1u,
    };
    const RHIDispatchDesc spectral_values_dispatch = {
      .group_count_x = (std::max(1u, _gpu_scene.spectrum_count) + 63u) / 64u,
      .group_count_y = 1u,
      .group_count_z = 1u,
    };
    const uint32_t heavy_continuation_chunk_count = 1u + ((wavefront_buffer_path_capacity - 1u) / kGPUWavefrontHeavyContinuationChunkSize);
    const RHIDispatchDesc build_dispatch_args_dispatch = {
      .group_count_x = (heavy_continuation_chunk_count + 63u) / 64u,
      .group_count_y = 1u,
      .group_count_z = 1u,
    };
    const auto rebuild_dispatch_args = [&](RHICommandBuffer cmd, uint32_t path_iteration) {
      barrier_wavefront_buffers(cmd);
      ctx.cmd_buffer_barrier(cmd, _wavefront_dispatch_args_buffer, _wavefront_dispatch_args_buffer_state, RHIResourceState::General);
      _wavefront_dispatch_args_buffer_state = RHIResourceState::General;
      dispatch_stage(cmd, PipelineStage::BuildDispatchArgs, build_dispatch_args_dispatch, path_iteration);
      ctx.cmd_buffer_barrier(cmd, _wavefront_dispatch_args_buffer, RHIResourceState::General, RHIResourceState::IndirectArgument);
      _wavefront_dispatch_args_buffer_state = RHIResourceState::IndirectArgument;
    };
    const uint32_t scene_max_path_length = std::max(1u, scene.data().options.max_path_length);
    const GPUIntegratorMode integrator_mode = static_cast<GPUIntegratorMode>(_integrator_mode);
    const uint32_t wavefront_hard_iteration_cap = scene_max_path_length;
    std::vector<SubmittedCommand> submitted_commands = {};
    submitted_commands.reserve(8u);
    std::vector<uint64_t> timestamp_values = {};
    if (capture_kernel_timings) {
      timestamp_values.reserve(timestamp_query_capacity);
    }
    RHIResult deferred_command_wait_result = RHIResult::Success;
    const auto wait_and_destroy_submitted_commands = [&](const char* stage_name) {
      RHIResult wait_result = deferred_command_wait_result;
      deferred_command_wait_result = RHIResult::Success;
      for (auto& submitted_command : submitted_commands) {
        const RHIResult command_wait_result = ctx.wait_for_command_buffer(submitted_command.command_buffer);
        if ((wait_result == RHIResult::Success) && (command_wait_result != RHIResult::Success)) {
          wait_result = command_wait_result;
        }
        if ((command_wait_result == RHIResult::Success) && (submitted_command.timestamp_query_count > 0u)) {
          timestamp_values.resize(submitted_command.timestamp_query_count);
          const RHIResult timestamp_result = ctx.read_timestamps(submitted_command.command_buffer, 0u, submitted_command.timestamp_query_count, timestamp_values.data());
          if (timestamp_result == RHIResult::Success) {
            for (const KernelTimestampSpan& span : submitted_command.timestamp_spans) {
              const uint64_t begin_tick = timestamp_values[span.begin_query];
              const uint64_t end_tick = timestamp_values[span.end_query];
              const uint64_t elapsed_tick_count = (end_tick >= begin_tick) ? (end_tick - begin_tick) : 0u;
              KernelTimingAccumulator& timing = _kernel_timing_accumulators[static_cast<uint32_t>(span.stage)];
              timing.dispatch_count += 1u;
              timing.total_ms += static_cast<double>(elapsed_tick_count) * timestamp_tick_to_ms;
            }
          } else {
            _kernel_timing_stats.dropped_dispatch_count += static_cast<uint64_t>(submitted_command.timestamp_spans.size());
            log::warning("GPU RT: failed to read kernel timestamps after %s (%u)", stage_name, static_cast<uint32_t>(timestamp_result));
          }
        } else if (command_wait_result != RHIResult::Success) {
          _kernel_timing_stats.dropped_dispatch_count += static_cast<uint64_t>(submitted_command.timestamp_spans.size());
        }
      }
      if (wait_result != RHIResult::Success) {
        log::warning("GPU RT: command wait failed after %s (%u)", stage_name, static_cast<uint32_t>(wait_result));
      }
      for (const auto& submitted_command : submitted_commands) {
        ctx.destroy_command_buffer(submitted_command.command_buffer);
      }
      submitted_commands.clear();
      if (capture_kernel_timings) {
        update_kernel_timing_stats();
      }
      return wait_result;
    };
    const auto record_and_submit = [&](const auto& record_commands) {
      constexpr size_t kPendingCommandLimit = 16u;
      constexpr size_t kTimestampedPendingCommandLimit = 1u;
      const bool bound_pending_commands = capture_kernel_timings || (upbp_mode && (_wavefront_render_step == WavefrontRenderStep::UPBPEvaluateLightBatch));
      const size_t pending_command_limit = capture_kernel_timings ? kTimestampedPendingCommandLimit : kPendingCommandLimit;
      if (bound_pending_commands && (submitted_commands.size() >= pending_command_limit)) {
        const RHIResult batch_wait_result = wait_and_destroy_submitted_commands(capture_kernel_timings ? "kernel timing batch" : "UPBP density batch");
        if (batch_wait_result != RHIResult::Success) {
          deferred_command_wait_result = batch_wait_result;
          return;
        }
      }
      if (deferred_command_wait_result != RHIResult::Success) {
        return;
      }

      SubmittedCommand submitted_command = {};
      submitted_command.command_buffer = ctx.get_command_buffer();
      if (capture_kernel_timings) {
        submitted_command.timestamp_spans.reserve(std::min(timestamp_query_capacity / 2u, 64u));
      }
      const RHICommandBuffer cmd = submitted_command.command_buffer;
      ctx.command_buffer_begin(cmd);
      if (capture_kernel_timings) {
        ctx.cmd_reset_timestamps(cmd, 0u, timestamp_query_capacity);
        active_submitted_command = &submitted_command;
      }
      record_commands(cmd);
      active_submitted_command = nullptr;
      ctx.command_buffer_end(cmd);
      ctx.submit_command_buffer({cmd});
      submitted_commands.push_back(std::move(submitted_command));
    };
    const auto submit_stage_chunked = [&](PipelineStage stage, uint32_t item_count, uint32_t path_iteration, bool from_camera) {
      const uint64_t argument_base_offset =
        kGPUWavefrontFixedDispatchArgsBufferSize + static_cast<uint64_t>(from_camera ? 0u : heavy_continuation_chunk_count) * kGPUWavefrontDispatchArgsStride;
      for (uint32_t item_offset = 0u; item_offset < item_count; item_offset += kGPUWavefrontHeavyContinuationChunkSize) {
        const uint32_t chunk_count = std::min(kGPUWavefrontHeavyContinuationChunkSize, item_count - item_offset);
        const uint64_t argument_buffer_offset =
          argument_base_offset + static_cast<uint64_t>(item_offset / kGPUWavefrontHeavyContinuationChunkSize) * kGPUWavefrontDispatchArgsStride;
        record_and_submit([&](RHICommandBuffer cmd) {
          barrier_wavefront_buffers(cmd);
          dispatch_stage_window(cmd, stage, argument_buffer_offset, item_offset, chunk_count, path_iteration);
          barrier_wavefront_buffers(cmd);
        });
      }
    };
    const auto submit_upbp_density_stage_chunked = [&](PipelineStage stage, uint32_t item_count, uint32_t work_queue_index) {
      for (uint32_t item_offset = 0u; item_offset < item_count; item_offset += kUPBPDensityLinearDispatchChunkSize) {
        const uint32_t chunk_count = std::min(kUPBPDensityLinearDispatchChunkSize, item_count - item_offset);
        record_and_submit([&](RHICommandBuffer cmd) {
          barrier_wavefront_buffers(cmd);
          dispatch_stage_range(cmd, stage, item_offset, chunk_count, 0u, work_queue_index);
          barrier_wavefront_buffers(cmd);
        });
      }
    };
    const auto submit_upbp_density_linear_queries = [&](PipelineStage stage, uint32_t item_count, uint32_t work_queue_index) {
      for (uint32_t item_offset = 0u; item_offset < item_count; item_offset += kUPBPDensityQueryDispatchChunkSize) {
        const uint32_t chunk_count = std::min(kUPBPDensityQueryDispatchChunkSize, item_count - item_offset);
        record_and_submit([&](RHICommandBuffer cmd) {
          barrier_wavefront_buffers(cmd);
          dispatch_stage_range(cmd, stage, item_offset, chunk_count, 0u, work_queue_index);
          barrier_wavefront_buffers(cmd);
        });
      }
    };
    const auto submit_upbp_density_query_groups = [&](PipelineStage stage, uint32_t item_count, uint32_t work_queue_index) {
      for (uint32_t item_offset = 0u; item_offset < item_count; item_offset += kDensityQueryGroupDispatchChunkSize) {
        const uint32_t chunk_count = std::min(kDensityQueryGroupDispatchChunkSize, item_count - item_offset);
        record_and_submit([&](RHICommandBuffer cmd) {
          barrier_wavefront_buffers(cmd);
          dispatch_stage_query_groups(cmd, stage, item_offset, chunk_count, constants.path_iteration, work_queue_index);
          barrier_wavefront_buffers(cmd);
        });
      }
    };
    const bool enable_camera_path = gpu_integrator_feature_enabled(_integrator_features, GPUIntegratorFeatures::CameraPath);
    const bool enable_light_path = gpu_integrator_feature_enabled(_integrator_features, GPUIntegratorFeatures::LightPath);
    const bool enable_direct_hit = gpu_integrator_feature_enabled(_integrator_features, GPUIntegratorFeatures::DirectHit) && (upbp_mode == false);
    const bool enable_connect_to_light = gpu_integrator_feature_enabled(_integrator_features, GPUIntegratorFeatures::ConnectToLight);
    const bool enable_connect_to_camera =
      gpu_integrator_feature_enabled(_integrator_features, GPUIntegratorFeatures::ConnectToCamera) && ((upbp_mode == false) || (_wavefront_tile_index == 0u));
    const bool enable_connect_vertices = gpu_integrator_feature_enabled(_integrator_features, GPUIntegratorFeatures::ConnectVertices);
    const bool enable_merge_vertices = gpu_integrator_feature_enabled(_integrator_features, GPUIntegratorFeatures::MergeVertices) && (upbp_mode == false);
    const bool store_complete_light_history = (enable_connect_vertices || enable_merge_vertices) && (integrator_mode != GPUIntegratorMode::UPBP);
    const bool phase_light_before_camera = ((integrator_mode == GPUIntegratorMode::BDPTFull) || (integrator_mode == GPUIntegratorMode::VCM) || upbp_mode) && enable_camera_path &&
                                           enable_light_path && (store_complete_light_history || upbp_mode);
    const bool has_various_continue = material_compile_mask_has_various_continue(_material_compile_mask);
    const bool has_various_connect = material_compile_mask_has_various_connect(_material_compile_mask);
    const bool has_plastic = material_compile_mask_has(_material_compile_mask, MaterialClass::Plastic);
    const bool has_conductor = material_compile_mask_has_conductor_stage(_material_compile_mask);
    const bool has_dielectric = material_compile_mask_has(_material_compile_mask, MaterialClass::Dielectric);
    const bool has_connectible_conductor = material_compile_mask_has_connectible_conductor(_material_compile_mask);
    const bool has_connectible_dielectric = material_compile_mask_has_connectible_dielectric(_material_compile_mask);
    const bool has_connectible_material = has_various_connect || has_plastic || has_connectible_conductor || has_connectible_dielectric;
    const bool has_thinfilm = material_compile_mask_has(_material_compile_mask, MaterialClass::Thinfilm);
    const bool use_material_work_queues = material_compile_mask_work_queue_count(_material_compile_mask) > 1u;
    const auto dispatch_stage_material_indirect = [&](RHICommandBuffer cmd, PipelineStage stage, bool from_camera, uint32_t material_queue_index, uint32_t path_iteration) {
      if (use_material_work_queues) {
        dispatch_stage_work_queue_indirect(cmd, stage, material_dispatch_args_offset(from_camera, material_queue_index), path_iteration, material_queue_index);
        return;
      }

      dispatch_stage_indirect(cmd, stage, from_camera ? kGPUWavefrontCameraDispatchArgsOffset : kGPUWavefrontLightDispatchArgsOffset, path_iteration);
    };
    const auto dispatch_vcm_merge_material = [&](RHICommandBuffer cmd, PipelineStage stage, uint32_t material_queue_index, uint32_t path_iteration) {
      const bool cooperative = (stage == PipelineStage::VCMMergeConductor) || (stage == PipelineStage::VCMMergeDielectric);
      const uint32_t chunk_size = cooperative ? kDensityQueryGroupDispatchChunkSize : kGPUWavefrontHeavyContinuationChunkSize;
      for (uint32_t item_offset = 0u; item_offset < _wavefront_camera_queue_count; item_offset += chunk_size) {
        const uint32_t item_count = std::min(chunk_size, _wavefront_camera_queue_count - item_offset);
        if (cooperative) {
          dispatch_stage_query_groups(cmd, stage, item_offset, item_count, path_iteration, material_queue_index);
        } else {
          dispatch_stage_range(cmd, stage, item_offset, item_count, path_iteration, material_queue_index);
        }
        barrier_wavefront_buffers(cmd);
      }
    };
    const uint32_t light_history_bounces = (store_complete_light_history || upbp_mode) ? scene_max_path_length : kWavefrontLightHistoryBounces;
    const uint32_t render_pixel_count = render_dim.x * render_dim.y;
    uint32_t initial_camera_queue_count = enable_camera_path ? render_pixel_count : 0u;
    const uint32_t initial_light_queue_count = enable_light_path ? (upbp_mode ? _upbp.light_batch_count : render_pixel_count) : 0u;
    const uint32_t batch_queue_readback_interval = _batch_coarse_progress ? kWavefrontCoarseQueueReadbackInterval : 1u;
    if ((_sample_index == 0u) && (_frame_index == 0u)) {
      log::info("GPU path mode: %s", gpu_integrator_mode_to_string(integrator_mode));
    }
    const auto finalize_wavefront_sample = [&]() {
      const bool final_tile = (_wavefront_tile_index + 1u) >= wavefront_tile_count_value;
      if (final_tile) {
        record_and_submit([&](RHICommandBuffer cmd) {
          ctx.cmd_texture_barrier(cmd, render_output_texture, render_output_texture_state, RHIResourceState::General);
          barrier_wavefront_buffers(cmd);
          GPURTConstants finalize_constants = constants;
          RHIDispatchDesc finalize_dispatch = film_dispatch;
          if (use_wavefront_tiling) {
            finalize_constants.render_window_origin_x = base_render_origin.x;
            finalize_constants.render_window_origin_y = base_render_origin.y;
            finalize_constants.render_window_width = base_render_dim.x;
            finalize_constants.render_window_height = base_render_dim.y;
            finalize_dispatch = {
              .group_count_x = (base_render_dim.x + 7u) / 8u,
              .group_count_y = (base_render_dim.y + 7u) / 8u,
              .group_count_z = 1u,
            };
          }
          finalize_constants.path_iteration = _wavefront_hard_iteration_cap;
          ctx.cmd_set_pipeline(cmd, _pipelines[static_cast<uint32_t>(PipelineStage::FinalizeSample)]);
          ctx.cmd_push_constants(cmd, &finalize_constants, sizeof(finalize_constants));
          const uint32_t timing_end_query = begin_kernel_timing(cmd, PipelineStage::FinalizeSample);
          ctx.cmd_dispatch(cmd, finalize_dispatch);
          end_kernel_timing(cmd, timing_end_query);
          ctx.cmd_texture_barrier(cmd, render_output_texture, RHIResourceState::General, RHIResourceState::ShaderReadOnly);
        });
      }

      const RHIResult finalize_result = wait_and_destroy_submitted_commands(final_tile ? "finalize sample submit" : "finish tile submit");
      if (finalize_result != RHIResult::Success) {
        set_runtime_failure(
          std::string(final_tile ? "GPU RT finalize sample submit failed (" : "GPU RT tile completion failed (") + std::to_string(static_cast<uint32_t>(finalize_result)) + ")");
        _wavefront_camera_queue_count = 0u;
        _wavefront_light_queue_count = 0u;
        return false;
      }
      if (final_tile) {
        render_output_texture_state = RHIResourceState::ShaderReadOnly;
      }
      _wavefront_render_step = WavefrontRenderStep::InitSample;
      _wavefront_path_iteration = 0u;
      _wavefront_hard_iteration_cap = 0u;
      _wavefront_camera_queue_count = 0u;
      _wavefront_light_queue_count = 0u;
      _wavefront_light_max_path_length = 0u;
      _wavefront_connect_light_vertex_length = 0u;
      _wavefront_connect_light_history_bounces = 0u;
      _wavefront_camera_phase_initialized = false;
      _wavefront_light_vertex_sample_peak_count = std::max(_wavefront_light_vertex_sample_peak_count, _wavefront_light_vertex_reserved_count);
      _wavefront_vcm_light_vertex_count = 0u;
      if (final_tile) {
        _wavefront_tile_index = 0u;
        completed_sample = true;
      } else {
        _wavefront_tile_index += 1u;
      }
      dispatch_submit_ms = elapsed_ms(dispatch_submit_begin, std::chrono::steady_clock::now());
      return true;
    };
    const auto snapshot_upbp_matching_light_history = [&](const uint32_t* upbp_counters) {
      const uint32_t light_vertex_count = std::min(upbp_counters[GPUUPBPCounterIndex::LightVertex], _upbp.resources.light_vertex_capacity);
      const uint64_t light_vertex_bytes = static_cast<uint64_t>(light_vertex_count) * kGPUUPBPBPTVertexStride;
      const uint64_t light_path_state_bytes = static_cast<uint64_t>(_upbp.light_batch_count) * kGPUUPBPBPTPathStateStride;
      const RHIBufferUsage bpt_usage = RHIBufferUsage::Storage;
      const bool vertex_buffer_reused = _upbp.bpt_light_vertex_buffer.handle.valid() && (_upbp.bpt_light_vertex_buffer.size >= light_vertex_bytes);
      const bool path_state_buffer_reused = _upbp.bpt_light_path_state_buffer.handle.valid() && (_upbp.bpt_light_path_state_buffer.size >= light_path_state_bytes);
      if ((ensure_storage_buffer_capacity(device, std::max<uint64_t>(light_vertex_bytes, sizeof(uint32_t)), bpt_usage, _upbp.bpt_light_vertex_buffer.handle,
             _upbp.bpt_light_vertex_buffer.size, _upbp.bpt_light_vertex_buffer.descriptor_index, "upbp_bpt_light_vertex") == false) ||
          (ensure_storage_buffer_capacity(device, std::max<uint64_t>(light_path_state_bytes, sizeof(uint32_t)), bpt_usage, _upbp.bpt_light_path_state_buffer.handle,
             _upbp.bpt_light_path_state_buffer.size, _upbp.bpt_light_path_state_buffer.descriptor_index, "upbp_bpt_light_path_state") == false)) {
        set_runtime_failure("GPU UPBP failed to allocate compact matching light history");
        return false;
      }
      _upbp.bpt_light_vertex_state = vertex_buffer_reused ? _upbp.bpt_light_vertex_state : RHIResourceState::Undefined;
      _upbp.bpt_light_path_state_state = path_state_buffer_reused ? _upbp.bpt_light_path_state_state : RHIResourceState::Undefined;
      _upbp.resources.bpt_light_vertex_buffer = _upbp.bpt_light_vertex_buffer.descriptor_index;
      _upbp.resources.bpt_light_path_state_buffer = _upbp.bpt_light_path_state_buffer.descriptor_index;
      if (update_upbp_iteration_resources(device, scene, upbp_global_path_count) == false) {
        set_runtime_failure("GPU UPBP failed to bind compact matching light history");
        return false;
      }
      record_and_submit([&](RHICommandBuffer cmd) {
        barrier_wavefront_buffers(cmd);
        if (light_vertex_bytes > 0u) {
          ctx.cmd_buffer_barrier(cmd, _upbp.bpt_light_vertex_buffer.handle, _upbp.bpt_light_vertex_state, RHIResourceState::General);
          dispatch_stage_range(cmd, PipelineStage::UPBPDensityCompact, 0u, light_vertex_count, 0u, GPUUPBPDensityCompactMode::BPTVertices);
          ctx.cmd_buffer_barrier(cmd, _upbp.bpt_light_vertex_buffer.handle, RHIResourceState::General, RHIResourceState::General);
          _upbp.bpt_light_vertex_state = RHIResourceState::General;
        }
        if (light_path_state_bytes > 0u) {
          ctx.cmd_buffer_barrier(cmd, _upbp.bpt_light_path_state_buffer.handle, _upbp.bpt_light_path_state_state, RHIResourceState::General);
          dispatch_stage_range(cmd, PipelineStage::UPBPDensityCompact, 0u, _upbp.light_batch_count, 0u, GPUUPBPDensityCompactMode::BPTPathStates);
          ctx.cmd_buffer_barrier(cmd, _upbp.bpt_light_path_state_buffer.handle, RHIResourceState::General, RHIResourceState::General);
          _upbp.bpt_light_path_state_state = RHIResourceState::General;
        }
      });
      const RHIResult copy_result = wait_and_destroy_submitted_commands("UPBP compact matching light history");
      if (copy_result != RHIResult::Success) {
        set_runtime_failure("GPU UPBP compact matching light history failed (" + std::to_string(static_cast<uint32_t>(copy_result)) + ")");
        return false;
      }
      return true;
    };
    const auto initialize_deferred_camera_phase = [&]() {
      _wavefront_path_iteration = 0u;
      _wavefront_camera_queue_count = initial_camera_queue_count;
      _wavefront_light_queue_count = 0u;
      _wavefront_camera_phase_initialized = true;
      _wavefront_vcm_light_vertex_count = std::min(_wavefront_light_vertex_reserved_count, _wavefront_resources.light_vertex_capacity);
      constants.vcm_light_vertex_count = _wavefront_vcm_light_vertex_count;
      record_and_submit([&](RHICommandBuffer cmd) {
        barrier_wavefront_buffers(cmd);
        if (enable_merge_vertices) {
          const uint32_t grid_clear_groups = divide_round_up(constants.vcm_grid_mask + 1u, 256u);
          const uint32_t grid_build_groups = divide_round_up(constants.vcm_light_vertex_count, 256u);
          const RHIDispatchDesc grid_clear_dispatch = {
            .group_count_x = std::min(grid_clear_groups, 65535u),
            .group_count_y = divide_round_up(grid_clear_groups, 65535u),
            .group_count_z = 1u,
          };
          const RHIDispatchDesc grid_build_dispatch = {
            .group_count_x = std::min(grid_build_groups, 65535u),
            .group_count_y = divide_round_up(grid_build_groups, 65535u),
            .group_count_z = 1u,
          };
          dispatch_stage(cmd, PipelineStage::VCMGridClear, grid_clear_dispatch, 0u);
          barrier_wavefront_buffers(cmd);
          dispatch_stage(cmd, PipelineStage::VCMGridBuild, grid_build_dispatch, 0u);
          barrier_wavefront_buffers(cmd);
          uint32_t scan_offset = 0u;
          uint32_t scan_count = constants.vcm_grid_mask + 1u;
          for (;;) {
            GPURTConstants scan_constants = constants;
            scan_constants.dispatch_item_offset = scan_offset;
            scan_constants.dispatch_item_count = scan_count;
            const uint32_t scan_groups = divide_round_up(scan_count, 64u);
            ctx.cmd_set_pipeline(cmd, _pipelines[static_cast<uint32_t>(PipelineStage::VCMGridPrefix)]);
            ctx.cmd_push_constants(cmd, &scan_constants, sizeof(scan_constants));
            const uint32_t timing_end_query = begin_kernel_timing(cmd, PipelineStage::VCMGridPrefix);
            ctx.cmd_dispatch(cmd, {.group_count_x = std::min(scan_groups, 65535u), .group_count_y = divide_round_up(scan_groups, 65535u), .group_count_z = 1u});
            end_kernel_timing(cmd, timing_end_query);
            barrier_wavefront_buffers(cmd);
            if (scan_groups == 1u) {
              break;
            }
            scan_offset = (scan_offset == 0u) ? (3u * scan_count) : (scan_offset + scan_count);
            scan_count = scan_groups;
          }
          dispatch_stage(cmd, PipelineStage::VCMGridScatter, grid_build_dispatch, 0u);
          barrier_wavefront_buffers(cmd);
        }
        dispatch_stage(cmd, PipelineStage::InitCameraPath0, film_dispatch, 0u);
        barrier_wavefront_buffers(cmd);
      });

      const RHIResult init_result = wait_and_destroy_submitted_commands("deferred camera init submit");
      if (init_result != RHIResult::Success) {
        set_runtime_failure("GPU RT deferred camera init submit failed (" + std::to_string(static_cast<uint32_t>(init_result)) + ")");
        _wavefront_camera_queue_count = 0u;
        return false;
      }
      dispatch_submit_ms = elapsed_ms(dispatch_submit_begin, std::chrono::steady_clock::now());
      return true;
    };
    const auto complete_upbp_trace_batch = [&]() {
      _wavefront_camera_queue_count = 0u;
      _wavefront_light_queue_count = 0u;
      _wavefront_path_iteration = 0u;
      _wavefront_render_step = WavefrontRenderStep::UPBPEvaluateLightBatch;
    };

    bool finished_current_tile = false;
    bool wavefront_auto_measurement_valid = true;
    uint32_t executed_wavefront_steps = 0u;
    const uint32_t wavefront_step_budget = std::max(1u, _wavefront_steps_per_render);
    const auto wavefront_batch_begin = std::chrono::steady_clock::now();
    for (uint32_t wavefront_step_index = 0u; (wavefront_step_index < wavefront_step_budget) && (finished_current_tile == false); ++wavefront_step_index) {
      executed_wavefront_steps += 1u;
      if (_wavefront_render_step == WavefrontRenderStep::InitSample) {
        if (_wavefront_tile_index == 0u) {
          _wavefront_light_vertex_sample_peak_count = 0u;
        }
        _wavefront_path_iteration = 0u;
        _wavefront_hard_iteration_cap = wavefront_hard_iteration_cap;
        _wavefront_camera_phase_initialized = enable_camera_path && (phase_light_before_camera == false);
        _wavefront_camera_queue_count = _wavefront_camera_phase_initialized ? initial_camera_queue_count : 0u;
        _wavefront_light_queue_count = initial_light_queue_count;
        _wavefront_light_max_path_length = 0u;
        _wavefront_connect_light_vertex_length = 0u;
        _wavefront_connect_light_history_bounces = 0u;
        _wavefront_light_vertex_reserved_count = wavefront_path_capacity;
        _wavefront_vcm_light_vertex_count = 0u;
        constants.vcm_light_vertex_count = 0u;

        record_and_submit([&](RHICommandBuffer cmd) {
          barrier_wavefront_buffers(cmd);
          if (upbp_mode) {
            dispatch_stage_mode(cmd, PipelineStage::UPBPClear, scalar_dispatch, 0u, GPUUPBPClearMode::All);
            barrier_wavefront_buffers(cmd);
          }
          if (use_spectral_values) {
            dispatch_stage(cmd, PipelineStage::PrepareSpectralValues, spectral_values_dispatch, 0u);
            barrier_wavefront_buffers(cmd);
          }
          dispatch_prepare_sample(cmd);
          barrier_wavefront_buffers(cmd);
          if (_wavefront_camera_phase_initialized) {
            dispatch_stage(cmd, PipelineStage::InitCameraPath0, film_dispatch, 0u);
            barrier_wavefront_buffers(cmd);
          }
          if (enable_light_path) {
            if (upbp_mode) {
              dispatch_upbp_light_init(cmd);
            } else {
              dispatch_stage(cmd, PipelineStage::InitLightPath0, film_dispatch, 0u);
            }
            barrier_wavefront_buffers(cmd);
          }
        });
        const RHIResult init_result = wait_and_destroy_submitted_commands("init sample submit");
        if (init_result != RHIResult::Success) {
          set_runtime_failure("GPU RT init sample submit failed (" + std::to_string(static_cast<uint32_t>(init_result)) + ")");
          _wavefront_camera_queue_count = 0u;
          _wavefront_light_queue_count = 0u;
          return;
        }
        _wavefront_render_step = WavefrontRenderStep::TraceBounce;
        dispatch_submit_ms = elapsed_ms(dispatch_submit_begin, std::chrono::steady_clock::now());
      } else if (_wavefront_render_step == WavefrontRenderStep::TraceBounce) {
        const bool waiting_for_deferred_camera_phase = phase_light_before_camera && enable_camera_path && (_wavefront_camera_phase_initialized == false);
        if (waiting_for_deferred_camera_phase && ((_wavefront_light_queue_count == 0u) || (_wavefront_path_iteration >= _wavefront_hard_iteration_cap))) {
          if (upbp_mode) {
            complete_upbp_trace_batch();
          } else {
            if (initialize_deferred_camera_phase() == false) {
              return;
            }
          }
        } else if ((_wavefront_camera_queue_count == 0u) && (_wavefront_light_queue_count == 0u)) {
          if (upbp_mode) {
            complete_upbp_trace_batch();
          } else {
            if (finalize_wavefront_sample() == false) {
              return;
            }
          }
        } else if (_wavefront_path_iteration >= _wavefront_hard_iteration_cap) {
          if (upbp_mode) {
            complete_upbp_trace_batch();
          } else {
            if (finalize_wavefront_sample() == false) {
              return;
            }
          }
        } else {
          const uint32_t path_iteration = _wavefront_path_iteration;
          const uint32_t current_path_length = path_iteration + 1u;
          if (_wavefront_camera_phase_initialized && (_wavefront_camera_queue_count > 0u)) {
            _wavefront_max_observed_camera_path_length = std::max(_wavefront_max_observed_camera_path_length, current_path_length);
          }
          if (_wavefront_light_queue_count > 0u) {
            _wavefront_max_observed_light_path_length = std::max(_wavefront_max_observed_light_path_length, current_path_length);
          }
          const bool connect_light_batch_in_progress = _wavefront_connect_light_vertex_length > 0u;
          if ((connect_light_batch_in_progress == false) && store_complete_light_history && (_wavefront_light_queue_count > 0u)) {
            const uint64_t required_light_vertex_capacity = static_cast<uint64_t>(_wavefront_light_vertex_reserved_count) + static_cast<uint64_t>(_wavefront_light_queue_count);
            if (required_light_vertex_capacity > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
              set_runtime_failure("GPU RT light path history exceeded the addressable buffer capacity");
              _wavefront_camera_queue_count = 0u;
              _wavefront_light_queue_count = 0u;
              return;
            }
            if (required_light_vertex_capacity > _wavefront_resources.light_vertex_capacity) {
              wavefront_auto_measurement_valid = false;
              if (submitted_commands.empty() == false) {
                const RHIResult pending_result = wait_and_destroy_submitted_commands("light history growth");
                if (pending_result != RHIResult::Success) {
                  set_runtime_failure("GPU RT light history synchronization failed (" + std::to_string(static_cast<uint32_t>(pending_result)) + ")");
                  _wavefront_camera_queue_count = 0u;
                  _wavefront_light_queue_count = 0u;
                  return;
                }
              }
              if (ensure_light_vertex_capacity(ctx, static_cast<uint32_t>(required_light_vertex_capacity)) == false) {
                set_runtime_failure("GPU RT light path history exceeded the addressable buffer capacity");
                _wavefront_camera_queue_count = 0u;
                _wavefront_light_queue_count = 0u;
                return;
              }
            }
          }
          const bool continue_paths = (path_iteration + 1u) < _wavefront_hard_iteration_cap;
          const bool render_step_budget_end = (wavefront_step_index + 1u) >= wavefront_step_budget;
          const bool queue_readback_due = (((path_iteration + 1u) % batch_queue_readback_interval) == 0u) || (continue_paths == false) || render_step_budget_end;
          const bool deferred_camera_light_terminal_step =
            phase_light_before_camera && enable_camera_path && enable_light_path && (_wavefront_camera_phase_initialized == false) && (continue_paths == false);
          const bool copy_upbp_terminal_light_path_length = upbp_mode && deferred_camera_light_terminal_step;
          const bool copy_queue_counts = (continue_paths && queue_readback_due) || deferred_camera_light_terminal_step;
          const bool copy_light_vertex_count = store_complete_light_history && (_wavefront_light_queue_count > 0u) && queue_readback_due;
          const RHIBindlessHandle next_camera_queue_buffer = ((path_iteration & 1u) == 0u) ? _camera_queue_b_buffer : _camera_queue_a_buffer;
          const RHIBindlessHandle next_light_queue_buffer = ((path_iteration & 1u) == 0u) ? _light_queue_b_buffer : _light_queue_a_buffer;

          RHIResult trace_step_result = RHIResult::Success;

          if (connect_light_batch_in_progress == false) {
            record_and_submit([&](RHICommandBuffer cmd) {
              rebuild_dispatch_args(cmd, path_iteration);
              if (_wavefront_light_queue_count > 0u) {
                if (enable_connect_to_camera) {
                  dispatch_stage_indirect(cmd, PipelineStage::LightConnectCameraClear, kGPUWavefrontLightDispatchArgsOffset, path_iteration);
                  barrier_wavefront_buffers(cmd);
                }
                dispatch_stage_indirect(cmd, PipelineStage::TraceLight, kGPUWavefrontLightDispatchArgsOffset, path_iteration);
                barrier_wavefront_buffers(cmd);
                if (use_material_work_queues) {
                  dispatch_stage_work_queue_indirect(cmd, PipelineStage::LightSurfaceClassify, kGPUWavefrontLightDispatchArgsOffset, path_iteration,
                    kGPUWavefrontMaterialQueueVarious);
                } else {
                  dispatch_stage_indirect(cmd, PipelineStage::LightSurfaceClassify, kGPUWavefrontLightDispatchArgsOffset, path_iteration);
                }
                barrier_wavefront_buffers(cmd);
              }
              if (_wavefront_camera_queue_count > 0u) {
                dispatch_stage_indirect(cmd, PipelineStage::TraceCamera, kGPUWavefrontCameraDispatchArgsOffset, path_iteration);
                barrier_wavefront_buffers(cmd);
                if (use_material_work_queues) {
                  dispatch_stage_work_queue_indirect(cmd, PipelineStage::CameraSurfaceClassify, kGPUWavefrontCameraDispatchArgsOffset, path_iteration,
                    kGPUWavefrontMaterialQueueVarious);
                } else {
                  dispatch_stage_indirect(cmd, PipelineStage::CameraSurfaceClassify, kGPUWavefrontCameraDispatchArgsOffset, path_iteration);
                }
                barrier_wavefront_buffers(cmd);
              }
              rebuild_dispatch_args(cmd, path_iteration);
            });
            const bool submit_non_dielectric_continue = (continue_paths || upbp_mode) && (has_various_continue || has_plastic || has_conductor || has_thinfilm);
            if (submit_non_dielectric_continue) {
              record_and_submit([&](RHICommandBuffer cmd) {
                barrier_wavefront_buffers(cmd);
                if (_wavefront_light_queue_count > 0u) {
                  if (has_various_continue) {
                    dispatch_stage_material_indirect(cmd, PipelineStage::LightContinuePrepareDiffuse, false, kGPUWavefrontMaterialQueueVarious, path_iteration);
                  }
                  if (has_plastic) {
                    dispatch_stage_material_indirect(cmd, PipelineStage::LightContinuePreparePlastic, false, kGPUWavefrontMaterialQueuePlastic, path_iteration);
                  }
                  if (has_conductor) {
                    dispatch_stage_material_indirect(cmd, PipelineStage::LightContinuePrepareConductor, false, kGPUWavefrontMaterialQueueConductor, path_iteration);
                  }
                  if (has_thinfilm) {
                    dispatch_stage_material_indirect(cmd, PipelineStage::LightContinuePrepareThinfilm, false, kGPUWavefrontMaterialQueueThinfilm, path_iteration);
                  }
                }
                if (_wavefront_camera_queue_count > 0u) {
                  if (has_various_continue) {
                    dispatch_stage_material_indirect(cmd, PipelineStage::CameraContinuePrepareDiffuse, true, kGPUWavefrontMaterialQueueVarious, path_iteration);
                  }
                  if (has_plastic) {
                    dispatch_stage_material_indirect(cmd, PipelineStage::CameraContinuePreparePlastic, true, kGPUWavefrontMaterialQueuePlastic, path_iteration);
                  }
                  if (has_conductor) {
                    dispatch_stage_material_indirect(cmd, PipelineStage::CameraContinuePrepareConductor, true, kGPUWavefrontMaterialQueueConductor, path_iteration);
                  }
                  if (has_thinfilm) {
                    dispatch_stage_material_indirect(cmd, PipelineStage::CameraContinuePrepareThinfilm, true, kGPUWavefrontMaterialQueueThinfilm, path_iteration);
                  }
                }
                barrier_wavefront_buffers(cmd);
              });
            }

            if ((continue_paths || upbp_mode) && has_dielectric && (_wavefront_light_queue_count > 0u)) {
              submit_stage_chunked(PipelineStage::LightContinuePrepareDielectric, _wavefront_light_queue_count, path_iteration, false);
            }

            if ((continue_paths || upbp_mode) && has_dielectric && (_wavefront_camera_queue_count > 0u)) {
              submit_stage_chunked(PipelineStage::CameraContinuePrepareDielectric, _wavefront_camera_queue_count, path_iteration, true);
            }
          }

          if ((connect_light_batch_in_progress == false) && enable_connect_vertices && has_connectible_material && (_wavefront_camera_queue_count > 0u)) {
            const uint32_t generated_light_history_bounces =
              (phase_light_before_camera && _wavefront_camera_phase_initialized) ? _wavefront_light_max_path_length : std::min(light_history_bounces, path_iteration + 1u);
            const uint32_t camera_path_length = path_iteration + 1u;
            const uint32_t maximum_connected_light_path_length = (scene_max_path_length > (camera_path_length + 1u)) ? (scene_max_path_length - camera_path_length - 1u) : 0u;
            _wavefront_connect_light_history_bounces = std::min(maximum_connected_light_path_length, std::min(light_history_bounces, generated_light_history_bounces));
            _wavefront_connect_light_vertex_length = _wavefront_connect_light_history_bounces;
          }
          const uint32_t connect_light_vertex_count = std::min(kGPUWavefrontConnectDispatchArgsCount, _wavefront_connect_light_vertex_length);
          const uint32_t remaining_connect_light_vertex_length = _wavefront_connect_light_vertex_length - connect_light_vertex_count;
          const bool finish_trace_bounce = remaining_connect_light_vertex_length == 0u;

          record_and_submit([&](RHICommandBuffer cmd) {
            if (connect_light_batch_in_progress == false) {
              barrier_wavefront_buffers(cmd);
              if (_wavefront_light_queue_count > 0u) {
                if (enable_connect_to_camera) {
                  if (has_various_connect) {
                    dispatch_stage_material_indirect(cmd, PipelineStage::LightConnectCameraPrepareDiffuse, false, kGPUWavefrontMaterialQueueVarious, path_iteration);
                  }
                  if (has_thinfilm && use_material_work_queues) {
                    dispatch_stage_material_indirect(cmd, PipelineStage::LightConnectCameraPrepareDiffuse, false, kGPUWavefrontMaterialQueueThinfilm, path_iteration);
                  }
                  if (has_plastic) {
                    dispatch_stage_material_indirect(cmd, PipelineStage::LightConnectCameraPreparePlastic, false, kGPUWavefrontMaterialQueuePlastic, path_iteration);
                  }
                  if (has_connectible_conductor) {
                    dispatch_stage_material_indirect(cmd, PipelineStage::LightConnectCameraPrepareConductor, false, kGPUWavefrontMaterialQueueConductor, path_iteration);
                  }
                  if (has_connectible_dielectric) {
                    dispatch_stage_material_indirect(cmd, PipelineStage::LightConnectCameraPrepareDielectric, false, kGPUWavefrontMaterialQueueDielectric, path_iteration);
                  }
                  rebuild_dispatch_args(cmd, path_iteration);
                  dispatch_stage_indirect(cmd, PipelineStage::LightConnectCameraShadow, shadow_dispatch_args_offset(kGPUWavefrontShadowQueueConnectCamera), path_iteration);
                  barrier_wavefront_buffers(cmd);
                  dispatch_stage_indirect(cmd, PipelineStage::LightConnectCameraAccumulate, shadow_dispatch_args_offset(kGPUWavefrontShadowQueueConnectCamera), path_iteration);
                  barrier_wavefront_buffers(cmd);
                }
                if (continue_paths) {
                  dispatch_stage_indirect(cmd, PipelineStage::LightContinueFinalize, kGPUWavefrontLightDispatchArgsOffset, path_iteration);
                  barrier_wavefront_buffers(cmd);
                }
              }
              if (_wavefront_camera_queue_count > 0u) {
                if (enable_connect_to_light) {
                  dispatch_stage_indirect(cmd, PipelineStage::CameraDirectLightSample, kGPUWavefrontCameraDispatchArgsOffset, path_iteration);
                  barrier_wavefront_buffers(cmd);
                  if (has_various_connect) {
                    dispatch_stage_material_indirect(cmd, PipelineStage::CameraDirectLightPrepareDiffuse, true, kGPUWavefrontMaterialQueueVarious, path_iteration);
                  }
                  if (has_thinfilm && use_material_work_queues) {
                    dispatch_stage_material_indirect(cmd, PipelineStage::CameraDirectLightPrepareDiffuse, true, kGPUWavefrontMaterialQueueThinfilm, path_iteration);
                  }
                  if (has_plastic) {
                    dispatch_stage_material_indirect(cmd, PipelineStage::CameraDirectLightPreparePlastic, true, kGPUWavefrontMaterialQueuePlastic, path_iteration);
                  }
                  if (has_connectible_conductor) {
                    dispatch_stage_material_indirect(cmd, PipelineStage::CameraDirectLightPrepareConductor, true, kGPUWavefrontMaterialQueueConductor, path_iteration);
                  }
                  if (has_connectible_dielectric) {
                    dispatch_stage_material_indirect(cmd, PipelineStage::CameraDirectLightPrepareDielectric, true, kGPUWavefrontMaterialQueueDielectric, path_iteration);
                  }
                  rebuild_dispatch_args(cmd, path_iteration);
                  dispatch_stage_indirect(cmd, PipelineStage::CameraDirectLightShadow, shadow_dispatch_args_offset(kGPUWavefrontShadowQueueDirectLight), path_iteration);
                  barrier_wavefront_buffers(cmd);
                  dispatch_stage_indirect(cmd, PipelineStage::CameraDirectLightAccumulate, shadow_dispatch_args_offset(kGPUWavefrontShadowQueueDirectLight), path_iteration);
                  barrier_wavefront_buffers(cmd);
                }
                if (enable_direct_hit) {
                  dispatch_stage_indirect(cmd, PipelineStage::CameraDirectHitAccumulate, kGPUWavefrontCameraDispatchArgsOffset, path_iteration);
                  barrier_wavefront_buffers(cmd);
                }
                if (enable_merge_vertices) {
                  if (has_various_connect) {
                    dispatch_vcm_merge_material(cmd, PipelineStage::VCMMergeDiffuse, kGPUWavefrontMaterialQueueVarious, path_iteration);
                  }
                  if (has_thinfilm && use_material_work_queues) {
                    dispatch_vcm_merge_material(cmd, PipelineStage::VCMMergeDiffuse, kGPUWavefrontMaterialQueueThinfilm, path_iteration);
                  }
                  if (has_plastic) {
                    dispatch_vcm_merge_material(cmd, PipelineStage::VCMMergePlastic, kGPUWavefrontMaterialQueuePlastic, path_iteration);
                  }
                  if (has_connectible_conductor) {
                    dispatch_vcm_merge_material(cmd, PipelineStage::VCMMergeConductor, kGPUWavefrontMaterialQueueConductor, path_iteration);
                  }
                  if (has_connectible_dielectric) {
                    dispatch_vcm_merge_material(cmd, PipelineStage::VCMMergeDielectric, kGPUWavefrontMaterialQueueDielectric, path_iteration);
                  }
                }
              }
            }
            if (connect_light_vertex_count > 0u) {
              const bool reset_light_cursor = _wavefront_connect_light_vertex_length == _wavefront_connect_light_history_bounces;
              const uint32_t connect_light_batch_index =
                (_wavefront_connect_light_history_bounces - _wavefront_connect_light_vertex_length) / kGPUWavefrontConnectDispatchArgsCount;
              const uint32_t connect_light_cursor_slot = connect_light_batch_index & 1u;
              const uint64_t connect_light_argument_buffer_offset =
                kGPUWavefrontConnectDispatchArgsOffset + static_cast<uint64_t>(connect_light_vertex_count - 1u) * kGPUWavefrontDispatchArgsStride;
              const uint32_t connect_light_prepare_stage_count = static_cast<uint32_t>(has_various_connect) + static_cast<uint32_t>(has_plastic) +
                                                                 static_cast<uint32_t>(has_connectible_conductor) + static_cast<uint32_t>(has_connectible_dielectric);
              const bool compact_connections = connect_light_prepare_stage_count > 1u;
              if (compact_connections) {
                dispatch_stage_mode(cmd, PipelineStage::CameraConnectLightCompact, scalar_dispatch, path_iteration, 0u);
                barrier_wavefront_buffers(cmd);
                dispatch_stage_with_connect_light_length(cmd, PipelineStage::CameraConnectLightClassify, connect_light_argument_buffer_offset, path_iteration,
                  _wavefront_connect_light_vertex_length, connect_light_vertex_count, reset_light_cursor, true, connect_light_cursor_slot);
                barrier_wavefront_buffers(cmd);
                ctx.cmd_buffer_barrier(cmd, _wavefront_dispatch_args_buffer, _wavefront_dispatch_args_buffer_state, RHIResourceState::General);
                _wavefront_dispatch_args_buffer_state = RHIResourceState::General;
                dispatch_stage_mode(cmd, PipelineStage::CameraConnectLightCompact, scalar_dispatch, path_iteration, 1u);
                ctx.cmd_buffer_barrier(cmd, _wavefront_dispatch_args_buffer, RHIResourceState::General, RHIResourceState::IndirectArgument);
                _wavefront_dispatch_args_buffer_state = RHIResourceState::IndirectArgument;
                barrier_wavefront_buffers(cmd);
                dispatch_stage_with_connect_light_length(cmd, PipelineStage::CameraConnectLightClassify, connect_light_argument_buffer_offset, path_iteration,
                  _wavefront_connect_light_vertex_length, connect_light_vertex_count, false, false, 0u);
                barrier_wavefront_buffers(cmd);
              }
              bool initialize_connect_light_batch = true;
              const auto connection_arguments = [&](uint32_t queue) {
                return compact_connections ? (kGPUWavefrontConnectQueueDispatchArgsOffset + static_cast<uint64_t>(queue) * kGPUWavefrontDispatchArgsStride)
                                           : connect_light_argument_buffer_offset;
              };
              const auto dispatch_connect_light_prepare = [&](PipelineStage stage, uint32_t queue) {
                const bool initialize_batch = initialize_connect_light_batch && (compact_connections == false);
                dispatch_stage_with_connect_light_length(cmd, stage, connection_arguments(queue), path_iteration, _wavefront_connect_light_vertex_length,
                  connect_light_vertex_count, reset_light_cursor && initialize_batch, initialize_batch, connect_light_cursor_slot);
                initialize_connect_light_batch = false;
              };
              if (has_various_connect) {
                dispatch_connect_light_prepare(PipelineStage::CameraConnectLightPrepareDiffuse, 0u);
              }
              if (has_plastic) {
                dispatch_connect_light_prepare(PipelineStage::CameraConnectLightPreparePlastic, 1u);
              }
              if (has_connectible_conductor) {
                dispatch_connect_light_prepare(PipelineStage::CameraConnectLightPrepareConductor, 2u);
              }
              if (has_connectible_dielectric) {
                dispatch_connect_light_prepare(PipelineStage::CameraConnectLightPrepareDielectric, 3u);
              }
              barrier_wavefront_buffers(cmd);
              if (has_various_connect) {
                dispatch_stage_with_connect_light_length(cmd, PipelineStage::CameraConnectLightResolveDiffuse, connection_arguments(4u), path_iteration,
                  _wavefront_connect_light_vertex_length, connect_light_vertex_count, false, false, 0u);
              }
              if (has_plastic) {
                dispatch_stage_with_connect_light_length(cmd, PipelineStage::CameraConnectLightResolvePlastic, connection_arguments(5u), path_iteration,
                  _wavefront_connect_light_vertex_length, connect_light_vertex_count, false, false, 0u);
              }
              if (has_connectible_conductor) {
                dispatch_stage_with_connect_light_length(cmd, PipelineStage::CameraConnectLightResolveConductor, connection_arguments(6u), path_iteration,
                  _wavefront_connect_light_vertex_length, connect_light_vertex_count, false, false, 0u);
              }
              if (has_connectible_dielectric) {
                dispatch_stage_with_connect_light_length(cmd, PipelineStage::CameraConnectLightResolveDielectric, connection_arguments(7u), path_iteration,
                  _wavefront_connect_light_vertex_length, connect_light_vertex_count, false, false, 0u);
              }
              rebuild_dispatch_args(cmd, path_iteration);
              barrier_wavefront_buffers(cmd);
              dispatch_stage_indirect(cmd, PipelineStage::CameraConnectLightShadow, shadow_dispatch_args_offset(kGPUWavefrontShadowQueueConnectLight), path_iteration);
              barrier_wavefront_buffers(cmd);
            }
            if (finish_trace_bounce && (_wavefront_camera_queue_count > 0u) && continue_paths) {
              dispatch_stage_indirect(cmd, PipelineStage::CameraContinueFinalize, kGPUWavefrontCameraDispatchArgsOffset, path_iteration);
              barrier_wavefront_buffers(cmd);
            }
            if (finish_trace_bounce) {
              dispatch_stage(cmd, PipelineStage::SwapQueues, scalar_dispatch, path_iteration);
              barrier_wavefront_buffers(cmd);

              if (finish_trace_bounce && copy_queue_counts) {
                if ((enable_camera_path) && _wavefront_camera_phase_initialized && (_wavefront_camera_queue_count > 0u)) {
                  ctx.cmd_buffer_barrier(cmd, next_camera_queue_buffer, RHIResourceState::General, RHIResourceState::TransferSrc);
                  ctx.cmd_buffer_barrier(cmd, _camera_queue_count_readback_buffer, _camera_queue_count_readback_state, RHIResourceState::TransferDst);
                  ctx.cmd_copy_buffer(cmd, next_camera_queue_buffer, _camera_queue_count_readback_buffer, kGPUWavefrontQueueHeaderSize);
                  ctx.cmd_buffer_barrier(cmd, next_camera_queue_buffer, RHIResourceState::TransferSrc, RHIResourceState::General);
                  _camera_queue_count_readback_state = RHIResourceState::TransferDst;
                }
                if ((enable_light_path) && (_wavefront_light_queue_count > 0u)) {
                  ctx.cmd_buffer_barrier(cmd, next_light_queue_buffer, RHIResourceState::General, RHIResourceState::TransferSrc);
                  ctx.cmd_buffer_barrier(cmd, _light_queue_count_readback_buffer, _light_queue_count_readback_state, RHIResourceState::TransferDst);
                  ctx.cmd_copy_buffer(cmd, next_light_queue_buffer, _light_queue_count_readback_buffer, kGPUWavefrontQueueHeaderSize);
                  ctx.cmd_buffer_barrier(cmd, next_light_queue_buffer, RHIResourceState::TransferSrc, RHIResourceState::General);
                  _light_queue_count_readback_state = RHIResourceState::TransferDst;
                }
              }
              if (finish_trace_bounce && copy_light_vertex_count) {
                ctx.cmd_buffer_barrier(cmd, _light_vertex_counter_buffer, RHIResourceState::General, RHIResourceState::TransferSrc);
                ctx.cmd_buffer_barrier(cmd, _light_vertex_counter_readback_buffer, _light_vertex_counter_readback_state, RHIResourceState::TransferDst);
                ctx.cmd_copy_buffer(cmd, _light_vertex_counter_buffer, _light_vertex_counter_readback_buffer, kWavefrontLightVertexCounterSize);
                ctx.cmd_buffer_barrier(cmd, _light_vertex_counter_buffer, RHIResourceState::TransferSrc, RHIResourceState::General);
                _light_vertex_counter_readback_state = RHIResourceState::TransferDst;
              }
              if (finish_trace_bounce && copy_upbp_terminal_light_path_length) {
                ctx.cmd_buffer_barrier(cmd, _upbp.counter_buffer.handle, RHIResourceState::General, RHIResourceState::TransferSrc);
                ctx.cmd_buffer_barrier(cmd, _upbp.counter_readback_buffer.handle, _upbp.counter_readback_state, RHIResourceState::TransferDst);
                ctx.cmd_copy_buffer(cmd, _upbp.counter_buffer.handle, _upbp.counter_readback_buffer.handle, sizeof(uint32_t),
                  GPUUPBPCounterIndex::MaximumLightPathLength * sizeof(uint32_t));
                ctx.cmd_buffer_barrier(cmd, _upbp.counter_buffer.handle, RHIResourceState::TransferSrc, RHIResourceState::General);
                _upbp.counter_readback_state = RHIResourceState::TransferDst;
              }
            }
          });
          if ((connect_light_batch_in_progress == false) && store_complete_light_history && (_wavefront_light_queue_count > 0u)) {
            _wavefront_light_vertex_reserved_count += _wavefront_light_queue_count;
          }
          _wavefront_connect_light_vertex_length = remaining_connect_light_vertex_length;
          if (finish_trace_bounce) {
            _wavefront_connect_light_history_bounces = 0u;
          }
          if (queue_readback_due) {
            trace_step_result = wait_and_destroy_submitted_commands("trace bounce submit");
          }

          if (trace_step_result != RHIResult::Success) {
            set_runtime_failure("GPU RT trace bounce submit failed (" + std::to_string(static_cast<uint32_t>(trace_step_result)) + ")");
            _wavefront_camera_queue_count = 0u;
            _wavefront_light_queue_count = 0u;
            return;
          }

          if (finish_trace_bounce == false) {
            dispatch_submit_ms = elapsed_ms(dispatch_submit_begin, std::chrono::steady_clock::now());
            continue;
          }

          _wavefront_path_iteration += 1u;

          if (copy_queue_counts) {
            if ((enable_camera_path) && _wavefront_camera_phase_initialized && (_wavefront_camera_queue_count > 0u)) {
              GPUWavefrontQueueHeader queue_header = {};
              const RHIResult read_result = device.read_buffer(_camera_queue_count_readback_buffer, &queue_header, static_cast<uint64_t>(sizeof(queue_header)));
              if (read_result != RHIResult::Success) {
                set_runtime_failure("GPU RT failed to read the camera queue count (" + std::to_string(static_cast<uint32_t>(read_result)) + ")");
                _wavefront_camera_queue_count = 0u;
                _wavefront_light_queue_count = 0u;
                return;
              }
              _wavefront_camera_queue_count = queue_header.count;
            } else {
              _wavefront_camera_queue_count = 0u;
            }

            if ((enable_light_path) && (_wavefront_light_queue_count > 0u)) {
              GPUWavefrontQueueHeader queue_header = {};
              const RHIResult read_result = device.read_buffer(_light_queue_count_readback_buffer, &queue_header, static_cast<uint64_t>(sizeof(queue_header)));
              if (read_result != RHIResult::Success) {
                set_runtime_failure("GPU RT failed to read the light queue count (" + std::to_string(static_cast<uint32_t>(read_result)) + ")");
                _wavefront_camera_queue_count = 0u;
                _wavefront_light_queue_count = 0u;
                return;
              }
              _wavefront_light_queue_count = queue_header.count;
              _wavefront_light_max_path_length = std::max(_wavefront_light_max_path_length, queue_header.max_path_length);
            } else {
              _wavefront_light_queue_count = 0u;
            }
          } else if (continue_paths) {
            if (store_complete_light_history && enable_light_path) {
              _wavefront_light_max_path_length = std::max(_wavefront_light_max_path_length, std::min(light_history_bounces, path_iteration + 1u));
            }
          } else {
            _wavefront_camera_queue_count = 0u;
            _wavefront_light_queue_count = 0u;
          }

          if (copy_light_vertex_count) {
            uint32_t light_vertex_count = 0u;
            const RHIResult counter_read_result = device.read_buffer(_light_vertex_counter_readback_buffer, &light_vertex_count, kWavefrontLightVertexCounterSize);
            if (counter_read_result != RHIResult::Success) {
              set_runtime_failure("GPU RT failed to read compact light history size (" + std::to_string(static_cast<uint32_t>(counter_read_result)) + ")");
              _wavefront_camera_queue_count = 0u;
              _wavefront_light_queue_count = 0u;
              return;
            }
            _wavefront_light_vertex_reserved_count = light_vertex_count;
          }

          if (copy_upbp_terminal_light_path_length) {
            uint32_t maximum_light_path_length = 0u;
            const RHIResult maximum_length_result = device.read_buffer(_upbp.counter_readback_buffer.handle, &maximum_light_path_length, sizeof(maximum_light_path_length));
            if (maximum_length_result != RHIResult::Success) {
              set_runtime_failure("GPU UPBP failed to read the maximum light-path length (" + std::to_string(static_cast<uint32_t>(maximum_length_result)) + ")");
              _wavefront_camera_queue_count = 0u;
              _wavefront_light_queue_count = 0u;
              return;
            }
            _wavefront_light_max_path_length = maximum_light_path_length;
          }

          const bool waiting_for_deferred_camera_phase_after_step = phase_light_before_camera && enable_camera_path && (_wavefront_camera_phase_initialized == false);
          if (waiting_for_deferred_camera_phase_after_step) {
            if (_wavefront_path_iteration >= _wavefront_hard_iteration_cap) {
              _wavefront_light_queue_count = 0u;
            }
          } else if ((_wavefront_path_iteration >= _wavefront_hard_iteration_cap) || ((_wavefront_camera_queue_count == 0u) && (_wavefront_light_queue_count == 0u))) {
            if (upbp_mode) {
              complete_upbp_trace_batch();
            } else {
              if (finalize_wavefront_sample() == false) {
                return;
              }
            }
          }
        }

        dispatch_submit_ms = elapsed_ms(dispatch_submit_begin, std::chrono::steady_clock::now());
      } else if (_wavefront_render_step == WavefrontRenderStep::UPBPEvaluateLightBatch) {
        record_and_submit([&](RHICommandBuffer cmd) {
          barrier_wavefront_buffers(cmd);
          dispatch_stage(cmd, PipelineStage::UPBPValidate, scalar_dispatch, 0u);
          barrier_wavefront_buffers(cmd);
          ctx.cmd_buffer_barrier(cmd, _upbp.counter_buffer.handle, RHIResourceState::General, RHIResourceState::TransferSrc);
          ctx.cmd_buffer_barrier(cmd, _upbp.counter_readback_buffer.handle, _upbp.counter_readback_state, RHIResourceState::TransferDst);
          ctx.cmd_copy_buffer(cmd, _upbp.counter_buffer.handle, _upbp.counter_readback_buffer.handle, static_cast<uint64_t>(GPUUPBPCounterIndex::Count) * sizeof(uint32_t));
          ctx.cmd_buffer_barrier(cmd, _upbp.counter_buffer.handle, RHIResourceState::TransferSrc, RHIResourceState::General);
          _upbp.counter_readback_state = RHIResourceState::TransferDst;
        });
        const RHIResult validation_result = wait_and_destroy_submitted_commands("UPBP batch validation");
        if (validation_result != RHIResult::Success) {
          set_runtime_failure("GPU UPBP batch validation failed (" + std::to_string(static_cast<uint32_t>(validation_result)) + ")");
          return;
        }
        uint32_t upbp_counters[GPUUPBPCounterIndex::Count] = {};
        const RHIResult counter_result = device.read_buffer(_upbp.counter_readback_buffer.handle, upbp_counters, sizeof(upbp_counters));
        if (counter_result != RHIResult::Success) {
          set_runtime_failure("GPU UPBP failed to read batch counters (" + std::to_string(static_cast<uint32_t>(counter_result)) + ")");
          return;
        }
        const uint32_t overflow_flags = upbp_counters[GPUUPBPCounterIndex::OverflowFlags];
        const uint32_t failed_camera_paths = upbp_counters[GPUUPBPCounterIndex::FailedCameraPaths];
        const uint32_t failed_light_paths = upbp_counters[GPUUPBPCounterIndex::FailedLightPaths];
        const uint32_t failed_connections = upbp_counters[GPUUPBPCounterIndex::FailedConnections];
        if (overflow_flags != 0u) {
          set_runtime_failure(
            "GPU UPBP resident storage overflowed while tracing a deterministic path batch (flags " + std::to_string(overflow_flags) + "; vertices " +
            std::to_string(upbp_counters[GPUUPBPCounterIndex::LightVertex]) + "/" + std::to_string(upbp_counters[GPUUPBPCounterIndex::CameraVertex]) + ", segments " +
            std::to_string(upbp_counters[GPUUPBPCounterIndex::LightSegment]) + "/" + std::to_string(upbp_counters[GPUUPBPCounterIndex::CameraSegment]) + ", intervals " +
            std::to_string(upbp_counters[GPUUPBPCounterIndex::LightInterval]) + "/" + std::to_string(upbp_counters[GPUUPBPCounterIndex::CameraInterval]) + ", events " +
            std::to_string(upbp_counters[GPUUPBPCounterIndex::LightEvent]) + "/" + std::to_string(upbp_counters[GPUUPBPCounterIndex::CameraEvent]) + ")");
          return;
        }
        if ((failed_camera_paths != 0u) || (failed_light_paths != 0u) || (failed_connections != 0u)) {
          const uint32_t first_failure = upbp_counters[GPUUPBPCounterIndex::FirstFailureCode];
          std::string failure_details =
            ", details " + std::to_string(upbp_counters[GPUUPBPCounterIndex::FirstFailureDetail0]) + "/" + std::to_string(upbp_counters[GPUUPBPCounterIndex::FirstFailureDetail1]) +
            "/" + std::to_string(upbp_counters[GPUUPBPCounterIndex::FirstFailureDetail2]) + "/" + std::to_string(upbp_counters[GPUUPBPCounterIndex::FirstFailureDetail3]);
          if (first_failure == GPUUPBPPathFailure::InvalidSegmentDistance) {
            failure_details = ", path length " + std::to_string(upbp_counters[GPUUPBPCounterIndex::FirstFailureDetail0]) + ", origin radius " +
                              std::to_string(std::bit_cast<float>(upbp_counters[GPUUPBPCounterIndex::FirstFailureDetail1])) + ", transport radius " +
                              std::to_string(std::bit_cast<float>(upbp_counters[GPUUPBPCounterIndex::FirstFailureDetail2])) + ", direction projection " +
                              std::to_string(std::bit_cast<float>(upbp_counters[GPUUPBPCounterIndex::FirstFailureDetail3]));
          }
          set_runtime_failure("GPU UPBP path construction failed for " + std::to_string(failed_camera_paths) + " camera paths and " + std::to_string(failed_light_paths) +
                              " light paths, with " + std::to_string(failed_connections) + " failed connections; first failure " + std::to_string(first_failure) +
                              " at global path " + std::to_string(upbp_counters[GPUUPBPCounterIndex::FirstFailureGlobalPath]) + failure_details);
          return;
        }

        const bool evaluating_upbp_camera_phase = _upbp.camera_phase_started;
        if ((evaluating_upbp_camera_phase == false) && (_upbp.light_batch_iteration == 0u) && (snapshot_upbp_matching_light_history(upbp_counters) == false)) {
          return;
        }

        const uint32_t point_count = std::min(upbp_counters[GPUUPBPCounterIndex::Point], _upbp.resources.point_capacity);
        const uint32_t light_vertex_count = std::min(upbp_counters[GPUUPBPCounterIndex::LightVertex], _upbp.resources.light_vertex_capacity);
        const uint32_t surface_point_count = std::min(upbp_counters[GPUUPBPCounterIndex::SurfacePoint], point_count);
        const uint32_t medium_point_count = std::min(upbp_counters[GPUUPBPCounterIndex::MediumPoint], point_count);
        const uint32_t beam_count = std::min(upbp_counters[GPUUPBPCounterIndex::Beam], _upbp.resources.beam_capacity);
        const uint32_t light_event_count = std::min(upbp_counters[GPUUPBPCounterIndex::LightEvent], _upbp.resources.light_event_capacity);
        const uint32_t camera_vertex_count = std::min(upbp_counters[GPUUPBPCounterIndex::CameraVertex], _upbp.resources.camera_vertex_capacity);
        const uint32_t camera_interval_count = std::min(upbp_counters[GPUUPBPCounterIndex::CameraInterval], _upbp.resources.camera_interval_capacity);
        uint32_t camera_surface_query_counts[GPUUPBPSurfaceQueryFamily::Count] = {};
        for (uint32_t& query_count : camera_surface_query_counts) {
          query_count = camera_vertex_count;
        }
        uint32_t camera_medium_vertex_query_count = camera_vertex_count;
        uint32_t camera_medium_interval_query_count = camera_interval_count;
        bool camera_surface_queries_compacted = false;
        bool camera_medium_vertex_queries_compacted = false;
        bool camera_interval_queries_compacted = false;
        if (capture_kernel_timings && (evaluating_upbp_camera_phase == false)) {
          log::info("GPU UPBP light batch %u/%u: paths=%u vertices=%u segments=%u intervals=%u maximum-path-length=%u path-state=%.2f MiB vertex-history=%.2f MiB",
            _upbp.light_batch_index + 1u, divide_round_up(_upbp.global_path_count, _upbp.resident_light_path_capacity), _upbp.light_batch_count, light_vertex_count,
            std::min(upbp_counters[GPUUPBPCounterIndex::LightSegment], _upbp.resources.light_segment_capacity),
            std::min(upbp_counters[GPUUPBPCounterIndex::LightInterval], _upbp.resources.light_interval_capacity), upbp_counters[GPUUPBPCounterIndex::MaximumLightPathLength],
            static_cast<double>(static_cast<uint64_t>(_upbp.light_batch_count) * kGPUUPBPBPTPathStateStride) / (1024.0 * 1024.0),
            static_cast<double>(static_cast<uint64_t>(light_vertex_count) * kGPUUPBPBPTVertexStride) / (1024.0 * 1024.0));
        }
        const bool bp2d_enabled = gpu_integrator_feature_enabled(_integrator_features, GPUIntegratorFeatures::UPBPBP2D);
        const bool bb1d_enabled = gpu_integrator_feature_enabled(_integrator_features, GPUIntegratorFeatures::UPBPBB1D);
        if (_upbp.density_cache_ready == false) {
          const RHIBufferUsage storage_usage = RHIBufferUsage::Storage | RHIBufferUsage::TransferDst | RHIBufferUsage::TransferSrc;
          const RHIBufferUsage aabb_usage = storage_usage | RHIBufferUsage::AccelerationStructureBuild | RHIBufferUsage::ShaderDeviceAddress;
          if (_upbp.density_batch_count >= _upbp.density_batches.size()) {
            _upbp.density_batches.emplace_back();
          }
          UPBPDensityBatchResources& density_batch = _upbp.density_batches[_upbp.density_batch_count];
          density_batch.surface_point_count = 0u;
          density_batch.medium_point_count = 0u;
          density_batch.beam_count = 0u;
          density_batch.event_count = 0u;
          density_batch.selected_beam_count = 0u;
          const auto ensure_density_buffer = [&device](UPBPBuffer& buffer, uint64_t size, RHIBufferUsage usage, const char* name) {
            return ensure_storage_buffer_capacity(device, std::max<uint64_t>(size, sizeof(uint32_t)), usage, buffer.handle, buffer.size, buffer.descriptor_index, name);
          };
          if (((surface_point_count > 0u) && ((ensure_density_buffer(density_batch.surface_point_buffer, static_cast<uint64_t>(surface_point_count) * kGPUUPBPDensityPointStride,
                                                 storage_usage, "upbp_density_surface_points") == false) ||
                                               (ensure_density_buffer(density_batch.surface_point_aabb_buffer, static_cast<uint64_t>(surface_point_count) * kGPUUPBPAABBStride,
                                                  aabb_usage, "upbp_density_surface_point_aabbs") == false))) ||
              ((medium_point_count > 0u) && ((ensure_density_buffer(density_batch.medium_point_buffer, static_cast<uint64_t>(medium_point_count) * kGPUUPBPDensityPointStride,
                                                storage_usage, "upbp_density_medium_points") == false) ||
                                              (ensure_density_buffer(density_batch.medium_point_aabb_buffer, static_cast<uint64_t>(medium_point_count) * kGPUUPBPAABBStride,
                                                 aabb_usage, "upbp_density_medium_point_aabbs") == false))) ||
              ((beam_count > 0u) &&
                (ensure_density_buffer(density_batch.beam_buffer, static_cast<uint64_t>(beam_count) * kGPUUPBPDensityBeamStride, storage_usage, "upbp_density_beams") == false)) ||
              ((light_event_count > 0u) && (ensure_density_buffer(density_batch.event_buffer, static_cast<uint64_t>(light_event_count) * kGPUUPBPTrackingEventStride, storage_usage,
                                              "upbp_density_events") == false))) {
            set_runtime_failure("GPU UPBP failed to allocate an exact compact density batch");
            return;
          }
          _upbp.resources.density_output_surface_point_buffer = density_batch.surface_point_buffer.descriptor_index;
          _upbp.resources.density_output_surface_point_capacity = surface_point_count;
          _upbp.resources.point_aabb_buffer = density_batch.surface_point_aabb_buffer.descriptor_index;
          _upbp.resources.point_aabb_capacity = surface_point_count;
          _upbp.resources.density_output_medium_point_buffer = density_batch.medium_point_buffer.descriptor_index;
          _upbp.resources.density_output_medium_point_capacity = medium_point_count;
          _upbp.resources.density_output_medium_point_aabb_buffer = density_batch.medium_point_aabb_buffer.descriptor_index;
          _upbp.resources.density_output_medium_point_aabb_capacity = medium_point_count;
          _upbp.resources.density_output_beam_buffer = density_batch.beam_buffer.descriptor_index;
          _upbp.resources.density_output_beam_capacity = beam_count;
          _upbp.resources.density_output_event_buffer = density_batch.event_buffer.descriptor_index;
          if (update_upbp_iteration_resources(device, scene, upbp_global_path_count) == false) {
            set_runtime_failure("GPU UPBP failed to bind compact density outputs");
            return;
          }
          record_and_submit([&](RHICommandBuffer cmd) {
            barrier_wavefront_buffers(cmd);
            if (light_event_count > 0u) {
              ctx.cmd_buffer_barrier(cmd, _upbp.event_buffer.handle, RHIResourceState::General, RHIResourceState::TransferSrc);
              ctx.cmd_buffer_barrier(cmd, density_batch.event_buffer.handle, RHIResourceState::Undefined, RHIResourceState::TransferDst);
              ctx.cmd_copy_buffer(cmd, _upbp.event_buffer.handle, density_batch.event_buffer.handle, static_cast<uint64_t>(light_event_count) * kGPUUPBPTrackingEventStride);
              ctx.cmd_buffer_barrier(cmd, density_batch.event_buffer.handle, RHIResourceState::TransferDst, RHIResourceState::General);
              ctx.cmd_buffer_barrier(cmd, _upbp.event_buffer.handle, RHIResourceState::TransferSrc, RHIResourceState::General);
            }
            dispatch_stage_mode(cmd, PipelineStage::UPBPClear, scalar_dispatch, 0u, GPUUPBPClearMode::DensityCompact);
            barrier_wavefront_buffers(cmd);
            if (point_count > 0u) {
              if (surface_point_count > 0u) {
                ctx.cmd_buffer_barrier(cmd, density_batch.surface_point_aabb_buffer.handle, RHIResourceState::Undefined, RHIResourceState::General);
              }
              if (medium_point_count > 0u) {
                ctx.cmd_buffer_barrier(cmd, density_batch.medium_point_aabb_buffer.handle, RHIResourceState::Undefined, RHIResourceState::General);
              }
              dispatch_stage_range(cmd, PipelineStage::UPBPDensityCompact, 0u, point_count, 0u, GPUUPBPDensityCompactMode::Points);
            }
            if (beam_count > 0u) {
              dispatch_stage_range(cmd, PipelineStage::UPBPDensityCompact, 0u, beam_count, 0u, GPUUPBPDensityCompactMode::Beams);
            }
            barrier_wavefront_buffers(cmd);
            ctx.cmd_buffer_barrier(cmd, _upbp.counter_buffer.handle, RHIResourceState::General, RHIResourceState::TransferSrc);
            ctx.cmd_buffer_barrier(cmd, _upbp.counter_readback_buffer.handle, _upbp.counter_readback_state, RHIResourceState::TransferDst);
            ctx.cmd_copy_buffer(cmd, _upbp.counter_buffer.handle, _upbp.counter_readback_buffer.handle, static_cast<uint64_t>(GPUUPBPCounterIndex::Count) * sizeof(uint32_t));
            ctx.cmd_buffer_barrier(cmd, _upbp.counter_buffer.handle, RHIResourceState::TransferSrc, RHIResourceState::General);
            _upbp.counter_readback_state = RHIResourceState::TransferDst;
          });
          const RHIResult compact_result = wait_and_destroy_submitted_commands("UPBP compact density batch");
          if (compact_result != RHIResult::Success) {
            set_runtime_failure("GPU UPBP compact density batch failed (" + std::to_string(static_cast<uint32_t>(compact_result)) + ")");
            return;
          }
          const RHIResult compact_counter_result = device.read_buffer(_upbp.counter_readback_buffer.handle, upbp_counters, sizeof(upbp_counters));
          if (compact_counter_result != RHIResult::Success) {
            set_runtime_failure("GPU UPBP failed to read compact density counts (" + std::to_string(static_cast<uint32_t>(compact_counter_result)) + ")");
            return;
          }
          density_batch.surface_point_count = upbp_counters[GPUUPBPCounterIndex::DensitySurfacePoint];
          density_batch.medium_point_count = upbp_counters[GPUUPBPCounterIndex::DensityMediumPoint];
          density_batch.beam_count = upbp_counters[GPUUPBPCounterIndex::DensityBeam];
          density_batch.selected_beam_count = upbp_counters[GPUUPBPCounterIndex::DensitySelectedBeam];
          density_batch.event_count = light_event_count;
          if ((density_batch.surface_point_count > surface_point_count) || (density_batch.medium_point_count > medium_point_count) || (density_batch.beam_count > beam_count) ||
              (density_batch.selected_beam_count > beam_count) ||
              ((upbp_counters[GPUUPBPCounterIndex::OverflowFlags] & (GPUUPBPOverflowFlags::Point | GPUUPBPOverflowFlags::Beam)) != 0u)) {
            set_runtime_failure("GPU UPBP compact density storage exceeded its exact source capacity");
            return;
          }
          _upbp.density_batch_count += 1u;
        }
        const bool final_density_batch = (_upbp.density_cache_ready == false) && ((_upbp.light_batch_iteration + 1u) >= _upbp.light_batch_count_total);
        if (final_density_batch) {
          const uint64_t released_resident_bytes = destroy_upbp_resident_path_buffers(device);
          if (released_resident_bytes > 0u) {
            const RHIResult reclaim_result = ctx.wait_idle();
            if (reclaim_result != RHIResult::Success) {
              set_runtime_failure("GPU UPBP failed to reclaim resident light-path storage (" + std::to_string(static_cast<uint32_t>(reclaim_result)) + ")");
              return;
            }
            log::info("GPU UPBP released %.2f MiB of resident light-path storage before density-cache consolidation",
              static_cast<double>(released_resident_bytes) / (1024.0 * 1024.0));
          }
          std::vector<GPUUPBPDensityBatch> batch_descriptors(_upbp.density_batch_count);
          std::vector<RHIAccelerationStructureInstance> surface_point_instances = {};
          std::vector<RHIAccelerationStructureInstance> medium_point_instances = {};
          surface_point_instances.reserve(kGPUUPBPSurfacePartitionCount);
          medium_point_instances.reserve(1u);
          uint64_t surface_point_count_total = 0u;
          uint64_t medium_point_count_total = 0u;
          uint64_t bp2d_beam_instance_count = 0u;
          uint64_t bb1d_beam_instance_count = 0u;
          uint64_t tracking_event_count_total = 0u;
          const auto append_density_instance = [&device](RHIBindlessHandle blas, uint32_t custom_index, uint8_t mask, std::vector<RHIAccelerationStructureInstance>& instances) {
            if (blas.valid() == false) {
              return true;
            }
            const uint64_t address = device.get_acceleration_structure_device_address(blas);
            if (address == 0u) {
              return false;
            }
            RHIAccelerationStructureInstance& instance = instances.emplace_back();
            instance.transform[0] = 1.0f;
            instance.transform[5] = 1.0f;
            instance.transform[10] = 1.0f;
            instance.instance_custom_index = custom_index;
            instance.mask = mask;
            instance.acceleration_structure_reference = address;
            return true;
          };
          for (uint32_t batch_index = 0u; batch_index < _upbp.density_batch_count; ++batch_index) {
            const UPBPDensityBatchResources& batch = _upbp.density_batches[batch_index];
            GPUUPBPDensityBatch& descriptor = batch_descriptors[batch_index];
            descriptor.surface_point_buffer = kInvalidDescriptorIndex;
            descriptor.surface_point_count = batch.surface_point_count;
            descriptor.medium_point_buffer = kInvalidDescriptorIndex;
            descriptor.medium_point_count = batch.medium_point_count;
            descriptor.beam_buffer = batch.beam_buffer.descriptor_index;
            descriptor.beam_count = batch.beam_count;
            descriptor.beam_instance_offset = static_cast<uint32_t>(bp2d_beam_instance_count);
            descriptor.selected_beam_count = batch.selected_beam_count;
            surface_point_count_total += batch.surface_point_count;
            medium_point_count_total += batch.medium_point_count;
            bp2d_beam_instance_count += bp2d_enabled ? batch.beam_count : 0u;
            bb1d_beam_instance_count += bb1d_enabled ? batch.selected_beam_count : 0u;
            tracking_event_count_total += batch.event_count;
          }
          const auto shader_buffer_addressable = [](uint64_t count, uint32_t stride) {
            return (count <= std::numeric_limits<uint32_t>::max()) && ((count * stride) <= kWavefrontMaxAddressableBufferSize);
          };
          const bool point_storage_addressable =
            shader_buffer_addressable(surface_point_count_total, kGPUUPBPDensityPointStride) && shader_buffer_addressable(surface_point_count_total, kGPUUPBPAABBStride) &&
            shader_buffer_addressable(medium_point_count_total, kGPUUPBPDensityPointStride) && shader_buffer_addressable(medium_point_count_total, kGPUUPBPAABBStride);
          if (point_storage_addressable == false) {
            set_runtime_failure("GPU UPBP compact point storage exceeds the 32-bit shader-addressable range");
            return;
          }
          const bool beam_storage_addressable = shader_buffer_addressable(bp2d_beam_instance_count, kGPUUPBPDensityBeamStride) &&
                                                shader_buffer_addressable(bp2d_beam_instance_count, kGPUUPBPBeamReferenceStride) &&
                                                shader_buffer_addressable(bb1d_beam_instance_count, kGPUUPBPDensityBeamStride);
          if (beam_storage_addressable == false) {
            set_runtime_failure("GPU UPBP compact beam storage exceeds the 32-bit shader-addressable range");
            return;
          }
          if ((_use_compute_upbp_beam_grid == false) && ((bp2d_beam_instance_count > (1ull << 24u)) || (bb1d_beam_instance_count > (1ull << 24u)))) {
            set_runtime_failure("GPU UPBP compact beam acceleration structure exceeds the 24-bit instance-index range");
            return;
          }
          _upbp.density_surface_point_count = static_cast<uint32_t>(surface_point_count_total);
          _upbp.density_medium_point_count = static_cast<uint32_t>(medium_point_count_total);
          _upbp.density_beam_count = static_cast<uint32_t>(bp2d_beam_instance_count);
          log::info("GPU UPBP density cache retained %llu tracking events in %u batch-local buffers (%.2f MiB)", static_cast<unsigned long long>(tracking_event_count_total),
            _upbp.density_batch_count, static_cast<double>(tracking_event_count_total * kGPUUPBPTrackingEventStride) / (1024.0 * 1024.0));

          const RHIBufferUsage compact_point_usage = RHIBufferUsage::Storage | RHIBufferUsage::TransferDst;
          const RHIBufferUsage compact_aabb_usage = compact_point_usage | RHIBufferUsage::AccelerationStructureBuild | RHIBufferUsage::ShaderDeviceAddress;
          const auto allocate_compact_points = [&device, compact_point_usage, compact_aabb_usage](uint64_t count, UPBPBuffer& point_buffer, UPBPBuffer& aabb_buffer,
                                                 const char* point_name, const char* aabb_name) {
            if (count == 0u) {
              return true;
            }
            return ensure_storage_buffer_capacity(device, count * kGPUUPBPDensityPointStride, compact_point_usage, point_buffer.handle, point_buffer.size,
                     point_buffer.descriptor_index, point_name) &&
                   ensure_storage_buffer_capacity(device, count * kGPUUPBPAABBStride, compact_aabb_usage, aabb_buffer.handle, aabb_buffer.size, aabb_buffer.descriptor_index,
                     aabb_name);
          };
          if ((allocate_compact_points(surface_point_count_total, _upbp.density_surface_point_buffer, _upbp.density_surface_point_aabb_buffer,
                 "upbp_density_surface_points_compact", "upbp_density_surface_point_aabbs_compact") == false) ||
              (allocate_compact_points(medium_point_count_total, _upbp.density_medium_point_buffer, _upbp.density_medium_point_aabb_buffer, "upbp_density_medium_points_compact",
                 "upbp_density_medium_point_aabbs_compact") == false)) {
            set_runtime_failure("GPU UPBP failed to allocate consolidated point storage");
            return;
          }
          if ((_upbp.density_beam_count > 0u) &&
              (ensure_storage_buffer_capacity(device, static_cast<uint64_t>(_upbp.density_beam_count) * kGPUUPBPDensityBeamStride, compact_point_usage,
                 _upbp.density_beam_buffer.handle, _upbp.density_beam_buffer.size, _upbp.density_beam_buffer.descriptor_index, "upbp_density_beams_compact") == false)) {
            set_runtime_failure("GPU UPBP failed to allocate consolidated beam storage");
            return;
          }
          record_and_submit([&](RHICommandBuffer cmd) {
            if (_upbp.density_surface_point_buffer.handle.valid()) {
              ctx.cmd_buffer_barrier(cmd, _upbp.density_surface_point_buffer.handle, RHIResourceState::Undefined, RHIResourceState::TransferDst);
              ctx.cmd_buffer_barrier(cmd, _upbp.density_surface_point_aabb_buffer.handle, RHIResourceState::Undefined, RHIResourceState::TransferDst);
            }
            if (_upbp.density_medium_point_buffer.handle.valid()) {
              ctx.cmd_buffer_barrier(cmd, _upbp.density_medium_point_buffer.handle, RHIResourceState::Undefined, RHIResourceState::TransferDst);
              ctx.cmd_buffer_barrier(cmd, _upbp.density_medium_point_aabb_buffer.handle, RHIResourceState::Undefined, RHIResourceState::TransferDst);
            }
            uint64_t surface_offset = 0u;
            uint64_t medium_offset = 0u;
            for (uint32_t batch_index = 0u; batch_index < _upbp.density_batch_count; ++batch_index) {
              const UPBPDensityBatchResources& batch = _upbp.density_batches[batch_index];
              if (batch.surface_point_count > 0u) {
                const uint64_t point_bytes = static_cast<uint64_t>(batch.surface_point_count) * kGPUUPBPDensityPointStride;
                const uint64_t aabb_bytes = static_cast<uint64_t>(batch.surface_point_count) * kGPUUPBPAABBStride;
                ctx.cmd_buffer_barrier(cmd, batch.surface_point_buffer.handle, RHIResourceState::General, RHIResourceState::TransferSrc);
                ctx.cmd_buffer_barrier(cmd, batch.surface_point_aabb_buffer.handle, RHIResourceState::General, RHIResourceState::TransferSrc);
                ctx.cmd_copy_buffer(cmd, batch.surface_point_buffer.handle, _upbp.density_surface_point_buffer.handle, point_bytes, 0u,
                  surface_offset * kGPUUPBPDensityPointStride);
                ctx.cmd_copy_buffer(cmd, batch.surface_point_aabb_buffer.handle, _upbp.density_surface_point_aabb_buffer.handle, aabb_bytes, 0u,
                  surface_offset * kGPUUPBPAABBStride);
                surface_offset += batch.surface_point_count;
              }
              if (batch.medium_point_count > 0u) {
                const uint64_t point_bytes = static_cast<uint64_t>(batch.medium_point_count) * kGPUUPBPDensityPointStride;
                const uint64_t aabb_bytes = static_cast<uint64_t>(batch.medium_point_count) * kGPUUPBPAABBStride;
                ctx.cmd_buffer_barrier(cmd, batch.medium_point_buffer.handle, RHIResourceState::General, RHIResourceState::TransferSrc);
                ctx.cmd_buffer_barrier(cmd, batch.medium_point_aabb_buffer.handle, RHIResourceState::General, RHIResourceState::TransferSrc);
                ctx.cmd_copy_buffer(cmd, batch.medium_point_buffer.handle, _upbp.density_medium_point_buffer.handle, point_bytes, 0u, medium_offset * kGPUUPBPDensityPointStride);
                ctx.cmd_copy_buffer(cmd, batch.medium_point_aabb_buffer.handle, _upbp.density_medium_point_aabb_buffer.handle, aabb_bytes, 0u, medium_offset * kGPUUPBPAABBStride);
                medium_offset += batch.medium_point_count;
              }
            }
            if (_upbp.density_surface_point_buffer.handle.valid()) {
              ctx.cmd_buffer_barrier(cmd, _upbp.density_surface_point_buffer.handle, RHIResourceState::TransferDst, RHIResourceState::General);
              ctx.cmd_buffer_barrier(cmd, _upbp.density_surface_point_aabb_buffer.handle, RHIResourceState::TransferDst, RHIResourceState::AccelerationStructure);
            }
            if (_upbp.density_medium_point_buffer.handle.valid()) {
              ctx.cmd_buffer_barrier(cmd, _upbp.density_medium_point_buffer.handle, RHIResourceState::TransferDst, RHIResourceState::General);
              ctx.cmd_buffer_barrier(cmd, _upbp.density_medium_point_aabb_buffer.handle, RHIResourceState::TransferDst, RHIResourceState::AccelerationStructure);
            }
          });
          const RHIResult point_copy_result = wait_and_destroy_submitted_commands("UPBP consolidated density storage");
          if (point_copy_result != RHIResult::Success) {
            set_runtime_failure("GPU UPBP failed to consolidate point storage (" + std::to_string(static_cast<uint32_t>(point_copy_result)) + ")");
            return;
          }
          {
            uint64_t released_bytes = 0u;
            for (UPBPDensityBatchResources& batch : _upbp.density_batches) {
              UPBPBuffer* buffers[] = {&batch.surface_point_buffer, &batch.surface_point_aabb_buffer, &batch.medium_point_buffer, &batch.medium_point_aabb_buffer};
              for (UPBPBuffer* buffer : buffers) {
                released_bytes += buffer->size;
                destroy_linear_scene_buffer(device, buffer->handle, buffer->size, buffer->descriptor_index);
              }
            }
            if (released_bytes > 0u) {
              const RHIResult reclaim_result = ctx.wait_idle();
              if (reclaim_result != RHIResult::Success) {
                set_runtime_failure("GPU UPBP failed to reclaim copied density point storage (" + std::to_string(static_cast<uint32_t>(reclaim_result)) + ")");
                return;
              }
            }
            log::info("GPU UPBP released %.2f MiB of copied density point storage before point indexing", static_cast<double>(released_bytes) / (1024.0 * 1024.0));
          }

          const auto ensure_compact_point_blas_capacity = [&device](const UPBPBuffer& aabb_buffer, uint32_t offset, uint32_t count, RHIBindlessHandle& result, uint32_t& capacity) {
            if (count == 0u) {
              return true;
            }
            if (result.valid() && (capacity >= count)) {
              return true;
            }
            if (result.valid()) {
              device.destroy_acceleration_structure(result);
              result = {};
              capacity = 0u;
            }
            RHIAccelerationStructureGeometry geometry = {};
            geometry.type = RHIAccelerationStructureGeometryType::AABBs;
            geometry.is_opaque = false;
            geometry.aabbs.buffer = aabb_buffer.handle;
            geometry.aabbs.buffer_offset = static_cast<uint64_t>(offset) * kGPUUPBPAABBStride;
            geometry.aabbs.stride = kGPUUPBPAABBStride;
            geometry.aabbs.count = count;
            RHIAccelerationStructureDesc desc = {};
            desc.type = RHIAccelerationStructureType::BottomLevel;
            desc.geometry_count = 1u;
            desc.geometries = &geometry;
            const RHICreateBindlessResult create_result = device.create_acceleration_structure(desc);
            if ((create_result.result != RHIResult::Success) || (create_result.handle.valid() == false)) {
              return false;
            }
            result = create_result.handle;
            capacity = count;
            return true;
          };
          bool point_acceleration_structures_created = ensure_compact_point_blas_capacity(_upbp.density_medium_point_aabb_buffer, 0u, _upbp.density_medium_point_count,
            _upbp.density_medium_point_blas, _upbp.density_medium_point_blas_capacity);
          if (point_acceleration_structures_created && (_upbp.density_medium_point_count > 0u)) {
            point_acceleration_structures_created = append_density_instance(_upbp.density_medium_point_blas, 0u, 0xffu, medium_point_instances);
          }
          for (uint32_t partition_index = 0u; (partition_index < kGPUUPBPSurfacePartitionCount) && point_acceleration_structures_created; ++partition_index) {
            const uint32_t partition_offset = upbp_partition_offset(_upbp.density_surface_point_count, partition_index, kGPUUPBPSurfacePartitionCount);
            const uint32_t partition_count = upbp_partition_size(_upbp.density_surface_point_count, partition_index, kGPUUPBPSurfacePartitionCount);
            point_acceleration_structures_created = ensure_compact_point_blas_capacity(_upbp.density_surface_point_aabb_buffer, partition_offset, partition_count,
              _upbp.density_surface_point_blas[partition_index], _upbp.density_surface_point_blas_capacities[partition_index]);
            if (point_acceleration_structures_created && (partition_count > 0u)) {
              point_acceleration_structures_created =
                append_density_instance(_upbp.density_surface_point_blas[partition_index], 0u, static_cast<uint8_t>(1u << partition_index), surface_point_instances);
            }
          }
          if (point_acceleration_structures_created == false) {
            set_runtime_failure("GPU UPBP failed to create consolidated point acceleration structures");
            return;
          }
          const uint64_t descriptor_bytes = static_cast<uint64_t>(batch_descriptors.size()) * kGPUUPBPDensityBatchStride;
          const RHIBufferUsage metadata_usage = RHIBufferUsage::Storage | RHIBufferUsage::TransferDst;
          if (ensure_storage_buffer_capacity(device, std::max<uint64_t>(descriptor_bytes, sizeof(uint32_t)), metadata_usage, _upbp.density_batch_buffer.handle,
                _upbp.density_batch_buffer.size, _upbp.density_batch_buffer.descriptor_index, "upbp_density_batches") == false) {
            set_runtime_failure("GPU UPBP failed to allocate compact density batch metadata");
            return;
          }
          if ((descriptor_bytes > 0u) && (device.update_buffer(_upbp.density_batch_buffer.handle, batch_descriptors.data(), descriptor_bytes) != RHIResult::Success)) {
            set_runtime_failure("GPU UPBP failed to upload compact density batch metadata");
            return;
          }

          const RHIBufferUsage instance_usage = RHIBufferUsage::AccelerationStructureBuild | RHIBufferUsage::ShaderDeviceAddress | RHIBufferUsage::TransferDst;
          const uint64_t beam_record_count = std::max(bp2d_beam_instance_count, bb1d_beam_instance_count);
          const uint64_t beam_reference_count = bp2d_enabled ? bp2d_beam_instance_count : 0u;
          if (beam_record_count > 0u) {
            uint64_t unit_beam_address = 0u;
            if (_use_compute_upbp_beam_grid == false) {
              GPUUPBPAABB unit_beam_aabb = {};
              unit_beam_aabb.minimum = {-1.0f, -1.0f, 0.0f};
              unit_beam_aabb.maximum = {1.0f, 1.0f, 1.0f};
              const RHIBufferUsage unit_aabb_usage =
                RHIBufferUsage::Storage | RHIBufferUsage::TransferDst | RHIBufferUsage::AccelerationStructureBuild | RHIBufferUsage::ShaderDeviceAddress;
              if ((ensure_storage_buffer_capacity(device, kGPUUPBPAABBStride, unit_aabb_usage, _upbp.density_beam_unit_aabb_buffer.handle, _upbp.density_beam_unit_aabb_buffer.size,
                     _upbp.density_beam_unit_aabb_buffer.descriptor_index, "upbp_density_unit_beam_aabb") == false) ||
                  (device.update_buffer(_upbp.density_beam_unit_aabb_buffer.handle, &unit_beam_aabb, kGPUUPBPAABBStride) != RHIResult::Success)) {
                set_runtime_failure("GPU UPBP failed to upload the canonical BP2D beam bounds");
                return;
              }
              RHIAccelerationStructureGeometry unit_beam_geometry = {};
              unit_beam_geometry.type = RHIAccelerationStructureGeometryType::AABBs;
              unit_beam_geometry.is_opaque = false;
              unit_beam_geometry.aabbs.buffer = _upbp.density_beam_unit_aabb_buffer.handle;
              unit_beam_geometry.aabbs.stride = kGPUUPBPAABBStride;
              unit_beam_geometry.aabbs.count = 1u;
              RHIAccelerationStructureDesc unit_beam_desc = {};
              unit_beam_desc.type = RHIAccelerationStructureType::BottomLevel;
              unit_beam_desc.geometry_count = 1u;
              unit_beam_desc.geometries = &unit_beam_geometry;
              if (_upbp.density_beam_unit_blas.valid() == false) {
                const RHICreateBindlessResult unit_beam_result = device.create_acceleration_structure(unit_beam_desc);
                if ((unit_beam_result.result != RHIResult::Success) || (unit_beam_result.handle.valid() == false)) {
                  set_runtime_failure("GPU UPBP failed to create the canonical BP2D beam acceleration structure");
                  return;
                }
                _upbp.density_beam_unit_blas = unit_beam_result.handle;
              }
              unit_beam_address = device.get_acceleration_structure_device_address(_upbp.density_beam_unit_blas);
              if (unit_beam_address == 0u) {
                set_runtime_failure("GPU UPBP failed to obtain the canonical BP2D beam acceleration-structure address");
                return;
              }
            }
            const uint64_t bp2d_instance_bytes = _use_compute_upbp_beam_grid ? 0u : bp2d_beam_instance_count * kGPUUPBPAccelerationStructureInstanceStride;
            const uint64_t bb1d_instance_bytes = _use_compute_upbp_beam_grid ? 0u : bb1d_beam_instance_count * kGPUUPBPAccelerationStructureInstanceStride;
            const uint64_t bb1d_beam_bytes = bb1d_beam_instance_count * kGPUUPBPDensityBeamStride;
            const uint64_t reference_bytes = beam_reference_count * kGPUUPBPBeamReferenceStride;
            const RHIBufferUsage gpu_instance_usage = RHIBufferUsage::Storage | RHIBufferUsage::AccelerationStructureBuild | RHIBufferUsage::ShaderDeviceAddress;
            if (((bp2d_instance_bytes > 0u) &&
                  (ensure_storage_buffer_capacity(device, bp2d_instance_bytes, gpu_instance_usage, _upbp.density_bp2d_beam_instance_buffer.handle,
                     _upbp.density_bp2d_beam_instance_buffer.size, _upbp.density_bp2d_beam_instance_buffer.descriptor_index, "upbp_density_bp2d_beam_instances") == false)) ||
                ((bb1d_instance_bytes > 0u) &&
                  (ensure_storage_buffer_capacity(device, bb1d_instance_bytes, gpu_instance_usage, _upbp.density_bb1d_beam_instance_buffer.handle,
                     _upbp.density_bb1d_beam_instance_buffer.size, _upbp.density_bb1d_beam_instance_buffer.descriptor_index, "upbp_density_bb1d_beam_instances") == false)) ||
                ((bb1d_beam_bytes > 0u) && (ensure_storage_buffer_capacity(device, bb1d_beam_bytes, RHIBufferUsage::Storage, _upbp.density_bb1d_beam_buffer.handle,
                                              _upbp.density_bb1d_beam_buffer.size, _upbp.density_bb1d_beam_buffer.descriptor_index, "upbp_density_bb1d_beams") == false)) ||
                ((reference_bytes > 0u) &&
                  (ensure_storage_buffer_capacity(device, reference_bytes, RHIBufferUsage::Storage, _upbp.density_beam_reference_buffer.handle,
                     _upbp.density_beam_reference_buffer.size, _upbp.density_beam_reference_buffer.descriptor_index, "upbp_density_beam_references") == false))) {
              set_runtime_failure("GPU UPBP failed to allocate GPU-authored compact beam instances");
              return;
            }
            _upbp.resources.density_batch_buffer = _upbp.density_batch_buffer.descriptor_index;
            _upbp.resources.density_batch_count = _upbp.density_batch_count;
            _upbp.resources.density_output_beam_instance_buffer = bp2d_instance_bytes > 0u ? _upbp.density_bp2d_beam_instance_buffer.descriptor_index : kInvalidDescriptorIndex;
            _upbp.resources.density_output_beam_reference_buffer = reference_bytes > 0u ? _upbp.density_beam_reference_buffer.descriptor_index : kInvalidDescriptorIndex;
            _upbp.resources.density_output_beam_instance_capacity = static_cast<uint32_t>(bp2d_beam_instance_count);
            _upbp.resources.density_output_beam_buffer = _upbp.density_beam_buffer.handle.valid() ? _upbp.density_beam_buffer.descriptor_index : kInvalidDescriptorIndex;
            _upbp.resources.density_output_beam_capacity = _upbp.density_beam_count;
            _upbp.resources.density_output_event_buffer = kInvalidDescriptorIndex;
            _upbp.resources.density_output_bb1d_beam_instance_buffer =
              bb1d_instance_bytes > 0u ? _upbp.density_bb1d_beam_instance_buffer.descriptor_index : kInvalidDescriptorIndex;
            _upbp.resources.density_output_bb1d_beam_instance_capacity = static_cast<uint32_t>(bb1d_beam_instance_count);
            _upbp.resources.bb1d_beam_buffer = bb1d_beam_instance_count > 0u ? _upbp.density_bb1d_beam_buffer.descriptor_index : kInvalidDescriptorIndex;
            _upbp.resources.density_beam_acceleration_structure_reference_low = static_cast<uint32_t>(unit_beam_address);
            _upbp.resources.density_beam_acceleration_structure_reference_high = static_cast<uint32_t>(unit_beam_address >> 32u);
            if (update_upbp_iteration_resources(device, scene, upbp_global_path_count) == false) {
              set_runtime_failure("GPU UPBP failed to bind GPU-authored compact beam instances");
              return;
            }
            record_and_submit([&](RHICommandBuffer cmd) {
              ctx.cmd_buffer_barrier(cmd, _upbp.density_batch_buffer.handle, RHIResourceState::TransferDst, RHIResourceState::General);
              if (_upbp.density_beam_buffer.handle.valid()) {
                ctx.cmd_buffer_barrier(cmd, _upbp.density_beam_buffer.handle, RHIResourceState::Undefined, RHIResourceState::General);
              }
              if (_upbp.density_bp2d_beam_instance_buffer.handle.valid()) {
                ctx.cmd_buffer_barrier(cmd, _upbp.density_bp2d_beam_instance_buffer.handle, RHIResourceState::Undefined, RHIResourceState::General);
              }
              if (_upbp.density_bb1d_beam_instance_buffer.handle.valid()) {
                ctx.cmd_buffer_barrier(cmd, _upbp.density_bb1d_beam_instance_buffer.handle, RHIResourceState::Undefined, RHIResourceState::General);
              }
              if (_upbp.density_bb1d_beam_buffer.handle.valid()) {
                ctx.cmd_buffer_barrier(cmd, _upbp.density_bb1d_beam_buffer.handle, RHIResourceState::Undefined, RHIResourceState::General);
              }
              if (_upbp.density_beam_reference_buffer.handle.valid()) {
                ctx.cmd_buffer_barrier(cmd, _upbp.density_beam_reference_buffer.handle, RHIResourceState::Undefined, RHIResourceState::General);
              }
              dispatch_stage_mode(cmd, PipelineStage::UPBPClear, scalar_dispatch, 0u, GPUUPBPClearMode::BeamInstances);
              barrier_wavefront_buffers(cmd);
              for (uint32_t batch_index = 0u; batch_index < _upbp.density_batch_count; ++batch_index) {
                const uint32_t batch_beam_count = _upbp.density_batches[batch_index].beam_count;
                if (batch_beam_count == 0u) {
                  continue;
                }
                dispatch_stage_range(cmd, PipelineStage::UPBPBeamInstances, 0u, batch_beam_count, 0u, batch_index);
                ctx.cmd_compute_barrier(cmd);
              }
              if (_upbp.density_bp2d_beam_instance_buffer.handle.valid()) {
                ctx.cmd_buffer_barrier(cmd, _upbp.density_bp2d_beam_instance_buffer.handle, RHIResourceState::General, RHIResourceState::AccelerationStructure);
              }
              if (_upbp.density_bb1d_beam_instance_buffer.handle.valid()) {
                ctx.cmd_buffer_barrier(cmd, _upbp.density_bb1d_beam_instance_buffer.handle, RHIResourceState::General, RHIResourceState::AccelerationStructure);
              }
              ctx.cmd_buffer_barrier(cmd, _upbp.counter_buffer.handle, RHIResourceState::General, RHIResourceState::TransferSrc);
              ctx.cmd_buffer_barrier(cmd, _upbp.counter_readback_buffer.handle, _upbp.counter_readback_state, RHIResourceState::TransferDst);
              ctx.cmd_copy_buffer(cmd, _upbp.counter_buffer.handle, _upbp.counter_readback_buffer.handle, static_cast<uint64_t>(GPUUPBPCounterIndex::Count) * sizeof(uint32_t));
              ctx.cmd_buffer_barrier(cmd, _upbp.counter_buffer.handle, RHIResourceState::TransferSrc, RHIResourceState::General);
              _upbp.counter_readback_state = RHIResourceState::TransferDst;
            });
            const RHIResult instance_generation_result = wait_and_destroy_submitted_commands("UPBP compact beam instance generation");
            if (instance_generation_result != RHIResult::Success) {
              set_runtime_failure("GPU UPBP compact beam instance generation failed (" + std::to_string(static_cast<uint32_t>(instance_generation_result)) + ")");
              return;
            }
            const RHIResult instance_counter_result = device.read_buffer(_upbp.counter_readback_buffer.handle, upbp_counters, sizeof(upbp_counters));
            if (instance_counter_result != RHIResult::Success) {
              set_runtime_failure("GPU UPBP failed to validate GPU-authored compact beam instances (" + std::to_string(static_cast<uint32_t>(instance_counter_result)) + ")");
              return;
            }
            if ((upbp_counters[GPUUPBPCounterIndex::OverflowFlags] & GPUUPBPOverflowFlags::BeamInstance) != 0u) {
              set_runtime_failure("GPU UPBP compact beam instance generation exceeded its exact measured capacity");
              return;
            }
            if (bb1d_enabled && (upbp_counters[GPUUPBPCounterIndex::DensityBeamInstance] != bb1d_beam_instance_count)) {
              set_runtime_failure("GPU UPBP compact BB1D beam instance count changed after density compaction");
              return;
            }
          }

          uint64_t released_beam_bytes = 0u;
          for (UPBPDensityBatchResources& batch : _upbp.density_batches) {
            released_beam_bytes += batch.beam_buffer.size;
            destroy_linear_scene_buffer(device, batch.beam_buffer.handle, batch.beam_buffer.size, batch.beam_buffer.descriptor_index);
          }
          if (released_beam_bytes > 0u) {
            const RHIResult reclaim_result = ctx.wait_idle();
            if (reclaim_result != RHIResult::Success) {
              set_runtime_failure("GPU UPBP failed to reclaim compacted density beam storage (" + std::to_string(static_cast<uint32_t>(reclaim_result)) + ")");
              return;
            }
            log::info("GPU UPBP released %.2f MiB of compacted density beam storage before beam indexing", static_cast<double>(released_beam_bytes) / (1024.0 * 1024.0));
          }

          const uint32_t bb1d_instance_count = static_cast<uint32_t>(bb1d_beam_instance_count);
          _upbp.density_bb1d_beam_count = bb1d_instance_count;
          if ((_use_compute_upbp_beam_grid == false) && bp2d_enabled && (bp2d_beam_instance_count > 0u)) {
            const uint32_t instance_count = static_cast<uint32_t>(bp2d_beam_instance_count);
            const uint32_t desired_partition_count = divide_round_up(instance_count, kUPBPBP2DTargetPartitionInstances);
            const RHIBindlessManager& bindless = ctx.bindless();
            const uint32_t maximum_acceleration_structure_count = bindless.get_max_acceleration_structures();
            const uint32_t acceleration_structure_count = bindless.get_acceleration_structure_count();
            const uint32_t usable_acceleration_structure_count = maximum_acceleration_structure_count > 0u ? maximum_acceleration_structure_count - 1u : 0u;
            const uint32_t free_acceleration_structure_slots =
              usable_acceleration_structure_count > acceleration_structure_count ? usable_acceleration_structure_count - acceleration_structure_count : 0u;

            uint32_t existing_bp2d_partition_count = 0u;
            for (const RHIBindlessHandle partition_tlas : _upbp.density_bp2d_beam_tlas) {
              existing_bp2d_partition_count += partition_tlas.valid() ? 1u : 0u;
            }

            uint32_t reserved_density_tlas_slots = 0u;
            reserved_density_tlas_slots += (surface_point_instances.empty() == false) && (_upbp.density_surface_point_tlas.valid() == false) ? 1u : 0u;
            reserved_density_tlas_slots += (medium_point_instances.empty() == false) && (_upbp.density_medium_point_tlas.valid() == false) ? 1u : 0u;
            for (uint32_t partition_index = 0u; partition_index < kGPUUPBPBB1DPartitionCount; ++partition_index) {
              const uint32_t partition_size = upbp_partition_size(bb1d_instance_count, partition_index, kGPUUPBPBB1DPartitionCount);
              reserved_density_tlas_slots += (partition_size > 0u) && (_upbp.density_bb1d_beam_tlas[partition_index].valid() == false) ? 1u : 0u;
            }

            const uint32_t reusable_or_free_slots = existing_bp2d_partition_count + free_acceleration_structure_slots;
            if (reusable_or_free_slots <= reserved_density_tlas_slots) {
              set_runtime_failure("GPU UPBP acceleration-structure descriptor capacity is insufficient for density indexing (" + std::to_string(acceleration_structure_count) +
                                  "/" + std::to_string(usable_acceleration_structure_count) + " in use, " + std::to_string(reserved_density_tlas_slots) +
                                  " required for point and BB1D structures)");
              return;
            }
            const uint32_t maximum_bp2d_partition_count = reusable_or_free_slots - reserved_density_tlas_slots;
            const uint32_t partition_count_total = std::min(desired_partition_count, maximum_bp2d_partition_count);
            const uint32_t maximum_partition_instances = divide_round_up(instance_count, partition_count_total);
            log::info("GPU UPBP BP2D acceleration structure: %u beams, radius %.9g, %u/%u partitions, at most %u instances per partition, AS slots %u/%u with %u reserved",
              instance_count, static_cast<double>(_upbp.resources.iteration.bp2d_radius), partition_count_total, desired_partition_count, maximum_partition_instances,
              acceleration_structure_count, usable_acceleration_structure_count, reserved_density_tlas_slots);
            while (_upbp.density_bp2d_beam_tlas.size() > partition_count_total) {
              RHIBindlessHandle& partition_tlas = _upbp.density_bp2d_beam_tlas.back();
              if (partition_tlas.valid()) {
                device.destroy_acceleration_structure(partition_tlas);
              }
              _upbp.density_bp2d_beam_tlas.pop_back();
              _upbp.density_bp2d_beam_tlas_capacities.pop_back();
            }
            _upbp.density_bp2d_beam_tlas.resize(partition_count_total);
            _upbp.density_bp2d_beam_tlas_capacities.resize(partition_count_total);
            for (uint32_t partition_index = 0u; partition_index < partition_count_total; ++partition_index) {
              const uint32_t partition_count = upbp_partition_size(instance_count, partition_index, partition_count_total);
              RHIBindlessHandle& partition_tlas = _upbp.density_bp2d_beam_tlas[partition_index];
              uint32_t& partition_capacity = _upbp.density_bp2d_beam_tlas_capacities[partition_index];
              if (partition_tlas.valid() && (partition_capacity < partition_count)) {
                device.destroy_acceleration_structure(partition_tlas);
                partition_tlas = {};
                partition_capacity = 0u;
              }
              if ((partition_count == 0u) || partition_tlas.valid()) {
                continue;
              }
              RHIAccelerationStructureDesc desc = {};
              desc.type = RHIAccelerationStructureType::TopLevel;
              desc.instance_count = partition_count;
              const RHICreateBindlessResult create_result = device.create_acceleration_structure(desc);
              if ((create_result.result != RHIResult::Success) || (create_result.handle.valid() == false)) {
                set_runtime_failure("GPU UPBP failed to create a BP2D beam acceleration-structure partition");
                return;
              }
              partition_tlas = create_result.handle;
              partition_capacity = partition_count;
            }
          }

          const auto ensure_density_tlas_capacity = [&](const std::vector<RHIAccelerationStructureInstance>& instances, const char* name, UPBPBuffer& instance_buffer,
                                                      RHIBindlessHandle& tlas, uint32_t& capacity) {
            if (instances.empty()) {
              return true;
            }
            const uint64_t instance_bytes = static_cast<uint64_t>(instances.size()) * sizeof(RHIAccelerationStructureInstance);
            if ((ensure_storage_buffer_capacity(device, instance_bytes, instance_usage, instance_buffer.handle, instance_buffer.size, instance_buffer.descriptor_index, name) ==
                  false) ||
                (device.update_buffer(instance_buffer.handle, instances.data(), instance_bytes) != RHIResult::Success)) {
              return false;
            }
            const uint32_t instance_count = static_cast<uint32_t>(instances.size());
            if (tlas.valid() && (capacity >= instance_count)) {
              return true;
            }
            if (tlas.valid()) {
              device.destroy_acceleration_structure(tlas);
              tlas = {};
              capacity = 0u;
            }
            RHIAccelerationStructureDesc desc = {};
            desc.type = RHIAccelerationStructureType::TopLevel;
            desc.instance_count = instance_count;
            const RHICreateBindlessResult create_result = device.create_acceleration_structure(desc);
            if ((create_result.result != RHIResult::Success) || (create_result.handle.valid() == false)) {
              return false;
            }
            tlas = create_result.handle;
            capacity = instance_count;
            return true;
          };
          const auto ensure_density_tlas_capacity_from_gpu_instances = [&](uint32_t instance_count, RHIBindlessHandle& tlas, uint32_t& capacity) {
            if (instance_count == 0u) {
              return true;
            }
            if (tlas.valid() && (capacity >= instance_count)) {
              return true;
            }
            if (tlas.valid()) {
              device.destroy_acceleration_structure(tlas);
              tlas = {};
              capacity = 0u;
            }
            RHIAccelerationStructureDesc desc = {};
            desc.type = RHIAccelerationStructureType::TopLevel;
            desc.instance_count = instance_count;
            const RHICreateBindlessResult create_result = device.create_acceleration_structure(desc);
            if ((create_result.result != RHIResult::Success) || (create_result.handle.valid() == false)) {
              return false;
            }
            tlas = create_result.handle;
            capacity = instance_count;
            return true;
          };
          bool density_tlas_created = ensure_density_tlas_capacity(surface_point_instances, "upbp_density_surface_point_instances", _upbp.density_surface_point_instance_buffer,
                                        _upbp.density_surface_point_tlas, _upbp.density_surface_point_tlas_capacity) &&
                                      ensure_density_tlas_capacity(medium_point_instances, "upbp_density_medium_point_instances", _upbp.density_medium_point_instance_buffer,
                                        _upbp.density_medium_point_tlas, _upbp.density_medium_point_tlas_capacity);
          if (_use_compute_upbp_beam_grid == false) {
            for (uint32_t partition_index = 0u; (partition_index < kGPUUPBPBB1DPartitionCount) && density_tlas_created; ++partition_index) {
              density_tlas_created = ensure_density_tlas_capacity_from_gpu_instances(upbp_partition_size(bb1d_instance_count, partition_index, kGPUUPBPBB1DPartitionCount),
                _upbp.density_bb1d_beam_tlas[partition_index], _upbp.density_bb1d_beam_tlas_capacities[partition_index]);
            }
          }
          if (density_tlas_created == false) {
            set_runtime_failure("GPU UPBP failed to create compact density top-level acceleration structures");
            return;
          }
          uint64_t tlas_scratch_size = 0u;
          for (const RHIBindlessHandle blas : _upbp.density_surface_point_blas) {
            if (blas.valid()) {
              tlas_scratch_size = std::max(tlas_scratch_size, device.get_acceleration_structure_build_scratch_size(blas));
            }
          }
          for (const RHIBindlessHandle blas : {_upbp.density_medium_point_blas, _upbp.density_beam_unit_blas}) {
            if (blas.valid()) {
              tlas_scratch_size = std::max(tlas_scratch_size, device.get_acceleration_structure_build_scratch_size(blas));
            }
          }
          for (const RHIBindlessHandle tlas : {_upbp.density_surface_point_tlas, _upbp.density_medium_point_tlas}) {
            if (tlas.valid()) {
              tlas_scratch_size = std::max(tlas_scratch_size, device.get_acceleration_structure_build_scratch_size(tlas));
            }
          }
          for (const RHIBindlessHandle tlas : _upbp.density_bp2d_beam_tlas) {
            if (tlas.valid()) {
              tlas_scratch_size = std::max(tlas_scratch_size, device.get_acceleration_structure_build_scratch_size(tlas));
            }
          }
          for (const RHIBindlessHandle tlas : _upbp.density_bb1d_beam_tlas) {
            if (tlas.valid()) {
              tlas_scratch_size = std::max(tlas_scratch_size, device.get_acceleration_structure_build_scratch_size(tlas));
            }
          }
          const RHIBufferUsage scratch_usage = RHIBufferUsage::AccelerationStructureBuild | RHIBufferUsage::ShaderDeviceAddress;
          if ((tlas_scratch_size > 0u) && (ensure_storage_buffer_capacity(device, tlas_scratch_size, scratch_usage, _upbp.density_as_scratch_buffer.handle,
                                             _upbp.density_as_scratch_buffer.size, _upbp.density_as_scratch_buffer.descriptor_index, "upbp_density_as_scratch") == false)) {
            set_runtime_failure("GPU UPBP failed to allocate compact density TLAS scratch storage");
            return;
          }
          record_and_submit([&](RHICommandBuffer cmd) {
            const auto build_compact_point_blas = [&](RHIBindlessHandle blas, const UPBPBuffer& aabb_buffer, uint32_t offset, uint32_t count) {
              if ((blas.valid() == false) || (count == 0u)) {
                return;
              }
              RHIAccelerationStructureGeometry geometry = {};
              geometry.type = RHIAccelerationStructureGeometryType::AABBs;
              geometry.is_opaque = false;
              geometry.aabbs.buffer = aabb_buffer.handle;
              geometry.aabbs.buffer_offset = static_cast<uint64_t>(offset) * kGPUUPBPAABBStride;
              geometry.aabbs.stride = kGPUUPBPAABBStride;
              geometry.aabbs.count = count;
              RHIAccelerationStructureBuildDesc desc = {};
              desc.as_handle = blas;
              desc.type = RHIAccelerationStructureType::BottomLevel;
              desc.geometry_count = 1u;
              desc.geometries = &geometry;
              ctx.cmd_build_acceleration_structure(cmd, desc, _upbp.density_as_scratch_buffer.handle, 0u);
              ctx.cmd_buffer_barrier(cmd, _upbp.density_as_scratch_buffer.handle, RHIResourceState::AccelerationStructure, RHIResourceState::AccelerationStructure);
            };
            for (uint32_t partition_index = 0u; partition_index < kGPUUPBPSurfacePartitionCount; ++partition_index) {
              const uint32_t partition_offset = upbp_partition_offset(_upbp.density_surface_point_count, partition_index, kGPUUPBPSurfacePartitionCount);
              const uint32_t partition_count = upbp_partition_size(_upbp.density_surface_point_count, partition_index, kGPUUPBPSurfacePartitionCount);
              build_compact_point_blas(_upbp.density_surface_point_blas[partition_index], _upbp.density_surface_point_aabb_buffer, partition_offset, partition_count);
            }
            build_compact_point_blas(_upbp.density_medium_point_blas, _upbp.density_medium_point_aabb_buffer, 0u, _upbp.density_medium_point_count);
            if (_upbp.density_beam_unit_blas.valid() && (beam_record_count > 0u)) {
              ctx.cmd_buffer_barrier(cmd, _upbp.density_beam_unit_aabb_buffer.handle, RHIResourceState::TransferDst, RHIResourceState::AccelerationStructure);
              RHIAccelerationStructureGeometry geometry = {};
              geometry.type = RHIAccelerationStructureGeometryType::AABBs;
              geometry.is_opaque = false;
              geometry.aabbs.buffer = _upbp.density_beam_unit_aabb_buffer.handle;
              geometry.aabbs.stride = kGPUUPBPAABBStride;
              geometry.aabbs.count = 1u;
              RHIAccelerationStructureBuildDesc desc = {};
              desc.as_handle = _upbp.density_beam_unit_blas;
              desc.type = RHIAccelerationStructureType::BottomLevel;
              desc.geometry_count = 1u;
              desc.geometries = &geometry;
              ctx.cmd_build_acceleration_structure(cmd, desc, _upbp.density_as_scratch_buffer.handle, 0u);
              ctx.cmd_buffer_barrier(cmd, _upbp.density_as_scratch_buffer.handle, RHIResourceState::AccelerationStructure, RHIResourceState::AccelerationStructure);
            }
            const auto build_density_tlas = [&](RHIBindlessHandle tlas, const UPBPBuffer& instance_buffer, uint64_t instance_buffer_offset, uint32_t instance_count,
                                              RHIBindlessHandle uniform_instance_blas) {
              if ((tlas.valid() == false) || (instance_count == 0u)) {
                return;
              }
              RHIAccelerationStructureBuildDesc desc = {};
              desc.as_handle = tlas;
              desc.type = RHIAccelerationStructureType::TopLevel;
              desc.instance_count = instance_count;
              desc.instance_buffer = instance_buffer.handle;
              desc.instance_buffer_offset = instance_buffer_offset;
              desc.uniform_instance_acceleration_structure = uniform_instance_blas;
              ctx.cmd_build_acceleration_structure(cmd, desc, _upbp.density_as_scratch_buffer.handle, 0u);
              ctx.cmd_buffer_barrier(cmd, _upbp.density_as_scratch_buffer.handle, RHIResourceState::AccelerationStructure, RHIResourceState::AccelerationStructure);
            };
            if (_upbp.density_surface_point_tlas.valid() && (surface_point_instances.empty() == false)) {
              ctx.cmd_buffer_barrier(cmd, _upbp.density_surface_point_instance_buffer.handle, RHIResourceState::TransferDst, RHIResourceState::AccelerationStructure);
            }
            if (_upbp.density_medium_point_tlas.valid() && (medium_point_instances.empty() == false)) {
              ctx.cmd_buffer_barrier(cmd, _upbp.density_medium_point_instance_buffer.handle, RHIResourceState::TransferDst, RHIResourceState::AccelerationStructure);
            }
            if ((_upbp.density_bp2d_beam_tlas.empty() == false) && _upbp.density_bp2d_beam_tlas[0].valid() && (bp2d_beam_instance_count > 0u)) {
              ctx.cmd_buffer_barrier(cmd, _upbp.density_bp2d_beam_instance_buffer.handle, RHIResourceState::AccelerationStructure, RHIResourceState::AccelerationStructure);
            }
            build_density_tlas(_upbp.density_surface_point_tlas, _upbp.density_surface_point_instance_buffer, 0u, static_cast<uint32_t>(surface_point_instances.size()), {});
            build_density_tlas(_upbp.density_medium_point_tlas, _upbp.density_medium_point_instance_buffer, 0u, static_cast<uint32_t>(medium_point_instances.size()), {});
            const uint32_t bp2d_instance_count = static_cast<uint32_t>(bp2d_beam_instance_count);
            const uint32_t bp2d_partition_count = static_cast<uint32_t>(_upbp.density_bp2d_beam_tlas.size());
            for (uint32_t partition_index = 0u; partition_index < bp2d_partition_count; ++partition_index) {
              const uint32_t partition_count = upbp_partition_size(bp2d_instance_count, partition_index, bp2d_partition_count);
              const uint64_t partition_offset =
                static_cast<uint64_t>(upbp_partition_offset(bp2d_instance_count, partition_index, bp2d_partition_count)) * kGPUUPBPAccelerationStructureInstanceStride;
              build_density_tlas(_upbp.density_bp2d_beam_tlas[partition_index], _upbp.density_bp2d_beam_instance_buffer, partition_offset, partition_count,
                _upbp.density_beam_unit_blas);
            }
            for (uint32_t partition_index = 0u; partition_index < kGPUUPBPBB1DPartitionCount; ++partition_index) {
              const uint32_t partition_count = upbp_partition_size(bb1d_instance_count, partition_index, kGPUUPBPBB1DPartitionCount);
              const uint64_t partition_offset =
                static_cast<uint64_t>(upbp_partition_offset(bb1d_instance_count, partition_index, kGPUUPBPBB1DPartitionCount)) * kGPUUPBPAccelerationStructureInstanceStride;
              build_density_tlas(_upbp.density_bb1d_beam_tlas[partition_index], _upbp.density_bb1d_beam_instance_buffer, partition_offset, partition_count,
                _upbp.density_beam_unit_blas);
            }
          });
          const RHIResult tlas_build_result = wait_and_destroy_submitted_commands("UPBP compact density TLAS build");
          if (tlas_build_result != RHIResult::Success) {
            set_runtime_failure("GPU UPBP compact density TLAS build failed (" + std::to_string(static_cast<uint32_t>(tlas_build_result)) + ")");
            return;
          }
          {
            uint64_t released_build_bytes = 0u;
            const auto release_build_buffer = [&device, &released_build_bytes](UPBPBuffer& buffer) {
              released_build_bytes += buffer.size;
              destroy_linear_scene_buffer(device, buffer.handle, buffer.size, buffer.descriptor_index);
            };
            release_build_buffer(_upbp.density_surface_point_aabb_buffer);
            release_build_buffer(_upbp.density_medium_point_aabb_buffer);
            release_build_buffer(_upbp.density_surface_point_instance_buffer);
            release_build_buffer(_upbp.density_medium_point_instance_buffer);
            release_build_buffer(_upbp.density_bp2d_beam_instance_buffer);
            release_build_buffer(_upbp.density_bb1d_beam_instance_buffer);
            release_build_buffer(_upbp.density_beam_unit_aabb_buffer);
            release_build_buffer(_upbp.density_as_scratch_buffer);
            if (released_build_bytes > 0u) {
              const RHIResult reclaim_result = ctx.wait_idle();
              if (reclaim_result != RHIResult::Success) {
                set_runtime_failure("GPU UPBP failed to reclaim density-index build storage (" + std::to_string(static_cast<uint32_t>(reclaim_result)) + ")");
                return;
              }
              log::info("GPU UPBP released %.2f MiB of density acceleration-structure build inputs", static_cast<double>(released_build_bytes) / (1024.0 * 1024.0));
            }
          }
          if (_use_compute_upbp_beam_grid) {
            const RHIBufferUsage grid_usage = RHIBufferUsage::Storage | RHIBufferUsage::TransferSrc;
            std::string beam_grid_failure_reason = {};
            if (ensure_host_visible_buffer(device, kGPUUPBPBeamGridMetadataStride, RHIBufferUsage::TransferDst, _upbp.density_beam_grid_metadata_readback_buffer.handle,
                  _upbp.density_beam_grid_metadata_readback_buffer.size, _upbp.density_beam_grid_metadata_readback_buffer.descriptor_index,
                  "upbp_beam_grid_metadata_readback") == false) {
              set_runtime_failure("GPU UPBP failed to allocate beam-grid metadata readback storage");
              return;
            }
            const auto build_gpu_beam_grid = [&](uint32_t grid_type, uint32_t beam_count, const UPBPBuffer& source, UPBPBeamGridResources& output, const char* name) {
              const auto fail = [&](std::string detail) {
                beam_grid_failure_reason = std::string("GPU UPBP ") + name + " " + std::move(detail);
                log::error("%s", beam_grid_failure_reason.c_str());
                return false;
              };
              output.beam_index_count = 0u;
              if (beam_count == 0u) {
                return true;
              }
              const std::string metadata_name = std::string(name) + "_metadata";
              const std::string offsets_name = std::string(name) + "_offsets";
              const std::string indices_name = std::string(name) + "_indices";
              const bool metadata_reallocated = (output.metadata_buffer.handle.valid() == false) || (output.metadata_buffer.size < kGPUUPBPBeamGridMetadataStride);
              if (ensure_storage_buffer_capacity(device, kGPUUPBPBeamGridMetadataStride, grid_usage, output.metadata_buffer.handle, output.metadata_buffer.size,
                    output.metadata_buffer.descriptor_index, metadata_name.c_str()) == false) {
                return fail("failed to allocate metadata");
              }
              if (metadata_reallocated) {
                output.metadata_state = RHIResourceState::Undefined;
              }
              GPUUPBPBeamGridResources build_resources = {};
              build_resources.metadata_buffer = output.metadata_buffer.descriptor_index;
              build_resources.beam_count = beam_count;
              build_resources.reserved1 = source.descriptor_index;
              const auto bind_build_resources = [&]() {
                if (grid_type == GPUUPBPBeamGridType::BP2D) {
                  _upbp.resources.bp2d_beam_grid = build_resources;
                } else {
                  _upbp.resources.bb1d_beam_grid = build_resources;
                }
                return device.update_buffer(_upbp.resources_buffer.handle, &_upbp.resources, kGPUUPBPResourcesStride) == RHIResult::Success;
              };
              if (bind_build_resources() == false) {
                return fail("failed to bind description resources");
              }
              record_and_submit([&](RHICommandBuffer cmd) {
                ctx.cmd_buffer_barrier(cmd, output.metadata_buffer.handle, output.metadata_state, RHIResourceState::General);
                output.metadata_state = RHIResourceState::General;
                dispatch_stage_range(cmd, PipelineStage::UPBPBeamGridBuild, 0u, 1u, grid_type, GPUUPBPBeamGridBuildMode::Describe);
                ctx.cmd_compute_barrier(cmd);
                ctx.cmd_buffer_barrier(cmd, output.metadata_buffer.handle, RHIResourceState::General, RHIResourceState::TransferSrc);
                ctx.cmd_buffer_barrier(cmd, _upbp.density_beam_grid_metadata_readback_buffer.handle, _upbp.density_beam_grid_metadata_readback_state,
                  RHIResourceState::TransferDst);
                ctx.cmd_copy_buffer(cmd, output.metadata_buffer.handle, _upbp.density_beam_grid_metadata_readback_buffer.handle, kGPUUPBPBeamGridMetadataStride);
                ctx.cmd_buffer_barrier(cmd, output.metadata_buffer.handle, RHIResourceState::TransferSrc, RHIResourceState::General);
                _upbp.density_beam_grid_metadata_readback_state = RHIResourceState::TransferDst;
              });
              RHIResult build_result = wait_and_destroy_submitted_commands("UPBP beam-grid description");
              if (build_result != RHIResult::Success) {
                return fail("description dispatch failed (" + std::to_string(static_cast<uint32_t>(build_result)) + ")");
              }
              GPUUPBPBeamGridMetadata metadata = {};
              const RHIResult description_read_result = device.read_buffer(_upbp.density_beam_grid_metadata_readback_buffer.handle, &metadata, sizeof(metadata));
              if (description_read_result != RHIResult::Success) {
                return fail("failed to read its description (" + std::to_string(static_cast<uint32_t>(description_read_result)) + ")");
              }
              if ((metadata.reserved0 != GPUUPBPBeamGridBuildFailure::None) || (metadata.beam_count != beam_count) || (metadata.cell_count == 0u) ||
                  (metadata.cell_count > (kUPBPBeamGridMaximumResolution * kUPBPBeamGridMaximumResolution * kUPBPBeamGridMaximumResolution))) {
                return fail("description is invalid: " + std::string(upbp_beam_grid_failure_to_string(metadata.reserved0)) + " (status " + std::to_string(metadata.reserved0) +
                            ", beams " + std::to_string(metadata.beam_count) + "/" + std::to_string(beam_count) + ", cells " + std::to_string(metadata.cell_count) + ")");
              }
              const uint64_t shard_stride_bytes = static_cast<uint64_t>(metadata.cell_count) * 2u * sizeof(uint32_t);
              const uint32_t scratch_limited_shard_count = static_cast<uint32_t>(kUPBPBeamGridScratchBudget / shard_stride_bytes);
              const uint32_t shard_count = std::min(beam_count, std::min(kUPBPBeamGridMaximumShardCount, std::max(1u, scratch_limited_shard_count)));
              const uint64_t shard_cell_count = static_cast<uint64_t>(shard_count) * metadata.cell_count;
              const uint64_t scratch_bytes = shard_cell_count * 2u * sizeof(uint32_t);
              const uint64_t offset_bytes = static_cast<uint64_t>(metadata.cell_count + 1u) * sizeof(uint32_t);
              if ((scratch_bytes > kWavefrontMaxAddressableBufferSize) || (offset_bytes > kWavefrontMaxAddressableBufferSize)) {
                return fail("construction storage exceeds the shader-addressable range");
              }
              const auto ensure_grid_capacity = [&](uint64_t required_size, uint64_t preferred_size, RHIBufferUsage usage, UPBPBuffer& buffer, RHIResourceState& state,
                                                  const char* buffer_name) {
                if (buffer.handle.valid() && (buffer.size >= required_size)) {
                  buffer.descriptor_index = get_bindless_descriptor_index(buffer.handle);
                  return true;
                }
                if (buffer.handle.valid()) {
                  const RHIResult destroy_result = device.destroy_buffer(buffer.handle);
                  if (destroy_result != RHIResult::Success) {
                    return false;
                  }
                  buffer = {};
                  state = RHIResourceState::Undefined;
                }
                RHIBufferDesc desc = {
                  .size = preferred_size,
                  .usage = usage,
                };
                RHICreateBindlessResult create_result = device.create_buffer(desc);
                if (((create_result.result != RHIResult::Success) || (create_result.handle.valid() == false)) && (preferred_size != required_size)) {
                  if (create_result.handle.valid()) {
                    device.destroy_buffer(create_result.handle);
                  }
                  desc.size = required_size;
                  create_result = device.create_buffer(desc);
                }
                if ((create_result.result != RHIResult::Success) || (create_result.handle.valid() == false)) {
                  if (create_result.handle.valid()) {
                    device.destroy_buffer(create_result.handle);
                  }
                  const RHIMemoryStats memory_stats = device.get_memory_statistics();
                  log::error("GPU RT: failed to create '%s' storage buffer (%u): required=%llu bytes (%.2fMB), preferred=%llu bytes (%.2fMB), device-local=%.2f/%.2fMB",
                    buffer_name, static_cast<uint32_t>(create_result.result), static_cast<unsigned long long>(required_size),
                    static_cast<double>(required_size) / (1024.0 * 1024.0), static_cast<unsigned long long>(preferred_size),
                    static_cast<double>(preferred_size) / (1024.0 * 1024.0), static_cast<double>(memory_stats.gpu_device_local_allocated_bytes) / (1024.0 * 1024.0),
                    static_cast<double>(memory_stats.gpu_device_local_budget_bytes) / (1024.0 * 1024.0));
                  return false;
                }
                buffer.handle = create_result.handle;
                buffer.size = desc.size;
                buffer.descriptor_index = get_bindless_descriptor_index(buffer.handle);
                state = RHIResourceState::Undefined;
                return true;
              };
              const uint64_t scratch_preferred_bytes = std::min(kUPBPBeamGridScratchBudget, scratch_bytes + scratch_bytes / 8u);
              if (ensure_grid_capacity(scratch_bytes, scratch_preferred_bytes, RHIBufferUsage::Storage, _upbp.density_beam_grid_scratch_buffer,
                    _upbp.density_beam_grid_scratch_state, "upbp_beam_grid_scratch") == false) {
                return fail("failed to allocate scratch storage");
              }
              const bool offsets_reallocated = (output.cell_offsets_buffer.handle.valid() == false) || (output.cell_offsets_buffer.size < offset_bytes);
              if (ensure_storage_buffer_capacity(device, offset_bytes, RHIBufferUsage::Storage, output.cell_offsets_buffer.handle, output.cell_offsets_buffer.size,
                    output.cell_offsets_buffer.descriptor_index, offsets_name.c_str()) == false) {
                return fail("failed to allocate cell offsets");
              }
              if (offsets_reallocated) {
                output.cell_offsets_state = RHIResourceState::Undefined;
              }
              build_resources.cell_offsets_buffer = output.cell_offsets_buffer.descriptor_index;
              build_resources.reserved0 = _upbp.density_beam_grid_scratch_buffer.descriptor_index;
              build_resources.reserved2 = shard_count;
              if (bind_build_resources() == false) {
                return fail("failed to bind count resources");
              }
              record_and_submit([&](RHICommandBuffer cmd) {
                ctx.cmd_buffer_barrier(cmd, _upbp.density_beam_grid_scratch_buffer.handle, _upbp.density_beam_grid_scratch_state, RHIResourceState::General);
                ctx.cmd_buffer_barrier(cmd, output.cell_offsets_buffer.handle, output.cell_offsets_state, RHIResourceState::General);
                _upbp.density_beam_grid_scratch_state = RHIResourceState::General;
                output.cell_offsets_state = RHIResourceState::General;
                dispatch_stage_range(cmd, PipelineStage::UPBPBeamGridBuild, 0u, shard_count, grid_type, GPUUPBPBeamGridBuildMode::Count);
                ctx.cmd_compute_barrier(cmd);
                dispatch_stage_range(cmd, PipelineStage::UPBPBeamGridBuild, 0u, metadata.cell_count, grid_type, GPUUPBPBeamGridBuildMode::CellTotals);
                ctx.cmd_compute_barrier(cmd);
                dispatch_stage_range(cmd, PipelineStage::UPBPBeamGridBuild, 0u, 1u, grid_type, GPUUPBPBeamGridBuildMode::Prefix);
                ctx.cmd_compute_barrier(cmd);
                dispatch_stage_range(cmd, PipelineStage::UPBPBeamGridBuild, 0u, metadata.cell_count, grid_type, GPUUPBPBeamGridBuildMode::ShardOffsets);
                ctx.cmd_compute_barrier(cmd);
                ctx.cmd_buffer_barrier(cmd, output.metadata_buffer.handle, RHIResourceState::General, RHIResourceState::TransferSrc);
                ctx.cmd_buffer_barrier(cmd, _upbp.density_beam_grid_metadata_readback_buffer.handle, _upbp.density_beam_grid_metadata_readback_state,
                  RHIResourceState::TransferDst);
                ctx.cmd_copy_buffer(cmd, output.metadata_buffer.handle, _upbp.density_beam_grid_metadata_readback_buffer.handle, kGPUUPBPBeamGridMetadataStride);
                ctx.cmd_buffer_barrier(cmd, output.metadata_buffer.handle, RHIResourceState::TransferSrc, RHIResourceState::General);
                _upbp.density_beam_grid_metadata_readback_state = RHIResourceState::TransferDst;
              });
              build_result = wait_and_destroy_submitted_commands("UPBP beam-grid count and prefix");
              if (build_result != RHIResult::Success) {
                return fail("count/prefix dispatch failed (" + std::to_string(static_cast<uint32_t>(build_result)) + ")");
              }
              const RHIResult count_read_result = device.read_buffer(_upbp.density_beam_grid_metadata_readback_buffer.handle, &metadata, sizeof(metadata));
              if (count_read_result != RHIResult::Success) {
                return fail("failed to read count/prefix results (" + std::to_string(static_cast<uint32_t>(count_read_result)) + ")");
              }
              if ((metadata.reserved0 != GPUUPBPBeamGridBuildFailure::None) || (metadata.entry_count == 0u)) {
                return fail("count/prefix failed: " + std::string(upbp_beam_grid_failure_to_string(metadata.reserved0)) + " (status " + std::to_string(metadata.reserved0) +
                            ", entries " + std::to_string(metadata.entry_count) + ")");
              }
              const uint64_t index_bytes = static_cast<uint64_t>(metadata.entry_count) * sizeof(uint32_t);
              if (index_bytes > kWavefrontMaxAddressableBufferSize) {
                return fail("indices exceed the shader-addressable range");
              }
              const uint64_t index_preferred_bytes = std::min(kWavefrontMaxAddressableBufferSize, index_bytes + index_bytes / 8u);
              if (ensure_grid_capacity(index_bytes, index_preferred_bytes, grid_usage, output.beam_indices_buffer, output.beam_indices_state, indices_name.c_str()) == false) {
                return fail("failed to allocate beam indices");
              }
              build_resources.beam_indices_buffer = output.beam_indices_buffer.descriptor_index;
              build_resources.beam_index_count = metadata.entry_count;
              if (bind_build_resources() == false) {
                return fail("failed to bind scatter resources");
              }
              record_and_submit([&](RHICommandBuffer cmd) {
                ctx.cmd_buffer_barrier(cmd, output.beam_indices_buffer.handle, output.beam_indices_state, RHIResourceState::General);
                output.beam_indices_state = RHIResourceState::General;
                dispatch_stage_range(cmd, PipelineStage::UPBPBeamGridBuild, 0u, shard_count, grid_type, GPUUPBPBeamGridBuildMode::Scatter);
                ctx.cmd_compute_barrier(cmd);
                dispatch_stage_range(cmd, PipelineStage::UPBPBeamGridBuild, 0u, metadata.cell_count, grid_type, GPUUPBPBeamGridBuildMode::Validate);
                ctx.cmd_compute_barrier(cmd);
                ctx.cmd_buffer_barrier(cmd, output.metadata_buffer.handle, RHIResourceState::General, RHIResourceState::TransferSrc);
                ctx.cmd_buffer_barrier(cmd, _upbp.density_beam_grid_metadata_readback_buffer.handle, _upbp.density_beam_grid_metadata_readback_state,
                  RHIResourceState::TransferDst);
                ctx.cmd_copy_buffer(cmd, output.metadata_buffer.handle, _upbp.density_beam_grid_metadata_readback_buffer.handle, kGPUUPBPBeamGridMetadataStride);
                ctx.cmd_buffer_barrier(cmd, output.metadata_buffer.handle, RHIResourceState::TransferSrc, RHIResourceState::General);
                _upbp.density_beam_grid_metadata_readback_state = RHIResourceState::TransferDst;
              });
              build_result = wait_and_destroy_submitted_commands("UPBP beam-grid scatter");
              if (build_result != RHIResult::Success) {
                return fail("scatter/validation dispatch failed (" + std::to_string(static_cast<uint32_t>(build_result)) + ")");
              }
              const RHIResult scatter_read_result = device.read_buffer(_upbp.density_beam_grid_metadata_readback_buffer.handle, &metadata, sizeof(metadata));
              if (scatter_read_result != RHIResult::Success) {
                return fail("failed to read scatter/validation results (" + std::to_string(static_cast<uint32_t>(scatter_read_result)) + ")");
              }
              if (metadata.reserved0 != GPUUPBPBeamGridBuildFailure::None) {
                return fail(
                  "scatter/validation failed: " + std::string(upbp_beam_grid_failure_to_string(metadata.reserved0)) + " (status " + std::to_string(metadata.reserved0) + ")");
              }
              output.beam_index_count = metadata.entry_count;
              log::info("GPU UPBP %s: %u beams, %u cells, %u CSR entries", name, beam_count, metadata.cell_count, metadata.entry_count);
              return true;
            };
            if ((build_gpu_beam_grid(GPUUPBPBeamGridType::BP2D, static_cast<uint32_t>(bp2d_beam_instance_count), _upbp.density_beam_reference_buffer, _upbp.density_bp2d_beam_grid,
                   "BP2D beam grid") == false) ||
                (build_gpu_beam_grid(GPUUPBPBeamGridType::BB1D, static_cast<uint32_t>(bb1d_beam_instance_count), _upbp.density_bb1d_beam_buffer, _upbp.density_bb1d_beam_grid,
                   "BB1D beam grid") == false)) {
              set_runtime_failure(beam_grid_failure_reason.empty() ? "GPU UPBP failed to build its compute beam grid on the GPU" : std::move(beam_grid_failure_reason));
              return;
            }
          }
          _upbp.density_cache_ready = true;
          bind_upbp_density_cache_resources();
          if (update_upbp_iteration_resources(device, scene, upbp_global_path_count) == false) {
            set_runtime_failure("GPU UPBP failed to bind its compact density cache");
            return;
          }
        }
        if (_upbp.density_cache_ready && evaluating_upbp_camera_phase) {
          record_and_submit([&](RHICommandBuffer cmd) {
            barrier_wavefront_buffers(cmd);
            dispatch_stage_mode(cmd, PipelineStage::UPBPClear, scalar_dispatch, 0u, GPUUPBPClearMode::CameraQueries);
            barrier_wavefront_buffers(cmd);
          });
          if (camera_vertex_count > 0u) {
            submit_upbp_density_stage_chunked(PipelineStage::UPBPDensityCompact, camera_vertex_count, GPUUPBPDensityCompactMode::CameraVertices);
          }
          if (camera_interval_count > 0u) {
            submit_upbp_density_stage_chunked(PipelineStage::UPBPDensityCompact, camera_interval_count, GPUUPBPDensityCompactMode::CameraIntervals);
          }
          record_and_submit([&](RHICommandBuffer cmd) {
            barrier_wavefront_buffers(cmd);
            ctx.cmd_buffer_barrier(cmd, _upbp.counter_buffer.handle, RHIResourceState::General, RHIResourceState::TransferSrc);
            ctx.cmd_buffer_barrier(cmd, _upbp.counter_readback_buffer.handle, _upbp.counter_readback_state, RHIResourceState::TransferDst);
            ctx.cmd_copy_buffer(cmd, _upbp.counter_buffer.handle, _upbp.counter_readback_buffer.handle, static_cast<uint64_t>(GPUUPBPCounterIndex::Count) * sizeof(uint32_t));
            ctx.cmd_buffer_barrier(cmd, _upbp.counter_buffer.handle, RHIResourceState::TransferSrc, RHIResourceState::General);
            _upbp.counter_readback_state = RHIResourceState::TransferDst;
          });
          const RHIResult compact_result = wait_and_destroy_submitted_commands("UPBP camera density query compaction");
          if (compact_result != RHIResult::Success) {
            set_runtime_failure("GPU UPBP camera density query compaction failed (" + std::to_string(static_cast<uint32_t>(compact_result)) + ")");
            return;
          }
          const RHIResult compact_counter_result = device.read_buffer(_upbp.counter_readback_buffer.handle, upbp_counters, sizeof(upbp_counters));
          if (compact_counter_result != RHIResult::Success) {
            set_runtime_failure("GPU UPBP failed to read compact camera density query counts (" + std::to_string(static_cast<uint32_t>(compact_counter_result)) + ")");
            return;
          }
          camera_surface_queries_compacted = true;
          for (uint32_t family = 0u; family < GPUUPBPSurfaceQueryFamily::Count; ++family) {
            const uint32_t query_count = upbp_counters[GPUUPBPCounterIndex::CameraSurfaceVariousQuery + family];
            camera_surface_queries_compacted = camera_surface_queries_compacted && (query_count <= _upbp.resources.point_capacity);
            camera_surface_query_counts[family] = query_count;
          }
          if (camera_surface_queries_compacted == false) {
            for (uint32_t& query_count : camera_surface_query_counts) {
              query_count = camera_vertex_count;
            }
          }
          camera_medium_vertex_queries_compacted = upbp_counters[GPUUPBPCounterIndex::CameraMediumVertexQuery] <= _upbp.resources.beam_capacity;
          if (camera_medium_vertex_queries_compacted) {
            camera_medium_vertex_query_count = upbp_counters[GPUUPBPCounterIndex::CameraMediumVertexQuery];
          }
          const uint64_t interval_query_capacity = static_cast<uint64_t>(_upbp.resources.beam_capacity) * ((kGPUUPBPBeamStride / sizeof(uint32_t)) - 1u);
          camera_interval_queries_compacted = upbp_counters[GPUUPBPCounterIndex::CameraMediumIntervalQuery] <= interval_query_capacity;
          if (camera_interval_queries_compacted) {
            camera_medium_interval_query_count = upbp_counters[GPUUPBPCounterIndex::CameraMediumIntervalQuery];
          }
          if (capture_kernel_timings) {
            log::info("GPU UPBP compact camera density queries: surface=%u/%u/%u/%u of %u medium-vertices=%u/%u medium-intervals=%u/%u",
              camera_surface_query_counts[GPUUPBPSurfaceQueryFamily::Various], camera_surface_query_counts[GPUUPBPSurfaceQueryFamily::Plastic],
              camera_surface_query_counts[GPUUPBPSurfaceQueryFamily::Conductor], camera_surface_query_counts[GPUUPBPSurfaceQueryFamily::Dielectric], camera_vertex_count,
              camera_medium_vertex_query_count, camera_vertex_count, camera_medium_interval_query_count, camera_interval_count);
          }
        }
        if (_upbp.density_cache_ready && (_upbp.density_surface_point_count > 0u) && _upbp.density_surface_point_tlas.valid() &&
            gpu_integrator_feature_enabled(_integrator_features, GPUIntegratorFeatures::UPBPSurface)) {
          const uint32_t query_mode = camera_surface_queries_compacted ? GPUUPBPDensityQueryMode::Compacted : GPUUPBPDensityQueryMode::Raw;
          const auto submit_surface_queries = [&](PipelineStage stage, uint32_t query_count, bool enabled) {
            if ((enabled == false) || (query_count == 0u)) {
              return;
            }
            for (uint32_t item_offset = 0u; item_offset < query_count; item_offset += kUPBPDensityQueryDispatchChunkSize) {
              const uint32_t chunk_count = std::min(kUPBPDensityQueryDispatchChunkSize, query_count - item_offset);
              record_and_submit([&](RHICommandBuffer cmd) {
                barrier_wavefront_buffers(cmd);
                dispatch_stage_query_groups(cmd, stage, item_offset, chunk_count, constants.path_iteration, query_mode);
                barrier_wavefront_buffers(cmd);
              });
            }
          };
          submit_surface_queries(PipelineStage::VCMMergeDiffuse, camera_surface_query_counts[GPUUPBPSurfaceQueryFamily::Various], has_various_continue);
          submit_surface_queries(PipelineStage::VCMMergePlastic, camera_surface_query_counts[GPUUPBPSurfaceQueryFamily::Plastic], has_plastic);
          submit_surface_queries(PipelineStage::VCMMergeConductor, camera_surface_query_counts[GPUUPBPSurfaceQueryFamily::Conductor], has_connectible_conductor);
          submit_surface_queries(PipelineStage::VCMMergeDielectric, camera_surface_query_counts[GPUUPBPSurfaceQueryFamily::Dielectric], has_connectible_dielectric || has_thinfilm);
        }
        if (_upbp.density_cache_ready && (camera_vertex_count > 0u)) {
          submit_upbp_density_stage_chunked(PipelineStage::UPBPDirectHit, camera_vertex_count, GPUUPBPDensityQueryMode::Raw);
        }
        if (_upbp.density_cache_ready && (_upbp.density_medium_point_count > 0u) && _upbp.density_medium_point_tlas.valid() && (camera_vertex_count > 0u) &&
            gpu_integrator_feature_enabled(_integrator_features, GPUIntegratorFeatures::UPBPPP3D)) {
          submit_upbp_density_linear_queries(PipelineStage::UPBPPP3D, camera_vertex_count, GPUUPBPDensityQueryMode::Raw);
        }
        if (_upbp.density_cache_ready && (_upbp.density_medium_point_count > 0u) && _upbp.density_medium_point_tlas.valid() && (camera_medium_interval_query_count > 0u) &&
            gpu_integrator_feature_enabled(_integrator_features, GPUIntegratorFeatures::UPBPPB2D)) {
          submit_upbp_density_linear_queries(PipelineStage::UPBPPB2D, camera_medium_interval_query_count,
            camera_interval_queries_compacted ? GPUUPBPDensityQueryMode::Compacted : GPUUPBPDensityQueryMode::Raw);
        }

        const bool bp2d_index_ready = _use_compute_upbp_beam_grid ? _upbp.density_bp2d_beam_grid.metadata_buffer.handle.valid() : (_upbp.density_bp2d_beam_tlas.empty() == false);
        if (_upbp.density_cache_ready && (_upbp.density_beam_count > 0u) && bp2d_index_ready && (camera_medium_vertex_query_count > 0u) && bp2d_enabled) {
          const uint32_t query_mode = camera_medium_vertex_queries_compacted ? GPUUPBPDensityQueryMode::Compacted : GPUUPBPDensityQueryMode::Raw;
          if (_use_compute_upbp_beam_grid) {
            submit_upbp_density_query_groups(PipelineStage::UPBPBP2D, camera_medium_vertex_query_count, query_mode);
          } else {
            const RHIResult preceding_density_result = wait_and_destroy_submitted_commands("UPBP density stages before BP2D");
            if (preceding_density_result != RHIResult::Success) {
              set_runtime_failure("GPU UPBP density evaluation before BP2D failed (" + std::to_string(static_cast<uint32_t>(preceding_density_result)) + ")");
              return;
            }
            const uint32_t partition_count = static_cast<uint32_t>(_upbp.density_bp2d_beam_tlas.size());
            for (uint32_t partition_offset = 0u; partition_offset < partition_count; partition_offset += kGPUUPBPBP2DPartitionCount) {
              for (uint32_t local_partition_index = 0u; local_partition_index < kGPUUPBPBP2DPartitionCount; ++local_partition_index) {
                const uint32_t partition_index = partition_offset + local_partition_index;
                const RHIBindlessHandle partition_tlas = partition_index < partition_count ? _upbp.density_bp2d_beam_tlas[partition_index] : RHIBindlessHandle{};
                _upbp.resources.bp2d_beam_acceleration_structures[local_partition_index] =
                  partition_tlas.valid() ? get_bindless_descriptor_index(partition_tlas) : kInvalidDescriptorIndex;
              }
              if (device.update_buffer(_upbp.resources_buffer.handle, &_upbp.resources, kGPUUPBPResourcesStride) != RHIResult::Success) {
                set_runtime_failure("GPU UPBP failed to bind a BP2D acceleration-structure window");
                return;
              }
              submit_upbp_density_query_groups(PipelineStage::UPBPBP2D, camera_medium_vertex_query_count, query_mode);
              const RHIResult window_result = wait_and_destroy_submitted_commands("UPBP BP2D acceleration-structure window");
              if (window_result != RHIResult::Success) {
                set_runtime_failure("GPU UPBP BP2D density evaluation failed (" + std::to_string(static_cast<uint32_t>(window_result)) + ")");
                return;
              }
            }
          }
        }
        const bool bb1d_index_ready = _use_compute_upbp_beam_grid ? _upbp.density_bb1d_beam_grid.metadata_buffer.handle.valid() : _upbp.density_bb1d_beam_tlas[0].valid();
        if (_upbp.density_cache_ready && (_upbp.density_bb1d_beam_count > 0u) && bb1d_index_ready && (camera_medium_interval_query_count > 0u) && bb1d_enabled) {
          submit_upbp_density_query_groups(PipelineStage::UPBPBB1D, camera_medium_interval_query_count,
            camera_interval_queries_compacted ? GPUUPBPDensityQueryMode::Compacted : GPUUPBPDensityQueryMode::Raw);
        }

        record_and_submit([&](RHICommandBuffer cmd) {
          barrier_wavefront_buffers(cmd);
          dispatch_stage(cmd, PipelineStage::UPBPValidate, scalar_dispatch, 0u);
          barrier_wavefront_buffers(cmd);
          ctx.cmd_buffer_barrier(cmd, _upbp.counter_buffer.handle, RHIResourceState::General, RHIResourceState::TransferSrc);
          ctx.cmd_buffer_barrier(cmd, _upbp.counter_readback_buffer.handle, _upbp.counter_readback_state, RHIResourceState::TransferDst);
          ctx.cmd_copy_buffer(cmd, _upbp.counter_buffer.handle, _upbp.counter_readback_buffer.handle, static_cast<uint64_t>(GPUUPBPCounterIndex::Count) * sizeof(uint32_t));
          ctx.cmd_buffer_barrier(cmd, _upbp.counter_buffer.handle, RHIResourceState::TransferSrc, RHIResourceState::General);
          _upbp.counter_readback_state = RHIResourceState::TransferDst;
        });
        const RHIResult grid_result = wait_and_destroy_submitted_commands("UPBP density evaluation");
        if (grid_result != RHIResult::Success) {
          set_runtime_failure("GPU UPBP density evaluation failed (" + std::to_string(static_cast<uint32_t>(grid_result)) + ")");
          return;
        }
        const RHIResult grid_counter_result = device.read_buffer(_upbp.counter_readback_buffer.handle, upbp_counters, sizeof(upbp_counters));
        if (grid_counter_result != RHIResult::Success) {
          set_runtime_failure("GPU UPBP failed to validate its streamed beam grid (" + std::to_string(static_cast<uint32_t>(grid_counter_result)) + ")");
          return;
        }
        if ((upbp_counters[GPUUPBPCounterIndex::OverflowFlags] & GPUUPBPOverflowFlags::BeamInstance) != 0u) {
          set_runtime_failure("GPU UPBP compact beam instance generation exceeded its exact measured capacity");
          return;
        }
        if (upbp_counters[GPUUPBPCounterIndex::FirstFailureCode] == GPUUPBPPathFailure::NonFiniteDensityContribution) {
          set_runtime_failure("GPU UPBP produced a non-finite density contribution for technique " + std::to_string(upbp_counters[GPUUPBPCounterIndex::FirstFailureDetail0]) +
                              " at global path " + std::to_string(upbp_counters[GPUUPBPCounterIndex::FirstFailureGlobalPath]) + ", spectral value bits " +
                              std::to_string(upbp_counters[GPUUPBPCounterIndex::FirstFailureDetail1]) + ", finite mask " +
                              std::to_string(upbp_counters[GPUUPBPCounterIndex::FirstFailureDetail2]) + ", light batch offset " +
                              std::to_string(upbp_counters[GPUUPBPCounterIndex::FirstFailureDetail3]));
          return;
        }
        if ((evaluating_upbp_camera_phase == false) && ((_upbp.light_batch_iteration + 1u) < _upbp.light_batch_count_total)) {
          _upbp.light_batch_iteration += 1u;
          const UPBPLightBatch next_light_batch = upbp_light_batch_for_iteration(_upbp.light_batch_iteration, _upbp.global_path_count, _upbp.resident_light_path_capacity,
            _upbp.camera_batch_offset, _upbp.camera_batch_count);
          _upbp.light_batch_index = _upbp.light_batch_iteration;
          _upbp.light_batch_offset = next_light_batch.offset;
          _upbp.light_batch_count = next_light_batch.count;
          if (update_upbp_iteration_resources(device, scene, upbp_global_path_count) == false) {
            set_runtime_failure("GPU UPBP failed to advance its deterministic light-path batch");
            return;
          }
          _wavefront_path_iteration = 0u;
          _wavefront_camera_queue_count = 0u;
          _wavefront_light_queue_count = _upbp.light_batch_count;
          record_and_submit([&](RHICommandBuffer cmd) {
            barrier_wavefront_buffers(cmd);
            dispatch_stage_mode(cmd, PipelineStage::UPBPClear, scalar_dispatch, 0u, GPUUPBPClearMode::LightBatch);
            barrier_wavefront_buffers(cmd);
            dispatch_stage_mode(cmd, PipelineStage::PrepareSample, scalar_dispatch, 0u, 1u);
            barrier_wavefront_buffers(cmd);
            dispatch_upbp_light_init(cmd);
            barrier_wavefront_buffers(cmd);
          });
          const RHIResult init_result = wait_and_destroy_submitted_commands("UPBP light batch init");
          if (init_result != RHIResult::Success) {
            set_runtime_failure("GPU UPBP light batch initialization failed (" + std::to_string(static_cast<uint32_t>(init_result)) + ")");
            return;
          }
          _wavefront_render_step = WavefrontRenderStep::TraceBounce;
        } else if (evaluating_upbp_camera_phase == false) {
          _upbp.camera_phase_started = true;
          // Keep the resident layout stable across frames. An edge tile can contain fewer active paths than the
          // tile-plan capacity, but shrinking storage to that active count invalidates an in-progress camera phase
          // when the next render call restores the planned capacity.
          uint32_t camera_resident_capacity = wavefront_buffer_path_capacity;
          const uint64_t released_wavefront_bytes = destroy_upbp_completed_light_wavefront_buffers(device);
          if (released_wavefront_bytes > 0u) {
            const RHIResult reclaim_result = ctx.wait_idle();
            if (reclaim_result != RHIResult::Success) {
              set_runtime_failure("GPU UPBP failed to reclaim completed light-phase storage (" + std::to_string(static_cast<uint32_t>(reclaim_result)) + ")");
              return;
            }
          }
          if (_upbp.vertex_buffer.handle.valid() == false) {
            UPBPOptions options = {};
            const auto settings = scene.integrator_data().settings.find(Integrator::Type::UPBP);
            if (settings != scene.integrator_data().settings.end()) {
              options.load(settings->second);
            }
            const bool merge_vertices_enabled = (scene.data().options.strategy_flags & Scene::Strategy::MergeVertices) != 0u;
            const uint32_t storage_technique_mask = upbp_effective_technique_mask(options, merge_vertices_enabled);
            bool scene_has_subsurface_material = false;
            for (const auto& material : scene.data().materials) {
              if (material.subsurface_cls != SubsurfaceMaterial::Disabled) {
                scene_has_subsurface_material = true;
                break;
              }
            }
            const RHIMemoryStats post_cache_memory_stats = device.get_memory_statistics();
            const uint64_t available_camera_bytes = gpu_resident_working_set_budget_bytes(post_cache_memory_stats);
            const bool connect_to_light = gpu_integrator_feature_enabled(_integrator_features, GPUIntegratorFeatures::ConnectToLight);
            const bool connect_to_camera = gpu_integrator_feature_enabled(_integrator_features, GPUIntegratorFeatures::ConnectToCamera);
            camera_resident_capacity = upbp_camera_resident_path_capacity(available_camera_bytes, wavefront_path_capacity, scene_max_path_length, options.maximum_boundary_count,
              storage_technique_mask, connect_to_light, connect_to_camera, scene_has_subsurface_material);
            if (camera_resident_capacity == 0u) {
              set_runtime_failure("GPU UPBP density cache leaves insufficient device-local memory for one camera path");
              return;
            }

            if (camera_resident_capacity < wavefront_path_capacity) {
              ETX_ASSERT(_wavefront_tile_index == 0u);
              _wavefront_tile_max_pixels = camera_resident_capacity;
              _wavefront_tile_path_capacity = wavefront_tile_path_capacity(active_base_dim, _wavefront_tile_max_pixels);
              _wavefront_tile_count = wavefront_tile_count(active_base_dim, _wavefront_tile_max_pixels);
              _upbp_light_path_capacity = _wavefront_tile_path_capacity;
              wavefront_buffer_path_capacity = _wavefront_tile_path_capacity;
              tile_max_pixels = _wavefront_tile_max_pixels;
              wavefront_tile_count_value = _wavefront_tile_count;
              active_window = wavefront_tile_window(active_base_origin, active_base_dim, tile_max_pixels, _wavefront_tile_index);
              active_path_offset = wavefront_tile_path_offset(active_base_origin, active_base_dim, tile_max_pixels, _wavefront_tile_index);
              render_dim = active_window.size;
              wavefront_path_capacity = render_dim.x * render_dim.y;
              camera_resident_capacity = wavefront_path_capacity;
              initial_camera_queue_count = wavefront_path_capacity;
              constants.render_window_origin_x = active_window.origin.x;
              constants.render_window_origin_y = active_window.origin.y;
              constants.render_window_width = render_dim.x;
              constants.render_window_height = render_dim.y;
              film_dispatch = {
                .group_count_x = (render_dim.x + 7u) / 8u,
                .group_count_y = (render_dim.y + 7u) / 8u,
                .group_count_z = 1u,
              };
              log::info(
                "GPU UPBP resized camera residency to %u paths after density indexing; released %.2f MiB of completed light-wavefront storage; post-cache available %.2f MiB, "
                "%u camera tiles",
                camera_resident_capacity, static_cast<double>(released_wavefront_bytes) / (1024.0 * 1024.0), static_cast<double>(available_camera_bytes) / (1024.0 * 1024.0),
                wavefront_tile_count_value);
            }
          }
          if (ensure_wavefront_buffers(ctx, scene, camera_resident_capacity, wavefront_path_capacity, false) == false) {
            set_runtime_failure("GPU UPBP failed to allocate camera-wavefront storage after density-cache construction");
            return;
          }
          if (ensure_upbp_buffers(ctx, scene, upbp_global_path_count, camera_resident_capacity, camera_resident_capacity, _wavefront_tile_index, active_path_offset,
                wavefront_path_capacity) == false) {
            set_runtime_failure("GPU UPBP failed to restore resident camera-path storage after density-cache construction");
            return;
          }
          _upbp.camera_phase_started = true;
          _wavefront_resources.upbp_resources_buffer = _upbp.resources_buffer.descriptor_index;
          if (device.update_buffer(_wavefront_resources_buffer, &_wavefront_resources, sizeof(_wavefront_resources)) != RHIResult::Success) {
            set_runtime_failure("GPU UPBP failed to bind the camera-phase resource layout");
            return;
          }
          if (update_upbp_iteration_resources(device, scene, upbp_global_path_count) == false) {
            set_runtime_failure("GPU UPBP failed to enable camera-independent terms for camera evaluation");
            return;
          }
          if (initialize_deferred_camera_phase() == false) {
            return;
          }
          _wavefront_render_step = WavefrontRenderStep::TraceBounce;
        } else {
          _wavefront_render_step = WavefrontRenderStep::FinalizeSample;
        }
        dispatch_submit_ms = elapsed_ms(dispatch_submit_begin, std::chrono::steady_clock::now());
      } else if (_wavefront_render_step == WavefrontRenderStep::FinalizeSample) {
        if (finalize_wavefront_sample() == false) {
          return;
        }
      }

      if (_wavefront_render_step == WavefrontRenderStep::InitSample) {
        finished_current_tile = true;
      }
    }
    const double wavefront_batch_ms = elapsed_ms(wavefront_batch_begin, std::chrono::steady_clock::now());
    const bool wavefront_budget_consumed = (executed_wavefront_steps >= wavefront_step_budget) && (finished_current_tile == false);
    update_wavefront_auto_tuning(executed_wavefront_steps, wavefront_batch_ms, wavefront_budget_consumed, wavefront_auto_measurement_valid);
    if ((render_output_texture_state == RHIResourceState::General) && frame_data.cmd.valid()) {
      ctx.cmd_texture_barrier(frame_data.cmd, render_output_texture, RHIResourceState::General, RHIResourceState::ShaderReadOnly);
      render_output_texture_state = RHIResourceState::ShaderReadOnly;
    }
  }

  _frame_index += 1u;
  if (completed_sample) {
    _display_output_valid = true;
    _sample_index += 1u;
    const bool compact_light_history = gpu_integrator_feature_enabled(_integrator_features, GPUIntegratorFeatures::LightPath) &&
                                       (gpu_integrator_feature_enabled(_integrator_features, GPUIntegratorFeatures::ConnectVertices) ||
                                         gpu_integrator_feature_enabled(_integrator_features, GPUIntegratorFeatures::MergeVertices));
    const bool rendering_will_stop = (_run_state == RunState::Finishing) || (_sample_index >= _last_target_samples);
    const uint32_t initial_light_history_bounces = wavefront_initial_light_history_bounces(scene_max_path_length_for_tiling);
    const uint64_t initial_light_vertex_capacity = static_cast<uint64_t>(wavefront_buffer_path_capacity) * static_cast<uint64_t>(initial_light_history_bounces + 1u);
    const uint64_t current_light_vertex_capacity = _wavefront_resources.light_vertex_capacity;
    uint32_t target_light_history_bounces = _wavefront_light_history_capacity_bounces;
    bool shrink_light_history = false;

    if (compact_light_history && (current_light_vertex_capacity > initial_light_vertex_capacity)) {
      if (rendering_will_stop) {
        target_light_history_bounces = initial_light_history_bounces;
        shrink_light_history = true;
      } else {
        const bool light_history_underused = (static_cast<uint64_t>(_wavefront_light_vertex_sample_peak_count) * 2ull) <= current_light_vertex_capacity;
        if (light_history_underused) {
          _wavefront_light_history_underuse_sample_count += 1u;
          _wavefront_light_history_underuse_peak_count = std::max(_wavefront_light_history_underuse_peak_count, _wavefront_light_vertex_sample_peak_count);
          if (_wavefront_light_history_underuse_sample_count >= kWavefrontLightHistoryShrinkSampleCount) {
            const uint64_t retained_light_vertex_capacity = std::max(initial_light_vertex_capacity, static_cast<uint64_t>(_wavefront_light_history_underuse_peak_count) * 2ull);
            const uint64_t retained_light_history_layers =
              (retained_light_vertex_capacity + static_cast<uint64_t>(wavefront_buffer_path_capacity) - 1ull) / static_cast<uint64_t>(wavefront_buffer_path_capacity);
            const uint32_t retained_light_history_bounces = static_cast<uint32_t>(retained_light_history_layers - 1ull);
            target_light_history_bounces = std::min(scene_max_path_length_for_tiling, std::max(initial_light_history_bounces, retained_light_history_bounces));
            shrink_light_history = target_light_history_bounces < _wavefront_light_history_capacity_bounces;
            _wavefront_light_history_underuse_sample_count = 0u;
            _wavefront_light_history_underuse_peak_count = 0u;
          }
        } else {
          _wavefront_light_history_underuse_sample_count = 0u;
          _wavefront_light_history_underuse_peak_count = 0u;
        }
      }
    } else {
      _wavefront_light_history_underuse_sample_count = 0u;
      _wavefront_light_history_underuse_peak_count = 0u;
    }

    if (shrink_light_history) {
      const uint32_t previous_light_history_bounces = _wavefront_light_history_capacity_bounces;
      _wavefront_light_history_capacity_bounces = target_light_history_bounces;
      if (ensure_wavefront_buffers(ctx, scene, wavefront_buffer_path_capacity, wavefront_path_capacity, true) == false) {
        _wavefront_light_history_capacity_bounces = previous_light_history_bounces;
        log::warning("GPU RT: failed to shrink compact light history buffer");
      } else {
        log::info("GPU RT: reduced compact light history capacity to %u vertices (%.2fMB allocation retained)", _wavefront_resources.light_vertex_capacity,
          static_cast<double>(_light_vertex_buffer_size) / (1024.0 * 1024.0));
      }
    }
    if (_run_state == RunState::Finishing) {
      preserve_render_statistics();
      _run_state = RunState::Stopped;
    } else if (_sample_index >= _last_target_samples) {
      preserve_render_statistics();
      _run_state = RunState::Completed;
    }
  }
}

void GPURaytracingRenderer::cleanup(RHIContext& ctx) {
  ETX_PROFILER_SCOPE();

  _cleanup_wait_succeeded = false;

  auto& device = ctx.device();
  cancel_preparation();
  finish_pipeline_publish_batch(device, true);
  release_inflight_preparation_tasks(true);

  const RHIResult wait_result = ctx.wait_idle();
  if (wait_result != RHIResult::Success) {
    log::warning("GPU RT: wait_idle failed during cleanup (%u)", static_cast<uint32_t>(wait_result));
  } else {
    _cleanup_wait_succeeded = true;
  }

  destroy_pipelines(device);

  destroy_wavefront_buffers(ctx);
  destroy_scene_buffers(ctx);
  destroy_linear_scene_buffer(device, _spectral_values_buffer, _spectral_values_buffer_size, _spectral_values_buffer_descriptor_index);
  destroy_blue_noise_buffer(ctx);
  destroy_acceleration_structures(ctx);
  destroy_linear_scene_buffer(device, _camera_buffer, _camera_buffer_size, _camera_buffer_descriptor_index);

  if (_output_texture.valid()) {
    device.destroy_texture(_output_texture);
    _output_texture = {};
  }
  _output_texture_state = RHIResourceState::Undefined;

  const RHIResult release_result = ctx.wait_idle();
  if (release_result != RHIResult::Success) {
    log::warning("GPU RT: wait_idle failed while releasing cleanup resources (%u)", static_cast<uint32_t>(release_result));
    _cleanup_wait_succeeded = false;
  }

  _initialized = false;
  _scene_valid = false;
  _run_state = RunState::Stopped;
  _current_scene_hashes = {};
  _current_integrator_data_revision = 0u;
  _current_camera_hash = 0;
  _integrator_data_revision_initialized = false;
  _frame_index = 0u;
  _sample_index = 0u;
  reset_render_timing();
  reset_kernel_timings();
  _preserved_render_elapsed_seconds = 0.0;
  _preserved_kernel_timing_stats = {};
  _preserved_timing_stats_valid = false;
  _wavefront_render_step = WavefrontRenderStep::InitSample;
  _wavefront_path_iteration = 0u;
  _wavefront_hard_iteration_cap = 0u;
  _wavefront_camera_queue_count = 0u;
  _wavefront_light_queue_count = 0u;
  _wavefront_light_max_path_length = 0u;
  _wavefront_connect_light_vertex_length = 0u;
  _wavefront_connect_light_history_bounces = 0u;
  _wavefront_vcm_light_vertex_count = 0u;
  _wavefront_tile_index = 0u;
  _wavefront_tile_max_pixels = 0u;
  _wavefront_tile_count = 1u;
  _wavefront_tile_path_capacity = 0u;
  _upbp_light_path_capacity = 0u;
  _wavefront_tile_base_origin = {};
  _wavefront_tile_base_size = {};
  _wavefront_tile_plan_valid = false;
  _wavefront_camera_phase_initialized = false;
  _integrator_mode = 0u;
  _integrator_features = 0u;
  _material_compile_mask = 0u;
  _spectral_mode = 0u;
  _scene_options_upload_pending = false;
  _render_window_origin = {};
  _render_window_size = {};
  _active_preparation.reset();
  _publish_preparation.reset();
  _preparation_generation = 0u;
  _published_pipeline_count = 0u;
  _publish_pipeline_index = 0u;
  _pipeline_publish_logged = false;
  invalidate_output();
  _preparation_canceled = false;
  reset_runtime_failure();
  set_preparation_ready();
  request_scene_update();
}

void GPURaytracingRenderer::on_scene_changed(SceneRepresentation& scene) {
  ETX_PROFILER_SCOPE();
  _scene_valid = scene.valid();
  Renderer::on_scene_changed(scene);
  restart_render_after_change();
}

void GPURaytracingRenderer::on_scene_transforms_changed(SceneRepresentation& scene) {
  ETX_PROFILER_SCOPE();
  _scene_valid = scene.valid();
  Renderer::on_scene_transforms_changed(scene);
}

bool GPURaytracingRenderer::refit_top_level_acceleration_structure(RHIContext& ctx, const SceneData& scene_data) {
  if ((_tlas.valid() == false) || (_tlas_instance_buffer.valid() == false) || (_as_scratch_buffer.valid() == false)) {
    log::warning("GPU RT: TLAS refit unavailable: tlas=%u instance_buffer=%u scratch_buffer=%u", _tlas.valid() ? 1u : 0u, _tlas_instance_buffer.valid() ? 1u : 0u,
      _as_scratch_buffer.valid() ? 1u : 0u);
    return false;
  }
  if (scene_data.hierarchy.mesh_instances.size() != _tlas_instance_count) {
    log::warning("GPU RT: TLAS refit unavailable: instance count changed from %u to %llu", _tlas_instance_count,
      static_cast<unsigned long long>(scene_data.hierarchy.mesh_instances.size()));
    return false;
  }

  _tlas_instance_staging.clear();
  _tlas_instance_staging.reserve(_tlas_instance_count);
  auto& device = ctx.device();
  for (uint32_t instance_index = 0u; instance_index < _tlas_instance_count; ++instance_index) {
    const ResolvedMeshInstance& resolved = scene_data.hierarchy.mesh_instances[instance_index];
    if (resolved.mesh_index >= _blas.size()) {
      log::warning("GPU RT: TLAS refit unavailable: instance %u references mesh %u with %llu BLAS entries", instance_index, resolved.mesh_index,
        static_cast<unsigned long long>(_blas.size()));
      return false;
    }
    RHIAccelerationStructureInstance& instance = _tlas_instance_staging.emplace_back();
    memcpy(instance.transform, resolved.object_to_world.rows, sizeof(instance.transform));
    instance.instance_custom_index = instance_index;
    instance.mask = (resolved.flags & ResolvedMeshInstance::Enabled) != 0u ? 0xffu : 0u;
    instance.instance_shader_binding_table_record_offset = 0u;
    instance.flags = 0u;
    instance.acceleration_structure_reference = device.get_acceleration_structure_device_address(_blas[resolved.mesh_index]);
  }

  const uint64_t upload_size = _tlas_instance_staging.size() * sizeof(RHIAccelerationStructureInstance);
  const RHIResult upload_result = device.update_buffer(_tlas_instance_buffer, _tlas_instance_staging.data(), upload_size);
  if (upload_result != RHIResult::Success) {
    log::error("GPU RT: failed to upload TLAS refit instances (%u)", static_cast<uint32_t>(upload_result));
    return false;
  }

  RHIAccelerationStructureBuildDesc build_desc = {};
  build_desc.as_handle = _tlas;
  build_desc.type = RHIAccelerationStructureType::TopLevel;
  build_desc.instance_count = _tlas_instance_count;
  build_desc.instance_buffer = _tlas_instance_buffer;
  build_desc.allow_update = true;
  build_desc.update = true;

  const RHICommandBuffer command_buffer = ctx.get_command_buffer();
  if (command_buffer.valid() == false) {
    log::warning("GPU RT: TLAS refit unavailable: failed to acquire a command buffer");
    return false;
  }
  ctx.command_buffer_begin(command_buffer);
  ctx.cmd_build_acceleration_structure(command_buffer, build_desc, _as_scratch_buffer, 0u);
  ctx.command_buffer_end(command_buffer);
  ctx.submit_command_buffer({command_buffer});
  const RHIResult wait_result = ctx.wait_for_command_buffer(command_buffer);
  ctx.destroy_command_buffer(command_buffer);
  if (wait_result != RHIResult::Success) {
    log::error("GPU RT: TLAS refit failed (%u)", static_cast<uint32_t>(wait_result));
    return false;
  }
  return true;
}

bool GPURaytracingRenderer::build_acceleration_structures(RHIContext& ctx, SceneRepresentation& scene) {
  ETX_PROFILER_SCOPE();
  const auto total_begin = std::chrono::steady_clock::now();
  double geometry_upload_ms = 0.0;
  double blas_create_ms = 0.0;
  double tlas_instance_upload_ms = 0.0;
  double tlas_create_ms = 0.0;
  double scratch_create_ms = 0.0;
  double build_submit_ms = 0.0;
  double build_wait_ms = 0.0;
  double scene_upload_ms = 0.0;

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
    const auto geometry_upload_begin = std::chrono::steady_clock::now();

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
    const auto geometry_upload_end = std::chrono::steady_clock::now();
    geometry_upload_ms = elapsed_ms(geometry_upload_begin, geometry_upload_end);
  }

  std::vector<RHIAccelerationStructureGeometry> geometries(s.meshes.size());
  std::vector<RHIAccelerationStructureBuildDesc> blas_build_descs;
  blas_build_descs.reserve(s.meshes.size());
  new_blas.reserve(s.meshes.size());

  const auto blas_create_begin = std::chrono::steady_clock::now();
  for (uint32_t mesh_index = 0u; mesh_index < s.meshes.size(); ++mesh_index) {
    const Mesh& mesh = s.meshes[mesh_index];
    if ((mesh.triangle_count == 0u) || ((mesh.triangle_offset + mesh.triangle_count) > s.triangles.size())) {
      log::error("GPU RT: mesh %u has an invalid triangle range", mesh_index);
      cleanup_failed_build();
      return false;
    }

    RHIAccelerationStructureGeometry& geometry = geometries[mesh_index];
    geometry.type = RHIAccelerationStructureGeometryType::Triangles;
    geometry.is_opaque = true;
    geometry.triangles.vertex_buffer = vb_res.handle;
    geometry.triangles.vertex_count = static_cast<uint32_t>(s.vertices.pos.size());
    geometry.triangles.vertex_stride = sizeof(float3);
    geometry.triangles.vertex_format = RHIVertexFormat::Float3;
    geometry.triangles.index_buffer = ib_res.handle;
    geometry.triangles.index_buffer_offset = static_cast<uint64_t>(mesh.triangle_offset) * 3u * sizeof(uint32_t);
    geometry.triangles.index_count = mesh.triangle_count * 3u;
    geometry.triangles.index_type = RHIIndexType::UInt32;

    RHIAccelerationStructureDesc blas_desc = {};
    blas_desc.type = RHIAccelerationStructureType::BottomLevel;
    blas_desc.geometry_count = 1u;
    blas_desc.geometries = &geometry;
    const RHICreateBindlessResult blas_result = device.create_acceleration_structure(blas_desc);
    if ((blas_result.result != RHIResult::Success) || (blas_result.handle.valid() == false)) {
      log::error("GPU RT: failed to create BLAS for mesh %u (%u)", mesh_index, static_cast<uint32_t>(blas_result.result));
      cleanup_failed_build();
      return false;
    }
    new_blas.push_back(blas_result.handle);

    RHIAccelerationStructureBuildDesc& build_desc = blas_build_descs.emplace_back();
    build_desc.as_handle = blas_result.handle;
    build_desc.type = RHIAccelerationStructureType::BottomLevel;
    build_desc.geometry_count = 1u;
    build_desc.geometries = &geometry;
  }
  const auto blas_create_end = std::chrono::steady_clock::now();
  blas_create_ms = elapsed_ms(blas_create_begin, blas_create_end);

  if (s.hierarchy.mesh_instances.empty()) {
    log::error("GPU RT: scene hierarchy contains no renderable mesh instances");
    cleanup_failed_build();
    return false;
  }

  std::vector<RHIAccelerationStructureInstance> rhi_instances;
  rhi_instances.reserve(s.hierarchy.mesh_instances.size());
  for (uint32_t instance_index = 0u; instance_index < s.hierarchy.mesh_instances.size(); ++instance_index) {
    const ResolvedMeshInstance& resolved = s.hierarchy.mesh_instances[instance_index];
    if (resolved.mesh_index >= new_blas.size()) {
      log::error("GPU RT: instance %u references invalid mesh %u", instance_index, resolved.mesh_index);
      cleanup_failed_build();
      return false;
    }
    if (instance_index > 0x00ffffffu) {
      log::error("GPU RT: TLAS instance count exceeds the 24-bit custom-index limit");
      cleanup_failed_build();
      return false;
    }

    RHIAccelerationStructureInstance& instance = rhi_instances.emplace_back();
    memcpy(instance.transform, resolved.object_to_world.rows, sizeof(instance.transform));
    instance.instance_custom_index = instance_index;
    instance.mask = (resolved.flags & ResolvedMeshInstance::Enabled) != 0u ? 0xffu : 0u;
    instance.instance_shader_binding_table_record_offset = 0u;
    instance.flags = 0u;
    instance.acceleration_structure_reference = device.get_acceleration_structure_device_address(new_blas[resolved.mesh_index]);
  }

  RHIBufferDesc inst_buf_desc = {};
  inst_buf_desc.size = rhi_instances.size() * sizeof(RHIAccelerationStructureInstance);
  inst_buf_desc.usage = RHIBufferUsage::ShaderDeviceAddress | RHIBufferUsage::AccelerationStructureBuild | RHIBufferUsage::TransferDst;
  const auto tlas_instance_upload_begin = std::chrono::steady_clock::now();
  auto inst_res = device.create_buffer(inst_buf_desc);
  if ((inst_res.result != RHIResult::Success) || (inst_res.handle.valid() == false)) {
    log::error("GPU RT: failed to create TLAS instance buffer (%u)", static_cast<uint32_t>(inst_res.result));
    cleanup_failed_build();
    return false;
  }
  const RHIResult inst_update_result = device.update_buffer(inst_res.handle, rhi_instances.data(), inst_buf_desc.size);
  if (inst_update_result != RHIResult::Success) {
    log::error("GPU RT: failed to upload TLAS instance buffer (%u)", static_cast<uint32_t>(inst_update_result));
    device.destroy_buffer(inst_res.handle);
    cleanup_failed_build();
    return false;
  }
  new_blas_buffers.push_back(inst_res.handle);
  const auto tlas_instance_upload_end = std::chrono::steady_clock::now();
  tlas_instance_upload_ms = elapsed_ms(tlas_instance_upload_begin, tlas_instance_upload_end);

  RHIAccelerationStructureDesc tlas_desc = {};
  tlas_desc.type = RHIAccelerationStructureType::TopLevel;
  tlas_desc.instance_count = static_cast<uint32_t>(rhi_instances.size());
  tlas_desc.allow_update = true;

  const auto tlas_create_begin = std::chrono::steady_clock::now();
  auto tlas_result = device.create_acceleration_structure(tlas_desc);
  const auto tlas_create_end = std::chrono::steady_clock::now();
  tlas_create_ms = elapsed_ms(tlas_create_begin, tlas_create_end);
  if ((tlas_result.result != RHIResult::Success) || (tlas_result.handle.valid() == false)) {
    log::error("GPU RT: failed to create TLAS (%u)", static_cast<uint32_t>(tlas_result.result));
    cleanup_failed_build();
    return false;
  }
  new_tlas = tlas_result.handle;

  uint64_t blas_scratch_size = 0u;
  for (RHIBindlessHandle blas : new_blas) {
    const uint64_t required_size = device.get_acceleration_structure_build_scratch_size(blas);
    if (required_size == 0u) {
      log::error("GPU RT: failed to query BLAS scratch size");
      cleanup_failed_build();
      return false;
    }
    blas_scratch_size = max(blas_scratch_size, required_size);
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
  const auto scratch_create_begin = std::chrono::steady_clock::now();
  auto scratch_res = device.create_buffer(scratch_desc);
  const auto scratch_create_end = std::chrono::steady_clock::now();
  scratch_create_ms = elapsed_ms(scratch_create_begin, scratch_create_end);
  if ((scratch_res.result != RHIResult::Success) || (scratch_res.handle.valid() == false)) {
    log::error("GPU RT: failed to create AS scratch buffer (%u)", static_cast<uint32_t>(scratch_res.result));
    cleanup_failed_build();
    return false;
  }
  new_blas_buffers.push_back(scratch_res.handle);

  RHIAccelerationStructureBuildDesc tlas_build_desc = {};
  tlas_build_desc.as_handle = new_tlas;
  tlas_build_desc.type = RHIAccelerationStructureType::TopLevel;
  tlas_build_desc.instance_count = static_cast<uint32_t>(rhi_instances.size());
  tlas_build_desc.instance_buffer = inst_res.handle;
  tlas_build_desc.allow_update = true;

  auto cmd = ctx.get_command_buffer();
  if (cmd.valid() == false) {
    log::error("GPU RT: failed to get command buffer for AS build");
    cleanup_failed_build();
    return false;
  }
  const auto build_submit_begin = std::chrono::steady_clock::now();
  {
    ETX_PROFILER_NAMED_SCOPE("gpu_rt_build_blas");
    ctx.command_buffer_begin(cmd);
    for (const RHIAccelerationStructureBuildDesc& build_desc : blas_build_descs) {
      ctx.cmd_build_acceleration_structure(cmd, build_desc, scratch_res.handle, 0u);
      ctx.cmd_buffer_barrier(cmd, scratch_res.handle, RHIResourceState::AccelerationStructure, RHIResourceState::AccelerationStructure);
    }
  }

  {
    ETX_PROFILER_NAMED_SCOPE("gpu_rt_build_tlas");
    ctx.cmd_build_acceleration_structure(cmd, tlas_build_desc, scratch_res.handle, 0);
    ctx.command_buffer_end(cmd);
    ctx.submit_command_buffer({cmd});
  }
  const auto build_submit_end = std::chrono::steady_clock::now();
  build_submit_ms = elapsed_ms(build_submit_begin, build_submit_end);

  const auto build_wait_begin = std::chrono::steady_clock::now();
  const RHIResult as_wait_result = ctx.wait_for_command_buffer(cmd);
  const auto build_wait_end = std::chrono::steady_clock::now();
  build_wait_ms = elapsed_ms(build_wait_begin, build_wait_end);
  if (as_wait_result != RHIResult::Success) {
    log::error("GPU RT: failed to wait for AS build completion (%u)", static_cast<uint32_t>(as_wait_result));
    ctx.destroy_command_buffer(cmd);
    cleanup_failed_build();
    return false;
  }
  ctx.destroy_command_buffer(cmd);

  const RHIResult index_destroy_result = device.destroy_buffer(ib_res.handle);
  if (index_destroy_result != RHIResult::Success) {
    log::warning("GPU RT: failed to release transient acceleration-structure index buffer (%u)", static_cast<uint32_t>(index_destroy_result));
  }
  new_blas_buffers.erase(new_blas_buffers.begin() + 1u);

  _vertex_positions_buffer = new_vertex_positions_buffer;
  _vertex_positions_buffer_size = vb_desc.size;
  _blas = std::move(new_blas);
  _blas_buffers = std::move(new_blas_buffers);
  _tlas_instance_staging = std::move(rhi_instances);
  _tlas_instance_count = static_cast<uint32_t>(_tlas_instance_staging.size());
  _tlas = new_tlas;
  _tlas_instance_buffer = inst_res.handle;
  _as_scratch_buffer = scratch_res.handle;
  _gpu_scene.vertex_positions = get_bindless_descriptor_index(_vertex_positions_buffer);

  ETX_PROFILER_NAMED_SCOPE("gpu_rt_upload_scene_after_as_build");
  const auto scene_upload_begin = std::chrono::steady_clock::now();
  const bool upload_success = upload_scene_data(ctx, scene, _vertex_positions_buffer);
  const auto scene_upload_end = std::chrono::steady_clock::now();
  scene_upload_ms = elapsed_ms(scene_upload_begin, scene_upload_end);
  if (upload_success == false) {
    log::error("GPU RT: failed to upload scene data after AS build");
  }
  return upload_success;
}

bool GPURaytracingRenderer::upload_scene_data(RHIContext& ctx, SceneRepresentation& scene, RHIBindlessHandle vertex_positions_buffer) {
  ETX_PROFILER_SCOPE();
  const auto total_begin = std::chrono::steady_clock::now();
  double packed_emitters_ms = 0.0;
  double geometry_buffers_ms = 0.0;
  double material_and_emitter_buffers_ms = 0.0;
  double images_ms = 0.0;
  double mediums_ms = 0.0;
  double scene_globals_ms = 0.0;
  double scene_options_ms = 0.0;
  double emitters_distribution_ms = 0.0;

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
    const auto packed_emitters_begin = std::chrono::steady_clock::now();
    packed_emitters = build_packed_emitters(data, _emitter_topology);
    const auto packed_emitters_end = std::chrono::steady_clock::now();
    packed_emitters_ms = elapsed_ms(packed_emitters_begin, packed_emitters_end);
  }

  bool upload_success = true;
  {
    ETX_PROFILER_NAMED_SCOPE("gpu_rt_upload_vertex_and_geometry_buffers");
    const auto geometry_buffers_begin = std::chrono::steady_clock::now();
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
    upload_success = upload_or_update_linear_scene_buffer(device, packed_emitters.instances.data(), packed_emitters.instances.size(), scene_buffer_usage, _instances_buffer,
                       _instances_buffer_size, _gpu_scene.instances, "instances") &&
                     upload_success;
    const auto geometry_buffers_end = std::chrono::steady_clock::now();
    geometry_buffers_ms = elapsed_ms(geometry_buffers_begin, geometry_buffers_end);
  }

  {
    ETX_PROFILER_NAMED_SCOPE("gpu_rt_upload_material_and_emitter_buffers");
    const auto material_and_emitter_buffers_begin = std::chrono::steady_clock::now();
    upload_success = upload_or_update_linear_scene_buffer(device, packed_emitters.emitter_profiles.data(), packed_emitters.emitter_profiles.size(), scene_buffer_usage,
                       _emitter_profiles_buffer, _emitter_profiles_buffer_size, _gpu_scene.emitter_profiles, "emitter_profiles") &&
                     upload_success;
    upload_success = upload_or_update_linear_scene_buffer(device, packed_emitters.emitter_instances.data(), packed_emitters.emitter_instances.size(), scene_buffer_usage,
                       _emitter_instances_buffer, _emitter_instances_buffer_size, _gpu_scene.emitter_instances, "emitter_instances") &&
                     upload_success;
    upload_success = upload_or_update_linear_scene_buffer(device, data.materials.data(), data.materials.size(), scene_buffer_usage, _materials_buffer, _materials_buffer_size,
                       _gpu_scene.materials, "materials") &&
                     upload_success;
    upload_success =
      upload_or_update_linear_scene_buffer(device, data.spectrum_values.data(), data.spectrum_values.size(), scene_buffer_usage, _spectrums_buffer, _spectrums_buffer_size,
        _gpu_scene.spectrums, "spectrums") &&
      upload_or_update_linear_scene_buffer(device, data.energy_compensation_interfaces.data(), data.energy_compensation_interfaces.size(), scene_buffer_usage,
        _energy_compensation_interfaces_buffer, _energy_compensation_interfaces_buffer_size, _gpu_scene.energy_compensation_interfaces, "energy_compensation_interfaces") &&
      upload_success;
    const auto material_and_emitter_buffers_end = std::chrono::steady_clock::now();
    material_and_emitter_buffers_ms = elapsed_ms(material_and_emitter_buffers_begin, material_and_emitter_buffers_end);
  }

  {
    ETX_PROFILER_NAMED_SCOPE("gpu_rt_upload_images");
    const auto images_begin = std::chrono::steady_clock::now();
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
    const auto images_end = std::chrono::steady_clock::now();
    images_ms = elapsed_ms(images_begin, images_end);
  }

  {
    ETX_PROFILER_NAMED_SCOPE("gpu_rt_upload_mediums");
    const auto mediums_begin = std::chrono::steady_clock::now();
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
    const auto mediums_end = std::chrono::steady_clock::now();
    mediums_ms = elapsed_ms(mediums_begin, mediums_end);
  }

  {
    ETX_PROFILER_NAMED_SCOPE("gpu_rt_upload_scene_globals");
    const auto scene_globals_begin = std::chrono::steady_clock::now();
    _host_transport_bounds = data.compute_transport_bounding_volumes();
    _host_scene_globals = build_scene_globals(data, scene.camera(), packed_emitters, _host_transport_bounds);
    const float3 geometry_center = 0.5f * (_host_scene_globals.bounding_box_min + _host_scene_globals.bounding_box_max);
    _scene_bounding_sphere_radius = length(_host_scene_globals.bounding_box_max - geometry_center);
    upload_success = upload_or_update_linear_scene_buffer(device, &_host_scene_globals, size_t(1), scene_buffer_usage, _scene_globals_buffer, _scene_globals_buffer_size,
                       _gpu_scene.scene_globals, "scene_globals") &&
                     upload_success;
    const auto scene_globals_end = std::chrono::steady_clock::now();
    scene_globals_ms = elapsed_ms(scene_globals_begin, scene_globals_end);
  }

  {
    ETX_PROFILER_NAMED_SCOPE("gpu_rt_upload_scene_options");
    const auto scene_options_begin = std::chrono::steady_clock::now();
    GPUSceneOptions options = build_scene_options(scene);
    upload_success = upload_or_update_linear_scene_buffer(device, &options, size_t(1), scene_buffer_usage, _scene_options_buffer, _scene_options_buffer_size,
                       _gpu_scene.scene_options, "scene_options") &&
                     upload_success;
    const auto scene_options_end = std::chrono::steady_clock::now();
    scene_options_ms = elapsed_ms(scene_options_begin, scene_options_end);
  }

  {
    ETX_PROFILER_NAMED_SCOPE("gpu_rt_upload_emitters_distribution");
    const auto emitters_distribution_begin = std::chrono::steady_clock::now();
    auto emitters_distribution = build_packed_emitter_distribution(packed_emitters);
    upload_success = upload_or_update_linear_scene_buffer(device, emitters_distribution.data(), emitters_distribution.size(), scene_buffer_usage, _emitters_distribution_buffer,
                       _emitters_distribution_buffer_size, _gpu_scene.emitters_distribution, "emitters_distribution") &&
                     upload_success;
    const auto emitters_distribution_end = std::chrono::steady_clock::now();
    emitters_distribution_ms = elapsed_ms(emitters_distribution_begin, emitters_distribution_end);
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
    if (changes[UpdateFlags::EnergyCompensationInterfaces]) {
      upload_success =
        upload_or_update_linear_scene_buffer(device, data.energy_compensation_interfaces.data(), data.energy_compensation_interfaces.size(), scene_buffer_usage,
          _energy_compensation_interfaces_buffer, _energy_compensation_interfaces_buffer_size, _gpu_scene.energy_compensation_interfaces, "energy_compensation_interfaces") &&
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
    if (changes[UpdateFlags::Mediums] || changes[UpdateFlags::Transforms]) {
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

  const bool packed_emitters_changed = changes[UpdateFlags::VerticesPos] || changes[UpdateFlags::Triangles] || changes[UpdateFlags::Meshes] || changes[UpdateFlags::Hierarchy] ||
                                       changes[UpdateFlags::Transforms] || changes[UpdateFlags::Attachments] || changes[UpdateFlags::Emitters] || changes[UpdateFlags::Materials] ||
                                       changes[UpdateFlags::Spectra];
  const bool transform_only_packing = changes[UpdateFlags::Transforms] && (changes[UpdateFlags::VerticesPos] == false) && (changes[UpdateFlags::Triangles] == false) &&
                                      (changes[UpdateFlags::Meshes] == false) && (changes[UpdateFlags::Hierarchy] == false) && (changes[UpdateFlags::Attachments] == false) &&
                                      (changes[UpdateFlags::Emitters] == false) && (changes[UpdateFlags::Materials] == false) && (changes[UpdateFlags::Spectra] == false);
  const bool scene_globals_changed = changes[UpdateFlags::VerticesPos] || changes[UpdateFlags::Triangles] || changes[UpdateFlags::Meshes] || changes[UpdateFlags::Hierarchy] ||
                                     changes[UpdateFlags::Transforms] || changes[UpdateFlags::Attachments] || changes[UpdateFlags::Materials] || changes[UpdateFlags::Spectra] ||
                                     changes[UpdateFlags::Emitters] || changes[UpdateFlags::EnergyCompensationInterfaces] || changes[UpdateFlags::Defaults] ||
                                     changes[UpdateFlags::Images] || changes[UpdateFlags::Mediums] || changes[UpdateFlags::PixelFilter];

  PackedEmitterData packed_emitters = {};
  if (packed_emitters_changed || scene_globals_changed) {
    ETX_PROFILER_NAMED_SCOPE("gpu_rt_partial_build_packed_emitters");
    packed_emitters = transform_only_packing ? build_packed_emitters_for_transforms(data, _emitter_topology) : build_packed_emitters(data, _emitter_topology);
  }

  {
    ETX_PROFILER_NAMED_SCOPE("gpu_rt_partial_dependent_buffer_updates");
    if (changes[UpdateFlags::Triangles] || changes[UpdateFlags::Emitters]) {
      upload_success = upload_or_update_linear_scene_buffer(device, packed_emitters.triangles.data(), packed_emitters.triangles.size(), scene_buffer_usage, _triangles_buffer,
                         _triangles_buffer_size, _gpu_scene.triangles, "triangles") &&
                       upload_success;
    }

    if (packed_emitters_changed) {
      upload_success = upload_or_update_linear_scene_buffer(device, packed_emitters.instances.data(), packed_emitters.instances.size(), scene_buffer_usage, _instances_buffer,
                         _instances_buffer_size, _gpu_scene.instances, "instances") &&
                       upload_success;

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
    _host_transport_bounds = data.compute_transport_bounding_volumes();
    _host_scene_globals = build_scene_globals(data, scene.camera(), packed_emitters, _host_transport_bounds);
    const float3 geometry_center = 0.5f * (_host_scene_globals.bounding_box_min + _host_scene_globals.bounding_box_max);
    // Keep density-kernel scale independent of the camera-dependent transport domain.
    _scene_bounding_sphere_radius = length(_host_scene_globals.bounding_box_max - geometry_center);
    upload_success = upload_or_update_linear_scene_buffer(device, &_host_scene_globals, size_t(1), scene_buffer_usage, _scene_globals_buffer, _scene_globals_buffer_size,
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

#if defined(ETX_ENABLE_SHADER_PACKAGER) && ETX_ENABLE_SHADER_PACKAGER
bool build_raytracer_shader_package(const std::filesystem::path& output_path, const std::filesystem::path& source_root, RHIBackend backend,
  RaytracerShaderPackageStatistics& statistics, std::string& error_message) {
  statistics = {};
  error_message.clear();

  std::map<uint64_t, ShaderPackageRequest> unique_requests = {};
  auto add_request = [&](const char* source_name, const char* entry_point, RHIShaderStage stage, const std::unordered_map<std::string, std::string>& defines) -> bool {
    ShaderPackageRequest request = {
      .source_name = source_name,
      .entry_point = entry_point,
      .stage = stage,
      .backend = backend,
      .defines = std::map<std::string, std::string>(defines.begin(), defines.end()),
    };
    const uint64_t request_hash = shader_package_request_hash(request);
    const auto [iterator, inserted] = unique_requests.emplace(request_hash, request);
    if (inserted) {
      return true;
    }
    if ((iterator->second.source_name != request.source_name) || (iterator->second.entry_point != request.entry_point) || (iterator->second.stage != request.stage) ||
        (iterator->second.backend != request.backend) || (iterator->second.defines != request.defines)) {
      error_message = "Shader package request hash collision detected while building the inventory.";
      return false;
    }
    return true;
  };

  constexpr uint32_t all_camera_features = GPUIntegratorFeatures::CameraPath | GPUIntegratorFeatures::DirectHit | GPUIntegratorFeatures::ConnectToLight;
  constexpr uint32_t all_light_features = GPUIntegratorFeatures::LightPath | GPUIntegratorFeatures::ConnectToCamera;
  constexpr uint32_t all_upbp_features = GPUIntegratorFeatures::UPBPMis | GPUIntegratorFeatures::UPBPSurface | GPUIntegratorFeatures::UPBPPP3D | GPUIntegratorFeatures::UPBPPB2D |
                                         GPUIntegratorFeatures::UPBPBP2D | GPUIntegratorFeatures::UPBPBB1D;
  struct PackageIntegratorConfiguration {
    GPUIntegratorMode mode = GPUIntegratorMode::PathTracing;
    uint32_t features = 0u;
  };
  constexpr PackageIntegratorConfiguration integrator_configurations[] = {
    {GPUIntegratorMode::PathTracing, all_camera_features},
    {GPUIntegratorMode::LightTracing, all_light_features},
    {GPUIntegratorMode::BDPTFast, all_camera_features | all_light_features},
    {GPUIntegratorMode::BDPTFull, all_camera_features | all_light_features | GPUIntegratorFeatures::ConnectVertices},
    {GPUIntegratorMode::VCM,
      all_camera_features | all_light_features | GPUIntegratorFeatures::ConnectVertices | GPUIntegratorFeatures::MergeVertices | GPUIntegratorFeatures::VCMMis},
    {GPUIntegratorMode::UPBP, all_camera_features | all_light_features | GPUIntegratorFeatures::ConnectVertices | GPUIntegratorFeatures::MergeVertices | all_upbp_features},
  };
  constexpr uint32_t material_mask_count = 1u << MaterialClass::Count;
  constexpr uint32_t open_pbr_bit = 1u << MaterialClass::OpenPBR;
  constexpr uint32_t spectral_modes[] = {static_cast<uint32_t>(GPUSpectralMode::RGB), static_cast<uint32_t>(GPUSpectralMode::Spectral)};

  for (const PackageIntegratorConfiguration& configuration : integrator_configurations) {
    for (uint32_t material_mask = 0u; material_mask < material_mask_count; ++material_mask) {
      if ((material_mask & open_pbr_bit) != 0u) {
        continue;
      }
      const uint32_t package_material_mask = material_mask | (material_compile_mask_has(material_mask, MaterialClass::Conductor) ? kMaterialCompileConnectibleConductor : 0u) |
                                             (material_compile_mask_has(material_mask, MaterialClass::Dielectric) ? kMaterialCompileConnectibleDielectric : 0u);
      for (const uint32_t spectral_mode : spectral_modes) {
        for (const WavefrontStage& stage : kWavefrontStages) {
          if (wavefront_stage_enabled(stage.stage, configuration.mode, configuration.features, package_material_mask, spectral_mode) == false) {
            continue;
          }
          const auto defines = wavefront_stage_defines(stage, configuration.mode, package_material_mask, spectral_mode);
          if (add_request(stage.source_file, stage.entry_point, RHIShaderStage::Compute, defines) == false) {
            return false;
          }
        }
      }
    }
  }

  const std::unordered_map<std::string, std::string> no_defines = {};
  const std::unordered_map<std::string, std::string> presentation_srgb_defines = {{"ETX_PRESENT_SRGB_TARGET", "1"}};
  const std::unordered_map<std::string, std::string> imgui_srgb_defines = {{"ETX_IMGUI_SRGB_TARGET", "1"}};
  const std::pair<const char*, RHIShaderStage> presentation_entry_points[] = {
    {"vertex_main", RHIShaderStage::Vertex},
    {"fragment_main", RHIShaderStage::Fragment},
  };
  const std::pair<const char*, RHIShaderStage> imgui_entry_points[] = {
    {"vs_main", RHIShaderStage::Vertex},
    {"ps_main", RHIShaderStage::Fragment},
  };
  for (const auto& [entry_point, stage] : presentation_entry_points) {
    if ((add_request("shaders/render.hlsl", entry_point, stage, no_defines) == false) ||
        (add_request("shaders/render.hlsl", entry_point, stage, presentation_srgb_defines) == false)) {
      return false;
    }
  }
  for (const auto& [entry_point, stage] : imgui_entry_points) {
    if ((add_request("shaders/imgui.hlsl", entry_point, stage, no_defines) == false) || (add_request("shaders/imgui.hlsl", entry_point, stage, imgui_srgb_defines) == false)) {
      return false;
    }
  }
  if ((add_request("shaders/atmosphere_optical_depth.hlsl", "optical_depth_main", RHIShaderStage::Compute, no_defines) == false) ||
      (add_request("shaders/atmosphere_sky.hlsl", "sky_raw_main", RHIShaderStage::Compute, no_defines) == false) ||
      (add_request("shaders/atmosphere_sky.hlsl", "sky_finalize_main", RHIShaderStage::Compute, no_defines) == false) ||
      (add_request("shaders/atmosphere_sun.hlsl", "sun_main", RHIShaderStage::Compute, no_defines) == false) ||
      (add_request("shaders/bsdf_energy_compensation.hlsl", "main", RHIShaderStage::Compute, no_defines) == false)) {
    return false;
  }

  struct PackageCompileGroup {
    std::string source_name = {};
    std::map<std::string, std::string> defines = {};
    std::vector<ShaderPackageRequest> requests = {};
  };
  std::map<std::string, PackageCompileGroup> groups = {};
  for (const auto& [request_hash, request] : unique_requests) {
    static_cast<void>(request_hash);
    std::string group_key = request.source_name;
    for (const auto& [name, value] : request.defines) {
      group_key += '\0';
      group_key += name;
      group_key += '\0';
      group_key += value;
    }
    PackageCompileGroup& group = groups[group_key];
    group.source_name = request.source_name;
    group.defines = request.defines;
    group.requests.push_back(request);
  }

  auto& compiler = ShaderCompiler::instance();
  compiler.set_runtime_compilation_allowed(true);
  compiler.set_shader_package_lookup_allowed(false);
  const RHIResult initialization_result = compiler.initialize();
  if (initialization_result != RHIResult::Success) {
    error_message = "Failed to initialize the shader compiler for package generation.";
    return false;
  }
  compiler.reset_statistics();
  std::vector<ShaderPackageBuildEntry> package_entries = {};
  package_entries.reserve(unique_requests.size());
  const auto compile_begin = std::chrono::steady_clock::now();
  const std::filesystem::path metal_library_cache_directory = output_path.parent_path() / "shader-package-cache";
# if defined(ETX_SHADER_PACKAGE_MACOS_DEPLOYMENT_TARGET)
  const std::string minimum_macos_version = ETX_SHADER_PACKAGE_MACOS_DEPLOYMENT_TARGET;
# else
  const std::string minimum_macos_version = {};
# endif
  std::vector<const PackageCompileGroup*> compile_groups = {};
  compile_groups.reserve(groups.size());
  for (const auto& [group_key, group] : groups) {
    static_cast<void>(group_key);
    compile_groups.push_back(&group);
  }

  const uint32_t available_threads = std::max(1u, std::thread::hardware_concurrency());
  const uint32_t worker_count = std::min<uint32_t>(static_cast<uint32_t>(compile_groups.size()), std::min(available_threads, 6u));
  const uint32_t total_variant_count = static_cast<uint32_t>(unique_requests.size());
  const uint32_t total_group_count = static_cast<uint32_t>(compile_groups.size());
  std::atomic<size_t> next_group_index = 0u;
  std::atomic<uint32_t> completed_variants = 0u;
  std::atomic<uint32_t> completed_groups = 0u;
  std::atomic<bool> compilation_failed = false;
  std::atomic<bool> compilation_finished = false;
  std::mutex package_entries_mutex = {};
  std::mutex error_mutex = {};
  std::mutex progress_mutex = {};
  std::condition_variable progress_condition = {};
  auto report_failure = [&](std::string message) {
    bool expected = false;
    if (compilation_failed.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
      std::lock_guard<std::mutex> error_lock(error_mutex);
      error_message = std::move(message);
    }
  };
  auto compile_group = [&]() {
    while (compilation_failed.load(std::memory_order_acquire) == false) {
      const size_t group_index = next_group_index.fetch_add(1u);
      if (group_index >= compile_groups.size()) {
        return;
      }
      const PackageCompileGroup& group = *compile_groups[group_index];
      const std::filesystem::path source_path = source_root / group.source_name;
      std::string read_error = {};
      const std::string source = compiler.read_file_content(source_path.string(), read_error);
      if (read_error.empty() == false) {
        report_failure("Failed to read shader source '" + group.source_name + "': " + read_error);
        return;
      }

      std::vector<ShaderCompiler::ShaderEntryPoint> entry_points = {};
      entry_points.reserve(group.requests.size());
      for (const ShaderPackageRequest& request : group.requests) {
        entry_points.push_back({request.entry_point, request.stage});
      }
      const std::unordered_map<std::string, std::string> defines(group.defines.begin(), group.defines.end());
      ShaderCompiler::MultiShaderCompilationResult compilation = compiler.compile(source, source_path.string(), entry_points, defines, backend);
      if ((compilation.result != RHIResult::Success) || (compilation.binaries.size() != group.requests.size())) {
        report_failure("Failed to compile shader package group '" + group.source_name + "': " + compilation.error_message);
        return;
      }

      std::vector<ShaderPackageBuildEntry> group_entries = {};
      group_entries.reserve(compilation.binaries.size());
      for (size_t binary_index = 0u; binary_index < compilation.binaries.size(); ++binary_index) {
        const RHIShaderBinary& compiled_binary = compilation.binaries[binary_index];
        ShaderPackageBuildEntry package_entry = {
          .request = group.requests[binary_index],
        };
        package_entry.binary.local_size_x = compiled_binary.local_size_x;
        package_entry.binary.local_size_y = compiled_binary.local_size_y;
        package_entry.binary.local_size_z = compiled_binary.local_size_z;
        package_entry.binary.metal_metadata = compiled_binary.metal_metadata;
        if (backend == RHIBackend::Metal) {
          package_entry.binary.format = RHIShaderBinaryFormat::MetalLibrary;
          std::string metal_error = {};
          if (compile_metal_shader_library(compiled_binary.spirv_data, compiled_binary.spirv_size, metal_library_cache_directory, minimum_macos_version, package_entry.binary.data,
                metal_error) == false) {
            report_failure("Failed to build Metal library for '" + group.requests[binary_index].entry_point + "': " + metal_error);
            return;
          }
        } else {
          package_entry.binary.format = RHIShaderBinaryFormat::SpirV;
          package_entry.binary.data.assign(compiled_binary.spirv_data, compiled_binary.spirv_data + compiled_binary.spirv_size);
        }
        group_entries.push_back(std::move(package_entry));
      }

      const uint32_t group_variant_count = static_cast<uint32_t>(group_entries.size());
      {
        std::lock_guard<std::mutex> entries_lock(package_entries_mutex);
        package_entries.insert(package_entries.end(), std::make_move_iterator(group_entries.begin()), std::make_move_iterator(group_entries.end()));
      }
      completed_variants.fetch_add(group_variant_count);
      completed_groups.fetch_add(1u);
    }
  };

  auto format_duration = [](const double seconds) {
    const uint64_t rounded_seconds = static_cast<uint64_t>(std::ceil(std::max(0.0, seconds)));
    const uint64_t hours = rounded_seconds / 3600u;
    const uint64_t minutes = (rounded_seconds % 3600u) / 60u;
    const uint64_t remaining_seconds = rounded_seconds % 60u;
    char buffer[64] = {};
    if (hours > 0u) {
      snprintf(buffer, sizeof(buffer), "%lluh %02llum %02llus", static_cast<unsigned long long>(hours), static_cast<unsigned long long>(minutes),
        static_cast<unsigned long long>(remaining_seconds));
    } else if (minutes > 0u) {
      snprintf(buffer, sizeof(buffer), "%llum %02llus", static_cast<unsigned long long>(minutes), static_cast<unsigned long long>(remaining_seconds));
    } else {
      snprintf(buffer, sizeof(buffer), "%llus", static_cast<unsigned long long>(remaining_seconds));
    }
    return std::string{buffer};
  };
  auto report_progress = [&]() {
    const uint32_t completed_variant_count = completed_variants.load(std::memory_order_acquire);
    const uint32_t completed_group_count = completed_groups.load(std::memory_order_acquire);
    const uint32_t claimed_group_count = std::min<uint32_t>(static_cast<uint32_t>(next_group_index.load(std::memory_order_acquire)), total_group_count);
    const uint32_t active_group_count = claimed_group_count > completed_group_count ? claimed_group_count - completed_group_count : 0u;
    const double elapsed_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - compile_begin).count();
    const double progress = total_variant_count > 0u ? static_cast<double>(completed_variant_count) / static_cast<double>(total_variant_count) : 1.0;
    std::string estimated_remaining = "estimating";
    if (compilation_finished.load(std::memory_order_acquire)) {
      estimated_remaining = compilation_failed.load(std::memory_order_acquire) ? "unavailable" : "0s";
    } else if (completed_variant_count > 0u) {
      estimated_remaining = format_duration(elapsed_seconds * (1.0 - progress) / progress);
    }
    log::info("Shader package compile progress: %u/%u variants (%.1f%%), groups %u/%u (%u active), elapsed %s, approximate remaining %s", completed_variant_count,
      total_variant_count, progress * 100.0, completed_group_count, total_group_count, active_group_count, format_duration(elapsed_seconds).c_str(), estimated_remaining.c_str());
    fflush(stdout);
  };

  log::info("Shader package compilation started: backend=%s variants=%u groups=%u workers=%u; progress updates every 10 seconds", backend == RHIBackend::Metal ? "metal" : "vulkan",
    total_variant_count, total_group_count, worker_count);
  fflush(stdout);
  std::thread progress_thread([&]() {
    while (true) {
      std::unique_lock<std::mutex> progress_lock(progress_mutex);
      progress_condition.wait_for(progress_lock, std::chrono::seconds(10), [&]() {
        return compilation_finished.load(std::memory_order_acquire);
      });
      progress_lock.unlock();
      report_progress();
      if (compilation_finished.load(std::memory_order_acquire)) {
        return;
      }
    }
  });

  std::vector<std::thread> workers = {};
  workers.reserve(worker_count);
  for (uint32_t worker_index = 0u; worker_index < worker_count; ++worker_index) {
    workers.emplace_back(compile_group);
  }
  for (std::thread& worker : workers) {
    worker.join();
  }
  compilation_finished.store(true, std::memory_order_release);
  progress_condition.notify_one();
  progress_thread.join();
  if (compilation_failed.load(std::memory_order_acquire)) {
    return false;
  }
  const auto compile_end = std::chrono::steady_clock::now();
  statistics.compile_time_ms = elapsed_ms(compile_begin, compile_end);
  compiler.log_statistics("shader-package");
  log::info("Shader compilation complete; writing and verifying the production package");
  fflush(stdout);

  ShaderPackageBuildStatistics package_statistics = {};
  const auto package_begin = std::chrono::steady_clock::now();
  if (write_shader_package(output_path, std::move(package_entries), package_statistics, error_message) == false) {
    return false;
  }

  ShaderPackage verification_package = {};
  if (verification_package.load(output_path, backend, error_message) == false) {
    error_message = "Generated shader package verification failed: " + error_message;
    return false;
  }
  if (verification_package.entry_count() != unique_requests.size()) {
    error_message = "Generated shader package contains an unexpected number of variants.";
    return false;
  }
  for (const auto& [request_hash, request] : unique_requests) {
    static_cast<void>(request_hash);
    ShaderPackageBinary verification_binary = {};
    if (verification_package.read(request, verification_binary, error_message) == false) {
      error_message = "Generated shader package entry verification failed: " + error_message;
      return false;
    }
  }
  const auto package_end = std::chrono::steady_clock::now();

  statistics.variant_count = package_statistics.entry_count;
  statistics.binary_size_bytes = package_statistics.binary_size_bytes;
  statistics.package_size_bytes = package_statistics.package_size_bytes;
  statistics.package_time_ms = elapsed_ms(package_begin, package_end);
  return true;
}
#endif

}  // namespace etx
