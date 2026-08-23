#include "gpu_renderer.hxx"
#include "shader_packager.hxx"
#include <interop/gpu_abi_constants.hxx>
#include <interop/gpu_rt_shared.hxx>
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
#include <etx/rt/shared/bdpt_mode.hxx>
#include <bluenoise.hxx>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <limits>
#include <map>
#include <set>
#include <thread>
#include <type_traits>
#include "gpu_renderer_abi_static_asserts.hxx"

namespace etx {

static bool update_host_visible_buffer(RHIDevice& device, const void* data, uint64_t required_size, RHIBufferUsage usage, RHIBindlessHandle& buffer, uint64_t& buffer_size,
  const char* buffer_name);

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
constexpr uint32_t kWavefrontBDPTTileSide = 512u;
// Seven in-place 112-byte connection records use less memory than the former
// four 160-byte tasks plus four 48-byte results.
constexpr uint32_t kWavefrontConnectLightBatchSize = 7u;
static_assert(kWavefrontConnectLightBatchSize == kGPUWavefrontConnectDispatchArgsCount);
constexpr uint32_t kWavefrontInitialLightHistoryBounces = 16u;
constexpr uint32_t kWavefrontLightHistoryShrinkSampleCount = 8u;
constexpr uint32_t kWavefrontAutoInitialSteps = 1u;
constexpr uint32_t kWavefrontAutoMaximumSteps = 1024u;
constexpr uint32_t kWavefrontAutoMaximumGrowthFactor = 4u;
constexpr double kWavefrontAutoTargetMs = 12.0;
constexpr double kWavefrontAutoLowerDeadZoneMs = 8.0;
constexpr double kWavefrontAutoUpperDeadZoneMs = 16.0;
constexpr double kWavefrontAutoSmoothingFactor = 0.25;
constexpr uint64_t kWavefrontLightVertexCounterSize = sizeof(uint32_t);
constexpr uint32_t kWavefrontCoarseQueueReadbackInterval = 16u;
constexpr uint32_t kWavefrontHeavyContinuationChunkSize = 65536u;
constexpr uint32_t kNonVulkanPipelinePublishBatchSize = 2u;
constexpr uint32_t kVulkanPipelineMaxWorkerCount = 6u;
constexpr uint64_t kVulkanPipelineWorkerMemoryReserve = 4ull * 1024ull * 1024ull * 1024ull;
constexpr uint64_t kWavefrontBDPTFallbackLightVertexBytes = 512ull * 1024ull * 1024ull;
constexpr uint64_t kWavefrontBDPTMemoryBudgetDivisor = 8ull;

static_assert(kGPUWavefrontDirectLightSampleStride == kGPUWavefrontConnectCameraTaskStride);
static_assert(kGPUWavefrontDirectLightTaskStride == kGPUWavefrontConnectCameraTaskStride);
static_assert(kGPUWavefrontDirectLightResultStride == kGPUWavefrontConnectCameraResultStride);

enum class GPUIntegratorMode : uint32_t {
  PathTracing = 0u,
  LightTracing = 1u,
  BDPTFast = 2u,
  BDPTFull = 3u,
  VCM = 4u,
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
  };
};

struct GPUIntegratorSelection {
  GPUIntegratorMode mode = GPUIntegratorMode::PathTracing;
  uint32_t features = 0u;
  Integrator::Type integrator_type = Integrator::Type::Invalid;
  BDPTMode requested_bdpt_mode = BDPTMode::Invalid;
  bool supported = true;
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
  {GPURaytracingRenderer::PipelineStage::CameraConnectLightClear, "shaders/gpu_rt_wavefront_connect_light_clear.hlsl", "wavefront_camera_connect_light_clear_main", nullptr,
    nullptr},
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
  {GPURaytracingRenderer::PipelineStage::BuildDispatchArgs, "shaders/gpu_rt_wavefront_dispatch_args.hlsl", "wavefront_build_dispatch_args_main", nullptr, nullptr},
  {GPURaytracingRenderer::PipelineStage::VCMGridClear, "shaders/gpu_rt_wavefront_vcm_grid.hlsl", "wavefront_vcm_grid_clear_main", "3", nullptr},
  {GPURaytracingRenderer::PipelineStage::VCMGridBuild, "shaders/gpu_rt_wavefront_vcm_grid.hlsl", "wavefront_vcm_grid_build_main", "3", nullptr},
  {GPURaytracingRenderer::PipelineStage::VCMMergeDiffuse, "shaders/gpu_rt_wavefront_vcm_merge_variant.hlsl", "wavefront_vcm_merge_diffuse_main", "3", "1", true},
  {GPURaytracingRenderer::PipelineStage::VCMMergePlastic, "shaders/gpu_rt_wavefront_vcm_merge_variant.hlsl", "wavefront_vcm_merge_plastic_main", "3", "2", true},
  {GPURaytracingRenderer::PipelineStage::VCMMergeConductor, "shaders/gpu_rt_wavefront_vcm_merge_variant.hlsl", "wavefront_vcm_merge_conductor_main", "3", "3", true},
  {GPURaytracingRenderer::PipelineStage::VCMMergeDielectric, "shaders/gpu_rt_wavefront_vcm_merge_variant.hlsl", "wavefront_vcm_merge_dielectric_main", "3", "4", true},
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

uint64_t material_dispatch_args_offset(bool from_camera, uint32_t material_queue_index) {
  const uint32_t path_queue_offset = from_camera ? 0u : kGPUWavefrontMaterialQueueCountPerPathType;
  return kGPUWavefrontMaterialDispatchArgsOffset + static_cast<uint64_t>(path_queue_offset + material_queue_index) * kGPUWavefrontDispatchArgsStride;
}

uint64_t shadow_dispatch_args_offset(uint32_t shadow_queue_index) {
  return kGPUWavefrontShadowDispatchArgsOffset + static_cast<uint64_t>(shadow_queue_index) * kGPUWavefrontDispatchArgsStride;
}

uint64_t wavefront_tile_budget_bytes(const RHIMemoryStats& memory_stats) {
  const uint64_t budget_bytes =
    (memory_stats.gpu_device_local_budget_bytes > 0ull) ? (memory_stats.gpu_device_local_budget_bytes / kWavefrontBDPTMemoryBudgetDivisor) : kWavefrontBDPTFallbackLightVertexBytes;
  return std::max<uint64_t>(1ull, std::min(budget_bytes, kWavefrontMaxAddressableBufferSize));
}

uint32_t wavefront_initial_light_history_bounces(uint32_t max_path_length) {
  return std::min(std::max(1u, max_path_length), kWavefrontInitialLightHistoryBounces);
}

uint64_t wavefront_tile_bytes_per_path(uint32_t integrator_features, bool has_subsurface_material, uint32_t light_history_capacity_bounces) {
  const bool enable_camera_path = (integrator_features & GPUIntegratorFeatures::CameraPath) != 0u;
  const bool enable_light_path = (integrator_features & GPUIntegratorFeatures::LightPath) != 0u;
  const bool enable_connect_to_light = (integrator_features & GPUIntegratorFeatures::ConnectToLight) != 0u;
  const bool enable_connect_to_camera = (integrator_features & GPUIntegratorFeatures::ConnectToCamera) != 0u;
  const bool enable_connect_vertices = (integrator_features & GPUIntegratorFeatures::ConnectVertices) != 0u;
  const bool enable_merge_vertices = (integrator_features & GPUIntegratorFeatures::MergeVertices) != 0u;
  const bool store_complete_light_history = enable_connect_vertices || enable_merge_vertices;
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
    result += static_cast<uint64_t>(kWavefrontConnectLightBatchSize) * kGPUWavefrontConnectLightTaskStride;
  }
  if (enable_merge_vertices) {
    // The power-of-two head table can approach two entries per retained light vertex,
    // plus one linked-list entry per vertex.
    result += static_cast<uint64_t>(light_history_bounces + 1u) * 3ull * sizeof(uint32_t);
  }
  return std::max<uint64_t>(1ull, result);
}

uint32_t wavefront_tile_max_pixels(uint64_t tile_budget_bytes, uint64_t tile_bytes_per_path, uint32_t requested_pixel_count) {
  const uint64_t max_pixels_by_budget = tile_budget_bytes / tile_bytes_per_path;
  if (static_cast<uint64_t>(requested_pixel_count) <= max_pixels_by_budget) {
    return std::max(1u, requested_pixel_count);
  }
  const uint64_t max_tile_pixels = static_cast<uint64_t>(kWavefrontBDPTTileSide) * static_cast<uint64_t>(kWavefrontBDPTTileSide);
  return static_cast<uint32_t>(std::max<uint64_t>(1ull, std::min(max_tile_pixels, max_pixels_by_budget)));
}

uint32_t wavefront_tile_width(uint2 base_size, uint32_t max_tile_pixels) {
  return std::min(base_size.x, std::min(kWavefrontBDPTTileSide, max_tile_pixels));
}

uint32_t wavefront_tile_count(uint2 base_size, uint32_t max_tile_pixels) {
  const uint32_t tile_width = wavefront_tile_width(base_size, max_tile_pixels);
  const uint32_t tile_height = std::max(1u, std::min(base_size.y, std::min(kWavefrontBDPTTileSide, max_tile_pixels / tile_width)));
  return divide_round_up(base_size.x, tile_width) * divide_round_up(base_size.y, tile_height);
}

WavefrontWindow wavefront_tile_window(uint2 base_origin, uint2 base_size, uint32_t max_tile_pixels, uint32_t tile_index) {
  const uint32_t tile_width = wavefront_tile_width(base_size, max_tile_pixels);
  const uint32_t tile_height = std::max(1u, std::min(base_size.y, std::min(kWavefrontBDPTTileSide, max_tile_pixels / tile_width)));
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
  uint32_t radius_decay = 256u;
  auto settings_it = scene.integrator_data().settings.find(Integrator::Type::VCM);
  if (settings_it != scene.integrator_data().settings.end()) {
    initial_radius = settings_it->second.get_float("vcm-initial_radius", initial_radius);
    radius_decay = settings_it->second.get_integral("vcm-radius_decay", radius_decay);
    result.kernel = settings_it->second.get_integral("vcm-kernel", result.kernel);
  }

  if (initial_radius == 0.0f) {
    const uint32_t max_dimension = std::max(1u, std::max(render_dimensions.x, render_dimensions.y));
    initial_radius = 5.0f * bounding_sphere_radius / static_cast<float>(max_dimension);
  }

  radius_decay = std::max(1u, radius_decay);
  result.radius = initial_radius / (1.0f + static_cast<float>(sample_index) / static_cast<float>(radius_decay));
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
    default:
      return "Unknown";
  }
}

const char* pipeline_stage_to_string(GPURaytracingRenderer::PipelineStage stage) {
  switch (stage) {
    case GPURaytracingRenderer::PipelineStage::PrepareSample:
      return "PrepareSample";
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
    case GPURaytracingRenderer::PipelineStage::CameraConnectLightClear:
      return "CameraConnectLightClear";
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
    case GPURaytracingRenderer::PipelineStage::BuildDispatchArgs:
      return "BuildDispatchArgs";
    case GPURaytracingRenderer::PipelineStage::VCMGridClear:
      return "VCMGridClear";
    case GPURaytracingRenderer::PipelineStage::VCMGridBuild:
      return "VCMGridBuild";
    case GPURaytracingRenderer::PipelineStage::VCMMergeDiffuse:
      return "VCMMergeDiffuse";
    case GPURaytracingRenderer::PipelineStage::VCMMergePlastic:
      return "VCMMergePlastic";
    case GPURaytracingRenderer::PipelineStage::VCMMergeConductor:
      return "VCMMergeConductor";
    case GPURaytracingRenderer::PipelineStage::VCMMergeDielectric:
      return "VCMMergeDielectric";
    case GPURaytracingRenderer::PipelineStage::Count:
      return "Count";
    default:
      return "Unknown";
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

uint32_t material_compile_mask_work_queue_count(uint32_t mask) {
  return static_cast<uint32_t>(material_compile_mask_has_various_continue(mask)) + static_cast<uint32_t>(material_compile_mask_has(mask, MaterialClass::Plastic)) +
         static_cast<uint32_t>(material_compile_mask_has_conductor_stage(mask)) + static_cast<uint32_t>(material_compile_mask_has(mask, MaterialClass::Dielectric)) +
         static_cast<uint32_t>(material_compile_mask_has(mask, MaterialClass::Thinfilm));
}

constexpr uint64_t pipeline_stage_bit(GPURaytracingRenderer::PipelineStage stage) {
  return 1ull << static_cast<uint32_t>(stage);
}

static_assert(static_cast<uint32_t>(GPURaytracingRenderer::PipelineStage::Count) <= 64u);

bool wavefront_stage_source_is(const WavefrontStage& stage, const char* source_file) {
  return (stage.source_file != nullptr) && (std::strcmp(stage.source_file, source_file) == 0);
}

struct WavefrontStageCompileOptions {
  bool path_tracing_only = false;
  bool work_queues = false;
  bool thinfilm = false;
  bool velvet = false;
};

WavefrontStageCompileOptions wavefront_stage_compile_options(const WavefrontStage& stage, GPUIntegratorMode mode, uint32_t material_compile_mask) {
  const bool diffuse_variant = (stage.bsdf_kind != nullptr) && (std::strcmp(stage.bsdf_kind, "1") == 0);
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
  result.thinfilm = diffuse_variant && material_compile_mask_has(material_compile_mask, MaterialClass::Thinfilm) &&
                    (direct_light_variant || connect_light_prepare_variant || connect_light_resolve_variant || connect_camera_variant || vcm_merge_variant);
  result.velvet =
    diffuse_variant && material_compile_mask_has(material_compile_mask, MaterialClass::Velvet) &&
    (surface_continue_variant || direct_light_variant || connect_light_prepare_variant || connect_light_resolve_variant || connect_camera_variant || vcm_merge_variant);
  return result;
}

uint64_t wavefront_stage_variant_key(const WavefrontStage& stage, GPUIntegratorMode mode, uint32_t material_compile_mask, uint32_t spectral_mode) {
  const WavefrontStageCompileOptions options = wavefront_stage_compile_options(stage, mode, material_compile_mask);
  uint64_t result = spectral_mode;
  result |= options.path_tracing_only ? (1ull << 8u) : 0ull;
  result |= options.work_queues ? (1ull << 9u) : 0ull;
  result |= options.thinfilm ? (1ull << 10u) : 0ull;
  result |= options.velvet ? (1ull << 11u) : 0ull;
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
  if (stage.uses_stage_entry_define) {
    result["ETX_STAGE_ENTRY"] = stage.entry_point;
  }
  return result;
}

bool gpu_integrator_feature_enabled(uint32_t features, uint32_t feature) {
  return (features & feature) != 0u;
}

bool wavefront_stage_enabled(GPURaytracingRenderer::PipelineStage stage, GPUIntegratorMode mode, uint32_t features, uint32_t material_compile_mask) {
  const bool enable_camera_path = gpu_integrator_feature_enabled(features, GPUIntegratorFeatures::CameraPath);
  const bool enable_light_path = gpu_integrator_feature_enabled(features, GPUIntegratorFeatures::LightPath);
  const bool enable_direct_hit = gpu_integrator_feature_enabled(features, GPUIntegratorFeatures::DirectHit);
  const bool enable_connect_to_light = gpu_integrator_feature_enabled(features, GPUIntegratorFeatures::ConnectToLight);
  const bool enable_connect_to_camera = gpu_integrator_feature_enabled(features, GPUIntegratorFeatures::ConnectToCamera);
  const bool enable_connect_vertices = gpu_integrator_feature_enabled(features, GPUIntegratorFeatures::ConnectVertices);
  const bool enable_merge_vertices = gpu_integrator_feature_enabled(features, GPUIntegratorFeatures::MergeVertices);
  const bool has_various_continue = material_compile_mask_has_various_continue(material_compile_mask);
  const bool has_various_connect = material_compile_mask_has_various_connect(material_compile_mask);
  const bool has_plastic = material_compile_mask_has(material_compile_mask, MaterialClass::Plastic);
  const bool has_conductor = material_compile_mask_has_conductor_stage(material_compile_mask);
  const bool has_dielectric = material_compile_mask_has(material_compile_mask, MaterialClass::Dielectric);
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
      return enable_connect_to_light && has_conductor;
    case GPURaytracingRenderer::PipelineStage::CameraDirectLightPrepareDielectric:
      return enable_connect_to_light && has_dielectric;
    case GPURaytracingRenderer::PipelineStage::CameraConnectLightClear:
      return enable_connect_vertices;
    case GPURaytracingRenderer::PipelineStage::CameraConnectLightPrepareDiffuse:
      return enable_connect_vertices && has_various_connect;
    case GPURaytracingRenderer::PipelineStage::CameraConnectLightPreparePlastic:
      return enable_connect_vertices && has_plastic;
    case GPURaytracingRenderer::PipelineStage::CameraConnectLightPrepareConductor:
      return enable_connect_vertices && has_conductor;
    case GPURaytracingRenderer::PipelineStage::CameraConnectLightPrepareDielectric:
      return enable_connect_vertices && has_dielectric;
    case GPURaytracingRenderer::PipelineStage::CameraConnectLightResolveDiffuse:
      return enable_connect_vertices && has_various_connect;
    case GPURaytracingRenderer::PipelineStage::CameraConnectLightResolvePlastic:
      return enable_connect_vertices && has_plastic;
    case GPURaytracingRenderer::PipelineStage::CameraConnectLightResolveConductor:
      return enable_connect_vertices && has_conductor;
    case GPURaytracingRenderer::PipelineStage::CameraConnectLightResolveDielectric:
      return enable_connect_vertices && has_dielectric;
    case GPURaytracingRenderer::PipelineStage::CameraConnectLightShadow:
      return enable_connect_vertices;
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
      return enable_connect_to_camera && has_conductor;
    case GPURaytracingRenderer::PipelineStage::LightConnectCameraPrepareDielectric:
      return enable_connect_to_camera && has_dielectric;
    case GPURaytracingRenderer::PipelineStage::VCMGridClear:
    case GPURaytracingRenderer::PipelineStage::VCMGridBuild:
      return (mode == GPUIntegratorMode::VCM) && enable_merge_vertices;
    case GPURaytracingRenderer::PipelineStage::VCMMergeDiffuse:
      return (mode == GPUIntegratorMode::VCM) && enable_merge_vertices && has_various_connect;
    case GPURaytracingRenderer::PipelineStage::VCMMergePlastic:
      return (mode == GPUIntegratorMode::VCM) && enable_merge_vertices && has_plastic;
    case GPURaytracingRenderer::PipelineStage::VCMMergeConductor:
      return (mode == GPUIntegratorMode::VCM) && enable_merge_vertices && has_conductor;
    case GPURaytracingRenderer::PipelineStage::VCMMergeDielectric:
      return (mode == GPUIntegratorMode::VCM) && enable_merge_vertices && has_dielectric;
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
  _wavefront_tile_base_origin = {};
  _wavefront_tile_base_size = {};
  _wavefront_tile_plan_valid = false;
  return true;
}

void GPURaytracingRenderer::set_wavefront_steps_per_render(uint32_t value) {
  _wavefront_auto_tuning_enabled = false;
  _wavefront_steps_per_render = std::clamp(value, 1u, kWavefrontAutoMaximumSteps);
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
  _wavefront_tile_base_origin = {};
  _wavefront_tile_base_size = {};
  _wavefront_tile_plan_valid = false;
}

void GPURaytracingRenderer::reset_runtime_failure() {
  _runtime_failed = false;
  _runtime_failure_reason.clear();
}

void GPURaytracingRenderer::set_runtime_failure(std::string message) {
  if ((_runtime_failed == false) || (_runtime_failure_reason != message)) {
    _runtime_failure_reason = std::move(message);
  }
  _runtime_failed = true;
  set_preparation_failed(_runtime_failure_reason, "Failed");
}

void GPURaytracingRenderer::init(RHIContext& ctx, SceneRepresentation& scene) {
  ETX_PROFILER_SCOPE();

  Renderer::init(ctx, scene);

  _cleanup_wait_succeeded = false;
  reset_runtime_failure();
  _backend = ctx.device().backend();
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

  if ((_preparation_state == RendererPreparationState::Failed) || _runtime_failed) {
    result.state = RendererStatusState::Failed;
    return result;
  }

  if (_preparation_state == RendererPreparationState::Preparing) {
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

  result.progress_kind = RendererProgressKind::Samples;
  result.completed_units = _sample_index;
  result.total_units = _last_target_samples;
  result.elapsed_seconds = _last_render_elapsed_seconds;
  if (_render_timing_active) {
    const auto now = std::chrono::steady_clock::now();
    result.elapsed_seconds = std::chrono::duration<double>(now - _render_started_at).count();
  }
  result.elapsed_available = _render_timing_active || (result.elapsed_seconds > 0.0);
  const double tile_count = static_cast<double>(std::max(1u, _wavefront_tile_count));
  const double tiled_sample_progress = (_wavefront_tile_plan_valid || (_wavefront_tile_count > 1u)) ? (static_cast<double>(_wavefront_tile_index) / tile_count) : 0.0;
  const double completed_sample_count = static_cast<double>(_sample_index) + tiled_sample_progress;
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
  const uint64_t shading_bytes = _materials_buffer_size + _spectrums_buffer_size + _energy_compensation_interfaces_buffer_size + _emitter_profiles_buffer_size +
                                 _emitter_instances_buffer_size + _emitters_distribution_buffer_size;
  const uint64_t scene_constants_bytes = _scene_globals_buffer_size + _scene_options_buffer_size + _camera_buffer_size + _wavefront_resources_buffer_size;
  const uint64_t path_state_bytes = _camera_state_buffer_size + _light_state_buffer_size + _camera_hit_buffer_size + _light_hit_buffer_size;
  const uint64_t queue_bytes = _camera_queue_a_buffer_size + _camera_queue_b_buffer_size + _light_queue_a_buffer_size + _light_queue_b_buffer_size + _material_queue_buffer_size +
                               _shadow_queue_buffer_size + _wavefront_dispatch_args_buffer_size;
  const uint64_t direct_light_bytes = _direct_light_sample_buffer_size + _direct_light_task_buffer_size + _direct_light_result_buffer_size;
  const uint64_t connection_bytes = _connect_light_task_buffer_size + _connect_camera_task_buffer_size + _connect_camera_result_buffer_size;
  const uint64_t subsurface_bytes = _camera_subsurface_state_buffer_size + _light_subsurface_state_buffer_size;
  const uint64_t readback_bytes = _camera_queue_count_readback_buffer_size + _light_queue_count_readback_buffer_size + _light_vertex_counter_readback_buffer_size;
  const uint64_t output_bytes = static_cast<uint64_t>(_output_dimensions.x) * static_cast<uint64_t>(_output_dimensions.y) * sizeof(float4);

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

  result.wavefront_path_capacity = _wavefront_path_capacity;
  result.light_vertex_capacity = _wavefront_resources.light_vertex_capacity;
  result.light_vertex_count = std::min(_wavefront_light_vertex_reserved_count, result.light_vertex_capacity);
  result.tile_index = _wavefront_tile_index;
  result.tile_count = _wavefront_tile_count;
  result.max_path_length = _wavefront_resources.max_path_length;
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
  _preview_visible = false;
  _display_output_valid = false;
}

void GPURaytracingRenderer::start() {
  if ((_initialized == false) || (_scene_valid == false) || _runtime_failed) {
    return;
  }

  _preparation_canceled = false;
  reset_render_progress();
  reset_preview_state();
  invalidate_output();
  _run_state = RunState::Running;
  request_scene_update();
}

void GPURaytracingRenderer::reset_render_timing() {
  _render_started_at = {};
  _last_render_elapsed_seconds = 0.0;
  _render_timing_active = false;
}

void GPURaytracingRenderer::reset_render_progress() {
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
  _wavefront_tile_base_origin = {};
  _wavefront_tile_base_size = {};
  _wavefront_tile_plan_valid = false;
  _wavefront_camera_phase_initialized = false;
  reset_wavefront_auto_tuning();
}

void GPURaytracingRenderer::reset_wavefront_auto_tuning() {
  _wavefront_last_batch_ms = 0.0;
  _wavefront_smoothed_ms_per_step = 0.0;
  if (_wavefront_auto_tuning_enabled) {
    _wavefront_steps_per_render = kWavefrontAutoInitialSteps;
  }
}

void GPURaytracingRenderer::update_wavefront_auto_tuning(uint32_t executed_steps, double elapsed_ms, bool budget_consumed, bool measurement_valid) {
  if ((_wavefront_auto_tuning_enabled == false) || (measurement_valid == false) || (executed_steps == 0u) || (elapsed_ms <= 0.0)) {
    return;
  }

  _wavefront_last_batch_ms = elapsed_ms;
  if ((budget_consumed == false) && (elapsed_ms <= kWavefrontAutoUpperDeadZoneMs)) {
    return;
  }

  const double measured_ms_per_step = elapsed_ms / static_cast<double>(executed_steps);
  if (_wavefront_smoothed_ms_per_step == 0.0) {
    _wavefront_smoothed_ms_per_step = measured_ms_per_step;
  } else {
    _wavefront_smoothed_ms_per_step += kWavefrontAutoSmoothingFactor * (measured_ms_per_step - _wavefront_smoothed_ms_per_step);
  }

  if (elapsed_ms > kWavefrontAutoUpperDeadZoneMs) {
    const double scaled_steps = static_cast<double>(executed_steps) * kWavefrontAutoTargetMs / elapsed_ms;
    uint32_t target_steps = static_cast<uint32_t>(std::clamp(std::floor(scaled_steps), 1.0, static_cast<double>(kWavefrontAutoMaximumSteps)));
    if ((target_steps >= _wavefront_steps_per_render) && (_wavefront_steps_per_render > 1u)) {
      target_steps = _wavefront_steps_per_render - 1u;
    }
    _wavefront_steps_per_render = target_steps;
    return;
  }

  if ((budget_consumed == false) || (elapsed_ms >= kWavefrontAutoLowerDeadZoneMs)) {
    return;
  }

  const uint32_t estimated_steps =
    static_cast<uint32_t>(std::clamp(std::floor(kWavefrontAutoTargetMs / _wavefront_smoothed_ms_per_step), 1.0, static_cast<double>(kWavefrontAutoMaximumSteps)));
  const uint32_t maximum_growth =
    static_cast<uint32_t>(std::min<uint64_t>(kWavefrontAutoMaximumSteps, static_cast<uint64_t>(_wavefront_steps_per_render) * kWavefrontAutoMaximumGrowthFactor));
  _wavefront_steps_per_render = std::max(_wavefront_steps_per_render, std::min(estimated_steps, maximum_growth));
}

void GPURaytracingRenderer::stop_render_timing() {
  if (_render_timing_active) {
    _last_render_elapsed_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - _render_started_at).count();
    _render_timing_active = false;
  }
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

bool GPURaytracingRenderer::ensure_preview_texture(RHIDevice& device, const uint2& dimensions) {
  if (_preview_texture.valid() && (_preview_texture_dimensions.x == dimensions.x) && (_preview_texture_dimensions.y == dimensions.y)) {
    return true;
  }

  _preview_visible = false;
  if (_preview_texture.valid()) {
    device.destroy_texture(_preview_texture);
    _preview_texture = {};
  }

  RHITextureDesc desc = {};
  desc.width = dimensions.x;
  desc.height = dimensions.y;
  desc.format = RHITextureFormat::R16G16B16A16_FLOAT;
  desc.usage = RHITextureUsage::Storage | RHITextureUsage::Sampled;
  const RHICreateBindlessResult result = device.create_texture(desc);
  if ((result.result != RHIResult::Success) || (result.handle.valid() == false)) {
    log::error("GPU RT: failed to create preview texture (%u)", static_cast<uint32_t>(result.result));
    _preview_texture_dimensions = {};
    _preview_texture_state = RHIResourceState::Undefined;
    return false;
  }

  _preview_texture = result.handle;
  _preview_texture_dimensions = dimensions;
  _preview_texture_state = RHIResourceState::Undefined;
  return true;
}

bool GPURaytracingRenderer::update_preview_camera_buffer(RHIDevice& device, const Camera& camera, const uint2& dimensions, uint32_t frame_index, uint32_t& descriptor_index) {
  ETX_CRITICAL(frame_index < kRHIMaxFrames);
  Camera preview_camera = camera;
  build_camera(preview_camera, camera.position, camera.direction, camera.up, dimensions, get_camera_fov(camera));
  const bool update_success = update_host_visible_buffer(device, &preview_camera, sizeof(preview_camera), RHIBufferUsage::Storage, _preview_camera_buffers[frame_index],
    _preview_camera_buffer_sizes[frame_index], "camera");
  if (update_success) {
    descriptor_index = get_bindless_descriptor_index(_preview_camera_buffers[frame_index]);
  }
  return update_success;
}

void GPURaytracingRenderer::destroy_preview_resources(RHIDevice& device) {
  if (_preview_texture.valid()) {
    device.destroy_texture(_preview_texture);
    _preview_texture = {};
  }
  _preview_texture_dimensions = {};
  _preview_texture_state = RHIResourceState::Undefined;
  for (uint32_t frame_index = 0u; frame_index < kRHIMaxFrames; ++frame_index) {
    device.destroy_buffer(_preview_camera_buffers[frame_index]);
    _preview_camera_buffers[frame_index] = {};
    _preview_camera_buffer_sizes[frame_index] = 0u;
  }
  _preview_visible = false;
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
    if ((result->requested_stage_mask & pipeline_stage_bit(stage_info.stage)) == 0u) {
      continue;
    }
    if ((result->compile_stage_filter.empty() == false) && (result->compile_stage_filter != stage_info.entry_point)) {
      continue;
    }
    if (result->compile_stage_filter.empty() &&
        (wavefront_stage_enabled(stage_info.stage, static_cast<GPUIntegratorMode>(result->integrator_mode), result->integrator_features, result->material_compile_mask) == false)) {
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
  for (const auto& group : compile_groups) {
    if (group.stages.size() == 1u) {
      log::info("Compiled %s - %.2fms", group.stages[0]->entry_point, group.compile_time_ms);
    } else {
      log::info("Compiled %zu stages from %s - %.2fms", group.stages.size(), group.source_file.c_str(), group.compile_time_ms);
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
  uint64_t requested_stage_mask = 0u;
  uint32_t requested_stage_count = 0u;
  for (const auto& stage_info : kWavefrontStages) {
    const bool filter_matches = (_compile_stage_filter.empty() == false) && (_compile_stage_filter == stage_info.entry_point);
    const bool stage_enabled = wavefront_stage_enabled(stage_info.stage, integrator_selection.mode, integrator_selection.features, material_compile_mask);
    if (((_compile_stage_filter.empty() == false) && (filter_matches == false)) || (_compile_stage_filter.empty() && (stage_enabled == false))) {
      continue;
    }

    const uint32_t stage_index = static_cast<uint32_t>(stage_info.stage);
    const uint64_t variant_key = wavefront_stage_variant_key(stage_info, integrator_selection.mode, material_compile_mask, spectral_mode);
    const bool stage_requires_preparation = force_reload || filter_matches || (_pipelines[stage_index].valid() == false) || (_pipeline_variant_keys[stage_index] != variant_key);
    if (stage_requires_preparation) {
      requested_stage_mask |= pipeline_stage_bit(stage_info.stage);
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
  reset_render_progress();
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
  _active_preparation->requested_stage_mask = requested_stage_mask;
  _active_preparation->compile_stage_filter = _compile_stage_filter;
  _active_preparation->queued_at = std::chrono::steady_clock::now();
  _publish_preparation.reset();
  _published_pipeline_count = 0u;
  _publish_pipeline_index = 0u;
  _pipeline_publish_logged = false;
  _preparation_started_at = _active_preparation->queued_at;
  set_preparation_state(RendererPreparationState::Preparing, "Compiling shaders", reason ? (std::string("Queued (") + reason + ")") : std::string("Queued"));
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
    if (wavefront_stage_enabled(stage_info.stage, static_cast<GPUIntegratorMode>(_integrator_mode), _integrator_features, _material_compile_mask) == false) {
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
  stop_render_timing();
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

void GPURaytracingRenderer::restart() {
  start();
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
  _wavefront_render_step = WavefrontRenderStep::InitSample;
  _wavefront_path_iteration = 0u;
  _wavefront_hard_iteration_cap = 0u;
  _wavefront_camera_queue_count = 0u;
  _wavefront_light_queue_count = 0u;
  _wavefront_light_max_path_length = 0u;
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
  _wavefront_tile_base_origin = {};
  _wavefront_tile_base_size = {};
  _wavefront_tile_plan_valid = false;
  _wavefront_camera_phase_initialized = false;
  _wavefront_dispatch_args_buffer_state = RHIResourceState::Undefined;
  _camera_queue_count_readback_state = RHIResourceState::Undefined;
  _light_queue_count_readback_state = RHIResourceState::Undefined;
  _light_vertex_counter_readback_state = RHIResourceState::Undefined;
}

bool GPURaytracingRenderer::ensure_wavefront_buffers(RHIContext& ctx, const SceneRepresentation& scene, uint32_t path_capacity, uint32_t active_path_capacity,
  bool allow_light_history_shrink) {
  ETX_PROFILER_SCOPE();

  const uint2 film_size = scene.camera().film_size;
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

  const bool enable_camera_path = gpu_integrator_feature_enabled(_integrator_features, GPUIntegratorFeatures::CameraPath);
  const bool enable_light_path = gpu_integrator_feature_enabled(_integrator_features, GPUIntegratorFeatures::LightPath);
  const bool enable_connect_to_light = gpu_integrator_feature_enabled(_integrator_features, GPUIntegratorFeatures::ConnectToLight);
  const bool enable_connect_to_camera = gpu_integrator_feature_enabled(_integrator_features, GPUIntegratorFeatures::ConnectToCamera);
  const bool enable_connect_vertices = gpu_integrator_feature_enabled(_integrator_features, GPUIntegratorFeatures::ConnectVertices);
  const bool enable_merge_vertices = gpu_integrator_feature_enabled(_integrator_features, GPUIntegratorFeatures::MergeVertices);
  const bool store_complete_light_history = enable_connect_vertices || enable_merge_vertices;
  bool scene_has_subsurface_material = false;
  for (const auto& material : scene.data().materials) {
    if (material.subsurface_cls != SubsurfaceMaterial::Disabled) {
      scene_has_subsurface_material = true;
      break;
    }
  }
  const bool enable_subsurface_state_buffers = scene_has_subsurface_material;
  const uint32_t scene_max_path_length = std::max(1u, scene.data().options.max_path_length);
  const uint32_t camera_history_bounces = enable_camera_path ? kWavefrontRollingHistoryBounces : 0u;
  const uint32_t initial_light_history_bounces = wavefront_initial_light_history_bounces(scene_max_path_length);
  const uint32_t retained_light_history_bounces = std::min(scene_max_path_length, std::max(initial_light_history_bounces, _wavefront_light_history_capacity_bounces));
  const bool use_fast_light_endpoints = enable_light_path && (static_cast<GPUIntegratorMode>(_integrator_mode) == GPUIntegratorMode::BDPTFast);
  const uint32_t rolling_light_history_bounces = use_fast_light_endpoints ? kWavefrontFastLightHistoryBounces : kWavefrontLightHistoryBounces;
  const uint32_t light_history_bounces = enable_light_path ? (store_complete_light_history ? retained_light_history_bounces : rolling_light_history_bounces) : 0u;
  const bool compact_light_history = enable_light_path && store_complete_light_history;
  const uint32_t wavefront_hard_iteration_cap = scene_max_path_length;
  const uint64_t camera_vertex_capacity_u64 = static_cast<uint64_t>(path_capacity) * static_cast<uint64_t>(camera_history_bounces + 1u);
  if (camera_vertex_capacity_u64 > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
    log::error("GPU RT: wavefront camera vertex capacity overflow");
    return false;
  }
  uint64_t light_vertex_capacity_u64 = static_cast<uint64_t>(path_capacity) * static_cast<uint64_t>(light_history_bounces + 1u);
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
                                            (kGPUWavefrontQueueHeaderSize + static_cast<uint64_t>(path_capacity) * kWavefrontConnectLightBatchSize * sizeof(uint32_t)) +
                                            (kGPUWavefrontQueueHeaderSize + static_cast<uint64_t>(path_capacity) * sizeof(uint32_t));
  const uint32_t heavy_continuation_chunk_count = 1u + ((path_capacity - 1u) / kWavefrontHeavyContinuationChunkSize);
  const uint64_t dispatch_args_buffer_size =
    kGPUWavefrontFixedDispatchArgsBufferSize + static_cast<uint64_t>(heavy_continuation_chunk_count) * 2u * kGPUWavefrontDispatchArgsStride;
  const uint64_t path_state_buffer_size = static_cast<uint64_t>(path_capacity) * kGPUWavefrontPathStateStride;
  const uint64_t hit_buffer_size = static_cast<uint64_t>(path_capacity) * kGPUWavefrontHitStride;
  const uint64_t camera_vertex_buffer_size = static_cast<uint64_t>(camera_vertex_capacity) * kGPUWavefrontPathVertexStride;
  const uint64_t light_vertex_buffer_size = static_cast<uint64_t>(light_vertex_capacity) * kGPUWavefrontLightPathVertexStride;
  const uint32_t vcm_grid_head_count = enable_merge_vertices ? wavefront_vcm_grid_head_count(light_vertex_capacity) : 0u;
  const uint64_t vcm_grid_heads_buffer_size = static_cast<uint64_t>(vcm_grid_head_count) * sizeof(uint32_t);
  const uint64_t vcm_grid_next_buffer_size = enable_merge_vertices ? static_cast<uint64_t>(light_vertex_capacity) * sizeof(uint32_t) : 0u;
  const uint64_t fast_light_endpoint_buffer_size = use_fast_light_endpoints ? static_cast<uint64_t>(path_capacity) * kGPUWavefrontFastLightEndpointStride : 0u;
  const uint64_t film_buffer_size = film_pixel_count_u64 * sizeof(float4);
  const uint64_t path_meta_buffer_size = static_cast<uint64_t>(path_capacity) * kGPUWavefrontPathMetaStride;
  const uint64_t direct_light_sample_buffer_size = static_cast<uint64_t>(path_capacity) * kGPUWavefrontDirectLightSampleStride;
  const uint64_t direct_light_task_buffer_size = static_cast<uint64_t>(path_capacity) * kGPUWavefrontDirectLightTaskStride;
  const uint64_t direct_light_work_buffer_size = std::max(direct_light_sample_buffer_size, direct_light_task_buffer_size);
  const uint64_t direct_light_result_buffer_size = static_cast<uint64_t>(path_capacity) * kGPUWavefrontDirectLightResultStride;
  const uint64_t connect_light_task_count = static_cast<uint64_t>(path_capacity) * static_cast<uint64_t>(kWavefrontConnectLightBatchSize);
  const uint64_t connect_light_task_buffer_size = connect_light_task_count * kGPUWavefrontConnectLightTaskStride;
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
  if (enable_camera_path) {
    wavefront_buffer_sizes_valid = wavefront_buffer_sizes_valid && validate_wavefront_buffer_size("wavefront_camera_state", path_state_buffer_size) &&
                                   validate_wavefront_buffer_size("wavefront_camera_hit", hit_buffer_size) &&
                                   validate_wavefront_buffer_size("wavefront_camera_queue", queue_buffer_size) &&
                                   validate_wavefront_buffer_size("wavefront_camera_vertex", camera_vertex_buffer_size);
  }
  if (enable_light_path) {
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
  if (enable_connect_to_camera) {
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
  if (enable_camera_path) {
    if (ensure_storage_buffer(device, path_state_buffer_size, wavefront_usage, _camera_state_buffer, _camera_state_buffer_size, _camera_state_buffer_descriptor_index,
          "wavefront_camera_state") == false) {
      return false;
    }
  } else {
    destroy_linear_scene_buffer(device, _camera_state_buffer, _camera_state_buffer_size, _camera_state_buffer_descriptor_index);
  }
  if (enable_light_path) {
    if (ensure_storage_buffer(device, path_state_buffer_size, wavefront_usage, _light_state_buffer, _light_state_buffer_size, _light_state_buffer_descriptor_index,
          "wavefront_light_state") == false) {
      return false;
    }
  } else {
    destroy_linear_scene_buffer(device, _light_state_buffer, _light_state_buffer_size, _light_state_buffer_descriptor_index);
  }
  if (enable_camera_path) {
    if (ensure_storage_buffer(device, hit_buffer_size, wavefront_usage, _camera_hit_buffer, _camera_hit_buffer_size, _camera_hit_buffer_descriptor_index, "wavefront_camera_hit") ==
        false) {
      return false;
    }
  } else {
    destroy_linear_scene_buffer(device, _camera_hit_buffer, _camera_hit_buffer_size, _camera_hit_buffer_descriptor_index);
  }
  if (enable_light_path) {
    if (ensure_storage_buffer(device, hit_buffer_size, wavefront_usage, _light_hit_buffer, _light_hit_buffer_size, _light_hit_buffer_descriptor_index, "wavefront_light_hit") ==
        false) {
      return false;
    }
  } else {
    destroy_linear_scene_buffer(device, _light_hit_buffer, _light_hit_buffer_size, _light_hit_buffer_descriptor_index);
  }
  if (enable_camera_path) {
    if (ensure_storage_buffer(device, queue_buffer_size, queue_buffer_usage, _camera_queue_a_buffer, _camera_queue_a_buffer_size, _camera_queue_a_buffer_descriptor_index,
          "wavefront_camera_queue_a") == false) {
      return false;
    }
  } else {
    destroy_linear_scene_buffer(device, _camera_queue_a_buffer, _camera_queue_a_buffer_size, _camera_queue_a_buffer_descriptor_index);
  }
  if (enable_camera_path) {
    if (ensure_storage_buffer(device, queue_buffer_size, queue_buffer_usage, _camera_queue_b_buffer, _camera_queue_b_buffer_size, _camera_queue_b_buffer_descriptor_index,
          "wavefront_camera_queue_b") == false) {
      return false;
    }
  } else {
    destroy_linear_scene_buffer(device, _camera_queue_b_buffer, _camera_queue_b_buffer_size, _camera_queue_b_buffer_descriptor_index);
  }
  if (enable_light_path) {
    if (ensure_storage_buffer(device, queue_buffer_size, queue_buffer_usage, _light_queue_a_buffer, _light_queue_a_buffer_size, _light_queue_a_buffer_descriptor_index,
          "wavefront_light_queue_a") == false) {
      return false;
    }
  } else {
    destroy_linear_scene_buffer(device, _light_queue_a_buffer, _light_queue_a_buffer_size, _light_queue_a_buffer_descriptor_index);
  }
  if (enable_light_path) {
    if (ensure_storage_buffer(device, queue_buffer_size, queue_buffer_usage, _light_queue_b_buffer, _light_queue_b_buffer_size, _light_queue_b_buffer_descriptor_index,
          "wavefront_light_queue_b") == false) {
      return false;
    }
  } else {
    destroy_linear_scene_buffer(device, _light_queue_b_buffer, _light_queue_b_buffer_size, _light_queue_b_buffer_descriptor_index);
  }
  if (ensure_storage_buffer(device, material_queue_buffer_size, wavefront_usage, _material_queue_buffer, _material_queue_buffer_size, _material_queue_buffer_descriptor_index,
        "wavefront_material_queue") == false) {
    return false;
  }
  if (ensure_storage_buffer(device, shadow_queue_buffer_size, wavefront_usage, _shadow_queue_buffer, _shadow_queue_buffer_size, _shadow_queue_buffer_descriptor_index,
        "wavefront_shadow_queue") == false) {
    return false;
  }
  const bool recreate_dispatch_args_buffer = (_wavefront_dispatch_args_buffer.valid() == false) || (_wavefront_dispatch_args_buffer_size != dispatch_args_buffer_size);
  if (ensure_storage_buffer(device, dispatch_args_buffer_size, dispatch_args_buffer_usage, _wavefront_dispatch_args_buffer, _wavefront_dispatch_args_buffer_size,
        _wavefront_dispatch_args_buffer_descriptor_index, "wavefront_dispatch_args") == false) {
    return false;
  }
  if (recreate_dispatch_args_buffer) {
    _wavefront_dispatch_args_buffer_state = RHIResourceState::Undefined;
  }
  if (enable_camera_path) {
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
  if (enable_light_path) {
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
  if (enable_camera_path) {
    if (ensure_storage_buffer(device, camera_vertex_buffer_size, wavefront_usage, _camera_vertex_buffer, _camera_vertex_buffer_size, _camera_vertex_buffer_descriptor_index,
          "wavefront_camera_vertex") == false) {
      return false;
    }
  } else {
    destroy_linear_scene_buffer(device, _camera_vertex_buffer, _camera_vertex_buffer_size, _camera_vertex_buffer_descriptor_index);
  }
  if (enable_light_path) {
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
    if (ensure_storage_buffer(device, fast_light_endpoint_buffer_size, wavefront_usage, _fast_light_endpoint_buffer, _fast_light_endpoint_buffer_size,
          _fast_light_endpoint_buffer_descriptor_index, "wavefront_fast_light_endpoint") == false) {
      return false;
    }
  } else {
    destroy_linear_scene_buffer(device, _fast_light_endpoint_buffer, _fast_light_endpoint_buffer_size, _fast_light_endpoint_buffer_descriptor_index);
  }
  if (ensure_storage_buffer(device, film_buffer_size, wavefront_usage, _film_buffer, _film_buffer_size, _film_buffer_descriptor_index, "wavefront_film") == false) {
    return false;
  }
  if (ensure_storage_buffer(device, path_meta_buffer_size, wavefront_usage, _path_meta_buffer, _path_meta_buffer_size, _path_meta_buffer_descriptor_index, "wavefront_path_meta") ==
      false) {
    return false;
  }
  if (enable_connect_to_light) {
    if (ensure_storage_buffer(device, direct_light_work_buffer_size, wavefront_usage, _direct_light_sample_buffer, _direct_light_sample_buffer_size,
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
    if (ensure_storage_buffer(device, direct_light_result_buffer_size, wavefront_usage, _direct_light_result_buffer, _direct_light_result_buffer_size,
          _direct_light_result_buffer_descriptor_index, "wavefront_direct_light_result") == false) {
      return false;
    }
  } else {
    destroy_linear_scene_buffer(device, _direct_light_result_buffer, _direct_light_result_buffer_size, _direct_light_result_buffer_descriptor_index);
  }
  if (enable_connect_vertices) {
    if (ensure_storage_buffer(device, connect_light_task_buffer_size, wavefront_usage, _connect_light_task_buffer, _connect_light_task_buffer_size,
          _connect_light_task_buffer_descriptor_index, "wavefront_connect_light_task") == false) {
      return false;
    }
  } else {
    destroy_linear_scene_buffer(device, _connect_light_task_buffer, _connect_light_task_buffer_size, _connect_light_task_buffer_descriptor_index);
  }
  if (enable_connect_to_camera) {
    if (enable_connect_to_light) {
      destroy_linear_scene_buffer(device, _connect_camera_task_buffer, _connect_camera_task_buffer_size, _connect_camera_task_buffer_descriptor_index);
      destroy_linear_scene_buffer(device, _connect_camera_result_buffer, _connect_camera_result_buffer_size, _connect_camera_result_buffer_descriptor_index);
      _connect_camera_task_buffer_descriptor_index = _direct_light_sample_buffer_descriptor_index;
      _connect_camera_result_buffer_descriptor_index = _direct_light_result_buffer_descriptor_index;
    } else {
      if (ensure_storage_buffer(device, connect_camera_task_buffer_size, wavefront_usage, _connect_camera_task_buffer, _connect_camera_task_buffer_size,
            _connect_camera_task_buffer_descriptor_index, "wavefront_connect_camera_task") == false) {
        return false;
      }
      if (ensure_storage_buffer(device, connect_camera_result_buffer_size, wavefront_usage, _connect_camera_result_buffer, _connect_camera_result_buffer_size,
            _connect_camera_result_buffer_descriptor_index, "wavefront_connect_camera_result") == false) {
        return false;
      }
    }
  } else {
    destroy_linear_scene_buffer(device, _connect_camera_task_buffer, _connect_camera_task_buffer_size, _connect_camera_task_buffer_descriptor_index);
    destroy_linear_scene_buffer(device, _connect_camera_result_buffer, _connect_camera_result_buffer_size, _connect_camera_result_buffer_descriptor_index);
  }
  if (enable_camera_path && enable_subsurface_state_buffers) {
    if (ensure_storage_buffer(device, subsurface_state_buffer_size, wavefront_usage, _camera_subsurface_state_buffer, _camera_subsurface_state_buffer_size,
          _camera_subsurface_state_buffer_descriptor_index, "wavefront_camera_subsurface_state") == false) {
      return false;
    }
  } else {
    destroy_linear_scene_buffer(device, _camera_subsurface_state_buffer, _camera_subsurface_state_buffer_size, _camera_subsurface_state_buffer_descriptor_index);
  }
  if (enable_light_path && enable_subsurface_state_buffers) {
    if (ensure_storage_buffer(device, subsurface_state_buffer_size, wavefront_usage, _light_subsurface_state_buffer, _light_subsurface_state_buffer_size,
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
  _wavefront_light_history_capacity_bounces = enable_light_path ? ((light_vertex_capacity / path_capacity) - 1u) : 0u;
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
    const uint64_t grid_heads_size = static_cast<uint64_t>(wavefront_vcm_grid_head_count(updated_resources.light_vertex_capacity)) * sizeof(uint32_t);
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
  const auto render_begin = std::chrono::steady_clock::now();
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

  update_camera(scene, frame_data.dt);

  if (_initialized == false)
    return;

  _scene_valid = scene.valid();
  if (_scene_valid == false)
    return;

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
  _last_target_samples = std::max(1u, scene.data().options.samples);
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

  auto& device = ctx.device();
  _last_memory_stats = device.get_memory_statistics();
  const bool render_preview_this_frame = _preview_active;

  const SceneUpdateScope scene_update_scope = consume_scene_update_request();
  const bool scene_check_requested = scene_update_scope != SceneUpdateScope::None;

  SceneHashes new_hashes = _current_scene_hashes;
  UpdateFlags changes = {};
  bool scene_changed = false;
  if (scene_check_requested) {
    ETX_PROFILER_NAMED_SCOPE("gpu_rt_scene_hashes_and_changes");
    const auto scene_hash_begin = std::chrono::steady_clock::now();
    if (scene_update_scope == SceneUpdateScope::Full) {
      scene.data().images.load_images(scheduler);
    }
    const bool hierarchy_resolve_required = (scene_update_scope == SceneUpdateScope::Full) || (scene.data().hierarchy.resolved_state_current() == false);
    if (hierarchy_resolve_required && (scene.data().resolve_hierarchy() == false)) {
      set_runtime_failure("Failed to resolve scene hierarchy");
      if (scene_update_scope == SceneUpdateScope::Full) {
        request_scene_update();
      } else {
        request_scene_transform_update();
      }
      return;
    }
    if (scene_update_scope == SceneUpdateScope::Full) {
      new_hashes = scene.data().compute_hashes();
    } else {
      new_hashes.transforms_hash = scene.data().compute_transforms_hash();
    }
    changes = new_hashes.compare(_current_scene_hashes);
    scene_changed = changes.any();
    const auto scene_hash_end = std::chrono::steady_clock::now();
    scene_hash_ms = elapsed_ms(scene_hash_begin, scene_hash_end);
  }

  const auto& camera = scene.camera();
  const uint64_t new_camera_hash = xxh64(&camera, sizeof(camera));
  const bool camera_changed = (new_camera_hash != _current_camera_hash);
  const bool restart_accumulation = scene_changed || integrator_mode_changed || integrator_features_changed || material_compile_mask_changed || camera_changed;
  if (restart_accumulation) {
    reset_render_progress();
    if (_run_state == RunState::Completed) {
      _run_state = RunState::Running;
    }
  }

  const bool wavefront_sample_in_progress = _wavefront_render_step != WavefrontRenderStep::InitSample;

  const bool geometry_structure_changed = changes[UpdateFlags::AnyGeometryStructure];
  bool needs_full_rebuild = scene_check_requested && geometry_structure_changed;
  const bool needs_scene_data_reupload = scene_check_requested && scene_changed && (geometry_structure_changed == false);
  bool scene_data_update_success = true;

  if (needs_scene_data_reupload && (_vertex_positions_buffer.valid() == false)) {
    needs_full_rebuild = true;
  }

  if (needs_full_rebuild) {
    ETX_PROFILER_NAMED_SCOPE("gpu_rt_full_rebuild_resources");
    const auto full_rebuild_destroy_begin = std::chrono::steady_clock::now();
    destroy_wavefront_buffers(ctx);
    destroy_scene_buffers(ctx);
    destroy_acceleration_structures(ctx);
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
    if (changes[UpdateFlags::Transforms]) {
      update_success = refit_top_level_acceleration_structure(ctx, scene.data());
      if (update_success == false) {
        destroy_wavefront_buffers(ctx);
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

  if (scene_check_requested && scene_data_update_success) {
    _current_scene_hashes = new_hashes;
  }
  _current_camera_hash = new_camera_hash;

  if (integrator_mode_changed || integrator_features_changed) {
    ETX_PROFILER_NAMED_SCOPE("gpu_rt_runtime_update_scene_options");
    const auto scene_options_upload_begin = std::chrono::steady_clock::now();
    const RHIBufferUsage scene_buffer_usage = RHIBufferUsage::Storage | RHIBufferUsage::TransferDst;
    const GPUSceneOptions options = build_scene_options(scene);
    const bool options_upload_success = upload_or_update_linear_scene_buffer(device, &options, size_t(1), scene_buffer_usage, _scene_options_buffer, _scene_options_buffer_size,
      _gpu_scene.scene_options, "scene_options");
    const auto scene_options_upload_end = std::chrono::steady_clock::now();
    scene_options_upload_ms = elapsed_ms(scene_options_upload_begin, scene_options_upload_end);
    if (options_upload_success == false) {
      log::error("GPU RT: failed to upload scene options buffer for integrator change");
      return;
    }
  }

  const uint2 full_dim = camera.film_size;
  const uint32_t preview_pixel_size = _preview_resolution.pixel_size();
  const uint2 preview_dim = {
    (full_dim.x + preview_pixel_size - 1u) / preview_pixel_size,
    (full_dim.y + preview_pixel_size - 1u) / preview_pixel_size,
  };
  uint32_t frame_camera_buffer_descriptor_index = _camera_buffer_descriptor_index;
  if (render_preview_this_frame) {
    const auto camera_upload_begin = std::chrono::steady_clock::now();
    if (update_preview_camera_buffer(device, camera, preview_dim, ctx.get_current_frame_index(), frame_camera_buffer_descriptor_index) == false) {
      return;
    }
    camera_upload_ms = elapsed_ms(camera_upload_begin, std::chrono::steady_clock::now());
  } else if (wavefront_sample_in_progress == false) {
    ETX_PROFILER_NAMED_SCOPE("gpu_rt_upload_camera");
    const auto camera_upload_begin = std::chrono::steady_clock::now();
    const RHIBufferUsage camera_buffer_usage = RHIBufferUsage::Storage | RHIBufferUsage::TransferDst;
    const bool camera_upload_success =
      upload_or_update_linear_scene_buffer(device, &camera, size_t(1), camera_buffer_usage, _camera_buffer, _camera_buffer_size, _camera_buffer_descriptor_index, "camera");
    const auto camera_upload_end = std::chrono::steady_clock::now();
    camera_upload_ms = elapsed_ms(camera_upload_begin, camera_upload_end);
    if (camera_upload_success == false) {
      log::error("GPU RT: failed to upload camera buffer");
      return;
    }
  }

  if ((wavefront_sample_in_progress == false) && (render_preview_this_frame == false)) {
    const auto blue_noise_update_begin = std::chrono::steady_clock::now();
    if (update_blue_noise_buffer(ctx, scene) == false) {
      log::warning("GPU RT: blue noise buffer is unavailable, falling back to white noise");
    }
    const auto blue_noise_update_end = std::chrono::steady_clock::now();
    blue_noise_update_ms = elapsed_ms(blue_noise_update_begin, blue_noise_update_end);
  }

  const bool has_render_window = (_render_window_size.x > 0u) && (_render_window_size.y > 0u);
  const uint2 base_render_origin = has_render_window ? _render_window_origin : uint2{};
  const uint2 base_render_dim = has_render_window ? _render_window_size : full_dim;
  const uint32_t scene_max_path_length_for_tiling = std::max(1u, scene.data().options.max_path_length);
  const bool use_complete_light_history = gpu_integrator_feature_enabled(_integrator_features, GPUIntegratorFeatures::ConnectVertices) ||
                                          gpu_integrator_feature_enabled(_integrator_features, GPUIntegratorFeatures::MergeVertices);
  const bool vcm_mode = static_cast<GPUIntegratorMode>(_integrator_mode) == GPUIntegratorMode::VCM;
  // VCM's light population, normalization, and spatial grid are iteration-global.
  const bool use_wavefront_tiling = (render_preview_this_frame == false) && use_complete_light_history && (vcm_mode == false);
  const uint64_t base_render_pixel_count_u64 = static_cast<uint64_t>(base_render_dim.x) * static_cast<uint64_t>(base_render_dim.y);
  if (base_render_pixel_count_u64 > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
    log::error("GPU RT: render window path capacity overflow");
    return;
  }
  const uint32_t base_render_pixel_count = static_cast<uint32_t>(base_render_pixel_count_u64);
  if (base_render_pixel_count == 0u) {
    log::error("GPU RT: render window is empty");
    return;
  }
  uint32_t tile_max_pixels = base_render_pixel_count;
  uint32_t wavefront_tile_count_value = 1u;
  uint32_t wavefront_buffer_path_capacity = base_render_pixel_count;
  uint2 active_base_origin = base_render_origin;
  uint2 active_base_dim = base_render_dim;
  if (use_wavefront_tiling) {
    const bool should_rebuild_tile_plan = (_wavefront_tile_plan_valid == false) || ((_wavefront_render_step == WavefrontRenderStep::InitSample) && (_wavefront_tile_index == 0u));
    if (should_rebuild_tile_plan) {
      bool scene_has_subsurface_material = false;
      for (const auto& material : scene.data().materials) {
        if (material.subsurface_cls != SubsurfaceMaterial::Disabled) {
          scene_has_subsurface_material = true;
          break;
        }
      }
      const uint64_t tile_budget = wavefront_tile_budget_bytes(_last_memory_stats);
      const uint32_t initial_light_history_bounces = wavefront_initial_light_history_bounces(scene_max_path_length_for_tiling);
      const uint64_t tile_bytes_per_path = wavefront_tile_bytes_per_path(_integrator_features, scene_has_subsurface_material, initial_light_history_bounces);
      _wavefront_tile_max_pixels = wavefront_tile_max_pixels(tile_budget, tile_bytes_per_path, base_render_pixel_count);
      _wavefront_tile_count = wavefront_tile_count(base_render_dim, _wavefront_tile_max_pixels);
      _wavefront_tile_path_capacity = std::min(_wavefront_tile_max_pixels, base_render_pixel_count);
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
  const WavefrontWindow active_window = use_wavefront_tiling ? wavefront_tile_window(active_base_origin, active_base_dim, tile_max_pixels, _wavefront_tile_index)
                                                             : WavefrontWindow{base_render_origin, base_render_dim};
  const uint2 render_dim = render_preview_this_frame ? preview_dim : active_window.size;
  const uint32_t wavefront_path_capacity = render_dim.x * render_dim.y;
  if (_output_dimensions.x != full_dim.x || _output_dimensions.y != full_dim.y) {
    ETX_PROFILER_NAMED_SCOPE("gpu_rt_recreate_output_texture");
    const auto output_texture_begin = std::chrono::steady_clock::now();
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
    _wavefront_tile_base_origin = {};
    _wavefront_tile_base_size = {};
    _wavefront_tile_plan_valid = false;
    _wavefront_camera_phase_initialized = false;
    const auto output_texture_end = std::chrono::steady_clock::now();
    output_texture_ms = elapsed_ms(output_texture_begin, output_texture_end);
    return;
  }

  if (render_preview_this_frame) {
    if (ensure_preview_texture(device, preview_dim) == false) {
      return;
    }
  }
  const RHITexture render_output_texture = render_preview_this_frame ? _preview_texture : _output_texture;
  RHIResourceState& render_output_texture_state = render_preview_this_frame ? _preview_texture_state : _output_texture_state;

  GPURTConstants constants = {
    .camera_buffer_index = frame_camera_buffer_descriptor_index,
    .as_index = get_bindless_descriptor_index(_tlas),
    .output_image_index = get_bindless_descriptor_index(render_preview_this_frame ? _preview_texture : _output_texture),
    .frame_index = _frame_index,
    .sample_index = _sample_index,
    .blue_noise_buffer_index = _blue_noise_buffer_descriptor_index,
    .wavefront_buffer_index = _wavefront_resources_buffer_descriptor_index,
    .path_iteration = 0u,
    .connect_light_vertex_length = 0u,
    .render_window_origin_x = render_preview_this_frame ? 0u : active_window.origin.x,
    .render_window_origin_y = render_preview_this_frame ? 0u : active_window.origin.y,
    .render_window_width = render_dim.x,
    .render_window_height = render_dim.y,
    .dispatch_item_offset = 0u,
    .dispatch_item_count = 0u,
    .work_queue_index = kInvalidIndex,
    .vcm_light_vertex_count = _wavefront_vcm_light_vertex_count,
    .pad2 = 0u,
    .scene = _gpu_scene,
  };
  const RHIDispatchDesc film_dispatch = {
    .group_count_x = (render_dim.x + 7u) / 8u,
    .group_count_y = (render_dim.y + 7u) / 8u,
    .group_count_z = 1u,
  };

  if ((_run_state == RunState::Stopped) || (_run_state == RunState::Completed)) {
    return;
  }

  if (_sample_index >= std::max(1u, scene.data().options.samples)) {
    stop_render_timing();
    _run_state = RunState::Completed;
    return;
  }

  if (_render_timing_active == false) {
    _render_started_at = std::chrono::steady_clock::now();
    _last_render_elapsed_seconds = 0.0;
    _render_timing_active = true;
  }

  const auto wavefront_buffer_begin = std::chrono::steady_clock::now();
  if (ensure_wavefront_buffers(ctx, scene, wavefront_buffer_path_capacity, wavefront_path_capacity, false) == false) {
    if (vcm_mode) {
      set_runtime_failure("GPU VCM requires whole-frame wavefront buffers; allocation failed for the current resolution and path length");
    } else {
      log::error("GPU RT: failed to allocate wavefront buffers");
    }
    return;
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
                                                            uint32_t connect_light_vertex_length, uint32_t connect_light_vertex_count, bool reset_light_cursor) {
      GPURTConstants stage_constants = constants;
      stage_constants.path_iteration = path_iteration;
      stage_constants.connect_light_vertex_length = connect_light_vertex_length;
      stage_constants.dispatch_item_count = connect_light_vertex_count;
      stage_constants.dispatch_item_offset = reset_light_cursor ? 1u : 0u;
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
    const uint32_t heavy_continuation_chunk_count = 1u + ((wavefront_path_capacity - 1u) / kWavefrontHeavyContinuationChunkSize);
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
    const auto wait_and_destroy_submitted_commands = [&](const char* stage_name) {
      RHIResult wait_result = RHIResult::Success;
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
      constexpr size_t kKernelTimingPendingCommandLimit = 16u;
      if (capture_kernel_timings && (submitted_commands.size() >= kKernelTimingPendingCommandLimit)) {
        wait_and_destroy_submitted_commands("kernel timing batch");
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
      for (uint32_t item_offset = 0u; item_offset < item_count; item_offset += kWavefrontHeavyContinuationChunkSize) {
        const uint32_t chunk_count = std::min(kWavefrontHeavyContinuationChunkSize, item_count - item_offset);
        const uint64_t argument_buffer_offset = argument_base_offset + static_cast<uint64_t>(item_offset / kWavefrontHeavyContinuationChunkSize) * kGPUWavefrontDispatchArgsStride;
        record_and_submit([&](RHICommandBuffer cmd) {
          barrier_wavefront_buffers(cmd);
          dispatch_stage_window(cmd, stage, argument_buffer_offset, item_offset, chunk_count, path_iteration);
          barrier_wavefront_buffers(cmd);
        });
      }
    };
    const bool enable_camera_path = gpu_integrator_feature_enabled(_integrator_features, GPUIntegratorFeatures::CameraPath);
    const bool enable_light_path = gpu_integrator_feature_enabled(_integrator_features, GPUIntegratorFeatures::LightPath);
    const bool enable_direct_hit = gpu_integrator_feature_enabled(_integrator_features, GPUIntegratorFeatures::DirectHit);
    const bool enable_connect_to_light = gpu_integrator_feature_enabled(_integrator_features, GPUIntegratorFeatures::ConnectToLight);
    const bool enable_connect_to_camera = gpu_integrator_feature_enabled(_integrator_features, GPUIntegratorFeatures::ConnectToCamera);
    const bool enable_connect_vertices = gpu_integrator_feature_enabled(_integrator_features, GPUIntegratorFeatures::ConnectVertices);
    const bool enable_merge_vertices = gpu_integrator_feature_enabled(_integrator_features, GPUIntegratorFeatures::MergeVertices);
    const bool store_complete_light_history = enable_connect_vertices || enable_merge_vertices;
    const bool phase_light_before_camera =
      ((integrator_mode == GPUIntegratorMode::BDPTFull) || (integrator_mode == GPUIntegratorMode::VCM)) && enable_camera_path && enable_light_path && store_complete_light_history;
    const bool has_various_continue = material_compile_mask_has_various_continue(_material_compile_mask);
    const bool has_various_connect = material_compile_mask_has_various_connect(_material_compile_mask);
    const bool has_plastic = material_compile_mask_has(_material_compile_mask, MaterialClass::Plastic);
    const bool has_conductor = material_compile_mask_has_conductor_stage(_material_compile_mask);
    const bool has_dielectric = material_compile_mask_has(_material_compile_mask, MaterialClass::Dielectric);
    const bool has_thinfilm = material_compile_mask_has(_material_compile_mask, MaterialClass::Thinfilm);
    const bool use_material_work_queues = material_compile_mask_work_queue_count(_material_compile_mask) > 1u;
    const auto dispatch_stage_material_indirect = [&](RHICommandBuffer cmd, PipelineStage stage, bool from_camera, uint32_t material_queue_index, uint32_t path_iteration) {
      if (use_material_work_queues) {
        dispatch_stage_work_queue_indirect(cmd, stage, material_dispatch_args_offset(from_camera, material_queue_index), path_iteration, material_queue_index);
        return;
      }

      dispatch_stage_indirect(cmd, stage, from_camera ? kGPUWavefrontCameraDispatchArgsOffset : kGPUWavefrontLightDispatchArgsOffset, path_iteration);
    };
    const uint32_t light_history_bounces = store_complete_light_history ? scene_max_path_length : kWavefrontLightHistoryBounces;
    const uint32_t render_pixel_count = render_dim.x * render_dim.y;
    const uint32_t initial_camera_queue_count = enable_camera_path ? render_pixel_count : 0u;
    const uint32_t initial_light_queue_count = enable_light_path ? render_pixel_count : 0u;
    const uint32_t batch_queue_readback_interval = _batch_coarse_progress ? kWavefrontCoarseQueueReadbackInterval : 1u;
    if ((_sample_index == 0u) && (_frame_index == 0u)) {
      log::info("GPU path mode: %s", gpu_integrator_mode_to_string(integrator_mode));
    }
    const auto finalize_wavefront_sample = [&]() {
      record_and_submit([&](RHICommandBuffer cmd) {
        ctx.cmd_texture_barrier(cmd, render_output_texture, render_output_texture_state, RHIResourceState::General);
        barrier_wavefront_buffers(cmd);
        dispatch_stage(cmd, PipelineStage::FinalizeSample, film_dispatch, _wavefront_hard_iteration_cap);
        ctx.cmd_texture_barrier(cmd, render_output_texture, RHIResourceState::General, RHIResourceState::ShaderReadOnly);
      });

      const RHIResult finalize_result = wait_and_destroy_submitted_commands("finalize sample submit");
      if (finalize_result != RHIResult::Success) {
        set_runtime_failure("GPU RT finalize sample submit failed (" + std::to_string(static_cast<uint32_t>(finalize_result)) + ")");
        _wavefront_camera_queue_count = 0u;
        _wavefront_light_queue_count = 0u;
        return false;
      }
      render_output_texture_state = RHIResourceState::ShaderReadOnly;
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
      if (_wavefront_tile_index + 1u >= wavefront_tile_count_value) {
        _wavefront_tile_index = 0u;
        _wavefront_tile_plan_valid = false;
        completed_sample = true;
      } else {
        _wavefront_tile_index += 1u;
      }
      dispatch_submit_ms = elapsed_ms(dispatch_submit_begin, std::chrono::steady_clock::now());
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
          const RHIDispatchDesc grid_clear_dispatch = {
            .group_count_x = divide_round_up(constants.vcm_grid_mask + 1u, 256u),
            .group_count_y = 1u,
            .group_count_z = 1u,
          };
          const RHIDispatchDesc grid_build_dispatch = {
            .group_count_x = divide_round_up(constants.vcm_light_vertex_count, 256u),
            .group_count_y = 1u,
            .group_count_z = 1u,
          };
          dispatch_stage(cmd, PipelineStage::VCMGridClear, grid_clear_dispatch, 0u);
          barrier_wavefront_buffers(cmd);
          dispatch_stage(cmd, PipelineStage::VCMGridBuild, grid_build_dispatch, 0u);
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

    if (render_output_texture_state != RHIResourceState::General) {
      record_and_submit([&](RHICommandBuffer cmd) {
        ctx.cmd_texture_barrier(cmd, render_output_texture, render_output_texture_state, RHIResourceState::General);
      });
      render_output_texture_state = RHIResourceState::General;
    }

    bool finished_current_tile = false;
    bool wavefront_auto_measurement_valid = true;
    uint32_t executed_wavefront_steps = 0u;
    const uint32_t preview_path_phase_count = phase_light_before_camera ? 2u : 1u;
    const uint32_t preview_step_budget = scene_max_path_length * preview_path_phase_count + 4u;
    const uint32_t wavefront_step_budget = render_preview_this_frame ? preview_step_budget : std::max(1u, _wavefront_steps_per_render);
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
          dispatch_stage(cmd, PipelineStage::PrepareSample, film_dispatch, 0u);
          barrier_wavefront_buffers(cmd);
          if (_wavefront_camera_phase_initialized) {
            dispatch_stage(cmd, PipelineStage::InitCameraPath0, film_dispatch, 0u);
            barrier_wavefront_buffers(cmd);
          }
          if (enable_light_path) {
            dispatch_stage(cmd, PipelineStage::InitLightPath0, film_dispatch, 0u);
            barrier_wavefront_buffers(cmd);
          }
        });
        render_output_texture_state = RHIResourceState::General;

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
          if (initialize_deferred_camera_phase() == false) {
            return;
          }
        } else if ((_wavefront_camera_queue_count == 0u) && (_wavefront_light_queue_count == 0u)) {
          if (finalize_wavefront_sample() == false) {
            return;
          }
        } else if (_wavefront_path_iteration >= _wavefront_hard_iteration_cap) {
          if (finalize_wavefront_sample() == false) {
            return;
          }
        } else {
          const uint32_t path_iteration = _wavefront_path_iteration;
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
          const bool copy_queue_counts = continue_paths && queue_readback_due;
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
            const bool submit_non_dielectric_continue = continue_paths && (has_various_continue || has_plastic || has_conductor || has_thinfilm);
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

            if (continue_paths && has_dielectric && (_wavefront_light_queue_count > 0u)) {
              submit_stage_chunked(PipelineStage::LightContinuePrepareDielectric, _wavefront_light_queue_count, path_iteration, false);
            }

            if (continue_paths && has_dielectric && (_wavefront_camera_queue_count > 0u)) {
              submit_stage_chunked(PipelineStage::CameraContinuePrepareDielectric, _wavefront_camera_queue_count, path_iteration, true);
            }
          }

          if ((connect_light_batch_in_progress == false) && enable_connect_vertices && (_wavefront_camera_queue_count > 0u)) {
            const uint32_t generated_light_history_bounces =
              (phase_light_before_camera && _wavefront_camera_phase_initialized) ? _wavefront_light_max_path_length : std::min(light_history_bounces, path_iteration + 1u);
            const uint32_t camera_path_length = path_iteration + 1u;
            const uint32_t maximum_connected_light_path_length = (scene_max_path_length > (camera_path_length + 1u)) ? (scene_max_path_length - camera_path_length - 1u) : 0u;
            _wavefront_connect_light_history_bounces = std::min(maximum_connected_light_path_length, std::min(light_history_bounces, generated_light_history_bounces));
            _wavefront_connect_light_vertex_length = _wavefront_connect_light_history_bounces;
          }
          const uint32_t connect_light_vertex_count = std::min(kWavefrontConnectLightBatchSize, _wavefront_connect_light_vertex_length);
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
                  if (has_conductor) {
                    dispatch_stage_material_indirect(cmd, PipelineStage::LightConnectCameraPrepareConductor, false, kGPUWavefrontMaterialQueueConductor, path_iteration);
                  }
                  if (has_dielectric) {
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
                  if (has_conductor) {
                    dispatch_stage_material_indirect(cmd, PipelineStage::CameraDirectLightPrepareConductor, true, kGPUWavefrontMaterialQueueConductor, path_iteration);
                  }
                  if (has_dielectric) {
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
                    dispatch_stage_material_indirect(cmd, PipelineStage::VCMMergeDiffuse, true, kGPUWavefrontMaterialQueueVarious, path_iteration);
                  }
                  if (has_thinfilm && use_material_work_queues) {
                    dispatch_stage_material_indirect(cmd, PipelineStage::VCMMergeDiffuse, true, kGPUWavefrontMaterialQueueThinfilm, path_iteration);
                  }
                  if (has_plastic) {
                    dispatch_stage_material_indirect(cmd, PipelineStage::VCMMergePlastic, true, kGPUWavefrontMaterialQueuePlastic, path_iteration);
                  }
                  if (has_conductor) {
                    dispatch_stage_material_indirect(cmd, PipelineStage::VCMMergeConductor, true, kGPUWavefrontMaterialQueueConductor, path_iteration);
                  }
                  if (has_dielectric) {
                    dispatch_stage_material_indirect(cmd, PipelineStage::VCMMergeDielectric, true, kGPUWavefrontMaterialQueueDielectric, path_iteration);
                  }
                  barrier_wavefront_buffers(cmd);
                }
              }
            }
            if (connect_light_vertex_count > 0u) {
              const bool reset_light_cursor = _wavefront_connect_light_vertex_length == _wavefront_connect_light_history_bounces;
              const uint64_t connect_light_argument_buffer_offset =
                kGPUWavefrontConnectDispatchArgsOffset + static_cast<uint64_t>(connect_light_vertex_count - 1u) * kGPUWavefrontDispatchArgsStride;
              dispatch_stage_with_connect_light_length(cmd, PipelineStage::CameraConnectLightClear, connect_light_argument_buffer_offset, path_iteration,
                _wavefront_connect_light_vertex_length, connect_light_vertex_count, reset_light_cursor);
              barrier_wavefront_buffers(cmd);
              if (has_various_connect) {
                dispatch_stage_with_connect_light_length(cmd, PipelineStage::CameraConnectLightPrepareDiffuse, connect_light_argument_buffer_offset, path_iteration,
                  _wavefront_connect_light_vertex_length, connect_light_vertex_count, reset_light_cursor);
              }
              if (has_plastic) {
                dispatch_stage_with_connect_light_length(cmd, PipelineStage::CameraConnectLightPreparePlastic, connect_light_argument_buffer_offset, path_iteration,
                  _wavefront_connect_light_vertex_length, connect_light_vertex_count, reset_light_cursor);
              }
              if (has_conductor) {
                dispatch_stage_with_connect_light_length(cmd, PipelineStage::CameraConnectLightPrepareConductor, connect_light_argument_buffer_offset, path_iteration,
                  _wavefront_connect_light_vertex_length, connect_light_vertex_count, reset_light_cursor);
              }
              if (has_dielectric) {
                dispatch_stage_with_connect_light_length(cmd, PipelineStage::CameraConnectLightPrepareDielectric, connect_light_argument_buffer_offset, path_iteration,
                  _wavefront_connect_light_vertex_length, connect_light_vertex_count, reset_light_cursor);
              }
              barrier_wavefront_buffers(cmd);
              if (has_various_connect) {
                dispatch_stage_with_connect_light_length(cmd, PipelineStage::CameraConnectLightResolveDiffuse, connect_light_argument_buffer_offset, path_iteration,
                  _wavefront_connect_light_vertex_length, connect_light_vertex_count, reset_light_cursor);
              }
              if (has_plastic) {
                dispatch_stage_with_connect_light_length(cmd, PipelineStage::CameraConnectLightResolvePlastic, connect_light_argument_buffer_offset, path_iteration,
                  _wavefront_connect_light_vertex_length, connect_light_vertex_count, reset_light_cursor);
              }
              if (has_conductor) {
                dispatch_stage_with_connect_light_length(cmd, PipelineStage::CameraConnectLightResolveConductor, connect_light_argument_buffer_offset, path_iteration,
                  _wavefront_connect_light_vertex_length, connect_light_vertex_count, reset_light_cursor);
              }
              if (has_dielectric) {
                dispatch_stage_with_connect_light_length(cmd, PipelineStage::CameraConnectLightResolveDielectric, connect_light_argument_buffer_offset, path_iteration,
                  _wavefront_connect_light_vertex_length, connect_light_vertex_count, reset_light_cursor);
              }
              rebuild_dispatch_args(cmd, path_iteration);
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
                log::warning("GPU RT: failed to read camera queue count (%u)", static_cast<uint32_t>(read_result));
                _wavefront_camera_queue_count = 0u;
              } else {
                _wavefront_camera_queue_count = queue_header.count;
              }
            } else {
              _wavefront_camera_queue_count = 0u;
            }

            if ((enable_light_path) && (_wavefront_light_queue_count > 0u)) {
              GPUWavefrontQueueHeader queue_header = {};
              const RHIResult read_result = device.read_buffer(_light_queue_count_readback_buffer, &queue_header, static_cast<uint64_t>(sizeof(queue_header)));
              if (read_result != RHIResult::Success) {
                log::warning("GPU RT: failed to read light queue count (%u)", static_cast<uint32_t>(read_result));
                _wavefront_light_queue_count = 0u;
              } else {
                _wavefront_light_queue_count = queue_header.count;
                _wavefront_light_max_path_length = std::max(_wavefront_light_max_path_length, queue_header.max_path_length);
              }
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

          const bool waiting_for_deferred_camera_phase_after_step = phase_light_before_camera && enable_camera_path && (_wavefront_camera_phase_initialized == false);
          if (waiting_for_deferred_camera_phase_after_step) {
            if (_wavefront_path_iteration >= _wavefront_hard_iteration_cap) {
              _wavefront_light_queue_count = 0u;
            }
          } else if ((_wavefront_path_iteration >= _wavefront_hard_iteration_cap) || ((_wavefront_camera_queue_count == 0u) && (_wavefront_light_queue_count == 0u))) {
            if (finalize_wavefront_sample() == false) {
              return;
            }
          }
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
    if (render_preview_this_frame == false) {
      update_wavefront_auto_tuning(executed_wavefront_steps, wavefront_batch_ms, wavefront_budget_consumed, wavefront_auto_measurement_valid);
    }
    if ((render_output_texture_state == RHIResourceState::General) && frame_data.cmd.valid()) {
      ctx.cmd_texture_barrier(frame_data.cmd, render_output_texture, RHIResourceState::General, RHIResourceState::ShaderReadOnly);
      render_output_texture_state = RHIResourceState::ShaderReadOnly;
    }
  }

  _frame_index += 1u;
  if (completed_sample) {
    if (_preview_active) {
      _preview_visible = true;
      const double preview_elapsed_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - render_begin).count();
      if (_preview_resolution.update(preview_elapsed_seconds, true)) {
        reset_render_progress();
        return;
      }

      _sample_index += 1u;
      if (_run_state == RunState::Finishing) {
        stop_render_timing();
        _run_state = RunState::Stopped;
      } else if (_sample_index >= _last_target_samples) {
        stop_render_timing();
        _run_state = RunState::Completed;
      }
      return;
    }

    _preview_visible = false;
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
      stop_render_timing();
      _run_state = RunState::Stopped;
    } else if (_sample_index >= _last_target_samples) {
      stop_render_timing();
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

  destroy_preview_resources(device);
  destroy_pipelines(device);

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
  _scene_valid = false;
  _run_state = RunState::Stopped;
  _current_scene_hashes = {};
  _current_camera_hash = 0;
  _frame_index = 0u;
  _sample_index = 0u;
  reset_render_timing();
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
  _wavefront_tile_base_origin = {};
  _wavefront_tile_base_size = {};
  _wavefront_tile_plan_valid = false;
  _wavefront_camera_phase_initialized = false;
  _integrator_mode = 0u;
  _integrator_features = 0u;
  _material_compile_mask = 0u;
  _spectral_mode = 0u;
  _render_window_origin = {};
  _render_window_size = {};
  _active_preparation.reset();
  _publish_preparation.reset();
  _preparation_generation = 0u;
  _published_pipeline_count = 0u;
  _publish_pipeline_index = 0u;
  _pipeline_publish_logged = false;
  reset_preview_state();
  invalidate_output();
  _preparation_canceled = false;
  reset_runtime_failure();
  set_preparation_ready();
  request_scene_update();
}

void GPURaytracingRenderer::on_camera_changed(SceneRepresentation& scene) {
  ETX_PROFILER_SCOPE();
  (void)scene;
  _preview_camera_active = true;
  update_preview_active_state();
  reset_render_progress();
  if ((_run_state == RunState::Completed) || (_run_state == RunState::Finishing)) {
    _run_state = RunState::Running;
  }
}

void GPURaytracingRenderer::on_camera_become_steady(SceneRepresentation& scene) {
  ETX_PROFILER_SCOPE();
  (void)scene;
  _preview_camera_active = false;
  if (update_preview_active_state() && (_preview_active == false)) {
    reset_render_progress();
    if ((_run_state == RunState::Completed) || (_run_state == RunState::Finishing)) {
      _run_state = RunState::Running;
    }
  }
}

void GPURaytracingRenderer::on_scene_changed(SceneRepresentation& scene) {
  ETX_PROFILER_SCOPE();
  _scene_valid = scene.valid();
  reset_render_progress();
  Renderer::on_scene_changed(scene);
}

void GPURaytracingRenderer::on_scene_transforms_changed(SceneRepresentation& scene) {
  ETX_PROFILER_SCOPE();
  _scene_valid = scene.valid();
  reset_render_progress();
  Renderer::on_scene_transforms_changed(scene);
}

static bool update_host_visible_buffer(RHIDevice& device, const void* data, uint64_t required_size, RHIBufferUsage usage, RHIBindlessHandle& buffer, uint64_t& buffer_size,
  const char* buffer_name) {
  if ((data == nullptr) || (required_size == 0u)) {
    return false;
  }

  if ((buffer.valid() == false) || (buffer_size != required_size)) {
    if (buffer.valid()) {
      device.destroy_buffer(buffer);
      buffer = {};
      buffer_size = 0u;
    }

    RHIBufferDesc desc = {};
    desc.size = required_size;
    desc.usage = usage;
    desc.host_visible = true;
    const RHICreateBindlessResult create_result = device.create_buffer(desc);
    if ((create_result.result != RHIResult::Success) || (create_result.handle.valid() == false)) {
      log::error("GPU RT: failed to create host-visible preview '%s' buffer (%u)", buffer_name, static_cast<uint32_t>(create_result.result));
      return false;
    }
    buffer = create_result.handle;
    buffer_size = required_size;
  }

  const RHIResult update_result = device.update_buffer(buffer, data, required_size);
  if (update_result != RHIResult::Success) {
    log::error("GPU RT: failed to update host-visible preview '%s' buffer (%u)", buffer_name, static_cast<uint32_t>(update_result));
    return false;
  }
  return true;
}

void GPURaytracingRenderer::on_scene_transform_interaction_started(SceneRepresentation& scene) {
  (void)scene;
  _preview_transform_active = true;
  update_preview_active_state();
  reset_render_progress();
}

void GPURaytracingRenderer::on_scene_transform_interaction_finished(SceneRepresentation& scene) {
  (void)scene;
  _preview_transform_active = false;
  if (update_preview_active_state() && (_preview_active == false)) {
    reset_render_progress();
    if ((_run_state == RunState::Completed) || (_run_state == RunState::Finishing)) {
      _run_state = RunState::Running;
    }
  }
}

bool GPURaytracingRenderer::refit_top_level_acceleration_structure(RHIContext& ctx, const SceneData& scene_data) {
  if ((_tlas.valid() == false) || (_tlas_instance_buffer.valid() == false) || (_as_scratch_buffer.valid() == false)) {
    return false;
  }
  if (scene_data.hierarchy.mesh_instances.size() != _tlas_instance_count) {
    return false;
  }

  _tlas_instance_staging.clear();
  _tlas_instance_staging.reserve(_tlas_instance_count);
  auto& device = ctx.device();
  for (uint32_t instance_index = 0u; instance_index < _tlas_instance_count; ++instance_index) {
    const ResolvedMeshInstance& resolved = scene_data.hierarchy.mesh_instances[instance_index];
    if (resolved.mesh_index >= _blas.size()) {
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
  _tlas = new_tlas;
  _tlas_instance_buffer = inst_res.handle;
  _as_scratch_buffer = scratch_res.handle;
  _tlas_instance_count = static_cast<uint32_t>(rhi_instances.size());
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
    GPUSceneGlobals globals = build_scene_globals(data, packed_emitters);
    _scene_bounding_sphere_radius = globals.bounding_sphere_radius;
    upload_success = upload_or_update_linear_scene_buffer(device, &globals, size_t(1), scene_buffer_usage, _scene_globals_buffer, _scene_globals_buffer_size,
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
    GPUSceneGlobals globals = build_scene_globals(data, packed_emitters);
    _scene_bounding_sphere_radius = globals.bounding_sphere_radius;
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

#if defined(ETX_ENABLE_SHADER_PACKAGER) && ETX_ENABLE_SHADER_PACKAGER
bool build_raytracer_shader_package(const std::filesystem::path& output_path, RHIBackend backend, RaytracerShaderPackageStatistics& statistics, std::string& error_message) {
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
  };
  constexpr uint32_t material_mask_count = 1u << MaterialClass::Count;
  constexpr uint32_t open_pbr_bit = 1u << MaterialClass::OpenPBR;
  constexpr uint32_t spectral_modes[] = {static_cast<uint32_t>(GPUSpectralMode::RGB), static_cast<uint32_t>(GPUSpectralMode::Spectral)};

  for (const PackageIntegratorConfiguration& configuration : integrator_configurations) {
    for (uint32_t material_mask = 0u; material_mask < material_mask_count; ++material_mask) {
      if ((material_mask & open_pbr_bit) != 0u) {
        continue;
      }
      for (const uint32_t spectral_mode : spectral_modes) {
        for (const WavefrontStage& stage : kWavefrontStages) {
          if (wavefront_stage_enabled(stage.stage, configuration.mode, configuration.features, material_mask) == false) {
            continue;
          }
          const auto defines = wavefront_stage_defines(stage, configuration.mode, material_mask, spectral_mode);
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

  std::atomic<size_t> next_group_index = 0u;
  std::atomic<uint32_t> completed_variants = 0u;
  std::atomic<bool> compilation_failed = false;
  std::mutex package_entries_mutex = {};
  std::mutex error_mutex = {};
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
      const std::filesystem::path source_path = std::filesystem::path(env().data_folder()) / group.source_name;
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
      const uint32_t previous_count = completed_variants.fetch_add(group_variant_count);
      const uint32_t completed_count = previous_count + group_variant_count;
      {
        std::lock_guard<std::mutex> entries_lock(package_entries_mutex);
        package_entries.insert(package_entries.end(), std::make_move_iterator(group_entries.begin()), std::make_move_iterator(group_entries.end()));
      }
      if ((completed_count == unique_requests.size()) || ((previous_count / 16u) != (completed_count / 16u))) {
        log::info("Shader package progress: %u/%zu variants", completed_count, unique_requests.size());
      }
    }
  };

  const uint32_t available_threads = std::max(1u, std::thread::hardware_concurrency());
  const uint32_t worker_count = std::min<uint32_t>(static_cast<uint32_t>(compile_groups.size()), std::min(available_threads, 8u));
  std::vector<std::thread> workers = {};
  workers.reserve(worker_count);
  for (uint32_t worker_index = 0u; worker_index < worker_count; ++worker_index) {
    workers.emplace_back(compile_group);
  }
  for (std::thread& worker : workers) {
    worker.join();
  }
  if (compilation_failed.load(std::memory_order_acquire)) {
    return false;
  }
  const auto compile_end = std::chrono::steady_clock::now();
  statistics.compile_time_ms = elapsed_ms(compile_begin, compile_end);
  compiler.log_statistics("shader-package");

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
