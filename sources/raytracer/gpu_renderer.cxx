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
#include <limits>
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
constexpr uint64_t kWavefrontMaxAddressableBufferSize = static_cast<uint64_t>(std::numeric_limits<uint32_t>::max());
constexpr uint32_t kWavefrontBDPTTileSide = 512u;
constexpr uint32_t kWavefrontConnectLightBatchSize = 4u;
constexpr uint64_t kWavefrontBDPTFallbackLightVertexBytes = 512ull * 1024ull * 1024ull;
constexpr uint64_t kWavefrontBDPTMemoryBudgetDivisor = 8ull;
constexpr const char* kPreviewShaderFile = "shaders/gpu_rt_preview_trace.hlsl";
constexpr const char* kPreviewShaderEntry = "gpu_preview_trace_main";

enum class GPUIntegratorMode : uint32_t {
  PathTracing = 0u,
  LightTracing = 1u,
  BDPTFast = 2u,
  BDPTFull = 3u,
  VCM = 4u,
};

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
  {GPURaytracingRenderer::PipelineStage::CameraDirectLightSample, "shaders/gpu_rt_wavefront_direct_light_sample.hlsl", "wavefront_camera_direct_light_sample_main", "0", nullptr},
  {GPURaytracingRenderer::PipelineStage::CameraDirectLightPrepareDiffuse, "shaders/gpu_rt_wavefront_direct_light_prepare_variant.hlsl",
    "wavefront_camera_direct_light_prepare_diffuse_main", "0", "1", true},
  {GPURaytracingRenderer::PipelineStage::CameraDirectLightPreparePlastic, "shaders/gpu_rt_wavefront_direct_light_prepare_variant.hlsl",
    "wavefront_camera_direct_light_prepare_plastic_main", "0", "2", true},
  {GPURaytracingRenderer::PipelineStage::CameraDirectLightPrepareConductor, "shaders/gpu_rt_wavefront_direct_light_prepare_variant.hlsl",
    "wavefront_camera_direct_light_prepare_conductor_main", "0", "3", true},
  {GPURaytracingRenderer::PipelineStage::CameraDirectLightPrepareDielectric, "shaders/gpu_rt_wavefront_direct_light_prepare_variant.hlsl",
    "wavefront_camera_direct_light_prepare_dielectric_main", "0", "4", true},
  {GPURaytracingRenderer::PipelineStage::CameraDirectLightShadow, "shaders/gpu_rt_wavefront_shadow.hlsl", "wavefront_camera_direct_light_shadow_main", nullptr, nullptr},
  {GPURaytracingRenderer::PipelineStage::CameraDirectLightAccumulate, "shaders/gpu_rt_wavefront_direct_light.hlsl", "wavefront_camera_direct_light_accumulate_main", nullptr, nullptr},
  {GPURaytracingRenderer::PipelineStage::CameraDirectHitAccumulate, "shaders/gpu_rt_wavefront_direct_hit.hlsl", "wavefront_camera_direct_hit_accumulate_main", nullptr, nullptr},
  {GPURaytracingRenderer::PipelineStage::CameraConnectLightClear, "shaders/gpu_rt_wavefront_connect_light_clear.hlsl", "wavefront_camera_connect_light_clear_main", nullptr, nullptr},
  {GPURaytracingRenderer::PipelineStage::CameraConnectLightPrepareDiffuse, "shaders/gpu_rt_wavefront_connect_light_prepare_variant.hlsl",
    "wavefront_camera_connect_light_prepare_diffuse_main", "0", "1", true},
  {GPURaytracingRenderer::PipelineStage::CameraConnectLightPreparePlastic, "shaders/gpu_rt_wavefront_connect_light_prepare_variant.hlsl",
    "wavefront_camera_connect_light_prepare_plastic_main", "0", "2", true},
  {GPURaytracingRenderer::PipelineStage::CameraConnectLightPrepareConductor, "shaders/gpu_rt_wavefront_connect_light_prepare_variant.hlsl",
    "wavefront_camera_connect_light_prepare_conductor_main", "0", "3", true},
  {GPURaytracingRenderer::PipelineStage::CameraConnectLightPrepareDielectric, "shaders/gpu_rt_wavefront_connect_light_prepare_variant.hlsl",
    "wavefront_camera_connect_light_prepare_dielectric_main", "0", "4", true},
  {GPURaytracingRenderer::PipelineStage::CameraConnectLightResolveDiffuse, "shaders/gpu_rt_wavefront_connect_light_resolve_variant.hlsl",
    "wavefront_camera_connect_light_resolve_diffuse_main", "0", "1", true},
  {GPURaytracingRenderer::PipelineStage::CameraConnectLightResolvePlastic, "shaders/gpu_rt_wavefront_connect_light_resolve_variant.hlsl",
    "wavefront_camera_connect_light_resolve_plastic_main", "0", "2", true},
  {GPURaytracingRenderer::PipelineStage::CameraConnectLightResolveConductor, "shaders/gpu_rt_wavefront_connect_light_resolve_variant.hlsl",
    "wavefront_camera_connect_light_resolve_conductor_main", "0", "3", true},
  {GPURaytracingRenderer::PipelineStage::CameraConnectLightResolveDielectric, "shaders/gpu_rt_wavefront_connect_light_resolve_variant.hlsl",
    "wavefront_camera_connect_light_resolve_dielectric_main", "0", "4", true},
  {GPURaytracingRenderer::PipelineStage::CameraConnectLightShadow, "shaders/gpu_rt_wavefront_shadow.hlsl", "wavefront_camera_connect_light_shadow_main", nullptr, nullptr},
  {GPURaytracingRenderer::PipelineStage::CameraConnectLightAccumulate, "shaders/gpu_rt_wavefront_connect_light_path.hlsl", "wavefront_camera_connect_light_accumulate_main",
    nullptr, nullptr},
  {GPURaytracingRenderer::PipelineStage::CameraContinuePrepareDiffuse, "shaders/gpu_rt_wavefront_surface_continue_prepare_camera_variant.hlsl",
    "wavefront_camera_continue_prepare_diffuse_main", "0", "1", true},
  {GPURaytracingRenderer::PipelineStage::CameraContinuePreparePlastic, "shaders/gpu_rt_wavefront_surface_continue_prepare_camera_variant.hlsl",
    "wavefront_camera_continue_prepare_plastic_main", "0", "2", true},
  {GPURaytracingRenderer::PipelineStage::CameraContinuePrepareConductor, "shaders/gpu_rt_wavefront_surface_continue_prepare_camera_variant.hlsl",
    "wavefront_camera_continue_prepare_conductor_main", "0", "3", true},
  {GPURaytracingRenderer::PipelineStage::CameraContinuePrepareDielectric, "shaders/gpu_rt_wavefront_surface_continue_prepare_camera_variant.hlsl",
    "wavefront_camera_continue_prepare_dielectric_main", "0", "4", true},
  {GPURaytracingRenderer::PipelineStage::CameraContinuePrepareThinfilm, "shaders/gpu_rt_wavefront_surface_continue_prepare_camera_variant.hlsl",
    "wavefront_camera_continue_prepare_thinfilm_main", "0", "5", true},
  {GPURaytracingRenderer::PipelineStage::CameraContinueFinalize, "shaders/gpu_rt_wavefront_surface_camera.hlsl", "wavefront_camera_continue_finalize_main", nullptr, nullptr},
  {GPURaytracingRenderer::PipelineStage::TraceLight, "shaders/gpu_rt_wavefront_trace_light.hlsl", "wavefront_trace_light_main", nullptr, nullptr},
  {GPURaytracingRenderer::PipelineStage::LightSurfaceClassify, "shaders/gpu_rt_wavefront_surface_light.hlsl", "wavefront_light_surface_classify_main", nullptr, nullptr},
  {GPURaytracingRenderer::PipelineStage::LightContinuePrepareDiffuse, "shaders/gpu_rt_wavefront_surface_continue_prepare_light_variant.hlsl",
    "wavefront_light_continue_prepare_diffuse_main", "0", "1", true},
  {GPURaytracingRenderer::PipelineStage::LightContinuePreparePlastic, "shaders/gpu_rt_wavefront_surface_continue_prepare_light_variant.hlsl",
    "wavefront_light_continue_prepare_plastic_main", "0", "2", true},
  {GPURaytracingRenderer::PipelineStage::LightContinuePrepareConductor, "shaders/gpu_rt_wavefront_surface_continue_prepare_light_variant.hlsl",
    "wavefront_light_continue_prepare_conductor_main", "0", "3", true},
  {GPURaytracingRenderer::PipelineStage::LightContinuePrepareDielectric, "shaders/gpu_rt_wavefront_surface_continue_prepare_light_variant.hlsl",
    "wavefront_light_continue_prepare_dielectric_main", "0", "4", true},
  {GPURaytracingRenderer::PipelineStage::LightContinuePrepareThinfilm, "shaders/gpu_rt_wavefront_surface_continue_prepare_light_variant.hlsl",
    "wavefront_light_continue_prepare_thinfilm_main", "0", "5", true},
  {GPURaytracingRenderer::PipelineStage::LightConnectCameraClear, "shaders/gpu_rt_wavefront_connect_camera.hlsl", "wavefront_light_connect_camera_prepare_main",
    nullptr, nullptr},
  {GPURaytracingRenderer::PipelineStage::LightConnectCameraPrepareDiffuse, "shaders/gpu_rt_wavefront_connect_camera_prepare_variant.hlsl",
    "wavefront_light_connect_camera_prepare_diffuse_main", "0", "1", true},
  {GPURaytracingRenderer::PipelineStage::LightConnectCameraPreparePlastic, "shaders/gpu_rt_wavefront_connect_camera_prepare_variant.hlsl",
    "wavefront_light_connect_camera_prepare_plastic_main", "0", "2", true},
  {GPURaytracingRenderer::PipelineStage::LightConnectCameraPrepareConductor, "shaders/gpu_rt_wavefront_connect_camera_prepare_variant.hlsl",
    "wavefront_light_connect_camera_prepare_conductor_main", "0", "3", true},
  {GPURaytracingRenderer::PipelineStage::LightConnectCameraPrepareDielectric, "shaders/gpu_rt_wavefront_connect_camera_prepare_variant.hlsl",
    "wavefront_light_connect_camera_prepare_dielectric_main", "0", "4", true},
  {GPURaytracingRenderer::PipelineStage::LightConnectCameraShadow, "shaders/gpu_rt_wavefront_shadow.hlsl", "wavefront_light_connect_camera_shadow_main", nullptr, nullptr},
  {GPURaytracingRenderer::PipelineStage::LightConnectCameraAccumulate, "shaders/gpu_rt_wavefront_connect_camera.hlsl", "wavefront_light_connect_camera_accumulate_main",
    nullptr, nullptr},
  {GPURaytracingRenderer::PipelineStage::LightContinueFinalize, "shaders/gpu_rt_wavefront_surface_light.hlsl", "wavefront_light_continue_finalize_main", nullptr, nullptr},
  {GPURaytracingRenderer::PipelineStage::SwapQueues, "shaders/gpu_rt_wavefront_prepare.hlsl", "wavefront_swap_queues_main", nullptr, nullptr},
  {GPURaytracingRenderer::PipelineStage::FinalizeSample, "shaders/gpu_rt_wavefront_prepare.hlsl", "wavefront_finalize_sample_main", nullptr, nullptr},
};

struct WavefrontWindow {
  uint2 origin = {};
  uint2 size = {};
};

uint32_t divide_round_up(uint32_t value, uint32_t divisor) {
  return (value / divisor) + (((value % divisor) == 0u) ? 0u : 1u);
}

uint64_t wavefront_tile_budget_bytes(const RHIMemoryStats& memory_stats) {
  const uint64_t budget_bytes =
    (memory_stats.gpu_device_local_budget_bytes > 0ull) ? (memory_stats.gpu_device_local_budget_bytes / kWavefrontBDPTMemoryBudgetDivisor) : kWavefrontBDPTFallbackLightVertexBytes;
  return std::max<uint64_t>(1ull, std::min(budget_bytes, kWavefrontMaxAddressableBufferSize));
}

uint64_t wavefront_tile_bytes_per_path(uint32_t integrator_features, bool has_subsurface_material, uint32_t max_path_length) {
  const bool enable_camera_path = (integrator_features & GPUIntegratorFeatures::CameraPath) != 0u;
  const bool enable_light_path = (integrator_features & GPUIntegratorFeatures::LightPath) != 0u;
  const bool enable_connect_to_light = (integrator_features & GPUIntegratorFeatures::ConnectToLight) != 0u;
  const bool enable_connect_to_camera = (integrator_features & GPUIntegratorFeatures::ConnectToCamera) != 0u;
  const bool enable_connect_vertices = (integrator_features & GPUIntegratorFeatures::ConnectVertices) != 0u;
  const uint32_t camera_history_bounces = enable_camera_path ? kWavefrontRollingHistoryBounces : 0u;
  const uint32_t light_history_bounces = enable_light_path ? (enable_connect_vertices ? max_path_length : kWavefrontLightHistoryBounces) : 0u;

  uint64_t result = kGPUWavefrontPathMetaStride;
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
  if (enable_connect_to_light) {
    result += kGPUWavefrontDirectLightSampleStride;
    result += kGPUWavefrontDirectLightTaskStride;
    result += kGPUWavefrontDirectLightResultStride;
  }
  if (enable_connect_vertices) {
    result += static_cast<uint64_t>(kWavefrontConnectLightBatchSize) * kGPUWavefrontConnectLightTaskStride;
    result += static_cast<uint64_t>(kWavefrontConnectLightBatchSize) * kGPUWavefrontConnectLightResultStride;
  }
  if (enable_connect_to_camera) {
    result += kGPUWavefrontConnectCameraTaskStride;
    result += kGPUWavefrontConnectCameraResultStride;
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
      result.features &= ~(GPUIntegratorFeatures::LightPath | GPUIntegratorFeatures::ConnectToCamera | GPUIntegratorFeatures::ConnectVertices | GPUIntegratorFeatures::MergeVertices |
                           GPUIntegratorFeatures::VCMMis);
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
    result.features |= GPUIntegratorFeatures::ConnectVertices | GPUIntegratorFeatures::MergeVertices | GPUIntegratorFeatures::VCMMis;
    result.supported = false;
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

GPUIntegratorMode gpu_integrator_mode_from_scene(const SceneRepresentation& scene) {
  return gpu_integrator_selection_from_scene(scene).mode;
}

std::string gpu_integrator_selection_error_message(const GPUIntegratorSelection& selection) {
  if (selection.integrator_type == Integrator::Type::Bidirectional) {
    if (bdpt_mode_valid(selection.requested_bdpt_mode) == false) {
      return "GPU RT does not support the requested bidirectional mode value.";
    }
  }

  if (selection.integrator_type == Integrator::Type::VCM) {
    return "GPU RT VCM is not complete yet. BDPT Full GPU support is available first; vertex merging still needs the GPU spatial grid stages.";
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
    case GPURaytracingRenderer::PipelineStage::CameraConnectLightAccumulate:
      return "CameraConnectLightAccumulate";
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
         (material_compile_mask_has(mask, MaterialClass::Velvet)) || (material_compile_mask_has(mask, MaterialClass::Void));
}

bool material_compile_mask_has_various_connect(uint32_t mask) {
  return (material_compile_mask_has_various_continue(mask)) || (material_compile_mask_has(mask, MaterialClass::Thinfilm));
}

bool material_compile_mask_has_conductor_stage(uint32_t mask) {
  return material_compile_mask_has(mask, MaterialClass::Conductor);
}

bool wavefront_stage_needs_low_opt(GPURaytracingRenderer::PipelineStage stage, uint32_t material_compile_mask) {
  (void)stage;
  (void)material_compile_mask;
  return false;
}

bool wavefront_stage_needs_spirv_compact_ids(GPURaytracingRenderer::PipelineStage stage, uint32_t material_compile_mask) {
  (void)material_compile_mask;
  switch (stage) {
    case GPURaytracingRenderer::PipelineStage::CameraDirectLightPreparePlastic:
    case GPURaytracingRenderer::PipelineStage::CameraConnectLightPreparePlastic:
    case GPURaytracingRenderer::PipelineStage::CameraConnectLightResolvePlastic:
    case GPURaytracingRenderer::PipelineStage::CameraContinuePreparePlastic:
    case GPURaytracingRenderer::PipelineStage::LightConnectCameraPreparePlastic:
    case GPURaytracingRenderer::PipelineStage::LightContinuePreparePlastic:
      return true;
    default:
      return false;
  }
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
    case GPURaytracingRenderer::PipelineStage::CameraConnectLightAccumulate:
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
    case GPURaytracingRenderer::PipelineStage::PrepareSample:
    case GPURaytracingRenderer::PipelineStage::SwapQueues:
    case GPURaytracingRenderer::PipelineStage::FinalizeSample:
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
    log::error("GPU RT: failed to create '%s' storage buffer (%u)", (buffer_name != nullptr) ? buffer_name : "unknown", static_cast<uint32_t>(create_result.result));
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
  _wavefront_steps_per_render = std::clamp(value, 1u, 1024u);
}

void GPURaytracingRenderer::set_batch_coarse_progress(bool value) {
  _batch_coarse_progress = value;
}

void GPURaytracingRenderer::reset_render_window() {
  _render_window_origin = {};
  _render_window_size = {};
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
  const GPUIntegratorSelection integrator_selection = gpu_integrator_selection_from_scene(scene);
  _integrator_mode = static_cast<uint32_t>(integrator_selection.mode);
  _integrator_features = integrator_selection.features;
  _material_compile_mask = build_material_compile_mask(scene.data());
  _initialized = true;

  if (integrator_selection.supported == false) {
    set_runtime_failure(gpu_integrator_selection_error_message(integrator_selection));
    return;
  }

  set_preparation_ready();
}

void GPURaytracingRenderer::reload_shaders(RHIContext& ctx, SceneRepresentation& scene) {
  ETX_PROFILER_SCOPE();
  auto& device = ctx.device();
  const bool recreate_preview_pipeline = _preview_pipeline.valid() || _preview_active;
  destroy_preview_pipeline(device);
  if (recreate_preview_pipeline) {
    create_preview_pipeline(ctx);
  }
  request_pipeline_preparation(scene, "reload");
}

void GPURaytracingRenderer::update_camera(SceneRepresentation& scene, float dt) {
  ETX_CRITICAL(_camera_controller);

  const bool camera_updated = _camera_controller->update(dt);
  const bool camera_navigation_input_active = _camera_controller->camera_navigation_input_active();
  if (camera_updated) {
    scene.store_active_camera();
    on_camera_changed(scene);
  } else if ((camera_navigation_input_active == false) && (_preview_active || (camera_updated != last_camera_update_state))) {
    on_camera_become_steady(scene);
  }
  last_camera_update_state = camera_updated;
}

void GPURaytracingRenderer::set_compile_stage_filter(const std::string& value) {
  _compile_stage_filter = value;
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
    result.total_steps = active->total_steps;
    if (_publish_preparation) {
      result.completed_steps = active->total_compile_groups + _published_pipeline_count;
    } else {
      result.completed_steps = active->completed_compile_groups.load();
    }
    result.completed_steps = std::min(result.completed_steps, result.total_steps);
  }

  return result;
}

RendererRuntimeStats GPURaytracingRenderer::runtime_stats() const {
  RendererRuntimeStats result = {};
  result.valid = _initialized;
  result.completed_samples = _sample_index;
  result.target_samples = _last_target_samples;
  result.elapsed_seconds = _last_render_elapsed_seconds;
  if (_render_timing_active) {
    const auto now = std::chrono::steady_clock::now();
    result.elapsed_seconds = std::chrono::duration<double>(now - _render_started_at).count();
  }
  const double tile_count = static_cast<double>(std::max(1u, _wavefront_tile_count));
  const double tiled_sample_progress = (_wavefront_tile_plan_valid || (_wavefront_tile_count > 1u)) ? (static_cast<double>(_wavefront_tile_index) / tile_count) : 0.0;
  const double completed_sample_count = static_cast<double>(_sample_index) + tiled_sample_progress;
  if ((result.elapsed_seconds > 0.0) && (_last_target_samples > 0u) && (completed_sample_count > 0.0)) {
    const double sample_rate = completed_sample_count / result.elapsed_seconds;
    if (sample_rate > 0.0) {
      const double remaining_samples = std::max(0.0, static_cast<double>(_last_target_samples) - completed_sample_count);
      result.estimated_remaining_seconds = remaining_samples / sample_rate;
    }
  } else if ((_last_target_samples > 0u) && (_sample_index >= _last_target_samples)) {
    result.estimated_remaining_seconds = 0.0;
  }
  return result;
}

void GPURaytracingRenderer::reset_render_timing() {
  _render_started_at = {};
  _last_render_elapsed_seconds = 0.0;
  _render_timing_active = false;
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
  for (auto& pipeline : _pipelines) {
    if (pipeline.valid()) {
      device.destroy_pipeline(pipeline);
      pipeline = {};
    }
  }
}

bool GPURaytracingRenderer::create_preview_pipeline(RHIContext& ctx) {
  if (_preview_pipeline.valid()) {
    return true;
  }
  if (_preview_pipeline_failed) {
    return false;
  }

  auto& device = ctx.device();
  auto& compiler = ShaderCompiler::instance();
  const auto compile_begin = std::chrono::steady_clock::now();
  const std::vector<ShaderCompiler::ShaderEntryPoint> entry_points = {
    {kPreviewShaderEntry, RHIShaderStage::Compute},
  };
  const auto compilation = compiler.compile(kPreviewShaderFile, entry_points, {}, _backend);
  const auto compile_end = std::chrono::steady_clock::now();
  if ((compilation.result != RHIResult::Success) || (compilation.binaries.size() != 1u)) {
    _preview_pipeline_failed = true;
    log::warning("GPU RT preview pipeline compilation failed after %.2fms: %s", elapsed_ms(compile_begin, compile_end), compilation.error_message.c_str());
    return false;
  }

  const auto create_begin = std::chrono::steady_clock::now();
  const RHIComputePipelineDesc desc = device.make_compute_pipeline_desc(compilation.binaries[0]);
  const auto pipeline_result = device.create_compute_pipeline(desc);
  const auto create_end = std::chrono::steady_clock::now();
  if ((pipeline_result.result != RHIResult::Success) || (pipeline_result.handle.valid() == false)) {
    _preview_pipeline_failed = true;
    log::warning("GPU RT preview pipeline creation failed after %.2fms (%u)", elapsed_ms(create_begin, create_end), static_cast<uint32_t>(pipeline_result.result));
    return false;
  }

  _preview_pipeline = pipeline_result.handle;
  return true;
}

void GPURaytracingRenderer::destroy_preview_pipeline(RHIDevice& device) {
  if (_preview_pipeline.valid()) {
    device.destroy_pipeline(_preview_pipeline);
    _preview_pipeline = {};
  }
  _preview_pipeline_failed = false;
}

bool GPURaytracingRenderer::render_preview(RHIContext& ctx, RHICommandBuffer frame_cmd, const GPURTConstants& constants, const RHIDispatchDesc& dispatch) {
  if (_preview_pipeline.valid() == false) {
    return false;
  }
  if (_output_texture.valid() == false) {
    return false;
  }

  if (frame_cmd.valid()) {
    ctx.cmd_texture_barrier(frame_cmd, _output_texture, _output_texture_state, RHIResourceState::General);
    ctx.cmd_set_pipeline(frame_cmd, _preview_pipeline);
    ctx.cmd_push_constants(frame_cmd, &constants, sizeof(constants));
    ctx.cmd_dispatch(frame_cmd, dispatch);
    ctx.cmd_texture_barrier(frame_cmd, _output_texture, RHIResourceState::General, RHIResourceState::ShaderReadOnly);
    _output_texture_state = RHIResourceState::ShaderReadOnly;
    return true;
  }

  RHICommandBuffer cmd = ctx.get_command_buffer();
  if (cmd.valid() == false) {
    log::warning("GPU RT preview: failed to allocate command buffer");
    return false;
  }

  ctx.command_buffer_begin(cmd);
  ctx.cmd_texture_barrier(cmd, _output_texture, _output_texture_state, RHIResourceState::General);
  ctx.cmd_set_pipeline(cmd, _preview_pipeline);
  ctx.cmd_push_constants(cmd, &constants, sizeof(constants));
  ctx.cmd_dispatch(cmd, dispatch);
  ctx.cmd_texture_barrier(cmd, _output_texture, RHIResourceState::General, RHIResourceState::ShaderReadOnly);
  ctx.command_buffer_end(cmd);
  ctx.submit_command_buffer({cmd});
  const RHIResult wait_result = ctx.wait_for_command_buffer(cmd);
  ctx.destroy_command_buffer(cmd);
  _output_texture_state = RHIResourceState::ShaderReadOnly;
  if (wait_result != RHIResult::Success) {
    log::warning("GPU RT preview: command wait failed (%u)", static_cast<uint32_t>(wait_result));
    return false;
  }

  return true;
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
  log::info("GPU RT preparation queued: generation=%u integrator=%s features=0x%08x material_mask=0x%08x filter=%s", result->generation,
    gpu_integrator_mode_to_string(static_cast<GPUIntegratorMode>(result->integrator_mode)), result->integrator_features, result->material_compile_mask,
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
    ShaderCompiler::MultiShaderCompilationResult compilation = {};
    double compile_time_ms = 0.0;
  };

  std::vector<StageCompileGroup> compile_groups = {};
  auto find_or_add_group = [&](const WavefrontStage& stage_info) -> StageCompileGroup& {
    const std::string source_file = stage_info.source_file ? stage_info.source_file : "";
    const std::string optimization_level =
      wavefront_stage_needs_low_opt(stage_info.stage, result->material_compile_mask) ? "0" : (stage_info.optimization_level ? stage_info.optimization_level : "");
    const std::string spirv_opt_config =
      wavefront_stage_needs_spirv_compact_ids(stage_info.stage, result->material_compile_mask) ? "--compact-ids" : "";
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

  result->total_compile_groups = static_cast<uint32_t>(compile_groups.size());
  result->total_steps = result->total_compile_groups + result->total_pipelines;

  if ((result->compile_stage_filter.empty() == false) && (result->compile_filter_matched == false)) {
    result->error_message = "GPU RT compile stage filter '" + result->compile_stage_filter + "' did not match any pipeline entry point";
    log::error("%s", result->error_message.c_str());
    result->compile_finished_at = std::chrono::steady_clock::now();
    return;
  }

  if (compile_groups.empty()) {
    result->success = true;
    result->compile_finished_at = std::chrono::steady_clock::now();
    log::info("GPU RT preparation background compile finished: generation=%u groups=0 stages=0 wall=0.00ms", result->generation);
    return;
  }

  compiler.reset_statistics();
  const auto compile_begin = std::chrono::steady_clock::now();
  const uint32_t worker_count = std::min<uint32_t>(static_cast<uint32_t>(compile_groups.size()), std::max(1u, scheduler.max_thread_count()));

  auto compile_group = [&](StageCompileGroup& group) {
    std::vector<ShaderCompiler::ShaderEntryPoint> entry_points = {};
    entry_points.reserve(group.stages.size());
    for (const WavefrontStage* stage_info : group.stages) {
      entry_points.push_back({stage_info->entry_point, RHIShaderStage::Compute});
    }

    std::unordered_map<std::string, std::string> defines = {};
    if (group.optimization_level.empty() == false) {
      defines["ETX_DXC_OPT_LEVEL"] = group.optimization_level;
    }
    if (group.spirv_opt_config.empty() == false) {
      defines["ETX_DXC_SPIRV_OPT_CONFIG"] = group.spirv_opt_config;
    }
    if (group.bsdf_kind.empty() == false) {
      defines["ETX_BSDF_KIND"] = group.bsdf_kind;
    }
    if (static_cast<GPUIntegratorMode>(result->integrator_mode) == GPUIntegratorMode::PathTracing) {
      defines["ETX_WAVEFRONT_PATH_TRACING_ONLY"] = "1";
    }
    // TODO(OpenPBR GPU parity): OpenPBR is intentionally not compiled into wavefront stages yet.
    // The current OpenPBR sample path overflows DXC/SPIR-V legalization and is not parity-ready.
    if (material_compile_mask_has(result->material_compile_mask, MaterialClass::Thinfilm)) {
      defines["ETX_ENABLE_THINFILM_STAGE"] = "1";
    }
    if (material_compile_mask_has(result->material_compile_mask, MaterialClass::Velvet)) {
      defines["ETX_ENABLE_VELVET_STAGE"] = "1";
    }
    if (material_compile_mask_has(result->material_compile_mask, MaterialClass::Plastic)) {
      defines["ETX_ENABLE_PLASTIC_STAGE"] = "1";
    }
    if (material_compile_mask_has_conductor_stage(result->material_compile_mask)) {
      defines["ETX_ENABLE_CONDUCTOR_STAGE"] = "1";
    }
    if (material_compile_mask_has(result->material_compile_mask, MaterialClass::Dielectric)) {
      defines["ETX_ENABLE_DIELECTRIC_STAGE"] = "1";
    }
    if (group.stage_entry_define.empty() == false) {
      defines["ETX_STAGE_ENTRY"] = group.stage_entry_define;
    }

    const auto group_compile_begin = std::chrono::steady_clock::now();
    group.compilation = compiler.compile(group.source_file, entry_points, defines, _backend);
    const auto group_compile_end = std::chrono::steady_clock::now();
    group.compile_time_ms = elapsed_ms(group_compile_begin, group_compile_end);
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

  const auto compile_end = std::chrono::steady_clock::now();
  const double compile_wall_time_ms = elapsed_ms(compile_begin, compile_end);

  for (const auto& group : compile_groups) {
    if ((group.compilation.result != RHIResult::Success) || (group.compilation.binaries.size() != group.stages.size())) {
      const char* failing_stage = group.stages.empty() ? "<unknown>" : group.stages.front()->entry_point;
      result->error_message = "GPU shader pipeline compilation failed at '" + std::string(failing_stage) + "'";
      log::error("Failed to compile GPU RT shader group rooted at '%s' after %.2fms: %s", failing_stage, group.compile_time_ms, group.compilation.error_message.c_str());
      compiler.log_statistics("GPU RT wavefront");
      result->compile_finished_at = std::chrono::steady_clock::now();
      return;
    }
  }

  result->compiled_stages.reserve(result->total_pipelines);
  for (const auto& group : compile_groups) {
    for (size_t binary_index = 0u; binary_index < group.stages.size(); ++binary_index) {
      const WavefrontStage& stage_info = *group.stages[binary_index];
      CompiledStageBinary compiled_stage = {
        .stage = stage_info.stage,
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
      result->compiled_stages.push_back(std::move(compiled_stage));
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
}

void GPURaytracingRenderer::request_pipeline_preparation(const SceneRepresentation& scene, const char* reason) {
  if (_initialized == false) {
    return;
  }

  const GPUIntegratorSelection integrator_selection = gpu_integrator_selection_from_scene(scene);
  if (integrator_selection.supported == false) {
    set_runtime_failure(gpu_integrator_selection_error_message(integrator_selection));
    return;
  }

  reset_runtime_failure();
  _frame_index = 0u;
  _sample_index = 0u;
  reset_render_timing();
  _wavefront_render_step = WavefrontRenderStep::InitSample;
  _wavefront_path_iteration = 0u;
  _wavefront_hard_iteration_cap = 0u;
  _wavefront_camera_queue_count = 0u;
  _wavefront_light_queue_count = 0u;
  _wavefront_light_max_path_length = 0u;
  _wavefront_tile_index = 0u;
  _wavefront_tile_max_pixels = 0u;
  _wavefront_tile_count = 1u;
  _wavefront_tile_path_capacity = 0u;
  _wavefront_tile_base_origin = {};
  _wavefront_tile_base_size = {};
  _wavefront_tile_plan_valid = false;
  _wavefront_camera_phase_initialized = false;
  _integrator_mode = static_cast<uint32_t>(integrator_selection.mode);
  _integrator_features = integrator_selection.features;
  _material_compile_mask = build_material_compile_mask(scene.data());
  _preparation_generation += 1u;
  _active_preparation = std::make_shared<PendingPipelinePreparation>();
  _active_preparation->generation = _preparation_generation;
  _active_preparation->integrator_mode = _integrator_mode;
  _active_preparation->integrator_features = _integrator_features;
  _active_preparation->material_compile_mask = _material_compile_mask;
  _active_preparation->compile_stage_filter = _compile_stage_filter;
  _active_preparation->queued_at = std::chrono::steady_clock::now();
  _publish_preparation.reset();
  _published_pipeline_count = 0u;
  _publish_pipeline_index = 0u;
  _pipeline_publish_logged = false;
  _preparation_started_at = _active_preparation->queued_at;
  set_preparation_state(RendererPreparationState::Preparing, "Compiling shaders", reason ? (std::string("Queued (") + reason + ")") : std::string("Queued"));
  _inflight_preparation_tasks.push_back({
    .handle = scheduler.schedule(1u, [this, result = _active_preparation](uint32_t, uint32_t, uint32_t) { compile_pipeline_preparation(result); }),
    .result = _active_preparation,
  });
}

void GPURaytracingRenderer::poll_preparation_tasks(RHIContext& ctx, bool wait_for_active) {
  auto& device = ctx.device();
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
      destroy_pipelines(device);
      continue;
    }

    begin_pipeline_publish(result);
  }
}

bool GPURaytracingRenderer::begin_pipeline_publish(std::shared_ptr<PendingPipelinePreparation> result) {
  if (result == nullptr) {
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
  set_preparation_state(RendererPreparationState::Preparing, "Creating pipelines");
  return true;
}

bool GPURaytracingRenderer::advance_pipeline_publish(RHIContext& ctx, uint32_t max_pipelines) {
  if (_publish_preparation == nullptr) {
    return false;
  }

  auto& device = ctx.device();
  if (_pipeline_publish_logged == false) {
    _pipeline_publish_logged = true;
    destroy_pipelines(device);
    log::info("GPU RT preparation pipeline creation started: generation=%u stages=%u", _publish_preparation->generation, _publish_preparation->total_pipelines);
  }

  uint32_t created_this_call = 0u;
  while ((_publish_pipeline_index < _publish_preparation->compiled_stages.size()) && (created_this_call < std::max(1u, max_pipelines))) {
    const auto& stage = _publish_preparation->compiled_stages[_publish_pipeline_index];
    RHIComputePipelineDesc desc = device.make_compute_pipeline_desc(stage.binary);
    const auto stage_pipeline_create_begin = std::chrono::steady_clock::now();
    auto pipeline_result = device.create_compute_pipeline(desc);
    const auto stage_pipeline_create_end = std::chrono::steady_clock::now();
    const double stage_pipeline_create_ms = elapsed_ms(stage_pipeline_create_begin, stage_pipeline_create_end);
    _publish_preparation->publish_timings.push_back({
      .stage = stage.stage,
      .entry_point = stage.entry_point,
      .source_file = stage.source_file,
      .optimization_level = stage.optimization_level,
      .bsdf_kind = stage.bsdf_kind,
      .uses_stage_entry_define = stage.uses_stage_entry_define,
      .elapsed_ms = stage_pipeline_create_ms,
    });
    log::info(
      "GPU RT pipeline create: generation=%u index=%u/%u stage=%s entry=%s source=%s material=%s opt=%s entry_define=%s time=%.2fms",
      _publish_preparation->generation, _publish_pipeline_index + 1u, _publish_preparation->total_pipelines, pipeline_stage_to_string(stage.stage), stage.entry_point.c_str(),
      stage.source_file.empty() ? "-" : stage.source_file.c_str(), bsdf_kind_to_string(stage.bsdf_kind), stage.optimization_level.empty() ? "-" : stage.optimization_level.c_str(),
      stage.uses_stage_entry_define ? "yes" : "no", stage_pipeline_create_ms);
    if ((pipeline_result.result != RHIResult::Success) || (pipeline_result.handle.valid() == false)) {
      const std::string message = "GPU pipeline creation failed at '" + stage.entry_point + "'";
      set_runtime_failure(message);
      set_preparation_failed(message, "Pipeline creation failed");
      log::error("GPU RT preparation failed: generation=%u phase=pipeline stage=%s", _publish_preparation->generation, stage.entry_point.c_str());
      destroy_pipelines(device);
      _publish_preparation.reset();
      return false;
    }

    _pipelines[static_cast<uint32_t>(stage.stage)] = pipeline_result.handle;
    _publish_pipeline_index += 1u;
    _published_pipeline_count += 1u;
    created_this_call += 1u;
  }

  if (_publish_pipeline_index >= _publish_preparation->compiled_stages.size()) {
    const auto ready_at = std::chrono::steady_clock::now();
    std::vector<PipelinePublishTiming> slowest_pipelines = _publish_preparation->publish_timings;
    std::sort(slowest_pipelines.begin(), slowest_pipelines.end(), [](const auto& lhs, const auto& rhs) { return lhs.elapsed_ms > rhs.elapsed_ms; });
    const size_t slow_pipeline_count = std::min<size_t>(5u, slowest_pipelines.size());
    for (size_t i = 0; i < slow_pipeline_count; ++i) {
      const auto& timing = slowest_pipelines[i];
      log::info("GPU RT pipeline slowest[%zu]: generation=%u stage=%s entry=%s source=%s material=%s opt=%s entry_define=%s time=%.2fms", i + 1u,
        _publish_preparation->generation, pipeline_stage_to_string(timing.stage), timing.entry_point.c_str(),
        timing.source_file.empty() ? "-" : timing.source_file.c_str(), bsdf_kind_to_string(timing.bsdf_kind),
        timing.optimization_level.empty() ? "-" : timing.optimization_level.c_str(), timing.uses_stage_entry_define ? "yes" : "no", timing.elapsed_ms);
    }
    log::info("GPU RT preparation ready: generation=%u compile=%.2fms publish=%.2fms total=%.2fms", _publish_preparation->generation,
      elapsed_ms(_publish_preparation->compile_started_at, _publish_preparation->compile_finished_at), elapsed_ms(_pipeline_publish_started_at, ready_at),
      elapsed_ms(_preparation_started_at, ready_at));
    set_preparation_ready();
    _publish_preparation.reset();
  }

  return created_this_call > 0u;
}

bool GPURaytracingRenderer::create_pipelines_sync(RHIContext& ctx, SceneRepresentation& scene, const char* reason) {
  request_pipeline_preparation(scene, reason);
  return finish_preparation(ctx, scene);
}

bool GPURaytracingRenderer::finish_preparation(RHIContext& ctx, SceneRepresentation& scene) {
  (void)scene;
  while (_preparation_state == RendererPreparationState::Preparing) {
    poll_preparation_tasks(ctx, true);
    if (_publish_preparation) {
      advance_pipeline_publish(ctx, std::max(1u, _publish_preparation->total_pipelines));
    }
  }

  return pipelines_valid();
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
    if (_pipelines[static_cast<uint32_t>(stage_info.stage)].valid() == false) {
      return false;
    }
  }

  return true;
}

void GPURaytracingRenderer::cancel_preparation() {
  if (_preparation_state != RendererPreparationState::Preparing) {
    return;
  }

  _preparation_generation += 1u;
  _active_preparation.reset();
  _publish_preparation.reset();
  _published_pipeline_count = 0u;
  _publish_pipeline_index = 0u;
  log::info("GPU RT preparation canceled: generation=%u", _preparation_generation - 1u);
  set_preparation_failed("Preparation canceled", "Canceled");
}

void GPURaytracingRenderer::stop() {
  cancel_preparation();
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
  _wavefront_tile_index = 0u;
  _wavefront_tile_max_pixels = 0u;
  _wavefront_tile_count = 1u;
  _wavefront_tile_path_capacity = 0u;
  _wavefront_tile_base_origin = {};
  _wavefront_tile_base_size = {};
  _wavefront_tile_plan_valid = false;
  _wavefront_camera_phase_initialized = false;
  _camera_queue_count_readback_state = RHIResourceState::Undefined;
  _light_queue_count_readback_state = RHIResourceState::Undefined;
}

bool GPURaytracingRenderer::ensure_wavefront_buffers(RHIContext& ctx, const SceneRepresentation& scene, uint32_t path_capacity) {
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
  const uint32_t light_history_bounces = enable_light_path ? (enable_connect_vertices ? scene_max_path_length : kWavefrontLightHistoryBounces) : 0u;
  const uint32_t wavefront_hard_iteration_cap = scene_max_path_length;
  const uint64_t camera_vertex_capacity_u64 = static_cast<uint64_t>(path_capacity) * static_cast<uint64_t>(camera_history_bounces + 1u);
  if (camera_vertex_capacity_u64 > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
    log::error("GPU RT: wavefront camera vertex capacity overflow");
    return false;
  }
  const uint64_t light_vertex_capacity_u64 = static_cast<uint64_t>(path_capacity) * static_cast<uint64_t>(light_history_bounces + 1u);
  if (light_vertex_capacity_u64 > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
    log::error("GPU RT: wavefront light vertex capacity overflow");
    return false;
  }
  const uint32_t camera_vertex_capacity = static_cast<uint32_t>(camera_vertex_capacity_u64);
  const uint32_t light_vertex_capacity = static_cast<uint32_t>(light_vertex_capacity_u64);

  const uint64_t queue_buffer_size = kGPUWavefrontQueueHeaderSize + static_cast<uint64_t>(path_capacity) * sizeof(uint32_t);
  const uint64_t path_state_buffer_size = static_cast<uint64_t>(path_capacity) * kGPUWavefrontPathStateStride;
  const uint64_t hit_buffer_size = static_cast<uint64_t>(path_capacity) * kGPUWavefrontHitStride;
  const uint64_t camera_vertex_buffer_size = static_cast<uint64_t>(camera_vertex_capacity) * kGPUWavefrontPathVertexStride;
  const uint64_t light_vertex_buffer_size = static_cast<uint64_t>(light_vertex_capacity) * kGPUWavefrontLightPathVertexStride;
  const uint64_t film_buffer_size = film_pixel_count_u64 * sizeof(float4);
  const uint64_t path_meta_buffer_size = static_cast<uint64_t>(path_capacity) * kGPUWavefrontPathMetaStride;
  const uint64_t direct_light_sample_buffer_size = static_cast<uint64_t>(path_capacity) * kGPUWavefrontDirectLightSampleStride;
  const uint64_t direct_light_task_buffer_size = static_cast<uint64_t>(path_capacity) * kGPUWavefrontDirectLightTaskStride;
  const uint64_t direct_light_result_buffer_size = static_cast<uint64_t>(path_capacity) * kGPUWavefrontDirectLightResultStride;
  const uint64_t connect_light_task_count = static_cast<uint64_t>(path_capacity) * static_cast<uint64_t>(kWavefrontConnectLightBatchSize);
  const uint64_t connect_light_task_buffer_size = connect_light_task_count * kGPUWavefrontConnectLightTaskStride;
  const uint64_t connect_light_result_buffer_size = connect_light_task_count * kGPUWavefrontConnectLightResultStride;
  const uint64_t connect_camera_task_buffer_size = static_cast<uint64_t>(path_capacity) * kGPUWavefrontConnectCameraTaskStride;
  const uint64_t connect_camera_result_buffer_size = static_cast<uint64_t>(path_capacity) * kGPUWavefrontConnectCameraResultStride;
  const uint64_t subsurface_state_buffer_size = static_cast<uint64_t>(path_capacity) * kGPUWavefrontSubsurfaceStateStride;
  const RHIBufferUsage wavefront_usage = RHIBufferUsage::Storage | RHIBufferUsage::TransferDst;
  const RHIBufferUsage queue_buffer_usage = RHIBufferUsage::Storage | RHIBufferUsage::TransferDst | RHIBufferUsage::TransferSrc;
  const RHIBufferUsage queue_readback_usage = RHIBufferUsage::TransferDst;

  const auto validate_wavefront_buffer_size = [](const char* buffer_name, uint64_t size) {
    if (size > kWavefrontMaxAddressableBufferSize) {
      log::error("GPU RT: '%s' wavefront buffer exceeds 32-bit shader byte-address range (%llu bytes)", buffer_name, size);
      return false;
    }
    return true;
  };

  bool wavefront_buffer_sizes_valid = (validate_wavefront_buffer_size("wavefront_film", film_buffer_size) &&
                                       validate_wavefront_buffer_size("wavefront_path_meta", path_meta_buffer_size));
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
  if (enable_connect_to_light) {
    wavefront_buffer_sizes_valid = wavefront_buffer_sizes_valid && validate_wavefront_buffer_size("wavefront_direct_light_sample", direct_light_sample_buffer_size) &&
                                   validate_wavefront_buffer_size("wavefront_direct_light_task", direct_light_task_buffer_size) &&
                                   validate_wavefront_buffer_size("wavefront_direct_light_result", direct_light_result_buffer_size);
  }
  if (enable_connect_vertices) {
    wavefront_buffer_sizes_valid = wavefront_buffer_sizes_valid && validate_wavefront_buffer_size("wavefront_connect_light_task", connect_light_task_buffer_size) &&
                                   validate_wavefront_buffer_size("wavefront_connect_light_result", connect_light_result_buffer_size);
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
    if (ensure_storage_buffer(device, hit_buffer_size, wavefront_usage, _camera_hit_buffer, _camera_hit_buffer_size, _camera_hit_buffer_descriptor_index,
          "wavefront_camera_hit") == false) {
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
  if (enable_camera_path) {
    const bool recreate_readback_buffer =
      (_camera_queue_count_readback_buffer.valid() == false) || (_camera_queue_count_readback_buffer_size != kGPUWavefrontQueueHeaderSize);
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
    const bool recreate_readback_buffer =
      (_light_queue_count_readback_buffer.valid() == false) || (_light_queue_count_readback_buffer_size != kGPUWavefrontQueueHeaderSize);
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
    if (ensure_storage_buffer(device, light_vertex_buffer_size, wavefront_usage, _light_vertex_buffer, _light_vertex_buffer_size, _light_vertex_buffer_descriptor_index,
          "wavefront_light_vertex") == false) {
      return false;
    }
  } else {
    destroy_linear_scene_buffer(device, _light_vertex_buffer, _light_vertex_buffer_size, _light_vertex_buffer_descriptor_index);
  }
  if (ensure_storage_buffer(device, film_buffer_size, wavefront_usage, _film_buffer, _film_buffer_size, _film_buffer_descriptor_index, "wavefront_film") == false) {
    return false;
  }
  if (ensure_storage_buffer(device, path_meta_buffer_size, wavefront_usage, _path_meta_buffer, _path_meta_buffer_size, _path_meta_buffer_descriptor_index, "wavefront_path_meta") ==
      false) {
    return false;
  }
  if (enable_connect_to_light) {
    if (ensure_storage_buffer(device, direct_light_sample_buffer_size, wavefront_usage, _direct_light_sample_buffer, _direct_light_sample_buffer_size,
          _direct_light_sample_buffer_descriptor_index, "wavefront_direct_light_sample") == false) {
      return false;
    }
  } else {
    destroy_linear_scene_buffer(device, _direct_light_sample_buffer, _direct_light_sample_buffer_size, _direct_light_sample_buffer_descriptor_index);
  }
  if (enable_connect_to_light) {
    if (ensure_storage_buffer(device, direct_light_task_buffer_size, wavefront_usage, _direct_light_task_buffer, _direct_light_task_buffer_size,
          _direct_light_task_buffer_descriptor_index, "wavefront_direct_light_task") == false) {
      return false;
    }
  } else {
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
    if (ensure_storage_buffer(device, connect_light_result_buffer_size, wavefront_usage, _connect_light_result_buffer, _connect_light_result_buffer_size,
          _connect_light_result_buffer_descriptor_index, "wavefront_connect_light_result") == false) {
      return false;
    }
  } else {
    destroy_linear_scene_buffer(device, _connect_light_task_buffer, _connect_light_task_buffer_size, _connect_light_task_buffer_descriptor_index);
    destroy_linear_scene_buffer(device, _connect_light_result_buffer, _connect_light_result_buffer_size, _connect_light_result_buffer_descriptor_index);
  }
  if (enable_connect_to_camera) {
    if (ensure_storage_buffer(device, connect_camera_task_buffer_size, wavefront_usage, _connect_camera_task_buffer, _connect_camera_task_buffer_size,
          _connect_camera_task_buffer_descriptor_index, "wavefront_connect_camera_task") == false) {
      return false;
    }
    if (ensure_storage_buffer(device, connect_camera_result_buffer_size, wavefront_usage, _connect_camera_result_buffer, _connect_camera_result_buffer_size,
          _connect_camera_result_buffer_descriptor_index, "wavefront_connect_camera_result") == false) {
      return false;
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
  resources.camera_subsurface_state_buffer = _camera_subsurface_state_buffer_descriptor_index;
  resources.light_subsurface_state_buffer = _light_subsurface_state_buffer_descriptor_index;
  resources.path_capacity = path_capacity;
  resources.max_path_length = wavefront_hard_iteration_cap;
  resources.camera_vertex_capacity = camera_vertex_capacity;
  resources.light_vertex_capacity = light_vertex_capacity;
  resources.camera_fixed_max_bounces = camera_history_bounces;
  resources.light_fixed_max_bounces = light_history_bounces;

  if (upload_or_update_linear_scene_buffer(device, &resources, size_t(1), wavefront_usage, _wavefront_resources_buffer, _wavefront_resources_buffer_size,
        _wavefront_resources_buffer_descriptor_index, "wavefront_resources") == false) {
    return false;
  }

  _wavefront_path_capacity = path_capacity;
  _wavefront_vertex_capacity = std::max(camera_vertex_capacity, light_vertex_capacity);
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

  poll_preparation_tasks(ctx);
  if (_publish_preparation) {
    advance_pipeline_publish(ctx, 1u);
  }

  const GPUIntegratorSelection integrator_selection = gpu_integrator_selection_from_scene(scene);
  const uint32_t new_integrator_mode = static_cast<uint32_t>(integrator_selection.mode);
  const uint32_t new_integrator_features = integrator_selection.features;
  const uint32_t new_material_compile_mask = build_material_compile_mask(scene.data());
  _last_target_samples = std::max(1u, scene.data().options.samples);
  const bool integrator_mode_changed = (_integrator_mode != new_integrator_mode);
  const bool integrator_features_changed = (_integrator_features != new_integrator_features);
  const bool material_compile_mask_changed = (_material_compile_mask != new_material_compile_mask);
  const bool missing_pipelines = (_preparation_state == RendererPreparationState::Ready) && (pipelines_valid() == false);
  const bool should_request_prepare =
    integrator_selection.supported &&
    (integrator_mode_changed || integrator_features_changed || material_compile_mask_changed || missing_pipelines || (_preparation_state == RendererPreparationState::Failed));
  if (should_request_prepare) {
    const auto pipeline_refresh_begin = std::chrono::steady_clock::now();
    request_pipeline_preparation(scene, (integrator_mode_changed || integrator_features_changed || material_compile_mask_changed) ? "scene pipeline change" : "missing pipelines");
    const auto pipeline_refresh_end = std::chrono::steady_clock::now();
    pipeline_refresh_ms = elapsed_ms(pipeline_refresh_begin, pipeline_refresh_end);
  }

  if (integrator_selection.supported == false) {
    set_runtime_failure(gpu_integrator_selection_error_message(integrator_selection));
    return;
  }

  const bool preview_pipeline_ready = _preview_active && create_preview_pipeline(ctx);
  const bool full_pipeline_ready = pipelines_valid();
  if ((preview_pipeline_ready == false) && (full_pipeline_ready == false)) {
    return;
  }

  auto& device = ctx.device();
  _last_memory_stats = device.get_memory_statistics();
  const bool render_preview_this_frame = _preview_active && preview_pipeline_ready;

  const bool scene_check_requested = consume_scene_update_request();

  SceneHashes new_hashes = _current_scene_hashes;
  UpdateFlags changes = {};
  bool scene_changed = false;
  if (scene_check_requested) {
    ETX_PROFILER_NAMED_SCOPE("gpu_rt_scene_hashes_and_changes");
    const auto scene_hash_begin = std::chrono::steady_clock::now();
    scene.data().images.load_images(scheduler);
    new_hashes = scene.data().compute_hashes();
    changes = new_hashes.compare(_current_scene_hashes);
    scene_changed = changes.any();
    const auto scene_hash_end = std::chrono::steady_clock::now();
    scene_hash_ms = elapsed_ms(scene_hash_begin, scene_hash_end);
  }

  const auto& camera = scene.camera();
  const uint64_t new_camera_hash = xxh64(&camera, sizeof(camera));
  const bool camera_changed = (new_camera_hash != _current_camera_hash);
  const bool restart_accumulation = scene_check_requested || integrator_mode_changed || integrator_features_changed || material_compile_mask_changed || camera_changed;
  if (restart_accumulation || scene_changed) {
    _frame_index = 0u;
    _sample_index = 0u;
    reset_render_timing();
    _wavefront_render_step = WavefrontRenderStep::InitSample;
    _wavefront_path_iteration = 0u;
    _wavefront_hard_iteration_cap = 0u;
    _wavefront_camera_queue_count = 0u;
    _wavefront_light_queue_count = 0u;
    _wavefront_light_max_path_length = 0u;
    _wavefront_tile_index = 0u;
    _wavefront_tile_max_pixels = 0u;
    _wavefront_tile_count = 1u;
    _wavefront_tile_path_capacity = 0u;
    _wavefront_tile_base_origin = {};
    _wavefront_tile_base_size = {};
    _wavefront_tile_plan_valid = false;
    _wavefront_camera_phase_initialized = false;
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
    const bool update_success = update_scene_data_partial(ctx, scene, changes);
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

  if (wavefront_sample_in_progress == false) {
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

  const uint2 full_dim = scene.camera().film_size;
  const bool has_render_window = (_render_window_size.x > 0u) && (_render_window_size.y > 0u);
  const uint2 base_render_origin = has_render_window ? _render_window_origin : uint2{};
  const uint2 base_render_dim = has_render_window ? _render_window_size : full_dim;
  const uint32_t scene_max_path_length_for_tiling = std::max(1u, scene.data().options.max_path_length);
  const bool use_wavefront_tiling = (render_preview_this_frame == false) && gpu_integrator_feature_enabled(_integrator_features, GPUIntegratorFeatures::ConnectVertices);
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
  uint32_t wavefront_path_capacity = base_render_pixel_count;
  uint2 active_base_origin = base_render_origin;
  uint2 active_base_dim = base_render_dim;
  if (use_wavefront_tiling) {
    const bool should_rebuild_tile_plan =
      (_wavefront_tile_plan_valid == false) || ((_wavefront_render_step == WavefrontRenderStep::InitSample) && (_wavefront_tile_index == 0u));
    if (should_rebuild_tile_plan) {
      bool scene_has_subsurface_material = false;
      for (const auto& material : scene.data().materials) {
        if (material.subsurface_cls != SubsurfaceMaterial::Disabled) {
          scene_has_subsurface_material = true;
          break;
        }
      }
      const uint64_t tile_budget = wavefront_tile_budget_bytes(_last_memory_stats);
      const uint64_t tile_bytes_per_path = wavefront_tile_bytes_per_path(_integrator_features, scene_has_subsurface_material, scene_max_path_length_for_tiling);
      _wavefront_tile_max_pixels = wavefront_tile_max_pixels(tile_budget, tile_bytes_per_path, base_render_pixel_count);
      _wavefront_tile_count = wavefront_tile_count(base_render_dim, _wavefront_tile_max_pixels);
      _wavefront_tile_path_capacity = std::min(_wavefront_tile_max_pixels, base_render_pixel_count);
      _wavefront_tile_base_origin = base_render_origin;
      _wavefront_tile_base_size = base_render_dim;
      _wavefront_tile_plan_valid = true;
    }
    tile_max_pixels = _wavefront_tile_max_pixels;
    wavefront_tile_count_value = _wavefront_tile_count;
    wavefront_path_capacity = _wavefront_tile_path_capacity;
    active_base_origin = _wavefront_tile_base_origin;
    active_base_dim = _wavefront_tile_base_size;
    if (_wavefront_tile_index >= wavefront_tile_count_value) {
      _wavefront_tile_index = 0u;
    }
  } else {
    _wavefront_tile_plan_valid = false;
  }
  const WavefrontWindow active_window =
    use_wavefront_tiling ? wavefront_tile_window(active_base_origin, active_base_dim, tile_max_pixels, _wavefront_tile_index) : WavefrontWindow{base_render_origin, base_render_dim};
  const uint2 render_dim = active_window.size;
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
    RHICommandBuffer init_texture_cmd = ctx.get_command_buffer();
    if (init_texture_cmd.valid()) {
      ctx.command_buffer_begin(init_texture_cmd);
      ctx.cmd_texture_barrier(init_texture_cmd, _output_texture, _output_texture_state, RHIResourceState::ShaderReadOnly);
      ctx.command_buffer_end(init_texture_cmd);
      ctx.submit_command_buffer({init_texture_cmd});
      const RHIResult init_texture_wait = ctx.wait_for_command_buffer(init_texture_cmd);
      ctx.destroy_command_buffer(init_texture_cmd);
      if (init_texture_wait == RHIResult::Success) {
        _output_texture_state = RHIResourceState::ShaderReadOnly;
      } else {
        log::warning("GPU RT: failed to initialize output texture layout (%u)", static_cast<uint32_t>(init_texture_wait));
      }
    }
    _wavefront_render_step = WavefrontRenderStep::InitSample;
    _wavefront_path_iteration = 0u;
    _wavefront_hard_iteration_cap = 0u;
    _wavefront_camera_queue_count = 0u;
    _wavefront_light_queue_count = 0u;
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
    .connect_light_vertex_length = 0u,
    .render_window_origin_x = active_window.origin.x,
    .render_window_origin_y = active_window.origin.y,
    .render_window_width = render_dim.x,
    .render_window_height = render_dim.y,
    .dispatch_item_offset = 0u,
    .dispatch_item_count = 0u,
    .pad2 = 0u,
    .scene = _gpu_scene,
  };
  const RHIDispatchDesc film_dispatch = {
    .group_count_x = (render_dim.x + 7u) / 8u,
    .group_count_y = (render_dim.y + 7u) / 8u,
    .group_count_z = 1u,
  };

  if (render_preview_this_frame) {
    const auto preview_dispatch_begin = std::chrono::steady_clock::now();
    const bool preview_success = render_preview(ctx, frame_data.cmd, constants, film_dispatch);
    dispatch_submit_ms = elapsed_ms(preview_dispatch_begin, std::chrono::steady_clock::now());
    if (preview_success) {
      _frame_index += 1u;
      return;
    }

    if (full_pipeline_ready == false) {
      return;
    }
  }

  if (full_pipeline_ready == false) {
    return;
  }

  if (_sample_index >= std::max(1u, scene.data().options.samples)) {
    return;
  }

  if (_render_timing_active == false) {
    _render_started_at = std::chrono::steady_clock::now();
    _last_render_elapsed_seconds = 0.0;
    _render_timing_active = true;
  }

  const auto wavefront_buffer_begin = std::chrono::steady_clock::now();
  if (ensure_wavefront_buffers(ctx, scene, wavefront_path_capacity) == false) {
    log::error("GPU RT: failed to allocate wavefront buffers");
    return;
  }
  const auto wavefront_buffer_end = std::chrono::steady_clock::now();
  wavefront_buffer_ms = elapsed_ms(wavefront_buffer_begin, wavefront_buffer_end);
  constants.wavefront_buffer_index = _wavefront_resources_buffer_descriptor_index;

  bool completed_sample = false;
  {
    ETX_PROFILER_NAMED_SCOPE("gpu_rt_dispatch_and_submit");
    const auto dispatch_submit_begin = std::chrono::steady_clock::now();
    const auto dispatch_stage_with_connect_light_length = [&](RHICommandBuffer cmd, PipelineStage stage, const RHIDispatchDesc& dispatch, uint32_t path_iteration,
                                                            uint32_t connect_light_vertex_length, uint32_t connect_light_vertex_count) {
      GPURTConstants stage_constants = constants;
      stage_constants.path_iteration = path_iteration;
      stage_constants.connect_light_vertex_length = connect_light_vertex_length;
      stage_constants.dispatch_item_count = connect_light_vertex_count;
      ctx.cmd_set_pipeline(cmd, _pipelines[static_cast<uint32_t>(stage)]);
      ctx.cmd_push_constants(cmd, &stage_constants, sizeof(stage_constants));
      ctx.cmd_dispatch(cmd, dispatch);
    };
    const auto dispatch_stage = [&](RHICommandBuffer cmd, PipelineStage stage, const RHIDispatchDesc& dispatch, uint32_t path_iteration) {
      GPURTConstants stage_constants = constants;
      stage_constants.path_iteration = path_iteration;
      ctx.cmd_set_pipeline(cmd, _pipelines[static_cast<uint32_t>(stage)]);
      ctx.cmd_push_constants(cmd, &stage_constants, sizeof(stage_constants));
      ctx.cmd_dispatch(cmd, dispatch);
    };
    const auto dispatch_stage_window = [&](RHICommandBuffer cmd, PipelineStage stage, uint32_t item_offset, uint32_t item_count, uint32_t path_iteration) {
      ctx.cmd_set_pipeline(cmd, _pipelines[static_cast<uint32_t>(stage)]);
      GPURTConstants stage_constants = constants;
      stage_constants.path_iteration = path_iteration;
      stage_constants.dispatch_item_offset = item_offset;
      stage_constants.dispatch_item_count = item_count;
      const RHIDispatchDesc chunk_dispatch = {
        .group_count_x = (item_count + 63u) / 64u,
        .group_count_y = 1u,
        .group_count_z = 1u,
      };
      ctx.cmd_push_constants(cmd, &stage_constants, sizeof(stage_constants));
      ctx.cmd_dispatch(cmd, chunk_dispatch);
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
        _camera_subsurface_state_buffer,
        _light_subsurface_state_buffer,
      };

      for (const auto& buffer : buffers) {
        if (buffer.valid()) {
          ctx.cmd_buffer_barrier(cmd, buffer, RHIResourceState::General, RHIResourceState::General);
        }
      }
    };

    const RHIDispatchDesc scalar_dispatch = {
      .group_count_x = 1u,
      .group_count_y = 1u,
      .group_count_z = 1u,
    };
    const uint32_t scene_max_path_length = std::max(1u, scene.data().options.max_path_length);
    const GPUIntegratorMode integrator_mode = static_cast<GPUIntegratorMode>(_integrator_mode);
    const uint32_t wavefront_hard_iteration_cap = scene_max_path_length;
    std::vector<RHICommandBuffer> submitted_commands = {};
    submitted_commands.reserve(2u);
    const auto record_and_submit = [&](const auto& record_commands) {
      RHICommandBuffer cmd = ctx.get_command_buffer();
      ctx.command_buffer_begin(cmd);
      record_commands(cmd);
      ctx.command_buffer_end(cmd);
      ctx.submit_command_buffer({cmd});
      submitted_commands.push_back(cmd);
    };
    const auto wait_and_destroy_submitted_commands = [&](const char* stage_name) {
      RHIResult wait_result = RHIResult::Success;
      for (const auto cmd : submitted_commands) {
        const RHIResult command_wait_result = ctx.wait_for_command_buffer(cmd);
        if ((wait_result == RHIResult::Success) && (command_wait_result != RHIResult::Success)) {
          wait_result = command_wait_result;
        }
      }
      if (wait_result != RHIResult::Success) {
        log::warning("GPU RT: command wait failed after %s (%u)", stage_name, static_cast<uint32_t>(wait_result));
      }
      for (const auto cmd : submitted_commands) {
        ctx.destroy_command_buffer(cmd);
      }
      submitted_commands.clear();
      return wait_result;
    };
    const auto submit_stage_chunked = [&](PipelineStage stage, uint32_t item_count, uint32_t path_iteration, uint32_t chunk_size, const char* stage_name) {
      for (uint32_t item_offset = 0u; item_offset < item_count; item_offset += chunk_size) {
        const uint32_t chunk_count = std::min(chunk_size, item_count - item_offset);
        record_and_submit([&](RHICommandBuffer cmd) {
          barrier_wavefront_buffers(cmd);
          dispatch_stage_window(cmd, stage, item_offset, chunk_count, path_iteration);
          barrier_wavefront_buffers(cmd);
        });
        const RHIResult chunk_result = wait_and_destroy_submitted_commands(stage_name);
        if (chunk_result != RHIResult::Success) {
          return chunk_result;
        }
      }
      return RHIResult::Success;
    };
    const bool enable_camera_path = gpu_integrator_feature_enabled(_integrator_features, GPUIntegratorFeatures::CameraPath);
    const bool enable_light_path = gpu_integrator_feature_enabled(_integrator_features, GPUIntegratorFeatures::LightPath);
    const bool enable_direct_hit = gpu_integrator_feature_enabled(_integrator_features, GPUIntegratorFeatures::DirectHit);
    const bool enable_connect_to_light = gpu_integrator_feature_enabled(_integrator_features, GPUIntegratorFeatures::ConnectToLight);
    const bool enable_connect_to_camera = gpu_integrator_feature_enabled(_integrator_features, GPUIntegratorFeatures::ConnectToCamera);
    const bool enable_connect_vertices = gpu_integrator_feature_enabled(_integrator_features, GPUIntegratorFeatures::ConnectVertices);
    const bool phase_light_before_camera =
      (integrator_mode == GPUIntegratorMode::BDPTFull) && enable_camera_path && enable_light_path && enable_connect_vertices;
    const bool has_various_continue = material_compile_mask_has_various_continue(_material_compile_mask);
    const bool has_various_connect = material_compile_mask_has_various_connect(_material_compile_mask);
    const bool has_plastic = material_compile_mask_has(_material_compile_mask, MaterialClass::Plastic);
    const bool has_conductor = material_compile_mask_has_conductor_stage(_material_compile_mask);
    const bool has_dielectric = material_compile_mask_has(_material_compile_mask, MaterialClass::Dielectric);
    const bool has_thinfilm = material_compile_mask_has(_material_compile_mask, MaterialClass::Thinfilm);
    const uint32_t light_history_bounces = enable_connect_vertices ? scene_max_path_length : kWavefrontLightHistoryBounces;
    const uint32_t render_pixel_count = render_dim.x * render_dim.y;
    const uint32_t initial_camera_queue_count = enable_camera_path ? render_pixel_count : 0u;
    const uint32_t initial_light_queue_count = enable_light_path ? render_pixel_count : 0u;
    const bool batch_coarse_progress = _batch_coarse_progress && (scene_max_path_length > _wavefront_steps_per_render);
    const uint32_t batch_queue_readback_interval = batch_coarse_progress ? std::max(1u, _wavefront_steps_per_render) : 1u;
    const auto queue_dispatch_count = [&](uint32_t queue_count) {
      return batch_coarse_progress ? std::min(wavefront_path_capacity, std::max(queue_count, 1u)) : queue_count;
    };
    if ((_sample_index == 0u) && (_frame_index == 0u)) {
      log::info("GPU path mode: %s", gpu_integrator_mode_to_string(integrator_mode));
    }
    const auto finalize_wavefront_sample = [&]() {
      record_and_submit([&](RHICommandBuffer cmd) {
        ctx.cmd_texture_barrier(cmd, _output_texture, _output_texture_state, RHIResourceState::General);
        barrier_wavefront_buffers(cmd);
        dispatch_stage(cmd, PipelineStage::FinalizeSample, film_dispatch, _wavefront_hard_iteration_cap);
        ctx.cmd_texture_barrier(cmd, _output_texture, RHIResourceState::General, RHIResourceState::ShaderReadOnly);
      });

      wait_and_destroy_submitted_commands("finalize sample submit");
      _output_texture_state = RHIResourceState::ShaderReadOnly;
      _wavefront_render_step = WavefrontRenderStep::InitSample;
      _wavefront_path_iteration = 0u;
      _wavefront_hard_iteration_cap = 0u;
      _wavefront_camera_queue_count = 0u;
      _wavefront_light_queue_count = 0u;
      _wavefront_light_max_path_length = 0u;
      _wavefront_camera_phase_initialized = false;
      if (_wavefront_tile_index + 1u >= wavefront_tile_count_value) {
        _wavefront_tile_index = 0u;
        _wavefront_tile_plan_valid = false;
        completed_sample = true;
      } else {
        _wavefront_tile_index += 1u;
      }
      dispatch_submit_ms = elapsed_ms(dispatch_submit_begin, std::chrono::steady_clock::now());
    };
    const auto initialize_deferred_camera_phase = [&]() {
      _wavefront_path_iteration = 0u;
      _wavefront_camera_queue_count = initial_camera_queue_count;
      _wavefront_light_queue_count = 0u;
      _wavefront_camera_phase_initialized = true;

      record_and_submit([&](RHICommandBuffer cmd) {
        barrier_wavefront_buffers(cmd);
        dispatch_stage(cmd, PipelineStage::InitCameraPath0, film_dispatch, 0u);
        barrier_wavefront_buffers(cmd);
      });

      wait_and_destroy_submitted_commands("deferred camera init submit");
      dispatch_submit_ms = elapsed_ms(dispatch_submit_begin, std::chrono::steady_clock::now());
    };

    bool finished_current_tile = false;
    const uint32_t wavefront_step_budget = std::max(1u, _wavefront_steps_per_render);
    for (uint32_t wavefront_step_index = 0u; (wavefront_step_index < wavefront_step_budget) && (finished_current_tile == false); ++wavefront_step_index) {
      if (_wavefront_render_step == WavefrontRenderStep::InitSample) {
        _wavefront_path_iteration = 0u;
        _wavefront_hard_iteration_cap = wavefront_hard_iteration_cap;
        _wavefront_camera_phase_initialized = enable_camera_path && (phase_light_before_camera == false);
        _wavefront_camera_queue_count = _wavefront_camera_phase_initialized ? initial_camera_queue_count : 0u;
        _wavefront_light_queue_count = initial_light_queue_count;
        _wavefront_light_max_path_length = 0u;

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

        wait_and_destroy_submitted_commands("init sample submit");
        _wavefront_render_step = WavefrontRenderStep::TraceBounce;
        dispatch_submit_ms = elapsed_ms(dispatch_submit_begin, std::chrono::steady_clock::now());
      } else if (_wavefront_render_step == WavefrontRenderStep::TraceBounce) {
        const bool waiting_for_deferred_camera_phase =
          phase_light_before_camera && enable_camera_path && (_wavefront_camera_phase_initialized == false);
        if (waiting_for_deferred_camera_phase && ((_wavefront_light_queue_count == 0u) || (_wavefront_path_iteration >= _wavefront_hard_iteration_cap))) {
          initialize_deferred_camera_phase();
        } else if ((_wavefront_camera_queue_count == 0u) && (_wavefront_light_queue_count == 0u)) {
          finalize_wavefront_sample();
        } else if (_wavefront_path_iteration >= _wavefront_hard_iteration_cap) {
          finalize_wavefront_sample();
        } else {
          const uint32_t path_iteration = _wavefront_path_iteration;
          const uint32_t camera_dispatch_count = queue_dispatch_count(_wavefront_camera_queue_count);
          const uint32_t light_dispatch_count = queue_dispatch_count(_wavefront_light_queue_count);
          const RHIDispatchDesc camera_queue_dispatch = {
            .group_count_x = (camera_dispatch_count + 63u) / 64u,
            .group_count_y = 1u,
            .group_count_z = 1u,
          };
          const RHIDispatchDesc light_queue_dispatch = {
            .group_count_x = (light_dispatch_count + 63u) / 64u,
            .group_count_y = 1u,
            .group_count_z = 1u,
          };
          const bool continue_paths = (path_iteration + 1u) < _wavefront_hard_iteration_cap;
          const bool queue_readback_due =
            (batch_coarse_progress == false) || (((path_iteration + 1u) % batch_queue_readback_interval) == 0u) || (continue_paths == false);
          const bool copy_queue_counts = continue_paths && queue_readback_due;
          const RHIBindlessHandle next_camera_queue_buffer = ((path_iteration & 1u) == 0u) ? _camera_queue_b_buffer : _camera_queue_a_buffer;
          const RHIBindlessHandle next_light_queue_buffer = ((path_iteration & 1u) == 0u) ? _light_queue_b_buffer : _light_queue_a_buffer;

          constexpr uint32_t kHeavyContinuationChunkSize = 65536u;
          RHIResult trace_step_result = RHIResult::Success;

          record_and_submit([&](RHICommandBuffer cmd) {
            barrier_wavefront_buffers(cmd);
            if (_wavefront_light_queue_count > 0u) {
              dispatch_stage(cmd, PipelineStage::TraceLight, light_queue_dispatch, path_iteration);
              barrier_wavefront_buffers(cmd);
              dispatch_stage(cmd, PipelineStage::LightSurfaceClassify, light_queue_dispatch, path_iteration);
              barrier_wavefront_buffers(cmd);
            }
            if (_wavefront_camera_queue_count > 0u) {
              dispatch_stage(cmd, PipelineStage::TraceCamera, camera_queue_dispatch, path_iteration);
              barrier_wavefront_buffers(cmd);
              dispatch_stage(cmd, PipelineStage::CameraSurfaceClassify, camera_queue_dispatch, path_iteration);
              barrier_wavefront_buffers(cmd);
            }
          });
          trace_step_result = wait_and_destroy_submitted_commands("trace/classify bounce submit");

          const bool submit_non_dielectric_continue = continue_paths && (has_various_continue || has_plastic || has_conductor || has_thinfilm);
          if ((trace_step_result == RHIResult::Success) && submit_non_dielectric_continue) {
            record_and_submit([&](RHICommandBuffer cmd) {
              barrier_wavefront_buffers(cmd);
              if (_wavefront_light_queue_count > 0u) {
                if (has_various_continue) {
                  dispatch_stage(cmd, PipelineStage::LightContinuePrepareDiffuse, light_queue_dispatch, path_iteration);
                  barrier_wavefront_buffers(cmd);
                }
                if (has_plastic) {
                  dispatch_stage(cmd, PipelineStage::LightContinuePreparePlastic, light_queue_dispatch, path_iteration);
                  barrier_wavefront_buffers(cmd);
                }
                if (has_conductor) {
                  dispatch_stage(cmd, PipelineStage::LightContinuePrepareConductor, light_queue_dispatch, path_iteration);
                  barrier_wavefront_buffers(cmd);
                }
                if (has_thinfilm) {
                  dispatch_stage(cmd, PipelineStage::LightContinuePrepareThinfilm, light_queue_dispatch, path_iteration);
                  barrier_wavefront_buffers(cmd);
                }
              }
              if (_wavefront_camera_queue_count > 0u) {
                if (has_various_continue) {
                  dispatch_stage(cmd, PipelineStage::CameraContinuePrepareDiffuse, camera_queue_dispatch, path_iteration);
                  barrier_wavefront_buffers(cmd);
                }
                if (has_plastic) {
                  dispatch_stage(cmd, PipelineStage::CameraContinuePreparePlastic, camera_queue_dispatch, path_iteration);
                  barrier_wavefront_buffers(cmd);
                }
                if (has_conductor) {
                  dispatch_stage(cmd, PipelineStage::CameraContinuePrepareConductor, camera_queue_dispatch, path_iteration);
                  barrier_wavefront_buffers(cmd);
                }
                if (has_thinfilm) {
                  dispatch_stage(cmd, PipelineStage::CameraContinuePrepareThinfilm, camera_queue_dispatch, path_iteration);
                  barrier_wavefront_buffers(cmd);
                }
              }
            });
            trace_step_result = wait_and_destroy_submitted_commands("continue prepare submit");
          }

          if ((trace_step_result == RHIResult::Success) && continue_paths && has_dielectric && (_wavefront_light_queue_count > 0u)) {
            trace_step_result = submit_stage_chunked(PipelineStage::LightContinuePrepareDielectric, _wavefront_light_queue_count, path_iteration,
              kHeavyContinuationChunkSize, "light dielectric continue prepare submit");
          }

          if ((trace_step_result == RHIResult::Success) && continue_paths && has_dielectric && (_wavefront_camera_queue_count > 0u)) {
            trace_step_result = submit_stage_chunked(PipelineStage::CameraContinuePrepareDielectric, _wavefront_camera_queue_count, path_iteration,
              kHeavyContinuationChunkSize, "camera dielectric continue prepare submit");
          }

          if (trace_step_result == RHIResult::Success) {
            record_and_submit([&](RHICommandBuffer cmd) {
              barrier_wavefront_buffers(cmd);
              if (_wavefront_light_queue_count > 0u) {
                if (enable_connect_to_camera) {
                  dispatch_stage(cmd, PipelineStage::LightConnectCameraClear, light_queue_dispatch, path_iteration);
                  barrier_wavefront_buffers(cmd);
                  if (has_various_connect) {
                    dispatch_stage(cmd, PipelineStage::LightConnectCameraPrepareDiffuse, light_queue_dispatch, path_iteration);
                    barrier_wavefront_buffers(cmd);
                  }
                  if (has_plastic) {
                    dispatch_stage(cmd, PipelineStage::LightConnectCameraPreparePlastic, light_queue_dispatch, path_iteration);
                    barrier_wavefront_buffers(cmd);
                  }
                  if (has_conductor) {
                    dispatch_stage(cmd, PipelineStage::LightConnectCameraPrepareConductor, light_queue_dispatch, path_iteration);
                    barrier_wavefront_buffers(cmd);
                  }
                  if (has_dielectric) {
                    dispatch_stage(cmd, PipelineStage::LightConnectCameraPrepareDielectric, light_queue_dispatch, path_iteration);
                    barrier_wavefront_buffers(cmd);
                  }
                  dispatch_stage(cmd, PipelineStage::LightConnectCameraShadow, light_queue_dispatch, path_iteration);
                  barrier_wavefront_buffers(cmd);
                  dispatch_stage(cmd, PipelineStage::LightConnectCameraAccumulate, light_queue_dispatch, path_iteration);
                  barrier_wavefront_buffers(cmd);
                }
                if (continue_paths) {
                  dispatch_stage(cmd, PipelineStage::LightContinueFinalize, light_queue_dispatch, path_iteration);
                  barrier_wavefront_buffers(cmd);
                }
              }
              if (_wavefront_camera_queue_count > 0u) {
                if (enable_connect_to_light) {
                  dispatch_stage(cmd, PipelineStage::CameraDirectLightSample, camera_queue_dispatch, path_iteration);
                  barrier_wavefront_buffers(cmd);
                  if (has_various_connect) {
                    dispatch_stage(cmd, PipelineStage::CameraDirectLightPrepareDiffuse, camera_queue_dispatch, path_iteration);
                    barrier_wavefront_buffers(cmd);
                  }
                  if (has_plastic) {
                    dispatch_stage(cmd, PipelineStage::CameraDirectLightPreparePlastic, camera_queue_dispatch, path_iteration);
                    barrier_wavefront_buffers(cmd);
                  }
                  if (has_conductor) {
                    dispatch_stage(cmd, PipelineStage::CameraDirectLightPrepareConductor, camera_queue_dispatch, path_iteration);
                    barrier_wavefront_buffers(cmd);
                  }
                  if (has_dielectric) {
                    dispatch_stage(cmd, PipelineStage::CameraDirectLightPrepareDielectric, camera_queue_dispatch, path_iteration);
                    barrier_wavefront_buffers(cmd);
                  }
                  dispatch_stage(cmd, PipelineStage::CameraDirectLightShadow, camera_queue_dispatch, path_iteration);
                  barrier_wavefront_buffers(cmd);
                  dispatch_stage(cmd, PipelineStage::CameraDirectLightAccumulate, camera_queue_dispatch, path_iteration);
                  barrier_wavefront_buffers(cmd);
                }
                if (enable_direct_hit) {
                  dispatch_stage(cmd, PipelineStage::CameraDirectHitAccumulate, camera_queue_dispatch, path_iteration);
                  barrier_wavefront_buffers(cmd);
                }
                if (enable_connect_vertices) {
                  const RHIDispatchDesc connect_light_dispatch = {
                    .group_count_x = (_wavefront_camera_queue_count + 63u) / 64u,
                    .group_count_y = 1u,
                    .group_count_z = 1u,
                  };
                  const uint32_t connect_light_history_bounces =
                    (phase_light_before_camera && _wavefront_camera_phase_initialized && (_wavefront_light_max_path_length > 0u))
                      ? std::min(light_history_bounces, _wavefront_light_max_path_length)
                      : light_history_bounces;
                  for (uint32_t light_vertex_length = 1u; light_vertex_length <= connect_light_history_bounces; light_vertex_length += kWavefrontConnectLightBatchSize) {
                    const uint32_t connect_light_vertex_count = std::min(kWavefrontConnectLightBatchSize, connect_light_history_bounces - light_vertex_length + 1u);
                    const RHIDispatchDesc connect_light_batch_dispatch = {
                      .group_count_x = connect_light_dispatch.group_count_x,
                      .group_count_y = connect_light_vertex_count,
                      .group_count_z = 1u,
                    };
                    dispatch_stage_with_connect_light_length(cmd, PipelineStage::CameraConnectLightClear, connect_light_batch_dispatch, path_iteration, light_vertex_length,
                      connect_light_vertex_count);
                    barrier_wavefront_buffers(cmd);
                    if (has_various_connect) {
                      dispatch_stage_with_connect_light_length(cmd, PipelineStage::CameraConnectLightPrepareDiffuse, connect_light_batch_dispatch, path_iteration, light_vertex_length,
                        connect_light_vertex_count);
                      barrier_wavefront_buffers(cmd);
                    }
                    if (has_plastic) {
                      dispatch_stage_with_connect_light_length(cmd, PipelineStage::CameraConnectLightPreparePlastic, connect_light_batch_dispatch, path_iteration, light_vertex_length,
                        connect_light_vertex_count);
                      barrier_wavefront_buffers(cmd);
                    }
                    if (has_conductor) {
                      dispatch_stage_with_connect_light_length(cmd, PipelineStage::CameraConnectLightPrepareConductor, connect_light_batch_dispatch, path_iteration, light_vertex_length,
                        connect_light_vertex_count);
                      barrier_wavefront_buffers(cmd);
                    }
                    if (has_dielectric) {
                      dispatch_stage_with_connect_light_length(cmd, PipelineStage::CameraConnectLightPrepareDielectric, connect_light_batch_dispatch, path_iteration, light_vertex_length,
                        connect_light_vertex_count);
                      barrier_wavefront_buffers(cmd);
                    }
                    if (has_various_connect) {
                      dispatch_stage_with_connect_light_length(cmd, PipelineStage::CameraConnectLightResolveDiffuse, connect_light_batch_dispatch, path_iteration, light_vertex_length,
                        connect_light_vertex_count);
                      barrier_wavefront_buffers(cmd);
                    }
                    if (has_plastic) {
                      dispatch_stage_with_connect_light_length(cmd, PipelineStage::CameraConnectLightResolvePlastic, connect_light_batch_dispatch, path_iteration, light_vertex_length,
                        connect_light_vertex_count);
                      barrier_wavefront_buffers(cmd);
                    }
                    if (has_conductor) {
                      dispatch_stage_with_connect_light_length(cmd, PipelineStage::CameraConnectLightResolveConductor, connect_light_batch_dispatch, path_iteration, light_vertex_length,
                        connect_light_vertex_count);
                      barrier_wavefront_buffers(cmd);
                    }
                    if (has_dielectric) {
                      dispatch_stage_with_connect_light_length(cmd, PipelineStage::CameraConnectLightResolveDielectric, connect_light_batch_dispatch, path_iteration, light_vertex_length,
                        connect_light_vertex_count);
                      barrier_wavefront_buffers(cmd);
                    }
                    dispatch_stage_with_connect_light_length(cmd, PipelineStage::CameraConnectLightShadow, connect_light_batch_dispatch, path_iteration, light_vertex_length,
                      connect_light_vertex_count);
                    barrier_wavefront_buffers(cmd);
                    dispatch_stage_with_connect_light_length(cmd, PipelineStage::CameraConnectLightAccumulate, connect_light_batch_dispatch, path_iteration, light_vertex_length,
                      connect_light_vertex_count);
                    barrier_wavefront_buffers(cmd);
                  }
                }
                if (continue_paths) {
                  dispatch_stage(cmd, PipelineStage::CameraContinueFinalize, camera_queue_dispatch, path_iteration);
                  barrier_wavefront_buffers(cmd);
                }
              }
              dispatch_stage(cmd, PipelineStage::SwapQueues, scalar_dispatch, path_iteration);
              barrier_wavefront_buffers(cmd);

              if (copy_queue_counts) {
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
            });

            trace_step_result = wait_and_destroy_submitted_commands("post-continue bounce submit");
          }

          if (trace_step_result != RHIResult::Success) {
            set_runtime_failure("GPU RT trace bounce submit failed (" + std::to_string(static_cast<uint32_t>(trace_step_result)) + ")");
            _wavefront_camera_queue_count = 0u;
            _wavefront_light_queue_count = 0u;
            return;
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
            if ((enable_connect_vertices) && enable_light_path) {
              _wavefront_light_max_path_length = light_history_bounces;
            }
          } else {
            _wavefront_camera_queue_count = 0u;
            _wavefront_light_queue_count = 0u;
          }

          const bool waiting_for_deferred_camera_phase_after_step =
            phase_light_before_camera && enable_camera_path && (_wavefront_camera_phase_initialized == false);
          if (waiting_for_deferred_camera_phase_after_step) {
            if (_wavefront_path_iteration >= _wavefront_hard_iteration_cap) {
              _wavefront_light_queue_count = 0u;
            }
          } else if ((_wavefront_path_iteration >= _wavefront_hard_iteration_cap) || ((_wavefront_camera_queue_count == 0u) && (_wavefront_light_queue_count == 0u))) {
            finalize_wavefront_sample();
          }
        }

        dispatch_submit_ms = elapsed_ms(dispatch_submit_begin, std::chrono::steady_clock::now());
      } else if (_wavefront_render_step == WavefrontRenderStep::FinalizeSample) {
        finalize_wavefront_sample();
      }

      if (_wavefront_render_step == WavefrontRenderStep::InitSample) {
        finished_current_tile = true;
      }
    }
  }

  _frame_index += 1u;
  if (completed_sample) {
    _sample_index += 1u;
    if (_sample_index >= _last_target_samples) {
      _last_render_elapsed_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - _render_started_at).count();
      _render_timing_active = false;
    }
  }
}

void GPURaytracingRenderer::cleanup(RHIContext& ctx) {
  ETX_PROFILER_SCOPE();

  _cleanup_wait_succeeded = false;

  cancel_preparation();
  release_inflight_preparation_tasks(true);

  auto& device = ctx.device();
  const RHIResult wait_result = ctx.wait_idle();
  if (wait_result != RHIResult::Success) {
    log::warning("GPU RT: wait_idle failed during cleanup (%u)", static_cast<uint32_t>(wait_result));
  } else {
    _cleanup_wait_succeeded = true;
  }

  destroy_preview_pipeline(device);
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
  _render_window_origin = {};
  _render_window_size = {};
  _active_preparation.reset();
  _publish_preparation.reset();
  _preparation_generation = 0u;
  _published_pipeline_count = 0u;
  _publish_pipeline_index = 0u;
  _pipeline_publish_logged = false;
  _preview_active = false;
  reset_runtime_failure();
  set_preparation_ready();
  request_scene_update();
}

void GPURaytracingRenderer::on_camera_changed(SceneRepresentation& scene) {
  ETX_PROFILER_SCOPE();
  (void)scene;
  _preview_active = true;
  _frame_index = 0u;
  _sample_index = 0u;
  reset_render_timing();
  _wavefront_render_step = WavefrontRenderStep::InitSample;
  _wavefront_path_iteration = 0u;
  _wavefront_hard_iteration_cap = 0u;
  _wavefront_camera_queue_count = 0u;
  _wavefront_light_queue_count = 0u;
  _wavefront_light_max_path_length = 0u;
  _wavefront_tile_index = 0u;
  _wavefront_tile_max_pixels = 0u;
  _wavefront_tile_count = 1u;
  _wavefront_tile_path_capacity = 0u;
  _wavefront_tile_base_origin = {};
  _wavefront_tile_base_size = {};
  _wavefront_tile_plan_valid = false;
  _wavefront_camera_phase_initialized = false;
}

void GPURaytracingRenderer::on_camera_become_steady(SceneRepresentation& scene) {
  ETX_PROFILER_SCOPE();
  (void)scene;
  _preview_active = false;
  _frame_index = 0u;
  _sample_index = 0u;
  reset_render_timing();
  _wavefront_render_step = WavefrontRenderStep::InitSample;
  _wavefront_path_iteration = 0u;
  _wavefront_hard_iteration_cap = 0u;
  _wavefront_camera_queue_count = 0u;
  _wavefront_light_queue_count = 0u;
  _wavefront_light_max_path_length = 0u;
  _wavefront_tile_index = 0u;
  _wavefront_tile_max_pixels = 0u;
  _wavefront_tile_count = 1u;
  _wavefront_tile_path_capacity = 0u;
  _wavefront_tile_base_origin = {};
  _wavefront_tile_base_size = {};
  _wavefront_tile_plan_valid = false;
  _wavefront_camera_phase_initialized = false;
}

void GPURaytracingRenderer::on_scene_changed(SceneRepresentation& scene) {
  ETX_PROFILER_SCOPE();
  Renderer::on_scene_changed(scene);
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

  const auto blas_create_begin = std::chrono::steady_clock::now();
  auto blas_result = device.create_acceleration_structure(blas_desc);
  const auto blas_create_end = std::chrono::steady_clock::now();
  blas_create_ms = elapsed_ms(blas_create_begin, blas_create_end);
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
  const auto tlas_instance_upload_begin = std::chrono::steady_clock::now();
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
  const auto tlas_instance_upload_end = std::chrono::steady_clock::now();
  tlas_instance_upload_ms = elapsed_ms(tlas_instance_upload_begin, tlas_instance_upload_end);

  RHIAccelerationStructureDesc tlas_desc = {};
  tlas_desc.type = RHIAccelerationStructureType::TopLevel;
  tlas_desc.instance_count = 1;

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
  tlas_build_desc.instance_count = 1;
  tlas_build_desc.instance_buffer = inst_res.handle;

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
    ctx.cmd_build_acceleration_structure(cmd, build_desc, scratch_res.handle, 0);
    ctx.cmd_buffer_barrier(cmd, scratch_res.handle, RHIResourceState::AccelerationStructure, RHIResourceState::AccelerationStructure);
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

  _vertex_positions_buffer = new_vertex_positions_buffer;
  _blas = std::move(new_blas);
  _blas_buffers = std::move(new_blas_buffers);
  _tlas = new_tlas;
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
    packed_emitters = build_packed_emitters(data);
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
    upload_success = upload_or_update_linear_scene_buffer(device, data.spectrum_values.data(), data.spectrum_values.size(), scene_buffer_usage, _spectrums_buffer,
                       _spectrums_buffer_size, _gpu_scene.spectrums, "spectrums") &&
                     upload_or_update_linear_scene_buffer(device, data.energy_compensation_interfaces.data(), data.energy_compensation_interfaces.size(), scene_buffer_usage,
                       _energy_compensation_interfaces_buffer, _energy_compensation_interfaces_buffer_size, _gpu_scene.energy_compensation_interfaces,
                       "energy_compensation_interfaces") &&
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
      upload_success = upload_or_update_linear_scene_buffer(device, data.energy_compensation_interfaces.data(), data.energy_compensation_interfaces.size(), scene_buffer_usage,
                         _energy_compensation_interfaces_buffer, _energy_compensation_interfaces_buffer_size, _gpu_scene.energy_compensation_interfaces,
                         "energy_compensation_interfaces") &&
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
  const bool scene_globals_changed = changes[UpdateFlags::Triangles] || changes[UpdateFlags::Meshes] || changes[UpdateFlags::Materials] || changes[UpdateFlags::Spectra] ||
                                     changes[UpdateFlags::Emitters] || changes[UpdateFlags::EnergyCompensationInterfaces] || changes[UpdateFlags::Defaults] ||
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
