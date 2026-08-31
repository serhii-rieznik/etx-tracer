#pragma once

#include "renderer.hxx"
#include <etx/render/host/emitter_packing.hxx>
#include <etx/render/host/tasks.hxx>
#include <etx/rhi/rhi.hxx>
#include <interop/gpu_rt_shared.hxx>
#include <interop/gpu_scene_shared.hxx>
#include <interop/gpu_upbp_shared.hxx>
#include <interop/gpu_wavefront_shared.hxx>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace etx {

struct GPURaytracingRenderer : public Renderer {
  enum class PipelineStage : uint32_t {
    PrepareSample = 0u,
    InitCameraPath0 = 1u,
    InitLightPath0 = 2u,
    TraceCamera = 3u,
    CameraSurfaceClassify = 4u,
    CameraDirectLightSample = 5u,
    CameraDirectLightPrepareDiffuse = 6u,
    CameraDirectLightPreparePlastic = 7u,
    CameraDirectLightPrepareConductor = 8u,
    CameraDirectLightPrepareDielectric = 9u,
    CameraDirectLightShadow = 10u,
    CameraDirectLightAccumulate = 11u,
    CameraDirectHitAccumulate = 12u,
    CameraConnectLightPrepareDiffuse = 13u,
    CameraConnectLightPreparePlastic = 14u,
    CameraConnectLightPrepareConductor = 15u,
    CameraConnectLightPrepareDielectric = 16u,
    CameraConnectLightResolveDiffuse = 17u,
    CameraConnectLightResolvePlastic = 18u,
    CameraConnectLightResolveConductor = 19u,
    CameraConnectLightResolveDielectric = 20u,
    CameraConnectLightShadow = 21u,
    PrepareSpectralValues = 22u,
    CameraContinuePrepareDiffuse = 23u,
    CameraContinuePreparePlastic = 24u,
    CameraContinuePrepareConductor = 25u,
    CameraContinuePrepareDielectric = 26u,
    CameraContinuePrepareThinfilm = 27u,
    CameraContinueFinalize = 28u,
    TraceLight = 29u,
    LightSurfaceClassify = 30u,
    LightConnectCameraPrepareDiffuse = 31u,
    LightConnectCameraPreparePlastic = 32u,
    LightConnectCameraPrepareConductor = 33u,
    LightConnectCameraPrepareDielectric = 34u,
    LightConnectCameraShadow = 35u,
    LightConnectCameraAccumulate = 36u,
    LightContinuePrepareDiffuse = 37u,
    LightContinuePreparePlastic = 38u,
    LightContinuePrepareConductor = 39u,
    LightContinuePrepareDielectric = 40u,
    LightContinuePrepareThinfilm = 41u,
    LightContinueFinalize = 42u,
    SwapQueues = 43u,
    FinalizeSample = 44u,
    UPBPClear = 45u,
    LightConnectCameraClear = 46u,
    BuildDispatchArgs = 47u,
    VCMGridClear = 48u,
    VCMGridBuild = 49u,
    VCMMergeDiffuse = 50u,
    VCMMergePlastic = 51u,
    VCMMergeConductor = 52u,
    VCMMergeDielectric = 53u,
    UPBPDensityCompact = 54u,
    UPBPPP3D = 55u,
    UPBPPB2D = 56u,
    UPBPBP2D = 57u,
    UPBPBB1D = 58u,
    UPBPDirectHit = 59u,
    UPBPValidate = 60u,
    UPBPBeamInstances = 61u,
    UPBPBeamGridBuild = 62u,
    Count = 63u,
  };

  GPURaytracingRenderer(TaskScheduler&);
  ~GPURaytracingRenderer() override;

  void init(RHIContext& ctx, SceneRepresentation& scene) override;
  void render(RHIContext& ctx, SceneRepresentation& scene, const FrameData& frame_data) override;

  void cleanup(RHIContext& ctx) override;

  void reload_shaders(RHIContext& ctx, SceneRepresentation& scene);
  bool finish_preparation(RHIContext& ctx, SceneRepresentation& scene);
  void poll_preparation(RHIContext& ctx);
  bool pipelines_valid() const;
  bool runtime_failed() const {
    return _runtime_failed;
  }
  bool cleanup_wait_succeeded() const {
    return _cleanup_wait_succeeded;
  }
  const std::string& runtime_failure_reason() const {
    return _runtime_failure_reason;
  }
  uint32_t completed_samples() const {
    return _sample_index;
  }
  void set_wavefront_steps_per_render(uint32_t value);
  void set_wavefront_auto_tuning(bool value);
  void set_batch_coarse_progress(bool value);
  void set_kernel_timing_enabled(bool value);
  const RendererKernelTimingStats& kernel_timing_stats() const {
    return (_preserved_timing_stats_valid && (_render_timing_active == false)) ? _preserved_kernel_timing_stats : _kernel_timing_stats;
  }
  uint32_t wavefront_steps_per_render() const {
    return _wavefront_steps_per_render;
  }
  double wavefront_last_batch_ms() const {
    return _wavefront_last_batch_ms;
  }
  bool wavefront_auto_tuning_enabled() const {
    return _wavefront_auto_tuning_enabled;
  }
  void set_compile_stage_filter(const std::string&);
  void set_sample_limit(uint32_t sample_limit);
  bool set_render_window(const uint2& origin, const uint2& size, const uint2& full_size);
  void reset_render_window();

  const char* name() const override {
    return "GPU Raytracing";
  }
  RendererMode mode() const override {
    return RendererMode::GPURaytracing;
  }
  RHITexture output_texture() const override {
    if (_preview_active || (_sample_index == 0u) || (_output_texture_state != RHIResourceState::ShaderReadOnly)) {
      return {};
    }
    return _output_texture;
  }
  RHITexture display_texture() const override {
    if (_preview_visible && (_preview_texture_state == RHIResourceState::ShaderReadOnly)) {
      return _preview_texture;
    }
    if ((_display_output_valid == false) || (_output_texture_state != RHIResourceState::ShaderReadOnly)) {
      return {};
    }
    return _output_texture;
  }
  RendererPreparationStatus preparation_status() const override;
  RendererStatus status() const override;
  RendererMemoryStats memory_stats() const override;
  RendererControlState control_state() const override;
  bool is_running() const override;
  void invalidate_output();
  void start() override;
  void cancel_preparation() override;
  void stop() override;
  void finish() override;
  void restart() override;

  void on_camera_changed(SceneRepresentation& scene) override;
  void on_camera_become_steady(SceneRepresentation& scene) override;
  void on_scene_changed(SceneRepresentation& scene) override;
  void on_scene_transforms_changed(SceneRepresentation& scene) override;
  void on_scene_transform_interaction_started(SceneRepresentation& scene) override;
  void on_scene_transform_interaction_finished(SceneRepresentation& scene) override;

 private:
  static constexpr uint32_t kGPUFixedMaxBounces = 32u;

  enum class RunState : uint32_t {
    Stopped,
    Running,
    Finishing,
    Completed,
  };

  struct CompiledStageBinary {
    PipelineStage stage = PipelineStage::PrepareSample;
    uint64_t variant_key = 0u;
    std::string entry_point = {};
    std::string source_file = {};
    std::string optimization_level = {};
    std::string bsdf_kind = {};
    bool uses_stage_entry_define = false;
    std::vector<uint8_t> blob = {};
    RHIShaderBinary binary = {};
  };

  struct PipelinePublishTiming {
    PipelineStage stage = PipelineStage::PrepareSample;
    std::string entry_point = {};
    std::string source_file = {};
    std::string optimization_level = {};
    std::string bsdf_kind = {};
    bool uses_stage_entry_define = false;
    uint64_t spirv_size_bytes = 0u;
    double elapsed_ms = 0.0;
  };

  struct PendingPipelinePreparation {
    uint32_t generation = 0u;
    uint32_t integrator_mode = 0u;
    uint32_t integrator_features = 0u;
    uint32_t material_compile_mask = 0u;
    uint32_t spectral_mode = 0u;
    uint64_t requested_stage_mask = 0u;
    std::string compile_stage_filter = {};
    uint32_t total_steps = 0u;
    uint32_t total_compile_groups = 0u;
    uint32_t total_pipelines = 0u;
    std::atomic<uint32_t> completed_compile_groups = 0u;
    std::atomic<uint32_t> completed_pipelines = 0u;
    std::atomic<uint32_t> compile_worker_count = 0u;
    std::atomic<bool> initialization_complete = false;
    std::atomic<bool> compilation_complete = false;
    std::atomic<bool> publish_started = false;
    std::vector<CompiledStageBinary> compiled_stages = {};
    std::vector<uint32_t> ready_pipeline_indices = {};
    std::vector<PipelinePublishTiming> publish_timings = {};
    mutable std::mutex progress_mutex = {};
    std::condition_variable progress_condition = {};
    std::vector<RendererPreparationStepStatus> pipeline_progress = {};
    std::string error_message = {};
    bool compile_filter_matched = false;
    bool success = false;
    std::chrono::steady_clock::time_point queued_at = {};
    std::chrono::steady_clock::time_point compile_started_at = {};
    std::chrono::steady_clock::time_point compile_finished_at = {};
  };

  struct InflightPreparationTask {
    Task::Handle handle = {};
    std::shared_ptr<PendingPipelinePreparation> result = {};
  };

  struct InflightPipelinePublishTask {
    Task::Handle handle = {};
    std::shared_ptr<PendingPipelinePreparation> preparation = {};
    std::vector<RHICreatePipelineBatchEntry> results = {};
    std::vector<uint32_t> pipeline_indices = {};
    uint32_t pipeline_count = 0u;
    uint32_t worker_count = 0u;
    std::chrono::steady_clock::time_point started_at = {};
    std::chrono::steady_clock::time_point finished_at = {};
  };

  enum class WavefrontRenderStep : uint32_t {
    InitSample = 0u,
    TraceBounce = 1u,
    FinalizeSample = 2u,
    UPBPEvaluateLightBatch = 3u,
  };

  struct UPBPBuffer {
    RHIBindlessHandle handle = {};
    uint64_t size = 0u;
    uint32_t descriptor_index = ~0u;
  };

  struct UPBPDensityBatchResources {
    UPBPBuffer surface_point_buffer = {};
    UPBPBuffer surface_point_aabb_buffer = {};
    UPBPBuffer medium_point_buffer = {};
    UPBPBuffer medium_point_aabb_buffer = {};
    UPBPBuffer beam_buffer = {};
    UPBPBuffer event_buffer = {};
    uint32_t surface_point_count = 0u;
    uint32_t medium_point_count = 0u;
    uint32_t beam_count = 0u;
    uint32_t event_count = 0u;
    uint32_t selected_beam_count = 0u;
  };

  struct UPBPBeamGridResources {
    UPBPBuffer metadata_buffer = {};
    UPBPBuffer cell_offsets_buffer = {};
    UPBPBuffer beam_indices_buffer = {};
    uint32_t beam_index_count = 0u;
    RHIResourceState metadata_state = RHIResourceState::Undefined;
    RHIResourceState cell_offsets_state = RHIResourceState::Undefined;
    RHIResourceState beam_indices_state = RHIResourceState::Undefined;
  };

  struct UPBPRuntimeResources {
    GPUUPBPResources resources = {};
    UPBPBuffer resources_buffer = {};
    UPBPBuffer vertex_buffer = {};
    UPBPBuffer segment_buffer = {};
    UPBPBuffer interval_buffer = {};
    UPBPBuffer event_buffer = {};
    UPBPBuffer point_buffer = {};
    UPBPBuffer beam_buffer = {};
    std::vector<UPBPDensityBatchResources> density_batches = {};
    UPBPBuffer density_batch_buffer = {};
    UPBPBuffer density_surface_point_buffer = {};
    UPBPBuffer density_surface_point_aabb_buffer = {};
    UPBPBuffer density_medium_point_buffer = {};
    UPBPBuffer density_medium_point_aabb_buffer = {};
    UPBPBuffer density_beam_buffer = {};
    UPBPBuffer density_surface_point_instance_buffer = {};
    UPBPBuffer density_medium_point_instance_buffer = {};
    UPBPBuffer density_bp2d_beam_instance_buffer = {};
    UPBPBuffer density_bb1d_beam_instance_buffer = {};
    UPBPBuffer density_bb1d_beam_buffer = {};
    UPBPBuffer density_beam_reference_buffer = {};
    UPBPBuffer density_beam_grid_metadata_readback_buffer = {};
    UPBPBuffer density_beam_grid_scratch_buffer = {};
    UPBPBeamGridResources density_bp2d_beam_grid = {};
    UPBPBeamGridResources density_bb1d_beam_grid = {};
    RHIResourceState density_beam_grid_metadata_readback_state = RHIResourceState::Undefined;
    RHIResourceState density_beam_grid_scratch_state = RHIResourceState::Undefined;
    UPBPBuffer density_beam_unit_aabb_buffer = {};
    UPBPBuffer density_as_scratch_buffer = {};
    UPBPBuffer counter_buffer = {};
    UPBPBuffer counter_readback_buffer = {};
    UPBPBuffer path_state_buffer = {};
    UPBPBuffer bpt_light_vertex_buffer = {};
    UPBPBuffer bpt_light_path_state_buffer = {};
    RHIBindlessHandle density_surface_point_tlas = {};
    RHIBindlessHandle density_medium_point_tlas = {};
    std::array<RHIBindlessHandle, kGPUUPBPSurfacePartitionCount> density_surface_point_blas = {};
    RHIBindlessHandle density_medium_point_blas = {};
    RHIBindlessHandle density_beam_unit_blas = {};
    std::vector<RHIBindlessHandle> density_bp2d_beam_tlas = {};
    std::array<RHIBindlessHandle, kGPUUPBPBB1DPartitionCount> density_bb1d_beam_tlas = {};
    uint32_t density_surface_point_tlas_capacity = 0u;
    uint32_t density_medium_point_tlas_capacity = 0u;
    std::array<uint32_t, kGPUUPBPSurfacePartitionCount> density_surface_point_blas_capacities = {};
    uint32_t density_medium_point_blas_capacity = 0u;
    std::vector<uint32_t> density_bp2d_beam_tlas_capacities = {};
    std::array<uint32_t, kGPUUPBPBB1DPartitionCount> density_bb1d_beam_tlas_capacities = {};
    uint32_t resident_light_path_capacity = 0u;
    uint32_t resident_camera_path_capacity = 0u;
    uint32_t density_surface_point_count = 0u;
    uint32_t density_medium_point_count = 0u;
    uint32_t density_beam_count = 0u;
    uint32_t density_bb1d_beam_count = 0u;
    uint32_t density_batch_count = 0u;
    uint32_t maximum_path_length = 0u;
    uint32_t maximum_boundary_count = 0u;
    uint32_t technique_mask = 0u;
    uint32_t light_batch_offset = 0u;
    uint32_t light_batch_count = 0u;
    uint32_t camera_batch_offset = 0u;
    uint32_t camera_batch_count = 0u;
    uint32_t light_batch_index = 0u;
    uint32_t light_batch_iteration = 0u;
    uint32_t camera_batch_index = 0u;
    uint32_t light_batch_count_total = 0u;
    uint32_t camera_batch_count_total = 0u;
    uint32_t global_path_count = 0u;
    uint32_t sample_index = ~0u;
    bool density_cache_ready = false;
    bool camera_phase_started = false;
    RHIResourceState counter_readback_state = RHIResourceState::Undefined;
    RHIResourceState bpt_light_vertex_state = RHIResourceState::Undefined;
    RHIResourceState bpt_light_path_state_state = RHIResourceState::Undefined;
  };

  void destroy_scene_buffers(RHIContext& ctx);
  void destroy_wavefront_buffers(RHIContext& ctx);
  void destroy_upbp_buffers(RHIDevice& device);
  void destroy_upbp_density_cache(RHIDevice& device, bool release_beam_grid_storage);
  void reset_upbp_density_cache();
  void bind_upbp_density_cache_resources();
  void destroy_blue_noise_buffer(RHIContext& ctx);
  bool update_blue_noise_buffer(RHIContext& ctx, const SceneRepresentation& scene);
  void destroy_acceleration_structures(RHIContext& ctx);
  bool build_acceleration_structures(RHIContext& ctx, SceneRepresentation& scene);
  bool refit_top_level_acceleration_structure(RHIContext& ctx, const SceneData& scene_data);
  bool update_preview_camera_buffer(RHIDevice& device, const Camera& camera, const uint2& dimensions, uint32_t frame_index, uint32_t& descriptor_index);
  bool ensure_preview_texture(RHIContext& ctx, const uint2& dimensions);
  void destroy_preview_resources(RHIDevice& device);
  bool upload_scene_data(RHIContext& ctx, SceneRepresentation& scene, RHIBindlessHandle vertex_positions_buffer);
  bool update_scene_data_partial(RHIContext& ctx, SceneRepresentation& scene, const UpdateFlags& changes);
  bool ensure_wavefront_buffers(RHIContext& ctx, const SceneRepresentation& scene, uint32_t path_capacity, uint32_t active_path_capacity, bool allow_light_history_shrink);
  bool ensure_upbp_buffers(RHIContext& ctx, const SceneRepresentation& scene, uint32_t global_path_count, uint32_t wavefront_path_capacity, uint32_t camera_batch_index,
    uint32_t camera_batch_offset, uint32_t camera_batch_count);
  bool update_upbp_iteration_resources(RHIDevice& device, const SceneRepresentation& scene, uint32_t global_path_count);
  bool ensure_light_vertex_capacity(RHIContext& ctx, uint32_t required_vertex_capacity);
  void request_pipeline_preparation(const SceneRepresentation& scene, const char* reason, bool force_reload);
  void poll_preparation_tasks(RHIContext& ctx, bool wait_for_active = false);
  bool begin_pipeline_publish(std::shared_ptr<PendingPipelinePreparation> result);
  bool advance_pipeline_publish(RHIContext& ctx, uint32_t max_pipelines, bool wait_for_batch);
  bool finish_pipeline_publish_batch(RHIDevice& device, bool wait);
  bool create_pipelines_sync(RHIContext& ctx, SceneRepresentation& scene, const char* reason);
  void release_inflight_preparation_tasks(bool wait);
  void compile_pipeline_preparation(std::shared_ptr<PendingPipelinePreparation> result);
  void destroy_pipelines(RHIDevice& device);
  void reset_render_timing();
  void reset_render_progress();
  void reset_wavefront_auto_tuning();
  void update_wavefront_auto_tuning(uint32_t executed_steps, double elapsed_ms, bool budget_consumed, bool measurement_valid);
  void stop_render_timing();
  void preserve_render_statistics();
  void reset_kernel_timings();
  void update_kernel_timing_stats();
  void set_preparation_failed(const std::string& message, const char* phase = "Failed");
  void set_preparation_ready(const char* message = nullptr);
  void set_preparation_state(RendererPreparationState state, const char* phase, const std::string& message = {}, uint32_t completed_steps = 0u, uint32_t total_steps = 0u);
  void reset_runtime_failure();
  void set_runtime_failure(std::string message);

 private:
  RHIPipeline _pipelines[static_cast<uint32_t>(PipelineStage::Count)] = {};
  uint64_t _pipeline_variant_keys[static_cast<uint32_t>(PipelineStage::Count)] = {};
  RHITexture _preview_texture = {};
  uint2 _preview_texture_dimensions = {};
  RHIResourceState _preview_texture_state = RHIResourceState::Undefined;
  RHIBindlessHandle _preview_camera_buffers[kRHIMaxFrames] = {};
  uint64_t _preview_camera_buffer_sizes[kRHIMaxFrames] = {};
  RHIBindlessHandle _tlas = {};
  std::vector<RHIBindlessHandle> _blas;
  std::vector<RHIBindlessHandle> _blas_buffers;
  std::vector<RHIAccelerationStructureInstance> _tlas_instance_staging;
  RHIBindlessHandle _tlas_instance_buffer = {};
  RHIBindlessHandle _as_scratch_buffer = {};
  uint32_t _tlas_instance_count = 0u;
  RHIBindlessHandle _vertex_positions_buffer = {};
  RHIBindlessHandle _vertex_normals_buffer = {};
  RHIBindlessHandle _vertex_tangents_buffer = {};
  RHIBindlessHandle _vertex_bitangents_buffer = {};
  RHIBindlessHandle _vertex_texcoords_buffer = {};
  RHIBindlessHandle _triangles_buffer = {};
  RHIBindlessHandle _meshes_buffer = {};
  RHIBindlessHandle _instances_buffer = {};
  RHIBindlessHandle _emitter_profiles_buffer = {};
  RHIBindlessHandle _emitter_instances_buffer = {};
  RHIBindlessHandle _materials_buffer = {};
  RHIBindlessHandle _spectrums_buffer = {};
  RHIBindlessHandle _spectral_values_buffer = {};
  RHIBindlessHandle _energy_compensation_interfaces_buffer = {};
  RHIBindlessHandle _scene_globals_buffer = {};
  RHIBindlessHandle _scene_options_buffer = {};
  RHIBindlessHandle _emitters_distribution_buffer = {};
  RHIBindlessHandle _camera_buffer = {};
  RHIBindlessHandle _blue_noise_buffer = {};
  RHIBindlessHandle _wavefront_resources_buffer = {};
  GPUWavefrontResources _wavefront_resources = {};
  UPBPRuntimeResources _upbp = {};
  RHIBindlessHandle _camera_state_buffer = {};
  RHIBindlessHandle _light_state_buffer = {};
  RHIBindlessHandle _camera_hit_buffer = {};
  RHIBindlessHandle _light_hit_buffer = {};
  RHIBindlessHandle _camera_queue_a_buffer = {};
  RHIBindlessHandle _camera_queue_b_buffer = {};
  RHIBindlessHandle _light_queue_a_buffer = {};
  RHIBindlessHandle _light_queue_b_buffer = {};
  RHIBindlessHandle _material_queue_buffer = {};
  RHIBindlessHandle _shadow_queue_buffer = {};
  RHIBindlessHandle _wavefront_dispatch_args_buffer = {};
  RHIBindlessHandle _camera_queue_count_readback_buffer = {};
  RHIBindlessHandle _light_queue_count_readback_buffer = {};
  RHIBindlessHandle _camera_vertex_buffer = {};
  RHIBindlessHandle _light_vertex_buffer = {};
  RHIBindlessHandle _fast_light_endpoint_buffer = {};
  RHIBindlessHandle _light_vertex_counter_buffer = {};
  RHIBindlessHandle _light_vertex_counter_readback_buffer = {};
  RHIBindlessHandle _vcm_grid_heads_buffer = {};
  RHIBindlessHandle _vcm_grid_next_buffer = {};
  RHIBindlessHandle _film_buffer = {};
  RHIBindlessHandle _path_meta_buffer = {};
  RHIBindlessHandle _direct_light_sample_buffer = {};
  RHIBindlessHandle _direct_light_task_buffer = {};
  RHIBindlessHandle _direct_light_result_buffer = {};
  RHIBindlessHandle _connect_light_task_buffer = {};
  RHIBindlessHandle _connect_camera_task_buffer = {};
  RHIBindlessHandle _connect_camera_result_buffer = {};
  RHIBindlessHandle _camera_subsurface_state_buffer = {};
  RHIBindlessHandle _light_subsurface_state_buffer = {};

  uint64_t _vertex_normals_buffer_size = 0;
  uint64_t _vertex_positions_buffer_size = 0;
  uint64_t _vertex_tangents_buffer_size = 0;
  uint64_t _vertex_bitangents_buffer_size = 0;
  uint64_t _vertex_texcoords_buffer_size = 0;
  uint64_t _triangles_buffer_size = 0;
  uint64_t _meshes_buffer_size = 0;
  uint64_t _instances_buffer_size = 0;
  uint64_t _emitter_profiles_buffer_size = 0;
  uint64_t _emitter_instances_buffer_size = 0;
  uint64_t _materials_buffer_size = 0;
  uint64_t _spectrums_buffer_size = 0;
  uint64_t _spectral_values_buffer_size = 0;
  uint64_t _energy_compensation_interfaces_buffer_size = 0;
  uint64_t _scene_globals_buffer_size = 0;
  float _scene_bounding_sphere_radius = 0.0f;
  uint64_t _scene_options_buffer_size = 0;
  uint64_t _emitters_distribution_buffer_size = 0;
  uint64_t _camera_buffer_size = 0;
  uint64_t _blue_noise_buffer_size = 0;
  uint64_t _wavefront_resources_buffer_size = 0;
  uint64_t _camera_state_buffer_size = 0;
  uint64_t _light_state_buffer_size = 0;
  uint64_t _camera_hit_buffer_size = 0;
  uint64_t _light_hit_buffer_size = 0;
  uint64_t _camera_queue_a_buffer_size = 0;
  uint64_t _camera_queue_b_buffer_size = 0;
  uint64_t _light_queue_a_buffer_size = 0;
  uint64_t _light_queue_b_buffer_size = 0;
  uint64_t _material_queue_buffer_size = 0;
  uint64_t _shadow_queue_buffer_size = 0;
  uint64_t _wavefront_dispatch_args_buffer_size = 0;
  uint64_t _camera_queue_count_readback_buffer_size = 0;
  uint64_t _light_queue_count_readback_buffer_size = 0;
  uint64_t _camera_vertex_buffer_size = 0;
  uint64_t _light_vertex_buffer_size = 0;
  uint64_t _fast_light_endpoint_buffer_size = 0;
  uint64_t _light_vertex_counter_buffer_size = 0;
  uint64_t _light_vertex_counter_readback_buffer_size = 0;
  uint64_t _vcm_grid_heads_buffer_size = 0;
  uint64_t _vcm_grid_next_buffer_size = 0;
  uint64_t _film_buffer_size = 0;
  uint64_t _path_meta_buffer_size = 0;
  uint64_t _direct_light_sample_buffer_size = 0;
  uint64_t _direct_light_task_buffer_size = 0;
  uint64_t _direct_light_result_buffer_size = 0;
  uint64_t _connect_light_task_buffer_size = 0;
  uint64_t _connect_camera_task_buffer_size = 0;
  uint64_t _connect_camera_result_buffer_size = 0;
  uint64_t _camera_subsurface_state_buffer_size = 0;
  uint64_t _light_subsurface_state_buffer_size = 0;
  uint32_t _camera_buffer_descriptor_index = ~0u;
  uint32_t _spectral_values_buffer_descriptor_index = ~0u;
  uint32_t _blue_noise_buffer_descriptor_index = ~0u;
  uint32_t _wavefront_resources_buffer_descriptor_index = ~0u;
  uint32_t _camera_state_buffer_descriptor_index = ~0u;
  uint32_t _light_state_buffer_descriptor_index = ~0u;
  uint32_t _camera_hit_buffer_descriptor_index = ~0u;
  uint32_t _light_hit_buffer_descriptor_index = ~0u;
  uint32_t _camera_queue_a_buffer_descriptor_index = ~0u;
  uint32_t _camera_queue_b_buffer_descriptor_index = ~0u;
  uint32_t _light_queue_a_buffer_descriptor_index = ~0u;
  uint32_t _light_queue_b_buffer_descriptor_index = ~0u;
  uint32_t _material_queue_buffer_descriptor_index = ~0u;
  uint32_t _shadow_queue_buffer_descriptor_index = ~0u;
  uint32_t _wavefront_dispatch_args_buffer_descriptor_index = ~0u;
  uint32_t _camera_queue_count_readback_buffer_descriptor_index = ~0u;
  uint32_t _light_queue_count_readback_buffer_descriptor_index = ~0u;
  uint32_t _camera_vertex_buffer_descriptor_index = ~0u;
  uint32_t _light_vertex_buffer_descriptor_index = ~0u;
  uint32_t _fast_light_endpoint_buffer_descriptor_index = ~0u;
  uint32_t _light_vertex_counter_buffer_descriptor_index = ~0u;
  uint32_t _light_vertex_counter_readback_buffer_descriptor_index = ~0u;
  uint32_t _vcm_grid_heads_buffer_descriptor_index = ~0u;
  uint32_t _vcm_grid_next_buffer_descriptor_index = ~0u;
  uint32_t _film_buffer_descriptor_index = ~0u;
  uint32_t _path_meta_buffer_descriptor_index = ~0u;
  uint32_t _direct_light_sample_buffer_descriptor_index = ~0u;
  uint32_t _direct_light_task_buffer_descriptor_index = ~0u;
  uint32_t _direct_light_result_buffer_descriptor_index = ~0u;
  uint32_t _connect_light_task_buffer_descriptor_index = ~0u;
  uint32_t _connect_camera_task_buffer_descriptor_index = ~0u;
  uint32_t _connect_camera_result_buffer_descriptor_index = ~0u;
  uint32_t _camera_subsurface_state_buffer_descriptor_index = ~0u;
  uint32_t _light_subsurface_state_buffer_descriptor_index = ~0u;
  uint32_t _blue_noise_target_samples = 0u;
  uint32_t _wavefront_path_capacity = 0u;
  uint32_t _wavefront_vertex_capacity = 0u;
  uint32_t _wavefront_allocated_integrator_mode = ~0u;
  uint32_t _wavefront_allocated_integrator_features = 0u;

  RHIChunkedBufferState _images_blob_state = {};
  RHIChunkedBufferState _mediums_blob_state = {};

  GPUScene _gpu_scene = {};
  SceneHashes _current_scene_hashes = {};
  PackedEmitterTopology _emitter_topology = {};
  uint64_t _current_camera_hash = 0;
  uint32_t _frame_index = 0u;
  uint32_t _sample_index = 0u;
  WavefrontRenderStep _wavefront_render_step = WavefrontRenderStep::InitSample;
  uint32_t _wavefront_path_iteration = 0u;
  uint32_t _wavefront_hard_iteration_cap = 0u;
  uint32_t _wavefront_camera_queue_count = 0u;
  uint32_t _wavefront_light_queue_count = 0u;
  uint32_t _wavefront_light_max_path_length = 0u;
  uint32_t _wavefront_max_observed_camera_path_length = 0u;
  uint32_t _wavefront_max_observed_light_path_length = 0u;
  uint32_t _wavefront_connect_light_vertex_length = 0u;
  uint32_t _wavefront_connect_light_history_bounces = 0u;
  uint32_t _wavefront_light_history_capacity_bounces = 0u;
  uint32_t _wavefront_light_vertex_reserved_count = 0u;
  uint32_t _wavefront_light_vertex_sample_peak_count = 0u;
  uint32_t _wavefront_light_history_underuse_sample_count = 0u;
  uint32_t _wavefront_light_history_underuse_peak_count = 0u;
  uint32_t _wavefront_vcm_light_vertex_count = 0u;
  uint32_t _wavefront_tile_index = 0u;
  uint32_t _wavefront_tile_max_pixels = 0u;
  uint32_t _wavefront_tile_count = 1u;
  uint32_t _wavefront_tile_path_capacity = 0u;
  uint32_t _wavefront_steps_per_render = 256u;
  double _wavefront_last_batch_ms = 0.0;
  double _wavefront_smoothed_ms_per_step = 0.0;
  uint2 _wavefront_tile_base_origin = {};
  uint2 _wavefront_tile_base_size = {};
  bool _wavefront_tile_plan_valid = false;
  bool _wavefront_camera_phase_initialized = false;
  bool _wavefront_auto_tuning_enabled = false;
  bool _batch_coarse_progress = true;
  RHIResourceState _wavefront_dispatch_args_buffer_state = RHIResourceState::Undefined;
  RHIResourceState _camera_queue_count_readback_state = RHIResourceState::Undefined;
  RHIResourceState _light_queue_count_readback_state = RHIResourceState::Undefined;
  RHIResourceState _light_vertex_counter_readback_state = RHIResourceState::Undefined;
  uint32_t _integrator_mode = 0u;
  uint32_t _integrator_features = 0u;
  uint32_t _material_compile_mask = 0u;
  uint32_t _spectral_mode = 0u;
  bool _use_compute_upbp_beam_grid = false;
  bool _scene_options_upload_pending = false;
  uint2 _render_window_origin = {};
  uint2 _render_window_size = {};
  RHIResourceState _output_texture_state = RHIResourceState::Undefined;
  RHIBackend _backend = RHIBackend::Metal;

  struct KernelTimingAccumulator {
    uint64_t dispatch_count = 0u;
    double total_ms = 0.0;
  };

  KernelTimingAccumulator _kernel_timing_accumulators[static_cast<uint32_t>(PipelineStage::Count)] = {};
  RendererKernelTimingStats _kernel_timing_stats = {};
  RendererKernelTimingStats _preserved_kernel_timing_stats = {};

  std::string _compile_stage_filter = {};
  std::string _runtime_failure_reason = {};
  std::string _preparation_phase = "Ready";
  std::string _preparation_message = {};
  RHIMemoryStats _last_memory_stats = {};
  std::vector<InflightPreparationTask> _inflight_preparation_tasks = {};
  std::shared_ptr<PendingPipelinePreparation> _active_preparation = {};
  std::shared_ptr<PendingPipelinePreparation> _publish_preparation = {};
  std::shared_ptr<InflightPipelinePublishTask> _pipeline_publish_task = {};
  std::chrono::steady_clock::time_point _preparation_started_at = {};
  std::chrono::steady_clock::time_point _pipeline_publish_started_at = {};
  std::chrono::steady_clock::time_point _render_started_at = {};
  std::chrono::steady_clock::time_point _kernel_timing_started_at = {};
  double _last_render_elapsed_seconds = 0.0;
  double _preserved_render_elapsed_seconds = 0.0;
  uint32_t _preparation_generation = 0u;
  uint32_t _published_pipeline_count = 0u;
  uint32_t _publish_pipeline_index = 0u;
  uint32_t _sample_limit = 0u;
  uint32_t _last_target_samples = 0u;
  bool _compile_filter_matched = false;
  bool _initialized = false;
  bool _runtime_failed = false;
  bool _preparation_canceled = false;
  bool _pipeline_publish_logged = false;
  bool _preview_visible = false;
  bool _display_output_valid = false;
  bool _cleanup_wait_succeeded = false;
  bool _render_timing_active = false;
  bool _preserved_timing_stats_valid = false;
  bool _kernel_timing_enabled = false;
  bool _scene_valid = false;
  RunState _run_state = RunState::Stopped;
  RendererPreparationState _preparation_state = RendererPreparationState::Ready;
};

}  // namespace etx
