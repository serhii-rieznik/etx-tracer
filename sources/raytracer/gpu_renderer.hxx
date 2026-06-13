#pragma once

#include "renderer.hxx"
#include <etx/render/host/tasks.hxx>
#include <etx/rhi/rhi.hxx>
#include <interop/gpu_rt_shared.hxx>
#include <interop/gpu_scene_shared.hxx>
#include <interop/gpu_wavefront_shared.hxx>
#include <atomic>
#include <chrono>
#include <memory>
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
    CameraConnectLightShadow = 17u,
    CameraConnectLightAccumulate = 18u,
    CameraContinuePrepareDiffuse = 19u,
    CameraContinuePreparePlastic = 20u,
    CameraContinuePrepareConductor = 21u,
    CameraContinuePrepareDielectric = 22u,
    CameraContinuePrepareThinfilm = 23u,
    CameraContinueFinalize = 24u,
    TraceLight = 25u,
    LightSurfaceClassify = 26u,
    LightConnectCameraPrepareDiffuse = 27u,
    LightConnectCameraPreparePlastic = 28u,
    LightConnectCameraPrepareConductor = 29u,
    LightConnectCameraPrepareDielectric = 30u,
    LightConnectCameraShadow = 31u,
    LightConnectCameraAccumulate = 32u,
    LightContinuePrepareDiffuse = 33u,
    LightContinuePreparePlastic = 34u,
    LightContinuePrepareConductor = 35u,
    LightContinuePrepareDielectric = 36u,
    LightContinuePrepareThinfilm = 37u,
    LightContinueFinalize = 38u,
    SwapQueues = 39u,
    FinalizeSample = 40u,
    CpuOrderPathTrace = 41u,
    Count = 42u,
  };

  GPURaytracingRenderer(TaskScheduler&);
  ~GPURaytracingRenderer() override;

  void init(RHIContext& ctx, SceneRepresentation& scene) override;
  void update_camera(SceneRepresentation& scene, float dt) override;
  void render(RHIContext& ctx, SceneRepresentation& scene, const FrameData& frame_data) override;

  void cleanup(RHIContext& ctx) override;

  void reload_shaders(RHIContext& ctx, SceneRepresentation& scene);
  bool finish_preparation(RHIContext& ctx, SceneRepresentation& scene);
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
  void set_compile_stage_filter(const std::string&);
  bool set_render_window(const uint2& origin, const uint2& size, const uint2& full_size);
  void reset_render_window();
  void set_cpu_order_parity_mode(bool enabled);

  const char* name() const override {
    return "GPU Raytracing";
  }
  RendererMode mode() const override {
    return RendererMode::GPURaytracing;
  }
  RendererPreparationStatus preparation_status() const override;
  void cancel_preparation() override;
  void stop() override;

  void on_camera_changed(SceneRepresentation& scene) override;
  void on_camera_become_steady(SceneRepresentation& scene) override;
  void on_scene_changed(SceneRepresentation& scene) override;

 private:
  static constexpr uint32_t kGPUFixedMaxBounces = 32u;

  struct CompiledStageBinary {
    PipelineStage stage = PipelineStage::PrepareSample;
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
    double elapsed_ms = 0.0;
  };

  struct PendingPipelinePreparation {
    uint32_t generation = 0u;
    uint32_t path_mode = 0u;
    uint32_t material_compile_mask = 0u;
    bool cpu_order_parity_mode = false;
    std::string compile_stage_filter = {};
    uint32_t total_steps = 0u;
    uint32_t total_compile_groups = 0u;
    uint32_t total_pipelines = 0u;
    std::atomic<uint32_t> completed_compile_groups = 0u;
    std::vector<CompiledStageBinary> compiled_stages = {};
    std::vector<PipelinePublishTiming> publish_timings = {};
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

  enum class WavefrontRenderStep : uint32_t {
    InitSample = 0u,
    TraceBounce = 1u,
    FinalizeSample = 2u,
  };

  void destroy_scene_buffers(RHIContext& ctx);
  void destroy_wavefront_buffers(RHIContext& ctx);
  void destroy_blue_noise_buffer(RHIContext& ctx);
  bool update_blue_noise_buffer(RHIContext& ctx, const SceneRepresentation& scene);
  void destroy_acceleration_structures(RHIContext& ctx);
  bool build_acceleration_structures(RHIContext& ctx, SceneRepresentation& scene);
  bool upload_scene_data(RHIContext& ctx, SceneRepresentation& scene, RHIBindlessHandle vertex_positions_buffer);
  bool update_scene_data_partial(RHIContext& ctx, SceneRepresentation& scene, const UpdateFlags& changes);
  bool ensure_wavefront_buffers(RHIContext& ctx, const SceneRepresentation& scene);
  void request_pipeline_preparation(const SceneRepresentation& scene, const char* reason);
  void poll_preparation_tasks(RHIContext& ctx, bool wait_for_active = false);
  bool begin_pipeline_publish(std::shared_ptr<PendingPipelinePreparation> result);
  bool advance_pipeline_publish(RHIContext& ctx, uint32_t max_pipelines);
  bool create_pipelines_sync(RHIContext& ctx, SceneRepresentation& scene, const char* reason);
  void release_inflight_preparation_tasks(bool wait);
  void compile_pipeline_preparation(std::shared_ptr<PendingPipelinePreparation> result);
  void destroy_pipelines(RHIDevice& device);
  bool create_preview_pipeline(RHIContext& ctx);
  void destroy_preview_pipeline(RHIDevice& device);
  bool render_preview(RHIContext& ctx, RHICommandBuffer frame_cmd, const GPURTConstants& constants, const RHIDispatchDesc& dispatch);
  void set_preparation_failed(const std::string& message, const char* phase = "Failed");
  void set_preparation_ready(const char* message = nullptr);
  void set_preparation_state(RendererPreparationState state, const char* phase, const std::string& message = {}, uint32_t completed_steps = 0u, uint32_t total_steps = 0u);
  void reset_runtime_failure();
  void set_runtime_failure(std::string message);

 private:
  RHIPipeline _pipelines[static_cast<uint32_t>(PipelineStage::Count)] = {};
  RHIPipeline _preview_pipeline = {};
  RHIBindlessHandle _tlas = {};
  std::vector<RHIBindlessHandle> _blas;
  std::vector<RHIBindlessHandle> _blas_buffers;
  RHIBindlessHandle _vertex_positions_buffer = {};
  RHIBindlessHandle _vertex_normals_buffer = {};
  RHIBindlessHandle _vertex_tangents_buffer = {};
  RHIBindlessHandle _vertex_bitangents_buffer = {};
  RHIBindlessHandle _vertex_texcoords_buffer = {};
  RHIBindlessHandle _triangles_buffer = {};
  RHIBindlessHandle _meshes_buffer = {};
  RHIBindlessHandle _emitter_profiles_buffer = {};
  RHIBindlessHandle _emitter_instances_buffer = {};
  RHIBindlessHandle _materials_buffer = {};
  RHIBindlessHandle _spectrums_buffer = {};
  RHIBindlessHandle _energy_compensation_interfaces_buffer = {};
  RHIBindlessHandle _scene_globals_buffer = {};
  RHIBindlessHandle _scene_options_buffer = {};
  RHIBindlessHandle _emitters_distribution_buffer = {};
  RHIBindlessHandle _camera_buffer = {};
  RHIBindlessHandle _blue_noise_buffer = {};
  RHIBindlessHandle _wavefront_resources_buffer = {};
  RHIBindlessHandle _camera_state_buffer = {};
  RHIBindlessHandle _light_state_buffer = {};
  RHIBindlessHandle _camera_hit_buffer = {};
  RHIBindlessHandle _light_hit_buffer = {};
  RHIBindlessHandle _camera_queue_a_buffer = {};
  RHIBindlessHandle _camera_queue_b_buffer = {};
  RHIBindlessHandle _light_queue_a_buffer = {};
  RHIBindlessHandle _light_queue_b_buffer = {};
  RHIBindlessHandle _camera_queue_count_readback_buffer = {};
  RHIBindlessHandle _light_queue_count_readback_buffer = {};
  RHIBindlessHandle _camera_vertex_buffer = {};
  RHIBindlessHandle _light_vertex_buffer = {};
  RHIBindlessHandle _film_buffer = {};
  RHIBindlessHandle _path_meta_buffer = {};
  RHIBindlessHandle _direct_light_sample_buffer = {};
  RHIBindlessHandle _direct_light_task_buffer = {};
  RHIBindlessHandle _direct_light_result_buffer = {};
  RHIBindlessHandle _connect_light_task_buffer = {};
  RHIBindlessHandle _connect_light_result_buffer = {};
  RHIBindlessHandle _connect_camera_task_buffer = {};
  RHIBindlessHandle _connect_camera_result_buffer = {};

  uint64_t _vertex_normals_buffer_size = 0;
  uint64_t _vertex_tangents_buffer_size = 0;
  uint64_t _vertex_bitangents_buffer_size = 0;
  uint64_t _vertex_texcoords_buffer_size = 0;
  uint64_t _triangles_buffer_size = 0;
  uint64_t _meshes_buffer_size = 0;
  uint64_t _emitter_profiles_buffer_size = 0;
  uint64_t _emitter_instances_buffer_size = 0;
  uint64_t _materials_buffer_size = 0;
  uint64_t _spectrums_buffer_size = 0;
  uint64_t _energy_compensation_interfaces_buffer_size = 0;
  uint64_t _scene_globals_buffer_size = 0;
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
  uint64_t _camera_queue_count_readback_buffer_size = 0;
  uint64_t _light_queue_count_readback_buffer_size = 0;
  uint64_t _camera_vertex_buffer_size = 0;
  uint64_t _light_vertex_buffer_size = 0;
  uint64_t _film_buffer_size = 0;
  uint64_t _path_meta_buffer_size = 0;
  uint64_t _direct_light_sample_buffer_size = 0;
  uint64_t _direct_light_task_buffer_size = 0;
  uint64_t _direct_light_result_buffer_size = 0;
  uint64_t _connect_light_task_buffer_size = 0;
  uint64_t _connect_light_result_buffer_size = 0;
  uint64_t _connect_camera_task_buffer_size = 0;
  uint64_t _connect_camera_result_buffer_size = 0;
  uint32_t _camera_buffer_descriptor_index = ~0u;
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
  uint32_t _camera_queue_count_readback_buffer_descriptor_index = ~0u;
  uint32_t _light_queue_count_readback_buffer_descriptor_index = ~0u;
  uint32_t _camera_vertex_buffer_descriptor_index = ~0u;
  uint32_t _light_vertex_buffer_descriptor_index = ~0u;
  uint32_t _film_buffer_descriptor_index = ~0u;
  uint32_t _path_meta_buffer_descriptor_index = ~0u;
  uint32_t _direct_light_sample_buffer_descriptor_index = ~0u;
  uint32_t _direct_light_task_buffer_descriptor_index = ~0u;
  uint32_t _direct_light_result_buffer_descriptor_index = ~0u;
  uint32_t _connect_light_task_buffer_descriptor_index = ~0u;
  uint32_t _connect_light_result_buffer_descriptor_index = ~0u;
  uint32_t _connect_camera_task_buffer_descriptor_index = ~0u;
  uint32_t _connect_camera_result_buffer_descriptor_index = ~0u;
  uint32_t _blue_noise_target_samples = 0u;
  uint32_t _wavefront_path_capacity = 0u;
  uint32_t _wavefront_vertex_capacity = 0u;

  RHIChunkedBufferState _images_blob_state = {};
  RHIChunkedBufferState _mediums_blob_state = {};

  GPUScene _gpu_scene = {};
  SceneHashes _current_scene_hashes = {};
  uint64_t _current_camera_hash = 0;
  uint32_t _frame_index = 0u;
  uint32_t _sample_index = 0u;
  WavefrontRenderStep _wavefront_render_step = WavefrontRenderStep::InitSample;
  uint32_t _wavefront_path_iteration = 0u;
  uint32_t _wavefront_hard_iteration_cap = 0u;
  uint32_t _wavefront_camera_queue_count = 0u;
  uint32_t _wavefront_light_queue_count = 0u;
  RHIResourceState _camera_queue_count_readback_state = RHIResourceState::Undefined;
  RHIResourceState _light_queue_count_readback_state = RHIResourceState::Undefined;
  uint32_t _path_mode = 0u;
  uint32_t _material_compile_mask = 0u;
  uint2 _render_window_origin = {};
  uint2 _render_window_size = {};
  RHIResourceState _output_texture_state = RHIResourceState::Undefined;
  RHIBackend _backend = RHIBackend::Metal;

  std::string _compile_stage_filter = {};
  std::string _runtime_failure_reason = {};
  std::string _preparation_phase = "Ready";
  std::string _preparation_message = {};
  std::vector<InflightPreparationTask> _inflight_preparation_tasks = {};
  std::shared_ptr<PendingPipelinePreparation> _active_preparation = {};
  std::shared_ptr<PendingPipelinePreparation> _publish_preparation = {};
  std::chrono::steady_clock::time_point _preparation_started_at = {};
  std::chrono::steady_clock::time_point _pipeline_publish_started_at = {};
  uint32_t _preparation_generation = 0u;
  uint32_t _published_pipeline_count = 0u;
  uint32_t _publish_pipeline_index = 0u;
  bool _compile_filter_matched = false;
  bool _initialized = false;
  bool _runtime_failed = false;
  bool _pipeline_publish_logged = false;
  bool _cpu_order_parity_mode = false;
  bool _preview_active = false;
  bool _preview_pipeline_failed = false;
  bool _cleanup_wait_succeeded = false;
  RendererPreparationState _preparation_state = RendererPreparationState::Ready;
};

}  // namespace etx
