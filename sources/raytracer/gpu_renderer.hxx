#pragma once

#include "renderer.hxx"
#include <etx/rhi/rhi.hxx>
#include <interop/gpu_scene_shared.hxx>
#include <interop/gpu_wavefront_shared.hxx>
#include <string>
#include <vector>

namespace etx {

struct GPURaytracingRenderer : public Renderer {
  GPURaytracingRenderer(TaskScheduler&);
  ~GPURaytracingRenderer() override;

  void init(RHIContext& ctx, SceneRepresentation& scene) override;
  void render(RHIContext& ctx, SceneRepresentation& scene, const FrameData& frame_data) override;

  void cleanup(RHIContext& ctx) override;

  void reload_shaders(RHIContext& ctx);
  bool pipelines_valid() const;
  void set_compile_stage_filter(const std::string&);
  bool set_render_window(const uint2& origin, const uint2& size, const uint2& full_size);
  void reset_render_window();

  const char* name() const override {
    return "GPU Raytracing";
  }
  RendererMode mode() const override {
    return RendererMode::GPURaytracing;
  }

  void on_camera_changed(SceneRepresentation& scene) override;
  void on_scene_changed(SceneRepresentation& scene) override;

 private:
  void destroy_scene_buffers(RHIContext& ctx);
  void destroy_wavefront_buffers(RHIContext& ctx);
  void destroy_blue_noise_buffer(RHIContext& ctx);
  bool update_blue_noise_buffer(RHIContext& ctx, const SceneRepresentation& scene);
  void destroy_acceleration_structures(RHIContext& ctx);
  bool build_acceleration_structures(RHIContext& ctx, SceneRepresentation& scene);
  bool upload_scene_data(RHIContext& ctx, SceneRepresentation& scene, RHIBindlessHandle vertex_positions_buffer);
  bool update_scene_data_partial(RHIContext& ctx, SceneRepresentation& scene, const UpdateFlags& changes);
  bool ensure_wavefront_buffers(RHIContext& ctx, const SceneRepresentation& scene);
  void create_pipelines(RHIContext& ctx);

 private:
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
    CameraDirectLightPrepareDielectricEval = 9u,
    CameraDirectLightPrepareDielectricPdf = 10u,
    CameraDirectLightShadow = 11u,
    CameraDirectLightAccumulate = 12u,
    CameraDirectHitAccumulate = 13u,
    CameraConnectLightPrepareDiffuse = 14u,
    CameraConnectLightPreparePlastic = 15u,
    CameraConnectLightPrepareConductor = 16u,
    CameraConnectLightPrepareDielectric = 17u,
    CameraConnectLightShadow = 18u,
    CameraConnectLightAccumulate = 19u,
    CameraContinuePrepareDiffuse = 20u,
    CameraContinuePreparePlastic = 21u,
    CameraContinuePrepareConductor = 22u,
    CameraContinuePrepareDielectric = 23u,
    CameraContinuePrepareThinfilm = 24u,
    CameraContinueFinalize = 25u,
    TraceLight = 26u,
    LightSurfaceClassify = 27u,
    LightConnectCameraPrepareDiffuse = 28u,
    LightConnectCameraPreparePlastic = 29u,
    LightConnectCameraPrepareConductor = 30u,
    LightConnectCameraPrepareDielectric = 31u,
    LightConnectCameraShadow = 32u,
    LightConnectCameraAccumulate = 33u,
    LightContinuePrepareDiffuse = 34u,
    LightContinuePreparePlastic = 35u,
    LightContinuePrepareConductor = 36u,
    LightContinuePrepareDielectric = 37u,
    LightContinuePrepareThinfilm = 38u,
    LightContinueFinalize = 39u,
    SwapQueues = 40u,
    FinalizeSample = 41u,
    Count = 42u,
  };

  RHIPipeline _pipelines[static_cast<uint32_t>(PipelineStage::Count)] = {};
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
  uint32_t _path_mode = 0u;
  uint32_t _material_compile_mask = 0u;
  uint2 _render_window_origin = {};
  uint2 _render_window_size = {};
  RHIResourceState _output_texture_state = RHIResourceState::Undefined;

  std::string _compile_stage_filter = {};
  bool _compile_filter_matched = false;
  bool _initialized = false;
};

}  // namespace etx
