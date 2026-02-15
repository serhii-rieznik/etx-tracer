#pragma once

#include "renderer.hxx"
#include <etx/rhi/rhi.hxx>
#include <interop/gpu_scene_shared.hxx>
#include <vector>

namespace etx {

struct GPURaytracingRenderer : public Renderer {
  GPURaytracingRenderer(TaskScheduler&);
  ~GPURaytracingRenderer() override;

  void init(RHIContext& ctx, SceneRepresentation& scene) override;
  void render(RHIContext& ctx, SceneRepresentation& scene, const FrameData& frame_data) override;

  void cleanup(RHIContext& ctx) override;

  void reload_shaders(RHIContext& ctx);

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
  void destroy_blue_noise_buffer(RHIContext& ctx);
  bool update_blue_noise_buffer(RHIContext& ctx, const SceneRepresentation& scene);
  void destroy_acceleration_structures(RHIContext& ctx);
  bool build_acceleration_structures(RHIContext& ctx, SceneRepresentation& scene);
  bool upload_scene_data(RHIContext& ctx, SceneRepresentation& scene, RHIBindlessHandle vertex_positions_buffer);
  bool update_scene_data_partial(RHIContext& ctx, SceneRepresentation& scene, const UpdateFlags& changes);
  void create_pipelines(RHIContext& ctx);

 private:
  RHIPipeline _pipeline = {};
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
  uint32_t _camera_buffer_descriptor_index = ~0u;
  uint32_t _blue_noise_buffer_descriptor_index = ~0u;
  uint32_t _blue_noise_target_samples = 0u;

  RHIChunkedBufferState _images_blob_state = {};
  RHIChunkedBufferState _mediums_blob_state = {};

  GPUScene _gpu_scene = {};
  SceneHashes _current_scene_hashes = {};
  uint64_t _current_camera_hash = 0;
  uint32_t _frame_index = 0u;
  uint32_t _sample_index = 0u;

  bool _initialized = false;
};

}  // namespace etx
