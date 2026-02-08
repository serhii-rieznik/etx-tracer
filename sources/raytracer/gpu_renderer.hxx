#pragma once

#include "renderer.hxx"
#include <etx/rhi/rhi_types.hxx>
#include <interop/gpu_scene_shared.hxx>

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
  void set_scene_updates_locked(bool locked);

 private:
  void destroy_scene_buffers(RHIContext& ctx);
  void destroy_acceleration_structures(RHIContext& ctx);
  void build_acceleration_structures(RHIContext& ctx, SceneRepresentation& scene);
  void upload_scene_data(RHIContext& ctx, SceneRepresentation& scene, RHIBindlessHandle vertex_positions_buffer);
  void update_scene_data_partial(RHIContext& ctx, SceneRepresentation& scene, const UpdateFlags& changes);
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

  GPUScene _gpu_scene = {};
  SceneHashes _current_scene_hashes = {};
  uint64_t _current_camera_hash = 0;
  uint32_t _frame_index = 0u;
  uint32_t _sample_index = 0u;

  bool _initialized = false;
  bool _scene_dirty = false;
  bool _scene_updates_locked = false;
};

}  // namespace etx
