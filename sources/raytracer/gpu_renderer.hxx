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

 private:
  void build_acceleration_structures(RHIContext& ctx, SceneRepresentation& scene);
  void upload_scene_data(RHIContext& ctx, SceneRepresentation& scene, RHIBindlessHandle vertex_positions_buffer);
  void create_pipelines(RHIContext& ctx);

 private:
  RHIPipeline _pipeline = {};
  RHIBindlessHandle _tlas = {};
  std::vector<RHIBindlessHandle> _blas;
  std::vector<RHIBindlessHandle> _blas_buffers;
  std::vector<RHIBindlessHandle> _scene_buffers;
  GPUScene _gpu_scene = {};
  uint32_t _frame_index = 0u;
  uint32_t _sample_index = 0u;

  bool _initialized = false;
  bool _scene_dirty = false;
};

}  // namespace etx
