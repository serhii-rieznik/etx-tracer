#pragma once

#include "renderer.hxx"
#include <etx/rhi/rhi_types.hxx>

namespace etx {

struct GPURaytracingRenderer : public Renderer {
  GPURaytracingRenderer(TaskScheduler&);
  ~GPURaytracingRenderer() override;

  void init(RHIContext* ctx, SceneRepresentation& scene) override;
  void render(RHIContext* ctx, SceneRepresentation& scene, const FrameData& frame_data) override;

  void cleanup(RHIContext* ctx) override;

  void reload_shaders(RHIContext* ctx);

  const char* name() const override {
    return "GPU Raytracing";
  }
  RendererMode mode() const override {
    return RendererMode::GPURaytracing;
  }

  void on_camera_changed(SceneRepresentation& scene) override;
  void on_scene_changed(SceneRepresentation& scene) override;

 private:
  void build_acceleration_structures(RHIContext* ctx, SceneRepresentation& scene);
  void create_pipelines(RHIContext* ctx);

 private:
  RHIPipeline _pipeline = {};

  RHIBindlessHandle _tlas = {};

  std::vector<RHIBindlessHandle> _blas;
  std::vector<RHIBindlessHandle> _blas_buffers;

  bool _initialized = false;
  bool _scene_dirty = false;
};

}  // namespace etx
