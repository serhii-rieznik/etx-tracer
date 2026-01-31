#pragma once

#include "renderer.hxx"
#include <etx/rhi/rhi_types.hxx>

namespace etx {

struct GPURaytracingRenderer : public Renderer {
  GPURaytracingRenderer(TaskScheduler&);
  ~GPURaytracingRenderer() override;

  void init(RenderContext& render_context, SceneRepresentation& scene) override;
  void frame(RenderContext& render_context, SceneRepresentation& scene, float dt) override;
  void cleanup(RenderContext& render_context) override;
  void process_event(const sapp_event* e) override;

  const char* name() const override {
    return "GPU Raytracing";
  }
  RendererMode mode() const override {
    return RendererMode::GPURaytracing;
  }

  void on_camera_changed(SceneRepresentation& scene, bool path_changed) override;
  void on_scene_changed(SceneRepresentation& scene) override;

 private:
  void build_acceleration_structures(RenderContext& render_context, SceneRepresentation& scene);

  RHIPipeline _pipeline = {};
  RHIBindlessHandle _tlas = 0;
  bool _initialized = false;
};

}  // namespace etx
