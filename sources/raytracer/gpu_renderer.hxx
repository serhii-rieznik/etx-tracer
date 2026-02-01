#pragma once

#include "renderer.hxx"
#include <etx/rhi/rhi_types.hxx>

namespace etx {

struct GPURaytracingRenderer : public Renderer {
  GPURaytracingRenderer(TaskScheduler&);
  ~GPURaytracingRenderer() override;

  void init(RHIContext* ctx, SceneRepresentation& scene) override;
  void frame(RHIContext* ctx, SceneRepresentation& scene, const FrameData&) override;
  void cleanup(RHIContext* ctx) override;
  void process_event(const sapp_event* e) override;

  const char* name() const override {
    return "GPU Raytracing";
  }
  RendererMode mode() const override {
    return RendererMode::GPURaytracing;
  }

  void on_camera_changed(SceneRepresentation& scene) override;
  void on_scene_changed(SceneRepresentation& scene) override;

 private:
  void build_acceleration_structures(RenderContext& render_context, SceneRepresentation& scene);

 private:
  RHIPipeline _pipeline = {};
  RHIBindlessHandle _tlas = {};
  bool _initialized = false;
};

}  // namespace etx
