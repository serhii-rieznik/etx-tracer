#pragma once

#include "renderer.hxx"

namespace etx {

struct RasterizationRenderer : public Renderer {
  RasterizationRenderer(TaskScheduler&);
  ~RasterizationRenderer() override;

  void init(RenderContext& render_context, SceneRepresentation& scene) override;
  void frame(RenderContext& render_context, SceneRepresentation& scene, float dt) override;
  void cleanup(RenderContext& render_context) override;

  const char* name() const override {
    return "Rasterization";
  }
  RendererMode mode() const override {
    return RendererMode::Rasterization;
  }

  void on_camera_changed(SceneRepresentation& scene, bool path_changed) override {
  }
  void on_scene_changed(SceneRepresentation& scene) override;

 private:
  void create_pipeline(RenderContext& render_context);

  RHIPipeline _pipeline = {};
  bool _initialized = false;
};

}  // namespace etx
