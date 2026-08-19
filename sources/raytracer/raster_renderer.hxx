#pragma once

#include "renderer.hxx"

namespace etx {

struct RasterizationRenderer : public Renderer {
  RasterizationRenderer(TaskScheduler&);
  ~RasterizationRenderer() override;

  void init(RHIContext& ctx, SceneRepresentation& scene) override;
  void render(RHIContext& ctx, SceneRepresentation& scene, const FrameData&) override;

  void cleanup(RHIContext& ctx) override;

  const char* name() const override {
    return "Rasterization";
  }

  RendererMode mode() const override {
    return RendererMode::Rasterization;
  }

  RendererStatus status() const override;

  void on_scene_changed(SceneRepresentation& scene) override;

 private:
  void create_pipeline();

 private:
  RHIPipeline _pipeline = {};

  bool _initialized = false;
};

}  // namespace etx
