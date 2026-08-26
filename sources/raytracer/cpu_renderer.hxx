#pragma once

#include "renderer.hxx"
#include <etx/render/shared/scene.hxx>

#include <etx/rt/rt.hxx>
#include <etx/rt/integrators/debug.hxx>
#include <etx/rt/integrators/bidirectional.hxx>
#include <etx/rt/integrators/vcm_cpu.hxx>

#include <chrono>

namespace etx {

struct CPURaytracingRenderer : public Renderer {
  CPURaytracingRenderer(Raytracing& rt, SceneRepresentation& scene);
  ~CPURaytracingRenderer() override;

  void init(RHIContext& ctx, SceneRepresentation& scene) override;

  void render(RHIContext& ctx, SceneRepresentation& scene, const FrameData&) override;

  void cleanup(RHIContext& ctx) override;

  const char* name() const override {
    return "CPU Raytracing";
  }

  RendererMode mode() const override {
    return RendererMode::CPURaytracing;
  }

  bool is_running() const override;
  RendererStatus status() const override;
  RendererControlState control_state() const override;
  RHITexture output_texture() const override {
    if (_preview_active || (_last_uploaded_completed_iterations == 0u) || (_output_texture_state != RHIResourceState::ShaderReadOnly)) {
      return {};
    }
    return _output_texture;
  }
  RHITexture display_texture() const override {
    if ((_display_output_valid == false) || (_output_texture_state != RHIResourceState::ShaderReadOnly)) {
      return {};
    }
    return _output_texture;
  }
  void start() override;
  void stop() override;
  void finish() override;
  void restart() override;

  void set_output_dimensions(RHIContext& ctx, const uint2& dim);
  void on_camera_changed(SceneRepresentation& scene) override;
  void on_camera_become_steady(SceneRepresentation& scene) override;
  void on_scene_changed(SceneRepresentation& scene) override;
  void on_scene_transforms_changed(SceneRepresentation& scene) override;
  void on_scene_transform_interaction_started(SceneRepresentation& scene) override;
  void on_scene_transform_interaction_finished(SceneRepresentation& scene) override;

  Integrator* current_integrator() const;
  void set_integrator(Integrator*);
  Integrator** integrator_list();
  uint64_t integrator_count() const;

  IntegratorThread& integrator_thread() {
    return _integrator_thread;
  }

  const Scene& scene() const {
    return _raytracing.scene();
  }

  Film& film() {
    return _raytracing.film();
  }

 private:
  void restart_render_at_pixel_size(uint32_t pixel_size);
  void start_render_timing();
  void stop_render_timing();
  void reset_render_timing();
  bool update_image(RHIContext& ctx, RHICommandBuffer cmd, const float4* camera);

 private:
  Raytracing& _raytracing;
  IntegratorThread _integrator_thread;

  CPUDebugIntegrator _debug = {_raytracing};
  CPUBidirectional _cpu_pt = {_raytracing, BDPTMode::PathTracing, Integrator::Type::PathTracing};
  CPUBidirectional _cpu_bidir = {_raytracing};
  CPUVCM _cpu_vcm = {_raytracing};

  Integrator* _integrator_array[4] = {
    &_debug,      // Debug = 0
    &_cpu_pt,     // PathTracing = 1
    &_cpu_bidir,  // Bidirectional = 2
    &_cpu_vcm,    // VCM = 3
  };

  std::chrono::steady_clock::time_point _render_started_at = {};
  double _last_render_elapsed_seconds = 0.0;
  bool _render_timing_active = false;

  RHIBindlessHandle _output_staging_buffers[kRHIMaxFrames] = {};
  uint64_t _output_staging_buffer_sizes[kRHIMaxFrames] = {};
  RHIResourceState _output_texture_state = RHIResourceState::Undefined;
  uint32_t _last_uploaded_completed_iterations = 0u;
  uint32_t _last_uploaded_view_layer = kInvalidIndex;
  bool _display_output_valid = false;

  RHIPipeline rhi_pipeline = {};
};

}  // namespace etx
