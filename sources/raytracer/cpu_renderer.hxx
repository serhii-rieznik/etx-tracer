#pragma once

#include "renderer.hxx"
#include <etx/render/shared/scene.hxx>

#include <etx/rt/rt.hxx>
#include <etx/rt/integrators/debug.hxx>
#include <etx/rt/integrators/path_tracing.hxx>
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
  void start() override;
  void stop() override;
  void finish() override;
  void restart() override;

  void set_output_dimensions(RHIContext& ctx, const uint2& dim);
  void on_camera_changed(SceneRepresentation& scene) override;
  void on_camera_become_steady(SceneRepresentation& scene) override;
  void on_scene_changed(SceneRepresentation& scene) override;

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
  void start_render_timing();
  void stop_render_timing();
  void reset_render_timing();
  void update_image(RHIContext& ctx, const float4* camera);

 private:
  Raytracing& _raytracing;
  IntegratorThread _integrator_thread;

  CPUDebugIntegrator _debug = {_raytracing};
  CPUPathTracing _cpu_pt = {_raytracing};
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

  RHIPipeline rhi_pipeline = {};
};

}  // namespace etx
