#pragma once

#include "renderer.hxx"
#include <etx/render/shared/scene.hxx>
#include <etx/rt/rt.hxx>
#include <etx/rt/integrators/debug.hxx>
#include <etx/rt/integrators/path_tracing.hxx>
#include <etx/rt/integrators/bidirectional.hxx>
#include <etx/rt/integrators/bdpt_distilled.hxx>
#include <etx/rt/integrators/vcm_cpu.hxx>

namespace etx {

struct CPURaytracingRenderer : public Renderer {
  CPURaytracingRenderer(TaskScheduler& scheduler, SceneRepresentation& scene);
  ~CPURaytracingRenderer() override;

  void init(RenderContext& render_context, SceneRepresentation& scene) override;
  void frame(RenderContext& render_context, SceneRepresentation& scene, float dt) override;
  void cleanup(RenderContext& render_context) override;

  const char* name() const override {
    return "CPU Raytracing";
  }
  RendererMode mode() const override {
    return RendererMode::CPURaytracing;
  }

  bool is_running() const override;
  void start() override;
  void stop() override;
  void restart() override;

  void on_camera_changed(SceneRepresentation& scene, bool path_changed) override;
  void on_scene_changed(SceneRepresentation& scene) override;

  // Extra methods for CPURaytracing
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
  Raytracing _raytracing;
  IntegratorThread _integrator_thread;

  CPUDebugIntegrator _debug = {_raytracing};
  CPUPathTracing _cpu_pt = {_raytracing};
  CPUBidirectional _cpu_bidir = {_raytracing};
  BDPTDistilled _bdpt_distilled = {_raytracing};
  CPUVCM _cpu_vcm = {_raytracing};

  Integrator* _integrator_array[5] = {
    &_debug,           // Debug = 0
    &_cpu_pt,          // PathTracing = 1
    &_cpu_bidir,       // Bidirectional = 2
    &_cpu_vcm,         // VCM = 3
    &_bdpt_distilled,  // BDPTDistilled = 4
  };
};

}  // namespace etx
