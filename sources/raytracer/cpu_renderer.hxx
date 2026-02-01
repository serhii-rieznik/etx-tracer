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
  CPURaytracingRenderer(Raytracing& rt, SceneRepresentation& scene);
  ~CPURaytracingRenderer() override;

  void init(RHIContext* ctx, SceneRepresentation& scene) override;
  void prepare_frame(RHIContext* ctx, SceneRepresentation& scene, const FrameData&) override;
  void frame(RHIContext* ctx, SceneRepresentation& scene, const FrameData&) override;
  void cleanup(RHIContext* ctx) override;

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

  void set_output_dimensions(const uint2& dim);
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

  void set_reference_image(const char*);
  void set_reference_image(const float4 data[], const uint2 dimensions);

 private:
  void apply_reference_image(RHIContext* ctx, uint32_t);
  void update_image(const float4* camera);

 private:
  Raytracing& _raytracing;
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

  std::vector<Image> images;
  std::vector<ImageStorage> images_storage;
  ImagePool image_pool;

  RHIContext* context = nullptr;
  RHIPipeline rhi_pipeline = {};
  RHITexture rhi_output_texture = {};
  RHITexture rhi_reference_texture = {};
  uint32_t def_image_handle = kInvalidIndex;
  uint32_t ref_image_handle = kInvalidIndex;

  uint2 output_dimensions = {};
};

}  // namespace etx
