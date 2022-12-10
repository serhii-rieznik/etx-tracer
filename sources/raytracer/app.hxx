#pragma once

#include <etx/core/core.hxx>
#include <etx/core/handle.hxx>

#include <etx/render/host/scene_loader.hxx>
#include <etx/rt/integrators/debug.hxx>
#include <etx/rt/integrators/direct.hxx>
#include <etx/rt/integrators/path_tracing.hxx>
#include <etx/rt/integrators/path_tracing_gpu.hxx>
#include <etx/rt/integrators/bidirectional.hxx>
#include <etx/rt/integrators/vcm_cpu.hxx>
#include <etx/rt/integrators/vcm_gpu.hxx>
#include <etx/rt/integrators/atmosphere.hxx>
#include <etx/rt/rt.hxx>

#include <etx/gpu/gpu.hxx>

#include "ui.hxx"
#include "render.hxx"
#include "camera_controller.hxx"

namespace etx {

struct RTApplication {
  RTApplication();

  void init();
  void frame();
  void cleanup();
  void process_event(const sapp_event*);
  void load_scene_file(const std::string&, uint32_t options, bool start_rendering);

 private:
  void on_referenece_image_selected(std::string);
  void on_save_image_selected(std::string, SaveImageMode);
  void on_scene_file_selected(std::string);
  void on_integrator_selected(Integrator*);
  void on_preview_selected();
  void on_run_selected();
  void on_stop_selected(bool wait_for_completion);
  void on_reload_scene_selected();
  void on_reload_geometry_selected();
  void on_options_changed();
  void on_reload_integrator_selected();
  void on_use_image_as_reference();
  void on_material_changed(uint32_t index);
  void on_medium_changed(uint32_t index);

 private:
  std::vector<float4> get_current_image(bool convert_to_rgb);
  void save_options();

 private:
  UI ui;
  TimeMeasure time_measure;
  Raytracing raytracing;

  RenderContext render;
  SceneRepresentation scene;
  CameraController camera_controller;

  CPUDebugIntegrator _preview = {raytracing};
  CPUDirectLighting _cpu_direct = {raytracing};
  CPUPathTracing _cpu_pt = {raytracing};
  GPUPathTracing _gpu_pt = {raytracing};
  CPUBidirectional _cpu_bidir = {raytracing};
  CPUVCM _cpu_vcm = {raytracing};
  GPUVCM _gpu_vcm = {raytracing};
  CPUAtmosphere _cpu_atmosphere = {raytracing};

  Integrator* _integrator_array[8] = {
    &_preview,
    &_cpu_direct,
    &_cpu_pt,
    &_cpu_bidir,
    &_cpu_vcm,
    &_gpu_pt,
    &_gpu_vcm,
    &_cpu_atmosphere,
  };

  Integrator* _current_integrator = nullptr;
  std::string _current_scene_file = {};
  Options _options;

  bool _reset_images = true;
};

}  // namespace etx
