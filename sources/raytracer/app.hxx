#pragma once

#include <etx/core/core.hxx>
#include <etx/core/handle.hxx>

#include <etx/render/host/scene_representation.hxx>
#include <etx/render/shared/ior_database.hxx>
#include <etx/render/host/film.hxx>

#include <etx/rt/integrators/debug.hxx>
#include <etx/rt/integrators/path_tracing.hxx>
#include <etx/rt/integrators/bidirectional.hxx>
#include <etx/rt/integrators/vcm_cpu.hxx>
#include <etx/rt/rt.hxx>

#include <etx/gpu/gpu.hxx>

#include "ui.hxx"
#include "render.hxx"
#include "camera_controller.hxx"

namespace etx {

struct RTApplication {
  RTApplication();
  ~RTApplication();

  void init();
  void frame();
  void cleanup();
  void process_event(const sapp_event*);

 private:
  void load_scene_file(const std::string&, uint32_t options, bool start_rendering);
  std::string save_scene_file(const std::string&);

  void on_referenece_image_selected(std::string);
  void on_save_image_selected(std::string, SaveImageMode);
  void on_scene_file_selected(std::string);
  void on_save_scene_file_selected(std::string);
  void on_save_scene_file_as_selected();
  void on_integrator_selected(Integrator*);
  void on_run_selected();
  void on_stop_selected(bool wait_for_completion);
  void on_restart_selected();
  void on_reload_scene_selected();
  void on_reload_geometry_selected();
  void on_options_changed();
  void on_use_image_as_reference();
  void on_material_changed(uint32_t index);
  void on_medium_added();
  void on_medium_changed(uint32_t index);
  void on_mesh_material_changed(uint32_t mesh_index, uint32_t material_index);
  void on_emitter_changed(uint32_t index);
  void on_camera_changed(bool film_changed);
  void on_scene_settings_changed();
  void on_denoise_selected();
  void on_view_scene(uint32_t direction);
  void on_clear_recent_files();

 private:
  void add_to_recent(const std::string&);
  void save_options();

 private:
  UI ui;
  Raytracing raytracing;
  RenderContext render;
  IORDatabase _ior_database;
  SceneRepresentation scene;
  CameraController camera_controller;
  IntegratorThread integrator_thread;

  CPUDebugIntegrator _debug = {raytracing};
  CPUPathTracing _cpu_pt = {raytracing};
  CPUBidirectional _cpu_bidir = {raytracing};
  CPUVCM _cpu_vcm = {raytracing};

  Integrator* _integrator_array[4] = {
    &_debug,
    &_cpu_pt,
    &_cpu_bidir,
    &_cpu_vcm,
  };

  Options _options;
  std::vector<std::string> _recent_files = {};
  std::string _current_scene_file = {};
  TimeMeasure time_measure = {};
  bool last_camera_controller_state = false;
};

}  // namespace etx
