#pragma once

#pragma once

#if !defined(WIN32_LEAN_AND_MEAN)
# define WIN32_LEAN_AND_MEAN 1
#endif

#include <etx/render/interop/interop.hxx>

#include <etx/core/core.hxx>
#include <etx/core/handle.hxx>
#include <etx/core/environment.hxx>
#include <etx/render/host/tasks.hxx>

#include <etx/render/host/scene_representation.hxx>
#include <etx/render/shared/ior_database.hxx>
#include <etx/render/host/film.hxx>

#include <etx/rt/integrators/integrator.hxx>
#include <etx/rt/integrators/debug.hxx>
#include <etx/rt/integrators/path_tracing.hxx>
#include <etx/rt/integrators/bidirectional.hxx>

#include "ui.hxx"
#include "render_context.hxx"
#include "renderer.hxx"
#include "cpu_renderer.hxx"
#include "raster_renderer.hxx"
#include "gpu_renderer.hxx"

#include <vector>
#include <string>
#include <memory>

namespace etx {

struct RTApplication {
  RTApplication();
  ~RTApplication();

  void init();
  void frame();
  void cleanup();
  void process_event(const sapp_event*);

  void set_renderer_mode(RendererMode mode);

 private:
  void load_scene_file(const std::string&, uint32_t options, bool start_rendering);
  std::string save_scene_file(const std::string&);

  void on_referenece_image_selected(std::string);
  void on_save_image_selected(std::string, SaveImageMode);
  void on_scene_file_selected(std::string);
  void on_save_scene_file_selected(std::string);
  void on_save_scene_file_as_selected();
  void on_integrator_selected(Integrator::Type);
  void on_run_selected();
  void on_stop_selected(bool wait_for_completion);
  void on_restart_selected();
  void on_reload_scene_selected();
  void on_reload_geometry_selected();
  void on_options_changed();
  void on_use_image_as_reference();
  void on_material_added();
  void on_material_renamed(uint32_t index, const std::string&);
  void on_material_changed(uint32_t index);
  void on_medium_added();
  void on_medium_renamed(uint32_t index, const std::string&);
  void on_medium_changed(uint32_t index);
  void on_mesh_material_changed(uint32_t mesh_index, uint32_t material_index);
  void on_mesh_renamed(uint32_t index, const std::string&);
  void on_emitter_changed(uint32_t index);
  void on_emitter_added(uint32_t type);
  void on_camera_changed(uint2 viewport, uint32_t pixel_size);
  void on_scene_settings_changed();
  void on_denoise_selected();
  void on_view_scene(uint32_t direction);
  void on_clear_recent_files();
  void on_camera_activated(uint32_t camera_index);
  void on_scene_update_requested();
  void on_reload_shaders_selected();

 private:
  void add_to_recent(const std::string&);
  void ensure_gpu_renderer_initialized();
  void save_options();
  void update_camera_to_fit_scene(const float3& view_direction);
  void notify_scene_might_have_changed();

 private:
  TaskScheduler scheduler;
  Film film;
  Raytracing rt;
  RenderContext render_context;
  IORDatabase _ior_database;
  SceneRepresentation scene;
  UI ui;

  CPURaytracingRenderer cpu_renderer;
  RasterizationRenderer raster_renderer;
  GPURaytracingRenderer gpu_renderer;
  Renderer* _active_renderer = nullptr;
  bool _gpu_renderer_initialized = false;
  bool _gpu_renderer_supported = false;

  Options _options;
  std::vector<std::string> _recent_files = {};
  std::string _current_scene_file = {};
  TimeMeasure time_measure = {};
  TimeMeasure scene_commit_time = {};
};

}  // namespace etx
