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
#include "platform_ui.hxx"
#include "application_control.hxx"

#include <vector>
#include <string>
#include <memory>
#include <atomic>
#include <mutex>

namespace etx {

struct ApplicationConfig {
  RuntimeMode runtime_mode = RuntimeMode::Desktop;
  uint32_t width = 1600u;
  uint32_t height = 900u;
  bool enable_imgui = true;
  bool enable_platform_ui = true;
  bool persist_options = true;
  bool override_renderer = false;
  RendererMode renderer = RendererMode::CPURaytracing;
};

struct RTApplication {
  RTApplication();
  ~RTApplication();

  void prepare_startup();
  void init(const ApplicationConfig& config = {});
  void frame();
  void cleanup();
  void process_event(const sapp_event*);

  uint64_t submit_command(ApplicationCommand command);
  ApplicationStateSnapshot state_snapshot() const;
  void drain_command_results(std::vector<ApplicationCommandResult>& output);
  bool quit_requested() const;
  bool initialized() const;
  bool capture_output_png(std::vector<uint8_t>& png_data, uint32_t& width, uint32_t& height);

  void set_renderer_mode(RendererMode mode);
  bool set_render_configuration(RendererMode mode, Integrator::Type integrator_type);

 private:
  bool load_scene_file(const std::string&, uint32_t options, bool start_rendering);
  std::string save_scene_file(const std::string&);

  void on_referenece_image_selected(std::string);
  void on_save_image_selected(std::string, SaveImageMode);
  void on_integrator_selected(Integrator::Type);
  void on_run_selected();
  void on_stop_selected(bool wait_for_completion);
  void on_restart_selected();
  void on_options_changed();
  void on_use_image_as_reference();
  uint32_t on_material_added();
  void on_material_renamed(uint32_t index, const std::string&);
  void on_material_changed(uint32_t index);
  void on_material_interaction_started();
  void on_material_interaction_finished(const std::vector<uint32_t>& material_indices);
  uint32_t on_medium_added();
  void on_medium_renamed(uint32_t index, const std::string&);
  void on_medium_changed(uint32_t index);
  void on_mesh_material_changed(uint32_t mesh_index, uint32_t material_index);
  uint32_t on_make_mesh_material_unique(uint32_t mesh_index, uint32_t material_index);
  void on_emitter_changed(uint32_t index);
  void on_emitter_added(uint32_t type);
  bool on_emitter_deleted(uint32_t index);
  void on_camera_changed(uint2 viewport, uint32_t pixel_size);
  void on_scene_settings_changed();
  void on_scene_transforms_changed();
  void on_scene_transform_interaction_started();
  void on_scene_transform_interaction_finished();
  void on_denoise_selected();
  void on_view_scene(uint32_t direction);
  void on_clear_recent_files();
  void on_camera_activated(uint32_t camera_index);
  void on_reload_shaders_selected();
  void on_cancel_renderer_preparation_selected();

 private:
  void add_to_recent(const std::string&);
  bool ensure_gpu_renderer_initialized();
  void process_pending_image_requests();
  bool read_active_renderer_output(std::vector<float4>& output, uint2& image_size);
  void save_options();
  void update_camera_to_fit_scene(const float3& view_direction);
  void notify_scene_might_have_changed();
  void notify_scene_transforms_changed();
  void sync_ui_renderer_state();
  void process_application_commands();
  bool execute_application_command(const ApplicationCommand& command, std::string& message);
  void publish_application_state();
  void sync_scene_integrator_data_from_current_integrator();
  void sync_platform_color_scheme();
  bool rebuild_material_render_resources();
  void poll_material_render_resource_preparation();
  void finish_material_render_resource_preparation(bool resources_ready);
  void mark_scene_dirty();

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
  Renderer* _scene_transform_interaction_renderer = nullptr;
  Renderer* _pending_reference_capture_renderer = nullptr;
  bool _gpu_renderer_initialized = false;
  bool _gpu_renderer_supported = false;
  bool _quit_preparation_cancel_requested = false;
  bool _pending_current_image_reference_capture = false;
  bool _pending_reference_file_load = false;
  bool _pending_gpu_save_image = false;
  bool _startup_frame_presented = false;
  bool _initialization_started = false;
  std::atomic<bool> _initialized = false;
  bool _scene_global_initialized = false;
  bool _scene_transform_interaction_active = false;
  bool _material_interaction_active = false;
  bool _material_interaction_cpu_was_running = false;
  bool _material_render_resource_preparation_active = false;
  bool _restart_cpu_after_material_resource_preparation = false;
  bool _scene_dirty = false;
  bool _platform_color_scheme_initialized = false;
  PlatformColorScheme _platform_color_scheme = PlatformColorScheme::Dark;
  SaveImageMode _pending_gpu_save_image_mode = SaveImageMode::RGB;

  Options _options;
  ViewParameters _view_parameters = {
    .exposure = 1.0f,
    .view_option = static_cast<uint32_t>(ViewOptions::Tonemapped),
    .view_image = static_cast<uint32_t>(OutputView::OutputImage),
    .view_layer = static_cast<uint32_t>(ViewLayer::Result),
  };
  std::vector<std::string> _recent_files = {};
  std::string _pending_reference_file = {};
  std::string _pending_gpu_save_image_file = {};
  std::string _current_scene_file = {};
  TimeMeasure time_measure = {};
  TimeMeasure scene_commit_time = {};

  mutable std::mutex _application_control_mutex = {};
  std::vector<ApplicationCommand> _pending_application_commands = {};
  std::vector<ApplicationCommandResult> _application_command_results = {};
  ApplicationStateSnapshot _application_state = {};
  std::atomic<uint64_t> _next_application_command_id = 1u;
  bool _application_quit_requested = false;
  ApplicationConfig _application_config = {};
};

}  // namespace etx
