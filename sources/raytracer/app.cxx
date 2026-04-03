#include <etx/core/core.hxx>
#include <etx/core/environment.hxx>
#include <etx/core/profiler.hxx>

#include <etx/render/host/scene_global.hxx>
#include <etx/rhi/shader/shader_compiler.hxx>
#include <etx/render/shared/camera.hxx>
#include <etx/rt/integrators/integrator.hxx>

#include "app.hxx"
#include "image_output.hxx"

#include <algorithm>
#include <cstring>
#include <filesystem>

#if defined(ETX_PLATFORM_WINDOWS)
# define WIN32_LEAN_AND_MEAN 1
# include <Windows.h>
#endif

#include <interop/render_options.hxx>

namespace etx {

namespace {

constexpr uint32_t kRecentFileLimit = 8u;

}

RTApplication::RTApplication()
  : rt(scheduler, film)
  , film(scheduler)
  , scene(scheduler, _ior_database)
  , render_context(scheduler)
  , cpu_renderer(rt, scene)
  , raster_renderer(scheduler)
  , gpu_renderer(scheduler) {
  _active_renderer = &cpu_renderer;
}

RTApplication::~RTApplication() {
  save_options();
}

void RTApplication::init() {
  ETX_PROFILER_SCOPE();

  scene_global_init();

  {
    ETX_PROFILER_NAMED_SCOPE("app_load_options");
    std::string options_file = env().file_in_data("options.json");
    _options.load_from_file(options_file);
  }

  {
    ETX_PROFILER_NAMED_SCOPE("app_init_render_context_and_ior");
    render_context.init();
    if (render_context.valid() == false) {
      log::error("Failed to initialize rendering context");
      return;
    }
    scene.set_scattering_rhi(render_context.get_context());
    _gpu_renderer_supported = render_context.get_context().capabilities().supports_ray_tracing;
    ui.set_gpu_renderer_available(_gpu_renderer_supported);
    if (_gpu_renderer_supported == false) {
      log::warning("GPU ray tracing is not supported by the active RHI backend; falling back to CPU or raster rendering");
    }
    std::string ior_folder = env().file_in_data("./spectrum/");
    _ior_database.load(ior_folder.c_str());
  }

  {
    ETX_PROFILER_NAMED_SCOPE("app_init_renderers");
    ui.set_integrator_list(cpu_renderer.integrator_list(), cpu_renderer.integrator_count());

    cpu_renderer.init(render_context.get_context(), scene);
    raster_renderer.init(render_context.get_context(), scene);
  }

  RendererMode mode = RendererMode::CPURaytracing;
  auto renderer_name = _options.get_string("renderer", "cpu");
  if (renderer_name == "gpu") {
    mode = RendererMode::GPURaytracing;
  } else if (renderer_name == "raster") {
    mode = RendererMode::Rasterization;
  }
  {
    ETX_PROFILER_NAMED_SCOPE("app_set_initial_renderer_mode");
    set_renderer_mode(mode);
  }

  {
    ETX_PROFILER_NAMED_SCOPE("app_bind_ui_callbacks");
    ui.callbacks.reference_image_selected = std::bind(&RTApplication::on_referenece_image_selected, this, std::placeholders::_1);
    ui.callbacks.save_image_selected = std::bind(&RTApplication::on_save_image_selected, this, std::placeholders::_1, std::placeholders::_2);
    ui.callbacks.scene_file_selected = std::bind(&RTApplication::on_scene_file_selected, this, std::placeholders::_1);
    ui.callbacks.save_scene_file_selected = std::bind(&RTApplication::on_save_scene_file_selected, this, std::placeholders::_1);
    ui.callbacks.save_scene_file_as_selected = std::bind(&RTApplication::on_save_scene_file_as_selected, this);
    ui.callbacks.renderer_selected = std::bind(&RTApplication::set_renderer_mode, this, std::placeholders::_1);
    ui.callbacks.run_selected = std::bind(&RTApplication::on_run_selected, this);
    ui.callbacks.stop_selected = std::bind(&RTApplication::on_stop_selected, this, std::placeholders::_1);
    ui.callbacks.restart_selected = std::bind(&RTApplication::on_restart_selected, this);
    ui.callbacks.reload_scene_selected = std::bind(&RTApplication::on_reload_scene_selected, this);
    ui.callbacks.reload_geometry_selected = std::bind(&RTApplication::on_reload_geometry_selected, this);
    ui.callbacks.options_changed = std::bind(&RTApplication::on_options_changed, this);
    ui.callbacks.reload_shaders_selected = std::bind(&RTApplication::on_reload_shaders_selected, this);
    ui.callbacks.use_image_as_reference = std::bind(&RTApplication::on_use_image_as_reference, this);
    ui.callbacks.material_added = std::bind(&RTApplication::on_material_added, this);
    ui.callbacks.material_renamed = std::bind(&RTApplication::on_material_renamed, this, std::placeholders::_1, std::placeholders::_2);
    ui.callbacks.material_changed = std::bind(&RTApplication::on_material_changed, this, std::placeholders::_1);
    ui.callbacks.medium_added = std::bind(&RTApplication::on_medium_added, this);
    ui.callbacks.medium_renamed = std::bind(&RTApplication::on_medium_renamed, this, std::placeholders::_1, std::placeholders::_2);
    ui.callbacks.medium_changed = std::bind(&RTApplication::on_medium_changed, this, std::placeholders::_1);
    ui.callbacks.mesh_material_changed = std::bind(&RTApplication::on_mesh_material_changed, this, std::placeholders::_1, std::placeholders::_2);
    ui.callbacks.mesh_renamed = std::bind(&RTApplication::on_mesh_renamed, this, std::placeholders::_1, std::placeholders::_2);
    ui.callbacks.emitter_changed = std::bind(&RTApplication::on_emitter_changed, this, std::placeholders::_1);
    ui.callbacks.emitter_added = std::bind(&RTApplication::on_emitter_added, this, std::placeholders::_1);
    ui.callbacks.camera_changed = std::bind(&RTApplication::on_camera_changed, this, std::placeholders::_1, std::placeholders::_2);
    ui.callbacks.scene_settings_changed = std::bind(&RTApplication::on_scene_settings_changed, this);
    ui.callbacks.denoise_selected = std::bind(&RTApplication::on_denoise_selected, this);
    ui.callbacks.view_scene = std::bind(&RTApplication::on_view_scene, this, std::placeholders::_1);
    ui.callbacks.clear_recent_files = std::bind(&RTApplication::on_clear_recent_files, this);
    ui.callbacks.camera_activated = std::bind(&RTApplication::on_camera_activated, this, std::placeholders::_1);
    ui.callbacks.scene_update_requested = std::bind(&RTApplication::on_scene_update_requested, this);
    ui.callbacks.integrator_selected = std::bind(&RTApplication::on_integrator_selected, this, std::placeholders::_1);
  }

  {
    ETX_PROFILER_NAMED_SCOPE("app_restore_recent_files");
    for (uint32_t i = 0; i < kRecentFileLimit; ++i) {
      const auto name = "recent-" + std::to_string(i);
      if (_options.has(name, Option::Class::String)) {
        const std::string restored_path = _options.get_string(name, {});
        if (restored_path.empty() == false) {
          add_to_recent(restored_path);
        }
      }
    }
  }

#if defined(ETX_PLATFORM_WINDOWS)
  if (GetAsyncKeyState(VK_ESCAPE)) {
    _options.set_string("integrator", {}, "Integrator");
  }
  if (GetAsyncKeyState(VK_ESCAPE) && GetAsyncKeyState(VK_SHIFT)) {
    _options.set_string("scene", {}, "Scene");
  }
#endif

  Integrator* integrator = nullptr;

  {
    ETX_PROFILER_NAMED_SCOPE("app_select_integrator");
    const auto& selected_integrator = _options.get_string("integrator", std::string{});
    for (uint64_t i = 0; (selected_integrator.empty() == false) && (i < (uint64_t)cpu_renderer.integrator_count()); ++i) {
      Integrator* it = cpu_renderer.integrator_list()[i];
      ETX_ASSERT(it != nullptr);
      if (selected_integrator == it->name()) {
        integrator = it;
      }
    }

    if (integrator == nullptr && cpu_renderer.integrator_count() > 0) {
      integrator = cpu_renderer.integrator_list()[1];  // pt
    }
  }

  cpu_renderer.set_integrator(integrator);
  ui.set_current_integrator(integrator);

  {
    ETX_PROFILER_NAMED_SCOPE("app_restore_last_scene_and_reference");
    _current_scene_file = _options.get_string("scene", std::string{});
    if (_current_scene_file.empty() == false) {
      on_scene_file_selected(_current_scene_file);
    }

    const auto& ref = _options.get_string("ref", std::string{});
    if (ref.empty() == false) {
      on_referenece_image_selected(ref);
    }
  }

  save_options();
}

void RTApplication::save_options() {
  ETX_PROFILER_SCOPE();

  for (uint32_t idx = 0; idx < kRecentFileLimit; ++idx) {
    _options.remove("recent-" + std::to_string(idx));
  }
  uint32_t i = 0;
  for (const auto& recent : _recent_files) {
    _options.set_string("recent-" + std::to_string(i++), recent, "Recent File");
  }
  if (_current_scene_file.empty() == false) {
    _options.set_string("scene", _current_scene_file, "Scene");
  }
  _options.save_to_file(env().file_in_data("options.json"));
}

bool RTApplication::ensure_gpu_renderer_initialized() {
  ETX_PROFILER_SCOPE();

  if (_gpu_renderer_supported == false) {
    log::warning("GPU ray tracing is unavailable for the active RHI backend");
    return false;
  }

  if (_gpu_renderer_initialized) {
    return true;
  }

  gpu_renderer.init(render_context.get_context(), scene);
  if (gpu_renderer.runtime_failed()) {
    log::warning("GPU ray tracing initialization failed: %s", gpu_renderer.runtime_failure_reason().c_str());
    return false;
  }
  _gpu_renderer_initialized = true;
  return true;
}

void RTApplication::set_renderer_mode(RendererMode mode) {
  ETX_PROFILER_SCOPE();

  if ((mode == RendererMode::GPURaytracing) && (_gpu_renderer_supported == false)) {
    log::warning("GPU ray tracing is unavailable for the active RHI backend; using CPU ray tracing instead");
    mode = RendererMode::CPURaytracing;
  }

  Renderer* next_renderer = nullptr;
  std::string renderer_name;
  switch (mode) {
    case RendererMode::Rasterization:
      next_renderer = &raster_renderer;
      renderer_name = "raster";
      break;

    case RendererMode::GPURaytracing:
      next_renderer = &gpu_renderer;
      renderer_name = "gpu";
      break;

    default:
      next_renderer = &cpu_renderer;
      renderer_name = "cpu";
      break;
  }

  _options.set_string("renderer", renderer_name, "Renderer");
  save_options();

  if (next_renderer == &gpu_renderer) {
    if (ensure_gpu_renderer_initialized() == false) {
      log::warning("GPU ray tracing is unavailable for the active RHI backend; using CPU ray tracing instead");
      next_renderer = &cpu_renderer;
      renderer_name = "cpu";
      mode = RendererMode::CPURaytracing;
    }
  }

  if (next_renderer == _active_renderer) {
    return;
  }

  if (_active_renderer != nullptr) {
    _active_renderer->stop();
  }

  _active_renderer = next_renderer;

  if (_active_renderer != nullptr) {
    _active_renderer->start();
  }

  ui.set_current_renderer_mode(mode);
}

void RTApplication::frame() {
  ETX_PROFILER_SCOPE();

  auto thread = _active_renderer && (_active_renderer->mode() == RendererMode::CPURaytracing) ? &cpu_renderer.integrator_thread() : nullptr;

  RenderContext::FrameData render_frame_data = {
    .dt = float(time_measure.lap()),
    .sample_count = thread ? thread->status().current_iteration : 0u,
    .view_parameters = ui.view_options(),
  };

  UI::FrameData ui_frame_data = {
    .ior_database = _ior_database,
    .recent_files = _recent_files,
    .film = film,
    .dt = render_frame_data.dt,
  };

  {
    ETX_PROFILER_NAMED_SCOPE("app_render_context_start_frame");
    render_context.start_frame(_active_renderer, scene, render_frame_data);
  }
  if (render_context.valid() && render_context.rhi_ui().initialized()) {
    ETX_PROFILER_NAMED_SCOPE("app_ui_build");
    ui.build(scene, ui_frame_data);
  }
  {
    ETX_PROFILER_NAMED_SCOPE("app_render_context_end_frame");
    render_context.end_frame();
  }

  if ((_active_renderer == &gpu_renderer) && gpu_renderer.runtime_failed()) {
    log::warning("GPU ray tracing failed at runtime: %s. Falling back to CPU ray tracing.", gpu_renderer.runtime_failure_reason().c_str());
    set_renderer_mode(RendererMode::CPURaytracing);
  }
}

void RTApplication::cleanup() {
  ETX_PROFILER_SCOPE();

  if (_active_renderer != nullptr) {
    _active_renderer->stop();
  }

  if (render_context.valid()) {
    auto& ctx = render_context.get_context();
    cpu_renderer.cleanup(ctx);
    raster_renderer.cleanup(ctx);
    if (_gpu_renderer_initialized) {
      gpu_renderer.cleanup(ctx);
    }
  }

  scene_global_deinit();
  render_context.cleanup();
  ShaderCompiler::instance().shutdown();
}

void RTApplication::process_event(const sapp_event* e) {
  ETX_PROFILER_SCOPE();

  {
    ETX_PROFILER_NAMED_SCOPE("app_process_event_imgui");
    if (render_context.valid() && render_context.rhi_ui().handle_event(e))
      return;
  }

  {
    ETX_PROFILER_NAMED_SCOPE("app_process_event_ui");
    if (ui.handle_event(e))
      return;
  }

  if (_active_renderer != nullptr) {
    ETX_PROFILER_NAMED_SCOPE("app_process_event_renderer");
    _active_renderer->process_event(e);
  }
}

void RTApplication::add_to_recent(const std::string& value) {
  ETX_PROFILER_SCOPE();

  if (value.empty()) {
    return;
  }

  const std::string absolute_path = env().resolve_to_absolute(value);
  if (absolute_path.empty()) {
    return;
  }

  auto e = std::remove_if(_recent_files.begin(), _recent_files.end(), [&](const std::string& entry) {
    return env().resolve_to_absolute(entry) == absolute_path;
  });
  _recent_files.erase(e, _recent_files.end());

  _recent_files.emplace_back(absolute_path);

  if (_recent_files.size() > kRecentFileLimit) {
    _recent_files.erase(_recent_files.begin());
  }
}

void RTApplication::load_scene_file(const std::string& file_name, uint32_t options, bool start_rendering) {
  ETX_PROFILER_SCOPE();

  _current_scene_file = env().resolve_to_absolute(file_name);

  cpu_renderer.stop();
  _options.set_string("scene", _current_scene_file, "Scene");
  save_options();

  log::warning("Loading scene %s...", _current_scene_file.c_str());
  SceneRepresentation::IntegratorData integrator_data;
  {
    ETX_PROFILER_NAMED_SCOPE("app_scene_load_from_file");
    if (scene.load_from_file(_current_scene_file.c_str(), options, &integrator_data) == false) {
      log::error("Failed to load scene from file: %s", _current_scene_file.c_str());
    }
  }
  log::warning("Setting output dimensions...");
  {
    ETX_PROFILER_NAMED_SCOPE("app_scene_set_output_dimensions");
    cpu_renderer.set_output_dimensions(render_context.get_context(), scene.camera().film_size);
  }

  if (scene.valid() == false) {
    return;
  }

  notify_scene_might_have_changed();

  {
    ETX_PROFILER_NAMED_SCOPE("app_scene_sync_integrator_settings");
    for (const auto& [type, options] : integrator_data.settings) {
      Integrator* integrator = integrator_type_to_instance(type, cpu_renderer.integrator_list(), cpu_renderer.integrator_count());
      if (integrator != nullptr) {
        integrator->sync_from_options(options);
        integrator->update_options();
      }
    }
  }

  Integrator* integrator = nullptr;
  {
    ETX_PROFILER_NAMED_SCOPE("app_scene_select_integrator");
    if (integrator_data.selected != Integrator::Type::Invalid) {
      integrator = integrator_type_to_instance(integrator_data.selected, cpu_renderer.integrator_list(), cpu_renderer.integrator_count());
    }

    if (integrator == nullptr && cpu_renderer.integrator_count() > 0) {
      integrator = cpu_renderer.integrator_list()[1];  // pt
    }
  }

  cpu_renderer.set_integrator(integrator);
  ui.set_current_integrator(integrator);

  add_to_recent(_current_scene_file);
  save_options();

  {
    ETX_PROFILER_NAMED_SCOPE("app_scene_restart_cpu_renderer");
    cpu_renderer.film().clear(Film::ClearEverything);
    cpu_renderer.start();
  }
}

std::string RTApplication::save_scene_file(const std::string& file_name) {
  ETX_PROFILER_SCOPE();

  log::info("Saving %s..", file_name.c_str());
  Integrator* current = cpu_renderer.current_integrator();
  Integrator::Type selected_type = integrator_to_type(current);
  std::string saved_path = scene.save_to_file(file_name.c_str(), selected_type, cpu_renderer.integrator_list(), cpu_renderer.integrator_count());
  if (saved_path.empty()) {
    log::error("Failed to save scene to %s", file_name.c_str());
    return {};
  }

  _current_scene_file = env().resolve_to_absolute(saved_path);
  _options.set_string("scene", _current_scene_file, "Scene");
  add_to_recent(_current_scene_file);
  save_options();

  return _current_scene_file;
}

void RTApplication::on_referenece_image_selected(std::string file_name) {
  ETX_PROFILER_SCOPE();

  log::warning("Loading reference image %s...", file_name.c_str());

  _options.set_string("ref", file_name, "Reference");
  save_options();

  render_context.set_reference_image(file_name.c_str());
}

void RTApplication::on_use_image_as_reference() {
  ETX_PROFILER_SCOPE();

  _options.set_string("ref", {}, "Reference");
  save_options();

  const float4* data = cpu_renderer.film().layer(ViewLayer::Result, cpu_renderer.scene().options.radiance_clamp);
  uint2 size = cpu_renderer.film().base_dimensions();
  render_context.set_reference_image(data, size);
}

void RTApplication::on_save_image_selected(std::string file_name, SaveImageMode mode) {
  ETX_PROFILER_SCOPE();

  uint2 image_size = {scene.camera().film_size.x, scene.camera().film_size.y};
  const float4* output = cpu_renderer.film().layer(ui.view_options().view_layer, cpu_renderer.scene().options.radiance_clamp);
  ImageOutputParameters params = {
    .mode = mode,
    .exposure = ui.view_options().exposure,
  };
  save_image_to_file(file_name, output, image_size, params);
}

void RTApplication::on_scene_file_selected(std::string file_name) {
  ETX_PROFILER_SCOPE();
  load_scene_file(file_name, SceneRepresentation::LoadEverything, false);
}

void RTApplication::on_save_scene_file_selected(std::string file_name) {
  ETX_PROFILER_SCOPE();

  std::string base = file_name;
  if (base.empty()) {
    base = _current_scene_file;
  }

  if (base.empty()) {
    log::warning("No scene file available for saving");
    return;
  }

  if (std::strlen(get_file_ext(base.c_str())) == 0) {
    base += ".json";
  }

  std::string saved_path = save_scene_file(base);
  if (saved_path.empty() == false) {
    log::info("Scene saved to %s", saved_path.c_str());
  }
}

void RTApplication::on_save_scene_file_as_selected() {
  ETX_PROFILER_SCOPE();
  std::string selected_file = save_file("json");
  if (selected_file.empty() == false) {
    on_save_scene_file_selected(selected_file);
  }
}

void RTApplication::on_integrator_selected(Integrator::Type itype) {
  ETX_PROFILER_SCOPE();

  Integrator* i = integrator_type_to_instance(itype, cpu_renderer.integrator_list(), cpu_renderer.integrator_count());
  if (i == nullptr)
    return;

  cpu_renderer.set_integrator(i);
  cpu_renderer.film().clear(Film::ClearEverything);

  if (scene.valid()) {
    cpu_renderer.start();
  }
}

void RTApplication::on_run_selected() {
  ETX_PROFILER_SCOPE();

  if (ui.view_options().view_layer == ViewLayer::Denoised) {
    ui.mutable_view_options().view_layer = ViewLayer::Result;
  }
  cpu_renderer.film().clear(Film::ClearEverything);
  cpu_renderer.start();
}

void RTApplication::on_stop_selected(bool wait_for_completion) {
  ETX_PROFILER_SCOPE();
  cpu_renderer.stop();
}

void RTApplication::on_restart_selected() {
  ETX_PROFILER_SCOPE();
  on_stop_selected(false);
  on_run_selected();
}

void RTApplication::on_reload_scene_selected() {
  ETX_PROFILER_SCOPE();
  if (_current_scene_file.empty() == false) {
    load_scene_file(_current_scene_file, SceneRepresentation::LoadEverything, cpu_renderer.is_running());
  }
}

void RTApplication::on_reload_geometry_selected() {
  ETX_PROFILER_SCOPE();
  if (_current_scene_file.empty() == false) {
    load_scene_file(_current_scene_file, SceneRepresentation::LoadGeometry, cpu_renderer.is_running());
  }
}

void RTApplication::on_options_changed() {
  ETX_PROFILER_SCOPE();
  notify_scene_might_have_changed();
  cpu_renderer.restart();
}

void RTApplication::on_material_added() {
  scene.add_material(nullptr);
  notify_scene_might_have_changed();
}

void RTApplication::on_material_renamed(uint32_t index, const std::string& name) {
  scene.rename_material(index, name.c_str());
  notify_scene_might_have_changed();
}

void RTApplication::on_material_changed(uint32_t index) {
  scene.create_area_emitters_from_materials();
  notify_scene_might_have_changed();
}

void RTApplication::on_medium_added() {
  scene.add_medium(nullptr);
  scene.update_medium_bounds();
  notify_scene_might_have_changed();
}

void RTApplication::on_medium_renamed(uint32_t index, const std::string& name) {
  scene.rename_medium(index, name.c_str());
  notify_scene_might_have_changed();
}

void RTApplication::on_medium_changed(uint32_t index) {
  scene.update_medium_bounds();
  notify_scene_might_have_changed();
}

void RTApplication::on_mesh_material_changed(uint32_t mesh_index, uint32_t material_index) {
  scene.set_mesh_material(mesh_index, material_index);
  scene.create_area_emitters_from_materials();
  notify_scene_might_have_changed();
}

void RTApplication::on_mesh_renamed(uint32_t index, const std::string& name) {
  scene.rename_mesh(index, name.c_str());
  notify_scene_might_have_changed();
}

void RTApplication::on_emitter_changed(uint32_t index) {
  bool cpu_was_running = cpu_renderer.is_running();
  bool atmosphere_related = false;
  if (index < scene.data().emitter_profiles.size()) {
    const auto& emitter = scene.data().emitter_profiles[index];
    if ((emitter.cls == EmitterProfile::Class::Environment) && ((emitter.meta & EmitterProfile::Meta::Atmosphere) != 0)) {
      atmosphere_related = true;
    } else if ((emitter.cls == EmitterProfile::Class::Directional) && (emitter.reference_emitter_index != kInvalidIndex) &&
               (emitter.reference_emitter_index < scene.data().emitter_profiles.size())) {
      const auto& referenced = scene.data().emitter_profiles[emitter.reference_emitter_index];
      atmosphere_related = (referenced.cls == EmitterProfile::Class::Environment) && ((referenced.meta & EmitterProfile::Meta::Atmosphere) != 0);
    }

    if (atmosphere_related) {
      if (cpu_was_running) {
        cpu_renderer.stop();
      }
      scene.rebuild_atmosphere_emitter(index);
    }
  }

  notify_scene_might_have_changed();
  if (atmosphere_related && cpu_was_running) {
    cpu_renderer.restart();
  }
}

void RTApplication::on_emitter_added(uint32_t type) {
  ETX_PROFILER_SCOPE();
  bool cpu_was_running = cpu_renderer.is_running();
  bool atmosphere_related = false;

  switch (type) {
    case 0: {
      scene.add_environment_emitter({1.0f, 1.0f, 1.0f}, kInvalidIndex);
      break;
    }
    case 1: {
      scene.add_directional_emitter({0.0f, 1.0f, 1.0f}, {1.0f, 1.0f, 1.0f}, 0.5422f, kInvalidIndex);
      break;
    }
    case 2: {
      atmosphere_related = true;
      if (cpu_was_running) {
        cpu_renderer.stop();
      }
      scene.add_atmosphere_emitter({
        .scattering = {.altitude = 1000.0f, .anisotropy = 0.825f, .rayleigh_scale = 1.0f, .mie_scale = 1.0f, .ozone_scale = 1.0f},
        .quality = 0.125f,
      });
      break;
    }
  }

  notify_scene_might_have_changed();
  if (atmosphere_related && cpu_was_running) {
    cpu_renderer.restart();
  }
}

void RTApplication::on_camera_changed(uint2 viewport, uint32_t pixel_size) {
  ETX_PROFILER_SCOPE();

  scene.update_active_camera();
  if ((viewport != film.base_dimensions()) || (pixel_size != film.pixel_size())) {
    cpu_renderer.set_output_dimensions(render_context.get_context(), scene.camera().film_size);
  }
  cpu_renderer.restart();
}

void RTApplication::on_scene_settings_changed() {
  notify_scene_might_have_changed();
}

void RTApplication::on_denoise_selected() {
  ETX_PROFILER_SCOPE();
  cpu_renderer.film().denoise(ui.view_options().view_layer, cpu_renderer.scene().options.radiance_clamp);
  ui.mutable_view_options().view_layer = ViewLayer::Denoised;
}

void RTApplication::update_camera_to_fit_scene(const float3& view_direction) {
  ETX_PROFILER_SCOPE();
  float3 position = {};
  float3 target = {};
  compute_camera_position_to_fit_scene(scene.data(), scene.camera(), view_direction, position, target);
  if (_active_renderer && _active_renderer->camera_controller()) {
    _active_renderer->camera_controller()->schedule(position, target);
  }
}

void RTApplication::on_view_scene(uint32_t direction) {
  ETX_PROFILER_SCOPE();
  constexpr float3 directions[] = {
    {1.0f, 1.0f, 1.0f},
    kWorldRight,
    -kWorldRight,
    kWorldUp,
    -kWorldUp,
    -kWorldForward,
    kWorldForward,
  };
  direction = clamp(direction, 0u, uint32_t(sizeof(directions) / sizeof(directions[0])));
  update_camera_to_fit_scene(directions[direction]);
}

void RTApplication::on_clear_recent_files() {
  ETX_PROFILER_SCOPE();
  _recent_files.clear();
  add_to_recent(_current_scene_file);
  save_options();
}

void RTApplication::on_camera_activated(uint32_t camera_index) {
  ETX_PROFILER_SCOPE();
  if (camera_index >= (uint32_t)scene.data().cameras.size()) {
    return;
  }

  for (auto& cam : scene.data().cameras) {
    cam.active = false;
  }

  scene.data().cameras[camera_index].active = true;

  scene.update_active_camera();

  cpu_renderer.restart();
}

void RTApplication::notify_scene_might_have_changed() {
  cpu_renderer.on_scene_changed(scene);
  raster_renderer.on_scene_changed(scene);
  gpu_renderer.on_scene_changed(scene);
}

void RTApplication::on_scene_update_requested() {
  ETX_PROFILER_SCOPE();
  notify_scene_might_have_changed();
}

void RTApplication::on_reload_shaders_selected() {
  ETX_PROFILER_SCOPE();
  if (_active_renderer == &gpu_renderer) {
    gpu_renderer.reload_shaders(render_context.get_context());
  }
}

}  // namespace etx
