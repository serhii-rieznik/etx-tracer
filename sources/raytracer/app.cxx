#include <etx/core/core.hxx>
#include <etx/core/environment.hxx>
#include <etx/core/profiler.hxx>

#include <etx/render/shared/camera.hxx>
#include <etx/rt/integrators/integrator.hxx>

#include "app.hxx"

#include <tinyexr.hxx>
#include <stb_image_write.hxx>

#include <algorithm>
#include <cstring>
#include <filesystem>

#if defined(ETX_PLATFORM_WINDOWS)
# define WIN32_LEAN_AND_MEAN 1
# include <Windows.h>
#endif

#include <shaders/shared/render_options.hxx>

namespace etx {

RTApplication::RTApplication()
  : _ior_database()
  , scene(scheduler, _ior_database)
  , render(scheduler)
  , cpu_renderer(scheduler, scene)
  , raster_renderer(scheduler)
  , gpu_renderer(scheduler) {
  _active_renderer = &cpu_renderer;
  ui.set_current_renderer_mode(RendererMode::CPURaytracing);
}

RTApplication::~RTApplication() {
  save_options();
}

void RTApplication::init() {
  render.init();
  std::string ior_folder = env().file_in_data("./spectrum/");
  _ior_database.load(ior_folder.c_str());

  ui.initialize(nullptr, &_ior_database, render.get_context());
  ui.set_integrator_list(cpu_renderer.integrator_list(), cpu_renderer.integrator_count());

  cpu_renderer.init(render, scene);
  raster_renderer.init(render, scene);
  gpu_renderer.init(render, scene);

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
  ui.callbacks.emitter_rebuild = std::bind(&RTApplication::on_emitter_rebuild, this, std::placeholders::_1);
  ui.callbacks.camera_changed = std::bind(&RTApplication::on_camera_changed, this, std::placeholders::_1);
  ui.callbacks.scene_settings_changed = std::bind(&RTApplication::on_scene_settings_changed, this);
  ui.callbacks.denoise_selected = std::bind(&RTApplication::on_denoise_selected, this);
  ui.callbacks.view_scene = std::bind(&RTApplication::on_view_scene, this, std::placeholders::_1);
  ui.callbacks.clear_recent_files = std::bind(&RTApplication::on_clear_recent_files, this);
  ui.callbacks.camera_activated = std::bind(&RTApplication::on_camera_activated, this, std::placeholders::_1);
  ui.callbacks.scene_updates_locked_changed = std::bind(&RTApplication::on_scene_updates_locked_changed, this, std::placeholders::_1);
  ui.callbacks.integrator_selected = std::bind(&RTApplication::on_integrator_selected, this, std::placeholders::_1);

  _options.load_from_file(env().file_in_data("options.json"));

  for (uint32_t i = 0; i < 7; ++i) {
    const auto name = "recent-" + std::to_string(i);
    if (_options.has(name, Option::Class::String)) {
      _recent_files.emplace_back(_options.get_string(name, {}));
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

  auto selected_integrator = _options.get_string("integrator", std::string{});
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

  cpu_renderer.set_integrator(integrator);
  ui.set_current_integrator(integrator);

  _current_scene_file = _options.get_string("scene", std::string{});
  if (_current_scene_file.empty() == false) {
    on_scene_file_selected(_current_scene_file);
  }

  const auto& ref = _options.get_string("ref", std::string{});
  if (ref.empty() == false) {
    on_referenece_image_selected(ref);
  }

  save_options();
}

void RTApplication::save_options() {
  constexpr uint32_t kRecentLimit = 8u;
  for (uint32_t idx = 0; idx < kRecentLimit; ++idx) {
    _options.remove("recent-" + std::to_string(idx));
  }
  uint32_t i = 0;
  for (const auto& recent : _recent_files) {
    _options.set_string("recent-" + std::to_string(i++), env().to_project_relative(recent), "Recent File");
  }
  _options.save_to_file(env().file_in_data("options.json"));
}

void RTApplication::set_renderer_mode(RendererMode mode) {
  Renderer* next_renderer = nullptr;
  switch (mode) {
    case RendererMode::Rasterization:
      next_renderer = &raster_renderer;
      break;

    case RendererMode::GPURaytracing:
      next_renderer = &gpu_renderer;
      break;

    default:
      next_renderer = &cpu_renderer;
      break;
  }

  if (next_renderer == _active_renderer)
    return;

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

  auto dt = time_measure.lap();

  render.begin_frame();

  if (_active_renderer != nullptr) {
    _active_renderer->frame(render, scene, dt);
  }

  uint32_t iteration = (_active_renderer != nullptr && _active_renderer->mode() == RendererMode::CPURaytracing) ? cpu_renderer.integrator_thread().status().current_iteration : 0;

  render.start_frame(iteration, ui.view_options());
  ui.build(dt, _recent_files, scene, scene.mutable_camera(), scene.material_mapping(), scene.medium_mapping(), scene.mesh_mapping(), scene.camera_mapping(),
    (_active_renderer != nullptr && _active_renderer->mode() == RendererMode::CPURaytracing) ? &cpu_renderer.integrator_thread() : nullptr, render.get_context());
  render.end_frame();
}

void RTApplication::cleanup() {
  render.cleanup([this]() {
    ui.cleanup();
  });
}

void RTApplication::process_event(const sapp_event* e) {
  ETX_PROFILER_SCOPE();
  if (ui.handle_event(e) == false && _active_renderer != nullptr) {
    _active_renderer->process_event(e);
  }
}

void RTApplication::add_to_recent(const std::string& value) {
  constexpr uint32_t kMaxRecentFiles = 8u;

  auto relative_path = env().to_project_relative(value);
  auto absolute_path = env().resolve_to_absolute(value);

  auto e = std::remove_if(_recent_files.begin(), _recent_files.end(), [&](const std::string& entry) {
    return env().resolve_to_absolute(entry) == absolute_path;
  });
  _recent_files.erase(e, _recent_files.end());

  if (value.empty() == false) {
    _recent_files.emplace_back(relative_path);
  }

  if (_recent_files.size() > kMaxRecentFiles) {
    _recent_files.erase(_recent_files.begin());
  }
}

void RTApplication::load_scene_file(const std::string& file_name, uint32_t options, bool start_rendering) {
  _current_scene_file = env().resolve_to_absolute(file_name);

  cpu_renderer.stop();
  _options.set_string("scene", env().to_project_relative(_current_scene_file), "Scene");
  save_options();

  log::warning("Loading scene %s...", _current_scene_file.c_str());
  SceneRepresentation::IntegratorData integrator_data;
  if (scene.load_from_file(_current_scene_file.c_str(), options, &integrator_data) == false) {
    log::error("Failed to load scene from file: %s", _current_scene_file.c_str());
  }
  log::warning("Setting output dimensions...");
  render.set_output_dimensions(scene.camera().film_size);

  if (scene.valid() == false) {
    return;
  }

  cpu_renderer.integrator_thread().reset_scene_hashes();

  for (const auto& [type, options] : integrator_data.settings) {
    Integrator* integrator = integrator_type_to_instance(type, cpu_renderer.integrator_list(), cpu_renderer.integrator_count());
    if (integrator != nullptr) {
      integrator->sync_from_options(options);
      integrator->update_options();
    }
  }

  Integrator* integrator = nullptr;
  if (integrator_data.selected != Integrator::Type::Invalid) {
    integrator = integrator_type_to_instance(integrator_data.selected, cpu_renderer.integrator_list(), cpu_renderer.integrator_count());
  }

  if (integrator == nullptr && cpu_renderer.integrator_count() > 0) {
    integrator = cpu_renderer.integrator_list()[1];  // pt
  }

  cpu_renderer.set_integrator(integrator);
  ui.set_current_integrator(integrator);

  add_to_recent(_current_scene_file);

  cpu_renderer.film().clear(Film::ClearEverything);
  cpu_renderer.start();
}

std::string RTApplication::save_scene_file(const std::string& file_name) {
  log::info("Saving %s..", file_name.c_str());
  Integrator* current = cpu_renderer.current_integrator();
  Integrator::Type selected_type = integrator_to_type(current);
  std::string saved_path = scene.save_to_file(file_name.c_str(), selected_type, cpu_renderer.integrator_list(), cpu_renderer.integrator_count());
  if (saved_path.empty()) {
    log::error("Failed to save scene to %s", file_name.c_str());
    return {};
  }

  _current_scene_file = saved_path;
  _options.set_string("scene", _current_scene_file, "Scene");
  add_to_recent(saved_path);
  save_options();

  return saved_path;
}

void RTApplication::on_referenece_image_selected(std::string file_name) {
  log::warning("Loading reference image %s...", file_name.c_str());

  _options.set_string("ref", file_name, "Reference");
  save_options();

  render.set_reference_image(file_name.c_str());
}

void RTApplication::on_use_image_as_reference() {
  _options.set_string("ref", {}, "Reference");
  save_options();

  const float4* data = cpu_renderer.film().layer(ViewLayer::Result, cpu_renderer.scene());
  uint2 size = cpu_renderer.film().dimensions();
  render.set_reference_image(data, size);
}

void RTApplication::on_save_image_selected(std::string file_name, SaveImageMode mode) {
  uint2 image_size = {scene.camera().film_size.x, scene.camera().film_size.y};
  const float4* output = cpu_renderer.film().layer(ui.view_options().view_layer, cpu_renderer.scene());

  if (mode == SaveImageMode::TonemappedLDR) {
    if (strlen(get_file_ext(file_name.c_str())) == 0) {
      file_name += ".png";
    }
    float exposure = ui.view_options().exposure;
    std::vector<ubyte4> tonemapped(image_size.x * image_size.y);
    for (uint32_t i = 0, e = image_size.x * image_size.y; i < e; ++i) {
      float3 tm = {
        1.0f - expf(-exposure * output[i].x),
        1.0f - expf(-exposure * output[i].y),
        1.0f - expf(-exposure * output[i].z),
      };
      float3 gamma = linear_to_gamma(tm);
      tonemapped[i].x = static_cast<uint8_t>(255.0f * saturate(gamma.x));
      tonemapped[i].y = static_cast<uint8_t>(255.0f * saturate(gamma.y));
      tonemapped[i].z = static_cast<uint8_t>(255.0f * saturate(gamma.z));
      tonemapped[i].w = 255u;
    }
    if (stbi_write_png(file_name.c_str(), image_size.x, image_size.y, 4, tonemapped.data(), 0) != 1) {
      log::error("Failed to save PNG image to %s", file_name.c_str());
    }
  } else {
    if (strlen(get_file_ext(file_name.c_str())) == 0) {
      file_name += ".exr";
    }
    const char* error = nullptr;
    if (SaveEXR(reinterpret_cast<const float*>(output), image_size.x, image_size.y, 4, false, file_name.c_str(), &error) != TINYEXR_SUCCESS) {
      log::error("Failed to save EXR image to %s: %s", file_name.c_str(), error);
    }
  }
}

void RTApplication::on_scene_file_selected(std::string file_name) {
  load_scene_file(file_name, SceneRepresentation::LoadEverything, false);
}

void RTApplication::on_save_scene_file_selected(std::string file_name) {
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

  base = env().to_project_relative(base);
  std::string saved_path = save_scene_file(base);
  if (saved_path.empty() == false) {
    log::info("Scene saved to %s", saved_path.c_str());
  }
}

void RTApplication::on_save_scene_file_as_selected() {
  std::string selected_file = save_file("json");
  if (selected_file.empty() == false) {
    on_save_scene_file_selected(selected_file);
  }
}

void RTApplication::on_integrator_selected(Integrator::Type itype) {
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
  if (ui.view_options().view_layer == ViewLayer::Denoised) {
    ui.mutable_view_options().view_layer = ViewLayer::Result;
  }
  cpu_renderer.film().clear(Film::ClearEverything);
  cpu_renderer.start();
}

void RTApplication::on_stop_selected(bool wait_for_completion) {
  cpu_renderer.stop();
}

void RTApplication::on_restart_selected() {
  on_stop_selected(false);
  on_run_selected();
}

void RTApplication::on_reload_scene_selected() {
  if (_current_scene_file.empty() == false) {
    load_scene_file(_current_scene_file, SceneRepresentation::LoadEverything, cpu_renderer.is_running());
  }
}

void RTApplication::on_reload_geometry_selected() {
  if (_current_scene_file.empty() == false) {
    load_scene_file(_current_scene_file, SceneRepresentation::LoadGeometry, cpu_renderer.is_running());
  }
}

void RTApplication::on_options_changed() {
  cpu_renderer.restart();
}

void RTApplication::on_material_added() {
  scene.add_material(nullptr);
}

void RTApplication::on_material_renamed(uint32_t index, const std::string& name) {
  scene.rename_material(index, name.c_str());
}

void RTApplication::on_material_changed(uint32_t index) {
  scene.create_area_emitters_from_materials();
}

void RTApplication::on_medium_added() {
  scene.add_medium(nullptr);
  scene.update_medium_bounds();
}

void RTApplication::on_medium_renamed(uint32_t index, const std::string& name) {
  scene.rename_medium(index, name.c_str());
}

void RTApplication::on_medium_changed(uint32_t index) {
  scene.update_medium_bounds();
}

void RTApplication::on_mesh_material_changed(uint32_t mesh_index, uint32_t material_index) {
  scene.set_mesh_material(mesh_index, material_index);
  scene.create_area_emitters_from_materials();
}

void RTApplication::on_mesh_renamed(uint32_t index, const std::string& name) {
  scene.rename_mesh(index, name.c_str());
}

void RTApplication::on_emitter_changed(uint32_t index) {
}

void RTApplication::on_emitter_added(uint32_t type) {
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
      scene.add_atmosphere_emitter({
        .scattering = {.altitude = 1000.0f, .anisotropy = 0.825f, .rayleigh_scale = 1.0f, .mie_scale = 1.0f, .ozone_scale = 1.0f},
        .quality = 0.125f,
      });
      break;
    }
  }
}

void RTApplication::on_emitter_rebuild(uint32_t index) {
  scene.rebuild_atmosphere_emitter(index);
}

void RTApplication::on_camera_changed(bool film_changed) {
  scene.update_active_camera();

  if (film_changed) {
    cpu_renderer.stop();
    render.set_output_dimensions(scene.camera().film_size);
  }
  cpu_renderer.restart();
}

void RTApplication::on_scene_settings_changed() {
}

void RTApplication::on_denoise_selected() {
  cpu_renderer.film().denoise(ui.view_options().view_layer, cpu_renderer.scene());
  ui.mutable_view_options().view_layer = ViewLayer::Denoised;
}

void RTApplication::update_camera_to_fit_scene(const float3& view_direction) {
  float3 position = {};
  float3 target = {};
  compute_camera_position_to_fit_scene(scene.data(), scene.camera(), view_direction, position, target);
  if (_active_renderer && _active_renderer->camera_controller()) {
    _active_renderer->camera_controller()->schedule(position, target);
  }
}

void RTApplication::on_view_scene(uint32_t direction) {
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
  _recent_files.clear();
  add_to_recent(_current_scene_file);
  save_options();
}

void RTApplication::on_camera_activated(uint32_t camera_index) {
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

void RTApplication::on_scene_updates_locked_changed(bool locked) {
  cpu_renderer.integrator_thread().set_scene_updates_locked(locked);
}

}  // namespace etx
