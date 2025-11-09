#include <etx/core/environment.hxx>
#include <etx/core/environment.hxx>
#include <etx/core/profiler.hxx>

#include <etx/render/shared/camera.hxx>

#include "app.hxx"

#include <tinyexr.hxx>
#include <stb_image_write.hxx>

#include <algorithm>
#include <cstring>

#if defined(ETX_PLATFORM_WINDOWS)

// TODO : fix hacks
# define WIN32_LEAN_AND_MEAN 1
# include <Windows.h>

#endif

namespace etx {

RTApplication::RTApplication()
  : render(raytracing.scheduler())
  , _ior_database()
  , scene(raytracing.scheduler(), _ior_database)
  , camera_controller(scene.camera())
  , integrator_thread(raytracing.scheduler(), IntegratorThread::Mode::ExternalControl) {
  raytracing.link_scene(scene.scene());
  raytracing.link_camera(scene.camera());
}

RTApplication::~RTApplication() {
  save_options();
}

void RTApplication::init() {
  render.init();
  std::string ior_folder = env().file_in_data("spectrum/");
  _ior_database.load(ior_folder.c_str());

  ui.initialize(&raytracing.film(), &_ior_database);
  ui.set_integrator_list(_integrator_array, std::size(_integrator_array));

  ui.callbacks.reference_image_selected = std::bind(&RTApplication::on_referenece_image_selected, this, std::placeholders::_1);
  ui.callbacks.save_image_selected = std::bind(&RTApplication::on_save_image_selected, this, std::placeholders::_1, std::placeholders::_2);
  ui.callbacks.scene_file_selected = std::bind(&RTApplication::on_scene_file_selected, this, std::placeholders::_1);
  ui.callbacks.save_scene_file_selected = std::bind(&RTApplication::on_save_scene_file_selected, this, std::placeholders::_1);
  ui.callbacks.integrator_selected = std::bind(&RTApplication::on_integrator_selected, this, std::placeholders::_1);
  ui.callbacks.run_selected = std::bind(&RTApplication::on_run_selected, this);
  ui.callbacks.stop_selected = std::bind(&RTApplication::on_stop_selected, this, std::placeholders::_1);
  ui.callbacks.restart_selected = std::bind(&RTApplication::on_restart_selected, this);
  ui.callbacks.reload_scene_selected = std::bind(&RTApplication::on_reload_scene_selected, this);
  ui.callbacks.reload_geometry_selected = std::bind(&RTApplication::on_reload_geometry_selected, this);
  ui.callbacks.options_changed = std::bind(&RTApplication::on_options_changed, this);
  ui.callbacks.use_image_as_reference = std::bind(&RTApplication::on_use_image_as_reference, this);
  ui.callbacks.material_changed = std::bind(&RTApplication::on_material_changed, this, std::placeholders::_1);
  ui.callbacks.medium_added = std::bind(&RTApplication::on_medium_added, this);
  ui.callbacks.medium_changed = std::bind(&RTApplication::on_medium_changed, this, std::placeholders::_1);
  ui.callbacks.emitter_changed = std::bind(&RTApplication::on_emitter_changed, this, std::placeholders::_1);
  ui.callbacks.camera_changed = std::bind(&RTApplication::on_camera_changed, this, std::placeholders::_1);
  ui.callbacks.scene_settings_changed = std::bind(&RTApplication::on_scene_settings_changed, this);
  ui.callbacks.denoise_selected = std::bind(&RTApplication::on_denoise_selected, this);
  ui.callbacks.view_scene = std::bind(&RTApplication::on_view_scene, this, std::placeholders::_1);

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
  for (uint64_t i = 0; (selected_integrator.empty() == false) && (i < std::size(_integrator_array)); ++i) {
    ETX_ASSERT(_integrator_array[i] != nullptr);
    if (selected_integrator == _integrator_array[i]->name()) {
      integrator = _integrator_array[i];
    }
  }

  integrator_thread.start(integrator);
  ui.set_current_integrator(integrator);

  _current_scene_file = _options.get_string("scene", std::string{});
  if (_current_scene_file.empty() == false) {
    on_scene_file_selected(_current_scene_file);
  }

  auto ref = _options.get_string("ref", std::string{});
  if (ref.empty() == false) {
    on_referenece_image_selected(ref);
  }

  save_options();
}

void RTApplication::save_options() {
  uint32_t i = 0;
  for (const auto& recent : _recent_files) {
    _options.set_string("recent-" + std::to_string(i++), recent, "Recent File");
  }
  _options.save_to_file(env().file_in_data("options.json"));
}

void RTApplication::frame() {
  ETX_PROFILER_SCOPE();

  auto dt = time_measure.lap();

  bool can_change_camera = scene.valid();
  if (can_change_camera) {
    bool camera_controller_state = camera_controller.update(dt);
    if (camera_controller_state) {
      raytracing.film().set_pixel_size(8u);
      integrator_thread.restart();
    } else if (camera_controller_state != last_camera_controller_state) {
      raytracing.film().set_pixel_size(1u);
      integrator_thread.restart();
    }
    last_camera_controller_state = camera_controller_state;
  }

  auto options = ui.view_options();
  if (options.layer == Film::Normals) {
    options.options = ViewOptions::SkipColorConversion;
  }

  ETX_PROFILER_NAMED_SCOPE("process");
  integrator_thread.update();

  render.start_frame(integrator_thread.status().current_iteration, options);

  const auto frame_data = raytracing.film().layer(options.layer);
  render.update_image(frame_data);

  ui.build(dt, _recent_files, scene.mutable_scene(), scene.mutable_camera(), scene.material_mapping(), scene.medium_mapping());
  render.end_frame();
}

void RTApplication::cleanup() {
  render.cleanup();
  ui.cleanup();
}

void RTApplication::process_event(const sapp_event* e) {
  ETX_PROFILER_SCOPE();
  if (ui.handle_event(e) == false) {
    camera_controller.handle_event(e);
  }
}

void RTApplication::load_scene_file(const std::string& file_name, uint32_t options, bool start_rendering) {
  _current_scene_file = file_name;

  integrator_thread.stop(Integrator::Stop::Immediate);
  _options.set_string("scene", _current_scene_file, "Scene");
  save_options();

  log::warning("Loading scene %s...", _current_scene_file.c_str());
  if (scene.load_from_file(_current_scene_file.c_str(), options) == false) {
    log::error("Failed to load scene from file: %s", _current_scene_file.c_str());
  }
  log::warning("Committing changes...");
  raytracing.commit_changes();
  render.set_output_dimensions(raytracing.film().dimensions());

  if (scene.valid() == false) {
    return;
  }

  auto existing = std::find(_recent_files.begin(), _recent_files.end(), file_name);
  if (existing != _recent_files.end()) {
    _recent_files.erase(existing);
  }

  _recent_files.emplace_back(file_name);

  if (_recent_files.size() > 8) {
    _recent_files.erase(_recent_files.begin());
  }

  raytracing.film().clear(Film::ClearEverything);
  integrator_thread.run();
}

std::string RTApplication::save_scene_file(const std::string& file_name) {
  log::info("Saving %s..", file_name.c_str());
  std::string saved_path = scene.save_to_file(file_name.c_str());
  if (saved_path.empty()) {
    log::error("Failed to save scene to %s", file_name.c_str());
    return {};
  }

  _current_scene_file = saved_path;

  auto existing = std::find(_recent_files.begin(), _recent_files.end(), saved_path);
  if (existing != _recent_files.end()) {
    _recent_files.erase(existing);
  }
  _recent_files.emplace_back(saved_path);
  if (_recent_files.size() > 8) {
    _recent_files.erase(_recent_files.begin());
  }

  _options.set_string("scene", _current_scene_file, "Scene");
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

  const float4* data = raytracing.film().layer(Film::Result);
  uint2 size = raytracing.film().dimensions();
  render.set_reference_image(data, size);
}

void RTApplication::on_save_image_selected(std::string file_name, SaveImageMode mode) {
  uint2 image_size = {raytracing.camera().film_size.x, raytracing.camera().film_size.y};
  const float4* output = raytracing.film().layer(ui.view_options().layer);

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

  std::string saved_path = save_scene_file(base);
  if (saved_path.empty() == false) {
    log::info("Scene saved to %s", saved_path.c_str());
  }
}

void RTApplication::on_integrator_selected(Integrator* i) {
  _options.set_string("integrator", i->name(), "Integrator");
  save_options();

  integrator_thread.set_integrator(i);
  raytracing.film().clear(Film::ClearEverything);

  if (scene.valid()) {
    integrator_thread.run();
  }
}

void RTApplication::on_run_selected() {
  if (ui.view_options().layer == Film::Denoised) {
    ui.mutable_view_options().layer = Film::Result;
  }
  raytracing.film().clear(Film::ClearEverything);
  integrator_thread.run();
}

void RTApplication::on_stop_selected(bool wait_for_completion) {
  integrator_thread.stop(wait_for_completion ? Integrator::Stop::WaitForCompletion : Integrator::Stop::Immediate);
}

void RTApplication::on_restart_selected() {
  on_stop_selected(false);
  on_run_selected();
}

void RTApplication::on_reload_scene_selected() {
  if (_current_scene_file.empty() == false) {
    load_scene_file(_current_scene_file, SceneRepresentation::LoadEverything, integrator_thread.running());
  }
}

void RTApplication::on_reload_geometry_selected() {
  if (_current_scene_file.empty() == false) {
    load_scene_file(_current_scene_file, SceneRepresentation::LoadGeometry, integrator_thread.running());
  }
}

void RTApplication::on_options_changed() {
  integrator_thread.restart();
}

void RTApplication::on_material_changed(uint32_t index) {
  integrator_thread.stop(Integrator::Stop::Immediate);
  scene.rebuild_area_emitters();
  integrator_thread.restart();
}

void RTApplication::on_medium_added() {
  integrator_thread.stop(Integrator::Stop::Immediate);
  scene.add_medium(nullptr);
  integrator_thread.restart();
}

void RTApplication::on_medium_changed(uint32_t index) {
  integrator_thread.restart();
}

void RTApplication::on_emitter_changed(uint32_t index) {
  integrator_thread.stop(Integrator::Stop::Immediate);
  scene.rebuild_area_emitters();
  integrator_thread.restart();
}

void RTApplication::on_camera_changed(bool film_changed) {
  if (film_changed) {
    integrator_thread.stop(Integrator::Stop::Immediate);
    raytracing.commit_changes();
    render.set_output_dimensions(raytracing.film().dimensions());
    integrator_thread.restart();
  } else {
    integrator_thread.restart();
  }
}

void RTApplication::on_scene_settings_changed() {
  integrator_thread.restart();
}

void RTApplication::on_denoise_selected() {
  raytracing.film().denoise(ui.view_options().layer);
  ui.mutable_view_options().layer = Film::Denoised;
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

  const float3 position = 3.0f * scene.scene().bounding_sphere_radius * normalize(directions[direction]);
  const auto view_center = scene.scene().bounding_sphere_center;
  camera_controller.schedule(position, view_center);
}

}  // namespace etx
