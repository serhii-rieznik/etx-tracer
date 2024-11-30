#include <etx/core/environment.hxx>
#include <etx/core/profiler.hxx>

#include <etx/render/host/scene_saver.hxx>

#include "app.hxx"

#include <tinyexr.hxx>
#include <stb_image_write.hxx>

#if defined(ETX_PLATFORM_WINDOWS)

// TODO : fix hacks
# define WIN32_LEAN_AND_MEAN 1
# include <Windows.h>

#endif

namespace etx {

RTApplication::RTApplication()
  : render(raytracing.scheduler())
  , scene(raytracing.scheduler())
  , camera_controller(scene.camera())
  , integrator_thread(raytracing.scheduler(), IntegratorThread::Mode::ExternalControl) {
}

void RTApplication::init() {
  render.init();
  ui.initialize();
  ui.set_integrator_list(_integrator_array, std::size(_integrator_array));
  ui.set_film(&raytracing.film());

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
  ui.callbacks.medium_changed = std::bind(&RTApplication::on_medium_changed, this, std::placeholders::_1);
  ui.callbacks.emitter_changed = std::bind(&RTApplication::on_emitter_changed, this, std::placeholders::_1);
  ui.callbacks.camera_changed = std::bind(&RTApplication::on_camera_changed, this);
  ui.callbacks.scene_settings_changed = std::bind(&RTApplication::on_scene_settings_changed, this);
  ui.callbacks.denoise_selected = std::bind(&RTApplication::on_denoise_selected, this);
  ui.callbacks.view_scene = std::bind(&RTApplication::on_view_scene, this);

  _options.load_from_file(env().file_in_data("options.json"));
  if (_options.has("integrator") == false) {
    _options.add("integrator", "none");
  }
  if (_options.has("scene") == false) {
    _options.add("scene", "none");
  }
  if (_options.has("ref") == false) {
    _options.add("ref", "none");
  }

#if defined(ETX_PLATFORM_WINDOWS)
  if (GetAsyncKeyState(VK_ESCAPE)) {
    _options.set("integrator", std::string());
  }
  if (GetAsyncKeyState(VK_ESCAPE) && GetAsyncKeyState(VK_SHIFT)) {
    _options.set("scene", std::string());
  }
#endif

  Integrator* integrator = nullptr;

  auto selected_integrator = _options.get("integrator", std::string{}).name;
  for (uint64_t i = 0; (selected_integrator.empty() == false) && (i < std::size(_integrator_array)); ++i) {
    ETX_ASSERT(_integrator_array[i] != nullptr);
    if (selected_integrator == _integrator_array[i]->name()) {
      integrator = _integrator_array[i];
    }
  }

  integrator_thread.start(integrator);
  ui.set_current_integrator(integrator);

  _current_scene_file = _options.get("scene", std::string{}).name;
  if (_current_scene_file.empty() == false) {
    on_scene_file_selected(_current_scene_file);
  }

  auto ref = _options.get("ref", std::string{}).name;
  if (ref.empty() == false) {
    on_referenece_image_selected(ref);
  }

  save_options();
  ETX_PROFILER_RESET_COUNTERS();
}

void RTApplication::save_options() {
  _options.save_to_file(env().file_in_data("options.json"));
}

void RTApplication::frame() {
  ETX_FUNCTION_SCOPE();
  auto dt = time_measure.lap();

  // if (_current_integrator != nullptr) {
  //   _current_integrator->update();
  //   // TODO : lock
  // }

  bool can_change_camera = true;  // _current_integrator->state() == Integrator::State::Preview;
  if (can_change_camera && camera_controller.update(dt)) {
    integrator_thread.restart();
    // _current_integrator->run(ui.integrator_options());
  }

  auto options = ui.view_options();
  if (options.layer == Film::Normals) {
    options.options = ViewOptions::SkipColorConversion;
  }

  integrator_thread.update();

  render.start_frame(integrator_thread.status().current_iteration, options);
  render.update_image(raytracing.film().layer(options.layer));
  ui.build(dt);
  render.end_frame();
}

void RTApplication::cleanup() {
  render.cleanup();
  ui.cleanup();
}

void RTApplication::process_event(const sapp_event* e) {
  if (ui.handle_event(e) || (raytracing.has_scene() == false)) {
    return;
  }
  camera_controller.handle_event(e);
}

void RTApplication::load_scene_file(const std::string& file_name, uint32_t options, bool start_rendering) {
  _current_scene_file = file_name;

  log::warning("Loading scene %s...", _current_scene_file.c_str());

  integrator_thread.stop(Integrator::Stop::Immediate);

  _options.set("scene", _current_scene_file);
  save_options();

  if (scene.load_from_file(_current_scene_file.c_str(), options) == false) {
    ui.set_scene(nullptr, {}, {});
    log::error("Failed to load scene from file: %s", _current_scene_file.c_str());
    return;
  }

  raytracing.set_scene(scene.scene());
  ui.set_scene(scene.mutable_scene_pointer(), scene.material_mapping(), scene.medium_mapping());
  render.set_output_dimensions(scene.scene().camera.image_size);

  if (scene.valid() == false) {
    return;
  }

  integrator_thread.run(ui.integrator_options());
}

void RTApplication::save_scene_file(const std::string& file_name) const {
  log::info("Saving %s..", file_name.c_str());
  save_scene_to_file(scene, file_name.c_str());
}

void RTApplication::on_referenece_image_selected(std::string file_name) {
  log::warning("Loading reference image %s...", file_name.c_str());

  _options.set("ref", file_name);
  save_options();

  render.set_reference_image(file_name.c_str());
}

void RTApplication::on_use_image_as_reference() {
  _options.set("ref", std::string());
  save_options();

  const float4* data = raytracing.film().combined_result();
  uint2 size = raytracing.film().dimensions();
  render.set_reference_image(data, size);
}

void RTApplication::on_save_image_selected(std::string file_name, SaveImageMode mode) {
  uint2 image_size = {raytracing.scene().camera.image_size.x, raytracing.scene().camera.image_size.y};
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
  if (strlen(get_file_ext(file_name.c_str())) == 0) {
    file_name += ".json";
  }
  save_scene_file(file_name);
}

void RTApplication::on_integrator_selected(Integrator* i) {
  _options.set("integrator", i->name());
  save_options();

  integrator_thread.set_integrator(i);
  raytracing.film().clear();

  if (scene.valid()) {
    integrator_thread.run(ui.integrator_options());
  }
}

void RTApplication::on_run_selected() {
  if (ui.view_options().layer == Film::Denoised) {
    ui.mutable_view_options().layer = Film::Result;
  }
  raytracing.film().clear();
  integrator_thread.run(ui.integrator_options());
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
  integrator_thread.restart(ui.integrator_options());
}

void RTApplication::on_material_changed(uint32_t index) {
  // TODO : re-upload to GPU
  integrator_thread.restart();
}

void RTApplication::on_medium_changed(uint32_t index) {
  // TODO : re-upload to GPU
  integrator_thread.restart();
}

void RTApplication::on_emitter_changed(uint32_t index) {
  // TODO : re-upload to GPU
  integrator_thread.stop(Integrator::Stop::Immediate);

  build_emitters_distribution(scene.mutable_scene());
  integrator_thread.run(ui.integrator_options());
}

void RTApplication::on_camera_changed() {
  integrator_thread.restart();
}

void RTApplication::on_scene_settings_changed() {
  integrator_thread.restart();
}

void RTApplication::on_denoise_selected() {
  raytracing.film().denoise();

  if (ui.view_options().layer == Film::Result) {
    ui.mutable_view_options().layer = Film::Denoised;
  }
}

void RTApplication::on_view_scene() {
  const float3 position = 2.0f * scene.scene().bounding_sphere_radius * float3{1.0f, 1.0f, 1.0f};
  const auto view_center = scene.scene().bounding_sphere_center;
  camera_controller.schedule(position, view_center);
}

}  // namespace etx
