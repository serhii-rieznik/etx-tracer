#include <etx/core/environment.hxx>
#include <etx/log/log.hxx>
#include <etx/render/host/image_pool.hxx>
#include <etx/render/shared/scene.hxx>

#include "app.hxx"

#include <tinyexr/tinyexr.hxx>

namespace etx {

RTApplication::RTApplication()
  : camera_controller(scene.camera()) {
}

void RTApplication::init() {
  render.init();
  ui.initialize();
  ui.set_integrator_list(_integrator_array, std::size(_integrator_array));
  ui.callbacks.reference_image_selected = std::bind(&RTApplication::on_referenece_image_selected, this, std::placeholders::_1);
  ui.callbacks.save_image_selected = std::bind(&RTApplication::on_save_image_selected, this, std::placeholders::_1, std::placeholders::_2);
  ui.callbacks.scene_file_selected = std::bind(&RTApplication::on_scene_file_selected, this, std::placeholders::_1);
  ui.callbacks.integrator_selected = std::bind(&RTApplication::on_integrator_selected, this, std::placeholders::_1);
  ui.callbacks.preview_selected = std::bind(&RTApplication::on_preview_selected, this);
  ui.callbacks.run_selected = std::bind(&RTApplication::on_run_selected, this);
  ui.callbacks.stop_selected = std::bind(&RTApplication::on_stop_selected, this, std::placeholders::_1);
  ui.callbacks.reload_scene_selected = std::bind(&RTApplication::on_reload_scene_selected, this);
  ui.callbacks.reload_geometry_selected = std::bind(&RTApplication::on_reload_geometry_selected, this);
  ui.callbacks.options_changed = std::bind(&RTApplication::on_options_changed, this);

  _options.load_from_file(env().file_in_data("options.json"));

  auto integrator = _options.get("integrator", std::string{}).name;
  for (uint64_t i = 0; (integrator.empty() == false) && (i < std::size(_integrator_array)); ++i) {
    if (integrator == _integrator_array[i]->name()) {
      _current_integrator = _integrator_array[i];
      break;
    }
  }

  ui.set_current_integrator(_current_integrator);

  _current_scene_file = _options.get("scene", std::string{}).name;
  if (_current_scene_file.empty() == false) {
    on_scene_file_selected(_current_scene_file);
  }

  auto ref = _options.get("ref", std::string{}).name;
  if (ref.empty() == false) {
    on_referenece_image_selected(ref);
  }
}

void RTApplication::save_options() {
  _options.save_to_file(env().file_in_data("options.json"));
}

void RTApplication::frame() {
  float4* c_image = nullptr;
  float4* l_image = nullptr;
  const char* status = "Not running";

  bool can_change_camera = true;

  if (_current_integrator != nullptr) {
    _current_integrator->update();
    status = _current_integrator->status();
    c_image = _current_integrator->get_camera_image(false);
    l_image = _current_integrator->get_light_image(false);
    can_change_camera = _current_integrator->state() == Integrator::State::Preview;
  }

  auto dt = time_measure.lap();
  if (can_change_camera) {
    if (camera_controller.update(dt)) {
      _current_integrator->preview(ui.integrator_options());
    }
  }

  render.set_view_options(ui.view_options());
  render.start_frame();
  render.update_output_images(c_image, l_image);
  ui.build(dt, status);
  render.end_frame();
}

void RTApplication::cleanup() {
  render.cleanup();
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
  if (_current_integrator) {
    _current_integrator->stop(false);
  }

  _options.set("scene", _current_scene_file);
  save_options();

  if (scene.load_from_file(_current_scene_file.c_str(), options) == false) {
    log::error("Failed to load scene from file: %s", _current_scene_file.c_str());
    return;
  }

  raytracing.set_scene(scene.scene());

  if (scene) {
    render.set_output_dimensions(scene.scene().camera.image_size);

    if (_current_integrator != nullptr) {
      if (start_rendering) {
        _current_integrator->run(ui.integrator_options());
      } else {
        _current_integrator->set_output_size(scene.scene().camera.image_size);
        _current_integrator->preview(ui.integrator_options());
      }
    }
  }
}

void RTApplication::on_referenece_image_selected(std::string file_name) {
  log::warning("Loading reference image %s...", file_name.c_str());

  _options.set("ref", file_name);
  save_options();

  render.set_reference_image(file_name.c_str());
}

void RTApplication::on_save_image_selected(std::string file_name, bool as_xyz) {
  if (_current_integrator == nullptr) {
    return;
  }

  if (strlen(get_file_ext(file_name.c_str())) == 0) {
    file_name += ".exr";
  }

  auto c_image = _current_integrator->get_camera_image(true);
  auto l_image = _current_integrator->get_light_image(true);

  uint2 image_size = raytracing.scene().camera.image_size;
  std::vector<float4> output(image_size.x * image_size.y, float4{});
  for (uint32_t i = 0, e = image_size.x * image_size.y; (c_image != nullptr) && (i < e); ++i) {
    output[i] = c_image[i];
  }
  for (uint32_t i = 0, e = image_size.x * image_size.y; (l_image != nullptr) && (i < e); ++i) {
    output[i] += l_image[i];
  }
  for (uint32_t i = 0, e = image_size.x * image_size.y; (as_xyz == false) && (i < e); ++i) {
    output[i] = {spectrum::xyz_to_rgb(output[i]), 1.0f};
  }

  const char* error = nullptr;
  if (SaveEXR(output.data()->data.data, image_size.x, image_size.y, 4, false, file_name.c_str(), &error) != TINYEXR_SUCCESS) {
    log::error("Failed to save EXR image to %s: %s", file_name.c_str(), error);
  }
}

void RTApplication::on_scene_file_selected(std::string file_name) {
  load_scene_file(file_name, SceneRepresentation::LoadEverything, false);
}

void RTApplication::on_integrator_selected(Integrator* i) {
  if (_current_integrator == i) {
    return;
  }

  _options.set("integrator", i->name());
  save_options();

  if (_current_integrator != nullptr) {
    _current_integrator->stop(false);
  }

  _current_integrator = i;
  ui.set_current_integrator(_current_integrator);

  if (scene) {
    _current_integrator->set_output_size(scene.scene().camera.image_size);
    _current_integrator->preview(ui.integrator_options());
  }
}

void RTApplication::on_preview_selected() {
  ETX_ASSERT(_current_integrator != nullptr);
  _current_integrator->preview(ui.integrator_options());
}

void RTApplication::on_run_selected() {
  ETX_ASSERT(_current_integrator != nullptr);
  _current_integrator->run(ui.integrator_options());
}

void RTApplication::on_stop_selected(bool wait_for_completion) {
  ETX_ASSERT(_current_integrator != nullptr);
  _current_integrator->stop(wait_for_completion);
}

void RTApplication::on_reload_scene_selected() {
  if (_current_scene_file.empty() == false) {
    bool start_render = (_current_integrator != nullptr) && (_current_integrator->state() == Integrator::State::Running);
    load_scene_file(_current_scene_file, SceneRepresentation::LoadEverything, start_render);
  }
}

void RTApplication::on_reload_geometry_selected() {
  if (_current_scene_file.empty() == false) {
    bool start_render = (_current_integrator != nullptr) && (_current_integrator->state() == Integrator::State::Running);
    load_scene_file(_current_scene_file, SceneRepresentation::LoadGeometry, start_render);
  }
}

void RTApplication::on_options_changed() {
  ETX_ASSERT(_current_integrator);
  _current_integrator->update_options(ui.integrator_options());
}

}  // namespace etx