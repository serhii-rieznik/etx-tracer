#include <etx/log/log.hxx>

#include <etx/render/host/image_pool.hxx>

#include <etx/render/shared/scene.hxx>

#include "app.hxx"

namespace etx {

void RTApplication::init() {
  render.init();
  ui.initialize();
  ui.set_integrator_list(_integrator_array, std::size(_integrator_array));
  ui.callbacks.reference_image_selected = std::bind(&RTApplication::on_referenece_image_selected, this, std::placeholders::_1);
  ui.callbacks.scene_file_selected = std::bind(&RTApplication::on_scene_file_selected, this, std::placeholders::_1);
  ui.callbacks.integrator_selected = std::bind(&RTApplication::on_integrator_selected, this, std::placeholders::_1);
}

void RTApplication::frame() {
  float4* c_image = _current_integrator ? _current_integrator->get_updated_camera_image() : nullptr;
  float4* l_image = _current_integrator ? _current_integrator->get_updated_light_image() : nullptr;
  const char* status = _current_integrator ? _current_integrator->status() : nullptr;

  ui.build(time_measure.lap(), status);

  render.set_view_options(ui.view_options());
  render.start_frame();
  render.update_output_images(c_image, l_image);
  ui.render();
  render.end_frame();
}

void RTApplication::cleanup() {
  ui.cleanup();
  render.cleanup();
}

void RTApplication::process_event(const sapp_event* e) {
  ui.handle_event(e);
}

void RTApplication::on_referenece_image_selected(std::string file_name) {
  log::warning("Loading reference image %s...", file_name.c_str());
  render.set_reference_image(file_name.c_str());
}

void RTApplication::on_scene_file_selected(std::string file_name) {
  log::warning("Loading scene %s...", file_name.c_str());
  if (_current_integrator) {
    _current_integrator->stop();
  }

  scene.load_from_file(file_name.c_str());
  render.set_output_dimensions(_scene_output_size);

  // TODO : load scene
  if (_current_integrator != nullptr) {
    _current_integrator->set_output_size(_scene_output_size);
    _current_integrator->preview();
  }
}

void RTApplication::on_integrator_selected(Integrator* i) {
  if (_current_integrator == i) {
    return;
  }

  if (_current_integrator != nullptr) {
    _current_integrator->stop();
  }

  _current_integrator = i;
  ui.set_current_integrator(_current_integrator);

  _current_integrator->set_output_size(_scene_output_size);
  _current_integrator->preview();
}

}  // namespace etx