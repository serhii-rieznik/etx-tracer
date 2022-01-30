#include <etx/log/log.hxx>

#include <etx/render/host/image_pool.hxx>

#include "app.hxx"

namespace etx {
void RTApplication::init() {
  render.init();
  ui.initialize();
  ui.callbacks.reference_image_selected = std::bind(&RTApplication::on_referenece_image_selected, this, std::placeholders::_1);
}

void RTApplication::frame() {
  ui.build(time_measure.lap());

  render.start_frame();
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
  log::warning("Loading %s...", file_name.c_str());
  render.set_reference_image(file_name.c_str());
}

}  // namespace etx