#include "app.hxx"

namespace etx {
void RTApplication::init() {
  render.init();
  ui.initialize();
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

}  // namespace etx