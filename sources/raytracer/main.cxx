#include <sokol_app.h>

void init_application() {
}

void frame() {
}

void cleanup_application() {
}

void handle_event(const sapp_event*) {
}

sapp_desc sokol_main(int argc, char* argv[]) {
  sapp_desc result = {};
  {
    result.init_cb = init_application;
    result.frame_cb = frame;
    result.cleanup_cb = cleanup_application;
    result.event_cb = handle_event;
    result.width = 1280;
    result.height = 720;
    result.window_title = "etx-tracer";
    result.high_dpi = true;
    result.win32_console_utf8 = true;
    result.win32_console_create = true;
  };
  return result;
}
