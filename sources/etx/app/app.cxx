#include <etx/app/app.hxx>

#include <Windows.h>

#include <sokol_app.h>

namespace etx {

struct ApplicationPrivate {
  ApplicationImpl& impl;

  ApplicationPrivate(ApplicationImpl& i)
    : impl(i) {
  }
};

ETX_IMPLEMENT_PIMPL(Application);

Application::Application(ApplicationImpl& impl) {
  ETX_PIMPL_INIT(Application, impl);
}

Application::~Application() {
  ETX_PIMPL_CLEANUP(Application);
}

void init_application() {
}

void frame() {
}

void cleanup_application() {
}

void handle_event(const sapp_event*) {
}

int Application::run(int argc, char* argv[]) {
  sapp_desc desc = {};
  {
    desc.init_cb = init_application;
    desc.frame_cb = frame;
    desc.cleanup_cb = cleanup_application;
    desc.event_cb = handle_event;
    desc.width = 1280;
    desc.height = 720;
    desc.window_title = "etx-tracer";
    desc.high_dpi = true;
    desc.win32_console_utf8 = true;
    desc.win32_console_create = true;
  };

  sapp_run(desc);
  return 0;
}

}  // namespace etx