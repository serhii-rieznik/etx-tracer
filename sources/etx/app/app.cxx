#include <etx/app/app.hxx>

#include <Windows.h>

#include <sokol_app.h>

namespace etx {

struct ApplicationPrivate {
  ApplicationImpl& impl;

  ApplicationPrivate(ApplicationImpl& i)
    : impl(i) {
  }

  void init() {
    impl.init();
  }

  void frame() {
    impl.process();
  }

  void cleanup() {
    impl.cleanup();
  }

  void add_event(const sapp_event*) {
  }
};

ETX_IMPLEMENT_PIMPL(Application);

Application::Application(ApplicationImpl& impl) {
  ETX_PIMPL_INIT(Application, impl);
}

Application::~Application() {
  ETX_PIMPL_CLEANUP(Application);
}

int Application::run(int argc, char* argv[]) {
  auto s_init = [](void* data) {
    reinterpret_cast<ApplicationPrivate*>(data)->init();
  };
  auto s_frame = [](void* data) {
    reinterpret_cast<ApplicationPrivate*>(data)->frame();
  };
  auto s_cleanup = [](void* data) {
    reinterpret_cast<ApplicationPrivate*>(data)->cleanup();
  };
  auto s_event = [](const sapp_event* e, void* data) {
    reinterpret_cast<ApplicationPrivate*>(data)->add_event(e);
  };

  sapp_desc desc = {};
  {
    desc.init_userdata_cb = s_init;
    desc.frame_userdata_cb = s_frame;
    desc.cleanup_userdata_cb = s_cleanup;
    desc.event_userdata_cb = s_event;
    desc.width = 1280;
    desc.height = 720;
    desc.window_title = "etx-tracer";
    desc.high_dpi = true;
    desc.win32_console_utf8 = true;
    desc.win32_console_create = true;
    desc.user_data = _private;
  };
  sapp_run(desc);
  return 0;
}

}  // namespace etx