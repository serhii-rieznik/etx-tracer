#include "app.hxx"

#include <sokol_app.h>

namespace etx {

extern "C" int main(int argc, char* argv[]) {
  RTApplication app;
  sapp_desc desc = {};
  {
    desc.init_userdata_cb = [](void* data) {
      reinterpret_cast<RTApplication*>(data)->init();
    };
    desc.frame_userdata_cb = [](void* data) {
      reinterpret_cast<RTApplication*>(data)->frame();
    };
    desc.cleanup_userdata_cb = [](void* data) {
      reinterpret_cast<RTApplication*>(data)->cleanup();
    };
    desc.event_userdata_cb = [](const sapp_event* e, void* data) {
      reinterpret_cast<RTApplication*>(data)->process_event(e);
    };
    desc.width = 1280;
    desc.height = 720;
    desc.window_title = "etx-tracer";
    desc.high_dpi = true;
    desc.win32_console_utf8 = true;
    desc.win32_console_create = true;
    desc.user_data = &app;
  };
  sapp_run(desc);
  return 0;
}

}  // namespace etx
