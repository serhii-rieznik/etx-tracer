#include <etx/core/environment.hxx>
#include <etx/core/profiler.hxx>
#include <etx/rhi/shader/shader_compiler.hxx>

#include "app.hxx"

#include <string>
#include <vector>
#include <cstring>

namespace etx {

extern "C" int main(int argc, char* argv[]) {
  ETX_PROFILER_MAIN_THREAD();

  init_platform();
  env().setup(argv[0]);

  PlaygroundApp app = {};
  sapp_desc desc = {};
  desc.init_userdata_cb = [](void* data) {
    reinterpret_cast<PlaygroundApp*>(data)->init();
  };
  desc.frame_userdata_cb = [](void* data) {
    reinterpret_cast<PlaygroundApp*>(data)->frame();
    ETX_END_PROFILER_FRAME();
  };
  desc.cleanup_userdata_cb = [](void* data) {
    reinterpret_cast<PlaygroundApp*>(data)->cleanup();
  };
  desc.event_userdata_cb = [](const sapp_event* e, void* data) {
    reinterpret_cast<PlaygroundApp*>(data)->process_event(e);
  };
  desc.width = 1600;
  desc.height = 900;
  desc.high_dpi = true;
  desc.window_title = "etx-playground";
  desc.win32.console_utf8 = true;
  desc.win32.console_create = true;
  desc.user_data = &app;
  desc.fullscreen = false;
  desc.alpha = false;

  sapp_run(desc);
  return 0;
}

}  // namespace etx
