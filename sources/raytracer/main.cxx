#include <etx/core/environment.hxx>
#include <etx/core/profiler.hxx>
#include <etx/rhi/shader/shader_compiler.hxx>

#include "app.hxx"
#include "batch_mode.hxx"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace etx {

extern "C" int main(int argc, char* argv[]) {
  ETX_PROFILER_MAIN_THREAD();

  init_platform();
  env().setup(argv[0]);

  BatchRenderOptions batch_options = {};
  std::string batch_message = {};
  const BatchModeCommand batch_command = parse_batch_command_line(argc, argv, batch_options, batch_message);
  if (batch_command == BatchModeCommand::Help) {
    printf("%s", batch_message.c_str());
    return 0;
  }
  if (batch_command == BatchModeCommand::Error) {
    fprintf(stderr, "%s", batch_message.c_str());
    return 1;
  }
  if (batch_command == BatchModeCommand::Run) {
    return run_batch_render(batch_options);
  }

  RTApplication rhi_app = {};
  sapp_desc desc = {};
  desc.init_userdata_cb = [](void* data) {
    reinterpret_cast<RTApplication*>(data)->init();
  };
  desc.frame_userdata_cb = [](void* data) {
    reinterpret_cast<RTApplication*>(data)->frame();
    ETX_END_PROFILER_FRAME();
  };
  desc.cleanup_userdata_cb = [](void* data) {
    reinterpret_cast<RTApplication*>(data)->cleanup();
  };
  desc.event_userdata_cb = [](const sapp_event* e, void* data) {
    reinterpret_cast<RTApplication*>(data)->process_event(e);
  };
  desc.width = 1600;
  desc.height = 900;
  desc.high_dpi = true;
  desc.window_title = "etx-tracer";
  desc.win32.console_utf8 = true;
  desc.win32.console_create = true;
  desc.user_data = &rhi_app;
  desc.fullscreen = false;
  desc.alpha = false;

  sapp_run(desc);
  return 0;
}

}  // namespace etx
