#include <etx/core/environment.hxx>
#include <etx/core/profiler.hxx>
#include <etx/rhi/shader/shader_compiler.hxx>
#include "rhi_test_app.hxx"

#include <string>
#include <vector>
#include <cstring>

namespace etx {

extern "C" int main(int argc, char* argv[]) {
  ETX_PROFILER_MAIN_THREAD();

  init_platform();
  env().setup(argv[0]);

  if (ShaderCompiler::initialize_global() != RHIResult::Success) {
    log::error("Failed to initialize global shader compiler");
    return 1;
  }

  RHITestApplication rhi_app = {};
  sapp_desc desc = {};
  desc.init_userdata_cb = [](void* data) {
    reinterpret_cast<RHITestApplication*>(data)->init();
  };
  desc.frame_userdata_cb = [](void* data) {
    reinterpret_cast<RHITestApplication*>(data)->frame();
  };
  desc.cleanup_userdata_cb = [](void* data) {
    reinterpret_cast<RHITestApplication*>(data)->cleanup();
  };
  desc.event_userdata_cb = [](const sapp_event* e, void* data) {
    reinterpret_cast<RHITestApplication*>(data)->process_event(e);
  };
  desc.width = 1600;
  desc.height = 900;
  desc.high_dpi = true;
  desc.window_title = "etx-tracer - RHI Test";
  desc.win32.console_utf8 = true;
  desc.win32.console_create = true;
  desc.user_data = &rhi_app;
  desc.fullscreen = false;
  desc.alpha = false;

  sapp_run(desc);
  ShaderCompiler::shutdown_global();
  return 0;
}

}  // namespace etx
