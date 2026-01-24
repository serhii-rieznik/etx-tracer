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

  // Initialize global shader compiler
  if (ShaderCompiler::initialize_global() != RHIResult::Success) {
    log::error("Failed to initialize global shader compiler");
    return 1;
  }

  // Parse command line arguments
  bool headless_mode = false;
  bool show_help = false;

  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "--headless") == 0 || strcmp(argv[i], "-h") == 0) {
      headless_mode = true;
    } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-?") == 0) {
      show_help = true;
    }
  }

  if (show_help) {
    printf("RHI Test Application\n");
    printf("Usage: %s [options]\n", argv[0]);
    printf("Options:\n");
    printf("  --headless, -h    Run in headless mode (no window)\n");
    printf("  --help, -?        Show this help message\n");
    printf("  --windowed        Run in windowed mode (default)\n");
    return 0;
  }

  log::info("Starting RHI test mode (%s)...", headless_mode ? "headless" : "windowed");

  RHITestApplication rhi_app;
  rhi_app.set_headless_mode(headless_mode);

  if (headless_mode) {
    // Run in headless mode - no window, run test and exit
    rhi_app.init();
    rhi_app.run_headless_test();
    rhi_app.cleanup();
    log::info("RHI headless test completed");
    ShaderCompiler::shutdown_global();
    return 0;
  } else {
    // Run in windowed mode with sokol_app
    log::info("Initializing windowed mode with sokol_app...");
    sapp_desc desc = {};
    {
      desc.init_userdata_cb = [](void* data) {
        log::info("sokol_app init callback called");
        reinterpret_cast<RHITestApplication*>(data)->init();
        log::info("Window created and RHI initialized");
      };
      desc.frame_userdata_cb = [](void* data) {
        reinterpret_cast<RHITestApplication*>(data)->frame();
      };
      desc.cleanup_userdata_cb = [](void* data) {
        log::info("sokol_app cleanup callback called");
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
      desc.swap_interval = 0;
      desc.fullscreen = false;  // Make sure we're not in fullscreen
      desc.alpha = false;
    }

    log::info("Starting sokol_app with %dx%d window...", desc.width, desc.height);
    sapp_run(desc);
    log::info("sokol_app finished");
    ShaderCompiler::shutdown_global();
    return 0;
  }
}

}  // namespace etx
