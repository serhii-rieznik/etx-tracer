#include <etx/core/environment.hxx>
#include <etx/core/profiler.hxx>
#include <etx/rhi/shader/shader_compiler.hxx>

#include "app.hxx"
#include "application_runner.hxx"
#include "batch_mode.hxx"
#include "bsdf_lut_generation.hxx"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace etx {

extern "C" int main(int argc, char* argv[]) {
  ETX_PROFILER_MAIN_THREAD();

  init_platform();
  env().setup(argv[0]);

  ApplicationRuntimeOptions runtime_options = parse_application_runtime_options(argc, argv);
  if (runtime_options.command == ApplicationRuntimeCommand::Help) {
    printf("%s", runtime_options.message.c_str());
    return 0;
  }
  if (runtime_options.command == ApplicationRuntimeCommand::Error) {
    fprintf(stderr, "%s", runtime_options.message.c_str());
    return 1;
  }
  if (runtime_options.command == ApplicationRuntimeCommand::Headless) {
    ApplicationRuntime runtime(std::move(runtime_options));
    return runtime.run_headless();
  }

  if (runtime_options.command == ApplicationRuntimeCommand::None) {
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
    if (batch_command == BatchModeCommand::GenerateBSDFLuts) {
      BSDFLutGenerationOptions lut_options = {};
      lut_options.output_directory = batch_options.output_file;
      lut_options.sample_count = batch_options.bsdf_lut_samples;
      return generate_bsdf_energy_compensation_luts(lut_options) ? 0 : 1;
    }
    if (batch_command == BatchModeCommand::PregenerateBSDFLutCache) {
      return pregenerate_named_bsdf_energy_compensation_lut_cache() ? 0 : 1;
    }
  }

  const uint32_t window_width = runtime_options.application.width;
  const uint32_t window_height = runtime_options.application.height;
  ApplicationRuntime runtime(std::move(runtime_options));
  sapp_desc desc = {};
  desc.init_userdata_cb = [](void* data) {
    reinterpret_cast<ApplicationRuntime*>(data)->prepare_startup();
  };
  desc.frame_userdata_cb = [](void* data) {
    reinterpret_cast<ApplicationRuntime*>(data)->frame();
    ETX_END_PROFILER_FRAME();
  };
  desc.cleanup_userdata_cb = [](void* data) {
    reinterpret_cast<ApplicationRuntime*>(data)->cleanup();
  };
  desc.event_userdata_cb = [](const sapp_event* e, void* data) {
    reinterpret_cast<ApplicationRuntime*>(data)->process_event(e);
  };
  desc.width = static_cast<int>(window_width);
  desc.height = static_cast<int>(window_height);
  desc.high_dpi = true;
#if defined(ETX_PLATFORM_APPLE)
  desc.window_title = "ETX Tracer";
#else
  desc.window_title = "etx-tracer";
#endif
  desc.win32.console_utf8 = true;
  desc.win32.console_create = true;
  desc.user_data = &runtime;
  desc.fullscreen = false;
  desc.alpha = false;

  platform_ui().prepare_application();
  sapp_run(desc);
  return 0;
}

}  // namespace etx
