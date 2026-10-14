#include <etx/core/environment.hxx>
#include <etx/core/log.hxx>
#include <etx/core/profiler.hxx>
#include <etx/rhi/shader/shader_compiler.hxx>

#include "app.hxx"
#include "application_runner.hxx"
#include "batch_mode.hxx"
#include "bsdf_lut_generation.hxx"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>
#if (ETX_PLATFORM_WINDOWS)
# include <windows.h>
# include <shellapi.h>
#endif

namespace etx {

extern "C" int main(int argc, char* argv[]) {
  ETX_PROFILER_MAIN_THREAD();

#if (ETX_PLATFORM_WINDOWS)
  wchar_t** wide_arguments = CommandLineToArgvW(GetCommandLineW(), &argc);
  if (wide_arguments == nullptr) {
    fprintf(stderr, "Cannot read the command line.\n");
    return 1;
  }
  std::vector<std::string> arguments;
  arguments.reserve(argc);
  for (int index = 0; index < argc; ++index) {
    const std::u8string value = std::filesystem::path(wide_arguments[index]).u8string();
    arguments.emplace_back(reinterpret_cast<const char*>(value.data()), value.size());
  }
  LocalFree(wide_arguments);
  std::vector<char*> argument_pointers;
  argument_pointers.reserve(arguments.size() + 1u);
  for (auto& argument : arguments)
    argument_pointers.push_back(argument.data());
  argument_pointers.push_back(nullptr);
  argv = argument_pointers.data();
#endif

  init_platform();
  env().setup(argv[0]);

  if ((argc == 2) && (std::strcmp(argv[1], "--list-importers") == 0)) {
    const auto& service = import_service();
    printf("Importer directory: %s\n", path_to_utf8(service.plugin_directory()).c_str());
    for (const auto& info : service.modules()) {
      printf("%s: %s [%s] (%s)\n", path_to_utf8(info.module.filename()).c_str(), info.name.c_str(), info.loaded ? "Loaded" : "Unavailable", info.extensions.c_str());
      if (info.error.empty() == false)
        printf("  %s\n", info.error.c_str());
    }
    if (service.discovery_error().empty() == false) {
      fprintf(stderr, "%s\n", service.discovery_error().c_str());
      return 1;
    }
    return 0;
  }
  for (int index = 1; index < argc; ++index) {
    if (std::strcmp(argv[index], "--import") != 0)
      continue;
    std::string destination, importer_id;
    bool valid = (index == 1) && (argc >= 5);
    for (int option = 3; valid && (option < argc); option += 2) {
      if ((option + 1) >= argc) {
        valid = false;
        break;
      }
      if ((std::strcmp(argv[option], "--output") == 0) && destination.empty())
        destination = argv[option + 1];
      else if ((std::strcmp(argv[option], "--importer") == 0) && importer_id.empty())
        importer_id = argv[option + 1];
      else
        valid = false;
    }
    if ((valid == false) || destination.empty()) {
      fprintf(stderr, "Usage: raytracer --import <source-file> --output <native.etx.json> [--importer <id>]\n");
      return 1;
    }
    std::string error;
    auto artifact = import_service().convert(
      std::filesystem::u8path(argv[2]), importer_id,
      [](uint32_t completed, uint32_t total, const char* stage) {
        printf("Import %u/%u: %s\n", completed, total, stage);
        return true;
      },
      error);
    std::string output;
    if ((artifact == nullptr) || (publish_import_document(*artifact, std::filesystem::u8path(destination), output, error) == false)) {
      fprintf(stderr, "Import failed: %s\n", error.c_str());
      return 1;
    }
    printf("Native document: %s\n", output.c_str());
    return 0;
  }

  log::info("Shader mode: on-demand source compilation");

  ApplicationRuntimeOptions runtime_options = parse_application_runtime_options(argc, argv);
  if (runtime_options.command == ApplicationRuntimeCommand::Help) {
    printf("%s", runtime_options.message.c_str());
    return 0;
  }
  if (runtime_options.command == ApplicationRuntimeCommand::Error) {
    fprintf(stderr, "%s", runtime_options.message.c_str());
    return 1;
  }
  if (runtime_options.command == ApplicationRuntimeCommand::ControlServer) {
    ApplicationRuntime runtime(std::move(runtime_options));
    return runtime.run_control_server();
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
  desc.window_title = "ETX Tracer";
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
