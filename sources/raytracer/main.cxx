#include <etx/core/environment.hxx>
#include <etx/core/log.hxx>
#include <etx/core/profiler.hxx>
#include <etx/rhi/shader/shader_compiler.hxx>

#include "app.hxx"
#include "application_runner.hxx"
#include "batch_mode.hxx"
#include "bsdf_lut_generation.hxx"
#include "shader_packager.hxx"

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

#if defined(ETX_ENABLE_SHADER_PACKAGER) && ETX_ENABLE_SHADER_PACKAGER
  for (int argument_index = 1; argument_index < argc; ++argument_index) {
    if (std::strcmp(argv[argument_index], "--build-shader-package") != 0) {
      continue;
    }
    if ((argument_index + 3) >= argc) {
      fprintf(stderr, "Usage: raytracer --build-shader-package <output-file> --shader-backend <metal|vulkan> [--shader-source-root <directory>]\n");
      return 1;
    }
    const std::filesystem::path output_path = argv[argument_index + 1];
    if (std::strcmp(argv[argument_index + 2], "--shader-backend") != 0) {
      fprintf(stderr, "Expected --shader-backend after the shader package output path.\n");
      return 1;
    }
    RHIBackend backend = RHIBackend::Vulkan;
    if (std::strcmp(argv[argument_index + 3], "metal") == 0) {
      backend = RHIBackend::Metal;
    } else if (std::strcmp(argv[argument_index + 3], "vulkan") != 0) {
      fprintf(stderr, "Unsupported shader package backend '%s'.\n", argv[argument_index + 3]);
      return 1;
    }

    std::filesystem::path source_root = env().data_folder();
    for (int option_index = argument_index + 4; option_index < argc; ++option_index) {
      if (std::strcmp(argv[option_index], "--shader-source-root") != 0) {
        fprintf(stderr, "Unsupported shader package option '%s'.\n", argv[option_index]);
        return 1;
      }
      if ((option_index + 1) >= argc) {
        fprintf(stderr, "Expected a directory after --shader-source-root.\n");
        return 1;
      }
      source_root = argv[++option_index];
    }

    RaytracerShaderPackageStatistics statistics = {};
    std::string error_message = {};
    if (build_raytracer_shader_package(output_path, source_root, backend, statistics, error_message) == false) {
      fprintf(stderr, "Shader package build failed: %s\n", error_message.c_str());
      return 1;
    }
    printf("Shader package built from %s: variants=%u binaries=%.2f MiB package=%.2f MiB compile=%.2f ms write-and-verify=%.2f ms\n", source_root.string().c_str(),
      statistics.variant_count, static_cast<double>(statistics.binary_size_bytes) / (1024.0 * 1024.0), static_cast<double>(statistics.package_size_bytes) / (1024.0 * 1024.0),
      statistics.compile_time_ms, statistics.package_time_ms);
    return 0;
  }
#endif

#if defined(ETX_REQUIRE_SHADER_PACKAGE) && ETX_REQUIRE_SHADER_PACKAGE
  ShaderCompiler::instance().set_runtime_compilation_allowed(false);
  log::info("Shader mode: production package");
#else
  ShaderCompiler::instance().set_shader_package_lookup_allowed(false);
  if (ShaderCompiler::instance().initialize() != RHIResult::Success) {
    log::error("Shader compiler initialization failed");
  }
  log::info("Shader mode: development source compilation (package lookup disabled)");
#endif

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
