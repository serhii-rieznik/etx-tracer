#include <etx/core/environment.hxx>
#include <etx/core/profiler.hxx>
#include <etx/rhi/shader/shader_compiler.hxx>

#include "app.hxx"

#include <etx/core/log.hxx>

#include <cstdint>
#include <string>
#include <vector>
#include <cstring>

namespace etx {

namespace {

struct PlaygroundCliOptions {
  bool headless = false;
  uint32_t frames = 2u;
  uint32_t width = 1600u;
  uint32_t height = 900u;
};

bool parse_u32_argument(const char* value, uint32_t& out_value) {
  if ((value == nullptr) || (*value == '\0')) {
    return false;
  }

  char* end_ptr = nullptr;
  const unsigned long parsed = std::strtoul(value, &end_ptr, 10);
  if ((end_ptr == value) || (end_ptr == nullptr) || (*end_ptr != '\0')) {
    return false;
  }

  out_value = static_cast<uint32_t>(parsed);
  return true;
}

bool parse_resolution_argument(const char* value, uint32_t& out_width, uint32_t& out_height) {
  if (value == nullptr) {
    return false;
  }

  const char* separator = std::strchr(value, 'x');
  if (separator == nullptr) {
    separator = std::strchr(value, 'X');
  }
  if (separator == nullptr) {
    return false;
  }

  const std::string width_str(value, static_cast<size_t>(separator - value));
  const std::string height_str(separator + 1u);
  uint32_t width = 0u;
  uint32_t height = 0u;
  if ((parse_u32_argument(width_str.c_str(), width) == false) || (parse_u32_argument(height_str.c_str(), height) == false)) {
    return false;
  }

  out_width = width;
  out_height = height;
  return (out_width > 0u) && (out_height > 0u);
}

void print_usage() {
  log::info("Usage:");
  log::info("  playground_app");
  log::info("  playground_app --headless [--frames N] [--resolution WIDTHxHEIGHT]");
  log::info("  playground_app --help");
}

bool parse_cli_options(int argc, char* argv[], PlaygroundCliOptions& options) {
  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i];
    if ((argument == "--help") || (argument == "-h")) {
      print_usage();
      return false;
    }
    if (argument == "--headless") {
      options.headless = true;
      continue;
    }
    if (argument == "--frames") {
      if ((i + 1) >= argc) {
        log::error("Missing value for --frames");
        return false;
      }
      if (parse_u32_argument(argv[++i], options.frames) == false) {
        log::error("Invalid value for --frames: %s", argv[i]);
        return false;
      }
      continue;
    }
    if (argument == "--resolution") {
      if ((i + 1) >= argc) {
        log::error("Missing value for --resolution");
        return false;
      }
      if (parse_resolution_argument(argv[++i], options.width, options.height) == false) {
        log::error("Invalid value for --resolution: %s", argv[i]);
        return false;
      }
      continue;
    }

    log::error("Unknown argument: %s", argument.c_str());
    return false;
  }

  return true;
}

}  // namespace

extern "C" int main(int argc, char* argv[]) {
  ETX_PROFILER_MAIN_THREAD();

  init_platform();
  env().setup(argv[0]);

  PlaygroundCliOptions options = {};
  if (parse_cli_options(argc, argv, options) == false) {
    return 1;
  }

  if (options.headless) {
    PlaygroundApp app = {};
    app.init_headless(options.width, options.height);
    if (app.initialized() == false) {
      log::error("Playground headless: initialization failed");
      return 1;
    }

    log::info("Playground headless: running %u frame(s) at %ux%u", options.frames, options.width, options.height);
    for (uint32_t frame_index = 0u; frame_index < options.frames; ++frame_index) {
      log::info("Playground headless: frame %u / %u", frame_index + 1u, options.frames);
      app.frame_headless();
      ETX_END_PROFILER_FRAME();
    }
    app.cleanup();
    return 0;
  }

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
