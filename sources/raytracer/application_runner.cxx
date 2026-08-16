#include "application_runner.hxx"

#include <etx/core/log.hxx>
#include <etx/core/profiler.hxx>

#include <chrono>
#include <charconv>
#include <csignal>
#include <cstring>
#include <limits>
#include <thread>
#include <utility>

namespace etx {
namespace {

volatile std::sig_atomic_t control_loop_interrupted = 0;

void interrupt_control_loop(int) {
  control_loop_interrupted = 1;
}

bool parse_u32(const char* text, uint32_t& value) {
  if ((text == nullptr) || (*text == 0)) {
    return false;
  }
  uint64_t parsed = 0u;
  const char* end = text + std::strlen(text);
  const auto result = std::from_chars(text, end, parsed, 10);
  if ((result.ec != std::errc()) || (result.ptr != end) || (parsed > std::numeric_limits<uint32_t>::max())) {
    return false;
  }
  value = static_cast<uint32_t>(parsed);
  return true;
}

bool parse_resolution(const char* text, uint32_t& width, uint32_t& height) {
  if (text == nullptr) {
    return false;
  }
  const char* separator = std::strchr(text, 'x');
  if (separator == nullptr) {
    separator = std::strchr(text, 'X');
  }
  if ((separator == nullptr) || (separator == text) || (separator[1] == 0)) {
    return false;
  }
  const std::string width_text(text, static_cast<size_t>(separator - text));
  return parse_u32(width_text.c_str(), width) && parse_u32(separator + 1, height) && (width > 0u) && (height > 0u);
}

std::string runtime_usage() {
  return "Interactive and UI-independent modes:\n"
         "  raytracer --control-server [--bind ADDRESS] [--port N] [--scene FILE] [--renderer cpu|raster|gpu]\n"
         "  raytracer --headless [--frames N] [--scene FILE] [--renderer cpu|raster|gpu]\n"
         "  raytracer --window-only [--bind ADDRESS] [--port N] [--window-size WIDTHxHEIGHT] [--scene FILE] [--renderer cpu|raster|gpu]\n\n"
         "--control-server starts a headless renderer controlled at http://127.0.0.1:PORT (default port: 1654).\n"
         "--bind explicitly permits another interface; the safe default accepts local connections only.\n"
         "--window-only keeps the rendered window, removes native and ImGui controls, and starts the same control server.\n";
}

}  // namespace

ApplicationRuntimeOptions parse_application_runtime_options(int argc, char* argv[]) {
  ApplicationRuntimeOptions options = {};
  bool activated = false;
  bool window_only = false;
  bool headless = false;
  bool frames_specified = false;
  bool port_specified = false;
  bool bind_specified = false;
  bool window_size_specified = false;

  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i];
    activated |= (argument == "--control-server") || (argument == "--browser-control") || (argument == "--headless") || (argument == "--window-only");
  }
  if (!activated) {
    return options;
  }

  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i];
    if ((argument == "--help") || (argument == "-h")) {
      options.command = ApplicationRuntimeCommand::Help;
      options.message = runtime_usage();
      return options;
    }
    if ((argument == "--control-server") || (argument == "--browser-control")) {
      options.control_server = true;
      continue;
    }
    if (argument == "--headless") {
      headless = true;
      continue;
    }
    if (argument == "--window-only") {
      window_only = true;
      options.control_server = true;
      continue;
    }
    auto require_value = [&](const char* option) -> const char* {
      if ((i + 1) >= argc) {
        options.message = std::string("Missing value for ") + option + "\n\n" + runtime_usage();
        return nullptr;
      }
      return argv[++i];
    };
    if (argument == "--port") {
      port_specified = true;
      const char* value = require_value("--port");
      uint32_t port = 0u;
      if ((value == nullptr) || !parse_u32(value, port) || (port == 0u) || (port > 65535u)) {
        if (value != nullptr)
          options.message = "Invalid port\n\n" + runtime_usage();
        options.command = ApplicationRuntimeCommand::Error;
        return options;
      }
      options.port = static_cast<uint16_t>(port);
      continue;
    }
    if (argument == "--bind") {
      bind_specified = true;
      const char* value = require_value("--bind");
      if ((value == nullptr) || (*value == 0)) {
        options.command = ApplicationRuntimeCommand::Error;
        return options;
      }
      options.bind_address = value;
      continue;
    }
    if (argument == "--frames") {
      frames_specified = true;
      const char* value = require_value("--frames");
      if ((value == nullptr) || !parse_u32(value, options.frames) || (options.frames == 0u)) {
        if (value != nullptr)
          options.message = "Invalid frame count\n\n" + runtime_usage();
        options.command = ApplicationRuntimeCommand::Error;
        return options;
      }
      continue;
    }
    if (argument == "--window-size") {
      window_size_specified = true;
      const char* value = require_value("--window-size");
      if ((value == nullptr) || !parse_resolution(value, options.application.width, options.application.height)) {
        if (value != nullptr)
          options.message = "Invalid window size\n\n" + runtime_usage();
        options.command = ApplicationRuntimeCommand::Error;
        return options;
      }
      continue;
    }
    if (argument == "--scene") {
      const char* value = require_value("--scene");
      if (value == nullptr) {
        options.command = ApplicationRuntimeCommand::Error;
        return options;
      }
      options.initial_scene = value;
      continue;
    }
    if (argument == "--renderer") {
      const char* value = require_value("--renderer");
      if (value == nullptr) {
        options.command = ApplicationRuntimeCommand::Error;
        return options;
      }
      const std::string renderer = value;
      if (renderer == "cpu")
        options.initial_renderer = RendererMode::CPURaytracing;
      else if (renderer == "raster")
        options.initial_renderer = RendererMode::Rasterization;
      else if (renderer == "gpu")
        options.initial_renderer = RendererMode::GPURaytracing;
      else {
        options.command = ApplicationRuntimeCommand::Error;
        options.message = "Invalid renderer\n\n" + runtime_usage();
        return options;
      }
      continue;
    }

    options.command = ApplicationRuntimeCommand::Error;
    options.message = "Unknown interactive option: " + argument + "\n\n" + runtime_usage();
    return options;
  }

  if (window_only && headless) {
    options.command = ApplicationRuntimeCommand::Error;
    options.message = "--headless and --window-only cannot be combined\n\n" + runtime_usage();
    return options;
  }
  if (window_only && frames_specified) {
    options.command = ApplicationRuntimeCommand::Error;
    options.message = "--frames is only available in headless mode\n\n" + runtime_usage();
    return options;
  }
  if (options.control_server && frames_specified) {
    options.command = ApplicationRuntimeCommand::Error;
    options.message = "--frames cannot be combined with the control server\n\n" + runtime_usage();
    return options;
  }
  if (!options.control_server && (port_specified || bind_specified)) {
    options.command = ApplicationRuntimeCommand::Error;
    options.message = "--bind and --port require --control-server or --window-only\n\n" + runtime_usage();
    return options;
  }
  if (!window_only && window_size_specified) {
    options.command = ApplicationRuntimeCommand::Error;
    options.message = "--window-size is only available with --window-only\n\n" + runtime_usage();
    return options;
  }
  options.command = window_only ? ApplicationRuntimeCommand::Desktop : ApplicationRuntimeCommand::Headless;
  options.application.runtime_mode = window_only ? RuntimeMode::Desktop : RuntimeMode::Headless;
  options.application.enable_imgui = !window_only;
  options.application.enable_platform_ui = !window_only;
  options.application.persist_options = false;
  options.application.override_renderer = true;
  options.application.renderer = options.initial_renderer;
  if (options.command == ApplicationRuntimeCommand::Headless) {
    options.application.width = 1u;
    options.application.height = 1u;
    options.application.enable_imgui = false;
    options.application.enable_platform_ui = false;
  }
  return options;
}

ApplicationRuntime::ApplicationRuntime(ApplicationRuntimeOptions options)
  : _options(std::move(options)) {
}

void ApplicationRuntime::prepare_startup() {
  if (_options.command == ApplicationRuntimeCommand::Desktop) {
    _application.init(_options.application);
  } else {
    _application.prepare_startup();
  }
}

bool ApplicationRuntime::start_control_server() {
  if (!_options.control_server || _control_server.running()) {
    return true;
  }
  return _control_server.init(
    {.port = _options.port, .bind_address = _options.bind_address},
    [this]() {
      return _application.state_snapshot();
    },
    [this](ApplicationCommand command) {
      return _application.submit_command(std::move(command));
    },
    [this](std::vector<ApplicationCommandResult>& results) {
      _application.drain_command_results(results);
    },
    [this](std::vector<uint8_t>& png, uint32_t& width, uint32_t& height) {
      return _application.capture_output_png(png, width, height);
    });
}

void ApplicationRuntime::submit_initial_commands() {
  if (_initial_commands_submitted || !_application.initialized()) {
    return;
  }
  _initial_commands_submitted = true;
  if (!_options.initial_scene.empty()) {
    _application.submit_command({.type = ApplicationCommandType::LoadScene, .path = _options.initial_scene});
  }
}

void ApplicationRuntime::submit_initial_run_when_ready() {
  if (_initial_run_submitted || _options.control_server || _options.initial_scene.empty()) {
    return;
  }
  const ApplicationStateSnapshot state = _application.state_snapshot();
  if (state.scene_loaded && state.controls.can_run) {
    _application.submit_command({.type = ApplicationCommandType::Run});
    _initial_run_submitted = true;
  }
}

void ApplicationRuntime::frame() {
  _application.frame();
  if (_application.initialized()) {
    submit_initial_commands();
    submit_initial_run_when_ready();
    if (!start_control_server()) {
      log::error("Failed to start application control server");
      _application.submit_command({.type = ApplicationCommandType::Quit});
    }
  }
  _control_server.poll();
}

void ApplicationRuntime::cleanup() {
  _application.cleanup();
  _control_server.shutdown();
}

void ApplicationRuntime::process_event(const sapp_event* event) {
  _application.process_event(event);
}

int ApplicationRuntime::run_headless() {
  _application.init(_options.application);
  if (!_application.initialized()) {
    log::error("Headless application initialization failed");
    _application.cleanup();
    return 1;
  }
  submit_initial_commands();
  if (!start_control_server()) {
    log::error("Failed to start application control server");
    _application.cleanup();
    return 1;
  }

  control_loop_interrupted = 0;
  std::signal(SIGINT, interrupt_control_loop);
  std::signal(SIGTERM, interrupt_control_loop);
  uint64_t frame_index = 0u;
  const bool continuous = _options.control_server;
  while ((control_loop_interrupted == 0) && !_application.quit_requested() && (continuous || (frame_index < _options.frames))) {
    frame();
    ETX_END_PROFILER_FRAME();
    ++frame_index;
    if (continuous) {
      std::this_thread::sleep_for(std::chrono::milliseconds(8));
    }
  }
  cleanup();
  return 0;
}

}  // namespace etx
