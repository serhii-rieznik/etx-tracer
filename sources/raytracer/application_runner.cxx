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

std::string runtime_usage() {
  return "Control-server mode:\n"
         "  raytracer --control-server [--bind ADDRESS] [--port N] [--scene FILE] [--renderer cpu|raster|gpu]\n\n"
         "--control-server starts a headless renderer controlled at http://127.0.0.1:PORT (default port: 1654).\n"
         "--bind explicitly permits another interface; the safe default accepts local connections only.\n";
}

}  // namespace

ApplicationRuntimeOptions parse_application_runtime_options(int argc, char* argv[]) {
  ApplicationRuntimeOptions options = {};
  bool activated = false;

  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i];
    activated |= (argument == "--control-server") || (argument == "--browser-control");
  }
  if (activated == false) {
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
      const char* value = require_value("--port");
      uint32_t port = 0u;
      if ((value == nullptr) || (parse_u32(value, port) == false) || (port == 0u) || (port > 65535u)) {
        if (value != nullptr)
          options.message = "Invalid port\n\n" + runtime_usage();
        options.command = ApplicationRuntimeCommand::Error;
        return options;
      }
      options.port = static_cast<uint16_t>(port);
      continue;
    }
    if (argument == "--bind") {
      const char* value = require_value("--bind");
      if ((value == nullptr) || (*value == 0)) {
        options.command = ApplicationRuntimeCommand::Error;
        return options;
      }
      options.bind_address = value;
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

  options.command = ApplicationRuntimeCommand::ControlServer;
  options.application.runtime_mode = RuntimeMode::Headless;
  options.application.persist_options = false;
  options.application.override_renderer = true;
  options.application.renderer = options.initial_renderer;
  options.application.width = 1u;
  options.application.height = 1u;
  return options;
}

ApplicationRuntime::ApplicationRuntime(ApplicationRuntimeOptions options)
  : _options(std::move(options)) {
}

void ApplicationRuntime::prepare_startup() {
  _application.prepare_startup();
}

bool ApplicationRuntime::start_control_server() {
  if (_control_server.running()) {
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
  if (_initial_commands_submitted || (_application.initialized() == false)) {
    return;
  }
  _initial_commands_submitted = true;
  if (_options.initial_scene.empty() == false) {
    _application.submit_command({.type = ApplicationCommandType::LoadScene, .path = _options.initial_scene});
  }
}

void ApplicationRuntime::frame() {
  _application.frame();
}

void ApplicationRuntime::cleanup() {
  _application.cleanup();
  _control_server.shutdown();
}

void ApplicationRuntime::process_event(const sapp_event* event) {
  _application.process_event(event);
}

int ApplicationRuntime::run_control_server() {
  _application.init(_options.application);
  if (_application.initialized() == false) {
    log::error("Control-server application initialization failed");
    _application.cleanup();
    return 1;
  }
  submit_initial_commands();
  if (start_control_server() == false) {
    log::error("Failed to start application control server");
    _application.cleanup();
    return 1;
  }

  control_loop_interrupted = 0;
  std::signal(SIGINT, interrupt_control_loop);
  std::signal(SIGTERM, interrupt_control_loop);
  while ((control_loop_interrupted == 0) && (_application.quit_requested() == false)) {
    frame();
    _control_server.poll();
    ETX_END_PROFILER_FRAME();
    std::this_thread::sleep_for(std::chrono::milliseconds(8));
  }
  cleanup();
  return 0;
}

}  // namespace etx
