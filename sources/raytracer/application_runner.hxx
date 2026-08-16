#pragma once

#include "app.hxx"
#include "application_control_server.hxx"

#include <cstdint>
#include <string>

struct sapp_event;

namespace etx {

enum class ApplicationRuntimeCommand : uint32_t {
  None,
  Help,
  Error,
  Desktop,
  Headless,
};

struct ApplicationRuntimeOptions {
  ApplicationRuntimeCommand command = ApplicationRuntimeCommand::None;
  ApplicationConfig application = {};
  bool control_server = false;
  uint16_t port = kDefaultApplicationControlPort;
  std::string bind_address = "127.0.0.1";
  uint32_t frames = 1u;
  std::string initial_scene = {};
  RendererMode initial_renderer = RendererMode::CPURaytracing;
  std::string message = {};
};

ApplicationRuntimeOptions parse_application_runtime_options(int argc, char* argv[]);

struct ApplicationRuntime {
  explicit ApplicationRuntime(ApplicationRuntimeOptions options);

  void prepare_startup();
  void frame();
  void cleanup();
  void process_event(const sapp_event* event);
  int run_headless();

 private:
  bool start_control_server();
  void submit_initial_commands();
  void submit_initial_run_when_ready();

 private:
  ApplicationRuntimeOptions _options = {};
  RTApplication _application = {};
  ApplicationControlServer _control_server = {};
  bool _initial_commands_submitted = false;
  bool _initial_run_submitted = false;
};

}  // namespace etx
