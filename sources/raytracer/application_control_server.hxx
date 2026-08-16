#pragma once

#include "application_control.hxx"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace etx {

inline constexpr uint16_t kDefaultApplicationControlPort = 1654u;

struct ApplicationUploadState;

struct ApplicationControlServerConfig {
  uint16_t port = kDefaultApplicationControlPort;
  std::string bind_address = "127.0.0.1";
};

struct ApplicationControlServer {
  using StateProvider = std::function<ApplicationStateSnapshot()>;
  using CommandSubmitter = std::function<uint64_t(ApplicationCommand)>;
  using ResultProvider = std::function<void(std::vector<ApplicationCommandResult>&)>;
  using ImageProvider = std::function<bool(std::vector<uint8_t>&, uint32_t&, uint32_t&)>;

  ApplicationControlServer();
  ~ApplicationControlServer();

  bool init(const ApplicationControlServerConfig& config, StateProvider state_provider, CommandSubmitter command_submitter, ResultProvider result_provider,
    ImageProvider image_provider);
  void poll();
  void shutdown();
  bool running() const;

 private:
  uint64_t _listen_socket = ~0ull;
  bool _socket_api_initialized = false;
  StateProvider _state_provider = {};
  CommandSubmitter _command_submitter = {};
  ResultProvider _result_provider = {};
  ImageProvider _image_provider = {};
  std::vector<ApplicationCommandResult> _retained_results = {};
  std::unique_ptr<ApplicationUploadState> _uploads = {};
};

}  // namespace etx
