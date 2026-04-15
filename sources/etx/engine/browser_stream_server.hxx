#pragma once

#include <cstddef>
#include <cstdint>

#include <functional>
#include <string>
#include <vector>

namespace etx {

enum class BrowserInputEventType : uint32_t {
  Invalid = 0u,
  MouseMove,
  MouseButton,
  MouseWheel,
  Key,
  Focus,
};

struct BrowserInputEvent {
  BrowserInputEventType type = BrowserInputEventType::Invalid;
  float delta_x = 0.0f;
  float delta_y = 0.0f;
  float wheel_delta = 0.0f;
  uint32_t mouse_button = 0u;
  uint32_t key_code = 0u;
  bool pressed = false;
  bool focused = false;
};

struct BrowserStreamServerConfig {
  uint16_t port = 8080u;
  const char* page_title = "etx browser stream";
};

struct BrowserStreamServer {
  bool init(const BrowserStreamServerConfig& config);
  void shutdown();
  void poll();

  void set_offer_handler(const std::function<bool(std::string&)>& handler);
  void set_answer_handler(const std::function<bool(const std::string&, std::string&)>& handler);

  void set_frame_png(const uint8_t* data, size_t data_size, uint32_t width, uint32_t height, uint64_t frame_index);
  void drain_input_events(std::vector<BrowserInputEvent>& output);

  bool initialized() const {
    return _initialized;
  }

  const std::string& base_url() const {
    return _base_url;
  }

private:
  void enqueue_input_event(const BrowserInputEvent& event);

private:
  uint16_t _port = 0u;
  uint64_t _listen_socket = ~0ull;
  bool _initialized = false;
  bool _wsa_started = false;
  std::string _page_title = {};
  std::string _base_url = {};
  std::function<bool(std::string&)> _offer_handler = {};
  std::function<bool(const std::string&, std::string&)> _answer_handler = {};
  std::vector<uint8_t> _frame_png = {};
  uint32_t _frame_width = 0u;
  uint32_t _frame_height = 0u;
  uint64_t _frame_index = 0u;
  std::vector<BrowserInputEvent> _pending_input_events = {};
};

}  // namespace etx
