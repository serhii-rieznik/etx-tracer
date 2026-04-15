#pragma once

#include <cstdint>

#include <mutex>
#include <string>
#include <vector>

#include "browser_stream_server.hxx"

namespace etx {

struct WebRtcStreamSessionConfig {
  uint32_t width = 1920u;
  uint32_t height = 1080u;
  uint32_t fps = 60u;
  uint32_t bitrate = 20u * 1024u * 1024u;
};

struct WebRtcStreamSession {
  bool init(const WebRtcStreamSessionConfig& config);
  void shutdown();

  bool create_offer_response_json(std::string& response_json);
  bool accept_answer_from_json(const std::string& answer_json, std::string& response_json);

  void push_h264_access_unit(const uint8_t* data, size_t data_size, uint64_t timestamp_us, bool keyframe);
  void drain_input_events(std::vector<BrowserInputEvent>& output);
  bool consume_keyframe_request();

private:
  void reset_peer_connection();
  void enqueue_input_event(const BrowserInputEvent& event);
  void handle_data_channel_message(const std::string& message);
  std::string generate_session_id();

private:
  WebRtcStreamSessionConfig _config = {};
  void* _peer_connection = nullptr;
  void* _video_track = nullptr;
  void* _input_fast_channel = nullptr;
  void* _input_control_channel = nullptr;
  bool _initialized = false;
  bool _track_open = false;
  bool _pending_keyframe_request = false;
  std::string _session_id = {};
  std::string _pending_offer_sdp = {};
  std::vector<uint8_t> _latest_keyframe = {};
  std::mutex _mutex = {};
  std::vector<BrowserInputEvent> _pending_input_events = {};
};

}  // namespace etx
