#include "webrtc_stream_session.hxx"

#include <etx/core/log.hxx>

#include <json.hpp>
#include <rtc/rtc.hpp>
#include <sokol_app.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <memory>

namespace etx {
namespace {

using Json = nlohmann::json;

template <typename T>
std::shared_ptr<T>* as_shared_ptr(void* value) {
  return reinterpret_cast<std::shared_ptr<T>*>(value);
}

struct OfferWaitState {
  std::mutex mutex = {};
  std::condition_variable condition = {};
  std::string offer_sdp = {};
  bool ready = false;
};

BrowserInputEvent parse_browser_input_event(const Json& event_json) {
  BrowserInputEvent event = {};
  if (event_json.is_object() == false) {
    return event;
  }

  const std::string type = event_json.value("type", std::string());
  if (type == "mouse_move") {
    event.type = BrowserInputEventType::MouseMove;
    event.delta_x = event_json.value("dx", 0.0f);
    event.delta_y = event_json.value("dy", 0.0f);
  } else if (type == "mouse_button") {
    event.type = BrowserInputEventType::MouseButton;
    event.mouse_button = event_json.value("button", 0u);
    event.pressed = event_json.value("pressed", false);
  } else if (type == "wheel") {
    event.type = BrowserInputEventType::MouseWheel;
    event.wheel_delta = event_json.value("delta_y", 0.0f);
  } else if (type == "key") {
    event.type = BrowserInputEventType::Key;
    const std::string code = event_json.value("code", std::string());
    if (code == "KeyW") {
      event.key_code = SAPP_KEYCODE_W;
    } else if (code == "KeyA") {
      event.key_code = SAPP_KEYCODE_A;
    } else if (code == "KeyS") {
      event.key_code = SAPP_KEYCODE_S;
    } else if (code == "KeyD") {
      event.key_code = SAPP_KEYCODE_D;
    } else if (code == "KeyQ") {
      event.key_code = SAPP_KEYCODE_Q;
    } else if (code == "KeyE") {
      event.key_code = SAPP_KEYCODE_E;
    } else if (code == "ShiftLeft") {
      event.key_code = SAPP_KEYCODE_LEFT_SHIFT;
    } else if (code == "ShiftRight") {
      event.key_code = SAPP_KEYCODE_RIGHT_SHIFT;
    } else if (code == "ControlLeft") {
      event.key_code = SAPP_KEYCODE_LEFT_CONTROL;
    } else if (code == "ControlRight") {
      event.key_code = SAPP_KEYCODE_RIGHT_CONTROL;
    }
    event.pressed = event_json.value("pressed", false);
  } else if (type == "focus") {
    event.type = BrowserInputEventType::Focus;
    event.focused = event_json.value("focused", false);
  }

  return event;
}

rtc::binary make_rtc_binary(const uint8_t* data, size_t data_size) {
  rtc::binary result = {};
  result.resize(data_size);
  if ((data != nullptr) && (data_size > 0u)) {
    std::memcpy(result.data(), data, data_size);
  }
  return result;
}

}  // namespace

bool WebRtcStreamSession::init(const WebRtcStreamSessionConfig& config) {
  shutdown();
  _config = config;

  static std::atomic<bool> logger_initialized = false;
  bool expected = false;
  if (logger_initialized.compare_exchange_strong(expected, true)) {
    rtc::InitLogger(rtc::LogLevel::Warning);
  }

  _initialized = true;
  _pending_keyframe_request = true;
  return true;
}

void WebRtcStreamSession::shutdown() {
  reset_peer_connection();

  std::lock_guard<std::mutex> lock(_mutex);
  _latest_keyframe.clear();
  _pending_input_events.clear();
  _pending_offer_sdp.clear();
  _session_id.clear();
  _pending_keyframe_request = false;
  _track_open = false;
  _initialized = false;
  _config = {};
}

std::string WebRtcStreamSession::generate_session_id() {
  static std::atomic<uint64_t> counter = 1u;
  const uint64_t value = counter.fetch_add(1u);
  return std::to_string(value);
}

void WebRtcStreamSession::reset_peer_connection() {
  if (_video_track != nullptr) {
    delete as_shared_ptr<rtc::Track>(_video_track);
    _video_track = nullptr;
  }
  if (_input_fast_channel != nullptr) {
    delete as_shared_ptr<rtc::DataChannel>(_input_fast_channel);
    _input_fast_channel = nullptr;
  }
  if (_input_control_channel != nullptr) {
    delete as_shared_ptr<rtc::DataChannel>(_input_control_channel);
    _input_control_channel = nullptr;
  }
  if (_peer_connection != nullptr) {
    std::shared_ptr<rtc::PeerConnection>* peer_connection = as_shared_ptr<rtc::PeerConnection>(_peer_connection);
    if ((peer_connection != nullptr) && (peer_connection->get() != nullptr)) {
      (*peer_connection)->close();
    }
    delete peer_connection;
    _peer_connection = nullptr;
  }

  std::lock_guard<std::mutex> lock(_mutex);
  _track_open = false;
  _pending_offer_sdp.clear();
}

void WebRtcStreamSession::enqueue_input_event(const BrowserInputEvent& event) {
  if (event.type == BrowserInputEventType::Invalid) {
    return;
  }

  std::lock_guard<std::mutex> lock(_mutex);
  _pending_input_events.push_back(event);
}

void WebRtcStreamSession::handle_data_channel_message(const std::string& message) {
  const Json event_json = Json::parse(message, nullptr, false);
  if (event_json.is_discarded()) {
    return;
  }

  const BrowserInputEvent event = parse_browser_input_event(event_json);
  if ((event.type == BrowserInputEventType::Key) && (event.key_code == 0u)) {
    return;
  }

  enqueue_input_event(event);
}

bool WebRtcStreamSession::create_offer_response_json(std::string& response_json) {
  response_json.clear();
  if (_initialized == false) {
    return false;
  }

  reset_peer_connection();
  _session_id = generate_session_id();

  rtc::Configuration configuration = {};
  configuration.disableAutoNegotiation = true;

  std::shared_ptr<rtc::PeerConnection> peer_connection = std::make_shared<rtc::PeerConnection>(configuration);
  std::shared_ptr<rtc::Track> video_track = {};
  std::shared_ptr<rtc::DataChannel> input_fast_channel = {};
  std::shared_ptr<rtc::DataChannel> input_control_channel = {};

  const std::shared_ptr<OfferWaitState> wait_state = std::make_shared<OfferWaitState>();
  peer_connection->onStateChange([this](rtc::PeerConnection::State state) {
    std::lock_guard<std::mutex> lock(_mutex);
    if (state == rtc::PeerConnection::State::Connected) {
      _track_open = true;
      _pending_keyframe_request = true;
      return;
    }

    if ((state == rtc::PeerConnection::State::Disconnected) || (state == rtc::PeerConnection::State::Failed) || (state == rtc::PeerConnection::State::Closed)) {
      _track_open = false;
    }
  });

  peer_connection->onGatheringStateChange([this, weak_peer_connection = std::weak_ptr<rtc::PeerConnection>(peer_connection),
                                            wait_state](rtc::PeerConnection::GatheringState state) {
    if (state == rtc::PeerConnection::GatheringState::Complete) {
      std::shared_ptr<rtc::PeerConnection> peer_connection_locked = weak_peer_connection.lock();
      if (peer_connection_locked == nullptr) {
        return;
      }

      const std::optional<rtc::Description> description = peer_connection_locked->localDescription();
      if (description.has_value() == false) {
        return;
      }

      {
        std::lock_guard<std::mutex> wait_lock(wait_state->mutex);
        wait_state->offer_sdp = std::string(description.value());
        wait_state->ready = true;
      }
      {
        std::lock_guard<std::mutex> session_lock(_mutex);
        _pending_offer_sdp = wait_state->offer_sdp;
      }
      wait_state->condition.notify_all();
    }
  });

  rtc::Description::Video video_description("video", rtc::Description::Direction::SendOnly);
  video_description.addH264Codec(96, "profile-level-id=42e02a;packetization-mode=1;level-asymmetry-allowed=1");
  video_description.addSSRC(1u, "etx-video", "etx-stream", "etx-video");
  video_track = peer_connection->addTrack(video_description);

  const std::shared_ptr<rtc::RtpPacketizationConfig> rtp_config =
    std::make_shared<rtc::RtpPacketizationConfig>(1u, "etx-video", 96u, rtc::H264RtpPacketizer::ClockRate);
  const std::shared_ptr<rtc::H264RtpPacketizer> packetizer =
    std::make_shared<rtc::H264RtpPacketizer>(rtc::NalUnit::Separator::StartSequence, rtp_config);
  const std::shared_ptr<rtc::RtcpSrReporter> sender_reporter = std::make_shared<rtc::RtcpSrReporter>(rtp_config);
  const std::shared_ptr<rtc::RtcpNackResponder> nack_responder = std::make_shared<rtc::RtcpNackResponder>();
  packetizer->addToChain(sender_reporter);
  packetizer->addToChain(nack_responder);
  video_track->setMediaHandler(packetizer);
  video_track->onOpen([this, weak_track = std::weak_ptr<rtc::Track>(video_track)]() {
    std::vector<uint8_t> latest_keyframe = {};
    {
      std::lock_guard<std::mutex> lock(_mutex);
      _track_open = true;
      _pending_keyframe_request = true;
      latest_keyframe = _latest_keyframe;
    }

    std::shared_ptr<rtc::Track> track = weak_track.lock();
    if ((track != nullptr) && (latest_keyframe.empty() == false)) {
      try {
        rtc::binary keyframe = make_rtc_binary(latest_keyframe.data(), latest_keyframe.size());
        track->sendFrame(keyframe, rtc::FrameInfo(std::chrono::duration<double>(0.0)));
      } catch (const std::exception& exception) {
        log::warning("WebRTC stream session: failed to send cached keyframe: %s", exception.what());
      }
    }
  });

  rtc::DataChannelInit input_fast_init = {};
  input_fast_init.reliability.unordered = true;
  input_fast_init.reliability.maxRetransmits = 0u;
  input_fast_channel = peer_connection->createDataChannel("input-fast", input_fast_init);
  input_control_channel = peer_connection->createDataChannel("input-control");

  input_fast_channel->onMessage([this](std::variant<rtc::binary, rtc::string> message) {
    if (std::holds_alternative<rtc::string>(message)) {
      handle_data_channel_message(std::get<rtc::string>(message));
    }
  });
  input_control_channel->onMessage([this](std::variant<rtc::binary, rtc::string> message) {
    if (std::holds_alternative<rtc::string>(message)) {
      handle_data_channel_message(std::get<rtc::string>(message));
    }
  });

  _peer_connection = new std::shared_ptr<rtc::PeerConnection>(peer_connection);
  _video_track = new std::shared_ptr<rtc::Track>(video_track);
  _input_fast_channel = new std::shared_ptr<rtc::DataChannel>(input_fast_channel);
  _input_control_channel = new std::shared_ptr<rtc::DataChannel>(input_control_channel);

  peer_connection->setLocalDescription();

  std::unique_lock<std::mutex> wait_lock(wait_state->mutex);
  const bool wait_completed = wait_state->condition.wait_for(wait_lock, std::chrono::seconds(5), [wait_state]() {
    return wait_state->ready;
  });
  if (wait_completed == false) {
    log::error("WebRTC stream session: timed out waiting for local offer");
    return false;
  }

  Json response = {
    {"ok", true},
    {"session_id", _session_id},
    {"type", "offer"},
    {"sdp", wait_state->offer_sdp},
  };
  response_json = response.dump();
  return true;
}

bool WebRtcStreamSession::accept_answer_from_json(const std::string& answer_json, std::string& response_json) {
  response_json.clear();
  if ((_initialized == false) || (_peer_connection == nullptr)) {
    return false;
  }

  const Json answer = Json::parse(answer_json, nullptr, false);
  if (answer.is_object() == false) {
    return false;
  }

  const std::string session_id = answer.value("session_id", std::string());
  const std::string sdp = answer.value("sdp", std::string());
  const std::string type = answer.value("type", std::string("answer"));
  if ((session_id != _session_id) || sdp.empty()) {
    return false;
  }

  std::shared_ptr<rtc::PeerConnection>* peer_connection = as_shared_ptr<rtc::PeerConnection>(_peer_connection);
  if ((peer_connection == nullptr) || (peer_connection->get() == nullptr)) {
    return false;
  }

  try {
    (*peer_connection)->setRemoteDescription(rtc::Description(sdp, type));
  } catch (const std::exception& exception) {
    log::error("WebRTC stream session: failed to set remote answer: %s", exception.what());
    return false;
  }

  Json response = {
    {"ok", true},
  };
  response_json = response.dump();
  return true;
}

void WebRtcStreamSession::push_h264_access_unit(const uint8_t* data, size_t data_size, uint64_t timestamp_us, bool keyframe) {
  if ((_initialized == false) || (_video_track == nullptr) || (data == nullptr) || (data_size == 0u)) {
    return;
  }

  std::shared_ptr<rtc::Track>* video_track = as_shared_ptr<rtc::Track>(_video_track);
  if ((video_track == nullptr) || (video_track->get() == nullptr)) {
    return;
  }

  const bool track_open = (*video_track)->isOpen();
  {
    std::lock_guard<std::mutex> lock(_mutex);
    _track_open = track_open;
    if (keyframe) {
      _latest_keyframe.assign(data, data + data_size);
    }
  }

  if (track_open == false) {
    return;
  }

  try {
    rtc::binary access_unit = make_rtc_binary(data, data_size);
    const std::chrono::duration<double> timestamp_seconds = std::chrono::duration<double>(static_cast<double>(timestamp_us) / 1'000'000.0);
    (*video_track)->sendFrame(access_unit, rtc::FrameInfo(timestamp_seconds));
  } catch (const std::exception& exception) {
    log::warning("WebRTC stream session: failed to send frame: %s", exception.what());
  }
}

void WebRtcStreamSession::drain_input_events(std::vector<BrowserInputEvent>& output) {
  std::lock_guard<std::mutex> lock(_mutex);
  output = std::move(_pending_input_events);
  _pending_input_events.clear();
}

bool WebRtcStreamSession::consume_keyframe_request() {
  std::lock_guard<std::mutex> lock(_mutex);
  const bool result = _pending_keyframe_request;
  _pending_keyframe_request = false;
  return result;
}

}  // namespace etx
