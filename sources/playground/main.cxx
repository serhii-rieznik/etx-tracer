#include <etx/core/environment.hxx>
#include <etx/core/profiler.hxx>
#include <etx/engine/browser_stream_server.hxx>
#include <etx/engine/mf_h264_encoder.hxx>
#include <etx/engine/webrtc_stream_session.hxx>
#include <etx/rhi/shader/shader_compiler.hxx>

#include "app.hxx"

#include <etx/core/log.hxx>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <string>
#include <thread>
#include <vector>
#include <cstring>

namespace etx {

namespace {

struct PlaygroundCliOptions {
  bool headless = false;
  bool browser = false;
  uint32_t frames = 2u;
  uint32_t width = 1600u;
  uint32_t height = 900u;
  uint16_t port = 8080u;
};

uint32_t default_streaming_bitrate(uint32_t width, uint32_t height, uint32_t fps) {
  constexpr double bits_per_pixel = 0.18;
  constexpr uint32_t min_bitrate = 6u * 1024u * 1024u;
  constexpr uint32_t max_bitrate = 40u * 1024u * 1024u;

  const double bitrate_value = static_cast<double>(width) * static_cast<double>(height) * static_cast<double>(fps) * bits_per_pixel;
  uint32_t bitrate = static_cast<uint32_t>(bitrate_value);
  bitrate = std::max(min_bitrate, bitrate);
  bitrate = std::min(max_bitrate, bitrate);
  return bitrate;
}

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
  log::info("  playground_app --browser [--port N] [--resolution WIDTHxHEIGHT]");
  log::info("  playground_app --streaming [--port N] [--resolution WIDTHxHEIGHT]");
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
    if (argument == "--browser") {
      options.browser = true;
      continue;
    }
    if (argument == "--streaming") {
      options.browser = true;
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
    if (argument == "--port") {
      if ((i + 1) >= argc) {
        log::error("Missing value for --port");
        return false;
      }

      uint32_t parsed_port = 0u;
      if (parse_u32_argument(argv[++i], parsed_port) == false) {
        log::error("Invalid value for --port: %s", argv[i]);
        return false;
      }
      if ((parsed_port == 0u) || (parsed_port > 65535u)) {
        log::error("Port is out of range: %u", parsed_port);
        return false;
      }
      options.port = static_cast<uint16_t>(parsed_port);
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

  if (options.browser) {
    if ((options.width == 1600u) && (options.height == 900u)) {
      options.width = 1920u;
      options.height = 1080u;
    }

    PlaygroundApp app = {};
    app.init_headless(options.width, options.height);
    if (app.initialized() == false) {
      log::error("Playground streaming: initialization failed");
      return 1;
    }

    BrowserStreamServer server = {};
    BrowserStreamServerConfig server_config = {
      .port = options.port,
      .page_title = "etx playground browser stream",
    };
    if (server.init(server_config) == false) {
      log::error("Playground streaming: failed to start local server");
      app.cleanup();
      return 1;
    }

    WebRtcStreamSession session = {};
    const uint32_t streaming_bitrate = default_streaming_bitrate(options.width, options.height, 60u);
    WebRtcStreamSessionConfig session_config = {
      .width = options.width,
      .height = options.height,
      .fps = 60u,
      .bitrate = streaming_bitrate,
    };
    if (session.init(session_config) == false) {
      log::error("Playground streaming: failed to initialize WebRTC session");
      server.shutdown();
      app.cleanup();
      return 1;
    }

    MediaFoundationH264Encoder encoder = {};
    MediaFoundationH264EncoderConfig encoder_config = {
      .width = options.width,
      .height = options.height,
      .fps = 60u,
      .target_bitrate = session_config.bitrate,
    };
    if (encoder.init(encoder_config) == false) {
      log::error("Playground streaming: failed to initialize H.264 encoder");
      session.shutdown();
      server.shutdown();
      app.cleanup();
      return 1;
    }

    server.set_offer_handler([&session](std::string& response_json) {
      return session.create_offer_response_json(response_json);
    });
    server.set_answer_handler([&session](const std::string& answer_json, std::string& response_json) {
      return session.accept_answer_from_json(answer_json, response_json);
    });

    log::info("Playground streaming: open %s", server.base_url().c_str());
    log::info("Playground streaming: WebRTC video at %ux%u, target bitrate %.1f Mbps, stop with Ctrl+C", options.width, options.height,
      static_cast<double>(session_config.bitrate) / (1024.0 * 1024.0));

    std::vector<BrowserInputEvent> browser_events = {};
    std::vector<uint8_t> frame_bgra = {};
    std::vector<H264EncodedFrame> encoded_frames = {};
    uint64_t frame_index = 0u;
    auto next_frame_time = std::chrono::steady_clock::now();

    while (server.initialized()) {
      server.poll();
      session.drain_input_events(browser_events);
      for (const BrowserInputEvent& event : browser_events) {
        app.process_browser_input_event(event);
      }
      if (session.consume_keyframe_request()) {
        encoder.request_keyframe();
      }

      app.frame_headless();

      uint32_t frame_width = 0u;
      uint32_t frame_height = 0u;
      if (app.capture_headless_frame_bgra(frame_bgra, frame_width, frame_height)) {
        const uint64_t timestamp_us = (frame_index * 1'000'000ull) / 60ull;
        if (encoder.encode_bgra_frame(frame_bgra.data(), frame_bgra.size(), timestamp_us, encoded_frames)) {
          for (const H264EncodedFrame& frame : encoded_frames) {
            session.push_h264_access_unit(frame.data.data(), frame.data.size(), frame.timestamp_us, frame.keyframe);
          }
        }
      }

      frame_index += 1u;
      ETX_END_PROFILER_FRAME();

      next_frame_time += std::chrono::milliseconds(16);
      const auto current_time = std::chrono::steady_clock::now();
      if (current_time < next_frame_time) {
        std::this_thread::sleep_until(next_frame_time);
      } else {
        next_frame_time = current_time;
      }
    }

    encoder.shutdown();
    session.shutdown();
    server.shutdown();
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
