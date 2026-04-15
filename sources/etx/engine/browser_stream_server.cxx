#include "browser_stream_server.hxx"

#include <etx/core/log.hxx>

#include <algorithm>
#include <array>
#include <charconv>
#include <cstring>
#include <utility>

#if ETX_PLATFORM_WINDOWS
# if !defined(WIN32_LEAN_AND_MEAN)
#   define WIN32_LEAN_AND_MEAN 1
# endif
# include <winsock2.h>
# include <ws2tcpip.h>
# pragma comment(lib, "ws2_32.lib")
#endif

namespace etx {
namespace {

#if ETX_PLATFORM_WINDOWS
constexpr uint64_t k_invalid_socket = static_cast<uint64_t>(INVALID_SOCKET);
#else
constexpr uint64_t k_invalid_socket = ~0ull;
#endif

constexpr uint32_t k_max_pending_input_events = 2048u;
constexpr uint32_t k_max_header_bytes = 64u * 1024u;
constexpr uint32_t k_max_request_bytes = 1024u * 1024u;

struct HttpRequest {
  std::string method = {};
  std::string path = {};
  std::string body = {};
};

const char* browser_client_html() {
  return R"html(<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>etx streaming</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #091018;
      --panel: rgba(10, 18, 28, 0.88);
      --line: #223446;
      --text: #e6eef7;
      --muted: #8da0b5;
      --accent: #5ad1ff;
    }
    html, body {
      margin: 0;
      width: 100%;
      height: 100%;
      background: radial-gradient(circle at top, #152535, var(--bg) 52%);
      color: var(--text);
      font: 14px/1.45 "Segoe UI", sans-serif;
    }
    body {
      display: grid;
      grid-template-rows: auto 1fr;
    }
    header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 16px;
      padding: 12px 16px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
      backdrop-filter: blur(16px);
    }
    .title {
      font-size: 15px;
      font-weight: 700;
      letter-spacing: 0.04em;
      text-transform: uppercase;
    }
    .meta {
      display: flex;
      gap: 16px;
      flex-wrap: wrap;
      color: var(--muted);
    }
    .viewport {
      display: grid;
      place-items: center;
      padding: 18px;
      overflow: hidden;
    }
    .shell {
      position: relative;
      border: 1px solid var(--line);
      border-radius: 14px;
      overflow: hidden;
      background: #04080c;
      box-shadow: 0 22px 80px rgba(0, 0, 0, 0.45);
    }
    video {
      display: block;
      max-width: min(100vw - 36px, 1920px);
      max-height: calc(100vh - 112px);
      background: #000;
      cursor: crosshair;
    }
    .hint {
      position: absolute;
      left: 14px;
      bottom: 14px;
      padding: 8px 10px;
      border-radius: 10px;
      background: rgba(8, 14, 20, 0.76);
      border: 1px solid rgba(255, 255, 255, 0.08);
      color: var(--muted);
      pointer-events: none;
    }
    .locked .hint {
      color: var(--accent);
    }
    button {
      appearance: none;
      border: 1px solid var(--line);
      background: #132131;
      color: var(--text);
      padding: 8px 12px;
      border-radius: 10px;
      cursor: pointer;
    }
  </style>
</head>
<body>
  <header>
    <div class="title">etx streaming</div>
    <div class="meta">
      <div id="status">starting</div>
      <div id="connection">new</div>
      <div>click video to lock pointer</div>
      <button id="reconnect" type="button">Reconnect</button>
    </div>
  </header>
  <div class="viewport">
    <div class="shell" id="shell">
      <video id="video" autoplay playsinline muted></video>
      <div class="hint">W/A/S/D/Q/E move, shift fast, ctrl slow, mouse look, wheel zoom</div>
    </div>
  </div>
  <script src="/client.js"></script>
</body>
</html>)html";
}

const char* browser_client_js() {
  return R"js((() => {
  const video = document.getElementById('video');
  const shell = document.getElementById('shell');
  const statusText = document.getElementById('status');
  const connectionText = document.getElementById('connection');
  const reconnectButton = document.getElementById('reconnect');

  let peerConnection = null;
  let inputFast = null;
  let inputControl = null;
  let reconnectTimer = null;
  let currentSessionId = '';

  function updateStatus(text) {
    statusText.textContent = text;
  }

  function updateConnection(text) {
    connectionText.textContent = text;
  }

  function clearReconnectTimer() {
    if (reconnectTimer !== null) {
      window.clearTimeout(reconnectTimer);
      reconnectTimer = null;
    }
  }

  function scheduleReconnect() {
    clearReconnectTimer();
    reconnectTimer = window.setTimeout(() => {
      startSession();
    }, 1000);
  }

  function sendInput(channel, payload) {
    if ((channel === null) || (channel.readyState !== 'open')) {
      return;
    }

    channel.send(JSON.stringify(payload));
  }

  function sendFastInput(payload) {
    sendInput(inputFast, payload);
  }

  function sendControlInput(payload) {
    sendInput(inputControl, payload);
  }

  async function waitForGatheringComplete(pc) {
    if (pc.iceGatheringState === 'complete') {
      return;
    }

    await new Promise((resolve) => {
      const onState = () => {
        if (pc.iceGatheringState === 'complete') {
          pc.removeEventListener('icegatheringstatechange', onState);
          resolve();
        }
      };
      pc.addEventListener('icegatheringstatechange', onState);
    });
  }

  function resetPeerConnection() {
    inputFast = null;
    inputControl = null;
    currentSessionId = '';

    if (peerConnection !== null) {
      peerConnection.getSenders().forEach((sender) => {
        if (sender.track !== null) {
          sender.track.stop();
        }
      });
      peerConnection.close();
      peerConnection = null;
    }
  }

  function installPeerConnectionCallbacks(pc) {
    pc.addEventListener('connectionstatechange', () => {
      updateConnection(pc.connectionState);
      if ((pc.connectionState === 'failed') || (pc.connectionState === 'disconnected') || (pc.connectionState === 'closed')) {
        updateStatus('reconnecting');
        scheduleReconnect();
      }
    });

    pc.ontrack = (event) => {
      if ((event.streams !== undefined) && (event.streams.length > 0)) {
        video.srcObject = event.streams[0];
      } else {
        const mediaStream = new MediaStream([event.track]);
        video.srcObject = mediaStream;
      }
      updateStatus('streaming');
    };

    pc.ondatachannel = (event) => {
      const channel = event.channel;
      if (channel.label === 'input-fast') {
        inputFast = channel;
      } else if (channel.label === 'input-control') {
        inputControl = channel;
      }
    };
  }

  async function startSession() {
    clearReconnectTimer();
    resetPeerConnection();
    updateStatus('requesting offer');
    updateConnection('new');

    const offerResponse = await fetch('/webrtc/offer', {
      method: 'POST',
      cache: 'no-store',
    });
    if (offerResponse.ok === false) {
      updateStatus('offer failed');
      scheduleReconnect();
      return;
    }

    const offer = await offerResponse.json();
    currentSessionId = offer.session_id;

    const pc = new RTCPeerConnection({
      bundlePolicy: 'max-bundle',
    });
    peerConnection = pc;
    installPeerConnectionCallbacks(pc);

    updateStatus('creating answer');
    await pc.setRemoteDescription({
      type: 'offer',
      sdp: offer.sdp,
    });
    await pc.setLocalDescription(await pc.createAnswer());
    await waitForGatheringComplete(pc);

    const answerResponse = await fetch('/webrtc/answer', {
      method: 'POST',
      cache: 'no-store',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        session_id: currentSessionId,
        type: pc.localDescription.type,
        sdp: pc.localDescription.sdp,
      }),
    });

    if (answerResponse.ok === false) {
      updateStatus('answer failed');
      scheduleReconnect();
      return;
    }

    updateStatus('connecting');
  }

  reconnectButton.addEventListener('click', () => {
    startSession();
  });

  video.addEventListener('click', () => {
    if (document.pointerLockElement !== video) {
      video.requestPointerLock();
    }
  });

  document.addEventListener('pointerlockchange', () => {
    const locked = (document.pointerLockElement === video);
    shell.classList.toggle('locked', locked);
    sendControlInput({
      type: 'focus',
      focused: locked,
    });
  });

  document.addEventListener('visibilitychange', () => {
    sendControlInput({
      type: 'focus',
      focused: (document.hidden === false),
    });
  });

  window.addEventListener('blur', () => {
    sendControlInput({
      type: 'focus',
      focused: false,
    });
  });

  video.addEventListener('mousemove', (event) => {
    if (document.pointerLockElement !== video) {
      return;
    }

    sendFastInput({
      type: 'mouse_move',
      dx: event.movementX,
      dy: event.movementY,
    });
  });

  video.addEventListener('mousedown', (event) => {
    sendControlInput({
      type: 'mouse_button',
      button: event.button,
      pressed: true,
    });
  });

  window.addEventListener('mouseup', (event) => {
    sendControlInput({
      type: 'mouse_button',
      button: event.button,
      pressed: false,
    });
  });

  video.addEventListener('wheel', (event) => {
    event.preventDefault();
    sendFastInput({
      type: 'wheel',
      delta_y: event.deltaY,
    });
  }, { passive: false });

  window.addEventListener('keydown', (event) => {
    const code = event.code;
    if ((code !== 'KeyW') && (code !== 'KeyA') && (code !== 'KeyS') && (code !== 'KeyD') && (code !== 'KeyQ') && (code !== 'KeyE') &&
        (code !== 'ShiftLeft') && (code !== 'ShiftRight') && (code !== 'ControlLeft') && (code !== 'ControlRight')) {
      return;
    }

    event.preventDefault();
    sendControlInput({
      type: 'key',
      code,
      pressed: true,
    });
  });

  window.addEventListener('keyup', (event) => {
    const code = event.code;
    if ((code !== 'KeyW') && (code !== 'KeyA') && (code !== 'KeyS') && (code !== 'KeyD') && (code !== 'KeyQ') && (code !== 'KeyE') &&
        (code !== 'ShiftLeft') && (code !== 'ShiftRight') && (code !== 'ControlLeft') && (code !== 'ControlRight')) {
      return;
    }

    event.preventDefault();
    sendControlInput({
      type: 'key',
      code,
      pressed: false,
    });
  });

  startSession();
})(); )js";
}

std::string to_lower_ascii(std::string value) {
  for (char& c : value) {
    if ((c >= 'A') && (c <= 'Z')) {
      c = static_cast<char>(c - 'A' + 'a');
    }
  }
  return value;
}

std::string trim_ascii(std::string value) {
  size_t begin = 0u;
  while ((begin < value.size()) && ((value[begin] == ' ') || (value[begin] == '\t'))) {
    begin += 1u;
  }

  size_t end = value.size();
  while ((end > begin) && ((value[end - 1u] == ' ') || (value[end - 1u] == '\t') || (value[end - 1u] == '\r') || (value[end - 1u] == '\n'))) {
    end -= 1u;
  }

  return value.substr(begin, end - begin);
}

std::string normalize_path(std::string path) {
  const size_t query_position = path.find('?');
  if (query_position != std::string::npos) {
    path.resize(query_position);
  }
  return path;
}

#if ETX_PLATFORM_WINDOWS
SOCKET to_socket(uint64_t value) {
  return static_cast<SOCKET>(value);
}

bool read_socket_request(SOCKET socket_handle, HttpRequest& request) {
  std::string raw_request = {};
  raw_request.reserve(4096u);

  std::array<char, 4096> buffer = {};
  size_t header_end = std::string::npos;
  size_t content_length = 0u;

  while (raw_request.size() < k_max_request_bytes) {
    const int received = recv(socket_handle, buffer.data(), static_cast<int>(buffer.size()), 0);
    if (received == 0) {
      break;
    }
    if (received == SOCKET_ERROR) {
      const int error_code = WSAGetLastError();
      if (error_code == WSAETIMEDOUT) {
        break;
      }
      return false;
    }

    raw_request.append(buffer.data(), static_cast<size_t>(received));
    if ((header_end == std::string::npos) && (raw_request.size() <= k_max_header_bytes)) {
      header_end = raw_request.find("\r\n\r\n");
      if (header_end != std::string::npos) {
        const std::string headers = raw_request.substr(0u, header_end + 4u);
        size_t line_begin = 0u;
        while (line_begin < headers.size()) {
          const size_t line_end = headers.find("\r\n", line_begin);
          if (line_end == std::string::npos) {
            break;
          }

          const std::string line = headers.substr(line_begin, line_end - line_begin);
          const size_t separator = line.find(':');
          if (separator != std::string::npos) {
            const std::string header_name = to_lower_ascii(trim_ascii(line.substr(0u, separator)));
            if (header_name == "content-length") {
              const std::string value = trim_ascii(line.substr(separator + 1u));
              const char* begin = value.c_str();
              const char* end = begin + value.size();
              uint32_t parsed_length = 0u;
              const std::from_chars_result parse_result = std::from_chars(begin, end, parsed_length);
              if ((parse_result.ec == std::errc()) && (parse_result.ptr == end)) {
                content_length = parsed_length;
              }
            }
          }

          line_begin = line_end + 2u;
        }
      }
    }

    if ((header_end != std::string::npos) && (raw_request.size() >= (header_end + 4u + content_length))) {
      break;
    }
  }

  if (header_end == std::string::npos) {
    return false;
  }

  const std::string_view header_view(raw_request.data(), header_end);
  const size_t request_line_end = header_view.find("\r\n");
  if (request_line_end == std::string_view::npos) {
    return false;
  }

  const std::string_view request_line = header_view.substr(0u, request_line_end);
  const size_t method_separator = request_line.find(' ');
  if (method_separator == std::string_view::npos) {
    return false;
  }

  const size_t path_separator = request_line.find(' ', method_separator + 1u);
  if (path_separator == std::string_view::npos) {
    return false;
  }

  request.method.assign(request_line.substr(0u, method_separator));
  request.path.assign(request_line.substr(method_separator + 1u, path_separator - method_separator - 1u));
  request.body.clear();

  const size_t body_offset = header_end + 4u;
  if (body_offset < raw_request.size()) {
    request.body.assign(raw_request.data() + body_offset, raw_request.size() - body_offset);
  }

  return true;
}

bool send_socket_bytes(SOCKET socket_handle, const char* data, size_t data_size) {
  size_t sent_total = 0u;
  while (sent_total < data_size) {
    const int sent = send(socket_handle, data + sent_total, static_cast<int>(data_size - sent_total), 0);
    if (sent == SOCKET_ERROR) {
      return false;
    }
    if (sent == 0) {
      return false;
    }
    sent_total += static_cast<size_t>(sent);
  }

  return true;
}

bool send_http_response(SOCKET socket_handle, uint32_t status_code, const char* status_text, const char* content_type, const void* body_data, size_t body_size) {
  std::string response_header = {};
  response_header.reserve(256u);
  response_header += "HTTP/1.1 ";
  response_header += std::to_string(status_code);
  response_header += " ";
  response_header += status_text;
  response_header += "\r\n";
  response_header += "Connection: close\r\n";
  response_header += "Cache-Control: no-store\r\n";
  response_header += "Access-Control-Allow-Origin: *\r\n";
  if (content_type != nullptr) {
    response_header += "Content-Type: ";
    response_header += content_type;
    response_header += "\r\n";
  }
  response_header += "Content-Length: ";
  response_header += std::to_string(body_size);
  response_header += "\r\n\r\n";

  if (send_socket_bytes(socket_handle, response_header.data(), response_header.size()) == false) {
    return false;
  }

  if ((body_data != nullptr) && (body_size > 0u)) {
    return send_socket_bytes(socket_handle, reinterpret_cast<const char*>(body_data), body_size);
  }

  return true;
}
#endif

}  // namespace

void BrowserStreamServer::set_offer_handler(const std::function<bool(std::string&)>& handler) {
  _offer_handler = handler;
}

void BrowserStreamServer::set_answer_handler(const std::function<bool(const std::string&, std::string&)>& handler) {
  _answer_handler = handler;
}

void BrowserStreamServer::enqueue_input_event(const BrowserInputEvent& event) {
  if (_pending_input_events.size() >= k_max_pending_input_events) {
    const size_t remove_count = _pending_input_events.size() / 2u;
    _pending_input_events.erase(_pending_input_events.begin(), _pending_input_events.begin() + remove_count);
  }

  _pending_input_events.push_back(event);
}

bool BrowserStreamServer::init(const BrowserStreamServerConfig& config) {
  shutdown();

  _page_title = (config.page_title != nullptr) ? config.page_title : "etx browser stream";

#if ETX_PLATFORM_WINDOWS
  WSADATA wsa_data = {};
  if (WSAStartup(MAKEWORD(2, 2), &wsa_data) != 0) {
    log::error("Browser stream server: WSAStartup failed");
    return false;
  }
  _wsa_started = true;

  const SOCKET listen_socket = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
  if (listen_socket == INVALID_SOCKET) {
    log::error("Browser stream server: failed to create socket");
    shutdown();
    return false;
  }

  const BOOL reuse_address = TRUE;
  setsockopt(listen_socket, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<const char*>(&reuse_address), sizeof(reuse_address));

  sockaddr_in address = {};
  address.sin_family = AF_INET;
  address.sin_port = htons(config.port);
  address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  if (bind(listen_socket, reinterpret_cast<const sockaddr*>(&address), sizeof(address)) == SOCKET_ERROR) {
    log::error("Browser stream server: bind failed on port %u", static_cast<uint32_t>(config.port));
    closesocket(listen_socket);
    shutdown();
    return false;
  }

  if (listen(listen_socket, 8) == SOCKET_ERROR) {
    log::error("Browser stream server: listen failed");
    closesocket(listen_socket);
    shutdown();
    return false;
  }

  u_long non_blocking = 1u;
  if (ioctlsocket(listen_socket, FIONBIO, &non_blocking) == SOCKET_ERROR) {
    log::error("Browser stream server: failed to enable non-blocking mode");
    closesocket(listen_socket);
    shutdown();
    return false;
  }

  _port = config.port;
  _listen_socket = static_cast<uint64_t>(listen_socket);
  _base_url = "http://127.0.0.1:" + std::to_string(_port) + "/";
  _initialized = true;
  return true;
#else
  (void) config;
  log::error("Browser stream server: this build only supports Windows for now");
  return false;
#endif
}

void BrowserStreamServer::shutdown() {
#if ETX_PLATFORM_WINDOWS
  if (_listen_socket != k_invalid_socket) {
    closesocket(to_socket(_listen_socket));
  }
#endif

  _port = 0u;
  _listen_socket = k_invalid_socket;
  _initialized = false;
  _offer_handler = {};
  _answer_handler = {};
  _frame_png.clear();
  _frame_width = 0u;
  _frame_height = 0u;
  _frame_index = 0u;
  _pending_input_events.clear();
  _page_title.clear();
  _base_url.clear();

#if ETX_PLATFORM_WINDOWS
  if (_wsa_started) {
    WSACleanup();
    _wsa_started = false;
  }
#else
  _wsa_started = false;
#endif
}

void BrowserStreamServer::poll() {
  if (_initialized == false) {
    return;
  }

#if ETX_PLATFORM_WINDOWS
  const SOCKET listen_socket = to_socket(_listen_socket);
  for (uint32_t handled_socket_count = 0u; handled_socket_count < 8u; ++handled_socket_count) {
    sockaddr_in client_address = {};
    int client_length = sizeof(client_address);
    const SOCKET client_socket = accept(listen_socket, reinterpret_cast<sockaddr*>(&client_address), &client_length);
    if (client_socket == INVALID_SOCKET) {
      const int error_code = WSAGetLastError();
      if ((error_code == WSAEWOULDBLOCK) || (error_code == WSAECONNRESET)) {
        return;
      }
      return;
    }

    const DWORD timeout_ms = 50u;
    setsockopt(client_socket, SOL_SOCKET, SO_RCVTIMEO, reinterpret_cast<const char*>(&timeout_ms), sizeof(timeout_ms));
    setsockopt(client_socket, SOL_SOCKET, SO_SNDTIMEO, reinterpret_cast<const char*>(&timeout_ms), sizeof(timeout_ms));

    HttpRequest request = {};
    if (read_socket_request(client_socket, request)) {
      const std::string normalized_path = normalize_path(request.path);
      bool response_sent = false;

      if ((request.method == "GET") && (normalized_path == "/")) {
        const char* html = browser_client_html();
        response_sent = send_http_response(client_socket, 200u, "OK", "text/html; charset=utf-8", html, std::strlen(html));
      } else if ((request.method == "GET") && (normalized_path == "/client.js")) {
        const char* js = browser_client_js();
        response_sent = send_http_response(client_socket, 200u, "OK", "application/javascript; charset=utf-8", js, std::strlen(js));
      } else if ((request.method == "GET") && (normalized_path == "/health")) {
        static constexpr const char health_json[] = "{\"ok\":true}";
        response_sent = send_http_response(client_socket, 200u, "OK", "application/json; charset=utf-8", health_json, sizeof(health_json) - 1u);
      } else if ((request.method == "POST") && (normalized_path == "/webrtc/offer")) {
        std::string response_body = {};
        if ((_offer_handler) && _offer_handler(response_body)) {
          response_sent = send_http_response(client_socket, 200u, "OK", "application/json; charset=utf-8", response_body.data(), response_body.size());
        } else {
          static constexpr const char error_json[] = "{\"ok\":false,\"error\":\"offer_failed\"}";
          response_sent = send_http_response(client_socket, 500u, "Internal Server Error", "application/json; charset=utf-8", error_json,
            sizeof(error_json) - 1u);
        }
      } else if ((request.method == "POST") && (normalized_path == "/webrtc/answer")) {
        std::string response_body = {};
        if ((_answer_handler) && _answer_handler(request.body, response_body)) {
          response_sent = send_http_response(client_socket, 200u, "OK", "application/json; charset=utf-8", response_body.data(), response_body.size());
        } else {
          static constexpr const char error_json[] = "{\"ok\":false,\"error\":\"answer_failed\"}";
          response_sent = send_http_response(client_socket, 400u, "Bad Request", "application/json; charset=utf-8", error_json, sizeof(error_json) - 1u);
        }
      } else if ((request.method == "GET") && ((normalized_path == "/frame.png") || (normalized_path == "/frame"))) {
        if (_frame_png.empty()) {
          response_sent = send_http_response(client_socket, 204u, "No Content", "text/plain; charset=utf-8", nullptr, 0u);
        } else {
          response_sent = send_http_response(client_socket, 200u, "OK", "image/png", _frame_png.data(), _frame_png.size());
        }
      } else {
        static constexpr const char not_found_body[] = "Not found";
        response_sent = send_http_response(client_socket, 404u, "Not Found", "text/plain; charset=utf-8", not_found_body, sizeof(not_found_body) - 1u);
      }

      (void) response_sent;
    }

    closesocket(client_socket);
  }
#endif
}

void BrowserStreamServer::set_frame_png(const uint8_t* data, size_t data_size, uint32_t width, uint32_t height, uint64_t frame_index) {
  _frame_width = width;
  _frame_height = height;
  _frame_index = frame_index;

  if ((data == nullptr) || (data_size == 0u)) {
    _frame_png.clear();
    return;
  }

  _frame_png.assign(data, data + data_size);
}

void BrowserStreamServer::drain_input_events(std::vector<BrowserInputEvent>& output) {
  output = std::move(_pending_input_events);
  _pending_input_events.clear();
}

}  // namespace etx
