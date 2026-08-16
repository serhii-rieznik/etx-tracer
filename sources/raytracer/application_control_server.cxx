#include "application_control_server.hxx"
#include "scene_dependencies.hxx"

#include <etx/core/log.hxx>

#include <json.hpp>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cctype>
#include <charconv>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <random>
#include <sstream>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#if ETX_PLATFORM_WINDOWS
# if !defined(WIN32_LEAN_AND_MEAN)
#  define WIN32_LEAN_AND_MEAN 1
# endif
# include <winsock2.h>
# include <ws2tcpip.h>
# if defined(_MSC_VER)
#  pragma comment(lib, "ws2_32.lib")
# endif
#else
# include <arpa/inet.h>
# include <fcntl.h>
# include <netinet/in.h>
# include <sys/socket.h>
# include <unistd.h>
#endif

namespace etx {

struct ApplicationUploadFile {
  std::string relative_path = {};
  uint64_t size = 0u;
  uint64_t received = 0u;
  bool required = false;
  bool inspected = false;
  bool ignore_obj_material_library = false;
};

struct ApplicationUploadSession {
  std::string id = {};
  std::filesystem::path directory = {};
  std::string entry = {};
  std::vector<ApplicationUploadFile> files = {};
  std::vector<std::string> unavailable_references = {};
  uint64_t total_size = 0u;
  bool committed = false;
  bool dependencies_resolved = false;
  bool became_active = false;
  uint64_t load_command_id = 0u;
  std::chrono::steady_clock::time_point last_activity = std::chrono::steady_clock::now();
};

struct ApplicationUploadState {
  std::filesystem::path directory = {};
  std::unordered_map<std::string, ApplicationUploadSession> sessions = {};
  uint64_t total_reserved_size = 0u;
  std::mt19937_64 random{std::random_device{}()};
  std::chrono::steady_clock::time_point next_maintenance = {};
};

namespace {

using Json = nlohmann::json;

#if ETX_PLATFORM_WINDOWS
using NativeSocket = SOCKET;
constexpr uint64_t kInvalidSocket = static_cast<uint64_t>(INVALID_SOCKET);
#else
using NativeSocket = int;
constexpr uint64_t kInvalidSocket = ~0ull;
#endif

constexpr size_t kMaxRequestSize = 1024u * 1024u;
constexpr size_t kMaxUploadChunkSize = 512u * 1024u;
constexpr size_t kMaxUploadFileCount = 8192u;
constexpr size_t kMaxUploadSessionCount = 16u;
constexpr size_t kMaxUploadPathLength = 1024u;
constexpr size_t kRetainedCommandResultLimit = 256u;
constexpr uint64_t kMaxUploadFileSize = 8ull * 1024ull * 1024ull * 1024ull;
constexpr uint64_t kMaxUploadSessionSize = 32ull * 1024ull * 1024ull * 1024ull;
constexpr uint64_t kMaxReservedUploadSize = 64ull * 1024ull * 1024ull * 1024ull;

NativeSocket native_socket(uint64_t socket) {
  return static_cast<NativeSocket>(socket);
}

uint64_t stored_socket(NativeSocket socket) {
  return static_cast<uint64_t>(socket);
}

void close_socket(NativeSocket socket) {
#if ETX_PLATFORM_WINDOWS
  closesocket(socket);
#else
  close(socket);
#endif
}

bool would_block() {
#if ETX_PLATFORM_WINDOWS
  const int error = WSAGetLastError();
  return (error == WSAEWOULDBLOCK) || (error == WSAETIMEDOUT);
#else
  return (errno == EAGAIN) || (errno == EWOULDBLOCK);
#endif
}

bool interrupted() {
#if ETX_PLATFORM_WINDOWS
  return WSAGetLastError() == WSAEINTR;
#else
  return errno == EINTR;
#endif
}

bool ascii_iequals(std::string_view lhs, std::string_view rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }
  for (size_t index = 0u; index < lhs.size(); ++index) {
    const char left = (lhs[index] >= 'A') && (lhs[index] <= 'Z') ? static_cast<char>(lhs[index] - 'A' + 'a') : lhs[index];
    const char right = (rhs[index] >= 'A') && (rhs[index] <= 'Z') ? static_cast<char>(rhs[index] - 'A' + 'a') : rhs[index];
    if (left != right) {
      return false;
    }
  }
  return true;
}

std::string upload_id(ApplicationUploadState& state) {
  std::ostringstream stream = {};
  stream << std::hex << std::setfill('0') << std::setw(16) << state.random() << std::setw(16) << state.random();
  return stream.str();
}

bool prepare_upload_directory(ApplicationUploadState& state) {
  std::error_code error = {};
  const std::filesystem::path temporary = std::filesystem::temp_directory_path(error);
  if (error) {
    return false;
  }
  for (uint32_t attempt = 0u; attempt < 16u; ++attempt) {
    const std::filesystem::path candidate = temporary / ("etx-tracer-uploads-" + upload_id(state));
    if (std::filesystem::create_directory(candidate, error)) {
      std::filesystem::permissions(candidate, std::filesystem::perms::owner_all, std::filesystem::perm_options::replace, error);
      if (error) {
        std::error_code remove_error = {};
        std::filesystem::remove_all(candidate, remove_error);
        return false;
      }
      state.directory = candidate;
      return true;
    }
    if (error) {
      error.clear();
    }
  }
  return false;
}

void clear_uploads(ApplicationUploadState& state) {
  if (!state.directory.empty()) {
    std::error_code error = {};
    std::filesystem::remove_all(state.directory, error);
    if (error) {
      log::warning("Failed to remove application upload directory %s: %s", state.directory.string().c_str(), error.message().c_str());
    }
  }
  state.directory.clear();
  state.sessions.clear();
  state.total_reserved_size = 0u;
  state.next_maintenance = {};
}

std::string upload_path_key(std::string_view path) {
  std::string key(path);
  std::transform(key.begin(), key.end(), key.begin(), [](unsigned char character) {
    return static_cast<char>(std::tolower(character));
  });
  return key;
}

bool normalize_upload_path(const std::string& source, std::string& normalized) {
  normalized.clear();
  if (source.empty() || (source.size() > kMaxUploadPathLength) || (source.front() == '/') || (source.back() == '/')) {
    return false;
  }

  size_t component_begin = 0u;
  while (component_begin < source.size()) {
    const size_t separator = source.find('/', component_begin);
    const size_t component_end = separator == std::string::npos ? source.size() : separator;
    const std::string_view component(source.data() + component_begin, component_end - component_begin);
    if (component.empty() || (component == ".") || (component == "..") || (component.back() == '.') || (component.back() == ' ')) {
      return false;
    }
    for (const unsigned char character : component) {
      if ((character < 32u) || (character == '\\') || (character == ':') || (character == '<') || (character == '>') || (character == '"') || (character == '|') ||
          (character == '?') || (character == '*')) {
        return false;
      }
    }
    if (!normalized.empty()) {
      normalized.push_back('/');
    }
    normalized.append(component);
    if (separator == std::string::npos) {
      break;
    }
    component_begin = separator + 1u;
  }
  return !normalized.empty();
}

bool parse_unsigned(std::string_view text, uint64_t& value) {
  value = 0u;
  const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
  return !text.empty() && (result.ec == std::errc()) && (result.ptr == text.data() + text.size());
}

bool parse_upload_file_path(std::string_view path, std::string& id, uint64_t& file_index) {
  constexpr std::string_view prefix = "/api/uploads/";
  constexpr std::string_view files = "/files/";
  if (!path.starts_with(prefix)) {
    return false;
  }
  const size_t files_position = path.find(files, prefix.size());
  if (files_position == std::string_view::npos) {
    return false;
  }
  const std::string_view id_view = path.substr(prefix.size(), files_position - prefix.size());
  const std::string_view index_view = path.substr(files_position + files.size());
  if ((id_view.size() != 32u) || !parse_unsigned(index_view, file_index)) {
    return false;
  }
  id.assign(id_view);
  return true;
}

bool parse_upload_action_path(std::string_view path, std::string_view action, std::string& id) {
  constexpr std::string_view prefix = "/api/uploads/";
  if (!path.starts_with(prefix) || !path.ends_with(action)) {
    return false;
  }
  const size_t id_size = path.size() - prefix.size() - action.size();
  if (id_size != 32u) {
    return false;
  }
  id.assign(path.substr(prefix.size(), id_size));
  return true;
}

bool query_value(std::string_view request_path, std::string_view name, uint64_t& value) {
  const size_t query_position = request_path.find('?');
  if (query_position == std::string_view::npos) {
    return false;
  }
  const std::string_view query = request_path.substr(query_position + 1u);
  size_t begin = 0u;
  while (begin <= query.size()) {
    const size_t end = query.find('&', begin);
    const std::string_view item = query.substr(begin, end == std::string_view::npos ? query.size() - begin : end - begin);
    const size_t equals = item.find('=');
    if ((equals != std::string_view::npos) && (item.substr(0u, equals) == name)) {
      return parse_unsigned(item.substr(equals + 1u), value);
    }
    if (end == std::string_view::npos) {
      break;
    }
    begin = end + 1u;
  }
  return false;
}

const char* client_page() {
  return R"html(<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ETX Tracer</title>
  <style>
    :root {
      color-scheme: light;
      font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", sans-serif;
      font-size: 14px;
      --page: #f1f1ef;
      --panel: #fafaf8;
      --surface: #ffffff;
      --surface-hover: #f4f4f1;
      --surface-pressed: #ecece8;
      --canvas: #e8e8e5;
      --border: #d6d6d0;
      --border-strong: #b9b9b1;
      --text: #242421;
      --muted: #6c6c65;
      --faint: #8c8c84;
      --accent: #625e55;
      --accent-hover: #4f4b44;
      --accent-text: #ffffff;
      --success: #397255;
      --warning: #8a6528;
      --danger: #a34545;
      --focus: rgba(98, 94, 85, .24);
      --radius: 8px;
    }
    @media (prefers-color-scheme: dark) {
      :root {
        color-scheme: dark;
        --page: #1d1d1b;
        --panel: #252523;
        --surface: #2d2d2a;
        --surface-hover: #353532;
        --surface-pressed: #3b3b37;
        --canvas: #181817;
        --border: #454540;
        --border-strong: #5b5b54;
        --text: #efefe9;
        --muted: #aaa9a0;
        --faint: #85847d;
        --accent: #d1c9ba;
        --accent-hover: #e3dccf;
        --accent-text: #25231f;
        --success: #79b38f;
        --warning: #d1a75f;
        --danger: #df8585;
        --focus: rgba(209, 201, 186, .22);
      }
    }
    * { box-sizing: border-box; }
    html, body { height: 100%; }
    body { margin: 0; overflow: hidden; color: var(--text); background: var(--page); -webkit-font-smoothing: antialiased; }
    header {
      height: 52px;
      display: flex;
      align-items: center;
      gap: 16px;
      padding: 0 20px;
      border-bottom: 1px solid var(--border);
      background: var(--panel);
    }
    header strong { flex: none; font-size: 15px; font-weight: 650; letter-spacing: -.01em; }
    #status { flex: none; margin-left: auto; color: var(--muted); font-size: 13px; font-weight: 550; }
    #status[data-state="connected"] { color: var(--success); }
    #status[data-state="disconnected"] { color: var(--danger); }
    main { height: calc(100% - 52px); display: grid; grid-template-columns: 328px minmax(0, 1fr); }
    aside { min-height: 0; overflow: auto; padding: 20px; border-right: 1px solid var(--border); background: var(--panel); }
    section { min-width: 0; min-height: 0; display: flex; flex-direction: column; padding: 20px; overflow: hidden; }
    fieldset { min-width: 0; margin: 0 0 20px; padding: 0 0 20px; border: 0; border-bottom: 1px solid var(--border); }
    legend { width: 100%; padding: 0 0 12px; color: var(--text); font-size: 13px; font-weight: 650; }
    label { display: block; margin: 0 0 6px; color: var(--muted); font-size: 12px; font-weight: 550; }
    input, select, button {
      font: inherit;
      color: inherit;
      border: 1px solid var(--border-strong);
      border-radius: var(--radius);
      background: var(--surface);
      transition: background-color 140ms ease, border-color 140ms ease, color 140ms ease, opacity 140ms ease;
    }
    input, select { width: 100%; height: 36px; margin: 0 0 10px; padding: 0 10px; }
    input::placeholder { color: var(--faint); }
    input:hover, select:hover { border-color: var(--muted); }
    input:focus, select:focus { border-color: var(--accent); outline: 3px solid var(--focus); outline-offset: 0; }
    button { min-height: 36px; padding: 0 12px; font-weight: 560; cursor: pointer; }
    button:not(:disabled):hover { background: var(--surface-hover); border-color: var(--muted); }
    button:not(:disabled):active { background: var(--surface-pressed); }
    button:focus-visible, summary:focus-visible { outline: 3px solid var(--focus); outline-offset: 2px; }
    button.primary { color: var(--accent-text); background: var(--accent); border-color: var(--accent); }
    button.primary:not(:disabled):hover { background: var(--accent-hover); border-color: var(--accent-hover); }
    button.danger:not(:disabled) { color: var(--danger); }
    button:disabled { color: var(--faint); opacity: .58; cursor: default; }
    button[aria-busy="true"] { cursor: progress; }
    .row { display: flex; gap: 8px; }
    .row > * { min-width: 0; flex: 1; }
    .render-actions { display: grid; grid-template-columns: 1fr 1fr; }
    #denoise { width: 100%; margin-top: 8px; }
    .hint { margin: 8px 0 12px; color: var(--muted); font-size: 12px; line-height: 1.45; overflow-wrap: anywhere; }
    #upload-warning { color: var(--warning); }
    .hidden { display: none !important; }
    progress { display: block; width: 100%; height: 5px; margin: 12px 0; overflow: hidden; border: 0; border-radius: 3px; accent-color: var(--accent); background: var(--surface-pressed); }
    progress::-webkit-progress-bar { background: var(--surface-pressed); }
    progress::-webkit-progress-value { background: var(--accent); }
    details { margin-top: 16px; padding-top: 14px; border-top: 1px solid var(--border); }
    summary { width: fit-content; color: var(--muted); font-size: 12px; cursor: pointer; }
    details[open] summary { margin-bottom: 12px; color: var(--text); }
    details label { margin-top: 0; }
    .meta { display: grid; grid-template-columns: 76px minmax(0, 1fr); gap: 7px 12px; margin: 0; padding-bottom: 4px; font-variant-numeric: tabular-nums; }
    .meta dt { color: var(--muted); font-size: 12px; }
    .meta dd { min-width: 0; margin: 0; font-size: 12px; overflow-wrap: anywhere; }
    #scene-name { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    #message { min-height: 36px; margin-top: 14px; padding-top: 12px; border-top: 1px solid var(--border); color: var(--muted); font-size: 12px; line-height: 1.45; overflow-wrap: anywhere; }
    #message[data-tone="success"] { color: var(--success); }
    #message[data-tone="error"] { color: var(--danger); }
    .image-head { min-height: 36px; display: flex; align-items: center; gap: 12px; margin-bottom: 12px; }
    .image-head strong { font-size: 13px; font-weight: 650; }
    .image-head button { min-height: 32px; margin-left: auto; font-size: 12px; }
    .image-stage { min-height: 0; flex: 1; display: grid; place-items: center; overflow: auto; border: 1px solid var(--border); border-radius: var(--radius); background: var(--canvas); }
    .image-empty { max-width: 280px; padding: 24px; color: var(--muted); line-height: 1.5; text-align: center; }
    .image-stage.has-image .image-empty { display: none; }
    #image { display: none; max-width: 100%; max-height: 100%; object-fit: contain; }
    .image-stage.has-image #image { display: block; }
    @media (max-width: 820px) {
      body { overflow: auto; }
      header { position: sticky; top: 0; z-index: 1; padding: 0 16px; }
      main { height: auto; min-height: calc(100% - 52px); grid-template-columns: 1fr; }
      aside { overflow: visible; padding: 20px 16px; border-right: 0; border-bottom: 1px solid var(--border); }
      section { min-height: 440px; padding: 16px; overflow: visible; }
      .image-stage { min-height: 360px; }
      input, select, button { min-height: 44px; }
    }
    @media (prefers-reduced-motion: reduce) { input, select, button { transition: none; } }
  </style>
</head>
<body>
  <header><strong>ETX Tracer</strong><span id="status" data-state="connecting">Connecting…</span></header>
  <main>
    <aside aria-label="Renderer controls">
      <fieldset>
        <legend>Scene</legend>
        <input id="folder" type="file" webkitdirectory multiple hidden>
        <button id="choose-folder" type="button">Choose scene folder…</button>
        <div id="folder-summary" class="hint">No folder selected</div>
        <div id="entry-fields" class="hidden">
          <label for="entry">Scene file</label>
          <select id="entry"></select>
        </div>
        <progress id="upload-progress" class="hidden" max="1" value="0"></progress>
        <div id="upload-warning" class="hint hidden"></div>
        <div class="row upload-actions">
          <button id="upload" class="primary" type="button" disabled>Upload and load</button>
          <button id="cancel-upload" class="hidden" type="button">Cancel</button>
        </div>
        <details>
          <summary>Load from renderer host</summary>
          <label for="path">File path</label>
          <input id="path" type="text" autocomplete="off">
          <button id="load" type="button" disabled>Load scene</button>
        </details>
      </fieldset>
      <fieldset>
        <legend>Rendering</legend>
        <label for="renderer">Renderer</label>
        <select id="renderer" disabled><option value="cpu">CPU</option><option value="raster">Raster</option><option value="gpu">GPU</option></select>
        <label for="integrator">Integrator</label>
        <select id="integrator" disabled></select>
        <div class="row render-actions">
          <button id="run" class="primary" type="button" disabled>Run</button>
          <button id="finish" type="button" disabled>Finish</button>
          <button id="stop" class="danger" type="button" disabled>Stop</button>
          <button id="restart" type="button" disabled>Restart</button>
        </div>
        <button id="denoise" type="button" disabled>Denoise image</button>
      </fieldset>
      <dl class="meta">
        <dt>Scene</dt><dd id="scene-name">—</dd>
        <dt>Renderer</dt><dd id="renderer-name">—</dd>
        <dt>State</dt><dd id="run-state">—</dd>
        <dt>Samples</dt><dd id="samples">—</dd>
        <dt>Elapsed</dt><dd id="elapsed">—</dd>
      </dl>
      <div id="message" role="status" aria-live="polite"></div>
    </aside>
    <section aria-label="Rendered image">
      <div class="image-head"><strong>Output</strong><button id="refresh-image" type="button">Refresh image</button></div>
      <div id="image-stage" class="image-stage">
        <div class="image-empty">Rendered output will appear here after you load and run a scene.</div>
        <img id="image" alt="Latest rendered output" decoding="async">
      </div>
    </section>
  </main>
  <script>
    const byId = id => document.getElementById(id);
    let imageUrl = null;
    let imageRefreshPending = false;
    let lastImageKey = '';
    let currentImageKey = '';
    let currentSceneFile = '';
    let lastImageRefresh = 0;
    let folderFiles = [];
    let uploadController = null;
    let currentUploadId = '';
    let activeUploadId = '';
    let pendingUpload = null;
    let lastCommandResultId = 0;
    let latestState = null;
    let serverConnected = false;
    let resultsRefreshPending = false;
    let pollTimer = 0;
    const commandResults = new Map();
    const pendingCommands = new Map();
    const submittingCommands = new Set();
    const commandDetails = {
      load_scene: {label: 'Load scene', control: 'load', pending: 'Loading…'},
      set_renderer: {label: 'Change renderer', control: 'renderer'},
      set_integrator: {label: 'Change integrator', control: 'integrator'},
      run: {label: 'Start rendering', control: 'run', pending: 'Starting…'},
      finish: {label: 'Finish rendering', control: 'finish', pending: 'Finishing…'},
      stop: {label: 'Stop rendering', control: 'stop', pending: 'Stopping…'},
      restart: {label: 'Restart rendering', control: 'restart', pending: 'Restarting…'},
      denoise: {label: 'Denoise image', control: 'denoise', pending: 'Denoising…'}
    };

    async function responseJson(response) {
      let result = {};
      try { result = await response.json(); } catch (_) {}
      if (!response.ok) throw new Error(result.error || `Server returned ${response.status}`);
      return result;
    }

    async function fetchWithTimeout(url, options = {}, timeoutMs = 5000) {
      const controller = new AbortController();
      const sourceSignal = options.signal;
      const forwardAbort = () => controller.abort();
      if (sourceSignal) {
        if (sourceSignal.aborted) controller.abort();
        else sourceSignal.addEventListener('abort', forwardAbort, {once: true});
      }
      const timeout = setTimeout(() => controller.abort(), timeoutMs);
      try {
        return await fetch(url, {...options, signal: controller.signal});
      } finally {
        clearTimeout(timeout);
        if (sourceSignal) sourceSignal.removeEventListener('abort', forwardAbort);
      }
    }

    function setMessage(message, tone = 'info') {
      byId('message').textContent = message;
      byId('message').dataset.tone = tone;
    }

    function setConnectionStatus(message, state) {
      byId('status').textContent = message;
      byId('status').dataset.state = state;
    }

    function commandPending(type) {
      if (submittingCommands.has(type)) return true;
      for (const pending of pendingCommands.values()) {
        if (pending.type === type) return true;
      }
      return false;
    }

    function setCommandPresentation(type, pending) {
      const details = commandDetails[type];
      if (!details || !details.control) return;
      const control = byId(details.control);
      if (!control) return;
      if (pending) {
        control.disabled = true;
        control.setAttribute('aria-busy', 'true');
        if ((control.tagName === 'BUTTON') && details.pending) {
          if (!control.dataset.defaultLabel) control.dataset.defaultLabel = control.textContent;
          control.textContent = details.pending;
        }
      } else {
        control.removeAttribute('aria-busy');
        if ((control.tagName === 'BUTTON') && control.dataset.defaultLabel) {
          control.textContent = control.dataset.defaultLabel;
        }
      }
    }

    function finishPendingCommand(result) {
      const pending = pendingCommands.get(result.command_id);
      if (!pending) return false;
      pendingCommands.delete(result.command_id);
      setCommandPresentation(pending.type, commandPending(pending.type));
      if (latestState) applyState(latestState);
      setMessage(result.message || `${pending.label} ${result.success ? 'completed' : 'failed'}`, result.success ? 'success' : 'error');
      return true;
    }

    async function command(type, extra = {}) {
      if (!serverConnected) {
        setMessage('Renderer is not connected', 'error');
        return null;
      }
      if (commandPending(type)) return null;
      const details = commandDetails[type] || {label: type.replaceAll('_', ' ')};
      submittingCommands.add(type);
      setCommandPresentation(type, true);
      setMessage(`${details.label} requested…`);
      try {
        const response = await fetchWithTimeout('/api/commands', {
          method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({type, ...extra})
        }, 10000);
        const result = await responseJson(response);
        if (!result.accepted || !Number.isSafeInteger(result.command_id)) throw new Error('The renderer returned an invalid command response');
        submittingCommands.delete(type);
        pendingCommands.set(result.command_id, {type, label: details.label});
        setMessage(`${details.label} accepted · waiting for renderer`);
        const cachedResult = commandResults.get(result.command_id);
        if (cachedResult) finishPendingCommand(cachedResult);
        else updateResults();
        return result.command_id;
      } catch (error) {
        submittingCommands.delete(type);
        setCommandPresentation(type, false);
        if (latestState) applyState(latestState);
        const message = error.name === 'AbortError' ? `${details.label} timed out` : error.message;
        setMessage(message, 'error');
        return null;
      }
    }

    function formatSize(bytes) {
      if (bytes < 1024) return `${bytes} B`;
      const units = ['KB', 'MB', 'GB', 'TB'];
      let value = bytes / 1024;
      let unit = 0;
      while ((value >= 1024) && (unit < units.length - 1)) { value /= 1024; ++unit; }
      return `${value.toFixed(value >= 10 ? 1 : 2)} ${units[unit]}`;
    }

    function chooseFolder() {
      if (uploadController) return;
      byId('folder').value = '';
      byId('folder').click();
    }

    function folderSelected(event) {
      const files = Array.from(event.target.files || []);
      folderFiles = [];
      const entry = byId('entry');
      entry.replaceChildren();
      byId('entry-fields').classList.add('hidden');
      byId('upload-warning').classList.add('hidden');
      byId('upload').disabled = true;
      if (!files.length) {
        byId('folder-summary').textContent = 'No folder selected';
        return;
      }

      const firstPath = files[0].webkitRelativePath || files[0].name;
      const rootEnd = firstPath.indexOf('/');
      const rootName = rootEnd >= 0 ? firstPath.slice(0, rootEnd) : '';
      const rootPrefix = rootName ? `${rootName}/` : '';
      const stripRoot = rootPrefix && files.every(file => (file.webkitRelativePath || file.name).startsWith(rootPrefix));
      folderFiles = files.map(file => {
        const originalPath = file.webkitRelativePath || file.name;
        return {file, path: stripRoot ? originalPath.slice(rootPrefix.length) : originalPath};
      });

      const sceneExtensions = ['.etx.json', '.json', '.etx', '.obj', '.gltf', '.glb', '.wo3'];
      const sceneFiles = folderFiles.filter(item => sceneExtensions.some(extension => item.path.toLowerCase().endsWith(extension)));
      sceneFiles.sort((left, right) => {
        const leftNative = left.path.toLowerCase().endsWith('.etx.json') ? 0 : 1;
        const rightNative = right.path.toLowerCase().endsWith('.etx.json') ? 0 : 1;
        return (leftNative - rightNative) || left.path.localeCompare(right.path);
      });
      for (const item of sceneFiles) {
        const option = document.createElement('option');
        option.value = item.path;
        option.textContent = item.path;
        entry.append(option);
      }
      const totalSize = folderFiles.reduce((total, item) => total + item.file.size, 0);
      byId('folder-summary').textContent = `${rootName || 'Selected folder'} · ${folderFiles.length} files · ${formatSize(totalSize)}`;
      if (sceneFiles.length) {
        byId('entry-fields').classList.toggle('hidden', sceneFiles.length === 1);
        byId('upload').disabled = !serverConnected || pendingUpload !== null;
      } else {
        byId('folder-summary').textContent += ' · No supported scene found';
      }
    }

    async function deleteUpload(id) {
      if (!id) return;
      try { await fetch(`/api/uploads/${id}`, {method: 'DELETE'}); } catch (_) {}
    }

    function processUploadResult(result) {
      if (!pendingUpload || (result.command_id !== pendingUpload.commandId)) return;
      const completed = pendingUpload;
      pendingUpload = null;
      if (result.success) {
        activeUploadId = completed.id;
        if (completed.previousId && (completed.previousId !== completed.id)) deleteUpload(completed.previousId);
      } else {
        deleteUpload(completed.id);
      }
      byId('upload').disabled = !serverConnected || !folderFiles.length || !byId('entry').value;
    }

    async function uploadFolder() {
      if (!serverConnected || uploadController || pendingUpload || !folderFiles.length || !byId('entry').value) return;
      uploadController = new AbortController();
      const signal = uploadController.signal;
      let uploadedSize = 0;
      let requiredSize = 1;
      let committed = false;
      byId('choose-folder').disabled = true;
      byId('upload').disabled = true;
      byId('cancel-upload').classList.remove('hidden');
      byId('upload-progress').classList.remove('hidden');
      byId('upload-progress').max = 1;
      byId('upload-progress').value = 0;
      byId('upload').textContent = 'Uploading…';
      byId('upload').setAttribute('aria-busy', 'true');
      setMessage('Preparing upload…');
      byId('upload-warning').classList.add('hidden');
      try {
        const manifest = {
          entry: byId('entry').value,
          files: folderFiles.map(item => ({path: item.path, size: item.file.size}))
        };
        const created = await responseJson(await fetch('/api/uploads', {
          method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(manifest), signal
        }));
        currentUploadId = created.upload_id;
        const chunkSize = created.chunk_size;
        let requestedFiles = created.files;
        let unavailableReferences = [];
        requiredSize = Math.max(created.required_size, 1);
        while (true) {
          byId('upload-progress').max = requiredSize;
          for (const requested of requestedFiles) {
            const item = folderFiles[requested.index];
            if (!item || (item.path !== requested.path)) throw new Error('Upload manifest no longer matches the selected folder');
            for (let offset = requested.received; offset < item.file.size; offset += chunkSize) {
              const chunk = item.file.slice(offset, Math.min(offset + chunkSize, item.file.size));
              await responseJson(await fetch(`/api/uploads/${currentUploadId}/files/${requested.index}?offset=${offset}`, {
                method: 'PUT', headers: {'Content-Type': 'application/octet-stream'}, body: chunk, signal
              }));
              uploadedSize += chunk.size;
              byId('upload-progress').value = uploadedSize;
              setMessage(`Uploading required files · ${formatSize(uploadedSize)} of ${formatSize(requiredSize)}`);
            }
          }
          setMessage('Inspecting scene dependencies…');
          const resolved = await responseJson(await fetch(`/api/uploads/${currentUploadId}/resolve`, {method: 'POST', signal}));
          requiredSize = Math.max(resolved.required_size, 1);
          unavailableReferences = resolved.unavailable_references || [];
          if (resolved.complete) break;
          requestedFiles = resolved.files;
          if (!requestedFiles.length) throw new Error('The renderer could not complete dependency discovery');
        }
        setMessage('Loading scene…');
        const accepted = await responseJson(await fetch(`/api/uploads/${currentUploadId}/commit`, {method: 'POST', signal}));
        if (!accepted.accepted || !Number.isSafeInteger(accepted.command_id)) throw new Error('The renderer returned an invalid load response');
        committed = true;
        pendingUpload = {id: currentUploadId, commandId: accepted.command_id, previousId: activeUploadId};
        pendingCommands.set(accepted.command_id, {type: 'upload_scene', label: 'Load uploaded scene'});
        const cachedResult = commandResults.get(accepted.command_id);
        const warning = unavailableReferences.length ? ` · ${unavailableReferences.length} unavailable reference${unavailableReferences.length === 1 ? '' : 's'}` : '';
        if (unavailableReferences.length) {
          const visible = unavailableReferences.slice(0, 3).join('; ');
          const remainder = unavailableReferences.length > 3 ? `; and ${unavailableReferences.length - 3} more` : '';
          byId('upload-warning').textContent = `Not found in the selected folder: ${visible}${remainder}`;
          byId('upload-warning').classList.remove('hidden');
        }
        setMessage(`Upload complete · waiting for renderer${warning}`);
        if (cachedResult) {
          processUploadResult(cachedResult);
          finishPendingCommand(cachedResult);
        }
      } catch (error) {
        if (currentUploadId && !committed) deleteUpload(currentUploadId);
        setMessage(error.name === 'AbortError' ? 'Upload cancelled' : error.message, error.name === 'AbortError' ? 'info' : 'error');
      } finally {
        currentUploadId = '';
        uploadController = null;
        byId('upload').textContent = 'Upload and load';
        byId('upload').removeAttribute('aria-busy');
        byId('choose-folder').disabled = false;
        byId('upload').disabled = !serverConnected || pendingUpload !== null || !folderFiles.length || !byId('entry').value;
        byId('cancel-upload').classList.add('hidden');
        byId('upload-progress').classList.add('hidden');
      }
    }

    function cancelUpload() {
      if (uploadController) uploadController.abort();
    }
    function sceneDisplayName(path) {
      const normalized = String(path || '').replace(/\\/g, '/');
      return normalized.slice(normalized.lastIndexOf('/') + 1) || '—';
    }

    function clearDisplayedImage() {
      lastImageKey = '';
      if (imageUrl) URL.revokeObjectURL(imageUrl);
      imageUrl = null;
      byId('image').removeAttribute('src');
      byId('image-stage').classList.remove('has-image');
    }

    function applyState(state) {
      byId('scene-name').textContent = sceneDisplayName(state.scene_file);
      byId('renderer-name').textContent = state.renderer_name;
      byId('run-state').textContent = state.run_state;
      byId('samples').textContent = state.runtime.valid ? `${state.runtime.completed_samples} / ${state.runtime.target_samples}` : '—';
      byId('elapsed').textContent = state.runtime.valid ? `${state.runtime.elapsed_seconds.toFixed(1)} s` : '—';
      const renderer = byId('renderer');
      if (!commandPending('set_renderer')) renderer.value = state.renderer;
      renderer.disabled = !serverConnected || commandPending('set_renderer');
      renderer.querySelector('[value="gpu"]').disabled = !state.gpu_renderer_available;
      const integrator = byId('integrator');
      const values = state.integrators.map(item => `${item.value}:${item.enabled ? 1 : 0}`).join(',');
      if (integrator.dataset.values !== values) {
        integrator.replaceChildren(...state.integrators.map(item => {
          const option = document.createElement('option'); option.value = item.value; option.textContent = item.name; option.disabled = !item.enabled; return option;
        }));
        integrator.dataset.values = values;
      }
      if (!commandPending('set_integrator')) integrator.value = String(state.integrator_value);
      integrator.disabled = !serverConnected || commandPending('set_integrator');
      byId('load').disabled = !serverConnected || commandPending('load_scene');
      byId('upload').disabled = !serverConnected || pendingUpload !== null || !folderFiles.length || !byId('entry').value;
      byId('run').disabled = !serverConnected || !state.controls.can_run || commandPending('run');
      byId('finish').disabled = !serverConnected || !state.controls.can_finish || commandPending('finish');
      byId('stop').disabled = !serverConnected || !state.controls.can_stop || commandPending('stop');
      byId('restart').disabled = !serverConnected || !state.controls.can_restart || commandPending('restart');
      byId('denoise').disabled = !serverConnected || !state.can_denoise || commandPending('denoise');

      if (state.scene_file !== currentSceneFile) {
        currentSceneFile = state.scene_file;
        clearDisplayedImage();
      }
      const imageKey = state.scene_loaded ? JSON.stringify([
        state.scene_file, state.renderer, state.run_state, state.runtime.completed_samples,
        state.view.exposure, state.view.view_layer, state.view.output_view, state.view.display_transform
      ]) : '';
      currentImageKey = imageKey;
      if (!imageKey) {
        clearDisplayedImage();
        return;
      }
      const activelyRendering = (state.run_state === 'running') || (state.run_state === 'finishing');
      const refreshInterval = activelyRendering ? 1000 : 5000;
      const imageChanged = imageKey !== lastImageKey;
      if ((imageChanged || ((Date.now() - lastImageRefresh) >= refreshInterval)) && !imageRefreshPending) {
        refreshImage(false, imageKey);
      }
    }

    function disableCommandControls() {
      for (const id of ['load', 'upload', 'renderer', 'integrator', 'run', 'finish', 'stop', 'restart', 'denoise']) byId(id).disabled = true;
    }

    async function updateState() {
      try {
        const state = await responseJson(await fetchWithTimeout('/api/state', {cache: 'no-store'}));
        setConnectionStatus(state.initialized ? 'Connected' : 'Starting…', state.initialized ? 'connected' : 'connecting');
        serverConnected = true;
        latestState = state;
        applyState(state);
        return true;
      } catch (_) {
        serverConnected = false;
        setConnectionStatus('Disconnected · retrying', 'disconnected');
        disableCommandControls();
        return false;
      }
    }

    async function updateResults() {
      if (resultsRefreshPending) return;
      resultsRefreshPending = true;
      try {
        const results = await responseJson(await fetchWithTimeout(`/api/results?after=${lastCommandResultId}`, {cache: 'no-store'}));
        if (!Array.isArray(results)) throw new Error('The renderer returned invalid command results');
        for (const result of results) {
          if (!Number.isSafeInteger(result.command_id) || (typeof result.success !== 'boolean')) continue;
          lastCommandResultId = Math.max(lastCommandResultId, result.command_id);
          commandResults.set(result.command_id, result);
          processUploadResult(result);
          finishPendingCommand(result);
        }
        while (commandResults.size > 256) commandResults.delete(commandResults.keys().next().value);
      } catch (_) {
      } finally {
        resultsRefreshPending = false;
      }
    }
    async function refreshImage(showFailure = true, imageKey = '') {
      if (imageRefreshPending) return;
      imageRefreshPending = true;
      lastImageRefresh = Date.now();
      byId('refresh-image').disabled = true;
      byId('image-stage').setAttribute('aria-busy', 'true');
      try {
        const response = await fetchWithTimeout(`/api/image?revision=${Date.now()}`, {cache: 'no-store'}, 10000);
        if (!response.ok) {
          if (showFailure) setMessage('No rendered image is available yet', 'error');
          return;
        }
        const contentType = response.headers.get('Content-Type') || '';
        if (!contentType.toLowerCase().startsWith('image/png')) throw new Error('The renderer returned an invalid image');
        const blob = await response.blob();
        if (!blob.size) throw new Error('The renderer returned an empty image');
        if (imageKey && (imageKey !== currentImageKey)) return;
        const nextUrl = URL.createObjectURL(blob);
        const decoder = new Image();
        decoder.src = nextUrl;
        try {
          await decoder.decode();
        } catch (error) {
          URL.revokeObjectURL(nextUrl);
          throw error;
        }
        if (imageKey && (imageKey !== currentImageKey)) {
          URL.revokeObjectURL(nextUrl);
          return;
        }
        byId('image').src = nextUrl;
        if (imageUrl) URL.revokeObjectURL(imageUrl);
        imageUrl = nextUrl;
        lastImageKey = imageKey;
        byId('image-stage').classList.add('has-image');
      } catch (error) {
        if (showFailure) setMessage(error.name === 'AbortError' ? 'Image refresh timed out' : (error.message || 'Failed to refresh rendered image'), 'error');
      } finally {
        imageRefreshPending = false;
        byId('refresh-image').disabled = false;
        byId('image-stage').removeAttribute('aria-busy');
      }
    }
    byId('choose-folder').onclick = chooseFolder;
    byId('folder').onchange = folderSelected;
    byId('upload').onclick = uploadFolder;
    byId('cancel-upload').onclick = cancelUpload;
    byId('load').onclick = () => command('load_scene', {path: byId('path').value});
    byId('path').onkeydown = event => { if (event.key === 'Enter') byId('load').click(); };
    byId('renderer').onchange = e => command('set_renderer', {renderer: e.target.value});
    byId('integrator').onchange = e => command('set_integrator', {value: Number(e.target.value)});
    for (const type of ['run', 'finish', 'stop', 'restart']) byId(type).onclick = () => command(type);
    byId('denoise').onclick = () => command('denoise');
    byId('refresh-image').onclick = () => refreshImage(true, currentImageKey);
    async function pollServer() {
      clearTimeout(pollTimer);
      if (await updateState()) await updateResults();
      pollTimer = setTimeout(pollServer, 500);
    }
    pollServer();
  </script>
</body>
</html>)html";
}

struct HttpRequest {
  std::string method = {};
  std::string path = {};
  std::string body = {};
};

bool receive_request(NativeSocket socket, HttpRequest& request) {
  std::string data = {};
  data.reserve(4096u);
  std::array<char, 4096u> buffer = {};
  size_t expected_size = 0u;
  size_t content_length = 0u;
  bool headers_parsed = false;

  while (data.size() < kMaxRequestSize) {
    const int received = recv(socket, buffer.data(), static_cast<int>(buffer.size()), 0);
    if (received > 0) {
      if (data.size() + static_cast<size_t>(received) > kMaxRequestSize) {
        return false;
      }
      data.append(buffer.data(), static_cast<size_t>(received));
    } else if (received == 0) {
      break;
    } else {
      if (interrupted()) {
        continue;
      }
      if (would_block()) {
        break;
      }
      return false;
    }

    const size_t header_end = data.find("\r\n\r\n");
    if ((header_end != std::string::npos) && !headers_parsed) {
      headers_parsed = true;
      const size_t first_line_end = data.find("\r\n");
      size_t line_begin = first_line_end == std::string::npos ? header_end : first_line_end + 2u;
      while (line_begin < header_end) {
        const size_t line_end = data.find("\r\n", line_begin);
        if ((line_end == std::string::npos) || (line_end > header_end)) {
          return false;
        }
        const size_t colon = data.find(':', line_begin);
        if ((colon != std::string::npos) && (colon < line_end) && ascii_iequals(std::string_view(data.data() + line_begin, colon - line_begin), "content-length")) {
          size_t value_begin = colon + 1u;
          while ((value_begin < line_end) && ((data[value_begin] == ' ') || (data[value_begin] == '\t'))) {
            ++value_begin;
          }
          const auto parsed = std::from_chars(data.data() + value_begin, data.data() + line_end, content_length);
          if ((parsed.ec != std::errc()) || (parsed.ptr != data.data() + line_end)) {
            return false;
          }
        }
        line_begin = line_end + 2u;
      }
      if ((header_end > (kMaxRequestSize - 4u)) || (content_length > (kMaxRequestSize - header_end - 4u))) {
        return false;
      }
      expected_size = header_end + 4u + content_length;
    }
    if ((expected_size != 0u) && (data.size() >= expected_size)) {
      break;
    }
  }

  const size_t first_line_end = data.find("\r\n");
  const size_t header_end = data.find("\r\n\r\n");
  if ((first_line_end == std::string::npos) || (header_end == std::string::npos)) {
    return false;
  }
  if (!headers_parsed || (data.size() < expected_size)) {
    return false;
  }
  const size_t method_end = data.find(' ');
  const size_t path_end = data.find(' ', method_end + 1u);
  if ((method_end == std::string::npos) || (path_end == std::string::npos) || (path_end > first_line_end)) {
    return false;
  }
  request.method = data.substr(0u, method_end);
  request.path = data.substr(method_end + 1u, path_end - method_end - 1u);
  request.body = data.substr(header_end + 4u, content_length);
  return true;
}

bool send_all(NativeSocket socket, const uint8_t* data, size_t size) {
  size_t offset = 0u;
  while (offset < size) {
#if defined(MSG_NOSIGNAL)
    constexpr int send_flags = MSG_NOSIGNAL;
#else
    constexpr int send_flags = 0;
#endif
    const int sent = send(socket, reinterpret_cast<const char*>(data + offset), static_cast<int>(std::min<size_t>(size - offset, 64u * 1024u)), send_flags);
    if (sent < 0 && interrupted()) {
      continue;
    }
    if (sent <= 0) {
      return false;
    }
    offset += static_cast<size_t>(sent);
  }
  return true;
}

void send_response(NativeSocket socket, int status, const char* status_text, const char* content_type, const uint8_t* body, size_t body_size) {
  const std::string header = "HTTP/1.1 " + std::to_string(status) + " " + status_text + "\r\nContent-Type: " + content_type + "\r\nContent-Length: " + std::to_string(body_size) +
                             "\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n";
  send_all(socket, reinterpret_cast<const uint8_t*>(header.data()), header.size());
  if ((body != nullptr) && (body_size > 0u)) {
    send_all(socket, body, body_size);
  }
}

void send_text(NativeSocket socket, int status, const char* status_text, const char* content_type, const std::string& body) {
  send_response(socket, status, status_text, content_type, reinterpret_cast<const uint8_t*>(body.data()), body.size());
}

void send_json_error(NativeSocket socket, int status, const char* status_text, const std::string& error) {
  send_text(socket, status, status_text, "application/json", Json({{"error", error}}).dump());
}

void erase_upload_session(ApplicationUploadState& state, const std::string& id) {
  const auto session = state.sessions.find(id);
  if (session == state.sessions.end()) {
    return;
  }
  std::error_code error = {};
  std::filesystem::remove_all(session->second.directory, error);
  if (error) {
    log::warning("Failed to remove upload session %s: %s", id.c_str(), error.message().c_str());
  }
  state.total_reserved_size -= std::min(state.total_reserved_size, session->second.total_size);
  state.sessions.erase(session);
}

void collect_command_results(ApplicationUploadState& state, const ApplicationControlServer::ResultProvider& result_provider,
  std::vector<ApplicationCommandResult>& retained_results) {
  std::vector<ApplicationCommandResult> results = {};
  result_provider(results);
  for (const ApplicationCommandResult& result : results) {
    for (auto session = state.sessions.begin(); session != state.sessions.end();) {
      if (!session->second.committed || (session->second.load_command_id != result.command_id)) {
        ++session;
        continue;
      }
      if (result.success) {
        session->second.became_active = true;
        ++session;
      } else {
        const std::string id = session->first;
        ++session;
        erase_upload_session(state, id);
      }
    }
  }
  retained_results.insert(retained_results.end(), std::make_move_iterator(results.begin()), std::make_move_iterator(results.end()));
  if (retained_results.size() > kRetainedCommandResultLimit) {
    const size_t excess = retained_results.size() - kRetainedCommandResultLimit;
    retained_results.erase(retained_results.begin(), retained_results.begin() + static_cast<std::ptrdiff_t>(excess));
  }
}

void maintain_upload_sessions(ApplicationUploadState& state, const ApplicationControlServer::StateProvider& state_provider) {
  if (state.sessions.empty()) {
    return;
  }
  constexpr auto kIncompleteUploadLifetime = std::chrono::minutes(10);
  constexpr auto kPendingSceneLifetime = std::chrono::hours(1);
  const auto now = std::chrono::steady_clock::now();
  const bool activation_pending = std::any_of(state.sessions.begin(), state.sessions.end(), [](const auto& item) {
    return item.second.committed && !item.second.became_active;
  });
  if (!activation_pending && (now < state.next_maintenance)) {
    return;
  }
  state.next_maintenance = now + std::chrono::seconds(1);
  const ApplicationStateSnapshot snapshot = state_provider();
  const std::filesystem::path scene_file = std::filesystem::path(snapshot.scene_file).lexically_normal();
  for (auto session = state.sessions.begin(); session != state.sessions.end();) {
    const std::filesystem::path entry = (session->second.directory / std::filesystem::path(session->second.entry)).lexically_normal();
    std::error_code equivalent_error = {};
    const bool same_file = (scene_file == entry) || std::filesystem::equivalent(scene_file, entry, equivalent_error);
    const bool active = snapshot.scene_loaded && same_file;
    session->second.became_active |= active;
    const auto age = now - session->second.last_activity;
    const bool expired =
      (!session->second.committed && (age >= kIncompleteUploadLifetime)) || (session->second.committed && !session->second.became_active && (age >= kPendingSceneLifetime));
    if ((session->second.became_active && !active) || expired) {
      const std::string id = session->first;
      ++session;
      erase_upload_session(state, id);
    } else {
      ++session;
    }
  }
}

Json requested_upload_files(const ApplicationUploadSession& session) {
  Json files = Json::array();
  for (size_t index = 0u; index < session.files.size(); ++index) {
    const ApplicationUploadFile& file = session.files[index];
    if (file.required && (file.received != file.size)) {
      files.push_back({{"index", index}, {"path", file.relative_path}, {"size", file.size}, {"received", file.received}});
    }
  }
  return files;
}

size_t find_upload_file(const ApplicationUploadSession& session, std::string_view path) {
  const std::string key = upload_path_key(path);
  for (size_t index = 0u; index < session.files.size(); ++index) {
    if (upload_path_key(session.files[index].relative_path) == key) {
      return index;
    }
  }
  return session.files.size();
}

enum class UploadFileRequirement { Success, LimitExceeded, FilesystemError };

UploadFileRequirement require_upload_file(ApplicationUploadState& state, ApplicationUploadSession& session, size_t index, std::string& error) {
  ApplicationUploadFile& file = session.files[index];
  if (file.required) {
    return UploadFileRequirement::Success;
  }
  if ((file.size > kMaxUploadFileSize) || (session.total_size > kMaxUploadSessionSize - file.size) || (state.total_reserved_size > kMaxReservedUploadSize - file.size)) {
    error = "The scene dependency set exceeds the server limits";
    return UploadFileRequirement::LimitExceeded;
  }
  std::error_code filesystem_error = {};
  const std::filesystem::path file_path = session.directory / std::filesystem::path(file.relative_path);
  std::filesystem::create_directories(file_path.parent_path(), filesystem_error);
  if (filesystem_error) {
    error = "Failed to create a dependency directory";
    return UploadFileRequirement::FilesystemError;
  }
  std::ofstream output(file_path, std::ios::binary | std::ios::trunc);
  if (!output) {
    error = "Failed to create a dependency file";
    return UploadFileRequirement::FilesystemError;
  }
  file.required = true;
  session.total_size += file.size;
  state.total_reserved_size += file.size;
  return UploadFileRequirement::Success;
}

void create_upload_session(NativeSocket client, ApplicationUploadState& state, const std::string& body) {
  if (state.sessions.size() >= kMaxUploadSessionCount) {
    send_json_error(client, 429, "Too Many Requests", "Too many upload sessions are active");
    return;
  }

  const Json json = Json::parse(body, nullptr, false);
  if (json.is_discarded() || !json.is_object() || !json.contains("entry") || !json["entry"].is_string() || !json.contains("files") || !json["files"].is_array()) {
    send_json_error(client, 400, "Bad Request", "Upload manifest requires an entry and a files array");
    return;
  }
  const Json& json_files = json["files"];
  if (json_files.empty() || (json_files.size() > kMaxUploadFileCount)) {
    send_json_error(client, 400, "Bad Request", "Upload manifest has an invalid file count");
    return;
  }

  ApplicationUploadSession session = {};
  if (!normalize_upload_path(json["entry"].get<std::string>(), session.entry)) {
    send_json_error(client, 400, "Bad Request", "The scene entry path is invalid");
    return;
  }

  std::unordered_set<std::string> paths = {};
  paths.reserve(json_files.size());
  session.files.reserve(json_files.size());
  for (const Json& json_file : json_files) {
    if (!json_file.is_object() || !json_file.contains("path") || !json_file["path"].is_string() || !json_file.contains("size") || !json_file["size"].is_number_unsigned()) {
      send_json_error(client, 400, "Bad Request", "Every upload file requires a path and an unsigned size");
      return;
    }
    ApplicationUploadFile file = {};
    if (!normalize_upload_path(json_file["path"].get<std::string>(), file.relative_path)) {
      send_json_error(client, 400, "Bad Request", "An upload file path is invalid");
      return;
    }
    file.size = json_file["size"].get<uint64_t>();
    if (!paths.insert(upload_path_key(file.relative_path)).second) {
      send_json_error(client, 400, "Bad Request", "The upload manifest contains conflicting file paths");
      return;
    }
    session.files.push_back(std::move(file));
  }
  const size_t entry_index = find_upload_file(session, session.entry);
  if (entry_index == session.files.size()) {
    send_json_error(client, 400, "Bad Request", "The selected scene entry is not part of the uploaded folder");
    return;
  }
  ApplicationUploadFile& entry_file = session.files[entry_index];
  if ((entry_file.size > kMaxUploadFileSize) || (entry_file.size > kMaxUploadSessionSize) || (state.total_reserved_size > kMaxReservedUploadSize - entry_file.size)) {
    send_json_error(client, 413, "Content Too Large", "The scene entry exceeds the server limits");
    return;
  }
  entry_file.required = true;
  session.total_size = entry_file.size;

  do {
    session.id = upload_id(state);
  } while (state.sessions.contains(session.id));
  session.directory = state.directory / session.id;
  std::error_code filesystem_error = {};
  if (!std::filesystem::create_directory(session.directory, filesystem_error)) {
    send_json_error(client, 500, "Internal Server Error", "Failed to create the upload directory");
    return;
  }
  const std::filesystem::path entry_path = session.directory / std::filesystem::path(entry_file.relative_path);
  std::filesystem::create_directories(entry_path.parent_path(), filesystem_error);
  std::ofstream entry_output(entry_path, std::ios::binary | std::ios::trunc);
  entry_output.close();
  if (filesystem_error || !entry_output) {
    std::filesystem::remove_all(session.directory, filesystem_error);
    send_json_error(client, 500, "Internal Server Error", "Failed to create the scene entry file");
    return;
  }

  const std::string id = session.id;
  const uint64_t total_size = session.total_size;
  state.total_reserved_size += total_size;
  state.sessions.emplace(id, std::move(session));
  send_text(client, 201, "Created", "application/json",
    Json({{"upload_id", id}, {"chunk_size", kMaxUploadChunkSize}, {"required_size", total_size}, {"files", requested_upload_files(state.sessions.at(id))}}).dump());
}

void receive_upload_chunk(NativeSocket client, ApplicationUploadState& state, const HttpRequest& request, const std::string& id, uint64_t file_index) {
  const auto session_iterator = state.sessions.find(id);
  if (session_iterator == state.sessions.end()) {
    send_json_error(client, 404, "Not Found", "Upload session not found");
    return;
  }
  ApplicationUploadSession& session = session_iterator->second;
  session.last_activity = std::chrono::steady_clock::now();
  if (session.committed) {
    send_json_error(client, 409, "Conflict", "The upload session is already committed");
    return;
  }
  if (file_index >= session.files.size()) {
    send_json_error(client, 404, "Not Found", "Upload file not found");
    return;
  }
  uint64_t offset = 0u;
  if (!query_value(request.path, "offset", offset)) {
    send_json_error(client, 400, "Bad Request", "A valid chunk offset is required");
    return;
  }
  ApplicationUploadFile& file = session.files[static_cast<size_t>(file_index)];
  if (!file.required) {
    send_json_error(client, 409, "Conflict", "This file has not been requested by dependency discovery");
    return;
  }
  if (offset != file.received) {
    send_text(client, 409, "Conflict", "application/json", Json({{"error", "Chunk offset does not match"}, {"expected_offset", file.received}}).dump());
    return;
  }
  if (request.body.empty() || (request.body.size() > kMaxUploadChunkSize) || (static_cast<uint64_t>(request.body.size()) > file.size - file.received)) {
    send_json_error(client, 413, "Content Too Large", "The upload chunk has an invalid size");
    return;
  }

  const std::filesystem::path file_path = session.directory / std::filesystem::path(file.relative_path);
  std::error_code filesystem_error = {};
  if (std::filesystem::file_size(file_path, filesystem_error) != file.received || filesystem_error) {
    send_json_error(client, 409, "Conflict", "The uploaded file does not match the session state");
    return;
  }
  std::ofstream output(file_path, std::ios::binary | std::ios::app);
  output.write(request.body.data(), static_cast<std::streamsize>(request.body.size()));
  output.close();
  if (!output) {
    send_json_error(client, 500, "Internal Server Error", "Failed to write the upload chunk");
    return;
  }
  file.received += static_cast<uint64_t>(request.body.size());
  send_text(client, 200, "OK", "application/json", Json({{"received", file.received}, {"size", file.size}}).dump());
}

bool dependency_descriptor(std::string_view path) {
  const size_t dot = path.find_last_of('.');
  if (dot == std::string_view::npos)
    return false;
  std::string ext = upload_path_key(path.substr(dot));
  return (ext == ".json") || (ext == ".gltf") || (ext == ".glb") || (ext == ".obj") || (ext == ".mtl") || (ext == ".materials");
}

enum class DependencyPathResolution { Found, Missing, Unsafe };

DependencyPathResolution resolve_dependency_path(const ApplicationUploadSession& session, const ApplicationUploadFile& source, const std::string& reference,
  std::string& relative_path) {
  std::string portable_reference = reference;
  std::replace(portable_reference.begin(), portable_reference.end(), '\\', '/');
  const std::filesystem::path reference_path(portable_reference);
  if (reference_path.is_absolute() || reference_path.has_root_name()) {
    return DependencyPathResolution::Unsafe;
  }
  const std::filesystem::path source_path(source.relative_path);
  const std::string combined = (source_path.parent_path() / reference_path).lexically_normal().generic_string();
  if (!normalize_upload_path(combined, relative_path)) {
    return DependencyPathResolution::Unsafe;
  }
  return find_upload_file(session, relative_path) == session.files.size() ? DependencyPathResolution::Missing : DependencyPathResolution::Found;
}

void resolve_upload_dependencies(NativeSocket client, ApplicationUploadState& state, const std::string& id) {
  const auto session_iterator = state.sessions.find(id);
  if (session_iterator == state.sessions.end()) {
    send_json_error(client, 404, "Not Found", "Upload session not found");
    return;
  }
  ApplicationUploadSession& session = session_iterator->second;
  session.last_activity = std::chrono::steady_clock::now();
  if (session.committed) {
    send_json_error(client, 409, "Conflict", "The upload session is already committed");
    return;
  }

  bool inspected = true;
  while (inspected) {
    inspected = false;
    for (ApplicationUploadFile& file : session.files) {
      if (!file.required || file.inspected || (file.received != file.size)) {
        continue;
      }
      file.inspected = true;
      inspected = true;
      if (!dependency_descriptor(file.relative_path)) {
        continue;
      }
      if (file.ignore_obj_material_library && upload_path_key(file.relative_path).ends_with(".obj")) {
        continue;
      }
      const SceneDependencyInspection result = inspect_scene_dependencies(session.directory / std::filesystem::path(file.relative_path), file.relative_path);
      if (!result.error.empty()) {
        send_json_error(client, 422, "Unprocessable Content", file.relative_path + ": " + result.error);
        return;
      }
      for (const std::string& reference : result.references) {
        std::string dependency_path = {};
        const DependencyPathResolution path_resolution = resolve_dependency_path(session, file, reference, dependency_path);
        if (path_resolution == DependencyPathResolution::Unsafe) {
          send_json_error(client, 400, "Bad Request", file.relative_path + ": dependency path is outside the selected folder: " + reference);
          return;
        }
        if (path_resolution == DependencyPathResolution::Missing) {
          const std::string missing = file.relative_path + " → " + reference;
          if (std::find(session.unavailable_references.begin(), session.unavailable_references.end(), missing) == session.unavailable_references.end()) {
            session.unavailable_references.push_back(missing);
          }
          continue;
        }
        const size_t dependency_index = find_upload_file(session, dependency_path);
        std::string reservation_error = {};
        const UploadFileRequirement requirement = require_upload_file(state, session, dependency_index, reservation_error);
        if (requirement != UploadFileRequirement::Success) {
          if (requirement == UploadFileRequirement::LimitExceeded) {
            send_json_error(client, 413, "Content Too Large", reservation_error);
          } else {
            send_json_error(client, 500, "Internal Server Error", reservation_error);
          }
          return;
        }
        if (std::find_if(result.geometry_with_external_materials.begin(), result.geometry_with_external_materials.end(), [&](const std::string& geometry) {
              return upload_path_key(geometry) == upload_path_key(reference);
            }) != result.geometry_with_external_materials.end()) {
          session.files[dependency_index].ignore_obj_material_library = true;
        }
      }
    }
  }

  Json requested = requested_upload_files(session);
  session.dependencies_resolved = requested.empty();
  send_text(client, 200, "OK", "application/json",
    Json({{"complete", session.dependencies_resolved}, {"required_size", session.total_size}, {"files", std::move(requested)},
           {"unavailable_references", session.unavailable_references}})
      .dump());
}

void commit_upload_session(NativeSocket client, ApplicationUploadState& state, const std::string& id, const ApplicationControlServer::CommandSubmitter& submit_command) {
  const auto session_iterator = state.sessions.find(id);
  if (session_iterator == state.sessions.end()) {
    send_json_error(client, 404, "Not Found", "Upload session not found");
    return;
  }
  ApplicationUploadSession& session = session_iterator->second;
  session.last_activity = std::chrono::steady_clock::now();
  if (session.committed) {
    send_json_error(client, 409, "Conflict", "The upload session is already committed");
    return;
  }
  if (!session.dependencies_resolved) {
    send_json_error(client, 409, "Conflict", "Scene dependencies have not been resolved");
    return;
  }
  for (const ApplicationUploadFile& file : session.files) {
    if (!file.required) {
      continue;
    }
    if (file.received != file.size) {
      send_json_error(client, 409, "Conflict", "The upload is incomplete");
      return;
    }
    std::error_code filesystem_error = {};
    if (std::filesystem::file_size(session.directory / std::filesystem::path(file.relative_path), filesystem_error) != file.size || filesystem_error) {
      send_json_error(client, 409, "Conflict", "An uploaded file failed validation");
      return;
    }
  }

  ApplicationCommand command = {};
  command.type = ApplicationCommandType::LoadScene;
  command.path = (session.directory / std::filesystem::path(session.entry)).string();
  const uint64_t command_id = submit_command(std::move(command));
  session.committed = true;
  session.load_command_id = command_id;
  send_text(client, 202, "Accepted", "application/json", Json({{"accepted", true}, {"command_id", command_id}, {"upload_id", id}}).dump());
}

void cancel_upload_session(NativeSocket client, ApplicationUploadState& state, const std::string& id, const ApplicationControlServer::StateProvider& state_provider) {
  const auto session = state.sessions.find(id);
  if (session == state.sessions.end()) {
    send_json_error(client, 404, "Not Found", "Upload session not found");
    return;
  }
  if (session->second.committed) {
    const ApplicationStateSnapshot snapshot = state_provider();
    const std::filesystem::path scene_file = std::filesystem::path(snapshot.scene_file).lexically_normal();
    const std::filesystem::path upload_entry = (session->second.directory / std::filesystem::path(session->second.entry)).lexically_normal();
    std::error_code equivalent_error = {};
    const bool same_file = (scene_file == upload_entry) || std::filesystem::equivalent(scene_file, upload_entry, equivalent_error);
    if (snapshot.scene_loaded && same_file) {
      send_json_error(client, 409, "Conflict", "The renderer is still using this upload");
      return;
    }
  }
  erase_upload_session(state, id);
  send_text(client, 200, "OK", "application/json", R"({"deleted":true})");
}

const char* renderer_mode_name(RendererMode mode) {
  switch (mode) {
    case RendererMode::Rasterization:
      return "raster";
    case RendererMode::GPURaytracing:
      return "gpu";
    default:
      return "cpu";
  }
}

const char* run_state_name(RendererRunState state) {
  switch (state) {
    case RendererRunState::Running:
      return "running";
    case RendererRunState::Finishing:
      return "finishing";
    case RendererRunState::Completed:
      return "completed";
    default:
      return "stopped";
  }
}

Json state_json(const ApplicationStateSnapshot& state) {
  Json integrators = Json::array();
  for (const ApplicationIntegratorInfo& integrator : state.integrators) {
    integrators.push_back({{"value", integrator.value}, {"id", integrator.id}, {"name", integrator.name}, {"enabled", integrator.enabled}});
  }
  return {
    {"revision", state.revision},
    {"initialized", state.initialized},
    {"scene_loaded", state.scene_loaded},
    {"scene_file", state.scene_file},
    {"can_denoise", state.can_denoise},
    {"gpu_renderer_available", state.gpu_renderer_available},
    {"quit_requested", state.quit_requested},
    {"renderer", renderer_mode_name(state.renderer_mode)},
    {"renderer_name", state.renderer_name},
    {"integrator", state.integrator_name},
    {"integrator_value", static_cast<uint32_t>(state.integrator_type)},
    {"integrators", std::move(integrators)},
    {"run_state", run_state_name(state.controls.state)},
    {"controls",
      {{"can_run", state.controls.can_run}, {"can_finish", state.controls.can_finish}, {"can_stop", state.controls.can_stop}, {"can_restart", state.controls.can_restart}}},
    {"preparation", {{"state", static_cast<uint32_t>(state.preparation.state)}, {"phase", state.preparation.phase}, {"message", state.preparation.message},
                      {"completed_steps", state.preparation.completed_steps}, {"total_steps", state.preparation.total_steps}}},
    {"runtime", {{"valid", state.runtime.valid}, {"completed_samples", state.runtime.completed_samples}, {"target_samples", state.runtime.target_samples},
                  {"elapsed_seconds", state.runtime.elapsed_seconds}, {"estimated_remaining_seconds", state.runtime.estimated_remaining_seconds}}},
    {"view", {{"exposure", state.view.exposure}, {"view_layer", state.view.view_layer}, {"output_view", state.view.view_image}, {"display_transform", state.view.view_option}}},
  };
}

bool parse_command(const Json& json, ApplicationCommand& command, std::string& error) {
  if (!json.is_object() || !json.contains("type") || !json["type"].is_string()) {
    error = "Command type is required";
    return false;
  }
  try {
    const std::string type = json["type"].get<std::string>();
    const auto path = [&]() {
      return json.value("path", std::string{});
    };
    const auto unsigned_value = [&]() {
      return json.value("value", 0u);
    };

    if (type == "load_scene") {
      command.type = ApplicationCommandType::LoadScene;
      command.path = path();
    } else if (type == "save_scene") {
      command.type = ApplicationCommandType::SaveScene;
      command.path = path();
    } else if (type == "load_reference") {
      command.type = ApplicationCommandType::LoadReferenceImage;
      command.path = path();
    } else if (type == "save_image") {
      command.type = ApplicationCommandType::SaveImage;
      command.path = path();
      const std::string format = json.value("format", std::string("png"));
      if ((format != "png") && (format != "exr")) {
        error = "Image format must be png or exr";
        return false;
      }
      command.save_image_mode = format == "png" ? SaveImageMode::TonemappedLDR : SaveImageMode::RGB;
    } else if (type == "denoise") {
      command.type = ApplicationCommandType::Denoise;
    } else if (type == "set_renderer") {
      command.type = ApplicationCommandType::SetRenderer;
      const std::string renderer = json.value("renderer", std::string{});
      if (renderer == "cpu")
        command.renderer = RendererMode::CPURaytracing;
      else if (renderer == "raster")
        command.renderer = RendererMode::Rasterization;
      else if (renderer == "gpu")
        command.renderer = RendererMode::GPURaytracing;
      else {
        error = "Renderer must be cpu, raster, or gpu";
        return false;
      }
    } else if (type == "set_integrator") {
      command.type = ApplicationCommandType::SetIntegrator;
      command.integrator = static_cast<Integrator::Type>(unsigned_value());
    } else if (type == "run")
      command.type = ApplicationCommandType::Run;
    else if (type == "finish")
      command.type = ApplicationCommandType::Finish;
    else if (type == "stop")
      command.type = ApplicationCommandType::Stop;
    else if (type == "restart")
      command.type = ApplicationCommandType::Restart;
    else if (type == "reload_scene")
      command.type = ApplicationCommandType::ReloadScene;
    else if (type == "reload_geometry")
      command.type = ApplicationCommandType::ReloadGeometry;
    else if (type == "reload_shaders")
      command.type = ApplicationCommandType::ReloadShaders;
    else if (type == "cancel_preparation")
      command.type = ApplicationCommandType::CancelPreparation;
    else if (type == "set_exposure") {
      command.type = ApplicationCommandType::SetExposure;
      command.float_value = json.value("value", 1.0f);
    } else if (type == "set_view_layer") {
      command.type = ApplicationCommandType::SetViewLayer;
      command.unsigned_value = unsigned_value();
    } else if (type == "set_output_view") {
      command.type = ApplicationCommandType::SetOutputView;
      command.unsigned_value = unsigned_value();
    } else if (type == "set_display_transform") {
      command.type = ApplicationCommandType::SetDisplayTransform;
      command.unsigned_value = unsigned_value();
    } else if (type == "set_gpu_wavefront_steps") {
      command.type = ApplicationCommandType::SetGPUWavefrontSteps;
      command.unsigned_value = unsigned_value();
    } else if (type == "quit")
      command.type = ApplicationCommandType::Quit;
    else {
      error = "Unknown command type";
      return false;
    }
    return true;
  } catch (const Json::exception&) {
    error = "Command fields have invalid types";
    return false;
  }
}

}  // namespace

ApplicationControlServer::ApplicationControlServer()
  : _uploads(std::make_unique<ApplicationUploadState>()) {
}

ApplicationControlServer::~ApplicationControlServer() {
  shutdown();
}

bool ApplicationControlServer::init(const ApplicationControlServerConfig& config, StateProvider state_provider, CommandSubmitter command_submitter, ResultProvider result_provider,
  ImageProvider image_provider) {
  shutdown();
  if (!state_provider || !command_submitter || !result_provider) {
    log::error("Application control server requires state, command, and result providers");
    return false;
  }
  if (!_uploads || !prepare_upload_directory(*_uploads)) {
    log::error("Failed to create the application control upload directory");
    return false;
  }
#if ETX_PLATFORM_WINDOWS
  WSADATA data = {};
  if (WSAStartup(MAKEWORD(2, 2), &data) != 0) {
    clear_uploads(*_uploads);
    return false;
  }
#endif
  _socket_api_initialized = true;
  const NativeSocket socket = ::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
  if (stored_socket(socket) == kInvalidSocket) {
    shutdown();
    return false;
  }
  int reuse_address = 1;
  setsockopt(socket, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<const char*>(&reuse_address), sizeof(reuse_address));

  sockaddr_in address = {};
  address.sin_family = AF_INET;
  address.sin_port = htons(config.port);
  if (inet_pton(AF_INET, config.bind_address.c_str(), &address.sin_addr) != 1) {
    close_socket(socket);
    shutdown();
    return false;
  }
  if ((bind(socket, reinterpret_cast<const sockaddr*>(&address), sizeof(address)) != 0) || (listen(socket, 16) != 0)) {
    close_socket(socket);
    shutdown();
    return false;
  }
#if ETX_PLATFORM_WINDOWS
  u_long non_blocking = 1u;
  if (ioctlsocket(socket, FIONBIO, &non_blocking) != 0) {
    close_socket(socket);
    shutdown();
    return false;
  }
#else
  const int socket_flags = fcntl(socket, F_GETFL, 0);
  if ((socket_flags < 0) || (fcntl(socket, F_SETFL, socket_flags | O_NONBLOCK) != 0)) {
    close_socket(socket);
    shutdown();
    return false;
  }
#endif
  _listen_socket = stored_socket(socket);
  _state_provider = std::move(state_provider);
  _command_submitter = std::move(command_submitter);
  _result_provider = std::move(result_provider);
  _image_provider = std::move(image_provider);
  log::info("Application control server listening on http://%s:%u", config.bind_address.c_str(), config.port);
  if (config.bind_address != "127.0.0.1") {
    log::warning("Application control server is exposed beyond the loopback interface and has no authentication");
  }
  return true;
}

void ApplicationControlServer::poll() {
  if (!running()) {
    return;
  }
  collect_command_results(*_uploads, _result_provider, _retained_results);
  maintain_upload_sessions(*_uploads, _state_provider);
  sockaddr_in client_address = {};
#if ETX_PLATFORM_WINDOWS
  int address_size = sizeof(client_address);
#else
  socklen_t address_size = sizeof(client_address);
#endif
  const NativeSocket client = accept(native_socket(_listen_socket), reinterpret_cast<sockaddr*>(&client_address), &address_size);
  if (stored_socket(client) == kInvalidSocket) {
    return;
  }
#if ETX_PLATFORM_WINDOWS
  u_long blocking = 0u;
  ioctlsocket(client, FIONBIO, &blocking);
  DWORD receive_timeout = 100u;
  DWORD send_timeout = 2000u;
  setsockopt(client, SOL_SOCKET, SO_RCVTIMEO, reinterpret_cast<const char*>(&receive_timeout), sizeof(receive_timeout));
  setsockopt(client, SOL_SOCKET, SO_SNDTIMEO, reinterpret_cast<const char*>(&send_timeout), sizeof(send_timeout));
#else
  const int client_flags = fcntl(client, F_GETFL, 0);
  if (client_flags >= 0) {
    fcntl(client, F_SETFL, client_flags & ~O_NONBLOCK);
  }
  timeval receive_timeout = {.tv_sec = 0, .tv_usec = 100000};
  timeval send_timeout = {.tv_sec = 2, .tv_usec = 0};
  setsockopt(client, SOL_SOCKET, SO_RCVTIMEO, &receive_timeout, sizeof(receive_timeout));
  setsockopt(client, SOL_SOCKET, SO_SNDTIMEO, &send_timeout, sizeof(send_timeout));
# if defined(SO_NOSIGPIPE)
  int no_sigpipe = 1;
  setsockopt(client, SOL_SOCKET, SO_NOSIGPIPE, &no_sigpipe, sizeof(no_sigpipe));
# endif
#endif

  HttpRequest request = {};
  if (!receive_request(client, request)) {
    send_text(client, 400, "Bad Request", "text/plain; charset=utf-8", "Invalid HTTP request");
    close_socket(client);
    return;
  }
  const size_t query = request.path.find('?');
  const std::string path = request.path.substr(0u, query);
  std::string upload_id_value = {};
  uint64_t upload_file_index = 0u;

  if ((request.method == "GET") && (path == "/")) {
    send_text(client, 200, "OK", "text/html; charset=utf-8", client_page());
  } else if ((request.method == "GET") && (path == "/api/state")) {
    send_text(client, 200, "OK", "application/json", state_json(_state_provider()).dump());
  } else if ((request.method == "GET") && (path == "/api/results")) {
    uint64_t after = 0u;
    query_value(request.path, "after", after);
    Json response = Json::array();
    for (const auto& result : _retained_results) {
      if (result.command_id > after) {
        response.push_back({{"command_id", result.command_id}, {"success", result.success}, {"message", result.message}});
      }
    }
    send_text(client, 200, "OK", "application/json", response.dump());
  } else if ((request.method == "GET") && (path == "/api/image")) {
    std::vector<uint8_t> png = {};
    uint32_t width = 0u;
    uint32_t height = 0u;
    if (_image_provider && _image_provider(png, width, height)) {
      send_response(client, 200, "OK", "image/png", png.data(), png.size());
    } else {
      send_text(client, 404, "Not Found", "application/json", R"({"error":"No image is available"})");
    }
  } else if ((request.method == "POST") && (path == "/api/uploads")) {
    create_upload_session(client, *_uploads, request.body);
  } else if ((request.method == "PUT") && parse_upload_file_path(path, upload_id_value, upload_file_index)) {
    receive_upload_chunk(client, *_uploads, request, upload_id_value, upload_file_index);
  } else if ((request.method == "POST") && parse_upload_action_path(path, "/resolve", upload_id_value)) {
    resolve_upload_dependencies(client, *_uploads, upload_id_value);
  } else if ((request.method == "POST") && parse_upload_action_path(path, "/commit", upload_id_value)) {
    commit_upload_session(client, *_uploads, upload_id_value, _command_submitter);
  } else if ((request.method == "DELETE") && parse_upload_action_path(path, "", upload_id_value)) {
    cancel_upload_session(client, *_uploads, upload_id_value, _state_provider);
  } else if ((request.method == "POST") && (path == "/api/commands")) {
    const Json json = Json::parse(request.body, nullptr, false);
    ApplicationCommand command = {};
    std::string error = {};
    if (json.is_discarded() || !parse_command(json, command, error)) {
      send_text(client, 400, "Bad Request", "application/json", Json({{"accepted", false}, {"error", error.empty() ? "Invalid JSON" : error}}).dump());
    } else {
      const uint64_t id = _command_submitter(std::move(command));
      send_text(client, 202, "Accepted", "application/json", Json({{"accepted", true}, {"command_id", id}}).dump());
    }
  } else {
    send_text(client, 404, "Not Found", "text/plain; charset=utf-8", "Not found");
  }
  close_socket(client);
}

void ApplicationControlServer::shutdown() {
  if (_listen_socket != kInvalidSocket) {
    close_socket(native_socket(_listen_socket));
    _listen_socket = kInvalidSocket;
  }
#if ETX_PLATFORM_WINDOWS
  if (_socket_api_initialized) {
    WSACleanup();
  }
#endif
  _socket_api_initialized = false;
  _state_provider = {};
  _command_submitter = {};
  _result_provider = {};
  _image_provider = {};
  _retained_results.clear();
  if (_uploads) {
    clear_uploads(*_uploads);
  }
}

bool ApplicationControlServer::running() const {
  return _listen_socket != kInvalidSocket;
}

}  // namespace etx
