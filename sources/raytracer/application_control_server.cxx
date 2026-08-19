#include "application_control_server.hxx"
#include "application_control_html.hxx"
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

std::string_view client_page() {
  return {reinterpret_cast<const char*>(kApplicationControlHtml), sizeof(kApplicationControlHtml)};
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

const char* run_state_name(RendererStatusState state) {
  switch (state) {
    case RendererStatusState::Running:
      return "running";
    case RendererStatusState::Finishing:
      return "finishing";
    case RendererStatusState::Completed:
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
  const bool runtime_valid = state.status.progress_kind == RendererProgressKind::Samples;
  const uint32_t completed_samples = runtime_valid ? state.status.completed_units : 0u;
  const uint32_t target_samples = runtime_valid ? state.status.total_units : 0u;
  const double elapsed_seconds = state.status.elapsed_available ? state.status.elapsed_seconds : 0.0;
  const double estimated_remaining_seconds = state.status.remaining_available ? state.status.remaining_seconds : -1.0;
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
    {"run_state", run_state_name(state.status.state)},
    {"controls",
      {{"can_run", state.controls.can_run}, {"can_finish", state.controls.can_finish}, {"can_stop", state.controls.can_stop}, {"can_restart", state.controls.can_restart}}},
    {"preparation", {{"state", static_cast<uint32_t>(state.preparation.state)}, {"phase", state.preparation.phase}, {"message", state.preparation.message},
                      {"completed_steps", state.preparation.completed_steps}, {"total_steps", state.preparation.total_steps}}},
    {"runtime", {{"valid", runtime_valid}, {"completed_samples", completed_samples}, {"target_samples", target_samples}, {"elapsed_seconds", elapsed_seconds},
                  {"estimated_remaining_seconds", estimated_remaining_seconds}}},
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
#if ETX_PLATFORM_WINDOWS
  const BOOL exclusive_address = TRUE;
  if (setsockopt(socket, SOL_SOCKET, SO_EXCLUSIVEADDRUSE, reinterpret_cast<const char*>(&exclusive_address), sizeof(exclusive_address)) != 0) {
    log::error("Failed to make the application control server address exclusive: %d", WSAGetLastError());
    close_socket(socket);
    shutdown();
    return false;
  }
#else
  const int reuse_address = 1;
  if (setsockopt(socket, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<const char*>(&reuse_address), sizeof(reuse_address)) != 0) {
    log::error("Failed to enable application control server address reuse: %d", errno);
    close_socket(socket);
    shutdown();
    return false;
  }
#endif

  sockaddr_in address = {};
  address.sin_family = AF_INET;
  address.sin_port = htons(config.port);
  if (inet_pton(AF_INET, config.bind_address.c_str(), &address.sin_addr) != 1) {
    close_socket(socket);
    shutdown();
    return false;
  }
  if (bind(socket, reinterpret_cast<const sockaddr*>(&address), sizeof(address)) != 0) {
#if ETX_PLATFORM_WINDOWS
    log::error("Failed to bind application control server to %s:%u: %d", config.bind_address.c_str(), config.port, WSAGetLastError());
#else
    log::error("Failed to bind application control server to %s:%u: %d", config.bind_address.c_str(), config.port, errno);
#endif
    close_socket(socket);
    shutdown();
    return false;
  }
  if (listen(socket, 16) != 0) {
#if ETX_PLATFORM_WINDOWS
    log::error("Failed to listen on application control server socket: %d", WSAGetLastError());
#else
    log::error("Failed to listen on application control server socket: %d", errno);
#endif
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
    const std::string_view page = client_page();
    send_response(client, 200, "OK", "text/html; charset=utf-8", reinterpret_cast<const uint8_t*>(page.data()), page.size());
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
