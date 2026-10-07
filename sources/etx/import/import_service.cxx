#include <etx/core/core.hxx>
#include <etx/import/import_service.hxx>
#include <etx/render/host/native_document.hxx>
#include <etx/core/log.hxx>
#include <etx/core/environment.hxx>
#include <etx/render/host/scene_representation.hxx>
#include <etx/render/shared/ior_database.hxx>
#include <atomic>
#include <chrono>
#include <sstream>
#include <unordered_set>
#include <cstring>
#include <cctype>
#if defined(_WIN32)
# include <windows.h>
#else
# include <dlfcn.h>
#endif

namespace etx {

struct ImportService::Plugin {
  const EtxImporterApi* api = nullptr;
#if defined(_WIN32)
  HMODULE module = nullptr;
  ~Plugin() {
    if (module != nullptr)
      FreeLibrary(module);
  }
#else
  void* module = nullptr;
  ~Plugin() {
    if (module != nullptr)
      dlclose(module);
  }
#endif
};

ImportArtifact::~ImportArtifact() {
  std::error_code error;
  std::filesystem::remove_all(directory, error);
}

namespace {

std::string lowercase(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char character) {
    return static_cast<char>(std::tolower(character));
  });
  return value;
}

std::string normalize_extensions(const char* text) {
  std::string result;
  std::unordered_set<std::string> seen;
  std::istringstream stream(text);
  for (std::string value; std::getline(stream, value, ',');) {
    const auto first = value.find_first_not_of(" \t.");
    const auto last = value.find_last_not_of(" \t");
    if ((first == std::string::npos) || (last < first))
      continue;
    value = lowercase(value.substr(first, last - first + 1u));
    if (std::all_of(value.begin(), value.end(), [](unsigned char c) {
          return std::isalnum(c) || (c == '-') || (c == '_');
        }) == false)
      return {};
    if (seen.insert(value).second == false)
      continue;
    if (result.empty() == false)
      result += ',';
    result += value;
  }
  return result;
}

std::string module_error() {
#if defined(_WIN32)
  const DWORD code = GetLastError();
  wchar_t message[1024] = {};
  const DWORD count = FormatMessageW(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, nullptr, code, 0u, message, 1024u, nullptr);
  const int bytes = WideCharToMultiByte(CP_UTF8, 0u, message, static_cast<int>(count), nullptr, 0, nullptr, nullptr);
  std::string result(static_cast<size_t>(bytes), '\0');
  if (bytes > 0)
    WideCharToMultiByte(CP_UTF8, 0u, message, static_cast<int>(count), result.data(), bytes, nullptr, nullptr);
  while ((result.empty() == false) && ((result.back() == '\r') || (result.back() == '\n')))
    result.pop_back();
  return result.empty() ? "Windows loader error " + std::to_string(code) : result;
#else
  const char* error = dlerror();
  return error == nullptr ? "The dynamic loader did not provide an error." : error;
#endif
}

bool inside_directory(const std::filesystem::path& path, const std::filesystem::path& directory) {
  std::error_code error;
  const auto resolved = std::filesystem::weakly_canonical(path, error);
  if (error)
    return false;
  const auto root = std::filesystem::weakly_canonical(directory, error);
  if (error)
    return false;
  const auto relative = resolved.lexically_relative(root);
  return (relative.empty() == false) && (relative.is_absolute() == false) && (*relative.begin() != "..");
}

bool terminated(const char* buffer, size_t capacity) {
  return std::memchr(buffer, 0, capacity) != nullptr;
}

}  // namespace

ImportService::ImportService(const std::filesystem::path& runtime_directory) {
  std::error_code error;
  _runtime_directory = std::filesystem::absolute(runtime_directory, error).lexically_normal();
  if (error) {
    _discovery_error = error.message();
    return;
  }
  _plugin_directory = _runtime_directory / "importers";
#if defined(__APPLE__)
  if (std::filesystem::is_directory(_runtime_directory / "../PlugIns/importers", error)) {
    _plugin_directory = (_runtime_directory / "../PlugIns/importers").lexically_normal();
  }
#endif
  std::vector<std::filesystem::path> paths;
  for (std::filesystem::directory_iterator iterator(_plugin_directory, error), end; (error.value() == 0) && (iterator != end); iterator.increment(error)) {
    std::error_code entry_error;
    if (iterator->is_regular_file(entry_error)) {
      const auto extension = lowercase(path_to_utf8(iterator->path().extension()));
#if defined(_WIN32)
      const bool module = extension == ".dll";
#else
      const bool module = (extension == ".so") || (extension == ".dylib");
#endif
      if (module)
        paths.push_back(iterator->path());
    }
  }
  if (error)
    _discovery_error = "Cannot enumerate " + path_to_utf8(_plugin_directory) + ": " + error.message();
  std::sort(paths.begin(), paths.end());
  for (const auto& path : paths) {
    ImporterInfo info;
    info.module = path;
    info.name = path_to_utf8(path.filename());
    auto plugin = std::make_unique<Plugin>();
#if defined(_WIN32)
    plugin->module = LoadLibraryExW(path.c_str(), nullptr, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    if (plugin->module == nullptr)
      info.error = module_error();
    auto entry = plugin->module == nullptr ? nullptr : reinterpret_cast<EtxGetImporterApi>(GetProcAddress(plugin->module, "etx_get_importer_api"));
#else
    plugin->module = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (plugin->module == nullptr)
      info.error = module_error();
    auto entry = plugin->module == nullptr ? nullptr : reinterpret_cast<EtxGetImporterApi>(dlsym(plugin->module, "etx_get_importer_api"));
#endif
    if ((entry == nullptr) && info.error.empty())
      info.error = "Missing etx_get_importer_api entry point.";
    if (entry != nullptr) {
      plugin->api = entry(ETX_IMPORT_ABI_VERSION);
      const auto* api = plugin->api;
      if (api == nullptr)
        info.error = "The module does not support importer ABI " + std::to_string(ETX_IMPORT_ABI_VERSION) + ".";
      else if ((api->size < sizeof(EtxImporterApi)) || (api->abi_version != ETX_IMPORT_ABI_VERSION))
        info.error = "Incompatible importer ABI or API structure size.";
      else if ((api->id == nullptr) || (api->name == nullptr) || (api->extensions == nullptr) || (api->probe == nullptr) || (api->inspect == nullptr) || (api->convert == nullptr))
        info.error = "The importer API is incomplete.";
      else {
        info.id = api->id;
        info.name = api->name;
        info.extensions = normalize_extensions(api->extensions);
        if (info.id.empty() || info.name.empty() || info.extensions.empty())
          info.error = "Importer ID, name, and supported extensions must be present and valid.";
        else if (std::any_of(_importers.begin(), _importers.end(), [&](const ImporterInfo& existing) {
                   return existing.id == info.id;
                 }))
          info.error = "Duplicate importer ID: " + info.id;
      }
    }
    info.loaded = info.error.empty();
    if (info.loaded) {
      _importers.push_back(info);
      _plugins.push_back(std::move(plugin));
    } else
      log::warning("Importer %s: %s", path_to_utf8(path).c_str(), info.error.c_str());
    _modules.push_back(std::move(info));
  }
}

ImportService::~ImportService() = default;
const std::vector<ImporterInfo>& ImportService::importers() const {
  return _importers;
}
const std::vector<ImporterInfo>& ImportService::modules() const {
  return _modules;
}
const std::filesystem::path& ImportService::plugin_directory() const {
  return _plugin_directory;
}
const std::string& ImportService::discovery_error() const {
  return _discovery_error;
}

const ImportService::Plugin* ImportService::select_importer(const std::string& source, const std::string& importer_id, std::string& error, bool& unsupported,
  bool probe_source) const {
  unsupported = false;
  const Plugin* selected = nullptr;
  for (size_t index = 0u; index < _plugins.size(); ++index) {
    const auto& plugin = _plugins[index];
    const auto& info = _importers[index];
    if ((importer_id.empty() == false) && (importer_id != info.id))
      continue;
    if (probe_source && (plugin->api->probe(source.c_str()) == 0)) {
      if (importer_id.empty() == false) {
        error = info.name + " does not support this file.";
        return nullptr;
      }
      continue;
    }
    if (selected != nullptr) {
      error = "Multiple importers support this file. Select an importer explicitly.";
      return nullptr;
    }
    selected = plugin.get();
  }
  if (selected == nullptr) {
    unsupported = importer_id.empty();
    error = unsupported ? "No installed importer supports this file." : "The selected importer is unavailable: " + importer_id;
  }
  return selected;
}

ImportService& import_service() {
  static ImportService service(std::filesystem::u8path(env().data_folder()) / (env().bundled() ? "../MacOS" : "."));
  return service;
}

bool publish_import_document(const ImportArtifact& artifact, const std::filesystem::path& destination, std::string& output, std::string& error) {
  TaskScheduler scheduler;
  IORDatabase database;
  char spectrum_directory[2048] = {};
  database.load(env().file_in_data("spectrum", spectrum_directory, sizeof(spectrum_directory)));
  SceneRepresentation document(scheduler, database);
  if (document.load_from_file(path_to_utf8(artifact.document).c_str(), SceneRepresentation::DocumentOnly) == false) {
    error = "Cannot read the converted native document.";
    return false;
  }
  output = document.save_to_file(path_to_utf8(destination).c_str());
  if (output.empty()) {
    error = "Cannot publish the native document.";
    return false;
  }
  return true;
}

SceneDependencyInspection ImportService::inspect(const std::filesystem::path& source, uint32_t flags) const {
  return inspect(source, flags, {});
}

SceneDependencyInspection ImportService::inspect(const std::filesystem::path& source, uint32_t flags, const std::string& importer_id) const {
  std::string native_error;
  if (importer_id.empty() && validate_native_document(source, native_error))
    return inspect_scene_dependencies(source, path_to_utf8(source));
  const std::string path = path_to_utf8(source);
  std::string selection_error;
  bool unsupported = false;
  const auto* plugin = select_importer(path, importer_id, selection_error, unsupported, ((flags & ETX_IMPORT_DEPENDENCY) == 0u) || importer_id.empty());
  if (plugin != nullptr) {
    struct Context {
      SceneDependencyInspection result;
      bool callback_failed = false;
    } context;
    context.result.importer_id = plugin->api->id;
    char error[2048] = {};
    const auto callback = [](void* data, const char* reference, uint32_t override_materials) {
      auto& context = *static_cast<Context*>(data);
      try {
        if ((reference == nullptr) || (reference[0] == 0)) {
          context.callback_failed = true;
          return;
        }
        (override_materials == 0u ? context.result.references : context.result.geometry_with_external_materials).emplace_back(reference);
      } catch (...) {
        context.callback_failed = true;
      }
    };
    const bool success = plugin->api->inspect(path.c_str(), flags, &context, callback, error, sizeof(error)) != 0;
    if (context.callback_failed)
      context.result.error = "Importer dependency callback returned invalid data or failed.";
    else if (terminated(error, sizeof(error)) == false)
      context.result.error = "Importer returned an unterminated error message.";
    else if (success == false)
      context.result.error = error[0] == 0 ? "Importer dependency inspection failed." : error;
    return std::move(context.result);
  }
  if (unsupported == false) {
    SceneDependencyInspection result;
    result.error = selection_error;
    return result;
  }
  // Material sidecars are inspected by the native material codec; they are never root import formats.
  const auto extension = lowercase(path_to_utf8(source.extension()));
  if ((extension == ".mtl") || (extension == ".materials"))
    return inspect_scene_dependencies(source, path);
  return {};
}

std::shared_ptr<ImportArtifact> ImportService::convert(const std::filesystem::path& source, const std::function<bool(uint32_t, uint32_t, const char*)>& progress,
  std::string& error) const {
  return convert(source, {}, progress, error);
}

std::shared_ptr<ImportArtifact> ImportService::convert(const std::filesystem::path& source, const std::string& importer_id,
  const std::function<bool(uint32_t, uint32_t, const char*)>& progress, std::string& error) const {
  error.clear();
  const std::string path = path_to_utf8(std::filesystem::absolute(source));
  bool unsupported = false;
  const Plugin* selected = select_importer(path, importer_id, error, unsupported, true);
  if (selected == nullptr)
    return {};
  static std::atomic<uint64_t> sequence = 0u;
  auto artifact = std::make_shared<ImportArtifact>();
  std::error_code file_error;
  const auto temporary = std::filesystem::temp_directory_path(file_error);
  if (file_error) {
    error = file_error.message();
    return {};
  }
  for (uint32_t attempt = 0u; attempt < 32u; ++attempt) {
    artifact->directory = temporary / ("etx-import-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + "-" + std::to_string(sequence++));
    if (std::filesystem::create_directory(artifact->directory, file_error))
      break;
    artifact->directory.clear();
  }
  if (artifact->directory.empty()) {
    error = "Cannot create an import staging directory.";
    return {};
  }
  const std::string output = path_to_utf8(artifact->directory);
  const std::string runtime = path_to_utf8(_runtime_directory);
  struct ProgressContext {
    const std::function<bool(uint32_t, uint32_t, const char*)>& function;
    bool failed = false;
  } context{progress};
  const EtxImportRequest request = {sizeof(EtxImportRequest), path.c_str(), output.c_str(), runtime.c_str(), &context,
    [](void* data, uint32_t completed, uint32_t total, const char* stage) -> int32_t {
      auto& context = *static_cast<ProgressContext*>(data);
      try {
        return context.function(completed, total, stage == nullptr ? "" : stage) ? 1 : 0;
      } catch (...) {
        context.failed = true;
        return 0;
      }
    }};
  char document[4096] = {};
  char failure[2048] = {};
  const bool converted = selected->api->convert(&request, document, sizeof(document), failure, sizeof(failure)) != 0;
  if (context.failed) {
    error = "Import progress callback failed.";
    return {};
  }
  if ((terminated(document, sizeof(document)) == false) || (terminated(failure, sizeof(failure)) == false)) {
    error = "Importer returned an unterminated output path or error message.";
    return {};
  }
  if (converted == false) {
    error = failure[0] == 0 ? "Importer conversion failed." : failure;
    return {};
  }
  artifact->document = std::filesystem::u8path(document).lexically_normal();
  if ((artifact->document.is_absolute() == false) || (inside_directory(artifact->document, artifact->directory) == false)) {
    error = "Importer output must name an absolute native document inside its output directory.";
    return {};
  }
  if (validate_native_document(artifact->document, error) == false) {
    error = "Importer returned an invalid native document: " + error;
    return {};
  }
  TaskScheduler scheduler;
  IORDatabase database;
  char spectrum_directory[2048] = {};
  database.load(env().file_in_data("spectrum", spectrum_directory, sizeof(spectrum_directory)));
  SceneRepresentation validation(scheduler, database);
  if (validation.load_from_file(path_to_utf8(artifact->document).c_str(), SceneRepresentation::DocumentOnly) == false) {
    error = "Importer output failed native document validation.";
    return {};
  }
  const auto& data = validation.data();
  bool package_required = data.owns_assets == false;
  for (const auto* reference : {&data.geometry_file_name, &data.materials_file_name}) {
    package_required |= (reference->empty() == false) && (inside_directory(std::filesystem::u8path(*reference), artifact->directory) == false);
  }
  for (uint32_t index = 0u; index < data.images_vector.size(); ++index) {
    const auto& image = data.images.path(index);
    if (image.empty() || image.starts_with("##"))
      continue;
    package_required |= inside_directory(std::filesystem::u8path(image), artifact->directory) == false;
  }
  for (uint32_t index = 0u; index < data.mediums_vector.size(); ++index) {
    const auto& volume = data.mediums.volume_path(index);
    if (volume.empty() == false)
      package_required |= inside_directory(std::filesystem::u8path(volume), artifact->directory) == false;
  }
  if (package_required) {
    if (progress(0u, 1u, "Collecting native assets") == false) {
      error = "Import cancelled.";
      return {};
    }
    validation.data().owns_assets = true;
    const auto saved = validation.save_to_file(path_to_utf8(artifact->directory / "owned.etx.json").c_str());
    if (saved.empty()) {
      error = "Cannot collect the imported document assets.";
      return {};
    }
    artifact->document = std::filesystem::u8path(saved);
  }
  if (progress(1u, 1u, "Converted") == false) {
    error = "Import cancelled.";
    return {};
  }
  return artifact;
}

}  // namespace etx
