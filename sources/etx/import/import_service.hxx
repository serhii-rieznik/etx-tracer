#pragma once

#include <etx/import/importer_abi.hxx>
#include <etx/import/scene_dependencies.hxx>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>
#include <functional>

namespace etx {

struct ImportArtifact {
  std::filesystem::path directory;
  std::shared_ptr<ImportArtifact> parent;
  std::filesystem::path document;
  ~ImportArtifact();
};

struct ImporterInfo {
  std::string id;
  std::string name;
  std::string extensions;
  std::filesystem::path module;
  bool loaded = false;
  std::string error;
};

struct ImportService {
  explicit ImportService(const std::filesystem::path& runtime_directory);
  ~ImportService();
  ImportService(const ImportService&) = delete;
  ImportService& operator=(const ImportService&) = delete;

  const std::vector<ImporterInfo>& importers() const;
  const std::vector<ImporterInfo>& modules() const;
  const std::filesystem::path& plugin_directory() const;
  const std::string& discovery_error() const;
  SceneDependencyInspection inspect(const std::filesystem::path& source, uint32_t flags) const;
  SceneDependencyInspection inspect(const std::filesystem::path& source, uint32_t flags, const std::string& importer_id) const;
  std::shared_ptr<ImportArtifact> convert(const std::filesystem::path& source, const std::function<bool(uint32_t, uint32_t, const char*)>& progress, std::string& error) const;
  std::shared_ptr<ImportArtifact> convert(const std::filesystem::path& source, const std::string& importer_id, const std::function<bool(uint32_t, uint32_t, const char*)>& progress,
    std::string& error) const;

 private:
  struct Plugin;
  const Plugin* select_importer(const std::string& source, const std::string& importer_id, std::string& error, bool& unsupported, bool probe_source) const;
  std::filesystem::path _runtime_directory;
  std::filesystem::path _plugin_directory;
  std::string _discovery_error;
  std::vector<ImporterInfo> _modules;
  std::vector<std::unique_ptr<Plugin>> _plugins;
  std::vector<ImporterInfo> _importers;
};

ImportService& import_service();
bool publish_import_document(const ImportArtifact& artifact, const std::filesystem::path& destination, std::string& output, std::string& error);

}  // namespace etx
