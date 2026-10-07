#include <etx/std.hxx>
#include <etx/core/core.hxx>
#include "importer_plugin.hxx"
#include <etx/import/scene_dependencies.hxx>
#include <etx/render/host/native_document.hxx>
#include <etx/render/host/scene_representation.hxx>
#include <etx/render/shared/ior_database.hxx>
#include <etx/core/environment.hxx>
#include <unordered_set>

namespace {

void message(char* buffer, uint32_t capacity, const std::string& text) {
  if ((buffer != nullptr) && (capacity > 0u))
    snprintf(buffer, capacity, "%s", text.c_str());
}

std::string extension(const std::filesystem::path& path) {
  auto value = etx::path_to_utf8(path.extension());
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return value;
}

int32_t ETX_IMPORT_CALL inspect(const char* source, uint32_t flags, void* context, EtxImportDependency dependency, char* error, uint32_t capacity) {
  try {
    if (((flags & ETX_IMPORT_DEPENDENCY) != 0u) && etx::kImporterFormat.inspect_root_only)
      return 1;
    if (((flags & ETX_IMPORT_EXTERNAL_MATERIALS) != 0u) && (extension(std::filesystem::u8path(source)) == ".obj"))
      return 1;
    const auto inspection = etx::kImporterFormat.inspect_dependencies(std::filesystem::u8path(source), source);
    if (inspection.error.empty() == false) {
      message(error, capacity, inspection.error);
      return 0;
    }
    for (const auto& reference : inspection.references)
      dependency(context, reference.c_str(), 0u);
    for (const auto& reference : inspection.geometry_with_external_materials)
      dependency(context, reference.c_str(), 1u);
    return 1;
  } catch (const std::exception& failure) {
    message(error, capacity, failure.what());
    return 0;
  }
}

bool check_dependencies(const std::filesystem::path& source, std::unordered_set<std::string>& visited, bool material_override, std::string& error) {
  const auto path = std::filesystem::absolute(source).lexically_normal();
  std::error_code file_error;
  if (std::filesystem::is_regular_file(path, file_error) == false) {
    error = "Missing dependency: " + etx::path_to_utf8(path);
    return false;
  }
  if (visited.emplace(etx::path_to_utf8(path)).second == false)
    return true;
  if (material_override && (extension(path) == ".obj"))
    return true;
  const auto inspection = etx::kImporterFormat.inspect_dependencies(path, etx::path_to_utf8(path));
  if (inspection.error.empty() == false) {
    error = etx::path_to_utf8(path) + ": " + inspection.error;
    return false;
  }
  for (const auto& reference : inspection.references) {
    if (etx::kImporterFormat.inspect_root_only) {
      const auto dependency = path.parent_path() / std::filesystem::u8path(reference);
      if (std::filesystem::is_regular_file(dependency, file_error) == false) {
        error = "Missing dependency: " + etx::path_to_utf8(dependency);
        return false;
      }
      continue;
    }
    const bool overrides = std::find(inspection.geometry_with_external_materials.begin(), inspection.geometry_with_external_materials.end(), reference) !=
                           inspection.geometry_with_external_materials.end();
    if (check_dependencies(path.parent_path() / std::filesystem::u8path(reference), visited, overrides, error) == false)
      return false;
  }
  return true;
}

int32_t ETX_IMPORT_CALL convert(const EtxImportRequest* request, char* document, uint32_t document_capacity, char* error, uint32_t error_capacity) {
  try {
    if ((request == nullptr) || (request->size < sizeof(EtxImportRequest)) || (request->progress == nullptr)) {
      message(error, error_capacity, "Invalid import request.");
      return 0;
    }
    if (etx::kImporterFormat.probe(request->source_path) == 0) {
      message(error, error_capacity, std::string(etx::kImporterFormat.name) + " does not support this file.");
      return 0;
    }
    const auto progress = [&](uint32_t step, const char* stage) {
      if (request->progress(request->context, step, 4u, stage) == 0)
        throw std::runtime_error("Import cancelled.");
    };
    progress(0u, "Inspecting dependencies");
    std::string failure;
    std::unordered_set<std::string> visited;
    if (check_dependencies(std::filesystem::u8path(request->source_path), visited, false, failure) == false)
      throw std::runtime_error(failure);
    // Each module has its own runtime globals. Embedded/generated assets belong to this request.
    const auto runtime = std::filesystem::u8path(request->runtime_directory) / "raytracer";
    etx::env().setup(etx::path_to_utf8(runtime).c_str(), false);
    etx::TaskScheduler scheduler;
    etx::IORDatabase database;
    database.load(etx::env().file_in_data("spectrum"));
    etx::SceneRepresentation scene(scheduler, database);
    const auto directory = std::filesystem::u8path(request->output_directory);
    const auto embedded = directory / "embedded";
    std::filesystem::create_directory(embedded);
    progress(1u, "Decoding source");
    if (scene.load_document(request->source_path, etx::kImporterFormat.decode, etx::kImporterFormat.decode_root, etx::path_to_utf8(embedded).c_str(),
          etx::SceneRepresentation::DocumentOnly, nullptr) == false) {
      throw std::runtime_error("Source decoding failed; the native document was not published.");
    }
    if (etx::kImporterFormat.configure_scene != nullptr)
      etx::kImporterFormat.configure_scene(request->source_path, scene);
    scene.data().owns_assets = true;
    progress(2u, "Writing native ETX document");
    const std::string output = scene.save_to_file(etx::path_to_utf8(directory / "scene.etx.json").c_str());
    if (output.empty())
      throw std::runtime_error("Could not write the native document or its assets.");
    progress(3u, "Validating native ETX document");
    etx::SceneRepresentation validation(scheduler, database);
    if (validation.load_from_file(output.c_str(), etx::SceneRepresentation::DocumentOnly) == false)
      throw std::runtime_error("Converted native document failed validation.");
    if ((output.size() + 1u) > document_capacity)
      throw std::runtime_error("Output path buffer is too small.");
    message(document, document_capacity, output);
    progress(4u, "Conversion complete");
    return 1;
  } catch (const std::exception& failure) {
    message(error, error_capacity, failure.what());
    return 0;
  } catch (...) {
    message(error, error_capacity, "Importer failed with an unknown exception.");
    return 0;
  }
}

const EtxImporterApi api = {sizeof(EtxImporterApi), ETX_IMPORT_ABI_VERSION, etx::kImporterFormat.id, etx::kImporterFormat.name, etx::kImporterFormat.extensions,
  etx::kImporterFormat.probe, inspect, convert};

}  // namespace

ETX_IMPORT_EXPORT const EtxImporterApi* ETX_IMPORT_CALL etx_get_importer_api(uint32_t version) {
  return version == ETX_IMPORT_ABI_VERSION ? &api : nullptr;
}
