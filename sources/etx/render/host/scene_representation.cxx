#include <etx/render/interop/interop.hxx>

#include <etx/core/core.hxx>
#include <etx/core/environment.hxx>
#include <etx/core/json.hxx>

#include <etx/render/shared/scene.hxx>
#include <etx/render/shared/scattering.hxx>
#include <etx/render/shared/ior_database.hxx>
#include <etx/render/shared/vertex_utils.hxx>
#include <etx/render/shared/sampler.hxx>

#include <etx/render/host/scene_representation.hxx>
#include <etx/render/host/image_pool.hxx>
#include <etx/render/host/medium_pool.hxx>
#include <etx/render/host/bsdf_energy_compensation_lut.hxx>
#include <etx/render/host/scene_data.hxx>
#include <etx/render/host/scene_serialization.hxx>
#include <etx/render/host/scene_loader_utils.hxx>
#include <etx/render/host/scene_procedural_geometry.hxx>
#include <etx/render/host/exr.hxx>
#include <etx/rt/integrators/integrator.hxx>

#include <etx/render/host/scene_obj_loader.hxx>
#include <etx/render/host/scene_gltf_loader.hxx>
#include <etx/render/host/scene_tungsten_loader.hxx>

#include <mikktspace.h>

#include <array>
#include <fstream>
#include <unordered_set>
namespace etx {

namespace {

constexpr float kDefaultCameraClipNear = 0.1f;
constexpr float kDefaultCameraClipFar = 1000.0f;
constexpr uint2 kDefaultModelCameraFilmSize = {1280u, 720u};
constexpr float kDefaultModelSunAngularDiameter = 0.53f;
constexpr float kDefaultModelAtmosphereQuality = 0.125f;

struct NodeGeometryEditAnalysis {
  std::vector<uint32_t> attachment_indices;
  std::vector<uint32_t> mesh_indices;
};

bool finite_point(const float3& value) {
  return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}

bool valid_direction(const float3& value) {
  const float length_squared = dot(value, value);
  return finite_point(value) && std::isfinite(length_squared) && (length_squared > 0.0f);
}

bool write_file_contents(const std::filesystem::path& path, const std::string& contents) {
  FILE* file = fopen(path.string().c_str(), "wb");
  if (file == nullptr) {
    log::error("Failed to open file for writing: %s", path.string().c_str());
    return false;
  }

  const size_t bytes_written = fwrite(contents.data(), 1, contents.size(), file);
  const bool flush_succeeded = fflush(file) == 0;
  const bool close_succeeded = fclose(file) == 0;
  if ((bytes_written != contents.size()) || (flush_succeeded == false) || (close_succeeded == false)) {
    log::error("Failed to write file: %s", path.string().c_str());
    return false;
  }

  return true;
}

std::string content_hash_string(uint64_t hash) {
  char buffer[17] = {};
  snprintf(buffer, sizeof(buffer), "%016llx", static_cast<unsigned long long>(hash));
  return buffer;
}

bool path_is_within_directory(const std::filesystem::path& path, const std::filesystem::path& directory) {
  std::error_code error;
  const std::filesystem::path relative = std::filesystem::relative(path, directory, error);
  if (error || relative.empty() || relative.is_absolute()) {
    return false;
  }
  for (const std::filesystem::path& component : relative) {
    if (component == "..") {
      return false;
    }
  }
  return true;
}

bool managed_scene_asset_path(const std::filesystem::path& path) {
  const std::string directory_name = path.parent_path().filename().string();
  return directory_name.ends_with(".etx.assets");
}

bool inspect_scene_asset_path(const std::filesystem::path& path, bool& file_exists) {
  std::error_code error;
  const std::filesystem::file_status status = std::filesystem::symlink_status(path, error);
  if (status.type() == std::filesystem::file_type::not_found) {
    file_exists = false;
    return true;
  }
  if (error) {
    log::error("Failed to inspect scene asset: %s", path.string().c_str());
    return false;
  }
  if (std::filesystem::is_regular_file(status) == false) {
    log::error("Expected scene asset to be a regular file: %s", path.string().c_str());
    return false;
  }
  file_exists = true;
  return true;
}

bool binary_files_match(const std::filesystem::path& first, const std::filesystem::path& second) {
  std::error_code error;
  const uintmax_t first_size = std::filesystem::file_size(first, error);
  if (error) {
    return false;
  }
  const uintmax_t second_size = std::filesystem::file_size(second, error);
  if (error || (first_size != second_size)) {
    return false;
  }

  std::ifstream first_file(first, std::ios::binary);
  std::ifstream second_file(second, std::ios::binary);
  if ((first_file.is_open() == false) || (second_file.is_open() == false)) {
    return false;
  }
  std::array<uint8_t, 64u * 1024u> first_buffer = {};
  std::array<uint8_t, 64u * 1024u> second_buffer = {};
  uintmax_t remaining = first_size;
  while (remaining > 0u) {
    const size_t bytes_to_read = static_cast<size_t>(std::min<uintmax_t>(first_buffer.size(), remaining));
    first_file.read(reinterpret_cast<char*>(first_buffer.data()), static_cast<std::streamsize>(bytes_to_read));
    second_file.read(reinterpret_cast<char*>(second_buffer.data()), static_cast<std::streamsize>(bytes_to_read));
    if ((first_file.gcount() != static_cast<std::streamsize>(bytes_to_read)) || (second_file.gcount() != static_cast<std::streamsize>(bytes_to_read)) ||
        (memcmp(first_buffer.data(), second_buffer.data(), bytes_to_read) != 0)) {
      return false;
    }
    remaining -= bytes_to_read;
  }
  return true;
}

bool hash_binary_file(const std::filesystem::path& path, uint64_t& hash, uint64_t& total_size) {
  std::ifstream file(path, std::ios::binary);
  if (file.is_open() == false) {
    return false;
  }

  hash = 0u;
  total_size = 0u;
  std::array<uint8_t, 64u * 1024u> buffer = {};
  while (true) {
    file.read(reinterpret_cast<char*>(buffer.data()), static_cast<std::streamsize>(buffer.size()));
    const std::streamsize bytes_read = file.gcount();
    if (bytes_read > 0) {
      hash = etx_hash64_continue(buffer.data(), static_cast<uint64_t>(bytes_read), hash);
      total_size += static_cast<uint64_t>(bytes_read);
    }
    if (bytes_read < static_cast<std::streamsize>(buffer.size())) {
      break;
    }
  }
  if (file.bad() || (total_size == 0u)) {
    return false;
  }

  hash = etx_hash64_continue(&total_size, sizeof(total_size), hash);
  return true;
}

struct StagedSceneFile {
  std::filesystem::path destination;
  std::filesystem::path staged;
  std::filesystem::path backup;
  bool destination_existed = false;
  bool destination_backed_up = false;
  bool staged_installed = false;
};

struct SceneSavePaths {
  std::filesystem::path geometry;
  std::filesystem::path materials;
  std::filesystem::path json;
};

std::filesystem::path path_with_suffix(std::filesystem::path path, const char* suffix) {
  path += suffix;
  return path;
}

bool strip_scene_extension(std::string& name, const char* extension) {
  const size_t extension_length = std::strlen(extension);
  if ((name.size() < extension_length) || (name.compare(name.size() - extension_length, extension_length, extension) != 0)) {
    return false;
  }
  name.resize(name.size() - extension_length);
  return true;
}

SceneSavePaths scene_save_paths(const std::filesystem::path& source_path) {
  const std::filesystem::path normalized_source = source_path.lexically_normal();
  const std::filesystem::path base_directory = normalized_source.has_parent_path() ? normalized_source.parent_path() : std::filesystem::current_path();
  std::string base_name = normalized_source.filename().string();
  bool keep_stripping = true;
  while (keep_stripping) {
    keep_stripping = false;
    if (strip_scene_extension(base_name, ".json")) {
      keep_stripping = true;
    }
    if (strip_scene_extension(base_name, ".etx")) {
      keep_stripping = true;
    }
    if (strip_scene_extension(base_name, ".obj")) {
      keep_stripping = true;
    }
    if (strip_scene_extension(base_name, ".gltf")) {
      keep_stripping = true;
    }
    if (strip_scene_extension(base_name, ".glb")) {
      keep_stripping = true;
    }
  }
  if (base_name.empty()) {
    base_name = "scene";
  }

  return {
    .geometry = (base_directory / (base_name + ".etx")).lexically_normal(),
    .materials = (base_directory / (base_name + ".etx.materials")).lexically_normal(),
    .json = (base_directory / (base_name + ".etx.json")).lexically_normal(),
  };
}

std::array<StagedSceneFile, 3u> staged_scene_files(const SceneSavePaths& paths) {
  const auto staged_path = [](const std::filesystem::path& destination) {
    return path_with_suffix(destination, ".save-staged");
  };
  const auto backup_path = [](const std::filesystem::path& destination) {
    return path_with_suffix(destination, ".save-backup");
  };
  return {{
    {paths.geometry, staged_path(paths.geometry), backup_path(paths.geometry)},
    {paths.materials, staged_path(paths.materials), backup_path(paths.materials)},
    {paths.json, staged_path(paths.json), backup_path(paths.json)},
  }};
}

std::filesystem::path scene_save_marker(const std::array<StagedSceneFile, 3u>& files, const char* state) {
  std::filesystem::path marker = files[2].destination;
  marker += ".save-";
  marker += state;
  return marker;
}

bool inspect_scene_save_path(const std::filesystem::path& path, const char* description, bool& file_exists) {
  std::error_code error;
  const std::filesystem::file_status status = std::filesystem::symlink_status(path, error);
  if (status.type() == std::filesystem::file_type::not_found) {
    file_exists = false;
    return true;
  }
  if (error) {
    log::error("Failed to inspect %s: %s", description, path.string().c_str());
    return false;
  }
  if ((std::filesystem::is_regular_file(status) == false) && (std::filesystem::is_symlink(status) == false)) {
    log::error("Expected %s to be a file: %s", description, path.string().c_str());
    return false;
  }
  file_exists = true;
  return true;
}

bool remove_scene_save_file(const std::filesystem::path& path, const char* description) {
  std::error_code error;
  const std::filesystem::file_status status = std::filesystem::symlink_status(path, error);
  if (status.type() == std::filesystem::file_type::not_found) {
    return true;
  }
  if (error) {
    log::warning("Failed to inspect %s: %s", description, path.string().c_str());
    return false;
  }
  if ((std::filesystem::is_regular_file(status) == false) && (std::filesystem::is_symlink(status) == false)) {
    log::error("Refusing to remove non-file %s: %s", description, path.string().c_str());
    return false;
  }
  std::filesystem::remove(path, error);
  if (error) {
    log::warning("Failed to remove %s: %s", description, path.string().c_str());
    return false;
  }
  return true;
}

bool discard_staged_scene_files(std::array<StagedSceneFile, 3u>& files) {
  bool discarded = true;
  for (StagedSceneFile& file : files) {
    discarded = remove_scene_save_file(file.staged, "staged scene file") && discarded;
  }
  return discarded;
}

bool restore_scene_files(std::array<StagedSceneFile, 3u>& files) {
  bool restored = true;
  for (StagedSceneFile& file : files) {
    if (file.staged_installed == false) {
      continue;
    }
    std::error_code error;
    std::filesystem::remove(file.destination, error);
    if (error) {
      log::error("Failed to remove incomplete scene file during save rollback: %s", file.destination.string().c_str());
      restored = false;
      continue;
    }
    file.staged_installed = false;
  }

  for (auto file = files.rbegin(); file != files.rend(); ++file) {
    if (file->destination_backed_up == false) {
      continue;
    }
    std::error_code error;
    std::filesystem::rename(file->backup, file->destination, error);
    if (error) {
      log::error("Failed to restore scene file after save failure: %s", file->destination.string().c_str());
      restored = false;
      continue;
    }
    file->destination_backed_up = false;
  }

  restored = discard_staged_scene_files(files) && restored;
  return restored;
}

bool commit_staged_scene_files(std::array<StagedSceneFile, 3u>& files) {
  for (StagedSceneFile& file : files) {
    if (inspect_scene_save_path(file.destination, "scene destination", file.destination_existed) == false) {
      (void)restore_scene_files(files);
      return false;
    }
  }

  std::string transaction_state;
  transaction_state.reserve(files.size());
  for (const StagedSceneFile& file : files) {
    transaction_state.push_back(file.destination_existed ? '1' : '0');
  }
  const std::filesystem::path pending_marker = scene_save_marker(files, "pending");
  const std::filesystem::path committed_marker = scene_save_marker(files, "committed");
  const std::filesystem::path preparing_marker = scene_save_marker(files, "preparing");
  if (write_file_contents(preparing_marker, transaction_state) == false) {
    (void)restore_scene_files(files);
    remove_scene_save_file(preparing_marker, "scene save marker");
    return false;
  }
  std::error_code marker_error;
  std::filesystem::rename(preparing_marker, pending_marker, marker_error);
  if (marker_error) {
    log::error("Failed to start scene save transaction: %s", pending_marker.string().c_str());
    (void)restore_scene_files(files);
    remove_scene_save_file(preparing_marker, "scene save marker");
    return false;
  }

  for (StagedSceneFile& file : files) {
    if (file.destination_existed == false) {
      continue;
    }

    std::error_code error;
    std::filesystem::rename(file.destination, file.backup, error);
    if (error) {
      log::error("Failed to preserve scene file before saving: %s", file.destination.string().c_str());
      if (restore_scene_files(files)) {
        remove_scene_save_file(pending_marker, "scene save marker");
      }
      return false;
    }
    file.destination_backed_up = true;
  }

  for (StagedSceneFile& file : files) {
    std::error_code error;
    std::filesystem::rename(file.staged, file.destination, error);
    if (error) {
      log::error("Failed to install saved scene file: %s", file.destination.string().c_str());
      if (restore_scene_files(files)) {
        remove_scene_save_file(pending_marker, "scene save marker");
      }
      return false;
    }
    file.staged_installed = true;
  }

  marker_error.clear();
  std::filesystem::rename(pending_marker, committed_marker, marker_error);
  if (marker_error) {
    log::error("Failed to commit scene save transaction: %s", committed_marker.string().c_str());
    if (restore_scene_files(files)) {
      remove_scene_save_file(pending_marker, "scene save marker");
    }
    return false;
  }

  for (StagedSceneFile& file : files) {
    if (file.destination_backed_up == false) {
      continue;
    }
    std::error_code error;
    std::filesystem::remove(file.backup, error);
    if (error) {
      log::warning("Failed to remove scene save backup: %s", file.backup.string().c_str());
      continue;
    }
    file.destination_backed_up = false;
  }
  return true;
}

bool read_scene_save_state(const std::filesystem::path& marker, std::array<bool, 3u>& destination_existed) {
  FILE* file = fopen(marker.string().c_str(), "rb");
  if (file == nullptr) {
    log::error("Failed to open scene save marker: %s", marker.string().c_str());
    return false;
  }

  std::array<char, 3u> state = {};
  const size_t bytes_read = fread(state.data(), 1, state.size(), file);
  const bool close_succeeded = fclose(file) == 0;
  if ((bytes_read != state.size()) || (close_succeeded == false)) {
    log::error("Failed to read scene save marker: %s", marker.string().c_str());
    return false;
  }
  for (size_t index = 0u; index < state.size(); ++index) {
    if ((state[index] != '0') && (state[index] != '1')) {
      log::error("Invalid scene save marker: %s", marker.string().c_str());
      return false;
    }
    destination_existed[index] = state[index] == '1';
  }
  return true;
}

bool recover_interrupted_scene_save(std::array<StagedSceneFile, 3u>& files, bool& committed_scene_available) {
  committed_scene_available = false;
  const std::filesystem::path pending_marker = scene_save_marker(files, "pending");
  const std::filesystem::path committed_marker = scene_save_marker(files, "committed");
  const std::filesystem::path preparing_marker = scene_save_marker(files, "preparing");
  bool pending_exists = false;
  if (inspect_scene_save_path(pending_marker, "scene save marker", pending_exists) == false) {
    return false;
  }
  bool committed_exists = false;
  if (inspect_scene_save_path(committed_marker, "scene save marker", committed_exists) == false) {
    return false;
  }

  if (committed_exists) {
    bool complete_scene = true;
    for (const StagedSceneFile& file : files) {
      bool destination_exists = false;
      if (inspect_scene_save_path(file.destination, "committed scene file", destination_exists) == false) {
        return false;
      }
      if (destination_exists == false) {
        complete_scene = false;
      }
    }
    if (complete_scene) {
      bool cleanup_succeeded = true;
      for (StagedSceneFile& file : files) {
        cleanup_succeeded = remove_scene_save_file(file.backup, "scene save backup") && cleanup_succeeded;
      }
      cleanup_succeeded = discard_staged_scene_files(files) && cleanup_succeeded;
      cleanup_succeeded = remove_scene_save_file(preparing_marker, "scene save marker") && cleanup_succeeded;
      cleanup_succeeded = remove_scene_save_file(pending_marker, "scene save marker") && cleanup_succeeded;
      if (cleanup_succeeded) {
        (void)remove_scene_save_file(committed_marker, "scene save marker");
      }
      committed_scene_available = true;
      return true;
    }
  }

  if (pending_exists || committed_exists) {
    const std::filesystem::path& state_marker = pending_exists ? pending_marker : committed_marker;
    std::array<bool, 3u> destination_existed = {};
    if (read_scene_save_state(state_marker, destination_existed) == false) {
      return false;
    }

    bool recovered = true;
    for (size_t index = 0u; index < files.size(); ++index) {
      StagedSceneFile& file = files[index];
      bool backup_exists = false;
      if (inspect_scene_save_path(file.backup, "scene save backup", backup_exists) == false) {
        recovered = false;
        continue;
      }

      if (destination_existed[index]) {
        if (backup_exists) {
          if (remove_scene_save_file(file.destination, "incomplete scene file") == false) {
            log::error("Failed to remove incomplete scene file: %s", file.destination.string().c_str());
            recovered = false;
            continue;
          }
          std::error_code error;
          std::filesystem::rename(file.backup, file.destination, error);
          if (error) {
            log::error("Failed to recover scene save backup: %s", file.destination.string().c_str());
            recovered = false;
          }
          continue;
        }

        bool destination_exists = false;
        if ((inspect_scene_save_path(file.destination, "original scene file", destination_exists) == false) || (destination_exists == false)) {
          log::error("Original scene file is unavailable after an interrupted save: %s", file.destination.string().c_str());
          recovered = false;
        }
        continue;
      }

      if (remove_scene_save_file(file.destination, "incomplete scene file") == false) {
        log::error("Failed to remove incomplete scene file: %s", file.destination.string().c_str());
        recovered = false;
      }
      if (backup_exists) {
        recovered = remove_scene_save_file(file.backup, "unexpected scene save backup") && recovered;
      }
    }

    if (recovered == false) {
      return false;
    }
    (void)discard_staged_scene_files(files);
    remove_scene_save_file(preparing_marker, "scene save marker");
    remove_scene_save_file(pending_marker, "scene save marker");
    remove_scene_save_file(committed_marker, "scene save marker");
    log::warning("Recovered scene files after an interrupted save");
    return true;
  }

  for (StagedSceneFile& file : files) {
    bool backup_exists = false;
    if (inspect_scene_save_path(file.backup, "scene save backup", backup_exists) == false) {
      return false;
    }
    if (backup_exists == false) {
      continue;
    }
    bool destination_exists = false;
    if (inspect_scene_save_path(file.destination, "scene destination", destination_exists) == false) {
      return false;
    }
    if (destination_exists) {
      log::error("Ambiguous scene save backup requires manual recovery: %s", file.backup.string().c_str());
      return false;
    }
    std::error_code error;
    std::filesystem::rename(file.backup, file.destination, error);
    if (error) {
      log::error("Failed to recover orphaned scene save backup: %s", file.destination.string().c_str());
      return false;
    }
  }
  if (discard_staged_scene_files(files) == false) {
    return false;
  }
  remove_scene_save_file(preparing_marker, "scene save marker");
  return true;
}

NodeGeometryEditResult analyze_node_geometry_edit(const SceneData& data, uint32_t node_index, NodeGeometryOperation operation, bool validate_contents,
  NodeGeometryEditAnalysis* out_analysis) {
  if (node_index >= data.hierarchy.nodes.size()) {
    return NodeGeometryEditResult::InvalidNode;
  }

  const SceneNode& node = data.hierarchy.nodes[node_index];
  const uint32_t attachment_end = node.attachment_offset + node.attachment_count;
  if ((attachment_end < node.attachment_offset) || (attachment_end > data.hierarchy.attachments.size())) {
    return NodeGeometryEditResult::InvalidGeometry;
  }

  NodeGeometryEditAnalysis analysis = {};
  for (uint32_t attachment_index = node.attachment_offset; attachment_index < attachment_end; ++attachment_index) {
    const SceneAttachment& attachment = data.hierarchy.attachments[attachment_index];
    if (attachment.type != SceneAttachment::Type::Mesh) {
      return NodeGeometryEditResult::NonMeshAttachments;
    }
    if (attachment.resource_index >= data.meshes.size()) {
      return NodeGeometryEditResult::InvalidGeometry;
    }
    analysis.attachment_indices.push_back(attachment_index);
    if (std::find(analysis.mesh_indices.begin(), analysis.mesh_indices.end(), attachment.resource_index) == analysis.mesh_indices.end()) {
      analysis.mesh_indices.push_back(attachment.resource_index);
    }
  }
  if (analysis.attachment_indices.empty()) {
    return NodeGeometryEditResult::NoMeshAttachments;
  }

  const bool has_normals = data.vertices.nrm.empty() == false;
  const bool has_tangents = data.vertices.tan.empty() == false;
  const bool has_bitangents = data.vertices.btn.empty() == false;
  const bool has_texcoords = data.vertices.tex.empty() == false;
  if ((has_normals && (data.vertices.nrm.size() != data.vertices.pos.size())) || (has_tangents != has_bitangents) || (has_tangents && (has_normals == false)) ||
      (has_tangents && ((data.vertices.tan.size() != data.vertices.pos.size()) || (data.vertices.btn.size() != data.vertices.pos.size()))) ||
      (has_texcoords && (data.vertices.tex.size() != data.vertices.pos.size()))) {
    return NodeGeometryEditResult::InvalidGeometry;
  }

  uint64_t cloned_vertex_count = data.vertices.pos.size();
  uint64_t cloned_triangle_count = data.triangles.size();
  for (uint32_t mesh_index : analysis.mesh_indices) {
    const Mesh& mesh = data.meshes[mesh_index];
    const uint32_t triangle_end = mesh.triangle_offset + mesh.triangle_count;
    if ((mesh.triangle_count == 0u) || (triangle_end < mesh.triangle_offset) || (triangle_end > data.triangles.size())) {
      return NodeGeometryEditResult::InvalidGeometry;
    }
    cloned_triangle_count += mesh.triangle_count;
    if (cloned_triangle_count >= kInvalidIndex) {
      return NodeGeometryEditResult::InvalidGeometry;
    }
    if (validate_contents) {
      for (uint32_t triangle_index = mesh.triangle_offset; triangle_index < triangle_end; ++triangle_index) {
        const Triangle& triangle = data.triangles[triangle_index];
        for (uint32_t corner = 0u; corner < 3u; ++corner) {
          const uint32_t vertex_index = triangle.i[corner];
          if ((vertex_index >= data.vertices.pos.size()) || (has_normals && (vertex_index >= data.vertices.nrm.size())) ||
              (has_tangents && ((vertex_index >= data.vertices.tan.size()) || (vertex_index >= data.vertices.btn.size()))) ||
              (has_texcoords && (vertex_index >= data.vertices.tex.size())) || (finite_point(data.vertices.pos[vertex_index]) == false) ||
              (has_normals && (valid_direction(data.vertices.nrm[vertex_index]) == false)) ||
              (has_tangents && ((finite_point(data.vertices.tan[vertex_index]) == false) || (finite_point(data.vertices.btn[vertex_index]) == false))) ||
              (has_texcoords && ((std::isfinite(data.vertices.tex[vertex_index].x) == false) || (std::isfinite(data.vertices.tex[vertex_index].y) == false)))) {
            return NodeGeometryEditResult::InvalidGeometry;
          }
        }
      }
    }
    cloned_vertex_count += static_cast<uint64_t>(mesh.triangle_count) * 3u;
    if (cloned_vertex_count >= kInvalidIndex) {
      return NodeGeometryEditResult::InvalidGeometry;
    }
  }

  if (operation == NodeGeometryOperation::BakeLocalTransform) {
    AffineTransform inverse = {};
    double determinant = 0.0;
    if (invert_affine(node.local_transform, inverse, determinant) == false) {
      return NodeGeometryEditResult::SingularTransform;
    }
  }

  if (out_analysis != nullptr) {
    *out_analysis = std::move(analysis);
  }
  return NodeGeometryEditResult::Success;
}

NodeGeometryEditResult compute_node_surface_center(const SceneData& data, const NodeGeometryEditAnalysis& analysis, float3& center) {
  double weighted_x = 0.0;
  double weighted_y = 0.0;
  double weighted_z = 0.0;
  double total_area = 0.0;
  for (uint32_t attachment_index : analysis.attachment_indices) {
    const uint32_t mesh_index = data.hierarchy.attachments[attachment_index].resource_index;
    const Mesh& mesh = data.meshes[mesh_index];
    const uint32_t triangle_end = mesh.triangle_offset + mesh.triangle_count;
    for (uint32_t triangle_index = mesh.triangle_offset; triangle_index < triangle_end; ++triangle_index) {
      const Triangle& triangle = data.triangles[triangle_index];
      const float3& p0 = data.vertices.pos[triangle.i[0]];
      const float3& p1 = data.vertices.pos[triangle.i[1]];
      const float3& p2 = data.vertices.pos[triangle.i[2]];
      const double area = 0.5 * static_cast<double>(length(cross(p1 - p0, p2 - p0)));
      if ((area <= 0.0) || (std::isfinite(area) == false)) {
        continue;
      }
      const double scale = area / 3.0;
      weighted_x += scale * static_cast<double>(p0.x + p1.x + p2.x);
      weighted_y += scale * static_cast<double>(p0.y + p1.y + p2.y);
      weighted_z += scale * static_cast<double>(p0.z + p1.z + p2.z);
      total_area += area;
    }
  }
  if ((total_area <= 0.0) || (std::isfinite(total_area) == false)) {
    return NodeGeometryEditResult::DegenerateGeometry;
  }
  const double inverse_area = 1.0 / total_area;
  const double values[3] = {weighted_x * inverse_area, weighted_y * inverse_area, weighted_z * inverse_area};
  for (double value : values) {
    if ((std::isfinite(value) == false) || (std::abs(value) > static_cast<double>(std::numeric_limits<float>::max()))) {
      return NodeGeometryEditResult::DegenerateGeometry;
    }
  }
  center = {static_cast<float>(values[0]), static_cast<float>(values[1]), static_cast<float>(values[2])};
  return NodeGeometryEditResult::Success;
}

std::string source_mesh_name(const SceneData& data, uint32_t mesh_index) {
  std::string result;
  for (const auto& [name, index] : data.mesh_mapping) {
    if ((index == mesh_index) && (result.empty() || (name < result))) {
      result = name;
    }
  }
  return result.empty() ? ("mesh-" + std::to_string(mesh_index)) : result;
}

std::string unique_edited_mesh_name(const SceneData& data, uint32_t mesh_index, const std::vector<std::string>& reserved_names) {
  const std::string base = source_mesh_name(data, mesh_index) + "-edited";
  std::string result = base;
  uint32_t suffix = 1u;
  while (data.mesh_mapping.contains(result) || (std::find(reserved_names.begin(), reserved_names.end(), result) != reserved_names.end())) {
    result = base + "#" + std::to_string(suffix++);
  }
  return result;
}

struct PendingNodeGeometry {
  std::vector<float3> positions;
  std::vector<float3> normals;
  std::vector<float3> tangents;
  std::vector<float3> bitangents;
  std::vector<float2> texcoords;
  std::vector<Triangle> triangles;
  std::vector<Mesh> meshes;
  std::vector<std::string> mesh_names;
  std::unordered_map<uint32_t, uint32_t> source_to_clone;
};

NodeGeometryEditResult build_edited_meshes(const SceneData& data, const NodeGeometryEditAnalysis& analysis, NodeGeometryOperation operation,
  const AffineTransform& geometry_transform, const float3& center, PendingNodeGeometry& pending) {
  const bool has_normals = data.vertices.nrm.empty() == false;
  const bool has_tangents = data.vertices.tan.empty() == false;
  const bool has_texcoords = data.vertices.tex.empty() == false;

  AffineTransform inverse = {};
  double determinant = 1.0;
  SceneInstance bake_instance = {};
  if (operation == NodeGeometryOperation::BakeLocalTransform) {
    if (invert_affine(geometry_transform, inverse, determinant) == false) {
      return NodeGeometryEditResult::SingularTransform;
    }
    bake_instance.object_to_world = geometry_transform;
    bake_instance.world_to_object = inverse;
    if (determinant < 0.0) {
      bake_instance.flags |= SceneInstance::Mirrored;
    }
  }

  size_t maximum_new_vertices = 0u;
  size_t new_triangles = 0u;
  for (uint32_t source_mesh_index : analysis.mesh_indices) {
    const size_t triangle_count = data.meshes[source_mesh_index].triangle_count;
    maximum_new_vertices += triangle_count * 3u;
    new_triangles += triangle_count;
  }
  pending.positions.reserve(maximum_new_vertices);
  pending.normals.reserve(has_normals ? maximum_new_vertices : 0u);
  pending.tangents.reserve(has_tangents ? maximum_new_vertices : 0u);
  pending.bitangents.reserve(has_tangents ? maximum_new_vertices : 0u);
  pending.texcoords.reserve(has_texcoords ? maximum_new_vertices : 0u);
  pending.triangles.reserve(new_triangles);
  pending.meshes.reserve(analysis.mesh_indices.size());
  pending.mesh_names.reserve(analysis.mesh_indices.size());
  pending.source_to_clone.reserve(analysis.mesh_indices.size());

  for (uint32_t source_mesh_index : analysis.mesh_indices) {
    const Mesh& source_mesh = data.meshes[source_mesh_index];
    Mesh clone = {};
    clone.triangle_offset = static_cast<uint32_t>(data.triangles.size() + pending.triangles.size());
    clone.triangle_count = source_mesh.triangle_count;
    clone.bbox_min = {kMaxFloat, kMaxFloat, kMaxFloat};
    clone.bbox_max = {-kMaxFloat, -kMaxFloat, -kMaxFloat};

    std::unordered_map<uint32_t, uint32_t> vertex_mapping;
    const uint32_t triangle_end = source_mesh.triangle_offset + source_mesh.triangle_count;
    for (uint32_t triangle_index = source_mesh.triangle_offset; triangle_index < triangle_end; ++triangle_index) {
      Triangle triangle = data.triangles[triangle_index];
      for (uint32_t corner = 0u; corner < 3u; ++corner) {
        const uint32_t source_vertex_index = triangle.i[corner];
        auto existing = vertex_mapping.find(source_vertex_index);
        if (existing != vertex_mapping.end()) {
          triangle.i[corner] = existing->second;
          continue;
        }

        Vertex vertex = {};
        vertex.pos = data.vertices.pos[source_vertex_index];
        if (has_normals) {
          vertex.nrm = data.vertices.nrm[source_vertex_index];
        }
        if (has_tangents) {
          vertex.tan = data.vertices.tan[source_vertex_index];
          vertex.btn = data.vertices.btn[source_vertex_index];
        }
        if (has_texcoords) {
          vertex.tex = data.vertices.tex[source_vertex_index];
        }

        if (operation == NodeGeometryOperation::CenterPivot) {
          vertex.pos -= center;
        } else {
          vertex = scene_instance_transform_vertex(bake_instance, vertex);
        }
        if ((finite_point(vertex.pos) == false) || (has_normals && (valid_direction(vertex.nrm) == false)) ||
            (has_tangents && ((finite_point(vertex.tan) == false) || (finite_point(vertex.btn) == false)))) {
          return NodeGeometryEditResult::InvalidGeometry;
        }

        const uint32_t cloned_vertex_index = static_cast<uint32_t>(data.vertices.pos.size() + pending.positions.size());
        vertex_mapping.emplace(source_vertex_index, cloned_vertex_index);
        triangle.i[corner] = cloned_vertex_index;
        pending.positions.push_back(vertex.pos);
        if (has_normals) {
          pending.normals.push_back(vertex.nrm);
        }
        if (has_tangents) {
          pending.tangents.push_back(vertex.tan);
          pending.bitangents.push_back(vertex.btn);
        }
        if (has_texcoords) {
          pending.texcoords.push_back(vertex.tex);
        }
        clone.bbox_min = min(clone.bbox_min, vertex.pos);
        clone.bbox_max = max(clone.bbox_max, vertex.pos);
      }
      const float3& p0 = pending.positions[triangle.i[0] - data.vertices.pos.size()];
      const float3& p1 = pending.positions[triangle.i[1] - data.vertices.pos.size()];
      const float3& p2 = pending.positions[triangle.i[2] - data.vertices.pos.size()];
      const float3 geometric_normal = cross(p1 - p0, p2 - p0);
      if (dot(geometric_normal, geometric_normal) > 0.0f) {
        triangle.geo_n = normalize(geometric_normal);
      }
      pending.triangles.push_back(triangle);
    }

    const uint32_t clone_mesh_index = static_cast<uint32_t>(data.meshes.size() + pending.meshes.size());
    pending.source_to_clone.emplace(source_mesh_index, clone_mesh_index);
    pending.meshes.push_back(clone);
    pending.mesh_names.push_back(unique_edited_mesh_name(data, source_mesh_index, pending.mesh_names));
  }
  return NodeGeometryEditResult::Success;
}

void sanitize_camera_clip_planes(Camera& camera) {
  camera.clip_near = (camera.clip_near > 0.0f) ? camera.clip_near : kDefaultCameraClipNear;
  camera.clip_far = (camera.clip_far > camera.clip_near) ? camera.clip_far : max(camera.clip_near + 0.001f, kDefaultCameraClipFar);
}

Integrator::Type legacy_integrator_selection_to_type(const std::string& type_id) {
  if (type_id == "bdpt_distilled") {
    return Integrator::Type::Bidirectional;
  }

  return integrator_id_to_type(type_id.c_str());
}

std::string rename_entry(std::unordered_map<std::string, uint32_t>& mapping, uint32_t index, const char* desired_name, const char* fallback_prefix) {
  auto current = mapping.end();
  for (auto it = mapping.begin(); it != mapping.end(); ++it) {
    if (it->second == index) {
      current = it;
      break;
    }
  }
  if (current == mapping.end()) {
    return {};
  }

  std::string base = (desired_name != nullptr) ? desired_name : "";
  auto strip_prefix = [](const std::string& s) -> std::string {
    if (s.starts_with("etx::")) {
      return s.substr(5);
    }
    if (s.starts_with("et::")) {
      return s.substr(4);
    }
    return s;
  };
  base = strip_prefix(base);
  if (base.empty()) {
    base = current->first;
  }
  if (base.empty()) {
    base = std::string(fallback_prefix) + std::to_string(index);
  }

  std::string final = base;
  uint32_t suffix = 1;
  while (true) {
    auto found = mapping.find(final);
    if ((found == mapping.end()) || (found->second == index)) {
      break;
    }
    final = base + "#" + std::to_string(suffix++);
  }

  if (final != current->first) {
    mapping.erase(current);
    mapping.emplace(final, index);
  }
  return final;
}

std::string mapping_name(const std::unordered_map<std::string, uint32_t>& mapping, uint32_t index, const char* fallback_prefix) {
  for (const auto& [name, mapped_index] : mapping) {
    if (mapped_index == index) {
      return name;
    }
  }
  return std::string(fallback_prefix) + std::to_string(index);
}

std::string unique_mapping_name(const std::unordered_map<std::string, uint32_t>& mapping, const char* desired_name, const char* fallback_prefix) {
  const std::string base = ((desired_name != nullptr) && (desired_name[0] != 0)) ? desired_name : fallback_prefix;
  std::string result = base;
  uint32_t suffix = 2u;
  while (mapping.contains(result)) {
    result = base + " " + std::to_string(suffix++);
  }
  return result;
}

std::vector<uint32_t> removal_remapping(size_t resource_count, uint32_t removed_index) {
  std::vector<uint32_t> result(resource_count);
  for (uint32_t old_index = 0u; old_index < resource_count; ++old_index) {
    result[old_index] = old_index < removed_index ? old_index : ((old_index == removed_index) ? kInvalidIndex : old_index - 1u);
  }
  return result;
}

void erase_mapping_resource(std::unordered_map<std::string, uint32_t>& mapping, uint32_t removed_index) {
  for (auto it = mapping.begin(); it != mapping.end();) {
    if (it->second == removed_index) {
      it = mapping.erase(it);
      continue;
    }
    if ((it->second != kInvalidIndex) && (it->second > removed_index)) {
      --it->second;
    }
    ++it;
  }
}

void remap_hierarchy_resource(SceneHierarchy& hierarchy, SceneAttachment::Type type, uint32_t removed_index) {
  for (SceneAttachment& attachment : hierarchy.attachments) {
    if ((attachment.type == type) && (attachment.resource_index != kInvalidIndex) && (attachment.resource_index > removed_index)) {
      --attachment.resource_index;
    }
  }
}

std::string unique_named_resource(const std::vector<std::string>& names, const char* desired_name, const char* fallback, uint32_t excluded_index) {
  const std::string base = ((desired_name != nullptr) && (desired_name[0] != 0)) ? desired_name : fallback;
  auto available = [&](const std::string& candidate) {
    for (uint32_t index = 0u; index < names.size(); ++index) {
      if ((index != excluded_index) && (names[index] == candidate)) {
        return false;
      }
    }
    return true;
  };

  std::string result = base;
  uint32_t suffix = 2u;
  while (available(result) == false) {
    result = base + " " + std::to_string(suffix++);
  }
  return result;
}

const char* default_emitter_name(const EmitterProfile& profile) {
  if (profile.cls == EmitterProfile::Class::Area) {
    return "Area Light";
  }
  if (profile.cls == EmitterProfile::Class::Environment) {
    return (profile.meta & EmitterProfile::Meta::Atmosphere) ? "Atmosphere" : "Environment Light";
  }
  return "Directional Light";
}

void ensure_emitter_names(SceneData& data) {
  if (data.emitter_names.size() > data.emitter_profiles.size()) {
    data.emitter_names.resize(data.emitter_profiles.size());
  }
  while (data.emitter_names.size() < data.emitter_profiles.size()) {
    const uint32_t emitter_index = static_cast<uint32_t>(data.emitter_names.size());
    const char* fallback = default_emitter_name(data.emitter_profiles[emitter_index]);
    data.emitter_names.push_back(unique_named_resource(data.emitter_names, fallback, fallback, kInvalidIndex));
  }
}

std::vector<uint32_t> serialized_emitter_indices(const SceneData& data) {
  std::vector<uint32_t> result;
  result.reserve(data.emitter_profiles.size());
  for (uint32_t emitter_index = 0u; emitter_index < data.emitter_profiles.size(); ++emitter_index) {
    const EmitterProfile& emitter = data.emitter_profiles[emitter_index];
    if ((emitter.cls == EmitterProfile::Class::Environment) && ((emitter.meta & EmitterProfile::Meta::Atmosphere) != 0u)) {
      result.push_back(emitter_index);
    }
  }
  for (uint32_t emitter_index = 0u; emitter_index < data.emitter_profiles.size(); ++emitter_index) {
    const EmitterProfile& emitter = data.emitter_profiles[emitter_index];
    if ((emitter.cls == EmitterProfile::Class::Environment) && ((emitter.meta & EmitterProfile::Meta::Atmosphere) == 0u)) {
      result.push_back(emitter_index);
    }
  }
  for (uint32_t emitter_index = 0u; emitter_index < data.emitter_profiles.size(); ++emitter_index) {
    if (data.emitter_profiles[emitter_index].cls == EmitterProfile::Class::Directional) {
      result.push_back(emitter_index);
    }
  }
  return result;
}

bool internal_scene_resource_name(const std::string& name) {
  return (name.compare(0, 4, "et::") == 0) || (name.compare(0, 5, "etx::") == 0);
}

struct SerializedMaterialEntry {
  uint32_t material_index = kInvalidIndex;
  std::string id;
  std::vector<std::string> authored_names;
};

struct SerializedMediumEntry {
  uint32_t medium_index = kInvalidIndex;
  std::string id;
  std::string authored_name;
};

struct SerializedCameraEntry {
  uint32_t camera_index = kInvalidIndex;
  std::string id;
  std::string authored_name;
};

bool build_serialized_material_entries(const SceneData& data, std::vector<SerializedMaterialEntry>& result) {
  std::unordered_map<uint32_t, uint32_t> entry_mapping;
  result.clear();
  result.reserve(data.material_mapping.size());
  for (const auto& mapping : data.material_mapping) {
    if (internal_scene_resource_name(mapping.first)) {
      continue;
    }
    if (mapping.second >= data.materials.size()) {
      log::error("Cannot save material %s: material index %u is invalid", mapping.first.c_str(), mapping.second);
      return false;
    }

    const auto existing = entry_mapping.find(mapping.second);
    if (existing != entry_mapping.end()) {
      result[existing->second].authored_names.push_back(mapping.first);
      continue;
    }

    const uint32_t entry_index = static_cast<uint32_t>(result.size());
    entry_mapping[mapping.second] = entry_index;
    SerializedMaterialEntry& entry = result.emplace_back();
    entry.material_index = mapping.second;
    entry.id = "saved_material_" + std::to_string(mapping.second);
    entry.authored_names.push_back(mapping.first);
  }

  for (SerializedMaterialEntry& entry : result) {
    std::sort(entry.authored_names.begin(), entry.authored_names.end());
  }
  std::sort(result.begin(), result.end(), [](const SerializedMaterialEntry& a, const SerializedMaterialEntry& b) {
    return a.material_index < b.material_index;
  });
  return true;
}

bool build_serialized_medium_entries(const SceneData& data, std::vector<SerializedMediumEntry>& result) {
  if (data.mediums.mapping().size() != data.mediums.array_size()) {
    log::error("Cannot save media: the name mapping does not match the medium array");
    return false;
  }

  std::unordered_set<uint32_t> mapped_indices;
  result.clear();
  result.reserve(data.mediums.mapping().size());
  for (const auto& [name, medium_index] : data.mediums.mapping()) {
    if ((name.empty()) || (medium_index >= data.mediums.array_size()) || (mapped_indices.insert(medium_index).second == false)) {
      log::error("Cannot save medium %s: its name mapping is invalid", name.c_str());
      return false;
    }

    SerializedMediumEntry& entry = result.emplace_back();
    entry.medium_index = medium_index;
    entry.id = "saved_medium_" + std::to_string(medium_index);
    entry.authored_name = name;
  }

  std::sort(result.begin(), result.end(), [](const SerializedMediumEntry& a, const SerializedMediumEntry& b) {
    return a.medium_index < b.medium_index;
  });
  return true;
}

std::vector<SerializedCameraEntry> build_serialized_camera_entries(const SceneData& data) {
  std::vector<SerializedCameraEntry> result;
  result.reserve(data.cameras.size());
  for (uint32_t camera_index = 0u; camera_index < data.cameras.size(); ++camera_index) {
    SerializedCameraEntry& entry = result.emplace_back();
    entry.camera_index = camera_index;
    entry.id = "saved_camera_" + std::to_string(camera_index);
    entry.authored_name = data.cameras[camera_index].id;
  }
  return result;
}

bool json_float_value(const nlohmann::json& value, float& result) {
  if (value.is_number() == false) {
    return false;
  }

  double parsed = 0.0;
  try {
    parsed = value.get<double>();
  } catch (const nlohmann::json::exception&) {
    return false;
  }
  const double maximum = static_cast<double>(std::numeric_limits<float>::max());
  if ((std::isfinite(parsed) == false) || (parsed < -maximum) || (parsed > maximum)) {
    return false;
  }

  result = static_cast<float>(parsed);
  return std::isfinite(result);
}

nlohmann::json serialize_spectral_distribution(const SpectralDistribution& spectrum) {
  nlohmann::json result = nlohmann::json::object();
  result["integrated"] = {spectrum.integrated_value.x, spectrum.integrated_value.y, spectrum.integrated_value.z};
  nlohmann::json samples = nlohmann::json::array();
  for (uint32_t sample_index = 0u; sample_index < spectrum.spectral_entry_count; ++sample_index) {
    samples.push_back(spectrum.spectral_entries[sample_index].wavelength);
    samples.push_back(spectrum.spectral_entries[sample_index].power);
  }
  result["samples"] = std::move(samples);
  return result;
}

bool spectral_distribution_serializable(const SpectralDistribution& spectrum) {
  if ((spectrum.spectral_entry_count > WavelengthCount) || (finite_point(spectrum.integrated_value) == false)) {
    return false;
  }
  float previous_wavelength = -std::numeric_limits<float>::infinity();
  for (uint32_t sample_index = 0u; sample_index < spectrum.spectral_entry_count; ++sample_index) {
    const SpectralDistribution::Entry& entry = spectrum.spectral_entries[sample_index];
    if ((std::isfinite(entry.wavelength) == false) || (std::isfinite(entry.power) == false) || (entry.wavelength <= previous_wavelength)) {
      return false;
    }
    previous_wavelength = entry.wavelength;
  }
  return true;
}

bool deserialize_spectral_distribution(const nlohmann::json& source, SpectralDistribution& result) {
  if ((source.is_object() == false) || (source.contains("integrated") == false) || (source["integrated"].is_array() == false) || (source["integrated"].size() != 3u) ||
      (source.contains("samples") == false) || (source["samples"].is_array() == false) || ((source["samples"].size() % 2u) != 0u) ||
      ((source["samples"].size() / 2u) > WavelengthCount)) {
    return false;
  }

  result = {};
  if ((json_float_value(source["integrated"][0u], result.integrated_value.x) == false) || (json_float_value(source["integrated"][1u], result.integrated_value.y) == false) ||
      (json_float_value(source["integrated"][2u], result.integrated_value.z) == false)) {
    return false;
  }

  const nlohmann::json& samples = source["samples"];
  result.spectral_entry_count = static_cast<uint32_t>(samples.size() / 2u);
  float previous_wavelength = -std::numeric_limits<float>::infinity();
  for (uint32_t sample_index = 0u; sample_index < result.spectral_entry_count; ++sample_index) {
    SpectralDistribution::Entry& entry = result.spectral_entries[sample_index];
    if ((json_float_value(samples[2u * sample_index], entry.wavelength) == false) || (json_float_value(samples[2u * sample_index + 1u], entry.power) == false) ||
        (entry.wavelength <= previous_wavelength)) {
      return false;
    }
    previous_wavelength = entry.wavelength;
  }
  return true;
}

nlohmann::json serialize_scene_spectral_overrides(const SceneData& data, const std::vector<SerializedMaterialEntry>& material_entries,
  const std::vector<SerializedMediumEntry>& medium_entries, bool& valid) {
  valid = true;
  nlohmann::json values = nlohmann::json::array();
  std::unordered_map<uint32_t, uint32_t> spectrum_mapping;

  auto spectrum_reference = [&](uint32_t spectrum_index, const char* context) -> nlohmann::json {
    if (spectrum_index == kInvalidIndex) {
      return nullptr;
    }
    if (spectrum_index >= data.spectrum_values.size()) {
      log::error("Cannot save %s: spectrum index %u is invalid", context, spectrum_index);
      valid = false;
      return nullptr;
    }
    if (spectral_distribution_serializable(data.spectrum_values[spectrum_index]) == false) {
      log::error("Cannot save %s: spectrum index %u contains invalid data", context, spectrum_index);
      valid = false;
      return nullptr;
    }
    const auto existing = spectrum_mapping.find(spectrum_index);
    if (existing != spectrum_mapping.end()) {
      return existing->second;
    }
    const uint32_t serialized_index = static_cast<uint32_t>(values.size());
    spectrum_mapping[spectrum_index] = serialized_index;
    values.push_back(serialize_spectral_distribution(data.spectrum_values[spectrum_index]));
    return serialized_index;
  };

  auto serialize_ior = [&](const RefractiveIndex& ior, const char* context) {
    return nlohmann::json{
      {"class", ior.cls},
      {"eta", spectrum_reference(ior.eta_index, context)},
      {"k", spectrum_reference(ior.k_index, context)},
    };
  };

  nlohmann::json materials = nlohmann::json::array();
  for (const SerializedMaterialEntry& entry : material_entries) {
    const Material& material = data.materials[entry.material_index];
    materials.push_back({
      {"name", entry.id},
      {"reflectance", spectrum_reference(material.reflectance.spectrum_index, "material reflectance")},
      {"scattering", spectrum_reference(material.scattering.spectrum_index, "material scattering")},
      {"emission", spectrum_reference(material.emission.spectrum_index, "material emission")},
      {"subsurface", spectrum_reference(material.subsurface.spectrum_index, "material subsurface")},
      {"external_ior", serialize_ior(material.ext_ior, "external IOR")},
      {"internal_ior", serialize_ior(material.int_ior, "internal IOR")},
      {"thinfilm_ior", serialize_ior(material.thinfilm.ior, "thin-film IOR")},
    });
  }

  nlohmann::json mediums = nlohmann::json::array();
  for (const SerializedMediumEntry& entry : medium_entries) {
    if (entry.medium_index >= data.mediums.array_size()) {
      log::error("Cannot save medium %s: medium index %u is invalid", entry.authored_name.c_str(), entry.medium_index);
      valid = false;
      continue;
    }
    const Medium& medium = data.mediums.get(entry.medium_index);
    mediums.push_back({
      {"name", entry.id},
      {"absorption", spectrum_reference(medium.absorption_index, "medium absorption")},
      {"scattering", spectrum_reference(medium.scattering_index, "medium scattering")},
    });
  }

  nlohmann::json emitters = nlohmann::json::array();
  for (const uint32_t emitter_index : serialized_emitter_indices(data)) {
    const EmitterProfile& emitter = data.emitter_profiles[emitter_index];
    emitters.push_back(spectrum_reference(emitter.emission.spectrum_index, "emitter emission"));
  }

  return {
    {"version", 1u},
    {"values", std::move(values)},
    {"materials", std::move(materials)},
    {"mediums", std::move(mediums)},
    {"emitters", std::move(emitters)},
  };
}

bool apply_scene_spectral_overrides(const nlohmann::json& source, SceneData& data) {
  if ((source.is_object() == false) || (source.contains("version") == false) || (source["version"].is_number_unsigned() == false) || (source["version"].get<uint64_t>() != 1u) ||
      (source.contains("values") == false) || (source["values"].is_array() == false) || (source.contains("materials") == false) || (source["materials"].is_array() == false) ||
      (source.contains("mediums") == false) || (source["mediums"].is_array() == false) || (source.contains("emitters") == false) || (source["emitters"].is_array() == false)) {
    return false;
  }

  std::vector<SpectralDistribution> decoded_values;
  decoded_values.reserve(source["values"].size());
  for (const nlohmann::json& value : source["values"]) {
    SpectralDistribution& decoded = decoded_values.emplace_back();
    if (deserialize_spectral_distribution(value, decoded) == false) {
      return false;
    }
  }

  std::vector<uint32_t> spectrum_mapping;
  spectrum_mapping.reserve(decoded_values.size());
  for (const SpectralDistribution& value : decoded_values) {
    spectrum_mapping.push_back(data.add_spectrum(value));
  }

  auto resolve_reference = [&](const nlohmann::json& reference, uint32_t& result) {
    if (reference.is_null()) {
      result = kInvalidIndex;
      return true;
    }
    if (reference.is_number_unsigned() == false) {
      return false;
    }
    const uint64_t serialized_index = reference.get<uint64_t>();
    if (serialized_index >= spectrum_mapping.size()) {
      return false;
    }
    result = spectrum_mapping[serialized_index];
    return true;
  };

  auto apply_ior = [&](const nlohmann::json& serialized, RefractiveIndex& result) {
    if ((serialized.is_object() == false) || (serialized.contains("class") == false) || (serialized["class"].is_number_unsigned() == false) ||
        (serialized["class"].get<uint64_t>() > SpectralDistribution::Illuminant) || (serialized.contains("eta") == false) || (serialized.contains("k") == false)) {
      return false;
    }
    result.cls = serialized["class"].get<uint32_t>();
    return resolve_reference(serialized["eta"], result.eta_index) && resolve_reference(serialized["k"], result.k_index);
  };

  size_t expected_material_count = 0u;
  for (const auto& mapping : data.material_mapping) {
    if (internal_scene_resource_name(mapping.first) == false) {
      ++expected_material_count;
    }
  }
  if (source["materials"].size() != expected_material_count) {
    return false;
  }
  std::unordered_set<std::string> restored_materials;
  for (const nlohmann::json& serialized : source["materials"]) {
    if ((serialized.is_object() == false) || (serialized.contains("name") == false) || (serialized["name"].is_string() == false) || (serialized.contains("reflectance") == false) ||
        (serialized.contains("scattering") == false) || (serialized.contains("emission") == false) || (serialized.contains("subsurface") == false) ||
        (serialized.contains("external_ior") == false) || (serialized.contains("internal_ior") == false) || (serialized.contains("thinfilm_ior") == false)) {
      return false;
    }
    const std::string& name = serialized["name"].get_ref<const std::string&>();
    if (restored_materials.insert(name).second == false) {
      return false;
    }
    const auto mapping = data.material_mapping.find(name);
    if ((mapping == data.material_mapping.end()) || (mapping->second >= data.materials.size())) {
      return false;
    }
    Material& material = data.materials[mapping->second];
    if ((resolve_reference(serialized["reflectance"], material.reflectance.spectrum_index) == false) ||
        (resolve_reference(serialized["scattering"], material.scattering.spectrum_index) == false) ||
        (resolve_reference(serialized["emission"], material.emission.spectrum_index) == false) ||
        (resolve_reference(serialized["subsurface"], material.subsurface.spectrum_index) == false) || (apply_ior(serialized["external_ior"], material.ext_ior) == false) ||
        (apply_ior(serialized["internal_ior"], material.int_ior) == false) || (apply_ior(serialized["thinfilm_ior"], material.thinfilm.ior) == false)) {
      return false;
    }
  }

  if (source["mediums"].size() != data.mediums.mapping().size()) {
    return false;
  }
  std::unordered_set<std::string> restored_mediums;
  for (const nlohmann::json& serialized : source["mediums"]) {
    if ((serialized.is_object() == false) || (serialized.contains("name") == false) || (serialized["name"].is_string() == false) || (serialized.contains("absorption") == false) ||
        (serialized.contains("scattering") == false)) {
      return false;
    }
    const std::string& name = serialized["name"].get_ref<const std::string&>();
    if (restored_mediums.insert(name).second == false) {
      return false;
    }
    const auto mapping = data.mediums.mapping().find(name);
    if ((mapping == data.mediums.mapping().end()) || (mapping->second >= data.mediums.array_size())) {
      return false;
    }
    Medium& medium = data.mediums.get(mapping->second);
    if ((resolve_reference(serialized["absorption"], medium.absorption_index) == false) || (resolve_reference(serialized["scattering"], medium.scattering_index) == false)) {
      return false;
    }
  }

  const std::vector<uint32_t> emitter_indices = serialized_emitter_indices(data);
  if (source["emitters"].size() != emitter_indices.size()) {
    return false;
  }
  for (uint32_t serialized_index = 0u; serialized_index < emitter_indices.size(); ++serialized_index) {
    EmitterProfile& emitter = data.emitter_profiles[emitter_indices[serialized_index]];
    if (resolve_reference(source["emitters"][serialized_index], emitter.emission.spectrum_index) == false) {
      return false;
    }
  }
  return true;
}

bool restore_scene_material_names(const nlohmann::json& source, SceneData& data) {
  if (source.is_array() == false) {
    return false;
  }

  SceneRepresentation::MaterialMapping restored_mapping;
  restored_mapping.reserve(data.material_mapping.size());
  for (const auto& mapping : data.material_mapping) {
    if (internal_scene_resource_name(mapping.first)) {
      restored_mapping.emplace(mapping);
    }
  }

  std::unordered_set<std::string> serialized_ids;
  std::unordered_set<std::string> authored_names;
  for (const nlohmann::json& binding : source) {
    if ((binding.is_object() == false) || (binding.contains("id") == false) || (binding["id"].is_string() == false) || (binding.contains("names") == false) ||
        (binding["names"].is_array() == false) || binding["names"].empty()) {
      return false;
    }

    const std::string& id = binding["id"].get_ref<const std::string&>();
    if (id.empty() || internal_scene_resource_name(id) || (serialized_ids.insert(id).second == false)) {
      return false;
    }
    const auto material = data.material_mapping.find(id);
    if ((material == data.material_mapping.end()) || (material->second >= data.materials.size())) {
      return false;
    }
    for (const nlohmann::json& serialized_name : binding["names"]) {
      if (serialized_name.is_string() == false) {
        return false;
      }
      const std::string& name = serialized_name.get_ref<const std::string&>();
      if (name.empty() || internal_scene_resource_name(name) || (authored_names.insert(name).second == false)) {
        return false;
      }
      restored_mapping[name] = material->second;
    }
  }

  for (const auto& mapping : data.material_mapping) {
    if ((internal_scene_resource_name(mapping.first) == false) && (serialized_ids.count(mapping.first) == 0u)) {
      return false;
    }
  }

  data.material_mapping = std::move(restored_mapping);
  return true;
}

bool restore_scene_medium_names(const nlohmann::json& source, SceneData& data) {
  if ((source.is_array() == false) || (source.size() != data.mediums.mapping().size()) || (source.size() != data.mediums.array_size())) {
    return false;
  }

  MediumPool::Mapping restored_mapping;
  restored_mapping.reserve(source.size());
  std::unordered_set<std::string> serialized_ids;
  for (const nlohmann::json& binding : source) {
    if ((binding.is_object() == false) || (binding.contains("id") == false) || (binding["id"].is_string() == false) || (binding.contains("name") == false) ||
        (binding["name"].is_string() == false)) {
      return false;
    }

    const std::string& id = binding["id"].get_ref<const std::string&>();
    const std::string& name = binding["name"].get_ref<const std::string&>();
    if (id.empty() || name.empty() || (serialized_ids.insert(id).second == false)) {
      return false;
    }
    const auto medium = data.mediums.mapping().find(id);
    if ((medium == data.mediums.mapping().end()) || (medium->second >= data.mediums.array_size()) || (restored_mapping.emplace(name, medium->second).second == false)) {
      return false;
    }
  }

  for (const auto& mapping : data.mediums.mapping()) {
    if (serialized_ids.count(mapping.first) == 0u) {
      return false;
    }
  }
  return data.mediums.replace_mapping(std::move(restored_mapping));
}

bool restore_scene_camera_names(const nlohmann::json& source, SceneData& data) {
  if ((source.is_array() == false) || (source.size() != data.cameras.size())) {
    return false;
  }

  std::unordered_map<std::string, uint32_t> camera_indices;
  camera_indices.reserve(data.cameras.size());
  for (uint32_t camera_index = 0u; camera_index < data.cameras.size(); ++camera_index) {
    if ((data.cameras[camera_index].id.empty()) || (camera_indices.emplace(data.cameras[camera_index].id, camera_index).second == false)) {
      return false;
    }
  }

  std::vector<std::string> restored_names(data.cameras.size());
  std::vector<bool> restored_indices(data.cameras.size(), false);
  for (const nlohmann::json& binding : source) {
    if ((binding.is_object() == false) || (binding.contains("id") == false) || (binding["id"].is_string() == false) || (binding.contains("name") == false) ||
        (binding["name"].is_string() == false)) {
      return false;
    }

    const std::string& id = binding["id"].get_ref<const std::string&>();
    const auto camera = camera_indices.find(id);
    if ((camera == camera_indices.end()) || restored_indices[camera->second]) {
      return false;
    }
    restored_names[camera->second] = binding["name"].get<std::string>();
    restored_indices[camera->second] = true;
  }

  for (uint32_t camera_index = 0u; camera_index < data.cameras.size(); ++camera_index) {
    if (restored_indices[camera_index] == false) {
      return false;
    }
    data.cameras[camera_index].id = std::move(restored_names[camera_index]);
  }
  return true;
}

uint32_t clone_spectrum(SceneData& data, uint32_t spectrum_index) {
  return spectrum_index < data.spectrum_values.size() ? data.add_spectrum(data.spectrum_values[spectrum_index]) : kInvalidIndex;
}

Material clone_material_resources(SceneData& data, const Material& source) {
  Material result = source;
  result.reflectance.spectrum_index = clone_spectrum(data, source.reflectance.spectrum_index);
  result.scattering.spectrum_index = clone_spectrum(data, source.scattering.spectrum_index);
  result.emission.spectrum_index = clone_spectrum(data, source.emission.spectrum_index);
  result.subsurface.spectrum_index = clone_spectrum(data, source.subsurface.spectrum_index);
  result.thinfilm.ior.eta_index = clone_spectrum(data, source.thinfilm.ior.eta_index);
  result.thinfilm.ior.k_index = clone_spectrum(data, source.thinfilm.ior.k_index);
  result.ext_ior.eta_index = clone_spectrum(data, source.ext_ior.eta_index);
  result.ext_ior.k_index = clone_spectrum(data, source.ext_ior.k_index);
  result.int_ior.eta_index = clone_spectrum(data, source.int_ior.eta_index);
  result.int_ior.k_index = clone_spectrum(data, source.int_ior.k_index);
  result.energy_compensation_interface_index = kInvalidIndex;
  result.conductor_energy_compensation_interface_index = kInvalidIndex;
  return result;
}

std::string unique_node_name(const SceneHierarchy& hierarchy, const char* desired_name) {
  const std::string base = ((desired_name != nullptr) && (desired_name[0] != 0)) ? desired_name : "Empty Node";
  auto available = [&](const std::string& candidate) {
    return std::find(hierarchy.node_names.begin(), hierarchy.node_names.end(), candidate) == hierarchy.node_names.end();
  };
  if (available(base)) {
    return base;
  }

  uint32_t suffix = 2u;
  std::string candidate;
  do {
    candidate = base + " " + std::to_string(suffix++);
  } while (available(candidate) == false);
  return candidate;
}

std::string unique_renamed_node_name(const SceneHierarchy& hierarchy, uint32_t node_index, const char* desired_name) {
  if (node_index >= hierarchy.nodes.size()) {
    return {};
  }
  const std::string current_name = (node_index < hierarchy.node_names.size()) ? hierarchy.node_names[node_index] : std::string{};
  const std::string base = ((desired_name != nullptr) && (desired_name[0] != 0)) ? desired_name : (current_name.empty() ? "Node" : current_name);
  auto available = [&](const std::string& candidate) {
    for (uint32_t index = 0u; index < hierarchy.node_names.size(); ++index) {
      if ((index != node_index) && (hierarchy.node_names[index] == candidate)) {
        return false;
      }
    }
    return true;
  };
  if (available(base)) {
    return base;
  }

  uint32_t suffix = 2u;
  std::string candidate;
  do {
    candidate = base + " " + std::to_string(suffix++);
  } while (available(candidate) == false);
  return candidate;
}

bool valid_attachment_resource(const SceneData& data, SceneAttachment::Type type, uint32_t resource_index) {
  switch (type) {
    case SceneAttachment::Type::Mesh:
      return resource_index < data.meshes.size();
    case SceneAttachment::Type::Camera:
      return resource_index < data.cameras.size();
    case SceneAttachment::Type::Emitter:
      return (resource_index < data.emitter_profiles.size()) && (data.emitter_profiles[resource_index].cls != EmitterProfile::Class::Area);
    case SceneAttachment::Type::Medium:
      return resource_index < data.mediums.array_size();
  }
  return false;
}

bool resource_is_attached(const SceneHierarchy& hierarchy, SceneAttachment::Type type, uint32_t resource_index, uint32_t excluded_node_index) {
  for (uint32_t node_index = 0u; node_index < hierarchy.nodes.size(); ++node_index) {
    if (node_index == excluded_node_index) {
      continue;
    }
    const SceneNode& node = hierarchy.nodes[node_index];
    const uint32_t attachment_end = node.attachment_offset + node.attachment_count;
    if ((attachment_end < node.attachment_offset) || (attachment_end > hierarchy.attachments.size())) {
      continue;
    }
    for (uint32_t attachment_index = node.attachment_offset; attachment_index < attachment_end; ++attachment_index) {
      const SceneAttachment& attachment = hierarchy.attachments[attachment_index];
      if ((attachment.type == type) && (attachment.resource_index == resource_index)) {
        return true;
      }
    }
  }
  return false;
}

bool subtree_contains_active_camera(const SceneData& data, uint32_t node_index) {
  const SceneHierarchy& hierarchy = data.hierarchy;
  if (node_index >= hierarchy.nodes.size()) {
    return false;
  }

  for (uint32_t camera_node_index = 0u; camera_node_index < hierarchy.nodes.size(); ++camera_node_index) {
    const SceneNode& camera_node = hierarchy.nodes[camera_node_index];
    const uint32_t attachment_end = camera_node.attachment_offset + camera_node.attachment_count;
    if ((attachment_end < camera_node.attachment_offset) || (attachment_end > hierarchy.attachments.size())) {
      continue;
    }
    bool active_camera_attached = false;
    for (uint32_t attachment_index = camera_node.attachment_offset; attachment_index < attachment_end; ++attachment_index) {
      const SceneAttachment& attachment = hierarchy.attachments[attachment_index];
      if ((attachment.type == SceneAttachment::Type::Camera) && (attachment.resource_index < data.cameras.size()) && data.cameras[attachment.resource_index].active) {
        active_camera_attached = true;
        break;
      }
    }
    if (active_camera_attached == false) {
      continue;
    }

    uint32_t ancestor_index = camera_node_index;
    for (uint32_t depth = 0u; (depth <= hierarchy.nodes.size()) && (ancestor_index != kInvalidIndex); ++depth) {
      if (ancestor_index == node_index) {
        return true;
      }
      if (ancestor_index >= hierarchy.nodes.size()) {
        break;
      }
      ancestor_index = hierarchy.nodes[ancestor_index].parent_index;
    }
  }
  return false;
}

bool scene_has_environment_emitter(const SceneData& data) {
  for (const auto& profile : data.emitter_profiles) {
    if (profile.cls == EmitterProfile::Class::Environment) {
      return true;
    }
  }
  return false;
}

void add_default_raw_model_lighting(SceneData& data) {
  scattering::Parameters scattering_params = {};
  scattering_params.altitude = 1000.0f;
  scattering_params.anisotropy = 0.825f;
  scattering_params.rayleigh_scale = 1.0f;
  scattering_params.mie_scale = 1.0f;
  scattering_params.ozone_scale = 1.0f;

  const uint32_t atmosphere_index = data.add_atmosphere_emitter({scattering_params, kDefaultModelAtmosphereQuality});

  auto& sun = data.emitter_profiles.emplace_back(EmitterProfile::Class::Directional);
  sun.emission.spectrum_index = data.add_spectrum(SpectralDistribution::rgb_luminance({1.0f, 1.0f, 1.0f}));
  sun.emission.image_index = kInvalidIndex;
  sun.directional.direction = normalize(float3{0.0f, 1.0f, 1.0f});
  sun.directional.angular_size = kDefaultModelSunAngularDiameter * kPi / 180.0f;
  sun.directional.equivalent_disk_size = 2.0f * std::tan(sun.directional.angular_size * 0.5f);
  sun.directional.angular_size_cosine = std::cos(sun.directional.angular_size * 0.5f);
  sun.reference_emitter_index = atmosphere_index;
  sun.medium_index = kInvalidIndex;
}

}  // namespace

const char* node_geometry_edit_result_message(NodeGeometryEditResult result) {
  switch (result) {
    case NodeGeometryEditResult::Success:
      return "";
    case NodeGeometryEditResult::InvalidNode:
      return "The selected node is invalid.";
    case NodeGeometryEditResult::NoMeshAttachments:
      return "The node has no mesh geometry.";
    case NodeGeometryEditResult::NonMeshAttachments:
      return "The node also contains a camera, emitter, or medium.";
    case NodeGeometryEditResult::InvalidGeometry:
      return "The attached mesh data is invalid.";
    case NodeGeometryEditResult::SingularTransform:
      return "A singular transform cannot be baked.";
    case NodeGeometryEditResult::DegenerateGeometry:
      return "The mesh has no non-degenerate surface area.";
    case NodeGeometryEditResult::HierarchyUpdateFailed:
      return "The hierarchy could not be updated.";
  }
  return "The geometry edit failed.";
}

const char* scene_edit_status_message(SceneEditStatus status) {
  switch (status) {
    case SceneEditStatus::Success:
      return "";
    case SceneEditStatus::InvalidNode:
      return "The selected node is no longer available.";
    case SceneEditStatus::InvalidParent:
      return "The node cannot be parented there.";
    case SceneEditStatus::InvalidResource:
      return "The selected resource is no longer available.";
    case SceneEditStatus::DuplicateAttachment:
      return "This resource is already attached to the node.";
    case SceneEditStatus::ResourceAlreadyAttached:
      return "This resource is already owned by another node.";
    case SceneEditStatus::UnsupportedAttachments:
      return "Subtrees containing cameras, lights, or media cannot be duplicated.";
    case SceneEditStatus::ActiveCameraProtected:
      return "Keep the active camera enabled or activate another camera before this edit.";
    case SceneEditStatus::InvalidTransform:
      return "The edit would give an attached camera, light, or medium an invalid transform.";
    case SceneEditStatus::GeometryGenerationFailed:
      return "The object geometry could not be generated.";
    case SceneEditStatus::HierarchyUpdateFailed:
      return "The scene hierarchy could not be updated.";
  }
  return "The scene edit failed.";
}

const char* scene_resource_edit_status_message(SceneResourceEditStatus status) {
  switch (status) {
    case SceneResourceEditStatus::Success:
      return "";
    case SceneResourceEditStatus::InvalidResource:
      return "The selected resource is no longer available.";
    case SceneResourceEditStatus::ResourceInUse:
      return "The resource is still referenced by the scene.";
    case SceneResourceEditStatus::ActiveResource:
      return "Activate another camera before deleting this one.";
    case SceneResourceEditStatus::ManagedResource:
      return "This resource is managed by the scene and cannot be deleted.";
    case SceneResourceEditStatus::ResourceUpdateFailed:
      return "The scene resource could not be updated.";
  }
  return "The scene resource edit failed.";
}

const char* scene_primitive_name(ScenePrimitive primitive) {
  switch (primitive) {
    case ScenePrimitive::Sphere:
      return "Sphere";
    case ScenePrimitive::Box:
      return "Box";
    case ScenePrimitive::Plane:
      return "Plane";
    case ScenePrimitive::Disk:
      return "Disk";
    case ScenePrimitive::Cylinder:
      return "Cylinder";
    case ScenePrimitive::Cone:
      return "Cone";
    case ScenePrimitive::Capsule:
      return "Capsule";
    case ScenePrimitive::Torus:
      return "Torus";
    case ScenePrimitive::Ring:
      return "Ring";
    case ScenePrimitive::Tube:
      return "Tube";
    case ScenePrimitive::Tetrahedron:
      return "Tetrahedron";
    case ScenePrimitive::Cube:
      return "Cube";
    case ScenePrimitive::Octahedron:
      return "Octahedron";
    case ScenePrimitive::Dodecahedron:
      return "Dodecahedron";
    case ScenePrimitive::Icosahedron:
      return "Icosahedron";
  }
  return "Object";
}

void material_class_to_string(Material::Class cls, const char** str) {
  static const char* names[] = {
    "diffuse",
    "translucent",
    "plastic",
    "conductor",
    "dielectric",
    "thinfilm",
    "mirror",
    "boundary",
    "velvet",
    "openpbr",
    "void",
    "diffraction_grating",
    "undefined",
  };
  static_assert(sizeof(names) / sizeof(names[0]) == uint32_t(MaterialClass::Count) + 1);
  *str = cls < MaterialClass::Count ? names[uint32_t(cls)] : "undefined";
}

const char* material_class_to_string(Material::Class cls) {
  const char* result = nullptr;
  material_class_to_string(cls, &result);
  return result;
}

struct SceneRepresentationImpl {
  static constexpr float kDefaultDielectricEta = 1.5f;
  static constexpr float kDefaultConductorK = 1000000.0f;

  TaskScheduler& scheduler;
  SceneData data;
  Camera active_camera;
  bool scene_valid = false;
  std::mutex mt;
  RHIContext* rhi = nullptr;
  scattering::GpuContext scattering_gpu = {};
  bool scattering_gpu_ready = false;
  EnergyCompensationGenerationContext energy_compensation_generation = {};
  EnergyCompensationPreparationState energy_compensation_preparation_state = EnergyCompensationPreparationState::Ready;
  std::chrono::steady_clock::time_point energy_compensation_preparation_started_at = {};
  // Material-derived media overwrite the render-facing local bounds, so retain the authored volume bounds separately for later node attachment.
  std::vector<BoundingBox> medium_authored_bounds;
  std::vector<BoundingBox> medium_bounds_scratch;
  std::vector<uint8_t> medium_has_bounds_scratch;
  std::vector<uint32_t> medium_attachment_nodes_scratch;

  const IORDatabase& ior_database;
  SceneRepresentation::IntegratorData integrator_data = {};
  uint64_t integrator_data_revision = 1u;

  bool load_illuminant_from_identifier(const char* identifier, SpectralDistribution& spd) const {
    if ((identifier == nullptr) || (identifier[0] == 0))
      return false;

    if (const IORDefinition* def = ior_database.find_by_name(identifier, SpectralDistribution::Illuminant)) {
      spd = def->eta;
      return true;
    }

    std::filesystem::path candidate = locate_spectrum_file(identifier, {"emission"});
    if (candidate.empty())
      return false;

    std::string title;
    auto cls = SpectralDistribution::load_from_file(candidate.string().c_str(), spd, nullptr, false, title);
    return cls != SpectralDistribution::Invalid;
  }

  SceneRepresentationImpl(TaskScheduler& s, const IORDatabase& db)
    : scheduler(s)
    , data(s)
    , ior_database(db) {
    data.images.init(1024u);
    data.mediums.init(1024u);
    build_camera(active_camera, {5.0f, 5.0f, 5.0f}, normalize(float3{0.0f, 0.0f, 0.0f} - float3{5.0f, 5.0f, 5.0f}), kWorldUp, {1280u, 720u}, 26.99f);
  }

  ~SceneRepresentationImpl() {
    cleanup();
    if ((rhi != nullptr) && scattering_gpu.initialized) {
      scattering::gpu_cleanup(*rhi, scattering_gpu);
    }
    data.images.cleanup();
    data.mediums.cleanup();
  }

  void init_default_values() {
    data.defaults.black_spectrum = data.add_spectrum(SpectralDistribution::rgb_reflectance({0.0f, 0.0f, 0.0f}));
    data.defaults.white_spectrum = data.add_spectrum(SpectralDistribution::rgb_reflectance({1.0f, 1.0f, 1.0f}));
    data.defaults.rayleigh_spectrum = data.add_spectrum(scattering::rayleigh_spectrum());
    data.defaults.mie_spectrum = data.add_spectrum(scattering::mie_spectrum());
    data.defaults.ozone_spectrum = data.add_spectrum(scattering::ozone_spectrum());
    data.defaults.dielectric_eta = data.add_spectrum(SpectralDistribution::constant(kDefaultDielectricEta));
    data.defaults.conductor_eta = data.add_spectrum(SpectralDistribution::constant(0.0f));
    data.defaults.conductor_k = data.add_spectrum(SpectralDistribution::constant(kDefaultConductorK));

    data.options.properties[Scene::Properties::Spectral] = false;
    data.options.properties[Scene::Properties::MultipleImportanceSampling] = true;
    data.options.properties[Scene::Properties::BlueNoise] = true;

    data.defaults.subsurface_scatter_material = data.add_material("etx::subsurface-scatter");
    data.materials[data.defaults.subsurface_scatter_material].reflectance = {.spectrum_index = data.defaults.black_spectrum};
    data.materials[data.defaults.subsurface_scatter_material].scattering = {.spectrum_index = data.defaults.white_spectrum};
    data.materials[data.defaults.subsurface_scatter_material].cls = MaterialClass::Translucent;

    data.defaults.subsurface_exit_material = data.add_material("etx::subsurface-exit");
    data.materials[data.defaults.subsurface_exit_material].reflectance = {.spectrum_index = data.defaults.white_spectrum};
    data.materials[data.defaults.subsurface_exit_material].scattering = {.spectrum_index = data.defaults.white_spectrum};
    data.materials[data.defaults.subsurface_exit_material].cls = MaterialClass::Diffuse;

    data.defaults.missing_material = data.add_material("etx::missing");
    data.materials[data.defaults.missing_material].reflectance = {.spectrum_index = data.defaults.white_spectrum};
    data.materials[data.defaults.missing_material].scattering = {.spectrum_index = data.defaults.white_spectrum};
    data.materials[data.defaults.missing_material].cls = MaterialClass::Diffuse;
  }

  void cleanup() {
    if ((rhi != nullptr) && (energy_compensation_generation.pipeline.valid() || (energy_compensation_generation.pending_step != nullptr))) {
      cleanup_energy_compensation_generation(*rhi, energy_compensation_generation);
    }
    energy_compensation_generation = {};
    energy_compensation_preparation_state = EnergyCompensationPreparationState::Ready;
    scene_valid = false;
    data.clear(scheduler);
    medium_authored_bounds.clear();
    medium_bounds_scratch.clear();
    medium_has_bounds_scratch.clear();
    medium_attachment_nodes_scratch.clear();
    integrator_data = {};
    integrator_data_revision += 1u;

    active_camera = {};
    active_camera.lens_image = kInvalidIndex;
    active_camera.medium_index = kInvalidIndex;
    active_camera.up = kWorldUp;

    build_camera(active_camera, {5.0f, 5.0f, 5.0f}, normalize(float3{0.0f, 0.0f, 0.0f} - float3{5.0f, 5.0f, 5.0f}), kWorldUp, {1280u, 720u}, 26.99f);

    init_default_values();
  }

  float triangle_area(const Triangle& t) {
    return 0.5f * length(cross(data.vertices.pos[t.i[1]] - data.vertices.pos[t.i[0]], data.vertices.pos[t.i[2]] - data.vertices.pos[t.i[0]]));
  }

  void validate_materials() {
    std::mutex mt;
    scheduler.execute(data.materials.size(), [this, &mt](uint32_t begin, uint32_t end, uint32_t) {
      for (uint32_t i = begin; i < end; ++i) {
        auto& mtl = data.materials[i];
        mtl.thinfilm.weight = clamp(mtl.thinfilm.weight, 0.0f, 1.0f);
        mtl.thinfilm.min_thickness = max(0.0f, mtl.thinfilm.min_thickness);
        mtl.thinfilm.max_thickness = max(0.0f, mtl.thinfilm.max_thickness);
        if ((mtl.cls == MaterialClass::DiffractionGrating) &&
            ((std::isfinite(mtl.diffraction_grating.period_nm) == false) || (mtl.diffraction_grating.period_nm < kDiffractionGratingMinimumPeriodNm) ||
              (mtl.diffraction_grating.period_nm > kDiffractionGratingMaximumPeriodNm))) {
          log::warning("Material %u diffraction period %.9g nm is outside the supported [%g, %g] nm range; clamping it", i, mtl.diffraction_grating.period_nm,
            kDiffractionGratingMinimumPeriodNm, kDiffractionGratingMaximumPeriodNm);
          mtl.diffraction_grating.period_nm = clamp(mtl.diffraction_grating.period_nm, kDiffractionGratingMinimumPeriodNm, kDiffractionGratingMaximumPeriodNm);
          if (std::isfinite(mtl.diffraction_grating.period_nm) == false) {
            mtl.diffraction_grating.period_nm = 1600.0f;
          }
        }
        if ((mtl.cls == MaterialClass::DiffractionGrating) &&
            ((std::isfinite(mtl.diffraction_grating.duty_cycle) == false) || (mtl.diffraction_grating.duty_cycle < 0.0f) || (mtl.diffraction_grating.duty_cycle > 1.0f))) {
          log::warning("Material %u diffraction duty cycle %.9g is outside [0, 1]; clamping it", i, mtl.diffraction_grating.duty_cycle);
          mtl.diffraction_grating.duty_cycle = clamp(mtl.diffraction_grating.duty_cycle, 0.0f, 1.0f);
          if (std::isfinite(mtl.diffraction_grating.duty_cycle) == false) {
            mtl.diffraction_grating.duty_cycle = 0.5f;
          }
        }
        if ((mtl.cls == MaterialClass::DiffractionGrating) && ((std::isfinite(mtl.diffraction_grating.optical_path_difference_nm) == false) ||
                                                                (mtl.diffraction_grating.optical_path_difference_nm < kDiffractionGratingMinimumOpticalPathDifferenceNm) ||
                                                                (mtl.diffraction_grating.optical_path_difference_nm > kDiffractionGratingMaximumOpticalPathDifferenceNm))) {
          log::warning("Material %u diffraction optical path difference %.9g nm is outside the supported [%g, %g] nm range; clamping it", i,
            mtl.diffraction_grating.optical_path_difference_nm, kDiffractionGratingMinimumOpticalPathDifferenceNm, kDiffractionGratingMaximumOpticalPathDifferenceNm);
          mtl.diffraction_grating.optical_path_difference_nm =
            clamp(mtl.diffraction_grating.optical_path_difference_nm, kDiffractionGratingMinimumOpticalPathDifferenceNm, kDiffractionGratingMaximumOpticalPathDifferenceNm);
          if (std::isfinite(mtl.diffraction_grating.optical_path_difference_nm) == false) {
            mtl.diffraction_grating.optical_path_difference_nm = 280.0f;
          }
        }
        if ((mtl.cls == MaterialClass::DiffractionGrating) && (std::isfinite(mtl.diffraction_grating.rotation) == false)) {
          log::warning("Material %u diffraction rotation is not finite; resetting it to zero", i);
          mtl.diffraction_grating.rotation = 0.0f;
        }
        if (mtl.reflectance.spectrum_index == kInvalidIndex) {
          std::unique_lock lock(mt);
          mtl.reflectance.spectrum_index = data.add_spectrum(SpectralDistribution::rgb_reflectance({1.0f, 1.0f, 1.0f}));
        }
        if (mtl.scattering.spectrum_index == kInvalidIndex) {
          std::unique_lock lock(mt);
          mtl.scattering.spectrum_index = data.add_spectrum(SpectralDistribution::rgb_reflectance({1.0f, 1.0f, 1.0f}));
        }
        if (mtl.subsurface.spectrum_index == kInvalidIndex) {
          std::unique_lock lock(mt);
          mtl.subsurface.spectrum_index = data.add_spectrum(SpectralDistribution::rgb_reflectance({1.0f, 0.2f, 0.04f}));
        }
        if ((mtl.subsurface_cls != SubsurfaceMaterial::Disabled) && (mtl.subsurface_cls != SubsurfaceMaterial::RandomWalk)) {
          mtl.subsurface_cls = SubsurfaceMaterial::RandomWalk;
        }
        if (mtl.emission.spectrum_index == kInvalidIndex) {
          std::unique_lock lock(mt);
          mtl.emission.spectrum_index = data.add_spectrum(SpectralDistribution::constant(0.0f));
        }
        if ((mtl.roughness.value.x > 0.0f) || (mtl.roughness.value.y > 0.0f)) {
          mtl.roughness.value.x = max(kEpsilon, mtl.roughness.value.x);
          mtl.roughness.value.y = max(kEpsilon, mtl.roughness.value.y);
        }
        if (mtl.int_ior.eta_index == kInvalidIndex) {
          std::unique_lock lock(mt);
          if (mtl.cls == MaterialClass::Conductor) {
            mtl.int_ior.cls = SpectralDistribution::Conductor;
            mtl.int_ior.eta_index = data.add_spectrum(SpectralDistribution::constant(0.0f));
          } else {
            mtl.int_ior.cls = SpectralDistribution::Dielectric;
            mtl.int_ior.eta_index = data.add_spectrum(SpectralDistribution::constant(kDefaultDielectricEta));
          }
        }
        if (mtl.int_ior.k_index == kInvalidIndex) {
          std::unique_lock lock(mt);
          if (mtl.cls == MaterialClass::Conductor) {
            mtl.int_ior.k_index = data.add_spectrum(SpectralDistribution::constant(kDefaultConductorK));
          } else {
            mtl.int_ior.k_index = data.add_spectrum(SpectralDistribution::constant(0.0f));
          }
        }
        if (mtl.ext_ior.eta_index == kInvalidIndex) {
          std::unique_lock lock(mt);
          mtl.ext_ior.cls = SpectralDistribution::Dielectric;
          mtl.ext_ior.eta_index = data.add_spectrum(SpectralDistribution::constant(1.0f));
        }
        if (mtl.ext_ior.k_index == kInvalidIndex) {
          std::unique_lock lock(mt);
          mtl.ext_ior.k_index = data.add_spectrum(SpectralDistribution::constant(0.0f));
        }
        const bool thinfilm_requested = (mtl.thinfilm.weight > 0.0f) && (max(mtl.thinfilm.min_thickness, mtl.thinfilm.max_thickness) > 0.0f);
        if (thinfilm_requested && (mtl.thinfilm.ior.cls != SpectralDistribution::Dielectric)) {
          log::warning("Material %u uses a non-dielectric thin-film IOR; disabling its unsupported thin film", i);
          mtl.thinfilm.weight = 0.0f;
        }
        {
          std::unique_lock lock(mt);
          if (mtl.thinfilm.ior.k_index >= data.spectrum_values.size()) {
            mtl.thinfilm.ior.k_index = data.add_spectrum(SpectralDistribution::constant(0.0f));
          }
          if (mtl.thinfilm.ior.eta_index >= data.spectrum_values.size()) {
            mtl.thinfilm.ior.eta_index = data.add_spectrum(SpectralDistribution::constant(1.0f));
          }
          if (thinfilm_requested && (mtl.thinfilm.ior.cls == SpectralDistribution::Dielectric) && (data.spectrum_values[mtl.thinfilm.ior.k_index].is_zero() == false)) {
            log::warning("Material %u uses absorption in its thin film; forcing extinction to zero for the supported lossless-film model", i);
            mtl.thinfilm.ior.k_index = data.add_spectrum(SpectralDistribution::constant(0.0f));
          }
        }
      }
    });
  }

  void validate_mediums() {
    // Clamp medium densities to prevent extremely small mean free paths
    for (uint32_t i = 0; i < data.mediums.array_size(); ++i) {
      const Medium& medium = data.mediums.get(i);

      if (medium.absorption_index == kInvalidIndex || medium.absorption_index >= data.spectrum_values.size()) {
        continue;
      }

      if (medium.scattering_index == kInvalidIndex || medium.scattering_index >= data.spectrum_values.size()) {
        continue;
      }

      SpectralDistribution& absorption = data.spectrum_values[medium.absorption_index];
      SpectralDistribution& scattering = data.spectrum_values[medium.scattering_index];

      float max_absorption = absorption.maximum_spectral_power();
      float max_scattering = scattering.maximum_spectral_power();

      float max_extinction = max_absorption + max_scattering;
      if (max_extinction <= 0.0f) {
        continue;
      }

      constexpr float kMinMeanFreePathAbsolute = 0.01f;  // Minimum mean free path in absolute units
      float max_allowed_extinction = 1.0f / kMinMeanFreePathAbsolute;

      if (max_extinction <= max_allowed_extinction) {
        continue;
      }

      float scale_factor = max_allowed_extinction / max_extinction;

      // Scale both spectra by the same factor to preserve ratios
      absorption.scale(scale_factor);
      scattering.scale(scale_factor);
    }
  }

  void validate_normals(std::vector<bool>& referenced_vertices, bool& has_invalid_tangents) {
    std::vector<bool> init_normals(data.vertices.nrm.size(), false);
    std::vector<bool> reconstruct_normals(data.vertices.nrm.size(), false);
    referenced_vertices.resize(data.vertices.nrm.size());

    for (uint64_t i = 0, e = data.vertices.nrm.size(); i < e; ++i) {
      reconstruct_normals[i] = is_valid_vector(data.vertices.nrm[i]) == false;
    }

    bool has_tangents = data.vertices.tan.size() == data.vertices.nrm.size();
    if (has_tangents == false)
      has_invalid_tangents = true;

    for (const auto& tri : data.triangles) {
      const float tri_area = triangle_area(tri);
      for (uint32_t i = 0; i < 3; ++i) {
        uint32_t index = tri.i[i];
        ETX_CRITICAL(is_valid_vector(tri.geo_n));
        referenced_vertices[index] = true;

        if (has_tangents && (is_valid_vector(data.vertices.tan[index]) == false)) {
          has_invalid_tangents = true;
        }

        if (reconstruct_normals[index] == false) {
          continue;
        }

        if (init_normals[index]) {
          data.vertices.nrm[index] += tri.geo_n * tri_area;
        } else {
          init_normals[index] = true;
          data.vertices.nrm[index] = tri.geo_n * tri_area;
        }
      }
    }

    scheduler.execute(data.vertices.nrm.size(), [this, &referenced_vertices](uint32_t begin, uint32_t end, uint32_t) {
      for (uint32_t i = begin; i < end; ++i) {
        if (referenced_vertices[i]) {
          data.vertices.nrm[i] = normalize(data.vertices.nrm[i]);
        }
      }
    });
  }

  void build_tangents() {
    const uint64_t normal_count = data.vertices.nrm.size();
    if (data.vertices.tan.size() != normal_count)
      data.vertices.tan.resize(normal_count);
    if (data.vertices.btn.size() != normal_count)
      data.vertices.btn.resize(normal_count);

    TimeMeasure uv_timer = {};
    float2 min_uv = {kMaxFloat, kMaxFloat};
    float2 max_uv = {-kMaxFloat, -kMaxFloat};
    for (const auto& v : data.vertices.tex) {
      min_uv = min(min_uv, v);
      max_uv = max(max_uv, v);
    }
    auto uv_span = max_uv - min_uv;
    if (dot(uv_span, uv_span) <= kEpsilon) {
      log::warning("No texture coordinates: tangents will be computed automatically");
      return;
    }
    log::info("UV validation: %.4f sec", uv_timer.lap());

    TimeMeasure total_timer = {};

    // Pre-resolve vertex data to eliminate index lookups during computation
    // Use SoA (Structure of Arrays) for better cache performance
    TimeMeasure resolve_timer = {};
    const size_t total_vertices = data.triangles.size() * 3;
    std::vector<float3> resolved_positions(total_vertices);
    std::vector<float3> resolved_normals(total_vertices);
    std::vector<float2> resolved_texcoords(total_vertices);

    for (size_t tri_idx = 0; tri_idx < data.triangles.size(); ++tri_idx) {
      const auto& tri = data.triangles[tri_idx];
      const size_t base_idx = tri_idx * 3;
      for (uint32_t i = 0; i < 3; ++i) {
        uint32_t vertex_index = tri.i[i];
        resolved_positions[base_idx + i] = data.vertices.pos[vertex_index];
        resolved_normals[base_idx + i] = data.vertices.nrm[vertex_index];
        resolved_texcoords[base_idx + i] = data.vertices.tex[vertex_index];
      }
    }
    log::info("Vertex data resolution: %.4f sec", resolve_timer.lap());

    struct MikkTSpaceUserData {
      const std::vector<float3>& positions;
      const std::vector<float3>& normals;
      const std::vector<float2>& texcoords;
      SceneData& data;
      std::vector<bool> computed_flags;
    };
    MikkTSpaceUserData user_data = {resolved_positions, resolved_normals, resolved_texcoords, data, std::vector<bool>(normal_count, false)};

    TimeMeasure interface_timer = {};
    SMikkTSpaceInterface contextInterface = {};
    contextInterface.m_getNumFaces = [](const SMikkTSpaceContext* pContext) -> int {
      const auto& user_data = *reinterpret_cast<MikkTSpaceUserData*>(pContext->m_pUserData);
      return static_cast<int>(user_data.data.triangles.size());
    };
    contextInterface.m_getNumVerticesOfFace = [](const SMikkTSpaceContext* pContext, const int iFace) -> int {
      return 3;
    };
    contextInterface.m_getPosition = [](const SMikkTSpaceContext* pContext, float fvPosOut[], const int iFace, const int iVert) {
      const auto& user_data = *reinterpret_cast<MikkTSpaceUserData*>(pContext->m_pUserData);
      const auto& pos = user_data.positions[iFace * 3 + iVert];
      fvPosOut[0] = pos.x;
      fvPosOut[1] = pos.y;
      fvPosOut[2] = pos.z;
    };
    contextInterface.m_getNormal = [](const SMikkTSpaceContext* pContext, float fvNormOut[], const int iFace, const int iVert) {
      const auto& user_data = *reinterpret_cast<MikkTSpaceUserData*>(pContext->m_pUserData);
      const auto& nrm = user_data.normals[iFace * 3 + iVert];
      fvNormOut[0] = nrm.x;
      fvNormOut[1] = nrm.y;
      fvNormOut[2] = nrm.z;
    };
    contextInterface.m_getTexCoord = [](const SMikkTSpaceContext* pContext, float fvTexcOut[], const int iFace, const int iVert) {
      const auto& user_data = *reinterpret_cast<MikkTSpaceUserData*>(pContext->m_pUserData);
      const auto& tex = user_data.texcoords[iFace * 3 + iVert];
      fvTexcOut[0] = tex.x;
      fvTexcOut[1] = tex.y;
    };
    contextInterface.m_setTSpaceBasic = [](const SMikkTSpaceContext* pContext, const float fvTangent[], const float fSign, const int iFace, const int iVert) {
      auto& user_data = *reinterpret_cast<MikkTSpaceUserData*>(pContext->m_pUserData);
      const auto& tri = user_data.data.triangles[iFace];
      uint32_t vertex_index = tri.i[iVert];
      auto& nrm = user_data.data.vertices.nrm[vertex_index];
      auto& tan = user_data.data.vertices.tan[vertex_index];
      auto& btn = user_data.data.vertices.btn[vertex_index];

      // Let MikkTSpace set tangents for vertices it hasn't touched yet
      if (user_data.computed_flags[vertex_index] == false) {
        tan = normalize(float3{fvTangent[0], fvTangent[1], fvTangent[2]});
        btn = normalize(cross(tan, nrm) * fSign);
        user_data.computed_flags[vertex_index] = true;
      }
    };

    SMikkTSpaceContext context = {};
    context.m_pUserData = &user_data;
    context.m_pInterface = &contextInterface;

    log::info("MikkTSpace interface setup: %.4f sec", interface_timer.lap());

    TimeMeasure compute_timer = {};
    genTangSpaceDefault(&context);
    log::info("MikkTSpace computation: %.4f sec", compute_timer.lap());

    log::info("Total tangent building: %.4f sec", total_timer.lap());
  }

  void validate_tangents(std::vector<bool>& referenced_vertices, bool force) {
    for (uint64_t vertex_index = 0, e = data.vertices.tan.size(); vertex_index < e; ++vertex_index) {
      auto& v_nrm = data.vertices.nrm[vertex_index];
      auto& v_tan = data.vertices.tan[vertex_index];
      auto& v_btn = data.vertices.btn[vertex_index];

      bool tan_valid = is_valid_vector(v_tan);
      bool btn_valid = is_valid_vector(v_btn);

      if (tan_valid && btn_valid) {
        continue;
      }

      if (force || referenced_vertices[vertex_index]) {
        ETX_ASSERT(is_valid_vector(v_nrm));
        auto [t, b] = orthonormal_basis(v_nrm);
        v_tan = t;
        v_btn = b;
      }
    }
  }

  bool update_medium_bounds();
  bool update_active_camera();
  SceneEditStatus finalize_hierarchy_edit(SceneHierarchy& original_hierarchy);
  void set_mesh_material(uint32_t mesh_index, uint32_t material_index);

  void set_mesh_material_impl(uint32_t mesh_index, uint32_t material_index);
  void add_atmosphere_emitter(const AtmosphereEmitterParameters& params);
  void rebuild_atmosphere_emitter(uint32_t emitter_index);
  void set_scattering_rhi(RHIContext& rhi_context);
  bool ensure_energy_compensation_interfaces();
  bool begin_energy_compensation_interface_preparation();
  void cancel_energy_compensation_interface_preparation();
  EnergyCompensationPreparationState poll_energy_compensation_interface_preparation();
  EnergyCompensationPreparationStatus energy_compensation_interface_preparation_status() const;
  bool ensure_scattering_gpu_context();
  void generate_pixel_sampler_image(float radius);

  void create_area_emitters_from_materials();
  void setup_atmosphere_references();

  bool finalize_scene_loading(uint32_t options, const char* base_folder, uint32_t load_result, float camera_fov, bool use_focal_len, float camera_focal_len, bool force_tangents,
    bool spectral_scene, float pixel_filter_radius, bool preserve_unattached_cameras);
};

void build_camera(Camera& camera, const float3& position, const float3& direction, const float3& up, const uint2& viewport, const float fov) {
  sanitize_camera_clip_planes(camera);

  float3 target = position + direction;

  float4x4 view = look_at(position, target, up);
  float4x4 proj = perspective(fov * kPi / 180.0f, viewport.x, viewport.y, camera.clip_near, camera.clip_far);

  camera.position = position;
  camera.direction = normalize(direction);
  camera.side = {view.col[0].x, view.col[1].x, view.col[2].x};
  camera.up = {view.col[0].y, view.col[1].y, view.col[2].y};
  camera.tan_half_fov = 1.0f / std::abs(proj.col[0].x);
  camera.aspect = proj.col[1].y / proj.col[0].x;
  camera.view_proj = proj * view;

  float plane_w = 2.0f * camera.tan_half_fov;
  float plane_h = 2.0f * camera.tan_half_fov / camera.aspect;
  camera.area = plane_w * plane_h;
  camera.film_size = viewport;
  camera.image_plane = float(camera.film_size.x) / (2.0f * camera.tan_half_fov);
}

float get_camera_fov(const Camera& camera) {
  return 2.0f * atanf(camera.tan_half_fov) * 180.0f / kPi;
}

float get_camera_focal_length(const Camera& camera) {
  return 0.5f * Film::kFilmHorizontalSize / camera.tan_half_fov;
}

float fov_to_focal_length(float fov) {
  return 0.5f * Film::kFilmHorizontalSize / tanf(0.5f * fov);
}

float focal_length_to_fov(float focal_len) {
  return 2.0f * atanf(Film::kFilmHorizontalSize / (2.0f * focal_len));
}

float horizontal_fov_to_vertical_fov(float horizontal_fov) {
  float aspect_ratio = Film::kFilmHorizontalSize / Film::kFilmVerticalSize;
  return 2.0f * atanf(tanf(0.5f * horizontal_fov) / aspect_ratio);
}

float vertical_fov_to_horizontal_fov(float vertical_fov) {
  float aspect_ratio = Film::kFilmHorizontalSize / Film::kFilmVerticalSize;
  return 2.0f * atanf(tanf(0.5f * vertical_fov) * aspect_ratio);
}

void compute_camera_position_to_fit_scene(const Scene& scene_data, const Camera& camera, const float3& view_direction, float3& out_position, float3& out_target) {
  const float3 bbox_min = scene_data.bounding_box_min;
  const float3 bbox_max = scene_data.bounding_box_max;
  const float3 center = 0.5f * (bbox_min + bbox_max);
  constexpr float kMinCosineThreshold = 0.99f;
  const float3 view_dir = clamp_view_direction_away_from_up(view_direction, kWorldUp, kMinCosineThreshold);

  float distance = 3.0f * scene_data.bounding_sphere_radius;

  if ((camera.cls == Camera::Class::Perspective) && (camera.tan_half_fov > kEpsilon)) {
    const float3 bbox_size = bbox_max - bbox_min;
    float3 right = cross(view_dir, kWorldUp);
    if (length(right) < kEpsilon) {
      right = cross(view_dir, kWorldRight);
    }
    right = normalize(right);
    const float3 up = normalize(cross(right, view_dir));

    const float3 bbox_half_size = 0.5f * bbox_size;
    const float3 bbox_corners[8] = {
      center + float3{-bbox_half_size.x, -bbox_half_size.y, -bbox_half_size.z},
      center + float3{+bbox_half_size.x, -bbox_half_size.y, -bbox_half_size.z},
      center + float3{-bbox_half_size.x, +bbox_half_size.y, -bbox_half_size.z},
      center + float3{+bbox_half_size.x, +bbox_half_size.y, -bbox_half_size.z},
      center + float3{-bbox_half_size.x, -bbox_half_size.y, +bbox_half_size.z},
      center + float3{+bbox_half_size.x, -bbox_half_size.y, +bbox_half_size.z},
      center + float3{-bbox_half_size.x, +bbox_half_size.y, +bbox_half_size.z},
      center + float3{+bbox_half_size.x, +bbox_half_size.y, +bbox_half_size.z},
    };

    const float tan_half_fov = camera.tan_half_fov;
    const float aspect = camera.aspect;
    const float margin = 1.1f;

    float min_distance = 0.0f;

    for (uint32_t i = 0; i < 8; ++i) {
      const float3 corner_rel = bbox_corners[i] - center;
      const float right_proj = dot(corner_rel, right);
      const float up_proj = dot(corner_rel, up);
      const float forward_proj = dot(corner_rel, view_dir);

      const float corner_dist_for_width = margin * fabsf(right_proj) / tan_half_fov;
      const float corner_dist_for_height = margin * fabsf(up_proj) * aspect / tan_half_fov;
      const float corner_required_dist = max(corner_dist_for_width, corner_dist_for_height);

      const float corner_distance = forward_proj + corner_required_dist;
      min_distance = max(min_distance, corner_distance);
    }

    distance = min_distance;
  }

  out_position = center + distance * view_dir;
  out_target = center;
}

void compute_camera_position_to_fit_scene(const SceneData& scene_data, const Camera& camera, const float3& view_direction, float3& out_position, float3& out_target) {
  auto bbox = scene_data.compute_bounding_volumes();
  const float3 bbox_size = bbox.p_max - bbox.p_min;
  const float3 center = 0.5f * (bbox.p_min + bbox.p_max);

  constexpr float kMinCosineThreshold = 0.99f;
  const float3 view_dir = clamp_view_direction_away_from_up(view_direction, kWorldUp, kMinCosineThreshold);

  float distance = 3.0f * length(bbox_size);

  if ((camera.cls == Camera::Class::Perspective) && (camera.tan_half_fov > kEpsilon)) {
    float3 right = cross(view_dir, kWorldUp);
    if (length(right) < kEpsilon) {
      right = cross(view_dir, kWorldRight);
    }
    right = normalize(right);
    const float3 up = normalize(cross(right, view_dir));

    const float3 bbox_half_size = 0.5f * bbox_size;
    const float3 bbox_corners[8] = {
      center + float3{-bbox_half_size.x, -bbox_half_size.y, -bbox_half_size.z},
      center + float3{+bbox_half_size.x, -bbox_half_size.y, -bbox_half_size.z},
      center + float3{-bbox_half_size.x, +bbox_half_size.y, -bbox_half_size.z},
      center + float3{+bbox_half_size.x, +bbox_half_size.y, -bbox_half_size.z},
      center + float3{-bbox_half_size.x, -bbox_half_size.y, +bbox_half_size.z},
      center + float3{+bbox_half_size.x, -bbox_half_size.y, +bbox_half_size.z},
      center + float3{-bbox_half_size.x, +bbox_half_size.y, +bbox_half_size.z},
      center + float3{+bbox_half_size.x, +bbox_half_size.y, +bbox_half_size.z},
    };

    const float tan_half_fov = camera.tan_half_fov;
    const float aspect = camera.aspect;
    const float margin = 1.1f;

    float min_distance = 0.0f;

    for (uint32_t i = 0; i < 8; ++i) {
      const float3 corner_rel = bbox_corners[i] - center;
      const float right_proj = dot(corner_rel, right);
      const float up_proj = dot(corner_rel, up);
      const float forward_proj = dot(corner_rel, view_dir);

      const float corner_dist_for_width = margin * fabsf(right_proj) / tan_half_fov;
      const float corner_dist_for_height = margin * fabsf(up_proj) * aspect / tan_half_fov;
      const float corner_required_dist = max(corner_dist_for_width, corner_dist_for_height);

      const float corner_distance = forward_proj + corner_required_dist;
      min_distance = max(min_distance, corner_distance);
    }

    distance = max(distance, min_distance);
  }

  out_position = center + distance * view_dir;
  out_target = center;
}

struct AttachmentTransform {
  AffineTransform object_to_world = {};
  AffineTransform world_to_object = {};
  AffineTransform orientation_to_world = {};
  AffineTransform orientation_to_object = {};
  uint32_t node_index = kInvalidIndex;
};

bool find_attachment_transform(SceneData& data, SceneAttachment::Type type, uint32_t resource_index, AttachmentTransform& result) {
  if ((data.hierarchy.resolved_state_current() == false) && (data.resolve_hierarchy() == false)) {
    return false;
  }

  for (uint32_t node_index : data.hierarchy.evaluation_order) {
    const SceneNode& node = data.hierarchy.nodes[node_index];
    if ((node_index >= data.hierarchy.effective_enabled.size()) || (data.hierarchy.effective_enabled[node_index] == 0u)) {
      continue;
    }
    const uint32_t attachment_end = node.attachment_offset + node.attachment_count;
    if (attachment_end > data.hierarchy.attachments.size()) {
      return false;
    }
    for (uint32_t attachment_index = node.attachment_offset; attachment_index < attachment_end; ++attachment_index) {
      const SceneAttachment& attachment = data.hierarchy.attachments[attachment_index];
      if ((attachment.type != type) || (attachment.resource_index != resource_index)) {
        continue;
      }
      if ((node_index >= data.hierarchy.orientation_valid.size()) || (data.hierarchy.orientation_valid[node_index] == 0u) ||
          (node_index >= data.hierarchy.world_orientations.size())) {
        log::warning("Attachment %u of type %u is below a sheared or singular node and has no valid orientation", resource_index, static_cast<uint32_t>(type));
        return false;
      }
      double determinant = 0.0;
      result.object_to_world = data.hierarchy.world_transforms[node_index];
      if (invert_affine(result.object_to_world, result.world_to_object, determinant) == false) {
        return false;
      }
      result.orientation_to_world = data.hierarchy.world_orientations[node_index];
      result.node_index = node_index;
      return invert_affine(result.orientation_to_world, result.orientation_to_object, determinant);
    }
  }
  return false;
}

Camera transform_camera(const Camera& source, const AffineTransform& position_transform, const AffineTransform& orientation_transform) {
  Camera result = source;
  const float3 position = transform_point(position_transform, source.position);
  const float3 direction = normalize(transform_vector(orientation_transform, source.direction));
  const float3 up = normalize(transform_vector(orientation_transform, source.up));
  build_camera(result, position, direction, up, source.film_size, get_camera_fov(source));
  return result;
}

bool hierarchy_attachment_transforms_valid(const SceneData& data) {
  const SceneHierarchy& hierarchy = data.hierarchy;
  for (uint32_t node_index : hierarchy.evaluation_order) {
    if ((node_index >= hierarchy.nodes.size()) || (node_index >= hierarchy.effective_enabled.size())) {
      return false;
    }
    if (hierarchy.effective_enabled[node_index] == 0u) {
      continue;
    }

    const SceneNode& node = hierarchy.nodes[node_index];
    const uint32_t attachment_end = node.attachment_offset + node.attachment_count;
    if ((attachment_end < node.attachment_offset) || (attachment_end > hierarchy.attachments.size())) {
      return false;
    }
    for (uint32_t attachment_index = node.attachment_offset; attachment_index < attachment_end; ++attachment_index) {
      const SceneAttachment& attachment = hierarchy.attachments[attachment_index];
      const bool camera_attachment = attachment.type == SceneAttachment::Type::Camera;
      const bool medium_attachment = attachment.type == SceneAttachment::Type::Medium;
      const bool distant_emitter_attachment = (attachment.type == SceneAttachment::Type::Emitter) && (attachment.resource_index < data.emitter_profiles.size()) &&
                                              (data.emitter_profiles[attachment.resource_index].cls != EmitterProfile::Class::Area);
      if ((camera_attachment == false) && (medium_attachment == false) && (distant_emitter_attachment == false)) {
        continue;
      }
      if (node_index >= hierarchy.world_transforms.size()) {
        return false;
      }

      if (camera_attachment || distant_emitter_attachment) {
        if ((node_index >= hierarchy.orientation_valid.size()) || (hierarchy.orientation_valid[node_index] == 0u) || (node_index >= hierarchy.world_orientations.size())) {
          return false;
        }
      }

      double determinant = 0.0;
      AffineTransform inverse = {};
      if ((camera_attachment || medium_attachment) && (invert_affine(hierarchy.world_transforms[node_index], inverse, determinant) == false)) {
        return false;
      }
      if (camera_attachment && (invert_affine(hierarchy.world_orientations[node_index], inverse, determinant) == false)) {
        return false;
      }
    }
  }
  return true;
}

bool active_camera_attachment_enabled(const SceneData& data) {
  const auto camera_it = std::find_if(data.cameras.begin(), data.cameras.end(), [](const auto& entry) {
    return entry.active;
  });
  if (camera_it == data.cameras.end()) {
    return true;
  }

  const uint32_t active_camera_index = static_cast<uint32_t>(std::distance(data.cameras.begin(), camera_it));
  bool attachment_found = false;
  for (uint32_t node_index = 0u; node_index < data.hierarchy.nodes.size(); ++node_index) {
    const SceneNode& node = data.hierarchy.nodes[node_index];
    const uint32_t attachment_end = node.attachment_offset + node.attachment_count;
    if ((attachment_end < node.attachment_offset) || (attachment_end > data.hierarchy.attachments.size())) {
      return false;
    }
    for (uint32_t attachment_index = node.attachment_offset; attachment_index < attachment_end; ++attachment_index) {
      const SceneAttachment& attachment = data.hierarchy.attachments[attachment_index];
      if ((attachment.type != SceneAttachment::Type::Camera) || (attachment.resource_index != active_camera_index)) {
        continue;
      }
      attachment_found = true;
      if ((node_index < data.hierarchy.effective_enabled.size()) && (data.hierarchy.effective_enabled[node_index] != 0u)) {
        return true;
      }
    }
  }
  return attachment_found == false;
}

bool SceneRepresentationImpl::update_active_camera() {
  if ((data.hierarchy.resolved_state_current() == false) && (data.resolve_hierarchy() == false)) {
    return false;
  }

  const auto camera_it = std::find_if(data.cameras.begin(), data.cameras.end(), [](const auto& entry) {
    return entry.active;
  });
  if (camera_it == data.cameras.end()) {
    return true;
  }

  Camera updated_camera = camera_it->cam;
  const uint32_t camera_index = static_cast<uint32_t>(std::distance(data.cameras.begin(), camera_it));
  bool enabled_attachment_found = false;
  for (uint32_t node_index : data.hierarchy.evaluation_order) {
    if ((node_index >= data.hierarchy.nodes.size()) || (node_index >= data.hierarchy.effective_enabled.size()) || (data.hierarchy.effective_enabled[node_index] == 0u)) {
      continue;
    }
    const SceneNode& node = data.hierarchy.nodes[node_index];
    const uint32_t attachment_end = node.attachment_offset + node.attachment_count;
    if ((attachment_end < node.attachment_offset) || (attachment_end > data.hierarchy.attachments.size())) {
      return false;
    }
    for (uint32_t attachment_index = node.attachment_offset; attachment_index < attachment_end; ++attachment_index) {
      const SceneAttachment& attachment = data.hierarchy.attachments[attachment_index];
      if ((attachment.type == SceneAttachment::Type::Camera) && (attachment.resource_index == camera_index)) {
        enabled_attachment_found = true;
        break;
      }
    }
    if (enabled_attachment_found) {
      break;
    }
  }

  if (enabled_attachment_found) {
    AttachmentTransform transform = {};
    if (find_attachment_transform(data, SceneAttachment::Type::Camera, camera_index, transform) == false) {
      return false;
    }
    updated_camera = transform_camera(camera_it->cam, transform.object_to_world, transform.orientation_to_world);
  }
  active_camera = updated_camera;
  return true;
}

SceneEditStatus SceneRepresentationImpl::finalize_hierarchy_edit(SceneHierarchy& original_hierarchy) {
  const auto restore_original = [&]() {
    data.hierarchy = std::move(original_hierarchy);
    const bool hierarchy_restored = data.resolve_hierarchy();
    const bool medium_state_restored = hierarchy_restored && update_medium_bounds();
    const bool camera_state_restored = medium_state_restored && update_active_camera();
    if (camera_state_restored == false) {
      log::error("Failed to restore scene state after rejecting a hierarchy edit");
    }
  };

  if (data.resolve_hierarchy() == false) {
    restore_original();
    return SceneEditStatus::HierarchyUpdateFailed;
  }
  if (active_camera_attachment_enabled(data) == false) {
    restore_original();
    return SceneEditStatus::ActiveCameraProtected;
  }
  if ((hierarchy_attachment_transforms_valid(data) == false) || (update_medium_bounds() == false) || (update_active_camera() == false)) {
    restore_original();
    return SceneEditStatus::InvalidTransform;
  }
  return SceneEditStatus::Success;
}

bool camera_pose_is_canonical(const Camera& camera) {
  return (camera.position.x == 0.0f) && (camera.position.y == 0.0f) && (camera.position.z == 0.0f) && (camera.direction.x == kWorldForward.x) &&
         (camera.direction.y == kWorldForward.y) && (camera.direction.z == kWorldForward.z) && (camera.up.x == kWorldUp.x) && (camera.up.y == kWorldUp.y) &&
         (camera.up.z == kWorldUp.z);
}

bool camera_pose_transform(const Camera& camera, AffineTransform& result) {
  const float3 direction = camera.direction;
  const float3 up_hint = camera.up;
  const float3 side = cross(direction, up_hint);
  const float fov = get_camera_fov(camera);
  if ((value_is_correct(camera.position) == false) || (is_valid_vector(direction) == false) || (is_valid_vector(up_hint) == false) || (is_valid_vector(side) == false) ||
      (camera.film_size.x == 0u) || (camera.film_size.y == 0u) || (std::isfinite(fov) == false) || (fov <= 0.0f)) {
    return false;
  }

  const float3 forward = normalize(direction);
  const float3 right = normalize(side);
  const float3 up = normalize(cross(right, forward));
  result.rows[0] = {right.x, up.x, -forward.x, camera.position.x};
  result.rows[1] = {right.y, up.y, -forward.y, camera.position.y};
  result.rows[2] = {right.z, up.z, -forward.z, camera.position.z};
  return true;
}

bool ensure_camera_nodes(SceneData& data, bool preserve_unattached_cameras) {
  for (uint32_t camera_index = 0u; camera_index < data.cameras.size(); ++camera_index) {
    uint32_t attachment_node_index = kInvalidIndex;
    uint32_t local_attachment_index = kInvalidIndex;
    SceneAttachment source_attachment = {};
    uint32_t attachment_count = 0u;
    for (uint32_t node_index = 0u; node_index < data.hierarchy.nodes.size(); ++node_index) {
      const SceneNode& node = data.hierarchy.nodes[node_index];
      const uint32_t attachment_end = node.attachment_offset + node.attachment_count;
      for (uint32_t attachment_index = node.attachment_offset; attachment_index < attachment_end; ++attachment_index) {
        const SceneAttachment& attachment = data.hierarchy.attachments[attachment_index];
        if ((attachment.type != SceneAttachment::Type::Camera) || (attachment.resource_index != camera_index)) {
          continue;
        }
        ++attachment_count;
        attachment_node_index = node_index;
        local_attachment_index = attachment_index - node.attachment_offset;
        source_attachment = attachment;
      }
    }
    if (attachment_count > 1u) {
      log::error("Camera %u is attached to multiple scene nodes; camera resources require one owning node", camera_index);
      return false;
    }
    if ((attachment_count == 0u) && preserve_unattached_cameras) {
      continue;
    }
    if ((attachment_count == 1u) && camera_pose_is_canonical(data.cameras[camera_index].cam)) {
      continue;
    }

    Camera& camera = data.cameras[camera_index].cam;
    AffineTransform camera_transform = {};
    if (camera_pose_transform(camera, camera_transform) == false) {
      log::error("Camera %u has an invalid pose or projection and cannot be attached to a scene node", camera_index);
      return false;
    }

    const std::string node_name = data.cameras[camera_index].id.empty() ? ("camera-" + std::to_string(camera_index)) : data.cameras[camera_index].id;
    const uint32_t node_index = data.hierarchy.add_node(node_name.c_str(), attachment_node_index, camera_transform);
    if (node_index == kInvalidIndex) {
      log::error("Failed to create a scene node for camera %u", camera_index);
      return false;
    }
    if (attachment_node_index != kInvalidIndex) {
      if (data.hierarchy.remove_attachment(attachment_node_index, local_attachment_index) == false) {
        log::error("Failed to move camera %u to its dedicated scene node", camera_index);
        return false;
      }
    } else {
      source_attachment = {SceneAttachment::Type::Camera, camera_index, 0u, 0u};
    }
    if (data.hierarchy.add_attachment(node_index, source_attachment) == false) {
      log::error("Failed to attach camera %u to its scene node", camera_index);
      return false;
    }

    const float fov = get_camera_fov(camera);
    build_camera(camera, {}, kWorldForward, kWorldUp, camera.film_size, fov);
  }
  return true;
}

SceneData::CameraInfo& select_active_camera(SceneData& data) {
  auto selected = std::find_if(data.cameras.begin(), data.cameras.end(), [](const SceneData::CameraInfo& entry) {
    return entry.active;
  });
  if (selected == data.cameras.end()) {
    selected = data.cameras.begin();
  }
  for (auto& entry : data.cameras) {
    entry.active = (&entry == &*selected);
  }
  return *selected;
}

ETX_PIMPL_IMPLEMENT(SceneRepresentation, Impl);

SceneRepresentation::SceneRepresentation(TaskScheduler& s, const IORDatabase& db) {
  ETX_PIMPL_INIT(SceneRepresentation, s, db);
}

SceneRepresentation::~SceneRepresentation() {
  ETX_PIMPL_CLEANUP(SceneRepresentation);
}

SceneData& SceneRepresentation::data() {
  return _private->data;
}

const SceneData& SceneRepresentation::data() const {
  return _private->data;
}

Camera& SceneRepresentation::mutable_camera() {
  return _private->active_camera;
}

const SceneRepresentation::MaterialMapping& SceneRepresentation::material_mapping() const {
  return _private->data.material_mapping;
}

const SceneRepresentation::MediumMapping& SceneRepresentation::medium_mapping() const {
  return _private->data.mediums.mapping();
}

const SceneRepresentation::MeshMapping& SceneRepresentation::mesh_mapping() const {
  return _private->data.mesh_mapping;
}

void SceneRepresentation::replace_loaded_scene(SceneRepresentation& source) {
  if (this == &source) {
    return;
  }
  cancel_energy_compensation_interface_preparation();
  _private->data.swap_contents(source._private->data);
  using std::swap;
  swap(_private->active_camera, source._private->active_camera);
  swap(_private->scene_valid, source._private->scene_valid);
  swap(_private->medium_authored_bounds, source._private->medium_authored_bounds);
  swap(_private->medium_bounds_scratch, source._private->medium_bounds_scratch);
  swap(_private->medium_has_bounds_scratch, source._private->medium_has_bounds_scratch);
  swap(_private->medium_attachment_nodes_scratch, source._private->medium_attachment_nodes_scratch);
  swap(_private->integrator_data, source._private->integrator_data);
  _private->integrator_data_revision += 1u;
  source._private->integrator_data_revision += 1u;
  swap(_private->rhi, source._private->rhi);
  swap(_private->scattering_gpu, source._private->scattering_gpu);
  swap(_private->scattering_gpu_ready, source._private->scattering_gpu_ready);
  swap(_private->energy_compensation_generation, source._private->energy_compensation_generation);
  swap(_private->energy_compensation_preparation_state, source._private->energy_compensation_preparation_state);
  swap(_private->energy_compensation_preparation_started_at, source._private->energy_compensation_preparation_started_at);
}

uint32_t SceneRepresentation::add_material(const char* name) {
  uint32_t index = _private->data.add_material(name);
  auto& mat = _private->data.materials[index];
  mat.cls = MaterialClass::Diffuse;
  mat.reflectance.spectrum_index = _private->data.add_spectrum(SpectralDistribution::rgb_reflectance({1.0f, 1.0f, 1.0f}));
  mat.scattering.spectrum_index = _private->data.add_spectrum(SpectralDistribution::rgb_reflectance({1.0f, 1.0f, 1.0f}));
  mat.emission.spectrum_index = _private->data.add_spectrum(SpectralDistribution::constant(0.0f));
  mat.subsurface.spectrum_index = _private->data.add_spectrum(SpectralDistribution::rgb_reflectance({1.0f, 0.2f, 0.04f}));
  mat.int_ior.cls = SpectralDistribution::Dielectric;
  mat.int_ior.eta_index = _private->data.add_spectrum(SpectralDistribution::constant(kDefaultDielectricEta));
  mat.int_ior.k_index = _private->data.add_spectrum(SpectralDistribution::constant(0.0f));
  mat.ext_ior.cls = SpectralDistribution::Dielectric;
  mat.ext_ior.eta_index = _private->data.add_spectrum(SpectralDistribution::constant(1.0f));
  mat.ext_ior.k_index = _private->data.add_spectrum(SpectralDistribution::constant(0.0f));
  return index;
}

SceneResourceEditResult SceneRepresentation::create_material(const char* name) {
  const std::string unique_name = unique_mapping_name(_private->data.material_mapping, name, "Material");
  const uint32_t material_index = add_material(unique_name.c_str());
  return {.status = SceneResourceEditStatus::Success, .resource_index = material_index};
}

SceneResourceEditResult SceneRepresentation::duplicate_material(uint32_t index) {
  SceneData& scene_data = _private->data;
  if (index >= scene_data.materials.size()) {
    return {.status = SceneResourceEditStatus::InvalidResource};
  }

  const std::string source_name = mapping_name(scene_data.material_mapping, index, "material-");
  const std::string duplicate_name = unique_mapping_name(scene_data.material_mapping, (source_name + " Copy").c_str(), "Material Copy");
  const Material material = clone_material_resources(scene_data, scene_data.materials[index]);
  const uint32_t duplicate_index = scene_data.clone_material(material, duplicate_name.c_str());
  return {.status = SceneResourceEditStatus::Success, .resource_index = duplicate_index};
}

SceneResourceEditResult SceneRepresentation::delete_material(uint32_t index) {
  SceneData& scene_data = _private->data;
  if (index >= scene_data.materials.size()) {
    return {.status = SceneResourceEditStatus::InvalidResource};
  }
  if ((scene_data.defaults.subsurface_scatter_material == index) || (scene_data.defaults.subsurface_exit_material == index) || (scene_data.defaults.missing_material == index)) {
    return {.status = SceneResourceEditStatus::ManagedResource};
  }
  for (const Triangle& triangle : scene_data.triangles) {
    if (triangle.material_index == index) {
      return {.status = SceneResourceEditStatus::ResourceInUse};
    }
  }

  SceneResourceEditResult result = {
    .status = SceneResourceEditStatus::Success,
    .resource_index = kInvalidIndex,
    .resource_remapping = removal_remapping(scene_data.materials.size(), index),
  };
  scene_data.materials.erase(scene_data.materials.begin() + index);
  erase_mapping_resource(scene_data.material_mapping, index);
  for (Triangle& triangle : scene_data.triangles) {
    if ((triangle.material_index != kInvalidIndex) && (triangle.material_index > index)) {
      --triangle.material_index;
    }
  }
  for (auto gltf_material = scene_data.gltf_material_mapping.begin(); gltf_material != scene_data.gltf_material_mapping.end();) {
    if (gltf_material->second == index) {
      gltf_material = scene_data.gltf_material_mapping.erase(gltf_material);
      continue;
    }
    if ((gltf_material->second != kInvalidIndex) && (gltf_material->second > index)) {
      --gltf_material->second;
    }
    ++gltf_material;
  }
  auto remap_default = [index](uint32_t& default_index) {
    if ((default_index != kInvalidIndex) && (default_index > index)) {
      --default_index;
    }
  };
  remap_default(scene_data.defaults.subsurface_scatter_material);
  remap_default(scene_data.defaults.subsurface_exit_material);
  remap_default(scene_data.defaults.missing_material);
  scene_data.material_to_emitter_profile.clear();
  return result;
}

std::string SceneRepresentation::rename_material(uint32_t index, const char* name) {
  return rename_entry(_private->data.material_mapping, index, name, "material-");
}

uint32_t SceneRepresentation::add_medium(const char* name) {
  SpectralDistribution absorption_spectrum = SpectralDistribution::constant(0.0f);
  SpectralDistribution scattering_spectrum = SpectralDistribution::constant(1.0f);
  uint32_t absorption_index = _private->data.add_spectrum(absorption_spectrum);
  uint32_t scattering_index = _private->data.add_spectrum(scattering_spectrum);
  std::string id = name && name[0] ? name : ("medium-" + std::to_string(_private->data.mediums.array_size()));
  return _private->data.mediums.add(Medium::Homogeneous, id, nullptr, absorption_index, scattering_index, 0.0f, true);
}

SceneResourceEditResult SceneRepresentation::create_medium(const char* name) {
  const std::string unique_name = unique_mapping_name(_private->data.mediums.mapping(), name, "Medium");
  const uint32_t medium_index = add_medium(unique_name.c_str());
  const Medium& medium = _private->data.mediums.get(medium_index);
  if (_private->medium_authored_bounds.size() <= medium_index) {
    _private->medium_authored_bounds.resize(static_cast<size_t>(medium_index) + 1u, medium.bounds);
  }
  return {.status = SceneResourceEditStatus::Success, .resource_index = medium_index};
}

SceneResourceEditResult SceneRepresentation::duplicate_medium(uint32_t index) {
  SceneData& scene_data = _private->data;
  if (index >= scene_data.mediums.array_size()) {
    return {.status = SceneResourceEditStatus::InvalidResource};
  }

  const std::string source_name = mapping_name(scene_data.mediums.mapping(), index, "medium-");
  const std::string duplicate_name = unique_mapping_name(scene_data.mediums.mapping(), (source_name + " Copy").c_str(), "Medium Copy");
  const uint32_t duplicate_index = scene_data.mediums.duplicate(index, duplicate_name);
  if (duplicate_index == kInvalidIndex) {
    return {.status = SceneResourceEditStatus::ResourceUpdateFailed};
  }
  Medium& duplicate = scene_data.mediums.get(duplicate_index);
  duplicate.absorption_index = clone_spectrum(scene_data, duplicate.absorption_index);
  duplicate.scattering_index = clone_spectrum(scene_data, duplicate.scattering_index);
  const BoundingBox authored_bounds = index < _private->medium_authored_bounds.size() ? _private->medium_authored_bounds[index] : duplicate.bounds;
  _private->medium_authored_bounds.push_back(authored_bounds);
  return {.status = SceneResourceEditStatus::Success, .resource_index = duplicate_index};
}

SceneResourceEditResult SceneRepresentation::delete_medium(uint32_t index) {
  SceneData& scene_data = _private->data;
  if (index >= scene_data.mediums.array_size()) {
    return {.status = SceneResourceEditStatus::InvalidResource};
  }
  for (const Material& material : scene_data.materials) {
    if ((material.int_medium == index) || (material.ext_medium == index)) {
      return {.status = SceneResourceEditStatus::ResourceInUse};
    }
  }
  for (const EmitterProfile& emitter : scene_data.emitter_profiles) {
    if (emitter.medium_index == index) {
      return {.status = SceneResourceEditStatus::ResourceInUse};
    }
  }
  for (const SceneData::CameraInfo& camera_info : scene_data.cameras) {
    if (camera_info.cam.medium_index == index) {
      return {.status = SceneResourceEditStatus::ResourceInUse};
    }
  }
  if ((_private->active_camera.medium_index == index) || resource_is_attached(scene_data.hierarchy, SceneAttachment::Type::Medium, index, kInvalidIndex)) {
    return {.status = SceneResourceEditStatus::ResourceInUse};
  }

  std::vector<uint32_t> remapping;
  if (scene_data.mediums.remove(index, remapping) == false) {
    return {.status = SceneResourceEditStatus::ResourceUpdateFailed};
  }
  auto remap_medium = [index](uint32_t& medium_index) {
    if ((medium_index != kInvalidIndex) && (medium_index > index)) {
      --medium_index;
    }
  };
  for (Material& material : scene_data.materials) {
    remap_medium(material.int_medium);
    remap_medium(material.ext_medium);
  }
  for (EmitterProfile& emitter : scene_data.emitter_profiles) {
    remap_medium(emitter.medium_index);
  }
  for (SceneData::CameraInfo& camera_info : scene_data.cameras) {
    remap_medium(camera_info.cam.medium_index);
  }
  remap_medium(_private->active_camera.medium_index);
  remap_hierarchy_resource(scene_data.hierarchy, SceneAttachment::Type::Medium, index);
  if (index < _private->medium_authored_bounds.size()) {
    _private->medium_authored_bounds.erase(_private->medium_authored_bounds.begin() + index);
  }
  _private->medium_bounds_scratch.clear();
  _private->medium_has_bounds_scratch.clear();
  _private->medium_attachment_nodes_scratch.clear();
  return {.status = SceneResourceEditStatus::Success, .resource_index = kInvalidIndex, .resource_remapping = std::move(remapping)};
}

std::string SceneRepresentation::rename_medium(uint32_t index, const char* name) {
  return _private->data.mediums.rename(index, (name != nullptr) ? name : "");
}

SceneResourceEditResult SceneRepresentation::create_camera(const char* name) {
  SceneData& scene_data = _private->data;
  std::vector<std::string> camera_names;
  camera_names.reserve(scene_data.cameras.size());
  for (const SceneData::CameraInfo& camera_info : scene_data.cameras) {
    camera_names.push_back(camera_info.id);
  }
  SceneData::CameraInfo camera_info = {};
  camera_info.cam = _private->active_camera;
  camera_info.id = unique_named_resource(camera_names, name, "Camera", kInvalidIndex);
  camera_info.active = scene_data.cameras.empty();
  const uint32_t camera_index = static_cast<uint32_t>(scene_data.cameras.size());
  scene_data.cameras.push_back(std::move(camera_info));
  return {.status = SceneResourceEditStatus::Success, .resource_index = camera_index};
}

SceneResourceEditResult SceneRepresentation::duplicate_camera(uint32_t index) {
  SceneData& scene_data = _private->data;
  if (index >= scene_data.cameras.size()) {
    return {.status = SceneResourceEditStatus::InvalidResource};
  }
  std::vector<std::string> camera_names;
  camera_names.reserve(scene_data.cameras.size());
  for (const SceneData::CameraInfo& camera_info : scene_data.cameras) {
    camera_names.push_back(camera_info.id);
  }
  SceneData::CameraInfo duplicate = scene_data.cameras[index];
  duplicate.active = false;
  duplicate.id = unique_named_resource(camera_names, (duplicate.id + " Copy").c_str(), "Camera Copy", kInvalidIndex);
  const uint32_t duplicate_index = static_cast<uint32_t>(scene_data.cameras.size());
  scene_data.cameras.push_back(std::move(duplicate));
  return {.status = SceneResourceEditStatus::Success, .resource_index = duplicate_index};
}

SceneResourceEditResult SceneRepresentation::delete_camera(uint32_t index) {
  SceneData& scene_data = _private->data;
  if (index >= scene_data.cameras.size()) {
    return {.status = SceneResourceEditStatus::InvalidResource};
  }
  if ((scene_data.cameras[index].active) || (scene_data.cameras.size() == 1u)) {
    return {.status = SceneResourceEditStatus::ActiveResource};
  }
  if (resource_is_attached(scene_data.hierarchy, SceneAttachment::Type::Camera, index, kInvalidIndex)) {
    return {.status = SceneResourceEditStatus::ResourceInUse};
  }
  SceneResourceEditResult result = {
    .status = SceneResourceEditStatus::Success,
    .resource_index = kInvalidIndex,
    .resource_remapping = removal_remapping(scene_data.cameras.size(), index),
  };
  scene_data.cameras.erase(scene_data.cameras.begin() + index);
  remap_hierarchy_resource(scene_data.hierarchy, SceneAttachment::Type::Camera, index);
  return result;
}

std::string SceneRepresentation::rename_camera(uint32_t index, const char* name) {
  SceneData& scene_data = _private->data;
  if (index >= scene_data.cameras.size()) {
    return {};
  }
  std::vector<std::string> camera_names;
  camera_names.reserve(scene_data.cameras.size());
  for (const SceneData::CameraInfo& camera_info : scene_data.cameras) {
    camera_names.push_back(camera_info.id);
  }
  scene_data.cameras[index].id = unique_named_resource(camera_names, name, "Camera", index);
  return scene_data.cameras[index].id;
}

SceneResourceEditResult SceneRepresentation::duplicate_emitter(uint32_t index) {
  SceneData& scene_data = _private->data;
  if (index >= scene_data.emitter_profiles.size()) {
    return {.status = SceneResourceEditStatus::InvalidResource};
  }
  if (scene_data.emitter_profiles[index].cls == EmitterProfile::Class::Area) {
    return {.status = SceneResourceEditStatus::ManagedResource};
  }

  ensure_emitter_names(scene_data);
  EmitterProfile duplicate = scene_data.emitter_profiles[index];
  duplicate.emission.spectrum_index = clone_spectrum(scene_data, duplicate.emission.spectrum_index);
  if ((duplicate.emission.image_index != kInvalidIndex) && (duplicate.emission.image_index < scene_data.images.array_size())) {
    duplicate.emission.image_index = scene_data.images.add_copy(duplicate.emission.image_index);
  }
  const uint32_t duplicate_index = static_cast<uint32_t>(scene_data.emitter_profiles.size());
  const std::string duplicate_name = unique_named_resource(scene_data.emitter_names, (scene_data.emitter_names[index] + " Copy").c_str(), "Light Copy", kInvalidIndex);
  scene_data.emitter_profiles.push_back(duplicate);
  scene_data.emitter_names.push_back(duplicate_name);
  return {.status = SceneResourceEditStatus::Success, .resource_index = duplicate_index};
}

SceneResourceEditResult SceneRepresentation::delete_emitter_profile(uint32_t index) {
  SceneData& scene_data = _private->data;
  if (index >= scene_data.emitter_profiles.size()) {
    return {.status = SceneResourceEditStatus::InvalidResource};
  }
  if (scene_data.emitter_profiles[index].cls == EmitterProfile::Class::Area) {
    return {.status = SceneResourceEditStatus::ManagedResource};
  }
  if (resource_is_attached(scene_data.hierarchy, SceneAttachment::Type::Emitter, index, kInvalidIndex)) {
    return {.status = SceneResourceEditStatus::ResourceInUse};
  }
  for (uint32_t emitter_index = 0u; emitter_index < scene_data.emitter_profiles.size(); ++emitter_index) {
    if ((emitter_index != index) && (scene_data.emitter_profiles[emitter_index].reference_emitter_index == index)) {
      return {.status = SceneResourceEditStatus::ResourceInUse};
    }
  }
  for (const Triangle& triangle : scene_data.triangles) {
    if (triangle.emitter_index == index) {
      return {.status = SceneResourceEditStatus::ResourceInUse};
    }
  }
  for (const auto& material_emitter : scene_data.material_to_emitter_profile) {
    if (material_emitter.second == index) {
      return {.status = SceneResourceEditStatus::ResourceInUse};
    }
  }

  ensure_emitter_names(scene_data);
  SceneResourceEditResult result = {
    .status = SceneResourceEditStatus::Success,
    .resource_index = kInvalidIndex,
    .resource_remapping = removal_remapping(scene_data.emitter_profiles.size(), index),
  };
  scene_data.emitter_profiles.erase(scene_data.emitter_profiles.begin() + index);
  scene_data.emitter_names.erase(scene_data.emitter_names.begin() + index);
  remap_hierarchy_resource(scene_data.hierarchy, SceneAttachment::Type::Emitter, index);
  for (EmitterProfile& emitter : scene_data.emitter_profiles) {
    if ((emitter.reference_emitter_index != kInvalidIndex) && (emitter.reference_emitter_index > index)) {
      --emitter.reference_emitter_index;
    }
  }
  for (Triangle& triangle : scene_data.triangles) {
    if ((triangle.emitter_index != kInvalidIndex) && (triangle.emitter_index > index)) {
      --triangle.emitter_index;
    }
  }
  for (auto& material_emitter : scene_data.material_to_emitter_profile) {
    if ((material_emitter.second != kInvalidIndex) && (material_emitter.second > index)) {
      --material_emitter.second;
    }
  }
  return result;
}

std::string SceneRepresentation::rename_emitter(uint32_t index, const char* name) {
  SceneData& scene_data = _private->data;
  if ((index >= scene_data.emitter_profiles.size()) || (scene_data.emitter_profiles[index].cls == EmitterProfile::Class::Area)) {
    return {};
  }
  ensure_emitter_names(scene_data);
  scene_data.emitter_names[index] = unique_named_resource(scene_data.emitter_names, name, default_emitter_name(scene_data.emitter_profiles[index]), index);
  return scene_data.emitter_names[index];
}

const std::vector<std::string>& SceneRepresentation::emitter_names() const {
  return _private->data.emitter_names;
}

void SceneRepresentation::update_medium_bounds() {
  _private->update_medium_bounds();
}

bool SceneRepresentation::synchronize_render_dependencies(const UpdateFlags& changes, bool full_update, bool& state_updated) {
  state_updated = false;
  const bool hierarchy_update = full_update || changes[UpdateFlags::AnyGeometry] || (_private->data.hierarchy.resolved_state_current() == false);
  if (hierarchy_update && (_private->data.resolve_hierarchy() == false)) {
    return false;
  }
  state_updated = hierarchy_update;

  if (full_update || changes[UpdateFlags::Materials] || changes[UpdateFlags::Meshes] || changes[UpdateFlags::Triangles]) {
    _private->create_area_emitters_from_materials();
    state_updated = true;
  }

  const bool medium_bounds_update = full_update || changes[UpdateFlags::Mediums] || changes[UpdateFlags::AnyMaterials] || changes[UpdateFlags::VerticesPos] ||
                                    changes[UpdateFlags::Triangles] || changes[UpdateFlags::Meshes] || changes[UpdateFlags::Hierarchy] || changes[UpdateFlags::Transforms] ||
                                    changes[UpdateFlags::Attachments];
  if (medium_bounds_update && (_private->update_medium_bounds() == false)) {
    return false;
  }
  state_updated = state_updated || medium_bounds_update;

  if (full_update || changes[UpdateFlags::Emitters]) {
    for (uint32_t emitter_index = 0u; emitter_index < _private->data.emitter_profiles.size(); ++emitter_index) {
      const EmitterProfile& emitter = _private->data.emitter_profiles[emitter_index];
      if ((emitter.cls == EmitterProfile::Class::Environment) && ((emitter.meta & EmitterProfile::Meta::Atmosphere) != 0u)) {
        _private->rebuild_atmosphere_emitter(emitter_index);
        state_updated = true;
      }
    }
  }
  return true;
}

void SceneRepresentation::update_active_camera() {
  if (_private->update_active_camera() == false) {
    log::error("Failed to update the active camera from the scene hierarchy");
  }
}

bool SceneRepresentation::store_active_camera() {
  auto it = std::find_if(_private->data.cameras.begin(), _private->data.cameras.end(), [](const auto& e) {
    return e.active;
  });
  if (it != _private->data.cameras.end()) {
    const uint32_t camera_index = static_cast<uint32_t>(std::distance(_private->data.cameras.begin(), it));
    AttachmentTransform transform = {};
    const bool canonical_local_pose = camera_pose_is_canonical(it->cam);
    if (canonical_local_pose && find_attachment_transform(_private->data, SceneAttachment::Type::Camera, camera_index, transform) &&
        (transform.node_index < _private->data.hierarchy.nodes.size())) {
      AffineTransform desired_world_pose = {};
      AffineTRS current_local_trs = {};
      SceneNode& camera_node = _private->data.hierarchy.nodes[transform.node_index];
      if ((camera_pose_transform(_private->active_camera, desired_world_pose) == false) || (affine_to_trs(camera_node.local_transform, current_local_trs) == false)) {
        log::error("Failed to derive a valid node transform for active camera %u", camera_index);
        return false;
      }

      AffineTransform desired_local_orientation = desired_world_pose;
      float3 desired_local_position = _private->active_camera.position;
      if (camera_node.parent_index != kInvalidIndex) {
        const uint32_t parent_index = camera_node.parent_index;
        if ((parent_index >= _private->data.hierarchy.world_transforms.size()) || (parent_index >= _private->data.hierarchy.world_orientations.size())) {
          log::error("Camera %u has an invalid parent transform", camera_index);
          return false;
        }
        AffineTransform parent_to_local = {};
        AffineTransform parent_orientation_to_local = {};
        double determinant = 0.0;
        if ((invert_affine(_private->data.hierarchy.world_transforms[parent_index], parent_to_local, determinant) == false) ||
            (invert_affine(_private->data.hierarchy.world_orientations[parent_index], parent_orientation_to_local, determinant) == false)) {
          log::error("Camera %u has a singular parent transform", camera_index);
          return false;
        }
        desired_local_position = transform_point(parent_to_local, _private->active_camera.position);
        desired_local_orientation = multiply_affine(parent_orientation_to_local, desired_world_pose);
      }

      desired_local_orientation.rows[0].x *= current_local_trs.scale.x;
      desired_local_orientation.rows[1].x *= current_local_trs.scale.x;
      desired_local_orientation.rows[2].x *= current_local_trs.scale.x;
      desired_local_orientation.rows[0].y *= current_local_trs.scale.y;
      desired_local_orientation.rows[1].y *= current_local_trs.scale.y;
      desired_local_orientation.rows[2].y *= current_local_trs.scale.y;
      desired_local_orientation.rows[0].z *= current_local_trs.scale.z;
      desired_local_orientation.rows[1].z *= current_local_trs.scale.z;
      desired_local_orientation.rows[2].z *= current_local_trs.scale.z;
      desired_local_orientation.rows[0].w = desired_local_position.x;
      desired_local_orientation.rows[1].w = desired_local_position.y;
      desired_local_orientation.rows[2].w = desired_local_position.z;

      const AffineTransform previous_local_transform = camera_node.local_transform;
      if ((_private->data.hierarchy.set_local_transform(transform.node_index, desired_local_orientation) == false) || (_private->data.resolve_hierarchy() == false)) {
        _private->data.hierarchy.set_local_transform(transform.node_index, previous_local_transform);
        if (_private->data.resolve_hierarchy() == false) {
          log::error("Failed to restore camera hierarchy after rejecting an invalid camera transform");
        }
        return false;
      }

      bool affects_scene_resources = false;
      const SceneHierarchy& hierarchy = _private->data.hierarchy;
      if ((transform.node_index < hierarchy.order_position.size()) && (transform.node_index < hierarchy.subtree_end_position.size())) {
        const uint32_t subtree_begin = hierarchy.order_position[transform.node_index];
        const uint32_t subtree_end = std::min<uint32_t>(hierarchy.subtree_end_position[transform.node_index], static_cast<uint32_t>(hierarchy.evaluation_order.size()));
        for (uint32_t order_position = subtree_begin; (order_position < subtree_end) && (affects_scene_resources == false); ++order_position) {
          const uint32_t descendant_index = hierarchy.evaluation_order[order_position];
          if (descendant_index >= hierarchy.nodes.size()) {
            continue;
          }
          const SceneNode& descendant = hierarchy.nodes[descendant_index];
          const uint32_t attachment_end = descendant.attachment_offset + descendant.attachment_count;
          for (uint32_t attachment_index = descendant.attachment_offset; attachment_index < attachment_end; ++attachment_index) {
            affects_scene_resources = hierarchy.attachments[attachment_index].type != SceneAttachment::Type::Camera;
            if (affects_scene_resources) {
              break;
            }
          }
        }
      }

      const float fov = get_camera_fov(_private->active_camera);
      it->cam = _private->active_camera;
      build_camera(it->cam, {}, kWorldForward, kWorldUp, it->cam.film_size, fov);
      return affects_scene_resources;
    }

    it->cam = find_attachment_transform(_private->data, SceneAttachment::Type::Camera, camera_index, transform)
                ? transform_camera(_private->active_camera, transform.world_to_object, transform.orientation_to_object)
                : _private->active_camera;
  }
  return false;
}

std::string SceneRepresentation::rename_mesh(uint32_t index, const char* name) {
  return rename_entry(_private->data.mesh_mapping, index, name, "mesh-");
}

void SceneRepresentation::set_mesh_material(uint32_t mesh_index, uint32_t material_index) {
  _private->set_mesh_material_impl(mesh_index, material_index);
}

SceneEditResult SceneRepresentation::create_empty_node() {
  SceneHierarchy& hierarchy = _private->data.hierarchy;
  const std::string name = unique_node_name(hierarchy, "Empty Node");
  const uint32_t node_index = hierarchy.add_node(name.c_str(), kInvalidIndex, {});
  if (node_index == kInvalidIndex) {
    return {.status = SceneEditStatus::HierarchyUpdateFailed};
  }
  if (_private->data.resolve_hierarchy() == false) {
    std::vector<uint32_t> remapping;
    hierarchy.remove_subtree(node_index, remapping);
    _private->data.resolve_hierarchy();
    return {.status = SceneEditStatus::HierarchyUpdateFailed};
  }
  return {.status = SceneEditStatus::Success, .node_index = node_index};
}

SceneEditResult SceneRepresentation::create_primitive(ScenePrimitive primitive) {
  SceneData& scene_data = _private->data;
  const size_t position_count = scene_data.vertices.pos.size();
  if ((scene_data.vertices.nrm.size() != position_count) || (scene_data.vertices.tan.size() != position_count) || (scene_data.vertices.btn.size() != position_count) ||
      (scene_data.vertices.tex.size() != position_count)) {
    return {.status = SceneEditStatus::GeometryGenerationFailed};
  }

  const size_t normal_count = scene_data.vertices.nrm.size();
  const size_t tangent_count = scene_data.vertices.tan.size();
  const size_t bitangent_count = scene_data.vertices.btn.size();
  const size_t texcoord_count = scene_data.vertices.tex.size();
  const size_t triangle_count = scene_data.triangles.size();
  const size_t mesh_count = scene_data.meshes.size();
  const size_t material_count = scene_data.materials.size();
  const size_t spectrum_count = scene_data.spectrum_values.size();
  const size_t spectrum_name_count = scene_data.spectrum_names.size();
  const SceneHierarchy original_hierarchy = scene_data.hierarchy;
  const SceneRepresentation::MeshMapping original_mesh_mapping = scene_data.mesh_mapping;
  const SceneRepresentation::MaterialMapping original_material_mapping = scene_data.material_mapping;
  auto rollback = [&]() {
    scene_data.vertices.pos.resize(position_count);
    scene_data.vertices.nrm.resize(normal_count);
    scene_data.vertices.tan.resize(tangent_count);
    scene_data.vertices.btn.resize(bitangent_count);
    scene_data.vertices.tex.resize(texcoord_count);
    scene_data.triangles.resize(triangle_count);
    scene_data.meshes.resize(mesh_count);
    scene_data.materials.resize(material_count);
    scene_data.spectrum_values.resize(spectrum_count);
    scene_data.spectrum_names.resize(spectrum_name_count);
    scene_data.hierarchy = original_hierarchy;
    scene_data.mesh_mapping = original_mesh_mapping;
    scene_data.material_mapping = original_material_mapping;
    scene_data.resolve_hierarchy();
  };

  uint32_t material_index = kInvalidIndex;
  const auto default_material = scene_data.material_mapping.find("Default Material");
  if ((default_material != scene_data.material_mapping.end()) && (default_material->second < scene_data.materials.size())) {
    material_index = default_material->second;
  } else {
    material_index = create_material("Default Material").resource_index;
  }

  ProceduralGeometryDefinition definition = {};
  definition.id = unique_node_name(scene_data.hierarchy, scene_primitive_name(primitive));
  definition.material_index = material_index;
  definition.dimensions = {1.0f, 1.0f, 1.0f};
  definition.radius = 0.5f;
  definition.segments = 64u;
  definition.subdivisions = 3u;
  switch (primitive) {
    case ScenePrimitive::Sphere:
      definition.cls = ProceduralGeometryDefinition::Class::Sphere;
      break;
    case ScenePrimitive::Box:
    case ScenePrimitive::Cube:
      definition.cls = ProceduralGeometryDefinition::Class::Box;
      break;
    case ScenePrimitive::Plane:
      definition.cls = ProceduralGeometryDefinition::Class::Plane;
      definition.dimensions.y = 0.0f;
      break;
    case ScenePrimitive::Disk:
      definition.cls = ProceduralGeometryDefinition::Class::Disk;
      definition.thickness = 0.0f;
      break;
    case ScenePrimitive::Cylinder:
      definition.cls = ProceduralGeometryDefinition::Class::Disk;
      definition.thickness = 1.0f;
      break;
    case ScenePrimitive::Cone:
      definition.cls = ProceduralGeometryDefinition::Class::Cone;
      break;
    case ScenePrimitive::Capsule:
      definition.cls = ProceduralGeometryDefinition::Class::Capsule;
      definition.dimensions.y = 2.0f;
      definition.subdivisions = 12u;
      break;
    case ScenePrimitive::Torus:
      definition.cls = ProceduralGeometryDefinition::Class::Torus;
      definition.inner_radius = 0.15f;
      definition.subdivisions = 24u;
      break;
    case ScenePrimitive::Ring:
      definition.cls = ProceduralGeometryDefinition::Class::Disk;
      definition.inner_radius = 0.25f;
      definition.thickness = 0.0f;
      break;
    case ScenePrimitive::Tube:
      definition.cls = ProceduralGeometryDefinition::Class::Disk;
      definition.inner_radius = 0.35f;
      definition.thickness = 1.0f;
      break;
    case ScenePrimitive::Tetrahedron:
      definition.cls = ProceduralGeometryDefinition::Class::Tetrahedron;
      break;
    case ScenePrimitive::Octahedron:
      definition.cls = ProceduralGeometryDefinition::Class::Octahedron;
      break;
    case ScenePrimitive::Dodecahedron:
      definition.cls = ProceduralGeometryDefinition::Class::Dodecahedron;
      break;
    case ScenePrimitive::Icosahedron:
      definition.cls = ProceduralGeometryDefinition::Class::Icosahedron;
      break;
  }

  if ((generate_procedural_geometry(scene_data, {definition}) != 1u) || (scene_data.meshes.size() != (mesh_count + 1u)) ||
      (scene_data.hierarchy.nodes.size() != (original_hierarchy.nodes.size() + 1u))) {
    rollback();
    return {.status = SceneEditStatus::GeometryGenerationFailed};
  }
  const uint32_t node_index = static_cast<uint32_t>(original_hierarchy.nodes.size());
  if (scene_data.resolve_hierarchy() == false) {
    rollback();
    return {.status = SceneEditStatus::HierarchyUpdateFailed};
  }
  return {.status = SceneEditStatus::Success, .node_index = node_index, .mesh_index = static_cast<uint32_t>(mesh_count)};
}

SceneEditResult SceneRepresentation::duplicate_node_subtree(uint32_t node_index) {
  SceneData& scene_data = _private->data;
  SceneHierarchy& hierarchy = scene_data.hierarchy;
  if (node_index >= hierarchy.nodes.size()) {
    return {.status = SceneEditStatus::InvalidNode};
  }
  if ((hierarchy.rebuild_topology() == false) || (node_index >= hierarchy.order_position.size()) || (node_index >= hierarchy.subtree_end_position.size())) {
    return {.status = SceneEditStatus::HierarchyUpdateFailed};
  }

  const uint32_t subtree_begin = hierarchy.order_position[node_index];
  const uint32_t subtree_end = std::min<uint32_t>(hierarchy.subtree_end_position[node_index], static_cast<uint32_t>(hierarchy.evaluation_order.size()));
  for (uint32_t order_position = subtree_begin; order_position < subtree_end; ++order_position) {
    const uint32_t source_index = hierarchy.evaluation_order[order_position];
    if (source_index >= hierarchy.nodes.size()) {
      return {.status = SceneEditStatus::HierarchyUpdateFailed};
    }
    const SceneNode& source_node = hierarchy.nodes[source_index];
    const uint32_t attachment_end = source_node.attachment_offset + source_node.attachment_count;
    if ((attachment_end < source_node.attachment_offset) || (attachment_end > hierarchy.attachments.size())) {
      return {.status = SceneEditStatus::HierarchyUpdateFailed};
    }
    for (uint32_t attachment_index = source_node.attachment_offset; attachment_index < attachment_end; ++attachment_index) {
      if (hierarchy.attachments[attachment_index].type != SceneAttachment::Type::Mesh) {
        return {.status = SceneEditStatus::UnsupportedAttachments};
      }
    }
  }

  SceneHierarchy original = hierarchy;
  const uint32_t original_node_count = static_cast<uint32_t>(hierarchy.nodes.size());
  const uint32_t duplicate_index = hierarchy.duplicate_subtree(node_index);
  if (duplicate_index == kInvalidIndex) {
    return {.status = SceneEditStatus::HierarchyUpdateFailed};
  }
  for (uint32_t new_node_index = original_node_count; new_node_index < hierarchy.nodes.size(); ++new_node_index) {
    const std::string source_name = (new_node_index < hierarchy.node_names.size()) ? hierarchy.node_names[new_node_index] : std::string{"Node"};
    const std::string desired_name = source_name + " Copy";
    hierarchy.node_names[new_node_index] = unique_renamed_node_name(hierarchy, new_node_index, desired_name.c_str());
  }
  const SceneEditStatus finalize_status = _private->finalize_hierarchy_edit(original);
  if (finalize_status != SceneEditStatus::Success) {
    return {.status = finalize_status};
  }
  return {.status = SceneEditStatus::Success, .node_index = duplicate_index};
}

SceneEditResult SceneRepresentation::delete_node_subtree(uint32_t node_index) {
  SceneData& scene_data = _private->data;
  SceneHierarchy& hierarchy = scene_data.hierarchy;
  if (node_index >= hierarchy.nodes.size()) {
    return {.status = SceneEditStatus::InvalidNode};
  }
  if (subtree_contains_active_camera(scene_data, node_index)) {
    return {.status = SceneEditStatus::ActiveCameraProtected};
  }

  SceneHierarchy original = hierarchy;
  std::vector<uint32_t> remapping;
  if (hierarchy.remove_subtree(node_index, remapping) == false) {
    return {.status = SceneEditStatus::HierarchyUpdateFailed};
  }
  const SceneEditStatus finalize_status = _private->finalize_hierarchy_edit(original);
  if (finalize_status != SceneEditStatus::Success) {
    return {.status = finalize_status};
  }
  return {.status = SceneEditStatus::Success, .node_index = kInvalidIndex, .node_remapping = std::move(remapping)};
}

SceneEditResult SceneRepresentation::reparent_node(uint32_t node_index, uint32_t parent_index) {
  SceneHierarchy& hierarchy = _private->data.hierarchy;
  if (node_index >= hierarchy.nodes.size()) {
    return {.status = SceneEditStatus::InvalidNode};
  }
  if ((parent_index != kInvalidIndex) && (parent_index >= hierarchy.nodes.size())) {
    return {.status = SceneEditStatus::InvalidParent};
  }
  if (hierarchy.nodes[node_index].parent_index == parent_index) {
    return {.status = SceneEditStatus::Success, .node_index = node_index};
  }

  SceneHierarchy original = hierarchy;
  if (hierarchy.reparent_preserve_world(node_index, parent_index) == false) {
    return {.status = SceneEditStatus::InvalidParent};
  }
  const SceneEditStatus finalize_status = _private->finalize_hierarchy_edit(original);
  if (finalize_status != SceneEditStatus::Success) {
    return {.status = finalize_status};
  }
  return {.status = SceneEditStatus::Success, .node_index = node_index};
}

SceneEditResult SceneRepresentation::set_node_enabled(uint32_t node_index, bool enabled) {
  SceneHierarchy& hierarchy = _private->data.hierarchy;
  if (node_index >= hierarchy.nodes.size()) {
    return {.status = SceneEditStatus::InvalidNode};
  }
  if ((enabled == false) && subtree_contains_active_camera(_private->data, node_index)) {
    return {.status = SceneEditStatus::ActiveCameraProtected};
  }

  SceneHierarchy original = hierarchy;
  if (hierarchy.set_enabled(node_index, enabled) == false) {
    return {.status = SceneEditStatus::HierarchyUpdateFailed};
  }
  const SceneEditStatus finalize_status = _private->finalize_hierarchy_edit(original);
  if (finalize_status != SceneEditStatus::Success) {
    return {.status = finalize_status};
  }
  return {.status = SceneEditStatus::Success, .node_index = node_index};
}

SceneEditResult SceneRepresentation::set_node_local_transform(uint32_t node_index, const AffineTransform& transform) {
  SceneHierarchy& hierarchy = _private->data.hierarchy;
  if (node_index >= hierarchy.nodes.size()) {
    return {.status = SceneEditStatus::InvalidNode};
  }

  SceneHierarchy original = hierarchy;
  if (hierarchy.set_local_transform(node_index, transform) == false) {
    return {.status = SceneEditStatus::HierarchyUpdateFailed};
  }
  const SceneEditStatus finalize_status = _private->finalize_hierarchy_edit(original);
  if (finalize_status != SceneEditStatus::Success) {
    return {.status = finalize_status};
  }
  return {.status = SceneEditStatus::Success, .node_index = node_index};
}

SceneEditResult SceneRepresentation::attach_node_resource(uint32_t node_index, SceneAttachment::Type type, uint32_t resource_index) {
  SceneData& scene_data = _private->data;
  SceneHierarchy& hierarchy = scene_data.hierarchy;
  if (node_index >= hierarchy.nodes.size()) {
    return {.status = SceneEditStatus::InvalidNode};
  }
  if (valid_attachment_resource(scene_data, type, resource_index) == false) {
    return {.status = SceneEditStatus::InvalidResource};
  }

  const SceneNode& node = hierarchy.nodes[node_index];
  const uint32_t attachment_end = node.attachment_offset + node.attachment_count;
  if ((attachment_end < node.attachment_offset) || (attachment_end > hierarchy.attachments.size())) {
    return {.status = SceneEditStatus::HierarchyUpdateFailed};
  }
  for (uint32_t attachment_index = node.attachment_offset; attachment_index < attachment_end; ++attachment_index) {
    const SceneAttachment& attachment = hierarchy.attachments[attachment_index];
    if ((attachment.type == type) && (attachment.resource_index == resource_index)) {
      return {.status = SceneEditStatus::DuplicateAttachment};
    }
  }
  if (((type == SceneAttachment::Type::Camera) || (type == SceneAttachment::Type::Medium)) && resource_is_attached(hierarchy, type, resource_index, node_index)) {
    return {.status = SceneEditStatus::ResourceAlreadyAttached};
  }

  SceneHierarchy original = hierarchy;
  if (hierarchy.add_attachment(node_index, {type, resource_index, 0u, 0u}) == false) {
    return {.status = SceneEditStatus::HierarchyUpdateFailed};
  }
  const SceneEditStatus finalize_status = _private->finalize_hierarchy_edit(original);
  if (finalize_status != SceneEditStatus::Success) {
    return {.status = finalize_status};
  }
  return {.status = SceneEditStatus::Success, .node_index = node_index};
}

SceneEditResult SceneRepresentation::detach_node_resource(uint32_t node_index, uint32_t local_attachment_index) {
  SceneData& scene_data = _private->data;
  SceneHierarchy& hierarchy = scene_data.hierarchy;
  if (node_index >= hierarchy.nodes.size()) {
    return {.status = SceneEditStatus::InvalidNode};
  }
  const SceneNode& node = hierarchy.nodes[node_index];
  if (local_attachment_index >= node.attachment_count) {
    return {.status = SceneEditStatus::InvalidResource};
  }
  const uint32_t attachment_index = node.attachment_offset + local_attachment_index;
  if (attachment_index >= hierarchy.attachments.size()) {
    return {.status = SceneEditStatus::HierarchyUpdateFailed};
  }
  const SceneAttachment& attachment = hierarchy.attachments[attachment_index];
  if ((attachment.type == SceneAttachment::Type::Camera) && (attachment.resource_index < scene_data.cameras.size()) && scene_data.cameras[attachment.resource_index].active) {
    return {.status = SceneEditStatus::ActiveCameraProtected};
  }

  SceneHierarchy original = hierarchy;
  if (hierarchy.remove_attachment(node_index, local_attachment_index) == false) {
    return {.status = SceneEditStatus::HierarchyUpdateFailed};
  }
  const SceneEditStatus finalize_status = _private->finalize_hierarchy_edit(original);
  if (finalize_status != SceneEditStatus::Success) {
    return {.status = finalize_status};
  }
  return {.status = SceneEditStatus::Success, .node_index = node_index};
}

std::string SceneRepresentation::rename_node(uint32_t node_index, const char* name) {
  SceneHierarchy& hierarchy = _private->data.hierarchy;
  if (node_index >= hierarchy.nodes.size()) {
    return {};
  }
  if (hierarchy.node_names.size() < hierarchy.nodes.size()) {
    hierarchy.node_names.resize(hierarchy.nodes.size());
  }
  hierarchy.node_names[node_index] = unique_renamed_node_name(hierarchy, node_index, name);
  return hierarchy.node_names[node_index];
}

NodeGeometryEditResult SceneRepresentation::validate_node_geometry_edit(uint32_t node_index, NodeGeometryOperation operation) const {
  return analyze_node_geometry_edit(_private->data, node_index, operation, false, nullptr);
}

NodeGeometryEditResult SceneRepresentation::edit_node_geometry(uint32_t node_index, NodeGeometryOperation operation) {
  SceneData& scene_data = _private->data;
  NodeGeometryEditAnalysis analysis = {};
  NodeGeometryEditResult result = analyze_node_geometry_edit(scene_data, node_index, operation, true, &analysis);
  if (result != NodeGeometryEditResult::Success) {
    return result;
  }

  SceneHierarchy& hierarchy = scene_data.hierarchy;
  const AffineTransform original_node_transform = hierarchy.nodes[node_index].local_transform;
  float3 center = {};
  if (operation == NodeGeometryOperation::CenterPivot) {
    result = compute_node_surface_center(scene_data, analysis, center);
    if (result != NodeGeometryEditResult::Success) {
      return result;
    }
  }

  AffineTransform node_transform = {};
  AffineTransform child_compensation = {};
  if (operation == NodeGeometryOperation::CenterPivot) {
    AffineTransform to_center = {};
    to_center.rows[0].w = center.x;
    to_center.rows[1].w = center.y;
    to_center.rows[2].w = center.z;
    node_transform = multiply_affine(original_node_transform, to_center);
    child_compensation.rows[0].w = -center.x;
    child_compensation.rows[1].w = -center.y;
    child_compensation.rows[2].w = -center.z;
  }

  std::vector<uint32_t> child_indices;
  std::vector<AffineTransform> original_child_transforms;
  std::vector<AffineTransform> edited_child_transforms;
  for (uint32_t candidate_index = 0u; candidate_index < hierarchy.nodes.size(); ++candidate_index) {
    const SceneNode& candidate = hierarchy.nodes[candidate_index];
    if (candidate.parent_index != node_index) {
      continue;
    }
    child_indices.push_back(candidate_index);
    original_child_transforms.push_back(candidate.local_transform);
    edited_child_transforms.push_back(operation == NodeGeometryOperation::BakeLocalTransform ? multiply_affine(original_node_transform, candidate.local_transform)
                                                                                             : multiply_affine(child_compensation, candidate.local_transform));
  }

  PendingNodeGeometry pending = {};
  result = build_edited_meshes(scene_data, analysis, operation, original_node_transform, center, pending);
  if (result != NodeGeometryEditResult::Success) {
    return result;
  }

  const size_t original_position_count = scene_data.vertices.pos.size();
  const size_t original_normal_count = scene_data.vertices.nrm.size();
  const size_t original_tangent_count = scene_data.vertices.tan.size();
  const size_t original_bitangent_count = scene_data.vertices.btn.size();
  const size_t original_texcoord_count = scene_data.vertices.tex.size();
  const size_t original_triangle_count = scene_data.triangles.size();
  const size_t original_mesh_count = scene_data.meshes.size();
  std::vector<uint32_t> original_attachment_meshes;
  original_attachment_meshes.reserve(analysis.attachment_indices.size());
  for (uint32_t attachment_index : analysis.attachment_indices) {
    original_attachment_meshes.push_back(hierarchy.attachments[attachment_index].resource_index);
  }

  scene_data.vertices.pos.insert(scene_data.vertices.pos.end(), pending.positions.begin(), pending.positions.end());
  scene_data.vertices.nrm.insert(scene_data.vertices.nrm.end(), pending.normals.begin(), pending.normals.end());
  scene_data.vertices.tan.insert(scene_data.vertices.tan.end(), pending.tangents.begin(), pending.tangents.end());
  scene_data.vertices.btn.insert(scene_data.vertices.btn.end(), pending.bitangents.begin(), pending.bitangents.end());
  scene_data.vertices.tex.insert(scene_data.vertices.tex.end(), pending.texcoords.begin(), pending.texcoords.end());
  scene_data.triangles.insert(scene_data.triangles.end(), pending.triangles.begin(), pending.triangles.end());
  scene_data.meshes.insert(scene_data.meshes.end(), pending.meshes.begin(), pending.meshes.end());
  for (uint32_t pending_index = 0u; pending_index < pending.mesh_names.size(); ++pending_index) {
    scene_data.mesh_mapping.emplace(pending.mesh_names[pending_index], static_cast<uint32_t>(original_mesh_count + pending_index));
  }
  for (uint32_t attachment_index : analysis.attachment_indices) {
    SceneAttachment& attachment = hierarchy.attachments[attachment_index];
    attachment.resource_index = pending.source_to_clone.at(attachment.resource_index);
  }

  bool hierarchy_updated = hierarchy.set_local_transform(node_index, node_transform);
  for (uint32_t child = 0u; hierarchy_updated && (child < child_indices.size()); ++child) {
    hierarchy_updated = hierarchy.set_local_transform(child_indices[child], edited_child_transforms[child]);
  }
  hierarchy_updated = hierarchy_updated && scene_data.resolve_hierarchy();
  if (hierarchy_updated) {
    return NodeGeometryEditResult::Success;
  }

  for (uint32_t attachment = 0u; attachment < analysis.attachment_indices.size(); ++attachment) {
    hierarchy.attachments[analysis.attachment_indices[attachment]].resource_index = original_attachment_meshes[attachment];
  }
  hierarchy.set_local_transform(node_index, original_node_transform);
  for (uint32_t child = 0u; child < child_indices.size(); ++child) {
    hierarchy.set_local_transform(child_indices[child], original_child_transforms[child]);
  }
  for (const std::string& mesh_name : pending.mesh_names) {
    scene_data.mesh_mapping.erase(mesh_name);
  }
  scene_data.vertices.pos.resize(original_position_count);
  scene_data.vertices.nrm.resize(original_normal_count);
  scene_data.vertices.tan.resize(original_tangent_count);
  scene_data.vertices.btn.resize(original_bitangent_count);
  scene_data.vertices.tex.resize(original_texcoord_count);
  scene_data.triangles.resize(original_triangle_count);
  scene_data.meshes.resize(original_mesh_count);
  if (scene_data.resolve_hierarchy() == false) {
    log::error("Failed to restore scene hierarchy after rejecting a node geometry edit");
  }
  return NodeGeometryEditResult::HierarchyUpdateFailed;
}

Camera& SceneRepresentation::camera() {
  return _private->active_camera;
}

const Camera& SceneRepresentation::camera() const {
  return _private->active_camera;
}

const SceneRepresentation::IntegratorData& SceneRepresentation::integrator_data() const {
  return _private->integrator_data;
}

uint64_t SceneRepresentation::integrator_data_revision() const {
  return _private->integrator_data_revision;
}

void SceneRepresentation::set_integrator_data(const IntegratorData& integrator_data) {
  _private->integrator_data = integrator_data;
  _private->integrator_data_revision += 1u;
}

bool SceneRepresentation::valid() const {
  return _private->scene_valid;
}

uint32_t SceneRepresentation::add_environment_emitter(const float3& color, uint32_t medium_index) {
  ensure_emitter_names(_private->data);
  uint32_t profile_index = uint32_t(_private->data.emitter_profiles.size());

  auto& e = _private->data.emitter_profiles.emplace_back(EmitterProfile::Class::Environment);
  e.emission.spectrum_index = _private->data.add_spectrum(SpectralDistribution::rgb_luminance(color));

  constexpr uint2 kUniformEnvImageDimensions = uint2{1u, 1u};
  constexpr float4 white_color = {1.0f, 1.0f, 1.0f, 1.0f};

  std::vector<float4> uniform_image_data(1, white_color);
  uint32_t image_options = Image::BuildSamplingTable | Image::RepeatU;
  e.emission.image_index = _private->data.add_image(uniform_image_data.data(), kUniformEnvImageDimensions, image_options, {}, {1.0f, 1.0f});
  e.medium_index = medium_index;
  _private->data.emitter_names.push_back(unique_named_resource(_private->data.emitter_names, "Environment Light", "Environment Light", kInvalidIndex));
  return profile_index;
}

uint32_t SceneRepresentation::add_directional_emitter(const float3& direction, const float3& color, float angular_diameter_degrees, uint32_t medium_index) {
  ensure_emitter_names(_private->data);
  uint32_t profile_index = uint32_t(_private->data.emitter_profiles.size());

  auto& e = _private->data.emitter_profiles.emplace_back(EmitterProfile::Class::Directional);
  e.emission.spectrum_index = _private->data.add_spectrum(SpectralDistribution::rgb_luminance(color));
  e.emission.image_index = kInvalidIndex;
  e.directional.direction = normalize(direction);
  e.directional.angular_size = angular_diameter_degrees * kPi / 180.0f;
  e.medium_index = medium_index;
  _private->data.emitter_names.push_back(unique_named_resource(_private->data.emitter_names, "Directional Light", "Directional Light", kInvalidIndex));

  return profile_index;
}

void SceneRepresentation::create_area_emitters_from_materials() {
  _private->create_area_emitters_from_materials();
}

bool SceneRepresentation::delete_emitter(uint32_t emitter_index) {
  return delete_emitter_profile(emitter_index).succeeded();
}

void SceneRepresentation::add_atmosphere_emitter(const AtmosphereEmitterParameters& params) {
  _private->add_atmosphere_emitter(params);
  ensure_emitter_names(_private->data);
}

void SceneRepresentation::rebuild_atmosphere_emitter(uint32_t emitter_index) {
  _private->rebuild_atmosphere_emitter(emitter_index);
}

void SceneRepresentation::set_scattering_rhi(RHIContext& rhi) {
  _private->set_scattering_rhi(rhi);
}

bool SceneRepresentation::ensure_energy_compensation_interfaces() {
  return _private->ensure_energy_compensation_interfaces();
}

bool SceneRepresentation::begin_energy_compensation_interface_preparation() {
  return _private->begin_energy_compensation_interface_preparation();
}

void SceneRepresentation::cancel_energy_compensation_interface_preparation() {
  _private->cancel_energy_compensation_interface_preparation();
}

EnergyCompensationPreparationState SceneRepresentation::poll_energy_compensation_interface_preparation() {
  return _private->poll_energy_compensation_interface_preparation();
}

EnergyCompensationPreparationStatus SceneRepresentation::energy_compensation_interface_preparation_status() const {
  return _private->energy_compensation_interface_preparation_status();
}

void SceneRepresentationImpl::set_scattering_rhi(RHIContext& rhi_context) {
  if ((rhi == &rhi_context) && scattering_gpu_ready) {
    return;
  }

  if ((rhi != nullptr) && ((energy_compensation_preparation_state == EnergyCompensationPreparationState::Preparing) || energy_compensation_generation.pipeline.valid())) {
    if (energy_compensation_generation.pipeline.valid()) {
      cleanup_energy_compensation_generation(*rhi, energy_compensation_generation);
    } else {
      energy_compensation_generation = {};
    }
    energy_compensation_preparation_state = EnergyCompensationPreparationState::Ready;
  }
  if ((rhi != nullptr) && scattering_gpu.initialized) {
    scattering::gpu_cleanup(*rhi, scattering_gpu);
  }

  rhi = &rhi_context;
  scattering_gpu = {};
  scattering_gpu_ready = false;
}

bool SceneRepresentationImpl::ensure_energy_compensation_interfaces() {
  if ((rhi != nullptr) && rhi->valid()) {
    return etx::ensure_energy_compensation_interfaces(data, scheduler, *rhi);
  }
  return etx::ensure_energy_compensation_interfaces(data, scheduler);
}

bool SceneRepresentationImpl::begin_energy_compensation_interface_preparation() {
  if ((rhi == nullptr) || (rhi->valid() == false)) {
    energy_compensation_preparation_state = EnergyCompensationPreparationState::Failed;
    return false;
  }

  if (energy_compensation_generation.pipeline.valid()) {
    cleanup_energy_compensation_generation(*rhi, energy_compensation_generation);
  }
  energy_compensation_generation = {};
  energy_compensation_preparation_started_at = std::chrono::steady_clock::now();
  energy_compensation_preparation_state = EnergyCompensationPreparationState::Preparing;
  return true;
}

void SceneRepresentationImpl::cancel_energy_compensation_interface_preparation() {
  if (rhi != nullptr) {
    cleanup_energy_compensation_generation(*rhi, energy_compensation_generation);
  } else {
    energy_compensation_generation = {};
  }
  energy_compensation_preparation_state = EnergyCompensationPreparationState::Ready;
}

EnergyCompensationPreparationState SceneRepresentationImpl::poll_energy_compensation_interface_preparation() {
  if (energy_compensation_preparation_state != EnergyCompensationPreparationState::Preparing) {
    return energy_compensation_preparation_state;
  }

  const EnergyCompensationGenerationResult result = generate_energy_compensation_interfaces_step(data, scheduler, *rhi, energy_compensation_generation);
  if (result == EnergyCompensationGenerationResult::Pending) {
    return energy_compensation_preparation_state;
  }

  if (result == EnergyCompensationGenerationResult::Complete) {
    energy_compensation_preparation_state = EnergyCompensationPreparationState::Ready;
    return energy_compensation_preparation_state;
  }

  cleanup_energy_compensation_generation(*rhi, energy_compensation_generation);
  energy_compensation_preparation_state = EnergyCompensationPreparationState::Failed;
  return energy_compensation_preparation_state;
}

EnergyCompensationPreparationStatus SceneRepresentationImpl::energy_compensation_interface_preparation_status() const {
  EnergyCompensationPreparationStatus result = {
    .state = energy_compensation_preparation_state,
    .completed_steps = energy_compensation_generation.completed_steps,
    .total_steps = energy_compensation_generation.total_steps,
  };
  if (energy_compensation_preparation_state != EnergyCompensationPreparationState::Preparing) {
    return result;
  }

  result.elapsed_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - energy_compensation_preparation_started_at).count();
  if (result.completed_steps > 0u) {
    const double seconds_per_step = result.elapsed_seconds / static_cast<double>(result.completed_steps);
    result.remaining_seconds = seconds_per_step * static_cast<double>(result.total_steps - result.completed_steps);
    result.remaining_available = true;
  }
  return result;
}

bool SceneRepresentationImpl::ensure_scattering_gpu_context() {
  if (rhi == nullptr) {
    log::error("SceneRepresentation atmosphere generation requires an RHI context. Call SceneRepresentation::set_scattering_rhi() first.");
    return false;
  }

  if (scattering_gpu_ready) {
    return true;
  }

  if ((scattering_gpu.initialized == false) && (scattering::gpu_init(*rhi, scattering_gpu) == false)) {
    log::error("Failed to initialize GPU atmosphere scattering context");
    return false;
  }

  if (scattering::gpu_precompute_optical_depth(*rhi, scattering_gpu) == false) {
    log::error("Failed to precompute GPU atmosphere optical depth");
    if (scattering_gpu.initialized) {
      scattering::gpu_cleanup(*rhi, scattering_gpu);
    }
    scattering_gpu = {};
    return false;
  }

  scattering_gpu_ready = true;
  return true;
}

void SceneRepresentationImpl::add_atmosphere_emitter(const AtmosphereEmitterParameters& params) {
  uint32_t emitter_index = data.add_atmosphere_emitter(params);
  if (ensure_scattering_gpu_context() == false) {
    return;
  }

  ETX_ASSERT(rhi != nullptr);
  data.build_atmosphere_and_sun_images(emitter_index, *rhi, scattering_gpu);
}

void SceneRepresentationImpl::rebuild_atmosphere_emitter(uint32_t emitter_index) {
  if (ensure_scattering_gpu_context() == false) {
    return;
  }

  ETX_ASSERT(rhi != nullptr);
  data.rebuild_atmosphere_emitter(emitter_index, *rhi, scattering_gpu);
}

template <class T>
inline void get_values(const std::vector<T>& a, T* ptr, uint64_t count) {
  for (uint64_t i = 0, e = a.size() < count ? a.size() : count; i < e; ++i) {
    *ptr++ = a[i];
  }
}

const char* scene_attachment_type_name(SceneAttachment::Type type) {
  switch (type) {
    case SceneAttachment::Type::Mesh:
      return "mesh";
    case SceneAttachment::Type::Camera:
      return "camera";
    case SceneAttachment::Type::Emitter:
      return "emitter";
    case SceneAttachment::Type::Medium:
      return "medium";
    default:
      return nullptr;
  }
}

bool scene_attachment_type_from_name(const std::string& name, SceneAttachment::Type& result) {
  if (name == "mesh") {
    result = SceneAttachment::Type::Mesh;
    return true;
  }
  if (name == "camera") {
    result = SceneAttachment::Type::Camera;
    return true;
  }
  if (name == "emitter") {
    result = SceneAttachment::Type::Emitter;
    return true;
  }
  if (name == "medium") {
    result = SceneAttachment::Type::Medium;
    return true;
  }
  return false;
}

void remap_emitter_attachments(SceneHierarchy& hierarchy, const std::vector<uint32_t>& old_to_new) {
  for (uint32_t node_index = 0u; node_index < hierarchy.nodes.size(); ++node_index) {
    uint32_t local_attachment_index = 0u;
    while (local_attachment_index < hierarchy.nodes[node_index].attachment_count) {
      const uint32_t attachment_index = hierarchy.nodes[node_index].attachment_offset + local_attachment_index;
      const SceneAttachment attachment = hierarchy.attachments[attachment_index];
      if (attachment.type != SceneAttachment::Type::Emitter) {
        ++local_attachment_index;
        continue;
      }
      if ((attachment.resource_index >= old_to_new.size()) || (old_to_new[attachment.resource_index] == kInvalidIndex)) {
        hierarchy.remove_attachment(node_index, local_attachment_index);
        continue;
      }
      const uint32_t new_index = old_to_new[attachment.resource_index];
      if (new_index != attachment.resource_index) {
        hierarchy.attachments[attachment_index].resource_index = new_index;
      }
      ++local_attachment_index;
    }
  }
}

void remap_emitter_references(std::vector<EmitterProfile>& profiles, const std::vector<uint32_t>& old_to_new) {
  for (EmitterProfile& profile : profiles) {
    if (profile.reference_emitter_index == kInvalidIndex) {
      continue;
    }
    profile.reference_emitter_index = profile.reference_emitter_index < old_to_new.size() ? old_to_new[profile.reference_emitter_index] : kInvalidIndex;
  }
}

std::string scene_attachment_resource_name(const SceneData& data, SceneAttachment::Type type, uint32_t resource_index) {
  switch (type) {
    case SceneAttachment::Type::Mesh:
      return resource_index < data.meshes.size() ? mapping_name(data.mesh_mapping, resource_index, "mesh-") : std::string{};
    case SceneAttachment::Type::Camera:
      return resource_index < data.cameras.size() ? data.cameras[resource_index].id : std::string{};
    case SceneAttachment::Type::Emitter:
      return resource_index < data.emitter_names.size() ? data.emitter_names[resource_index] : std::string{};
    case SceneAttachment::Type::Medium:
      return resource_index < data.mediums.array_size() ? mapping_name(data.mediums.mapping(), resource_index, "medium-") : std::string{};
  }
  return {};
}

uint32_t scene_attachment_resource_index(const SceneData& data, SceneAttachment::Type type, const std::string& name) {
  switch (type) {
    case SceneAttachment::Type::Mesh: {
      const auto found = data.mesh_mapping.find(name);
      return found != data.mesh_mapping.end() ? found->second : kInvalidIndex;
    }
    case SceneAttachment::Type::Camera: {
      uint32_t result = kInvalidIndex;
      for (uint32_t index = 0u; index < data.cameras.size(); ++index) {
        if (data.cameras[index].id == name) {
          if (result != kInvalidIndex) {
            return kInvalidIndex;
          }
          result = index;
        }
      }
      return result;
    }
    case SceneAttachment::Type::Emitter: {
      uint32_t result = kInvalidIndex;
      for (uint32_t index = 0u; index < data.emitter_names.size(); ++index) {
        if (data.emitter_names[index] == name) {
          if (result != kInvalidIndex) {
            return kInvalidIndex;
          }
          result = index;
        }
      }
      return result;
    }
    case SceneAttachment::Type::Medium: {
      const auto found = data.mediums.mapping().find(name);
      return found != data.mediums.mapping().end() ? found->second : kInvalidIndex;
    }
  }
  return kInvalidIndex;
}

nlohmann::json serialize_scene_hierarchy(const SceneData& data) {
  const SceneHierarchy& hierarchy = data.hierarchy;
  const std::vector<uint32_t> emitter_serialization_order = serialized_emitter_indices(data);
  std::vector<uint32_t> emitter_serialized_indices(data.emitter_profiles.size(), kInvalidIndex);
  for (uint32_t serialized_index = 0u; serialized_index < emitter_serialization_order.size(); ++serialized_index) {
    emitter_serialized_indices[emitter_serialization_order[serialized_index]] = serialized_index;
  }
  nlohmann::json result = nlohmann::json::object();
  result["version"] = 1u;
  nlohmann::json nodes = nlohmann::json::array();
  for (uint32_t node_index = 0u; node_index < hierarchy.nodes.size(); ++node_index) {
    const SceneNode& node = hierarchy.nodes[node_index];
    nlohmann::json node_json = nlohmann::json::object();
    node_json["name"] = node_index < hierarchy.node_names.size() ? hierarchy.node_names[node_index] : ("node-" + std::to_string(node_index));
    node_json["parent"] = node.parent_index == kInvalidIndex ? nlohmann::json(nullptr) : nlohmann::json(node.parent_index);
    node_json["flags"] = node.flags;
    nlohmann::json transform = nlohmann::json::array();
    for (const float4& row : node.local_transform.rows) {
      transform.push_back(row.x);
      transform.push_back(row.y);
      transform.push_back(row.z);
      transform.push_back(row.w);
    }
    node_json["transform"] = std::move(transform);

    nlohmann::json attachments = nlohmann::json::array();
    const uint32_t attachment_end = node.attachment_offset + node.attachment_count;
    if (attachment_end <= hierarchy.attachments.size()) {
      for (uint32_t attachment_index = node.attachment_offset; attachment_index < attachment_end; ++attachment_index) {
        const SceneAttachment& attachment = hierarchy.attachments[attachment_index];
        const char* type_name = scene_attachment_type_name(attachment.type);
        if (type_name == nullptr) {
          continue;
        }
        uint32_t serialized_resource_index = attachment.resource_index;
        if ((attachment.type == SceneAttachment::Type::Emitter) && (attachment.resource_index < emitter_serialized_indices.size()) &&
            (emitter_serialized_indices[attachment.resource_index] != kInvalidIndex)) {
          serialized_resource_index = emitter_serialized_indices[attachment.resource_index];
        }
        nlohmann::json attachment_json = {{"type", type_name}, {"index", serialized_resource_index}, {"flags", attachment.flags}};
        const std::string resource_name = scene_attachment_resource_name(data, attachment.type, attachment.resource_index);
        if (resource_name.empty() == false) {
          attachment_json["name"] = resource_name;
        }
        attachments.push_back(std::move(attachment_json));
      }
    }
    node_json["attachments"] = std::move(attachments);
    nodes.push_back(std::move(node_json));
  }
  result["nodes"] = std::move(nodes);
  return result;
}

bool deserialize_scene_hierarchy(const nlohmann::json& source, SceneData& data) {
  if ((source.is_object() == false) || (source.contains("version") == false) || (source["version"].is_number_unsigned() == false) || (source["version"].get<uint64_t>() != 1u) ||
      (source.contains("nodes") == false) || (source["nodes"].is_array() == false)) {
    return false;
  }

  SceneHierarchy hierarchy;
  const nlohmann::json& nodes = source["nodes"];
  std::vector<uint32_t> parent_indices;
  parent_indices.reserve(nodes.size());
  for (uint32_t node_index = 0u; node_index < nodes.size(); ++node_index) {
    const nlohmann::json& node_json = nodes[node_index];
    if ((node_json.is_object() == false) || (node_json.contains("transform") == false) || (node_json["transform"].is_array() == false) || (node_json["transform"].size() != 12u)) {
      return false;
    }

    AffineTransform transform = {};
    for (uint32_t row = 0u; row < 3u; ++row) {
      for (uint32_t column = 0u; column < 4u; ++column) {
        const nlohmann::json& value = node_json["transform"][4u * row + column];
        if (value.is_number() == false) {
          return false;
        }
        const float component = value.get<float>();
        if (std::isfinite(component) == false) {
          return false;
        }
        (&transform.rows[row].x)[column] = component;
      }
    }
    if ((node_json.contains("name") && (node_json["name"].is_string() == false)) || (node_json.contains("flags") && (node_json["flags"].is_number_unsigned() == false))) {
      return false;
    }
    const std::string name = node_json.contains("name") ? node_json["name"].get<std::string>() : ("node-" + std::to_string(node_index));
    const uint32_t new_node_index = hierarchy.add_node(name.c_str(), kInvalidIndex, transform);
    if (new_node_index == kInvalidIndex) {
      return false;
    }
    if (node_json.contains("flags")) {
      const uint64_t flags = node_json["flags"].get<uint64_t>();
      if (flags > kInvalidIndex) {
        return false;
      }
      hierarchy.nodes[new_node_index].flags = static_cast<uint32_t>(flags);
    }

    uint32_t parent_index = kInvalidIndex;
    if (node_json.contains("parent") && node_json["parent"].is_null() == false) {
      if (node_json["parent"].is_number_unsigned() == false) {
        return false;
      }
      const uint64_t parent_value = node_json["parent"].get<uint64_t>();
      if (parent_value >= kInvalidIndex) {
        return false;
      }
      parent_index = static_cast<uint32_t>(parent_value);
    }
    parent_indices.emplace_back(parent_index);
  }

  for (uint32_t node_index = 0u; node_index < nodes.size(); ++node_index) {
    if (hierarchy.set_parent(node_index, parent_indices[node_index]) == false) {
      return false;
    }
    const nlohmann::json& node_json = nodes[node_index];
    if (node_json.contains("attachments") == false) {
      continue;
    }
    if (node_json["attachments"].is_array() == false) {
      return false;
    }
    for (const nlohmann::json& attachment_json : node_json["attachments"]) {
      if ((attachment_json.is_object() == false) || (attachment_json.contains("type") == false) || (attachment_json["type"].is_string() == false) ||
          (attachment_json.contains("index") == false) || (attachment_json["index"].is_number_unsigned() == false)) {
        return false;
      }
      SceneAttachment attachment = {};
      if (scene_attachment_type_from_name(attachment_json["type"].get<std::string>(), attachment.type) == false) {
        return false;
      }
      const uint64_t resource_index = attachment_json["index"].get<uint64_t>();
      if ((resource_index >= kInvalidIndex) || (attachment_json.contains("flags") && (attachment_json["flags"].is_number_unsigned() == false))) {
        return false;
      }
      attachment.resource_index = static_cast<uint32_t>(resource_index);
      if (attachment_json.contains("name")) {
        if (attachment_json["name"].is_string() == false) {
          return false;
        }
        const uint32_t named_resource_index = scene_attachment_resource_index(data, attachment.type, attachment_json["name"].get<std::string>());
        if (named_resource_index != kInvalidIndex) {
          attachment.resource_index = named_resource_index;
        }
      }
      if (attachment_json.contains("flags")) {
        const uint64_t flags = attachment_json["flags"].get<uint64_t>();
        if (flags > kInvalidIndex) {
          return false;
        }
        attachment.flags = static_cast<uint32_t>(flags);
      }
      if (hierarchy.add_attachment(node_index, attachment) == false) {
        return false;
      }
    }
  }
  if (hierarchy.rebuild_topology() == false) {
    return false;
  }
  data.hierarchy = std::move(hierarchy);
  return true;
}

void synthesize_identity_scene_hierarchy(SceneData& data) {
  if (data.hierarchy.nodes.empty() == false) {
    return;
  }
  std::vector<std::string> mesh_names(data.meshes.size());
  for (const auto& mapping : data.mesh_mapping) {
    if ((mapping.second >= data.meshes.size()) || ((mesh_names[mapping.second].empty() == false) && (mesh_names[mapping.second] <= mapping.first))) {
      continue;
    }
    mesh_names[mapping.second] = mapping.first;
  }
  for (uint32_t mesh_index = 0u; mesh_index < data.meshes.size(); ++mesh_index) {
    const std::string node_name = mesh_names[mesh_index].empty() ? ("mesh-" + std::to_string(mesh_index)) : mesh_names[mesh_index];
    const uint32_t node_index = data.hierarchy.add_node(node_name.c_str(), kInvalidIndex, {});
    const SceneAttachment attachment = {SceneAttachment::Type::Mesh, mesh_index, 0u, 0u};
    ETX_CRITICAL(data.hierarchy.add_attachment(node_index, attachment));
  }
}

bool SceneRepresentation::load_from_file(const char* filename, uint32_t options, IntegratorData* out_integrator) {
  if ((filename == nullptr) || (filename[0] == 0)) {
    return false;
  }
  const SceneSavePaths interrupted_save_paths = scene_save_paths(filename);
  std::array<StagedSceneFile, 3u> interrupted_save_files = staged_scene_files(interrupted_save_paths);
  bool committed_scene_available = false;
  if (recover_interrupted_scene_save(interrupted_save_files, committed_scene_available) == false) {
    return false;
  }
  std::string committed_scene_file;
  if (committed_scene_available && ((options & PreferRecoveredSave) != 0u)) {
    committed_scene_file = interrupted_save_paths.json.generic_string();
    filename = committed_scene_file.c_str();
  }

  IntegratorData parsed_integrator_data = {};
  IntegratorData* integrator_data = out_integrator;
  if (integrator_data == nullptr) {
    integrator_data = &parsed_integrator_data;
  } else {
    *integrator_data = {};
  }

  char base_folder[2048] = {};
  get_file_folder(filename, base_folder, sizeof(base_folder));

  _private->cleanup();
  _private->data.json_file_name = {};
  _private->data.materials_file_name = {};
  _private->data.geometry_file_name = filename;
  _private->active_camera.lens_radius = 0.0f;
  _private->active_camera.focal_distance = 0.0f;
  _private->active_camera.lens_image = kInvalidIndex;
  _private->active_camera.medium_index = kInvalidIndex;
  _private->active_camera.up = kWorldUp;

  Camera default_camera = {};
  default_camera.lens_image = kInvalidIndex;
  default_camera.medium_index = kInvalidIndex;
  default_camera.up = kWorldUp;
  default_camera.cls = Camera::Class::Perspective;

  float3 camera_target = default_camera.position + default_camera.direction;
  bool has_target = false;
  bool has_direction = false;
  float camera_focal_len = 50.0f;
  float camera_fov = focal_length_to_fov(camera_focal_len) * 180.0f / kPi;
  bool use_focal_len = false;
  bool force_tangents = false;
  bool spectral_scene = false;
  float pixel_filter_radius = 1.5f;
  nlohmann::json hierarchy_json;
  nlohmann::json material_names_json;
  nlohmann::json medium_names_json;
  nlohmann::json camera_names_json;
  nlohmann::json spectral_overrides_json;

  const bool raw_model_file = (strcmp(get_file_ext(filename), ".json") != 0);

  if (raw_model_file == false) {
    std::string json_content;
    if (auto f = fopen(filename, "rb")) {
      size_t file_size = get_file_size(f);
      if (file_size > 0) {
        json_content.resize(file_size);
        size_t read_bytes = fread(json_content.data(), 1, json_content.size(), f);
        json_content.resize(read_bytes);
      }
      fclose(f);
    }

    nlohmann::json js = nlohmann::json::parse(json_content, nullptr, false);
    bool parsed = js.is_discarded() == false;
    bool has_bsdfs = parsed && js.is_object() && js.contains("bsdfs");
    bool is_tungsten = parsed && js.is_object() && has_bsdfs && (js.contains("primitives") || js.contains("renderer"));
    bool is_native = parsed && js.is_object() && (has_bsdfs == false) && (js.contains("geometry") || js.contains("materials") || js.contains("integrator"));

    if (is_tungsten && (is_native == false)) {
      if (js.contains("integrator") && js["integrator"].is_object()) {
        const auto& itg = js["integrator"];
        auto map_integrator = [](const std::string& s) {
          if (s == "bidirectional_path_tracer")
            return Integrator::Type::Bidirectional;
          if ((s == "vcm") || (s == "progressive_photon_map"))
            return Integrator::Type::VCM;
          if (s == "debug")
            return Integrator::Type::Debug;
          return Integrator::Type::PathTracing;
        };
        if (integrator_data != nullptr) {
          std::string t = itg.value("type", "");
          integrator_data->selected = map_integrator(t);
        }
        if (itg.contains("min_bounces") && itg["min_bounces"].is_number_integer()) {
          _private->data.options.min_path_length = static_cast<uint32_t>(std::max<int64_t>(0, itg["min_bounces"].get<int64_t>()));
        }
        if (itg.contains("max_bounces") && itg["max_bounces"].is_number_integer()) {
          _private->data.options.max_path_length = static_cast<uint32_t>(std::max<int64_t>(0, itg["max_bounces"].get<int64_t>()));
        }
      }

      if (js.contains("renderer") && js["renderer"].is_object()) {
        const auto& rnd = js["renderer"];
        if (rnd.contains("spp") && rnd["spp"].is_number_integer()) {
          _private->data.options.samples = static_cast<uint32_t>(std::max<int64_t>(1, rnd["spp"].get<int64_t>()));
        }
      }

      uint32_t load_result = load_from_tungsten_file(filename, _private->data, _private->ior_database, _private->scheduler, _private->active_camera);
      if ((load_result & SceneLoadSucceeded) == 0)
        return false;
      return _private->finalize_scene_loading(options, base_folder, load_result, camera_fov, use_focal_len, camera_focal_len, force_tangents, spectral_scene, pixel_filter_radius,
        false);
    }

    if (parsed == false) {
      log::error("Failed to parse JSON scene %s", filename);
      return false;
    }

    if (is_native) {
      _private->data.geometry_file_name.clear();
    }

    for (auto i = js.begin(), e = js.end(); i != e; ++i) {
      const auto& key = i.key();
      const auto& obj = i.value();
      std::string str_value = {};
      float float_value = 0.0f;
      int64_t int_value = 0;
      bool bool_value = false;
      if (json_get_int(i, "samples", int_value)) {
        _private->data.options.samples = static_cast<uint32_t>(max(int64_t(1), int_value));
      } else if (json_get_int(i, "random-termination-start", int_value)) {
        _private->data.options.random_path_termination = static_cast<uint32_t>(max(int64_t(1), int_value));
      } else if (json_get_int(i, "max-path-length", int_value)) {
        _private->data.options.max_path_length = static_cast<uint32_t>(max(int64_t(1), int_value));
      } else if (json_get_int(i, "min-path-length", int_value)) {
        _private->data.options.min_path_length = static_cast<uint32_t>(max(int64_t(1), int_value));
      } else if ((json_get_float(i, "noise-threshold", float_value)) && (std::isfinite(float_value))) {
        _private->data.options.noise_threshold = clamp(float_value, 0.0f, 1.0f);
      } else if ((json_get_float(i, "radiance-clamp", float_value)) && (std::isfinite(float_value))) {
        _private->data.options.radiance_clamp = max(float_value, 0.0f);
      } else if ((json_get_float(i, "pixel-filter-radius", float_value)) && (std::isfinite(float_value))) {
        pixel_filter_radius = clamp(float_value, 0.0f, 32.0f);
      } else if (json_get_string(i, "geometry", str_value)) {
        _private->data.geometry_file_name = std::string(base_folder) + str_value;
      } else if (json_get_string(i, "materials", str_value)) {
        _private->data.materials_file_name = std::string(base_folder) + str_value;
      } else if (json_get_bool(i, "spectral", bool_value)) {
        spectral_scene = bool_value;
      } else if (json_get_bool(i, "energy_compensated_specular", bool_value)) {
        (void)bool_value;
      } else if (json_get_bool(i, "multiple_importance_sampling", bool_value)) {
        _private->data.options.properties[Scene::Properties::MultipleImportanceSampling] = bool_value;
      } else if (json_get_bool(i, "blue_noise", bool_value)) {
        _private->data.options.properties[Scene::Properties::BlueNoise] = bool_value;
      } else if ((key == "scene_hierarchy") && obj.is_object()) {
        hierarchy_json = obj;
      } else if (key == "material_names") {
        material_names_json = obj;
      } else if (key == "medium_names") {
        medium_names_json = obj;
      } else if (key == "camera_names") {
        camera_names_json = obj;
      } else if (key == "spectral_overrides") {
        spectral_overrides_json = obj;
      } else if ((key == "emitter_names") && obj.is_array()) {
        _private->data.emitter_names.clear();
        for (const nlohmann::json& emitter_name : obj) {
          if (emitter_name.is_string() == false) {
            _private->data.emitter_names.clear();
            break;
          }
          _private->data.emitter_names.push_back(emitter_name.get<std::string>());
        }
      } else if (json_get_string(i, "light_sampling", str_value)) {
        if (str_value == "uniform") {
          _private->data.options.light_sampling = Scene::LightSampling::Uniform;
        } else if (str_value == "from_distribution") {
          _private->data.options.light_sampling = Scene::LightSampling::FromDistribution;
        } else if (str_value == "ris_uniform") {
          _private->data.options.light_sampling = Scene::LightSampling::RIS_Uniform;
        } else if (str_value == "ris_from_distribution") {
          _private->data.options.light_sampling = Scene::LightSampling::RIS_FromDistribution;
        }
      } else if (key == "strategies" && obj.is_object()) {
        uint32_t strategy_flags = Scene::Strategy::Default;
        for (auto strat_it = obj.begin(); strat_it != obj.end(); ++strat_it) {
          const std::string& strat_key = strat_it.key();
          if (strat_it.value().is_boolean() == false) {
            continue;
          }
          bool strat_value = strat_it.value().get<bool>();
          if (strat_key == "direct_hit") {
            strategy_flags = (strategy_flags & (~Scene::Strategy::DirectHit)) | (strat_value ? Scene::Strategy::DirectHit : 0u);
          } else if (strat_key == "next_event_estimation") {
            strategy_flags = (strategy_flags & (~Scene::Strategy::ConnectToLight)) | (strat_value ? Scene::Strategy::ConnectToLight : 0u);
          } else if (strat_key == "connect_to_light") {
            strategy_flags = (strategy_flags & (~Scene::Strategy::ConnectToLight)) | (strat_value ? Scene::Strategy::ConnectToLight : 0u);
          } else if (strat_key == "connect_to_camera") {
            strategy_flags = (strategy_flags & (~Scene::Strategy::ConnectToCamera)) | (strat_value ? Scene::Strategy::ConnectToCamera : 0u);
          } else if (strat_key == "connect_vertices") {
            strategy_flags = (strategy_flags & (~Scene::Strategy::ConnectVertices)) | (strat_value ? Scene::Strategy::ConnectVertices : 0u);
          } else if (strat_key == "merge_vertices") {
            strategy_flags = (strategy_flags & (~Scene::Strategy::MergeVertices)) | (strat_value ? Scene::Strategy::MergeVertices : 0u);
          } else if (strat_key == "multiple_importance_sampling") {
            _private->data.options.properties[Scene::Properties::MultipleImportanceSampling] = strat_value;
          } else if (strat_key == "blue_noise") {
            _private->data.options.properties[Scene::Properties::BlueNoise] = strat_value;
          }
        }
        _private->data.options.strategy_flags = strategy_flags;
      } else if (json_get_bool(i, "force-tangents", bool_value)) {
        force_tangents = bool_value;
      } else if ((key == "camera") && obj.is_object()) {
        for (auto ci = obj.begin(), ce = obj.end(); ci != ce; ++ci) {
          const auto& ckey = ci.key();
          const auto& cobj = ci.value();
          if (json_get_string(ci, "class", str_value)) {
            default_camera.cls = str_value == "eq" ? Camera::Class::Equirectangular : Camera::Class::Perspective;
          } else if (json_get_float(ci, "fov", float_value)) {
            camera_fov = float_value;
          } else if (json_get_float(ci, "focal-length", float_value)) {
            camera_focal_len = float_value;
            use_focal_len = true;
          } else if (json_get_float(ci, "lens-radius", float_value)) {
            default_camera.lens_radius = float_value;
          } else if (json_get_float(ci, "focal-distance", float_value)) {
            default_camera.focal_distance = float_value;
          } else if (json_get_float(ci, "clip-near", float_value)) {
            default_camera.clip_near = float_value;
          } else if (json_get_float(ci, "clip-far", float_value)) {
            default_camera.clip_far = float_value;
          } else if (cobj.is_array()) {
            if (ckey == "origin") {
              auto values = cobj.get<std::vector<float>>();
              get_values(values, &default_camera.position.x, 3llu);
            } else if (ckey == "target") {
              auto values = cobj.get<std::vector<float>>();
              get_values(values, &camera_target.x, 3llu);
              has_target = true;
            } else if (ckey == "direction") {
              auto values = cobj.get<std::vector<float>>();
              get_values(values, &default_camera.direction.x, 3llu);
              has_direction = true;
            } else if (ckey == "up") {
              auto values = cobj.get<std::vector<float>>();
              get_values(values, &default_camera.up.x, 3llu);
            } else if (ckey == "viewport") {
              auto values = cobj.get<std::vector<uint32_t>>();
              get_values(values, &default_camera.film_size.x, 2llu);
            } else {
              log::warning("Unhandled value in camera description : %s", key.c_str());
            }
          }
        }

        if (has_direction) {
          default_camera.direction = normalize(default_camera.direction);
        } else if (has_target) {
          default_camera.direction = normalize(camera_target - default_camera.position);
        } else {
          default_camera.direction = kWorldForward;
        }
      } else if ((key == "integrator") && obj.is_object()) {
        if (integrator_data != nullptr) {
          std::string selected_id_str;
          if (obj.contains("selected") && obj["selected"].is_string()) {
            selected_id_str = obj["selected"].get<std::string>();
            integrator_data->selected = legacy_integrator_selection_to_type(selected_id_str);
          }
          if (integrator_data->selected == Integrator::Type::Invalid) {
            if (obj.contains("type") && obj["type"].is_string()) {
              integrator_data->selected = legacy_integrator_selection_to_type(obj["type"].get<std::string>());
            } else if (obj.contains("name") && obj["name"].is_string()) {
              std::string name = obj["name"].get<std::string>();
              if (name.find("Path Tracing") != std::string::npos) {
                integrator_data->selected = Integrator::Type::PathTracing;
              } else if (name.find("Bidirectional") != std::string::npos) {
                integrator_data->selected = Integrator::Type::Bidirectional;
              } else if (name.find("Distilled") != std::string::npos) {
                integrator_data->selected = Integrator::Type::Bidirectional;
              } else if (name.find("VCM") != std::string::npos) {
                integrator_data->selected = Integrator::Type::VCM;
              } else if (name.find("UPBP") != std::string::npos) {
                integrator_data->selected = Integrator::Type::UPBP;
              } else if (name.find("Debug") != std::string::npos) {
                integrator_data->selected = Integrator::Type::Debug;
              }
            }
          }

          if (obj.contains("settings") && obj["settings"].is_object()) {
            const auto& settings_obj = obj["settings"];
            for (auto it = settings_obj.begin(); it != settings_obj.end(); ++it) {
              const std::string& type_id = it.key();
              const auto& options_array = it.value();

              Integrator::Type type = integrator_id_to_type(type_id.c_str());
              if (type == Integrator::Type::Invalid)
                continue;

              if (options_array.is_array()) {
                Options options;
                if (options.deserialize_from_json(options_array)) {
                  integrator_data->settings[type] = std::move(options);
                }
              }
            }
          }

          if ((integrator_data->selected != Integrator::Type::Invalid) && obj.contains("options") && obj["options"].is_array()) {
            Options options;
            if (options.deserialize_from_json(obj["options"])) {
              integrator_data->settings[integrator_data->selected] = std::move(options);
            }
          }
        }
      } else {
        log::warning("Unhandled value in scene description : %s", key.c_str());
      }
    }
    _private->data.json_file_name = filename;
  }

  _private->integrator_data = *integrator_data;
  _private->integrator_data_revision += 1u;

  uint32_t load_result = SceneLoadFailed;

  const char* materials_file_name = _private->data.materials_file_name.c_str();
  if (_private->data.geometry_file_name.empty()) {
    if ((materials_file_name == nullptr) || (materials_file_name[0] == 0)) {
      log::error("Scene %s does not provide geometry or materials", filename);
      return false;
    }

    char materials_base_dir[2048] = {};
    get_file_folder(materials_file_name, materials_base_dir, sizeof(materials_base_dir));
    SceneSerialization loader;
    if (loader.parse_materials_file(materials_file_name, materials_base_dir, _private->data, _private->ior_database, _private->scheduler) == false) {
      log::error("Failed to load materials from %s", materials_file_name);
      return false;
    }

    load_result = _private->data.triangles.empty() ? SceneLoadFailed : SceneLoadSucceeded;
  } else {
    const char* geometry_file_name = _private->data.geometry_file_name.c_str();
    auto ext = get_file_ext(geometry_file_name);
    if (strcmp(ext, ".etx") == 0) {
      SceneSerialization loader;
      if (loader.load_from_file(geometry_file_name, _private->data, materials_file_name, _private->ior_database, _private->scheduler) == false) {
        log::error("Failed to load ETX file from %s", geometry_file_name);
        return false;
      }
      load_result = SceneLoadSucceeded;
    } else if (strcmp(ext, ".obj") == 0) {
      load_result = load_from_obj_file(geometry_file_name, materials_file_name, _private->data, _private->ior_database, _private->scheduler);
    } else if (strcmp(ext, ".gltf") == 0) {
      load_result = load_from_gltf_file(geometry_file_name, false, _private->data, _private->scheduler, _private->active_camera);
    } else if (strcmp(ext, ".glb") == 0) {
      load_result = load_from_gltf_file(geometry_file_name, true, _private->data, _private->scheduler, _private->active_camera);
    }
  }

  if ((load_result & SceneLoadSucceeded) == 0) {
    return false;
  }

  if ((spectral_overrides_json.is_null() == false) && (apply_scene_spectral_overrides(spectral_overrides_json, _private->data) == false)) {
    log::error("Failed to restore spectral scene data from %s", filename);
    return false;
  }
  if ((medium_names_json.is_null() == false) && (restore_scene_medium_names(medium_names_json, _private->data) == false)) {
    log::error("Failed to restore medium names from %s", filename);
    return false;
  }
  if ((material_names_json.is_null() == false) && (restore_scene_material_names(material_names_json, _private->data) == false)) {
    log::error("Failed to restore material names from %s", filename);
    return false;
  }
  if ((camera_names_json.is_null() == false) && (restore_scene_camera_names(camera_names_json, _private->data) == false)) {
    log::error("Failed to restore camera names from %s", filename);
    return false;
  }

  if (hierarchy_json.is_null() == false) {
    _private->data.hierarchy.clear();
    if (deserialize_scene_hierarchy(hierarchy_json, _private->data) == false) {
      log::error("Failed to deserialize scene hierarchy from %s", filename);
      return false;
    }
  } else {
    synthesize_identity_scene_hierarchy(_private->data);
  }

  const bool setup_camera = (options & SceneRepresentation::SetupCamera) != 0u;

  if ((raw_model_file && setup_camera) && (scene_has_environment_emitter(_private->data) == false)) {
    add_default_raw_model_lighting(_private->data);
  }

  if (has_target || has_direction || default_camera.film_size.x > 0 || default_camera.lens_radius > 0.0f) {
    if (use_focal_len) {
      camera_fov = focal_length_to_fov(camera_focal_len) * 180.0f / kPi;
    }

    if (default_camera.film_size.x * default_camera.film_size.y == 0) {
      default_camera.film_size = {1280, 720};
    }

    auto& entry = _private->data.cameras.emplace_back();
    entry.id = "default";
    entry.active = _private->data.cameras.size() == 1;

    build_camera(entry.cam, default_camera.position, default_camera.direction, default_camera.up, default_camera.film_size, camera_fov);

    entry.cam.cls = default_camera.cls;
    entry.cam.lens_radius = default_camera.lens_radius;
    entry.cam.focal_distance = default_camera.focal_distance;
    entry.cam.clip_near = default_camera.clip_near;
    entry.cam.clip_far = default_camera.clip_far;
  }

  return _private->finalize_scene_loading(options, base_folder, load_result, camera_fov, use_focal_len, camera_focal_len, force_tangents, spectral_scene, pixel_filter_radius,
    hierarchy_json.is_null() == false);
}

bool SceneRepresentationImpl::update_medium_bounds() {
  const uint32_t medium_count = data.mediums.array_size();
  if (medium_count == 0u) {
    medium_authored_bounds.clear();
    return true;
  }

  medium_bounds_scratch.assign(medium_count, {});
  medium_has_bounds_scratch.assign(medium_count, 0u);
  medium_attachment_nodes_scratch.assign(medium_count, kInvalidIndex);
  bool valid = true;

  const size_t previous_authored_bound_count = medium_authored_bounds.size();
  medium_authored_bounds.resize(medium_count);
  for (size_t medium_index = previous_authored_bound_count; medium_index < medium_authored_bounds.size(); ++medium_index) {
    medium_authored_bounds[medium_index] = data.mediums.get(static_cast<uint32_t>(medium_index)).bounds;
  }

  for (uint32_t medium_index = 0u; medium_index < medium_count; ++medium_index) {
    Medium& medium = data.mediums.get(medium_index);
    medium.bounds = medium_authored_bounds[medium_index];
    medium.local_bounds = medium_authored_bounds[medium_index];
    medium.world_to_object = {};
  }

  auto add_bounds = [&](uint32_t medium_index, const float3& bounds_min, const float3& bounds_max) {
    if ((medium_index == kInvalidIndex) || (medium_index >= medium_count)) {
      return;
    }
    if (medium_has_bounds_scratch[medium_index] == 0u) {
      medium_bounds_scratch[medium_index] = {bounds_min, 0.0f, bounds_max, 0.0f};
      medium_has_bounds_scratch[medium_index] = 1u;
      return;
    }
    medium_bounds_scratch[medium_index].p_min = min(medium_bounds_scratch[medium_index].p_min, bounds_min);
    medium_bounds_scratch[medium_index].p_max = max(medium_bounds_scratch[medium_index].p_max, bounds_max);
  };

  for (const ResolvedMeshInstance& instance : data.hierarchy.mesh_instances) {
    if (((instance.flags & ResolvedMeshInstance::Enabled) == 0u) || (instance.mesh_index >= data.meshes.size())) {
      continue;
    }
    const Mesh& mesh = data.meshes[instance.mesh_index];
    const uint32_t triangle_end = mesh.triangle_offset + mesh.triangle_count;
    if (triangle_end > data.triangles.size()) {
      continue;
    }
    for (uint32_t triangle_index = mesh.triangle_offset; triangle_index < triangle_end; ++triangle_index) {
      const Triangle& triangle = data.triangles[triangle_index];
      if ((triangle.material_index >= data.materials.size()) || (triangle.i[0] >= data.vertices.pos.size()) || (triangle.i[1] >= data.vertices.pos.size()) ||
          (triangle.i[2] >= data.vertices.pos.size())) {
        continue;
      }
      const Material& material = data.materials[triangle.material_index];
      if (((material.int_medium == kInvalidIndex) || (material.int_medium >= medium_count)) && ((material.ext_medium == kInvalidIndex) || (material.ext_medium >= medium_count))) {
        continue;
      }
      const float3 v0 = transform_point(instance.object_to_world, data.vertices.pos[triangle.i[0]]);
      const float3 v1 = transform_point(instance.object_to_world, data.vertices.pos[triangle.i[1]]);
      const float3 v2 = transform_point(instance.object_to_world, data.vertices.pos[triangle.i[2]]);
      const float3 triangle_min = min(min(v0, v1), v2);
      const float3 triangle_max = max(max(v0, v1), v2);
      add_bounds(material.int_medium, triangle_min, triangle_max);
      add_bounds(material.ext_medium, triangle_min, triangle_max);
    }
  }

  for (uint32_t node_index : data.hierarchy.evaluation_order) {
    const SceneNode& node = data.hierarchy.nodes[node_index];
    const uint32_t attachment_end = node.attachment_offset + node.attachment_count;
    if ((node_index >= data.hierarchy.effective_enabled.size()) || (data.hierarchy.effective_enabled[node_index] == 0u) || (attachment_end > data.hierarchy.attachments.size()) ||
        (node_index >= data.hierarchy.world_transforms.size())) {
      continue;
    }
    for (uint32_t attachment_index = node.attachment_offset; attachment_index < attachment_end; ++attachment_index) {
      const SceneAttachment& attachment = data.hierarchy.attachments[attachment_index];
      if ((attachment.type != SceneAttachment::Type::Medium) || (attachment.resource_index >= data.mediums.array_size())) {
        continue;
      }
      if (medium_attachment_nodes_scratch[attachment.resource_index] != kInvalidIndex) {
        log::error("Medium %u is attached to multiple enabled nodes (%u and %u); medium resources support one transform", attachment.resource_index,
          medium_attachment_nodes_scratch[attachment.resource_index], node_index);
        valid = false;
        continue;
      }

      AffineTransform world_to_object = {};
      double determinant = 0.0;
      if (invert_affine(data.hierarchy.world_transforms[node_index], world_to_object, determinant) == false) {
        log::error("Medium %u is attached to node %u with a singular transform", attachment.resource_index, node_index);
        valid = false;
        continue;
      }

      medium_attachment_nodes_scratch[attachment.resource_index] = node_index;
      Medium& medium = data.mediums.get(attachment.resource_index);
      medium.world_to_object = world_to_object;
      medium.local_bounds = medium_authored_bounds[attachment.resource_index];
      medium.bounds = transform_bounding_box(data.hierarchy.world_transforms[node_index], medium.local_bounds);
    }
  }

  for (uint32_t medium_index = 0u; medium_index < medium_count; ++medium_index) {
    if ((medium_has_bounds_scratch[medium_index] == 0u) || (medium_attachment_nodes_scratch[medium_index] != kInvalidIndex)) {
      continue;
    }
    Medium& medium = data.mediums.get(medium_index);
    medium.bounds = medium_bounds_scratch[medium_index];
    medium.local_bounds = medium.bounds;
  }

  return valid;
}

void SceneRepresentationImpl::set_mesh_material_impl(uint32_t mesh_index, uint32_t material_index) {
  if (mesh_index >= data.meshes.size())
    return;

  const Mesh& mesh = data.meshes[mesh_index];
  for (uint32_t i = 0; i < mesh.triangle_count; ++i) {
    uint32_t triangle_index = mesh.triangle_offset + i;
    if (triangle_index < data.triangles.size()) {
      data.triangles[triangle_index].material_index = material_index;
    }
  }
}

std::string SceneRepresentation::save_to_file(const char* filename, Integrator::Type selected_type, Integrator* integrator_array[], size_t integrator_count) {
  auto save_start = std::chrono::high_resolution_clock::now();

  auto impl = _private;

  std::string base_file = {};
  if ((filename != nullptr) && (filename[0] != 0)) {
    base_file = filename;
  } else if (impl->data.json_file_name.empty() == false) {
    base_file = impl->data.json_file_name;
  } else if (impl->data.geometry_file_name.empty() == false) {
    base_file = impl->data.geometry_file_name;
  }

  if (base_file.empty()) {
    log::error("Unable to determine base file for saving scene");
    return {};
  }

  const SceneSavePaths paths = scene_save_paths(base_file);
  const std::filesystem::path& json_path = paths.json;
  const std::filesystem::path& materials_path = paths.materials;
  const std::filesystem::path& geometry_path = paths.geometry;
  const std::filesystem::path asset_directory = path_with_suffix(geometry_path, ".assets");
  std::array<StagedSceneFile, 3u> scene_files = staged_scene_files(paths);
  bool committed_scene_available = false;
  if (recover_interrupted_scene_save(scene_files, committed_scene_available) == false) {
    return {};
  }

  std::vector<SerializedMaterialEntry> serialized_material_entries;
  if (build_serialized_material_entries(impl->data, serialized_material_entries) == false) {
    return {};
  }
  std::vector<SerializedMediumEntry> serialized_medium_entries;
  if (build_serialized_medium_entries(impl->data, serialized_medium_entries) == false) {
    return {};
  }
  const std::vector<SerializedCameraEntry> serialized_camera_entries = build_serialized_camera_entries(impl->data);
  SceneSerialization::MaterialNameMapping serialized_material_names;
  serialized_material_names.reserve(serialized_material_entries.size());
  for (const SerializedMaterialEntry& entry : serialized_material_entries) {
    serialized_material_names[entry.material_index] = entry.id;
  }

  auto to_relative = [](const std::filesystem::path& target, const std::filesystem::path& base_folder) {
    std::error_code ec = {};
    auto relative_path = std::filesystem::relative(target, base_folder, ec);
    if (ec.value() == 0) {
      std::string result = relative_path.generic_string();
      if (result.empty()) {
        result = target.generic_string();
      }
      return result;
    }

    return target.generic_string();
  };

  std::string geometry_ref = to_relative(geometry_path, json_path.parent_path());
  std::string materials_ref = to_relative(materials_path, json_path.parent_path());

  nlohmann::json js = nlohmann::json::object();
  js["samples"] = impl->data.options.samples;
  js["random-termination-start"] = impl->data.options.random_path_termination;
  js["max-path-length"] = impl->data.options.max_path_length;
  js["min-path-length"] = impl->data.options.min_path_length;
  js["noise-threshold"] = impl->data.options.noise_threshold;
  js["radiance-clamp"] = impl->data.options.radiance_clamp;
  js["pixel-filter-radius"] = impl->data.pixel_filter.radius;
  js["geometry"] = geometry_ref;
  if (materials_ref.empty() == false) {
    js["materials"] = materials_ref;
  }
  js["spectral"] = impl->data.options.properties[Scene::Properties::Spectral];
  js["multiple_importance_sampling"] = impl->data.options.properties[Scene::Properties::MultipleImportanceSampling];
  js["blue_noise"] = impl->data.options.properties[Scene::Properties::BlueNoise];
  ensure_emitter_names(impl->data);
  js["scene_hierarchy"] = serialize_scene_hierarchy(impl->data);
  nlohmann::json material_names_json = nlohmann::json::array();
  for (const SerializedMaterialEntry& entry : serialized_material_entries) {
    material_names_json.push_back({{"id", entry.id}, {"names", entry.authored_names}});
  }
  js["material_names"] = std::move(material_names_json);
  nlohmann::json medium_names_json = nlohmann::json::array();
  for (const SerializedMediumEntry& entry : serialized_medium_entries) {
    medium_names_json.push_back({{"id", entry.id}, {"name", entry.authored_name}});
  }
  js["medium_names"] = std::move(medium_names_json);
  nlohmann::json camera_names_json = nlohmann::json::array();
  for (const SerializedCameraEntry& entry : serialized_camera_entries) {
    camera_names_json.push_back({{"id", entry.id}, {"name", entry.authored_name}});
  }
  if (camera_names_json.empty() == false) {
    js["camera_names"] = std::move(camera_names_json);
  }
  bool spectral_overrides_valid = false;
  nlohmann::json spectral_overrides = serialize_scene_spectral_overrides(impl->data, serialized_material_entries, serialized_medium_entries, spectral_overrides_valid);
  if (spectral_overrides_valid == false) {
    return {};
  }
  js["spectral_overrides"] = std::move(spectral_overrides);
  nlohmann::json emitter_names_json = nlohmann::json::array();
  for (uint32_t emitter_index : serialized_emitter_indices(impl->data)) {
    emitter_names_json.push_back(impl->data.emitter_names[emitter_index]);
  }
  js["emitter_names"] = std::move(emitter_names_json);

  switch (impl->data.options.light_sampling) {
    case Scene::LightSampling::Uniform:
      js["light_sampling"] = "uniform";
      break;
    case Scene::LightSampling::FromDistribution:
      js["light_sampling"] = "from_distribution";
      break;
    case Scene::LightSampling::RIS_Uniform:
      js["light_sampling"] = "ris_uniform";
      break;
    case Scene::LightSampling::RIS_FromDistribution:
      js["light_sampling"] = "ris_from_distribution";
      break;
    default:
      js["light_sampling"] = "ris_from_distribution";
      break;
  }

  nlohmann::json strategies = nlohmann::json::object();
  strategies["direct_hit"] = ((impl->data.options.strategy_flags & Scene::Strategy::DirectHit) == Scene::Strategy::DirectHit);
  strategies["connect_to_light"] = ((impl->data.options.strategy_flags & Scene::Strategy::ConnectToLight) == Scene::Strategy::ConnectToLight);
  strategies["connect_to_camera"] = ((impl->data.options.strategy_flags & Scene::Strategy::ConnectToCamera) == Scene::Strategy::ConnectToCamera);
  strategies["connect_vertices"] = ((impl->data.options.strategy_flags & Scene::Strategy::ConnectVertices) == Scene::Strategy::ConnectVertices);
  strategies["merge_vertices"] = ((impl->data.options.strategy_flags & Scene::Strategy::MergeVertices) == Scene::Strategy::MergeVertices);
  js["strategies"] = strategies;

  if (selected_type != Integrator::Type::Invalid && integrator_array != nullptr && integrator_count > 0) {
    nlohmann::json integrator_json;

    const char* selected_id = integrator_type_to_id(selected_type);
    if (selected_id != nullptr) {
      integrator_json["selected"] = selected_id;
    }

    nlohmann::json settings_json = nlohmann::json::object();

    for (size_t i = 0; i < integrator_count; ++i) {
      Integrator* integrator = integrator_array[i];
      if (integrator == nullptr)
        continue;

      Integrator::Type type = integrator_to_type(integrator);
      if (type == Integrator::Type::Invalid)
        continue;

      const char* type_id = integrator_type_to_id(type);
      if (type_id == nullptr)
        continue;

      nlohmann::json options_json;
      integrator->options().serialize_to_json(options_json);

      if (options_json.is_array() && options_json.size() > 0) {
        settings_json[type_id] = options_json;
      }
    }

    if (settings_json.empty() == false) {
      integrator_json["settings"] = settings_json;
    }

    if (integrator_json.empty() == false) {
      js["integrator"] = integrator_json;
    }
  }

  auto sanitize_name = [](const std::string& value) {
    std::string result = value;
    for (char& ch : result) {
      if (std::isalnum(static_cast<unsigned char>(ch)) == 0) {
        ch = '_';
      }
    }
    return result;
  };

  std::unordered_map<uint32_t, std::string> medium_names;
  medium_names.reserve(serialized_medium_entries.size());
  for (const SerializedMediumEntry& entry : serialized_medium_entries) {
    medium_names[entry.medium_index] = entry.id;
  }

  auto spectrum_rgb = [&](uint32_t index) -> float3 {
    if ((index == kInvalidIndex) || (index >= impl->data.spectrum_values.size())) {
      return {0.0f, 0.0f, 0.0f};
    }
    return impl->data.spectrum_values[index].integrated();
  };

  auto spectrum_scalar = [&](uint32_t index, float fallback) -> float {
    if ((index == kInvalidIndex) || (index >= impl->data.spectrum_values.size())) {
      return fallback;
    }
    float3 rgb = impl->data.spectrum_values[index].integrated();
    return (rgb.x + rgb.y + rgb.z) / 3.0f;
  };

  auto spectrum_by_index = [&](uint32_t index) -> const SpectralDistribution& {
    static const SpectralDistribution null_spectrum = SpectralDistribution::constant(0.0f);
    if ((index == kInvalidIndex) || (index >= impl->data.spectrum_values.size())) {
      return null_spectrum;
    }
    return impl->data.spectrum_values[index];
  };

  auto is_white_fallback_image = [&](uint32_t image_index) {
    if (image_index >= impl->data.images.array_size()) {
      return false;
    }
    const Image& image = impl->data.images.get(image_index);
    if ((image.isize.x != 1u) || (image.isize.y != 1u) || (image.isize.z != 1u) || (image.format != Image::Format::RGBA32F) || (image.pixels.f32.a == nullptr) ||
        (image.pixels.f32.count == 0u)) {
      return false;
    }
    const float4& pixel = image.pixels.f32.a[0u];
    return (pixel.x == 1.0f) && (pixel.y == 1.0f) && (pixel.z == 1.0f) && (pixel.w == 1.0f);
  };

  auto ensure_asset_directory = [&]() {
    std::error_code error;
    std::filesystem::create_directories(asset_directory, error);
    if (error) {
      log::error("Failed to create scene asset directory: %s", asset_directory.string().c_str());
      return false;
    }
    return true;
  };

  auto install_staged_asset = [&](const std::filesystem::path& staged, const std::filesystem::path& destination) {
    std::error_code error;
    std::filesystem::rename(staged, destination, error);
    if (error) {
      log::error("Failed to install scene asset: %s", destination.string().c_str());
      (void)remove_scene_save_file(staged, "staged scene asset");
      return false;
    }
    return true;
  };

  auto publish_staged_asset = [&](const std::filesystem::path& staged, const std::string& file_name) -> std::string {
    const std::filesystem::path destination = asset_directory / file_name;
    bool destination_exists = false;
    if (inspect_scene_asset_path(destination, destination_exists) == false) {
      (void)remove_scene_save_file(staged, "staged scene asset");
      return {};
    }
    if (destination_exists) {
      const bool matches = binary_files_match(destination, staged);
      (void)remove_scene_save_file(staged, "staged scene asset");
      if (matches == false) {
        log::error("Existing scene asset does not match its content hash: %s", destination.string().c_str());
        return {};
      }
      return to_relative(destination, materials_path.parent_path());
    }
    if (install_staged_asset(staged, destination) == false) {
      return {};
    }
    return to_relative(destination, materials_path.parent_path());
  };

  const std::filesystem::path application_temporary_directory = std::filesystem::path(env().tmp_folder()).lexically_normal();
  std::error_code system_temporary_path_error;
  const std::filesystem::path system_temporary_directory = std::filesystem::temp_directory_path(system_temporary_path_error).lexically_normal();
  auto managed_source_path = [&](const std::filesystem::path& source) {
    const bool system_temporary_source = (system_temporary_path_error.value() == 0) && path_is_within_directory(source, system_temporary_directory);
    return path_is_within_directory(source, application_temporary_directory) || system_temporary_source || managed_scene_asset_path(source);
  };

  auto persist_managed_image = [&](uint32_t image_index) -> std::string {
    if (ensure_asset_directory() == false) {
      return {};
    }

    const Image& image = impl->data.images.get(image_index);
    const uint64_t pixel_count = 1ull * image.isize.x * image.isize.y;
    bool pixel_data_available = false;
    if (image.format == Image::Format::RGBA32F) {
      pixel_data_available = (image.pixels.f32.a != nullptr) && (pixel_count <= image.pixels.f32.count);
    } else if (image.format == Image::Format::RGBA8) {
      pixel_data_available = (image.pixels.u8.a != nullptr) && (pixel_count <= image.pixels.u8.count);
    } else if (image.format == Image::Format::R32F) {
      pixel_data_available = (image.pixels.r32.a != nullptr) && (pixel_count <= image.pixels.r32.count);
    } else if (Image::is_compressed_bc_format(image.format)) {
      pixel_data_available = (image.pixels.compressed.a != nullptr) && (image.pixels.compressed.count > 0u);
    }
    if ((image.isize.x == 0u) || (image.isize.y == 0u) || (image.isize.z != 1u) || (pixel_count > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) ||
        (pixel_data_available == false)) {
      log::error("Cannot persist managed image %u: unsupported or incomplete pixel data", image_index);
      return {};
    }

    std::vector<float4> converted_pixels;
    const float4* pixels = image.format == Image::Format::RGBA32F ? image.pixels.f32.a : nullptr;
    if (pixels == nullptr) {
      converted_pixels.resize(static_cast<size_t>(pixel_count));
      pixels = converted_pixels.data();
    }
    for (uint32_t pixel_index = 0u; pixel_index < static_cast<uint32_t>(pixel_count); ++pixel_index) {
      const float4 pixel = image.pixel(pixel_index);
      if ((std::isfinite(pixel.x) == false) || (std::isfinite(pixel.y) == false) || (std::isfinite(pixel.z) == false) || (std::isfinite(pixel.w) == false) || (pixel.x < 0.0f) ||
          (pixel.y < 0.0f) || (pixel.z < 0.0f) || (pixel.w < 0.0f)) {
        log::error("Cannot persist managed image %u: EXR requires finite non-negative pixels", image_index);
        return {};
      }
      if (converted_pixels.empty() == false) {
        converted_pixels[pixel_index] = pixel;
      }
    }

    const std::filesystem::path staged = asset_directory / "image-save-staged.exr";
    bool staged_exists = false;
    if (inspect_scene_asset_path(staged, staged_exists) == false) {
      return {};
    }
    if (staged_exists && (remove_scene_save_file(staged, "staged scene asset") == false)) {
      return {};
    }

    std::string encode_error;
    if (save_exr_image(staged.string().c_str(), pixels, {image.isize.x, image.isize.y}, &encode_error) == false) {
      log::error("Failed to encode managed scene image %u: %s", image_index, encode_error.c_str());
      (void)remove_scene_save_file(staged, "staged scene asset");
      return {};
    }

    uint64_t hash = 0u;
    uint64_t encoded_size = 0u;
    if (hash_binary_file(staged, hash, encoded_size) == false) {
      log::error("Failed to hash managed scene image %u", image_index);
      (void)remove_scene_save_file(staged, "staged scene asset");
      return {};
    }
    return publish_staged_asset(staged, "image-" + content_hash_string(hash) + ".exr");
  };

  auto persist_managed_file = [&](const std::filesystem::path& source, const char* prefix, const char* extension) -> std::string {
    if (ensure_asset_directory() == false) {
      return {};
    }

    const std::filesystem::path staged = asset_directory / (std::string(prefix) + "save-staged");
    bool staged_exists = false;
    if (inspect_scene_asset_path(staged, staged_exists) == false) {
      return {};
    }
    if (staged_exists && (remove_scene_save_file(staged, "staged scene asset") == false)) {
      return {};
    }

    std::ifstream input(source, std::ios::binary);
    std::ofstream output(staged, std::ios::binary | std::ios::trunc);
    if ((input.is_open() == false) || (output.is_open() == false)) {
      log::error("Failed to open scene dependency for persistence: %s", source.string().c_str());
      (void)remove_scene_save_file(staged, "staged scene asset");
      return {};
    }

    std::array<uint8_t, 64u * 1024u> buffer = {};
    uint64_t hash = 0u;
    uint64_t total_size = 0u;
    while (true) {
      input.read(reinterpret_cast<char*>(buffer.data()), static_cast<std::streamsize>(buffer.size()));
      const std::streamsize bytes_read = input.gcount();
      if (bytes_read > 0) {
        output.write(reinterpret_cast<const char*>(buffer.data()), bytes_read);
        if (output.good() == false) {
          break;
        }
        hash = etx_hash64_continue(buffer.data(), static_cast<uint64_t>(bytes_read), hash);
        total_size += static_cast<uint64_t>(bytes_read);
      }
      if (bytes_read < static_cast<std::streamsize>(buffer.size())) {
        break;
      }
    }
    output.flush();
    const bool streams_succeeded = (input.bad() == false) && output.good() && (total_size > 0u);
    input.close();
    output.close();
    if ((streams_succeeded == false) || output.fail()) {
      log::error("Failed to persist scene dependency: %s", source.string().c_str());
      (void)remove_scene_save_file(staged, "staged scene asset");
      return {};
    }

    hash = etx_hash64_continue(&total_size, sizeof(total_size), hash);
    const std::filesystem::path destination = asset_directory / (std::string(prefix) + content_hash_string(hash) + extension);
    bool destination_exists = false;
    if (inspect_scene_asset_path(destination, destination_exists) == false) {
      (void)remove_scene_save_file(staged, "staged scene asset");
      return {};
    }
    if (destination_exists) {
      const bool matches = binary_files_match(staged, destination);
      (void)remove_scene_save_file(staged, "staged scene asset");
      if (matches == false) {
        log::error("Existing scene asset does not match its content hash: %s", destination.string().c_str());
        return {};
      }
      return to_relative(destination, materials_path.parent_path());
    }
    if (install_staged_asset(staged, destination) == false) {
      return {};
    }
    return to_relative(destination, materials_path.parent_path());
  };

  auto texture_path = [&](uint32_t image_index, bool omit_white_fallback) -> std::string {
    if ((image_index == kInvalidIndex) || (image_index >= impl->data.images.array_size())) {
      return {};
    }
    const std::string stored = impl->data.images.path(image_index);
    if ((stored.compare(0, 2, "##") == 0) || stored.empty()) {
      if (omit_white_fallback && is_white_fallback_image(image_index)) {
        return {};
      }
      return persist_managed_image(image_index);
    }
    const std::filesystem::path source = std::filesystem::path(stored).lexically_normal();
    if (managed_source_path(source)) {
      return persist_managed_image(image_index);
    }
    return to_relative(source, materials_path.parent_path());
  };

  auto write_path_token = [&](std::ostringstream& stream, const std::string& path, const char* context) {
    for (const unsigned char character : path) {
      if (character < 0x20u) {
        log::error("Cannot save %s path containing control characters", context);
        return false;
      }
    }
    stream << std::quoted(path);
    return true;
  };

  auto write_texture_line = [&](std::ostringstream& stream, const char* label, uint32_t image_index, uint32_t channel) {
    if (image_index == kInvalidIndex) {
      return true;
    }
    const std::string path = texture_path(image_index, false);
    if (path.empty()) {
      log::error("Cannot save %s texture: its source file is unavailable", label);
      return false;
    }
    stream << label << " ";
    if (write_path_token(stream, path, label) == false) {
      return false;
    }
    if (channel != kInvalidIndex) {
      stream << " channel " << channel;
    }
    stream << "\n";
    return true;
  };

  auto write_spectrum_line = [&](std::ostringstream& stream, const char* label, uint32_t index, bool use_gamma) {
    if ((index == kInvalidIndex) || (index >= impl->data.spectrum_values.size())) {
      return;
    }
    float3 value = spectrum_rgb(index);
    if (use_gamma) {
      value = linear_to_gamma(value);
    }
    stream << label << " " << value.x << " " << value.y << " " << value.z << "\n";
  };

  std::ostringstream materials_stream;
  materials_stream << std::setprecision(std::numeric_limits<float>::max_digits10);

  const IORDatabase& database = impl->ior_database;
  std::vector<std::pair<uint32_t, std::filesystem::path>> persisted_volume_paths;

  for (const SerializedMediumEntry& entry : serialized_medium_entries) {
    const uint32_t pool_index = entry.medium_index;
    const Medium& medium = impl->data.mediums.get(pool_index);
    materials_stream << "newmtl et::medium\n";
    materials_stream << "id " << entry.id << "\n";
    float3 absorption = impl->data.spectrum_values[medium.absorption_index].integrated();
    if ((std::fabs(absorption.x) >= kEpsilon) || (std::fabs(absorption.y) >= kEpsilon) || (std::fabs(absorption.z) >= kEpsilon)) {
      materials_stream << "absorption " << absorption.x << " " << absorption.y << " " << absorption.z << "\n";
    }
    float3 scattering = impl->data.spectrum_values[medium.scattering_index].integrated();
    if ((std::fabs(scattering.x) >= kEpsilon) || (std::fabs(scattering.y) >= kEpsilon) || (std::fabs(scattering.z) >= kEpsilon)) {
      materials_stream << "scattering " << scattering.x << " " << scattering.y << " " << scattering.z << "\n";
    }
    if (std::fabs(medium.phase_function_g) >= kEpsilon) {
      materials_stream << "anisotropy " << medium.phase_function_g << "\n";
    }
    if (medium.enable_explicit_connections == false) {
      materials_stream << "enclosed 1\n";
    }
    const std::string& source_volume_path = impl->data.mediums.volume_path(pool_index);
    if (source_volume_path.empty() == false) {
      const std::filesystem::path volume_source = std::filesystem::path(source_volume_path).lexically_normal();
      const bool persist_volume = managed_source_path(volume_source);
      const std::string saved_volume_path = persist_volume ? persist_managed_file(volume_source, "volume-", ".nvdb") : to_relative(volume_source, materials_path.parent_path());
      if (saved_volume_path.empty()) {
        log::error("Cannot save volume dependency: %s", source_volume_path.c_str());
        return {};
      }
      if (persist_volume) {
        std::filesystem::path persisted_path = saved_volume_path;
        if (persisted_path.is_relative()) {
          persisted_path = materials_path.parent_path() / persisted_path;
        }
        persisted_volume_paths.emplace_back(pool_index, persisted_path.lexically_normal());
      }
      materials_stream << "volume ";
      if (write_path_token(materials_stream, saved_volume_path, "volume") == false) {
        return {};
      }
      materials_stream << "\n";
    } else if ((medium.cls == Medium::Heterogeneous) && (medium.grid_type_enum() == DensityGrid::Type::Texture3D)) {
      log::error("Cannot save file-backed medium %s: its source volume path is unavailable", entry.authored_name.c_str());
      return {};
    } else if ((medium.cls == Medium::Heterogeneous) && (medium.grid_type_enum() == DensityGrid::Type::NoiseFunction)) {
      materials_stream << "noise type " << static_cast<uint32_t>(medium.noise_type_enum()) << " scale " << medium.grid.noise_scale << " octaves " << medium.grid.noise_octaves
                       << " lacunarity " << medium.grid.noise_lacunarity << " persistence " << medium.grid.noise_persistence << " seed " << medium.grid.noise_seed << " power "
                       << medium.grid.noise_power << " sharpness " << medium.grid.noise_sharpness << " offset " << medium.grid.noise_offset.x << " " << medium.grid.noise_offset.y
                       << " " << medium.grid.noise_offset.z << " border_fade " << medium.grid.noise_enable_border_fade << " border_fade_distance "
                       << medium.grid.noise_border_fade_distance << "\n";
    }
    materials_stream << "\n";
  }

  const auto write_camera = [&](const Camera& camera, const std::string& camera_id, bool active) {
    if ((camera.film_size.x == 0u) || (camera.film_size.y == 0u)) {
      if (camera_id.empty()) {
        return true;
      }
      log::error("Cannot save camera %s: its viewport is invalid", camera_id.c_str());
      return false;
    }
    const float3 target = camera.position + camera.direction;
    materials_stream << "newmtl et::camera\n";
    materials_stream << "class " << ((camera.cls == Camera::Class::Equirectangular) ? "eq" : "perspective") << "\n";
    materials_stream << "viewport " << camera.film_size.x << " " << camera.film_size.y << "\n";
    materials_stream << "origin " << camera.position.x << " " << camera.position.y << " " << camera.position.z << "\n";
    materials_stream << "target " << target.x << " " << target.y << " " << target.z << "\n";
    materials_stream << "up " << camera.up.x << " " << camera.up.y << " " << camera.up.z << "\n";
    materials_stream << "fov " << get_camera_fov(camera) << "\n";
    const float fov_from_focal = focal_length_to_fov(get_camera_focal_length(camera)) * 180.0f / kPi;
    if (std::fabs(fov_from_focal - get_camera_fov(camera)) > 0.01f) {
      materials_stream << "focal-length " << get_camera_focal_length(camera) << "\n";
    }
    if (camera.lens_radius > 0.0f) {
      materials_stream << "lens-radius " << camera.lens_radius << "\n";
    }
    if (camera.focal_distance > 0.0f) {
      materials_stream << "focal-distance " << camera.focal_distance << "\n";
    }
    if (camera.clip_near != 0.1f) {
      materials_stream << "clip-near " << camera.clip_near << "\n";
    }
    if (camera.clip_far != 1000.0f) {
      materials_stream << "clip-far " << camera.clip_far << "\n";
    }
    const std::string lens_shape = texture_path(camera.lens_image, false);
    if (lens_shape.empty() == false) {
      materials_stream << "shape ";
      if (write_path_token(materials_stream, lens_shape, "camera lens") == false) {
        return false;
      }
      materials_stream << "\n";
    } else if (camera.lens_image != kInvalidIndex) {
      log::error("Cannot save camera %s: its lens image source file is unavailable", camera_id.empty() ? "camera" : camera_id.c_str());
      return false;
    }
    const bool camera_medium_valid = (camera.medium_index != kInvalidIndex) && (medium_names.count(camera.medium_index) > 0);
    if (camera_medium_valid) {
      materials_stream << "ext_medium " << medium_names[camera.medium_index] << "\n";
    }
    if (camera_id.empty() == false) {
      materials_stream << "id " << camera_id << "\n";
    }
    materials_stream << "active " << (active ? 1 : 0) << "\n";
    materials_stream << "\n";
    return true;
  };

  if (impl->data.cameras.empty()) {
    if (write_camera(impl->active_camera, {}, true) == false) {
      return {};
    }
  } else {
    for (const SerializedCameraEntry& serialized_entry : serialized_camera_entries) {
      const SceneData::CameraInfo& camera_entry = impl->data.cameras[serialized_entry.camera_index];
      const Camera* camera_to_save = &camera_entry.cam;
      if (camera_entry.active) {
        AttachmentTransform transform = {};
        if (find_attachment_transform(impl->data, SceneAttachment::Type::Camera, serialized_entry.camera_index, transform) == false) {
          camera_to_save = &impl->active_camera;
        }
      }
      if (write_camera(*camera_to_save, serialized_entry.id, camera_entry.active) == false) {
        return {};
      }
    }
  }

  std::vector<uint32_t> atmosphere_emitter_indices;

  for (uint32_t i = 0; i < impl->data.emitter_profiles.size(); ++i) {
    const auto& profile = impl->data.emitter_profiles[i];
    if ((profile.meta & EmitterProfile::Meta::Atmosphere) && (profile.cls == EmitterProfile::Class::Environment)) {
      atmosphere_emitter_indices.push_back(i);
    }
  }

  for (uint32_t emitter_index : atmosphere_emitter_indices) {
    const auto& env_profile = impl->data.emitter_profiles[emitter_index];
    float3 env_color = spectrum_rgb(env_profile.emission.spectrum_index);
    const auto& scattering = env_profile.atmosphere.scattering;
    materials_stream << "newmtl et::atmosphere\n";
    materials_stream << "anisotropy " << scattering.anisotropy << "\n";
    materials_stream << "altitude " << scattering.altitude << "\n";
    materials_stream << "rayleigh " << scattering.rayleigh_scale << "\n";
    materials_stream << "mie " << scattering.mie_scale << "\n";
    materials_stream << "ozone " << scattering.ozone_scale << "\n";
    if (scattering.primary_scattering == 0u) {
      materials_stream << "primary-scattering 0\n";
    }
    if (scattering.secondary_scattering == 0u) {
      materials_stream << "secondary-scattering 0\n";
    }
    materials_stream << "quality " << env_profile.atmosphere.quality << "\n";
    materials_stream << "color " << env_color.x << " " << env_color.y << " " << env_color.z << "\n";
    const bool atmosphere_medium_valid = (env_profile.medium_index != kInvalidIndex) && (medium_names.count(env_profile.medium_index) > 0u);
    if (atmosphere_medium_valid) {
      materials_stream << "ext_medium " << medium_names[env_profile.medium_index] << "\n";
    }
    materials_stream << "\n";
  }

  for (uint32_t i = 0; i < impl->data.emitter_profiles.size(); ++i) {
    const auto& profile = impl->data.emitter_profiles[i];
    if (profile.cls != EmitterProfile::Class::Environment) {
      continue;
    }
    if (profile.meta & EmitterProfile::Meta::Atmosphere) {
      continue;
    }

    materials_stream << "newmtl et::env\n";
    std::string env_path = texture_path(profile.emission.image_index, true);
    if (env_path.empty() == false) {
      materials_stream << "image ";
      if (write_path_token(materials_stream, env_path, "environment image") == false) {
        return {};
      }
      materials_stream << "\n";
    } else if (is_white_fallback_image(profile.emission.image_index) == false) {
      log::error("Cannot save environment light: its image source file is unavailable");
      return {};
    }
    float3 env_color = spectrum_rgb(profile.emission.spectrum_index);
    materials_stream << "color " << env_color.x << " " << env_color.y << " " << env_color.z << "\n";
    float env_rotation_offset = 0.0f;
    float env_scale_u = 1.0f;
    if (profile.emission.image_index != kInvalidIndex) {
      const Image& env_image = impl->data.images.get(profile.emission.image_index);
      env_rotation_offset = env_image.offset.x;
      env_scale_u = env_image.scale.x;
    }
    if (std::fabs(env_rotation_offset) >= kEpsilon) {
      materials_stream << "rotation " << (-env_rotation_offset * 360.0f) << "\n";
    }
    if (std::fabs(env_scale_u - 1.0f) >= kEpsilon) {
      materials_stream << "scale " << env_scale_u << "\n";
    }
    bool env_medium_valid = (profile.medium_index != kInvalidIndex) && (medium_names.count(profile.medium_index) > 0);
    if (env_medium_valid) {
      materials_stream << "ext_medium " << medium_names[profile.medium_index] << "\n";
    }
    materials_stream << "\n";
  }

  for (uint32_t i = 0; i < impl->data.emitter_profiles.size(); ++i) {
    const auto& profile = impl->data.emitter_profiles[i];
    if (profile.cls != EmitterProfile::Class::Directional) {
      continue;
    }

    materials_stream << "newmtl et::dir\n";
    float3 dir_color = spectrum_rgb(profile.emission.spectrum_index);
    materials_stream << "color " << dir_color.x << " " << dir_color.y << " " << dir_color.z << "\n";
    materials_stream << "direction " << profile.directional.direction.x << " " << profile.directional.direction.y << " " << profile.directional.direction.z << "\n";
    if (profile.directional.angular_size >= kEpsilon) {
      materials_stream << "angular_diameter " << (profile.directional.angular_size * 180.0f / kPi) << "\n";
    }
    const bool references_atmosphere = (profile.reference_emitter_index != kInvalidIndex) && (profile.reference_emitter_index < impl->data.emitter_profiles.size()) &&
                                       (impl->data.emitter_profiles[profile.reference_emitter_index].cls == EmitterProfile::Class::Environment) &&
                                       ((impl->data.emitter_profiles[profile.reference_emitter_index].meta & EmitterProfile::Meta::Atmosphere) != 0u);
    if (references_atmosphere) {
      materials_stream << "use_as_sun 1\n";
      const auto atmosphere_position = std::find(atmosphere_emitter_indices.begin(), atmosphere_emitter_indices.end(), profile.reference_emitter_index);
      if (atmosphere_position != atmosphere_emitter_indices.end()) {
        materials_stream << "atmosphere_index " << std::distance(atmosphere_emitter_indices.begin(), atmosphere_position) << "\n";
      }
    }
    std::string dir_path;
    if (references_atmosphere == false) {
      dir_path = texture_path(profile.emission.image_index, false);
    }
    if (dir_path.empty() == false) {
      materials_stream << "image ";
      if (write_path_token(materials_stream, dir_path, "directional image") == false) {
        return {};
      }
      materials_stream << "\n";
    } else if ((profile.emission.image_index != kInvalidIndex) && (references_atmosphere == false)) {
      log::error("Cannot save directional light: its image source file is unavailable");
      return {};
    }
    bool dir_medium_valid = (profile.medium_index != kInvalidIndex) && (medium_names.count(profile.medium_index) > 0);
    if (dir_medium_valid) {
      materials_stream << "ext_medium " << medium_names[profile.medium_index] << "\n";
    }
    materials_stream << "\n";
  }

  for (const SerializedMaterialEntry& entry : serialized_material_entries) {
    const std::string& serialized_name = entry.id;
    const std::string& display_name = entry.authored_names.front();
    const uint32_t index = entry.material_index;
    const Material& material = impl->data.materials[index];

    materials_stream << "newmtl " << serialized_name << "\n";
    materials_stream << "material class " << material_class_to_string(material.cls) << "\n";

    write_spectrum_line(materials_stream, "Kd", material.scattering.spectrum_index, true);
    if ((material.cls == MaterialClass::Dielectric) || (material.cls == MaterialClass::Translucent) || (material.transmission.value.x > kEpsilon)) {
      write_spectrum_line(materials_stream, "Kt", material.scattering.spectrum_index, true);
    }
    write_spectrum_line(materials_stream, "Ks", material.reflectance.spectrum_index, true);

    float rough_u = material.roughness.value.x;
    float rough_v = material.roughness.value.y;
    if ((rough_u >= kEpsilon) || (rough_v >= kEpsilon)) {
      float value_u = std::sqrt(max(0.0f, rough_u));
      float value_v = std::sqrt(max(0.0f, rough_v));
      if (std::fabs(value_u - value_v) < kEpsilon) {
        materials_stream << "Pr " << value_u << "\n";
      } else {
        materials_stream << "Pr " << value_u << " " << value_v << "\n";
      }
    }

    if (material.metalness.value.x >= kEpsilon) {
      materials_stream << "metalness " << material.metalness.value.x << "\n";
    }
    if (material.transmission.value.x >= kEpsilon) {
      materials_stream << "transmission " << material.transmission.value.x << "\n";
    }

    if ((write_texture_line(materials_stream, "map_Kd", material.scattering.image_index, kInvalidIndex) == false) ||
        (write_texture_line(materials_stream, "map_Ks", material.reflectance.image_index, kInvalidIndex) == false) ||
        (write_texture_line(materials_stream, "map_Kt", material.scattering.image_index, kInvalidIndex) == false) ||
        (write_texture_line(materials_stream, "map_Pr", material.roughness.image_index, material.roughness.channel) == false) ||
        (write_texture_line(materials_stream, "map_Ml", material.metalness.image_index, material.metalness.channel) == false) ||
        (write_texture_line(materials_stream, "map_Tm", material.transmission.image_index, material.transmission.channel) == false)) {
      return {};
    }

    if ((material.normal_image_index != kInvalidIndex) || (std::fabs(material.normal_scale - 1.0f) >= kEpsilon)) {
      std::string normal_path = texture_path(material.normal_image_index, false);
      materials_stream << "normalmap";
      if (normal_path.empty() == false) {
        materials_stream << " image ";
        if (write_path_token(materials_stream, normal_path, "normal map") == false) {
          return {};
        }
      } else if (material.normal_image_index != kInvalidIndex) {
        log::error("Cannot save material %s: its normal map source file is unavailable", display_name.c_str());
        return {};
      }
      materials_stream << " scale " << material.normal_scale << "\n";
    }

    int matched_int_index = -1;
    if (material.int_ior.cls != SpectralDistribution::Invalid) {
      matched_int_index = database.find_matching_index(spectrum_by_index(material.int_ior.eta_index), spectrum_by_index(material.int_ior.k_index), material.int_ior.cls);
    }
    if ((matched_int_index >= 0) && (matched_int_index < static_cast<int>(database.definitions.size()))) {
      const IORDefinition& def = database.definitions[static_cast<size_t>(matched_int_index)];
      materials_stream << "int_ior " << def.name << "\n";
    } else if ((material.int_ior.eta_index != kInvalidIndex) && (material.int_ior.cls != SpectralDistribution::Invalid)) {
      float eta_value = spectrum_scalar(material.int_ior.eta_index, 1.0f);
      if (material.int_ior.cls == SpectralDistribution::Dielectric) {
        materials_stream << "int_ior " << eta_value << "\n";
      } else if (material.int_ior.cls == SpectralDistribution::Conductor) {
        float k_value = spectrum_scalar(material.int_ior.k_index, 0.0f);
        materials_stream << "int_ior " << eta_value << " " << k_value << "\n";
      }
    }

    int matched_ext_index = -1;
    if (material.ext_ior.cls != SpectralDistribution::Invalid) {
      matched_ext_index = database.find_matching_index(spectrum_by_index(material.ext_ior.eta_index), spectrum_by_index(material.ext_ior.k_index), material.ext_ior.cls);
    }
    if ((matched_ext_index >= 0) && (matched_ext_index < static_cast<int>(database.definitions.size()))) {
      const IORDefinition& def = database.definitions[static_cast<size_t>(matched_ext_index)];
      materials_stream << "ext_ior " << def.name << "\n";
    } else {
      float ext_eta_value = spectrum_scalar(material.ext_ior.eta_index, 1.0f);
      if ((material.ext_ior.eta_index != kInvalidIndex) && (material.ext_ior.cls != SpectralDistribution::Invalid) &&
          (material.ext_ior.cls != SpectralDistribution::Dielectric || std::fabs(ext_eta_value - 1.0f) >= kEpsilon)) {
        if (material.ext_ior.cls == SpectralDistribution::Dielectric) {
          materials_stream << "ext_ior " << ext_eta_value << "\n";
        } else if (material.ext_ior.cls == SpectralDistribution::Conductor) {
          float ext_k_value = spectrum_scalar(material.ext_ior.k_index, 0.0f);
          materials_stream << "ext_ior " << ext_eta_value << " " << ext_k_value << "\n";
        }
      } else {
        materials_stream << "ext_ior 1.0\n";
      }
    }

    if (medium_names.count(material.int_medium) > 0u) {
      materials_stream << "int_medium " << medium_names[material.int_medium] << "\n";
    }
    if (medium_names.count(material.ext_medium) > 0u) {
      materials_stream << "ext_medium " << medium_names[material.ext_medium] << "\n";
    }

    if (material.two_sided != 0u) {
      materials_stream << "two_sided 1\n";
    }
    if (std::fabs(material.opacity - 1.0f) >= kEpsilon) {
      materials_stream << "opacity " << material.opacity << "\n";
    }

    bool has_emission_texture = (material.emission.image_index != kInvalidIndex);
    bool has_emission_spectrum = (material.emission.spectrum_index != kInvalidIndex) && (material.emission.spectrum_index < impl->data.spectrum_values.size());
    if (has_emission_texture || has_emission_spectrum) {
      materials_stream << "emitter";
      if (has_emission_texture) {
        std::string emission_path = texture_path(material.emission.image_index, false);
        if (emission_path.empty() == false) {
          materials_stream << " image ";
          if (write_path_token(materials_stream, emission_path, "emission texture") == false) {
            return {};
          }
        } else {
          log::error("Cannot save material %s: its emission texture source file is unavailable", display_name.c_str());
          return {};
        }
      }
      if (has_emission_spectrum) {
        float3 emission_value = spectrum_rgb(material.emission.spectrum_index);
        materials_stream << " color " << emission_value.x << " " << emission_value.y << " " << emission_value.z;
      }
      if (material.two_sided != 0u) {
        materials_stream << " twosided";
      }
      if (material.emission_collimation >= kEpsilon) {
        materials_stream << " collimated " << material.emission_collimation;
      }
      materials_stream << "\n";
    }

    if (material.subsurface_cls != SubsurfaceMaterial::Disabled) {
      materials_stream << "subsurface";
      if (material.subsurface.image_index != kInvalidIndex) {
        const std::string subsurface_path = texture_path(material.subsurface.image_index, false);
        if (subsurface_path.empty()) {
          log::error("Cannot save material %s: its subsurface texture has no source file", display_name.c_str());
          return {};
        }
        materials_stream << " image ";
        if (write_path_token(materials_stream, subsurface_path, "subsurface texture") == false) {
          return {};
        }
      }
      if (material.subsurface_path == SubsurfaceMaterial::RefractedPath) {
        materials_stream << " path refracted";
      }
      float3 subsurface_color = spectrum_rgb(material.subsurface.spectrum_index);
      materials_stream << " distances " << subsurface_color.x << " " << subsurface_color.y << " " << subsurface_color.z;
      materials_stream << "\n";
    }

    if ((material.thinfilm.thinkness_image != kInvalidIndex) || (std::fabs(material.thinfilm.min_thickness) >= kEpsilon) ||
        (std::fabs(material.thinfilm.max_thickness) >= kEpsilon) || (std::fabs(material.thinfilm.weight - 1.0f) >= kEpsilon)) {
      materials_stream << "thinfilm";
      std::string thinfilm_path = texture_path(material.thinfilm.thinkness_image, false);
      if (thinfilm_path.empty() == false) {
        materials_stream << " image ";
        if (write_path_token(materials_stream, thinfilm_path, "thin-film texture") == false) {
          return {};
        }
      } else if (material.thinfilm.thinkness_image != kInvalidIndex) {
        log::error("Cannot save material %s: its thin-film texture source file is unavailable", display_name.c_str());
        return {};
      }
      materials_stream << " range " << material.thinfilm.min_thickness << " " << material.thinfilm.max_thickness;
      materials_stream << " weight " << clamp(material.thinfilm.weight, 0.0f, 1.0f);
      int matched_thinfilm_index = -1;
      if (material.thinfilm.ior.cls != SpectralDistribution::Invalid) {
        matched_thinfilm_index =
          database.find_matching_index(spectrum_by_index(material.thinfilm.ior.eta_index), spectrum_by_index(material.thinfilm.ior.k_index), material.thinfilm.ior.cls);
      }
      if ((matched_thinfilm_index >= 0) && (matched_thinfilm_index < static_cast<int>(database.definitions.size()))) {
        const IORDefinition& def = database.definitions[static_cast<size_t>(matched_thinfilm_index)];
        materials_stream << " ior " << def.name << "\n";
      } else {
        float thinfilm_eta = spectrum_scalar(material.thinfilm.ior.eta_index, 1.0f);
        materials_stream << " ior " << thinfilm_eta << "\n";
      }
    }

    if (material.cls == MaterialClass::DiffractionGrating) {
      materials_stream << "diffraction_grating period_nm " << material.diffraction_grating.period_nm;
      materials_stream << " optical_path_difference_nm " << material.diffraction_grating.optical_path_difference_nm;
      materials_stream << " duty_cycle " << material.diffraction_grating.duty_cycle;
      materials_stream << " rotation_degrees " << material.diffraction_grating.rotation * 180.0f / kPi;
      materials_stream << "\n";
    }

    materials_stream << "\n";
  }

  const std::string materials_string = materials_stream.str();

  std::string json_string;
  try {
    json_string = js.dump(2);
  } catch (const nlohmann::json::exception& error) {
    log::error("Failed to serialize scene config: %s", error.what());
    return {};
  }

  const auto geometry_export_start = std::chrono::high_resolution_clock::now();
  SceneSerialization archive;
  if (archive.save_to_file(impl->data, scene_files[0].staged, serialized_material_names) == false) {
    log::error("Failed to stage geometry for %s", geometry_path.string().c_str());
    discard_staged_scene_files(scene_files);
    return {};
  }
  const auto geometry_export_end = std::chrono::high_resolution_clock::now();
  const auto geometry_export_duration = std::chrono::duration_cast<std::chrono::milliseconds>(geometry_export_end - geometry_export_start);
  log::info("Geometry export: %lld ms", geometry_export_duration.count());

  const auto materials_write_start = std::chrono::high_resolution_clock::now();
  if (write_file_contents(scene_files[1].staged, materials_string) == false) {
    discard_staged_scene_files(scene_files);
    return {};
  }
  const auto materials_write_end = std::chrono::high_resolution_clock::now();
  const auto materials_write_duration = std::chrono::duration_cast<std::chrono::milliseconds>(materials_write_end - materials_write_start);
  log::info("Materials file write: %lld ms (%zu bytes)", materials_write_duration.count(), materials_string.size());

  const auto json_write_start = std::chrono::high_resolution_clock::now();
  if (write_file_contents(scene_files[2].staged, json_string) == false) {
    discard_staged_scene_files(scene_files);
    return {};
  }
  const auto json_write_end = std::chrono::high_resolution_clock::now();
  const auto json_write_duration = std::chrono::duration_cast<std::chrono::milliseconds>(json_write_end - json_write_start);
  log::info("JSON config write: %lld ms", json_write_duration.count());

  if (commit_staged_scene_files(scene_files) == false) {
    return {};
  }

  for (const auto& [medium_index, volume_path] : persisted_volume_paths) {
    impl->data.mediums.set_volume_path(medium_index, volume_path.generic_string());
  }
  impl->data.json_file_name = json_path.generic_string();
  impl->data.materials_file_name = materials_path.generic_string();

  auto save_end = std::chrono::high_resolution_clock::now();
  auto save_duration = std::chrono::duration_cast<std::chrono::milliseconds>(save_end - save_start);
  log::info("Scene save total: %lld ms", save_duration.count());

  return json_path.generic_string();
}

void SceneRepresentationImpl::generate_pixel_sampler_image(float radius) {
  std::vector<float4> sampler_image;
  Film::generate_filter_image(Film::PixelFilterBlackmanHarris, sampler_image);
  uint32_t image_options = Image::BuildSamplingTable | Image::UniformSamplingTable;
  uint32_t image_index = data.images.add_from_data(sampler_image.data(), {Film::PixelFilterSize, Film::PixelFilterSize}, image_options, {}, {1.0f, 1.0f});
  data.pixel_filter = {image_index, radius};
}

void SceneRepresentationImpl::setup_atmosphere_references() {
  for (auto& profile : data.emitter_profiles) {
    if (profile.cls != EmitterProfile::Class::Directional) {
      continue;
    }

    const bool invalid_reference = (profile.reference_emitter_index == kInvalidIndex) || (profile.reference_emitter_index >= data.emitter_profiles.size());
    if (invalid_reference) {
      profile.reference_emitter_index = kInvalidIndex;
      continue;
    }

    const auto& referenced = data.emitter_profiles[profile.reference_emitter_index];
    if ((referenced.cls != EmitterProfile::Class::Environment) || ((referenced.meta & EmitterProfile::Meta::Atmosphere) == 0u)) {
      profile.reference_emitter_index = kInvalidIndex;
    }
  }
}

bool SceneRepresentationImpl::finalize_scene_loading(uint32_t options, const char* base_folder, uint32_t load_result, float camera_fov, bool use_focal_len, float camera_focal_len,
  bool force_tangents, bool spectral_scene, float pixel_filter_radius, bool preserve_unattached_cameras) {
  if (data.resolve_hierarchy() == false) {
    log::error("Failed to resolve scene hierarchy");
    return false;
  }

  auto& camera = active_camera;
  bool needs_camera_positioning = false;
  if (data.options.max_path_length > kMaximumPathLength) {
    log::warning("Scene max path length %u exceeds the supported limit; clamping to %u", data.options.max_path_length, kMaximumPathLength);
    data.options.max_path_length = kMaximumPathLength;
  }
  data.options.min_path_length = std::min(data.options.min_path_length, data.options.max_path_length);
  data.options.properties[Scene::Properties::Spectral] = spectral_scene;

  if (options & SceneRepresentation::SetupCamera) {
    if (data.cameras.empty()) {
      if ((load_result & SceneLoadCameraInfo) == 0) {
        if (use_focal_len) {
          camera_fov = focal_length_to_fov(camera_focal_len) * 180.0f / kPi;
        }
        if ((camera.film_size.x == 0u) || (camera.film_size.y == 0u)) {
          camera.film_size = kDefaultModelCameraFilmSize;
        }
        if (length(camera.direction) <= kEpsilon) {
          camera.direction = kWorldForward;
        }
        build_camera(camera, camera.position, camera.direction, camera.up, camera.film_size, camera_fov);
        needs_camera_positioning = true;
      }
    } else {
      const auto& selected = select_active_camera(data);
      camera = selected.cam;
      const uint32_t camera_index = static_cast<uint32_t>(&selected - data.cameras.data());
      AttachmentTransform transform = {};
      if (find_attachment_transform(data, SceneAttachment::Type::Camera, camera_index, transform)) {
        camera = transform_camera(selected.cam, transform.object_to_world, transform.orientation_to_world);
      }
    }
  }

  bool has_emissive_materials = false;
  for (const auto& material : data.materials) {
    if ((material.emission.spectrum_index != kInvalidIndex) && (material.emission.spectrum_index < data.spectrum_values.size()) &&
        (data.spectrum_values[material.emission.spectrum_index].luminance() > 0.0f)) {
      has_emissive_materials = true;
      break;
    }
  }

  validate_materials();
  validate_mediums();

  generate_pixel_sampler_image(pixel_filter_radius);

  if (ensure_energy_compensation_interfaces() == false) {
    return false;
  }

  data.images.load_images(scheduler);

  {
    TimeMeasure m = {};
    log::warning("Validating normals and tangents...");
    bool has_invalid_tangents = false;
    std::vector<bool> referenced_vertices;
    validate_normals(referenced_vertices, has_invalid_tangents);
    log::warning("Normals validated: %.2f sec", m.lap());

    if (has_invalid_tangents || force_tangents) {
      build_tangents();
      log::warning("Tangents built: %.2f sec", m.lap());
    } else {
      log::warning("Tangents are valid, skipping rebuild");
    }

    validate_tangents(referenced_vertices, has_invalid_tangents || force_tangents);
    log::warning("Tangents validated: %.2f sec", m.lap());
  }

  setup_atmosphere_references();

  // Create area emitters from materials with emission
  create_area_emitters_from_materials();

  // Rebuild atmospheres now that references are set up
  for (uint32_t i = 0; i < data.emitter_profiles.size(); ++i) {
    const auto& profile = data.emitter_profiles[i];
    if ((profile.cls == EmitterProfile::Class::Environment) && (profile.meta & EmitterProfile::Meta::Atmosphere) != 0) {
      rebuild_atmosphere_emitter(i);
    }
  }

  if (update_medium_bounds() == false) {
    return false;
  }

  if (needs_camera_positioning) {
    constexpr float3 kDefaultViewDirection = {1.0f, 1.0f, 1.0f};
    float3 position = {};
    float3 target = {};
    compute_camera_position_to_fit_scene(data, camera, kDefaultViewDirection, position, target);
    const float3 direction = normalize(target - position);
    build_camera(camera, position, direction, kWorldUp, camera.film_size, camera_fov);
  }

  if (data.cameras.empty()) {
    auto& entry = data.cameras.emplace_back();
    entry.id = "camera";
    entry.active = true;
    entry.cam = camera;
  }

  SceneData::CameraInfo& selected_camera = select_active_camera(data);
  if ((ensure_camera_nodes(data, preserve_unattached_cameras) == false) || (data.resolve_hierarchy() == false)) {
    log::error("Failed to attach cameras to the scene hierarchy");
    return false;
  }

  camera = selected_camera.cam;
  const uint32_t selected_camera_index = static_cast<uint32_t>(&selected_camera - data.cameras.data());
  AttachmentTransform camera_transform = {};
  if (find_attachment_transform(data, SceneAttachment::Type::Camera, selected_camera_index, camera_transform)) {
    camera = transform_camera(selected_camera.cam, camera_transform.object_to_world, camera_transform.orientation_to_world);
  }

  scene_valid = true;
  return true;
}

void SceneRepresentationImpl::create_area_emitters_from_materials() {
  ensure_emitter_names(data);
  std::vector<uint32_t> old_to_new(data.emitter_profiles.size(), kInvalidIndex);
  std::vector<EmitterProfile> non_area_profiles;
  std::vector<std::string> non_area_names;
  non_area_profiles.reserve(data.emitter_profiles.size());
  non_area_names.reserve(data.emitter_profiles.size());
  for (uint32_t old_index = 0u; old_index < data.emitter_profiles.size(); ++old_index) {
    const EmitterProfile& profile = data.emitter_profiles[old_index];
    if (profile.cls == EmitterProfile::Class::Area) {
      continue;
    }
    old_to_new[old_index] = static_cast<uint32_t>(non_area_profiles.size());
    non_area_profiles.emplace_back(profile);
    non_area_names.emplace_back(data.emitter_names[old_index]);
  }
  remap_emitter_attachments(data.hierarchy, old_to_new);
  remap_emitter_references(non_area_profiles, old_to_new);
  data.emitter_profiles = std::move(non_area_profiles);
  data.emitter_names = std::move(non_area_names);

  // Clear triangle emitter references (now point to profiles, not instances)
  for (Triangle& tri : data.triangles) {
    tri.emitter_index = kInvalidIndex;
  }

  // Create area emitter profiles for emissive materials
  std::unordered_map<uint32_t, uint32_t> material_to_profile;

  for (size_t tri_index = 0; tri_index < data.triangles.size(); ++tri_index) {
    const Triangle& tri = data.triangles[tri_index];
    if (tri.material_index >= data.materials.size())
      continue;

    const Material& mtl = data.materials[tri.material_index];
    if (mtl.emission.spectrum_index == kInvalidIndex)
      continue;

    float spectrum_weight = data.spectrum_values[mtl.emission.spectrum_index].luminance();
    if (spectrum_weight <= kEpsilon)
      continue;

    // Get or create emitter profile for this material
    uint32_t profile_index = kInvalidIndex;
    auto mat_it = material_to_profile.find(tri.material_index);
    if (mat_it != material_to_profile.end()) {
      profile_index = mat_it->second;
    } else {
      profile_index = static_cast<uint32_t>(data.emitter_profiles.size());
      material_to_profile[tri.material_index] = profile_index;

      EmitterProfile& profile = data.emitter_profiles.emplace_back(EmitterProfile::Class::Area);
      profile.emission = mtl.emission;
      profile.medium_index = kInvalidIndex;
      const std::string material_name = mapping_name(data.material_mapping, tri.material_index, "material-");
      const std::string area_name = material_name + " Light";
      data.emitter_names.push_back(unique_named_resource(data.emitter_names, area_name.c_str(), "Area Light", kInvalidIndex));
    }

    // Mark triangle as referencing this emitter profile
    data.triangles[tri_index].emitter_index = profile_index;
  }
}

}  // namespace etx
