#pragma once

#include <etx/render/host/scene_data.hxx>
#include <etx/render/host/scene_loader_utils.hxx>
#include <etx/render/shared/math.hxx>

#include <string>
#include <map>
#include <unordered_map>
#include <vector>
#include <filesystem>

namespace etx {

inline std::string normalize_material_name(std::string name) {
  trim_whitespace(name);
  std::transform(name.begin(), name.end(), name.begin(), ::tolower);
  return name;
}


struct Scene;
struct IORDatabase;
struct TaskScheduler;

static constexpr uint32_t kBinaryGeometryMagic = ('B' << 24) | ('x' << 16) | ('t' << 8) | ('E' << 0);
static constexpr uint32_t kBinaryGeometryVersion = 1u;

#pragma pack(push, 1)
struct MappingEntry {
  uint32_t string_index;
  uint32_t target_index;
};

struct MaterialIndexMapping {
  uint32_t saved_index;
  uint32_t name_index;
};
#pragma pack(pop)

struct SceneSerialization {
  SceneSerialization();
  ~SceneSerialization();

  bool save_to_file(const SceneData& data, const std::filesystem::path& path);
  bool load_from_file(const std::filesystem::path& path, SceneData& data, const char* materials_file, const IORDatabase& database, TaskScheduler& scheduler);
  void parse_material_definitions(const char* base_dir, const std::vector<MaterialDefinition>& materials, SceneData& data, const IORDatabase& database, TaskScheduler& scheduler);
  bool parse_materials_file(const std::filesystem::path& path, const char* base_dir, SceneData& data, const IORDatabase& database, TaskScheduler& scheduler);

 private:
  ETX_DECLARE_PIMPL(SceneSerialization, 5120);
};

}  // namespace etx
