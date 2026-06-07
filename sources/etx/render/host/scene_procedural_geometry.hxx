#pragma once

#include <etx/render/host/scene_loader_utils.hxx>

namespace etx {

struct SceneData;

struct ProceduralGeometryDefinition {
  enum class Class : uint32_t {
    Invalid,
    Sphere,
    Plane,
  };

  Class cls = Class::Invalid;
  std::string id;
  std::string material_name;
  uint32_t material_index = kInvalidIndex;
  float3 center = {};
  float3 dimensions = {};
  uint32_t subdivisions = 4u;
};

bool is_procedural_geometry_entry(const std::string& name);
bool parse_procedural_geometry_definition(const MaterialDefinition& material, ProceduralGeometryDefinition& out_definition);
uint32_t generate_procedural_geometry(SceneData& data, const std::vector<ProceduralGeometryDefinition>& definitions);

}  // namespace etx
