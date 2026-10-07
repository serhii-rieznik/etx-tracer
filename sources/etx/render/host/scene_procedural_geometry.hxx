#pragma once

#include <etx/render/interop/interop_base.hxx>
#include <etx/render/shared/base.hxx>
#include <etx/render/host/scene_loader_utils.hxx>
#include <array>

namespace etx {

struct SceneData;

struct BilinearPatch {
  static constexpr uint32_t DefaultSubdivisions = 16u;
  static constexpr uint32_t MaximumSubdivisions = 256u;
  // Parameter order: (0,0), (1,0), (0,1), (1,1).
  std::array<float3, 4> positions = {};
  std::array<float3, 4> normals = {};
  std::array<float2, 4> texcoords = {float2{0.0f, 0.0f}, float2{1.0f, 0.0f}, float2{0.0f, 1.0f}, float2{1.0f, 1.0f}};
};

bool tessellate_bilinear_patch(const BilinearPatch& patch, uint32_t subdivisions, std::vector<Vertex>& vertices, std::vector<uint3>& triangles);

struct ProceduralGeometryDefinition {
  enum class Class : uint32_t {
    Invalid,
    Sphere,
    Plane,
    Disk,
    Box,
    Cone,
    Capsule,
    Torus,
    Tetrahedron,
    Octahedron,
    Dodecahedron,
    Icosahedron,
    Bilinear,
  };

  Class cls = Class::Invalid;
  std::string id;
  std::string material_name;
  uint32_t material_index = kInvalidIndex;
  float3 center = {};
  float3 dimensions = {};
  float3 normal = {0.0f, 1.0f, 0.0f};
  float radius = 0.5f;
  float inner_radius = 0.0f;
  float thickness = 0.0f;
  float bevel_radius = 0.0f;
  uint32_t subdivisions = 4u;
  uint32_t segments = 128u;
  uint32_t bevel_segments = 0u;
  BilinearPatch bilinear = {};
};

bool is_procedural_geometry_entry(const std::string& name);
bool parse_procedural_geometry_definition(const MaterialDefinition& material, ProceduralGeometryDefinition& out_definition);
uint32_t generate_procedural_geometry(SceneData& data, const std::vector<ProceduralGeometryDefinition>& definitions);

}  // namespace etx
