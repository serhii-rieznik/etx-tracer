#pragma once

#include <etx/render/host/scene_data.hxx>
#include <filesystem>

namespace etx {

struct ImportedMesh {
  std::vector<float3> positions;
  std::vector<float3> normals;
  std::vector<float2> texcoords;
  std::vector<uint3> indices;
  bool flat_shading = false;
  bool omitted_colors = false;
  bool omitted_textures = false;
};

ImportedMesh load_ply_mesh(const std::filesystem::path& path);
ImportedMesh load_stl_mesh(const std::filesystem::path& path);
uint32_t load_mesh_document(ImportedMesh mesh, const std::filesystem::path& path, SceneData& data);

}  // namespace etx
