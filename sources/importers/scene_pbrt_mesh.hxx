#pragma once

#include <etx/render/interop/interop_base.hxx>
#include <etx/render/shared/base.hxx>
#include <filesystem>
#include <vector>

namespace etx {

struct PbrtMesh {
  std::vector<float3> positions;
  std::vector<float3> normals;
  std::vector<float2> texcoords;
  std::vector<uint3> indices;
};

PbrtMesh load_pbrt_ply(const std::filesystem::path& path);
void discard_degenerate_pbrt_triangles(PbrtMesh& mesh);
PbrtMesh subdivide_pbrt_loop(PbrtMesh mesh, uint32_t levels);

}  // namespace etx
