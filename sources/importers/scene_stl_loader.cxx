#include <etx/std.hxx>
#include "scene_mesh_loader.hxx"
#include "scene_mesh_input.hxx"

namespace etx {
namespace {

void expect(MeshInput& input, std::string_view word) {
  if (input.token() != word)
    throw std::runtime_error("Invalid ASCII STL: expected " + std::string(word) + ".");
}

float3 vector(MeshInput& input) {
  const float x = mesh_float(input.number());
  const float y = mesh_float(input.number());
  const float z = mesh_float(input.number());
  return {x, y, z};
}

void append_facet(ImportedMesh& mesh, const float3 (&positions)[3]) {
  if (mesh.positions.size() > (std::numeric_limits<uint32_t>::max() - 3u))
    throw std::runtime_error("STL exceeds native geometry index capacity.");
  const uint32_t begin = static_cast<uint32_t>(mesh.positions.size());
  mesh.positions.insert(mesh.positions.end(), std::begin(positions), std::end(positions));
  mesh.indices.push_back({begin, begin + 1u, begin + 2u});
}

}  // namespace

ImportedMesh load_stl_mesh(const std::filesystem::path& path) {
  MeshInput input(path);
  bool binary = false;
  uint32_t count = 0u;
  if (input.size() >= 84u) {
    input.read(nullptr, 80u);
    count = input.binary<uint32_t>(false);
    binary = input.size() == (84ull + 50ull * count);
  }
  ImportedMesh mesh;
  mesh.flat_shading = true;
  if (binary) {
    if (count > (std::numeric_limits<uint32_t>::max() / 3u))
      throw std::runtime_error("STL exceeds native geometry index capacity.");
    mesh.positions.reserve(size_t(count) * 3u);
    mesh.indices.reserve(count);
    for (uint32_t index = 0u; index < count; ++index) {
      for (uint32_t component = 0u; component < 3u; ++component)
        mesh_float(input.binary<float>(false));
      float3 positions[3];
      for (auto& p : positions)
        p = {mesh_float(input.binary<float>(false)), mesh_float(input.binary<float>(false)), mesh_float(input.binary<float>(false))};
      if (input.binary<uint16_t>(false) != 0u)
        mesh.omitted_colors = true;
      append_facet(mesh, positions);
    }
  } else {
    MeshInput text(path);
    for (auto word = text.token(); word.empty() == false; word = text.token()) {
      if (word != "solid")
        throw std::runtime_error("STL has neither a valid binary length nor an ASCII solid declaration.");
      text.line();
      for (;;) {
        const auto declaration = text.token();
        if (declaration == "endsolid") {
          text.line();
          break;
        }
        if (declaration != "facet")
          throw std::runtime_error("Invalid ASCII STL: expected facet or endsolid.");
        expect(text, "normal");
        vector(text);
        expect(text, "outer");
        expect(text, "loop");
        float3 positions[3];
        for (auto& p : positions) {
          expect(text, "vertex");
          p = vector(text);
        }
        expect(text, "endloop");
        expect(text, "endfacet");
        append_facet(mesh, positions);
      }
    }
  }
  if (mesh.indices.empty())
    throw std::runtime_error("STL contains no facets.");
  return mesh;
}

}  // namespace etx
