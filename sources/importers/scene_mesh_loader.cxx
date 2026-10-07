#include <etx/std.hxx>
#include "scene_mesh_loader.hxx"
#include <etx/core/environment.hxx>
#include <etx/core/log.hxx>
#include <etx/render/host/scene_loader_utils.hxx>
#include <limits>
#include <stdexcept>

namespace etx {
namespace {

float3 unit_normal(const float3& value) {
  const double length = std::sqrt(double(value.x) * value.x + double(value.y) * value.y + double(value.z) * value.z);
  if (std::isfinite(length) == false)
    throw std::runtime_error("Mesh normal exceeds the native floating-point range.");
  if (length == 0.0)
    return {};
  return {static_cast<float>(value.x / length), static_cast<float>(value.y / length), static_cast<float>(value.z / length)};
}

}  // namespace

uint32_t load_mesh_document(ImportedMesh mesh, const std::filesystem::path& path, SceneData& data) {
  if ((mesh.positions.empty()) || (mesh.indices.empty()))
    throw std::runtime_error("Mesh contains no triangle geometry.");
  if ((mesh.positions.size() > std::numeric_limits<uint32_t>::max()) || (mesh.indices.size() > std::numeric_limits<uint32_t>::max()))
    throw std::runtime_error("Mesh exceeds native geometry index capacity.");
  const size_t vertex_count = mesh.positions.size();
  data.vertices.pos = std::move(mesh.positions);
  data.vertices.nrm = std::move(mesh.normals);
  data.vertices.nrm.resize(vertex_count);
  data.vertices.tex = std::move(mesh.texcoords);
  data.vertices.tex.resize(vertex_count);
  data.vertices.tan.resize(vertex_count);
  data.vertices.btn.resize(vertex_count);
  std::vector<bool> reconstruct(mesh.flat_shading ? 0u : vertex_count);
  for (size_t index = 0u; index < vertex_count; ++index) {
    if (mesh.flat_shading == false) {
      auto& normal = data.vertices.nrm[index];
      normal = unit_normal(normal);
      reconstruct[index] = dot(normal, normal) == 0.0f;
    }
    data.vertices.tex[index].y = 1.0f - data.vertices.tex[index].y;
  }
  const uint32_t material = data.add_material("Imported mesh");
  data.materials[material].cls = MaterialClass::Diffuse;
  data.materials[material].scattering = {.spectrum_index = data.add_spectrum(SpectralDistribution::rgb_reflectance({0.7f, 0.7f, 0.7f}))};
  data.materials[material].reflectance = {.spectrum_index = data.defaults.white_spectrum};
  float3 lower = {kMaxFloat, kMaxFloat, kMaxFloat}, upper = {-kMaxFloat, -kMaxFloat, -kMaxFloat};
  data.triangles.reserve(mesh.indices.size());
  for (const auto& indices : mesh.indices) {
    const float3& a = data.vertices.pos[indices.x];
    const float3& b = data.vertices.pos[indices.y];
    const float3& c = data.vertices.pos[indices.z];
    const float3 geometric = cross(b - a, c - a);
    const double area_squared = double(geometric.x) * geometric.x + double(geometric.y) * geometric.y + double(geometric.z) * geometric.z;
    if ((std::isfinite(area_squared) == false) || (area_squared > std::numeric_limits<float>::max()))
      throw std::runtime_error("Mesh triangle exceeds native geometric range.");
    if (area_squared == 0.0)
      continue;
    if (dot(geometric, geometric) == 0.0f)
      throw std::runtime_error("Mesh triangle is too small for native geometric precision.");
    Triangle triangle = {};
    triangle.i[0] = indices.x;
    triangle.i[1] = indices.y;
    triangle.i[2] = indices.z;
    triangle.material_index = material;
    triangle.geo_n = unit_normal(geometric);
    data.triangles.push_back(triangle);
    for (uint32_t index : triangle.i) {
      lower = min(lower, data.vertices.pos[index]);
      upper = max(upper, data.vertices.pos[index]);
      if (mesh.flat_shading)
        data.vertices.nrm[index] = triangle.geo_n;
      else if (reconstruct[index])
        data.vertices.nrm[index] += geometric;
    }
  }
  if (data.triangles.empty())
    throw std::runtime_error("Mesh has no non-degenerate triangles.");
  for (size_t index = 0u; index < reconstruct.size(); ++index) {
    if (reconstruct[index]) {
      data.vertices.nrm[index] = unit_normal(data.vertices.nrm[index]);
    }
  }
  for (const auto& triangle : data.triangles) {
    for (uint32_t index : triangle.i) {
      if (dot(data.vertices.nrm[index], data.vertices.nrm[index]) == 0.0f)
        data.vertices.nrm[index] = triangle.geo_n;
    }
  }
  const std::string name = path_to_utf8(path.stem());
  data.add_mesh(name.c_str(), 0u, static_cast<uint32_t>(data.triangles.size()), lower, upper);
  if (mesh.omitted_colors)
    log::warning("Mesh colors were not imported: native ETX geometry has no vertex color channel (%s).", path_to_utf8(path).c_str());
  if (mesh.omitted_textures)
    log::warning("PLY texture-file metadata is not supported; texture coordinates were retained (%s).", path_to_utf8(path).c_str());
  if (data.triangles.size() != mesh.indices.size())
    log::warning("Skipped %zu degenerate mesh triangles (%s).", mesh.indices.size() - data.triangles.size(), path_to_utf8(path).c_str());
  return SceneLoadSucceeded;
}

}  // namespace etx
