#include <etx/std.hxx>
#include "scene_pbrt_mesh.hxx"
#include <etx/render/shared/base.hxx>
#include <unordered_map>
#include <limits>
#include <stdexcept>

namespace etx {
namespace {

struct LoopEdge {
  uint32_t a, b, opposite;
  uint32_t other = kInvalidIndex;
};

struct LoopVertex {
  struct Link {
    uint32_t from, to;
  };
  std::vector<Link> links;
  std::vector<uint32_t> ring;
  bool boundary = false;
};

struct LoopTopology {
  explicit LoopTopology(const PbrtMesh& mesh)
    : vertices(mesh.positions.size()) {
    edges.reserve(mesh.indices.size() * 3u / 2u);
    edge_indices.reserve(mesh.indices.size() * 3u / 2u);
    for (const auto& triangle : mesh.indices) {
      const uint32_t index[] = {triangle.x, triangle.y, triangle.z};
      for (uint32_t corner = 0u; corner < 3u; ++corner) {
        const uint32_t a = index[corner], b = index[(corner + 1u) % 3u], c = index[(corner + 2u) % 3u];
        if ((a >= vertices.size()) || (a == b) || (a == c))
          throw std::runtime_error("Invalid Loop subdivision control triangle.");
        vertices[a].links.push_back({b, c});
        const auto [entry, inserted] = edge_indices.emplace(key(a, b), static_cast<uint32_t>(edges.size()));
        if (inserted)
          edges.push_back({a, b, c});
        else {
          auto& edge = edges[entry->second];
          if ((edge.other != kInvalidIndex) || (edge.a != b) || (edge.b != a))
            throw std::runtime_error("Loop subdivision requires consistently oriented manifold edges.");
          edge.other = c;
        }
      }
    }
    for (auto& vertex : vertices) {
      if (vertex.links.empty())
        throw std::runtime_error("Loop subdivision control mesh contains an unreferenced vertex.");
      for (const auto& link : vertex.links) {
        if (std::count_if(vertex.links.begin(), vertex.links.end(), [&](const auto& item) {
              return item.from == link.from;
            }) != 1)
          throw std::runtime_error("Non-manifold Loop subdivision vertex.");
      }
      uint32_t start = vertex.links.front().from;
      size_t boundary_ends = 0u;
      for (const auto& link : vertex.links) {
        if (std::none_of(vertex.links.begin(), vertex.links.end(), [&](const auto& item) {
              return item.to == link.from;
            })) {
          start = link.from;
          ++boundary_ends;
        }
      }
      if (boundary_ends > 1u)
        throw std::runtime_error("Disconnected Loop subdivision vertex fan.");
      vertex.boundary = boundary_ends == 1u;
      vertex.ring.reserve(vertex.links.size() + (vertex.boundary ? 1u : 0u));
      uint32_t current = start;
      do {
        if (vertex.ring.size() > vertex.links.size())
          throw std::runtime_error("Invalid Loop subdivision vertex fan.");
        vertex.ring.push_back(current);
        const auto link = std::find_if(vertex.links.begin(), vertex.links.end(), [&](const auto& item) {
          return item.from == current;
        });
        if (link == vertex.links.end())
          break;
        current = link->to;
      } while (current != start);
      const size_t expected = vertex.links.size() + (vertex.boundary ? 1u : 0u);
      if ((vertex.ring.size() != expected) || ((vertex.boundary == false) && (current != start)))
        throw std::runtime_error("Disconnected Loop subdivision vertex fan.");
    }
  }

  static uint64_t key(uint32_t a, uint32_t b) {
    return (uint64_t(std::min(a, b)) << 32u) | std::max(a, b);
  }

  std::vector<LoopVertex> vertices;
  std::vector<LoopEdge> edges;
  std::unordered_map<uint64_t, uint32_t> edge_indices;
};

float loop_beta(size_t valence) {
  return valence == 3u ? 3.0f / 16.0f : 3.0f / (8.0f * static_cast<float>(valence));
}

float3 loop_weight(const PbrtMesh& mesh, uint32_t index, const LoopVertex& vertex, float beta) {
  if (vertex.boundary)
    return (1.0f - 2.0f * beta) * mesh.positions[index] + beta * mesh.positions[vertex.ring.front()] + beta * mesh.positions[vertex.ring.back()];
  float3 result = (1.0f - static_cast<float>(vertex.ring.size()) * beta) * mesh.positions[index];
  for (const uint32_t neighbor : vertex.ring)
    result += beta * mesh.positions[neighbor];
  return result;
}

}  // namespace

PbrtMesh subdivide_pbrt_loop(PbrtMesh mesh, uint32_t levels) {
  discard_degenerate_pbrt_triangles(mesh);
  if (mesh.indices.empty())
    return mesh;
  for (uint32_t level = 0u; level < levels; ++level) {
    const LoopTopology topology(mesh);
    const size_t count = mesh.positions.size() + topology.edges.size();
    if ((count >= kInvalidIndex) || (mesh.indices.size() > (kInvalidIndex / 4u)))
      throw std::runtime_error("Loop subdivision exceeds the native mesh index range.");
    PbrtMesh refined;
    refined.positions.reserve(count);
    refined.indices.reserve(mesh.indices.size() * 4u);
    for (uint32_t index = 0u; index < mesh.positions.size(); ++index) {
      const auto& vertex = topology.vertices[index];
      refined.positions.push_back(loop_weight(mesh, index, vertex, vertex.boundary ? 1.0f / 8.0f : loop_beta(vertex.ring.size())));
    }
    const uint32_t edge_start = static_cast<uint32_t>(mesh.positions.size());
    for (const auto& edge : topology.edges) {
      const float3 endpoints = mesh.positions[edge.a] + mesh.positions[edge.b];
      refined.positions.push_back(
        edge.other == kInvalidIndex ? endpoints * 0.5f : endpoints * (3.0f / 8.0f) + (mesh.positions[edge.opposite] + mesh.positions[edge.other]) * (1.0f / 8.0f));
    }
    for (const auto& triangle : mesh.indices) {
      const uint32_t a = triangle.x, b = triangle.y, c = triangle.z;
      const uint32_t ab = edge_start + topology.edge_indices.at(LoopTopology::key(a, b));
      const uint32_t bc = edge_start + topology.edge_indices.at(LoopTopology::key(b, c));
      const uint32_t ca = edge_start + topology.edge_indices.at(LoopTopology::key(c, a));
      refined.indices.push_back({a, ab, ca});
      refined.indices.push_back({ab, b, bc});
      refined.indices.push_back({ca, bc, c});
      refined.indices.push_back({ab, bc, ca});
    }
    mesh = std::move(refined);
  }
  const LoopTopology topology(mesh);
  std::vector<float3> limit;
  limit.reserve(mesh.positions.size());
  for (uint32_t index = 0u; index < mesh.positions.size(); ++index) {
    const auto& vertex = topology.vertices[index];
    const float gamma = 1.0f / (static_cast<float>(vertex.ring.size()) + 3.0f / (8.0f * loop_beta(vertex.ring.size())));
    limit.push_back(loop_weight(mesh, index, vertex, vertex.boundary ? 1.0f / 5.0f : gamma));
  }
  mesh.positions = std::move(limit);
  mesh.normals.resize(mesh.positions.size());
  mesh.texcoords.clear();
  for (uint32_t index = 0u; index < mesh.positions.size(); ++index) {
    const auto& vertex = topology.vertices[index];
    const size_t count = vertex.ring.size();
    const auto point = [&](size_t ring_index) {
      return mesh.positions[vertex.ring[ring_index]];
    };
    float3 tangent = {}, bitangent = {};
    if (vertex.boundary == false) {
      for (size_t neighbor = 0u; neighbor < count; ++neighbor) {
        const float angle = kDoublePi * static_cast<float>(neighbor) / static_cast<float>(count);
        tangent += std::cos(angle) * point(neighbor);
        bitangent += std::sin(angle) * point(neighbor);
      }
    } else {
      tangent = point(count - 1u) - point(0u);
      if (count == 2u)
        bitangent = point(0u) + point(1u) - 2.0f * mesh.positions[index];
      else if (count == 3u)
        bitangent = point(1u) - mesh.positions[index];
      else if (count == 4u)
        bitangent = -point(0u) + 2.0f * point(1u) + 2.0f * point(2u) - point(3u) - 2.0f * mesh.positions[index];
      else {
        const float theta = kPi / static_cast<float>(count - 1u);
        bitangent = std::sin(theta) * (point(0u) + point(count - 1u));
        for (size_t neighbor = 1u; (neighbor + 1u) < count; ++neighbor)
          bitangent += (2.0f * std::cos(theta) - 2.0f) * std::sin(static_cast<float>(neighbor) * theta) * point(neighbor);
        bitangent = -bitangent;
      }
    }
    const float3 normal = cross(tangent, bitangent);
    const auto& link = vertex.links.front();
    const float3 geometric = cross(mesh.positions[link.from] - mesh.positions[index], mesh.positions[link.to] - mesh.positions[index]);
    const float normal_length_squared = dot(normal, normal);
    if (std::isfinite(normal_length_squared) == false)
      throw std::runtime_error("Loop subdivision normal exceeds the native floating-point range.");
    mesh.normals[index] = (normal_length_squared > 0.0f) ? normalize(dot(normal, geometric) < 0.0f ? -normal : normal) : float3{};
  }
  return mesh;
}

}  // namespace etx
