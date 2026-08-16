#include <etx/render/host/scene_procedural_geometry.hxx>

#include <etx/core/log.hxx>
#include <etx/render/host/scene_data.hxx>
#include <etx/render/host/scene_serialization.hxx>
#include <etx/render/shared/scene.hxx>

namespace etx {
namespace {

static constexpr uint32_t kMaxSphereSubdivisions = 6u;
static constexpr uint32_t kMinimumDiskSegments = 3u;
static constexpr uint32_t kMaximumDiskSegments = 4096u;
static constexpr uint32_t kMaximumDiskBevelSegments = 32u;

struct IcosphereFace {
  uint32_t i[3] = {};
};

bool read_property(const MaterialDefinition& material, const char* name, std::string& out_value) {
  auto it = material.properties.find(name);
  if (it == material.properties.end()) {
    return false;
  }

  out_value = it->second;
  trim_whitespace(out_value);
  return out_value.empty() == false;
}

bool read_float3_property(const MaterialDefinition& material, const char* name, float3& out_value) {
  std::string value;
  if (read_property(material, name, value) == false) {
    return false;
  }

  float parsed[3] = {};
  if (sscanf(value.c_str(), "%f %f %f", parsed + 0, parsed + 1, parsed + 2) != 3) {
    log::warning("Procedural geometry `%s` has invalid `%s` value: %s", material.name.c_str(), name, value.c_str());
    return false;
  }

  out_value = {parsed[0], parsed[1], parsed[2]};
  return true;
}

bool read_float2_property(const MaterialDefinition& material, const char* name, float2& out_value) {
  std::string value;
  if (read_property(material, name, value) == false) {
    return false;
  }

  float parsed[2] = {};
  if (sscanf(value.c_str(), "%f %f", parsed + 0, parsed + 1) != 2) {
    log::warning("Procedural geometry `%s` has invalid `%s` value: %s", material.name.c_str(), name, value.c_str());
    return false;
  }

  out_value = {parsed[0], parsed[1]};
  return true;
}

bool read_float_property(const MaterialDefinition& material, const char* name, float& out_value) {
  std::string value;
  if (read_property(material, name, value) == false) {
    return false;
  }

  float parsed = 0.0f;
  if (sscanf(value.c_str(), "%f", &parsed) != 1) {
    log::warning("Procedural geometry `%s` has invalid `%s` value: %s", material.name.c_str(), name, value.c_str());
    return false;
  }

  out_value = parsed;
  return true;
}

bool read_uint32_property(const MaterialDefinition& material, const char* name, uint32_t& out_value) {
  std::string value;
  if (read_property(material, name, value) == false) {
    return false;
  }

  uint32_t parsed = 0u;
  if (sscanf(value.c_str(), "%u", &parsed) != 1) {
    log::warning("Procedural geometry `%s` has invalid `%s` value: %s", material.name.c_str(), name, value.c_str());
    return false;
  }

  out_value = parsed;
  return true;
}

bool is_sphere_alias(const std::string& name) {
  return (name == "et::sphere") || (name == "etx::sphere");
}

bool is_plane_alias(const std::string& name) {
  return (name == "et::plane") || (name == "etx::plane");
}

bool is_disk_alias(const std::string& name) {
  return (name == "et::disk") || (name == "etx::disk");
}

bool is_geometry_entry(const std::string& name) {
  return (name == "et::geometry") || (name == "etx::geometry");
}

uint64_t edge_key(uint32_t a, uint32_t b) {
  const uint32_t lo = min(a, b);
  const uint32_t hi = max(a, b);
  return (static_cast<uint64_t>(lo) << 32u) | static_cast<uint64_t>(hi);
}

uint32_t midpoint_index(uint32_t a, uint32_t b, std::vector<float3>& vertices, std::unordered_map<uint64_t, uint32_t>& midpoint_cache) {
  const uint64_t key = edge_key(a, b);
  auto it = midpoint_cache.find(key);
  if (it != midpoint_cache.end()) {
    return it->second;
  }

  const float3 p = normalize(vertices[a] + vertices[b]);
  const uint32_t index = static_cast<uint32_t>(vertices.size());
  vertices.emplace_back(p);
  midpoint_cache[key] = index;
  return index;
}

void build_icosahedron(std::vector<float3>& vertices, std::vector<IcosphereFace>& faces) {
  const float t = (1.0f + sqrtf(5.0f)) * 0.5f;

  const float3 base_vertices[12] = {
    {-1.0f, +t, 0.0f},
    {+1.0f, +t, 0.0f},
    {-1.0f, -t, 0.0f},
    {+1.0f, -t, 0.0f},
    {0.0f, -1.0f, +t},
    {0.0f, +1.0f, +t},
    {0.0f, -1.0f, -t},
    {0.0f, +1.0f, -t},
    {+t, 0.0f, -1.0f},
    {+t, 0.0f, +1.0f},
    {-t, 0.0f, -1.0f},
    {-t, 0.0f, +1.0f},
  };

  vertices.reserve(12u);
  for (uint32_t i = 0u; i < 12u; ++i) {
    vertices.emplace_back(normalize(base_vertices[i]));
  }

  const IcosphereFace base_faces[20] = {
    {{0u, 11u, 5u}},
    {{0u, 5u, 1u}},
    {{0u, 1u, 7u}},
    {{0u, 7u, 10u}},
    {{0u, 10u, 11u}},
    {{1u, 5u, 9u}},
    {{5u, 11u, 4u}},
    {{11u, 10u, 2u}},
    {{10u, 7u, 6u}},
    {{7u, 1u, 8u}},
    {{3u, 9u, 4u}},
    {{3u, 4u, 2u}},
    {{3u, 2u, 6u}},
    {{3u, 6u, 8u}},
    {{3u, 8u, 9u}},
    {{4u, 9u, 5u}},
    {{2u, 4u, 11u}},
    {{6u, 2u, 10u}},
    {{8u, 6u, 7u}},
    {{9u, 8u, 1u}},
  };

  faces.assign(base_faces, base_faces + 20u);
}

void subdivide_icosphere(std::vector<float3>& vertices, std::vector<IcosphereFace>& faces, uint32_t subdivisions) {
  for (uint32_t level = 0u; level < subdivisions; ++level) {
    std::unordered_map<uint64_t, uint32_t> midpoint_cache;
    midpoint_cache.reserve(faces.size() * 3u);

    std::vector<IcosphereFace> subdivided_faces;
    subdivided_faces.reserve(faces.size() * 4u);

    for (const IcosphereFace& face : faces) {
      const uint32_t a = midpoint_index(face.i[0], face.i[1], vertices, midpoint_cache);
      const uint32_t b = midpoint_index(face.i[1], face.i[2], vertices, midpoint_cache);
      const uint32_t c = midpoint_index(face.i[2], face.i[0], vertices, midpoint_cache);

      subdivided_faces.push_back({{face.i[0], a, c}});
      subdivided_faces.push_back({{face.i[1], b, a}});
      subdivided_faces.push_back({{face.i[2], c, b}});
      subdivided_faces.push_back({{a, b, c}});
    }

    faces = std::move(subdivided_faces);
  }
}

float2 sphere_uv(const float3& nrm) {
  const float u = atan2f(nrm.z, nrm.x) / kDoublePi + 0.5f;
  const float v = 0.5f - asinf(clamp(nrm.y, -1.0f, 1.0f)) / kPi;
  return {u, v};
}

void sphere_tangent_frame(const float3& nrm, float3& out_tangent, float3& out_bitangent) {
  const float horizontal_length = sqrtf((nrm.x * nrm.x) + (nrm.z * nrm.z));
  if (horizontal_length > kEpsilon) {
    out_tangent = {-nrm.z / horizontal_length, 0.0f, nrm.x / horizontal_length};
  } else {
    out_tangent = {1.0f, 0.0f, 0.0f};
  }
  out_bitangent = normalize(cross(nrm, out_tangent));
}

std::string unique_mesh_name(const SceneData& data, const std::string& requested_name, const char* fallback_name) {
  std::string base_name = requested_name.empty() ? fallback_name : requested_name;
  if (base_name.empty()) {
    base_name = "procedural";
  }

  std::string candidate = base_name;
  uint32_t suffix = 1u;
  while (data.mesh_mapping.count(candidate) > 0u) {
    candidate = base_name + "-" + std::to_string(suffix);
    ++suffix;
  }
  return candidate;
}

uint32_t resolve_material_index(SceneData& data, const ProceduralGeometryDefinition& definition) {
  if (definition.material_index != kInvalidIndex) {
    if (definition.material_index < data.materials.size()) {
      return definition.material_index;
    }
    log::warning("Procedural geometry `%s` references invalid material-index %u", definition.id.c_str(), definition.material_index);
  }

  if (definition.material_name.empty() == false) {
    const std::string normalized_name = normalize_material_name(definition.material_name);
    auto it = data.material_mapping.find(normalized_name);
    if (it != data.material_mapping.end()) {
      return it->second;
    }
    log::warning("Procedural geometry `%s` references unknown material `%s`", definition.id.c_str(), definition.material_name.c_str());
  }

  if (data.defaults.missing_material < data.materials.size()) {
    return data.defaults.missing_material;
  }

  return kInvalidIndex;
}

float sphere_radius(const ProceduralGeometryDefinition& definition) {
  return definition.dimensions.x * 0.5f;
}

bool append_sphere_mesh(SceneData& data, const ProceduralGeometryDefinition& definition) {
  const float radius = sphere_radius(definition);
  if (radius <= kEpsilon) {
    log::warning("Procedural sphere `%s` has invalid dimensions %.6f %.6f %.6f - skipped", definition.id.c_str(), definition.dimensions.x, definition.dimensions.y,
      definition.dimensions.z);
    return false;
  }

  uint32_t subdivisions = definition.subdivisions;
  if (subdivisions > kMaxSphereSubdivisions) {
    log::warning("Procedural sphere `%s` subdivisions %u exceeds maximum %u - clamped", definition.id.c_str(), subdivisions, kMaxSphereSubdivisions);
    subdivisions = kMaxSphereSubdivisions;
  }

  const uint32_t material_index = resolve_material_index(data, definition);
  if (material_index == kInvalidIndex) {
    log::warning("Procedural sphere `%s` has no valid material - skipped", definition.id.c_str());
    return false;
  }

  std::vector<float3> local_vertices;
  std::vector<IcosphereFace> local_faces;
  build_icosahedron(local_vertices, local_faces);
  subdivide_icosphere(local_vertices, local_faces, subdivisions);

  const uint32_t vertex_start = static_cast<uint32_t>(data.vertices.pos.size());
  const uint32_t triangle_start = static_cast<uint32_t>(data.triangles.size());

  data.vertices.pos.reserve(data.vertices.pos.size() + local_vertices.size());
  data.vertices.nrm.reserve(data.vertices.nrm.size() + local_vertices.size());
  data.vertices.tan.reserve(data.vertices.tan.size() + local_vertices.size());
  data.vertices.btn.reserve(data.vertices.btn.size() + local_vertices.size());
  data.vertices.tex.reserve(data.vertices.tex.size() + local_vertices.size());

  for (const float3& local_position : local_vertices) {
    const float3 nrm = normalize(local_position);
    float3 tan = {};
    float3 btn = {};
    sphere_tangent_frame(nrm, tan, btn);

    data.vertices.pos.emplace_back(definition.center + nrm * radius);
    data.vertices.nrm.emplace_back(nrm);
    data.vertices.tan.emplace_back(tan);
    data.vertices.btn.emplace_back(btn);
    data.vertices.tex.emplace_back(sphere_uv(nrm));
  }

  data.triangles.reserve(data.triangles.size() + local_faces.size());

  for (const IcosphereFace& face : local_faces) {
    Triangle tri = {};
    tri.i[0] = vertex_start + face.i[0];
    tri.i[1] = vertex_start + face.i[1];
    tri.i[2] = vertex_start + face.i[2];
    tri.material_index = material_index;

    const float3 center_to_face = normalize((data.vertices.pos[tri.i[0]] + data.vertices.pos[tri.i[1]] + data.vertices.pos[tri.i[2]]) / 3.0f - definition.center);
    const float3 winding_normal = cross(data.vertices.pos[tri.i[1]] - data.vertices.pos[tri.i[0]], data.vertices.pos[tri.i[2]] - data.vertices.pos[tri.i[0]]);
    if (dot(winding_normal, center_to_face) < 0.0f) {
      std::swap(tri.i[1], tri.i[2]);
    }

    if (validate_triangle(tri, data.vertices.pos) == false) {
      continue;
    }

    data.triangles.emplace_back(tri);
  }

  const uint32_t triangle_end = static_cast<uint32_t>(data.triangles.size());
  const uint32_t triangle_count = triangle_end - triangle_start;
  if (triangle_count == 0u) {
    data.vertices.pos.resize(vertex_start);
    data.vertices.nrm.resize(vertex_start);
    data.vertices.tan.resize(vertex_start);
    data.vertices.btn.resize(vertex_start);
    data.vertices.tex.resize(vertex_start);
    return false;
  }

  const float3 radius_vector = {radius, radius, radius};
  const float3 bbox_min = definition.center - radius_vector;
  const float3 bbox_max = definition.center + radius_vector;
  const std::string mesh_name = unique_mesh_name(data, definition.id, "sphere");
  data.add_mesh(mesh_name.c_str(), triangle_start, triangle_count, bbox_min, bbox_max);
  return true;
}

bool append_plane_mesh(SceneData& data, const ProceduralGeometryDefinition& definition) {
  if ((definition.dimensions.x <= kEpsilon) || (definition.dimensions.z <= kEpsilon)) {
    log::warning("Procedural plane `%s` has invalid dimensions %.6f %.6f %.6f - skipped", definition.id.c_str(), definition.dimensions.x, definition.dimensions.y,
      definition.dimensions.z);
    return false;
  }

  const uint32_t material_index = resolve_material_index(data, definition);
  if (material_index == kInvalidIndex) {
    log::warning("Procedural plane `%s` has no valid material - skipped", definition.id.c_str());
    return false;
  }

  const uint32_t vertex_start = static_cast<uint32_t>(data.vertices.pos.size());
  const uint32_t triangle_start = static_cast<uint32_t>(data.triangles.size());
  const float half_width = definition.dimensions.x * 0.5f;
  const float half_depth = definition.dimensions.z * 0.5f;
  const float3 bbox_min = definition.center - float3{half_width, 0.0f, half_depth};
  const float3 bbox_max = definition.center + float3{half_width, 0.0f, half_depth};

  const float3 positions[4] = {
    {bbox_min.x, definition.center.y, bbox_min.z},
    {bbox_max.x, definition.center.y, bbox_min.z},
    {bbox_max.x, definition.center.y, bbox_max.z},
    {bbox_min.x, definition.center.y, bbox_max.z},
  };
  const float2 uvs[4] = {
    {0.0f, 0.0f},
    {1.0f, 0.0f},
    {1.0f, 1.0f},
    {0.0f, 1.0f},
  };

  data.vertices.pos.reserve(data.vertices.pos.size() + 4u);
  data.vertices.nrm.reserve(data.vertices.nrm.size() + 4u);
  data.vertices.tan.reserve(data.vertices.tan.size() + 4u);
  data.vertices.btn.reserve(data.vertices.btn.size() + 4u);
  data.vertices.tex.reserve(data.vertices.tex.size() + 4u);

  for (uint32_t i = 0u; i < 4u; ++i) {
    data.vertices.pos.emplace_back(positions[i]);
    data.vertices.nrm.emplace_back(float3{0.0f, 1.0f, 0.0f});
    data.vertices.tan.emplace_back(float3{1.0f, 0.0f, 0.0f});
    data.vertices.btn.emplace_back(float3{0.0f, 0.0f, 1.0f});
    data.vertices.tex.emplace_back(uvs[i]);
  }

  const uint32_t indices[6] = {
    0u,
    2u,
    1u,
    0u,
    3u,
    2u,
  };

  data.triangles.reserve(data.triangles.size() + 2u);
  for (uint32_t i = 0u; i < 6u; i += 3u) {
    Triangle tri = {};
    tri.i[0] = vertex_start + indices[i + 0u];
    tri.i[1] = vertex_start + indices[i + 1u];
    tri.i[2] = vertex_start + indices[i + 2u];
    tri.material_index = material_index;

    if (validate_triangle(tri, data.vertices.pos) == false) {
      continue;
    }

    data.triangles.emplace_back(tri);
  }

  const uint32_t triangle_end = static_cast<uint32_t>(data.triangles.size());
  const uint32_t triangle_count = triangle_end - triangle_start;
  if (triangle_count == 0u) {
    data.vertices.pos.resize(vertex_start);
    data.vertices.nrm.resize(vertex_start);
    data.vertices.tan.resize(vertex_start);
    data.vertices.btn.resize(vertex_start);
    data.vertices.tex.resize(vertex_start);
    return false;
  }

  const std::string mesh_name = unique_mesh_name(data, definition.id, "plane");
  data.add_mesh(mesh_name.c_str(), triangle_start, triangle_count, bbox_min, bbox_max);
  return true;
}

bool append_disk_mesh(SceneData& data, const ProceduralGeometryDefinition& definition) {
  if ((std::isfinite(definition.center.x) == false) || (std::isfinite(definition.center.y) == false) || (std::isfinite(definition.center.z) == false)) {
    log::warning("Procedural disk `%s` has a non-finite center - skipped", definition.id.c_str());
    return false;
  }
  if ((std::isfinite(definition.radius) == false) || (definition.radius <= kEpsilon)) {
    log::warning("Procedural disk `%s` has invalid radius %.6f - skipped", definition.id.c_str(), definition.radius);
    return false;
  }
  if ((std::isfinite(definition.inner_radius) == false) || (definition.inner_radius < 0.0f) || (definition.inner_radius >= definition.radius)) {
    log::warning("Procedural disk `%s` has invalid inner radius %.6f for outer radius %.6f - skipped", definition.id.c_str(), definition.inner_radius, definition.radius);
    return false;
  }
  if ((std::isfinite(definition.thickness) == false) || (definition.thickness < 0.0f) || ((definition.thickness > 0.0f) && (definition.thickness <= kEpsilon))) {
    log::warning("Procedural disk `%s` has invalid thickness %.6f - skipped", definition.id.c_str(), definition.thickness);
    return false;
  }
  if ((std::isfinite(definition.bevel_radius) == false) || (definition.bevel_radius < 0.0f)) {
    log::warning("Procedural disk `%s` has invalid bevel radius %.6f - skipped", definition.id.c_str(), definition.bevel_radius);
    return false;
  }
  if ((definition.thickness == 0.0f) && (definition.bevel_radius > 0.0f)) {
    log::warning("Procedural disk `%s` requires positive thickness for a bevel - skipped", definition.id.c_str());
    return false;
  }
  if (definition.bevel_radius > 0.0f) {
    const float maximum_radial_bevel = (definition.inner_radius > 0.0f) ? 0.5f * (definition.radius - definition.inner_radius) : definition.radius;
    const float maximum_bevel = min(0.5f * definition.thickness, maximum_radial_bevel);
    if ((definition.bevel_radius >= maximum_bevel) || (maximum_bevel <= kEpsilon)) {
      log::warning("Procedural disk `%s` bevel radius %.6f reaches or exceeds the geometric limit %.6f - skipped", definition.id.c_str(), definition.bevel_radius, maximum_bevel);
      return false;
    }
  }
  const float normal_length = length(definition.normal);
  if ((std::isfinite(definition.normal.x) == false) || (std::isfinite(definition.normal.y) == false) || (std::isfinite(definition.normal.z) == false) ||
      (std::isfinite(normal_length) == false) || (normal_length <= kEpsilon)) {
    log::warning("Procedural disk `%s` has invalid normal %.6f %.6f %.6f - skipped", definition.id.c_str(), definition.normal.x, definition.normal.y, definition.normal.z);
    return false;
  }

  uint32_t segments = definition.segments;
  if (segments < kMinimumDiskSegments) {
    log::warning("Procedural disk `%s` segments %u is below minimum %u - clamped", definition.id.c_str(), segments, kMinimumDiskSegments);
    segments = kMinimumDiskSegments;
  } else if (segments > kMaximumDiskSegments) {
    log::warning("Procedural disk `%s` segments %u exceeds maximum %u - clamped", definition.id.c_str(), segments, kMaximumDiskSegments);
    segments = kMaximumDiskSegments;
  }

  uint32_t bevel_segments = definition.bevel_segments;
  if (definition.bevel_radius > 0.0f) {
    if (bevel_segments == 0u) {
      log::warning("Procedural disk `%s` has a positive bevel radius but zero bevel segments - skipped", definition.id.c_str());
      return false;
    }
    if (bevel_segments > kMaximumDiskBevelSegments) {
      log::warning("Procedural disk `%s` bevel segments %u exceeds maximum %u - clamped", definition.id.c_str(), bevel_segments, kMaximumDiskBevelSegments);
      bevel_segments = kMaximumDiskBevelSegments;
    }
  }

  const uint32_t material_index = resolve_material_index(data, definition);
  if (material_index == kInvalidIndex) {
    log::warning("Procedural disk `%s` has no valid material - skipped", definition.id.c_str());
    return false;
  }

  const float3 normal = definition.normal / normal_length;
  const float3 tangent_axis = (abs(normal.y) < 0.999f) ? normalize(cross(float3{0.0f, 1.0f, 0.0f}, normal)) : float3{1.0f, 0.0f, 0.0f};
  const float3 bitangent_axis = normalize(cross(normal, tangent_axis));
  const bool has_center_hole = definition.inner_radius > 0.0f;
  const uint32_t vertex_start = static_cast<uint32_t>(data.vertices.pos.size());
  const uint32_t triangle_start = static_cast<uint32_t>(data.triangles.size());

  auto append_triangle = [&](const uint32_t i0, const uint32_t i1, const uint32_t i2) {
    Triangle triangle = {};
    triangle.i[0] = i0;
    triangle.i[1] = i1;
    triangle.i[2] = i2;
    triangle.material_index = material_index;
    if (validate_triangle(triangle, data.vertices.pos)) {
      data.triangles.emplace_back(triangle);
    }
  };

  auto append_surface_vertex = [&](const float radius, const float axial_offset, const float radial_normal, const float axial_normal, const float cosine, const float sine,
                                 const float2 texcoord) {
    const float3 radial_direction = normalize(cosine * tangent_axis + sine * bitangent_axis);
    const float3 surface_normal = normalize(radial_normal * radial_direction + axial_normal * normal);
    const float3 surface_tangent = normalize(axial_normal * radial_direction - radial_normal * normal);
    data.vertices.pos.emplace_back(definition.center + radius * radial_direction + axial_offset * normal);
    data.vertices.nrm.emplace_back(surface_normal);
    data.vertices.tan.emplace_back(surface_tangent);
    data.vertices.btn.emplace_back(normalize(cross(surface_normal, surface_tangent)));
    data.vertices.tex.emplace_back(texcoord);
  };

  auto append_strip = [&](const float radius0, const float offset0, const float radial_normal0, const float axial_normal0, const float radius1, const float offset1,
                        const float radial_normal1, const float axial_normal1) {
    const uint32_t strip_start = static_cast<uint32_t>(data.vertices.pos.size());
    for (uint32_t segment = 0u; segment <= segments; ++segment) {
      const float u = float(segment) / float(segments);
      const float angle = kDoublePi * u;
      const float cosine = cosf(angle);
      const float sine = sinf(angle);
      append_surface_vertex(radius0, offset0, radial_normal0, axial_normal0, cosine, sine, float2{u, 0.0f});
      append_surface_vertex(radius1, offset1, radial_normal1, axial_normal1, cosine, sine, float2{u, 1.0f});
    }
    for (uint32_t segment = 0u; segment < segments; ++segment) {
      const uint32_t current0 = strip_start + 2u * segment;
      const uint32_t current1 = current0 + 1u;
      const uint32_t next0 = current0 + 2u;
      const uint32_t next1 = current0 + 3u;
      append_triangle(current0, current1, next1);
      append_triangle(current0, next1, next0);
    }
  };

  auto append_cap = [&](const float axial_offset, const float axial_normal, const float inner_radius, const float outer_radius) {
    if (inner_radius > 0.0f) {
      const uint32_t cap_start = static_cast<uint32_t>(data.vertices.pos.size());
      for (uint32_t segment = 0u; segment <= segments; ++segment) {
        const float angle = kDoublePi * float(segment) / float(segments);
        const float cosine = cosf(angle);
        const float sine = sinf(angle);
        const float radius0 = (axial_normal > 0.0f) ? inner_radius : outer_radius;
        const float radius1 = (axial_normal > 0.0f) ? outer_radius : inner_radius;
        const float normalized_radius0 = radius0 / definition.radius;
        const float normalized_radius1 = radius1 / definition.radius;
        append_surface_vertex(radius0, axial_offset, 0.0f, axial_normal, cosine, sine, float2{0.5f + 0.5f * normalized_radius0 * cosine, 0.5f + 0.5f * normalized_radius0 * sine});
        append_surface_vertex(radius1, axial_offset, 0.0f, axial_normal, cosine, sine, float2{0.5f + 0.5f * normalized_radius1 * cosine, 0.5f + 0.5f * normalized_radius1 * sine});
      }
      for (uint32_t segment = 0u; segment < segments; ++segment) {
        const uint32_t current0 = cap_start + 2u * segment;
        const uint32_t current1 = current0 + 1u;
        const uint32_t next0 = current0 + 2u;
        const uint32_t next1 = current0 + 3u;
        append_triangle(current0, current1, next1);
        append_triangle(current0, next1, next0);
      }
      return;
    }

    const uint32_t center_index = static_cast<uint32_t>(data.vertices.pos.size());
    data.vertices.pos.emplace_back(definition.center + axial_offset * normal);
    data.vertices.nrm.emplace_back(axial_normal * normal);
    data.vertices.tan.emplace_back(tangent_axis);
    data.vertices.btn.emplace_back(normalize(cross(axial_normal * normal, tangent_axis)));
    data.vertices.tex.emplace_back(float2{0.5f, 0.5f});

    const uint32_t ring_start = static_cast<uint32_t>(data.vertices.pos.size());
    for (uint32_t segment = 0u; segment <= segments; ++segment) {
      const float angle = kDoublePi * float(segment) / float(segments);
      const float cosine = cosf(angle);
      const float sine = sinf(angle);
      append_surface_vertex(outer_radius, axial_offset, 0.0f, axial_normal, cosine, sine, float2{0.5f + 0.5f * cosine, 0.5f + 0.5f * sine});
    }
    for (uint32_t segment = 0u; segment < segments; ++segment) {
      if (axial_normal > 0.0f) {
        append_triangle(center_index, ring_start + segment, ring_start + segment + 1u);
      } else {
        append_triangle(center_index, ring_start + segment + 1u, ring_start + segment);
      }
    }
  };

  if (definition.thickness == 0.0f) {
    append_cap(0.0f, 1.0f, definition.inner_radius, definition.radius);
  } else {
    const float half_thickness = 0.5f * definition.thickness;
    const float bevel = definition.bevel_radius;
    const float top_inner_radius = has_center_hole ? definition.inner_radius + bevel : 0.0f;
    const float top_outer_radius = definition.radius - bevel;
    append_cap(half_thickness, 1.0f, top_inner_radius, top_outer_radius);

    if (bevel > 0.0f) {
      const float outer_center_radius = definition.radius - bevel;
      const float top_bevel_center = half_thickness - bevel;
      for (uint32_t i = 0u; i < bevel_segments; ++i) {
        const float angle0 = 0.5f * kPi * (1.0f - float(i) / float(bevel_segments));
        const float angle1 = 0.5f * kPi * (1.0f - float(i + 1u) / float(bevel_segments));
        append_strip(outer_center_radius + bevel * cosf(angle0), top_bevel_center + bevel * sinf(angle0), cosf(angle0), sinf(angle0), outer_center_radius + bevel * cosf(angle1),
          top_bevel_center + bevel * sinf(angle1), cosf(angle1), sinf(angle1));
      }
    }

    append_strip(definition.radius, half_thickness - bevel, 1.0f, 0.0f, definition.radius, -half_thickness + bevel, 1.0f, 0.0f);

    if (bevel > 0.0f) {
      const float outer_center_radius = definition.radius - bevel;
      const float bottom_bevel_center = -half_thickness + bevel;
      for (uint32_t i = 0u; i < bevel_segments; ++i) {
        const float angle0 = -0.5f * kPi * float(i) / float(bevel_segments);
        const float angle1 = -0.5f * kPi * float(i + 1u) / float(bevel_segments);
        append_strip(outer_center_radius + bevel * cosf(angle0), bottom_bevel_center + bevel * sinf(angle0), cosf(angle0), sinf(angle0), outer_center_radius + bevel * cosf(angle1),
          bottom_bevel_center + bevel * sinf(angle1), cosf(angle1), sinf(angle1));
      }
    }

    const float bottom_inner_radius = has_center_hole ? definition.inner_radius + bevel : 0.0f;
    append_cap(-half_thickness, -1.0f, bottom_inner_radius, top_outer_radius);

    if (has_center_hole) {
      if (bevel > 0.0f) {
        const float inner_center_radius = definition.inner_radius + bevel;
        const float bottom_bevel_center = -half_thickness + bevel;
        for (uint32_t i = 0u; i < bevel_segments; ++i) {
          const float angle0 = -0.5f * kPi - 0.5f * kPi * float(i) / float(bevel_segments);
          const float angle1 = -0.5f * kPi - 0.5f * kPi * float(i + 1u) / float(bevel_segments);
          append_strip(inner_center_radius + bevel * cosf(angle0), bottom_bevel_center + bevel * sinf(angle0), cosf(angle0), sinf(angle0),
            inner_center_radius + bevel * cosf(angle1), bottom_bevel_center + bevel * sinf(angle1), cosf(angle1), sinf(angle1));
        }
      }

      append_strip(definition.inner_radius, -half_thickness + bevel, -1.0f, 0.0f, definition.inner_radius, half_thickness - bevel, -1.0f, 0.0f);

      if (bevel > 0.0f) {
        const float inner_center_radius = definition.inner_radius + bevel;
        const float top_bevel_center = half_thickness - bevel;
        for (uint32_t i = 0u; i < bevel_segments; ++i) {
          const float angle0 = kPi - 0.5f * kPi * float(i) / float(bevel_segments);
          const float angle1 = kPi - 0.5f * kPi * float(i + 1u) / float(bevel_segments);
          append_strip(inner_center_radius + bevel * cosf(angle0), top_bevel_center + bevel * sinf(angle0), cosf(angle0), sinf(angle0), inner_center_radius + bevel * cosf(angle1),
            top_bevel_center + bevel * sinf(angle1), cosf(angle1), sinf(angle1));
        }
      }
    }
  }

  const uint32_t triangle_end = static_cast<uint32_t>(data.triangles.size());
  const uint32_t triangle_count = triangle_end - triangle_start;
  if (triangle_count == 0u) {
    data.vertices.pos.resize(vertex_start);
    data.vertices.nrm.resize(vertex_start);
    data.vertices.tan.resize(vertex_start);
    data.vertices.btn.resize(vertex_start);
    data.vertices.tex.resize(vertex_start);
    return false;
  }

  const float3 projected_radius = definition.radius * float3{
                                                        sqrtf(max(0.0f, 1.0f - normal.x * normal.x)),
                                                        sqrtf(max(0.0f, 1.0f - normal.y * normal.y)),
                                                        sqrtf(max(0.0f, 1.0f - normal.z * normal.z)),
                                                      };
  const float3 projected_half_thickness = 0.5f * definition.thickness * float3{abs(normal.x), abs(normal.y), abs(normal.z)};
  const std::string mesh_name = unique_mesh_name(data, definition.id, "disk");
  data.add_mesh(mesh_name.c_str(), triangle_start, triangle_count, definition.center - projected_radius - projected_half_thickness,
    definition.center + projected_radius + projected_half_thickness);
  return true;
}

}  // namespace

bool is_procedural_geometry_entry(const std::string& name) {
  return is_geometry_entry(name) || is_sphere_alias(name) || is_plane_alias(name) || is_disk_alias(name);
}

bool parse_procedural_geometry_definition(const MaterialDefinition& material, ProceduralGeometryDefinition& out_definition) {
  out_definition = {};

  if (is_sphere_alias(material.name)) {
    out_definition.cls = ProceduralGeometryDefinition::Class::Sphere;
  } else if (is_plane_alias(material.name)) {
    out_definition.cls = ProceduralGeometryDefinition::Class::Plane;
  } else if (is_disk_alias(material.name)) {
    out_definition.cls = ProceduralGeometryDefinition::Class::Disk;
  } else if (is_geometry_entry(material.name)) {
    std::string class_name;
    if (read_property(material, "class", class_name) == false) {
      log::warning("Procedural geometry entry requires `class` - skipped");
      return false;
    }

    if (class_name == "sphere") {
      out_definition.cls = ProceduralGeometryDefinition::Class::Sphere;
    } else if (class_name == "plane") {
      out_definition.cls = ProceduralGeometryDefinition::Class::Plane;
    } else if (class_name == "disk") {
      out_definition.cls = ProceduralGeometryDefinition::Class::Disk;
    } else {
      log::warning("Unsupported procedural geometry class `%s` - skipped", class_name.c_str());
      return false;
    }
  } else {
    return false;
  }

  if (out_definition.cls == ProceduralGeometryDefinition::Class::Sphere) {
    out_definition.dimensions = {2.0f, 2.0f, 2.0f};
  } else if (out_definition.cls == ProceduralGeometryDefinition::Class::Plane) {
    out_definition.dimensions = {1.0f, 0.0f, 1.0f};
  } else if (out_definition.cls == ProceduralGeometryDefinition::Class::Disk) {
    out_definition.radius = 0.5f;
    out_definition.inner_radius = 0.0f;
    out_definition.thickness = 0.0f;
    out_definition.bevel_radius = 0.0f;
    out_definition.segments = 128u;
    out_definition.bevel_segments = 0u;
  }

  std::string value;
  if (read_property(material, "id", value)) {
    out_definition.id = value;
  }

  if (read_property(material, "material", value)) {
    out_definition.material_name = value;
  }

  uint32_t material_index = kInvalidIndex;
  if (read_uint32_property(material, "material-index", material_index)) {
    out_definition.material_index = material_index;
  }

  read_float3_property(material, "center", out_definition.center);

  if (out_definition.cls == ProceduralGeometryDefinition::Class::Disk) {
    if (read_float_property(material, "radius", out_definition.radius) == false) {
      float diameter = 0.0f;
      if (read_float_property(material, "diameter", diameter)) {
        out_definition.radius = 0.5f * diameter;
      }
    }
    if (read_float_property(material, "inner-radius", out_definition.inner_radius) == false) {
      if (read_float_property(material, "inner_radius", out_definition.inner_radius) == false) {
        if (read_float_property(material, "hole-radius", out_definition.inner_radius) == false) {
          read_float_property(material, "hole_radius", out_definition.inner_radius);
        }
      }
    }
    read_float3_property(material, "normal", out_definition.normal);
    read_float_property(material, "thickness", out_definition.thickness);
    if (read_float_property(material, "bevel-radius", out_definition.bevel_radius) == false) {
      read_float_property(material, "bevel_radius", out_definition.bevel_radius);
    }
    read_uint32_property(material, "segments", out_definition.segments);
    if (read_uint32_property(material, "bevel-segments", out_definition.bevel_segments) == false) {
      read_uint32_property(material, "bevel_segments", out_definition.bevel_segments);
    }
  } else {
    const bool has_dimensions = read_float3_property(material, "dimensions", out_definition.dimensions);
    if (has_dimensions == false) {
      if (out_definition.cls == ProceduralGeometryDefinition::Class::Sphere) {
        float radius = 0.0f;
        if (read_float_property(material, "radius", radius)) {
          const float diameter = radius * 2.0f;
          out_definition.dimensions = {diameter, diameter, diameter};
        }
      } else if (out_definition.cls == ProceduralGeometryDefinition::Class::Plane) {
        float2 size = {};
        if (read_float2_property(material, "size", size)) {
          out_definition.dimensions = {size.x, 0.0f, size.y};
        }
      }
    }

    read_uint32_property(material, "subdivisions", out_definition.subdivisions);
  }

  return out_definition.cls != ProceduralGeometryDefinition::Class::Invalid;
}

uint32_t generate_procedural_geometry(SceneData& data, const std::vector<ProceduralGeometryDefinition>& definitions) {
  uint32_t generated_count = 0u;
  for (const ProceduralGeometryDefinition& definition : definitions) {
    if (definition.cls == ProceduralGeometryDefinition::Class::Sphere) {
      if (append_sphere_mesh(data, definition)) {
        ++generated_count;
      }
    } else if (definition.cls == ProceduralGeometryDefinition::Class::Plane) {
      if (append_plane_mesh(data, definition)) {
        ++generated_count;
      }
    } else if (definition.cls == ProceduralGeometryDefinition::Class::Disk) {
      if (append_disk_mesh(data, definition)) {
        ++generated_count;
      }
    }
  }
  return generated_count;
}

}  // namespace etx
