#include <etx/render/host/scene_procedural_geometry.hxx>

#include <etx/core/log.hxx>
#include <etx/render/host/scene_data.hxx>
#include <etx/render/host/scene_serialization.hxx>
#include <etx/render/shared/scene.hxx>

#include <algorithm>
#include <array>
#include <charconv>

namespace etx {
namespace {

static constexpr uint32_t kMaxSphereSubdivisions = 6u;
static constexpr uint32_t kMinimumDiskSegments = 3u;
static constexpr uint32_t kMaximumDiskSegments = 4096u;
static constexpr uint32_t kMaximumDiskBevelSegments = 32u;
static constexpr uint32_t kMinimumRoundSegments = 3u;
static constexpr uint32_t kMaximumRoundSegments = 4096u;
static constexpr uint32_t kMaximumRoundSubdivisions = 256u;

struct IcosphereFace {
  uint32_t i[3] = {};
};

struct UnitCircleSample {
  float u = 0.0f;
  float cosine = 1.0f;
  float sine = 0.0f;
};

UnitCircleSample cardinal_unit_circle_sample(float u, int32_t quarter_turn) {
  const int32_t normalized_quarter_turn = ((quarter_turn % 4) + 4) % 4;
  switch (normalized_quarter_turn) {
    case 0:
      return {u, 1.0f, 0.0f};
    case 1:
      return {u, 0.0f, 1.0f};
    case 2:
      return {u, -1.0f, 0.0f};
    default:
      return {u, 0.0f, -1.0f};
  }
}

UnitCircleSample unit_circle_arc_sample(uint32_t segment, uint32_t segment_count, int32_t start_quarter_turn, int32_t end_quarter_turn) {
  if (segment == 0u) {
    return cardinal_unit_circle_sample(0.0f, start_quarter_turn);
  }
  if (segment == segment_count) {
    return cardinal_unit_circle_sample(1.0f, end_quarter_turn);
  }
  const float u = float(segment) / float(segment_count);
  const float quarter_turn = float(start_quarter_turn) + float(end_quarter_turn - start_quarter_turn) * u;
  const float angle = 0.5f * kPi * quarter_turn;
  return {u, cosf(angle), sinf(angle)};
}

UnitCircleSample unit_circle_sample(uint32_t segment, uint32_t segment_count) {
  return unit_circle_arc_sample(segment, segment_count, 0, 4);
}

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
  const char* value_begin = value.data();
  const char* value_end = value_begin + value.size();
  const auto [parsed_end, error] = std::from_chars(value_begin, value_end, parsed);
  if ((error != std::errc{}) || (parsed_end != value_end)) {
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

bool is_box_alias(const std::string& name) {
  return (name == "et::box") || (name == "etx::box");
}

bool is_cone_alias(const std::string& name) {
  return (name == "et::cone") || (name == "etx::cone");
}

bool is_capsule_alias(const std::string& name) {
  return (name == "et::capsule") || (name == "etx::capsule");
}

bool is_torus_alias(const std::string& name) {
  return (name == "et::torus") || (name == "etx::torus");
}

bool is_tetrahedron_alias(const std::string& name) {
  return (name == "et::tetrahedron") || (name == "etx::tetrahedron");
}

bool is_octahedron_alias(const std::string& name) {
  return (name == "et::octahedron") || (name == "etx::octahedron");
}

bool is_dodecahedron_alias(const std::string& name) {
  return (name == "et::dodecahedron") || (name == "etx::dodecahedron");
}

bool is_icosahedron_alias(const std::string& name) {
  return (name == "et::icosahedron") || (name == "etx::icosahedron");
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

bool finite_float3(const float3& value) {
  return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}

bool finite_float2(const float2& value) {
  return std::isfinite(value.x) && std::isfinite(value.y);
}

bool validate_procedural_triangle(Triangle& triangle, const std::vector<float3>& positions) {
  if ((triangle.i[0] >= positions.size()) || (triangle.i[1] >= positions.size()) || (triangle.i[2] >= positions.size())) {
    return false;
  }
  if ((finite_float3(positions[triangle.i[0]]) == false) || (finite_float3(positions[triangle.i[1]]) == false) || (finite_float3(positions[triangle.i[2]]) == false)) {
    return false;
  }
  return validate_triangle(triangle, positions) && finite_float3(triangle.geo_n);
}

float3 stable_tangent(const float3& normal) {
  const float3 reference = (abs(normal.y) < 0.999f) ? float3{0.0f, 1.0f, 0.0f} : float3{1.0f, 0.0f, 0.0f};
  return normalize(cross(reference, normal));
}

struct BakedMeshBuilder {
  SceneData& data;
  uint32_t material_index = kInvalidIndex;
  uint32_t vertex_start = 0u;
  uint32_t triangle_start = 0u;
  bool valid = true;
  float3 bbox_min = {kMaxFloat, kMaxFloat, kMaxFloat};
  float3 bbox_max = {-kMaxFloat, -kMaxFloat, -kMaxFloat};

  BakedMeshBuilder(SceneData& scene_data, uint32_t material)
    : data(scene_data)
    , material_index(material)
    , vertex_start(static_cast<uint32_t>(scene_data.vertices.pos.size()))
    , triangle_start(static_cast<uint32_t>(scene_data.triangles.size())) {
  }

  void reserve(uint32_t vertex_count, uint32_t triangle_count) {
    data.vertices.pos.reserve(data.vertices.pos.size() + vertex_count);
    data.vertices.nrm.reserve(data.vertices.nrm.size() + vertex_count);
    data.vertices.tan.reserve(data.vertices.tan.size() + vertex_count);
    data.vertices.btn.reserve(data.vertices.btn.size() + vertex_count);
    data.vertices.tex.reserve(data.vertices.tex.size() + vertex_count);
    data.triangles.reserve(data.triangles.size() + triangle_count);
  }

  uint32_t append_vertex(const float3& position, const float3& normal, const float3& tangent, const float2& texcoord) {
    const float normal_length_squared = dot(normal, normal);
    if ((valid == false) || (finite_float3(position) == false) || (finite_float3(normal) == false) || (finite_float3(tangent) == false) || (finite_float2(texcoord) == false) ||
        (std::isfinite(normal_length_squared) == false) || (normal_length_squared <= 0.0f)) {
      valid = false;
      return kInvalidIndex;
    }

    const float3 normalized_normal = normal / sqrtf(normal_length_squared);
    float3 orthogonal_tangent = tangent - normalized_normal * dot(normalized_normal, tangent);
    const float tangent_length_squared = dot(orthogonal_tangent, orthogonal_tangent);
    if ((finite_float3(normalized_normal) == false) || (finite_float3(orthogonal_tangent) == false) || (std::isfinite(tangent_length_squared) == false)) {
      valid = false;
      return kInvalidIndex;
    }
    if (tangent_length_squared <= (kEpsilon * kEpsilon)) {
      orthogonal_tangent = stable_tangent(normalized_normal);
    } else {
      orthogonal_tangent /= sqrtf(tangent_length_squared);
    }

    const uint32_t vertex_index = static_cast<uint32_t>(data.vertices.pos.size());
    data.vertices.pos.emplace_back(position);
    data.vertices.nrm.emplace_back(normalized_normal);
    data.vertices.tan.emplace_back(orthogonal_tangent);
    data.vertices.btn.emplace_back(normalize(cross(normalized_normal, orthogonal_tangent)));
    data.vertices.tex.emplace_back(texcoord);
    bbox_min = min(bbox_min, position);
    bbox_max = max(bbox_max, position);
    return vertex_index;
  }

  bool append_triangle(uint32_t i0, uint32_t i1, uint32_t i2) {
    if ((valid == false) || (i0 >= data.vertices.pos.size()) || (i1 >= data.vertices.pos.size()) || (i2 >= data.vertices.pos.size())) {
      return false;
    }

    const float3 edge0 = data.vertices.pos[i1] - data.vertices.pos[i0];
    const float3 edge1 = data.vertices.pos[i2] - data.vertices.pos[i0];
    const float3 edge2 = data.vertices.pos[i2] - data.vertices.pos[i1];
    const float3 face_normal = cross(edge0, edge1);
    const float maximum_edge_length_squared = max(dot(edge0, edge0), max(dot(edge1, edge1), dot(edge2, edge2)));
    const float face_normal_length_squared = dot(face_normal, face_normal);
    const float minimum_face_normal_length = kEpsilon * maximum_edge_length_squared;
    if ((finite_float3(face_normal) == false) || (std::isfinite(maximum_edge_length_squared) == false) || (std::isfinite(face_normal_length_squared) == false) ||
        (maximum_edge_length_squared <= 0.0f) || (face_normal_length_squared <= (minimum_face_normal_length * minimum_face_normal_length))) {
      return false;
    }

    Triangle triangle = {};
    triangle.i[0] = i0;
    triangle.i[1] = i1;
    triangle.i[2] = i2;
    triangle.material_index = material_index;
    if (validate_procedural_triangle(triangle, data.vertices.pos) == false) {
      return false;
    }
    data.triangles.emplace_back(triangle);
    return true;
  }

  bool append_outward_triangle(uint32_t i0, uint32_t i1, uint32_t i2, const float3& center) {
    if ((valid == false) || (i0 >= data.vertices.pos.size()) || (i1 >= data.vertices.pos.size()) || (i2 >= data.vertices.pos.size())) {
      return false;
    }

    const float3& p0 = data.vertices.pos[i0];
    const float3& p1 = data.vertices.pos[i1];
    const float3& p2 = data.vertices.pos[i2];
    const float3 face_center = (p0 + p1 + p2) / 3.0f;
    const float3 face_normal = cross(p1 - p0, p2 - p0);
    if (dot(face_normal, face_center - center) < 0.0f) {
      std::swap(i1, i2);
    }
    return append_triangle(i0, i1, i2);
  }

  void rollback() {
    data.vertices.pos.resize(vertex_start);
    data.vertices.nrm.resize(vertex_start);
    data.vertices.tan.resize(vertex_start);
    data.vertices.btn.resize(vertex_start);
    data.vertices.tex.resize(vertex_start);
    data.triangles.resize(triangle_start);
  }

  bool finish(const ProceduralGeometryDefinition& definition, const char* fallback_name) {
    const uint32_t triangle_count = static_cast<uint32_t>(data.triangles.size()) - triangle_start;
    if ((valid == false) || (triangle_count == 0u)) {
      rollback();
      return false;
    }

    const std::string mesh_name = unique_mesh_name(data, definition.id, fallback_name);
    data.add_mesh(mesh_name.c_str(), triangle_start, triangle_count, bbox_min, bbox_max);
    return true;
  }
};

bool append_flat_polygon(BakedMeshBuilder& builder, const float3* positions, uint32_t count, const float3& center) {
  if ((count < 3u) || (count > 5u)) {
    return false;
  }

  float3 face_center = {};
  for (uint32_t i = 0u; i < count; ++i) {
    face_center += positions[i];
  }
  face_center /= float(count);

  const float3 source_normal = cross(positions[1] - positions[0], positions[2] - positions[0]);
  const float source_normal_length_squared = dot(source_normal, source_normal);
  if ((finite_float3(source_normal) == false) || (std::isfinite(source_normal_length_squared) == false) || (source_normal_length_squared <= 0.0f)) {
    return false;
  }
  const bool reverse_winding = dot(source_normal, face_center - center) < 0.0f;

  std::array<float3, 5u> ordered_positions = {};
  for (uint32_t i = 0u; i < count; ++i) {
    const uint32_t source_index = reverse_winding ? ((count - i) % count) : i;
    ordered_positions[i] = positions[source_index];
  }

  const float3 normal = normalize(cross(ordered_positions[1] - ordered_positions[0], ordered_positions[2] - ordered_positions[0]));
  const float3 tangent = normalize(ordered_positions[1] - ordered_positions[0]);
  const float3 bitangent = normalize(cross(normal, tangent));
  std::array<float2, 5u> projected = {};
  float2 projected_min = {kMaxFloat, kMaxFloat};
  float2 projected_max = {-kMaxFloat, -kMaxFloat};
  for (uint32_t i = 0u; i < count; ++i) {
    const float3 offset = ordered_positions[i] - face_center;
    projected[i] = {dot(offset, tangent), dot(offset, bitangent)};
    projected_min = min(projected_min, projected[i]);
    projected_max = max(projected_max, projected[i]);
  }

  const float2 projected_size = projected_max - projected_min;
  const uint32_t face_vertex_start = static_cast<uint32_t>(builder.data.vertices.pos.size());
  for (uint32_t i = 0u; i < count; ++i) {
    const float u = (projected_size.x > kEpsilon) ? ((projected[i].x - projected_min.x) / projected_size.x) : 0.5f;
    const float v = (projected_size.y > kEpsilon) ? ((projected[i].y - projected_min.y) / projected_size.y) : 0.5f;
    builder.append_vertex(ordered_positions[i], normal, tangent, {u, v});
  }

  bool valid = true;
  for (uint32_t i = 1u; i + 1u < count; ++i) {
    valid = builder.append_triangle(face_vertex_start, face_vertex_start + i, face_vertex_start + i + 1u) && valid;
  }
  return valid;
}

bool append_flat_triangular_mesh(SceneData& data, const ProceduralGeometryDefinition& definition, const std::vector<float3>& unit_vertices, const std::vector<IcosphereFace>& faces,
  const char* fallback_name) {
  if ((finite_float3(definition.center) == false) || (std::isfinite(definition.radius) == false) || (definition.radius <= kEpsilon)) {
    log::warning("Procedural %s `%s` has invalid center or radius - skipped", fallback_name, definition.id.c_str());
    return false;
  }

  const uint32_t material_index = resolve_material_index(data, definition);
  if (material_index == kInvalidIndex) {
    log::warning("Procedural %s `%s` has no valid material - skipped", fallback_name, definition.id.c_str());
    return false;
  }

  BakedMeshBuilder builder(data, material_index);
  builder.reserve(static_cast<uint32_t>(faces.size()) * 3u, static_cast<uint32_t>(faces.size()));
  for (const IcosphereFace& face : faces) {
    if ((face.i[0] >= unit_vertices.size()) || (face.i[1] >= unit_vertices.size()) || (face.i[2] >= unit_vertices.size())) {
      builder.rollback();
      return false;
    }
    const float3 positions[3] = {
      definition.center + definition.radius * unit_vertices[face.i[0]],
      definition.center + definition.radius * unit_vertices[face.i[1]],
      definition.center + definition.radius * unit_vertices[face.i[2]],
    };
    if (append_flat_polygon(builder, positions, 3u, definition.center) == false) {
      builder.rollback();
      return false;
    }
  }
  return builder.finish(definition, fallback_name);
}

float sphere_radius(const ProceduralGeometryDefinition& definition) {
  return definition.dimensions.x * 0.5f;
}

bool append_sphere_mesh(SceneData& data, const ProceduralGeometryDefinition& definition) {
  const float radius = sphere_radius(definition);
  if ((finite_float3(definition.center) == false) || (finite_float3(definition.dimensions) == false) || (radius <= kEpsilon)) {
    log::warning("Procedural sphere `%s` has invalid dimensions %.6f %.6f %.6f - skipped", definition.id.c_str(), definition.dimensions.x, definition.dimensions.y,
      definition.dimensions.z);
    return false;
  }

  const float3 radius_vector = {radius, radius, radius};
  const float3 bbox_min = definition.center - radius_vector;
  const float3 bbox_max = definition.center + radius_vector;
  if ((finite_float3(bbox_min) == false) || (finite_float3(bbox_max) == false)) {
    log::warning("Procedural sphere `%s` has non-finite bounds - skipped", definition.id.c_str());
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
  const auto rollback = [&]() {
    data.vertices.pos.resize(vertex_start);
    data.vertices.nrm.resize(vertex_start);
    data.vertices.tan.resize(vertex_start);
    data.vertices.btn.resize(vertex_start);
    data.vertices.tex.resize(vertex_start);
    data.triangles.resize(triangle_start);
  };

  const size_t emitted_vertex_count = 3u * local_faces.size();
  data.vertices.pos.reserve(data.vertices.pos.size() + emitted_vertex_count);
  data.vertices.nrm.reserve(data.vertices.nrm.size() + emitted_vertex_count);
  data.vertices.tan.reserve(data.vertices.tan.size() + emitted_vertex_count);
  data.vertices.btn.reserve(data.vertices.btn.size() + emitted_vertex_count);
  data.vertices.tex.reserve(data.vertices.tex.size() + emitted_vertex_count);
  data.triangles.reserve(data.triangles.size() + local_faces.size());

  for (const IcosphereFace& face : local_faces) {
    if ((face.i[0] >= local_vertices.size()) || (face.i[1] >= local_vertices.size()) || (face.i[2] >= local_vertices.size())) {
      rollback();
      return false;
    }

    std::array<float3, 3u> normals = {
      normalize(local_vertices[face.i[0]]),
      normalize(local_vertices[face.i[1]]),
      normalize(local_vertices[face.i[2]]),
    };
    std::array<float2, 3u> texcoords = {
      sphere_uv(normals[0]),
      sphere_uv(normals[1]),
      sphere_uv(normals[2]),
    };
    float minimum_u = texcoords[0].x;
    float maximum_u = texcoords[0].x;
    for (uint32_t vertex = 1u; vertex < texcoords.size(); ++vertex) {
      minimum_u = min(minimum_u, texcoords[vertex].x);
      maximum_u = max(maximum_u, texcoords[vertex].x);
    }
    if ((maximum_u - minimum_u) > 0.5f) {
      for (float2& texcoord : texcoords) {
        if (texcoord.x <= 0.5f) {
          texcoord.x += 1.0f;
        }
      }
    }
    for (uint32_t vertex = 0u; vertex < normals.size(); ++vertex) {
      const float horizontal_length_squared = (normals[vertex].x * normals[vertex].x) + (normals[vertex].z * normals[vertex].z);
      if (horizontal_length_squared <= (kEpsilon * kEpsilon)) {
        texcoords[vertex].x = 0.5f * (texcoords[(vertex + 1u) % 3u].x + texcoords[(vertex + 2u) % 3u].x);
      }
    }

    const uint32_t face_vertex_start = static_cast<uint32_t>(data.vertices.pos.size());
    for (uint32_t vertex = 0u; vertex < normals.size(); ++vertex) {
      float3 tangent = {};
      float3 bitangent = {};
      sphere_tangent_frame(normals[vertex], tangent, bitangent);
      const float horizontal_length_squared = (normals[vertex].x * normals[vertex].x) + (normals[vertex].z * normals[vertex].z);
      if (horizontal_length_squared <= (kEpsilon * kEpsilon)) {
        const float longitude = (texcoords[vertex].x - 0.5f) * kDoublePi;
        tangent = {-sinf(longitude), 0.0f, cosf(longitude)};
        bitangent = normalize(cross(normals[vertex], tangent));
      }
      data.vertices.pos.emplace_back(definition.center + normals[vertex] * radius);
      data.vertices.nrm.emplace_back(normals[vertex]);
      data.vertices.tan.emplace_back(tangent);
      data.vertices.btn.emplace_back(bitangent);
      data.vertices.tex.emplace_back(texcoords[vertex]);
    }

    Triangle tri = {};
    tri.i[0] = face_vertex_start;
    tri.i[1] = face_vertex_start + 1u;
    tri.i[2] = face_vertex_start + 2u;
    tri.material_index = material_index;

    const float3 center_to_face = normalize((data.vertices.pos[tri.i[0]] + data.vertices.pos[tri.i[1]] + data.vertices.pos[tri.i[2]]) / 3.0f - definition.center);
    const float3 winding_normal = cross(data.vertices.pos[tri.i[1]] - data.vertices.pos[tri.i[0]], data.vertices.pos[tri.i[2]] - data.vertices.pos[tri.i[0]]);
    if (dot(winding_normal, center_to_face) < 0.0f) {
      std::swap(tri.i[1], tri.i[2]);
    }

    if (validate_procedural_triangle(tri, data.vertices.pos) == false) {
      rollback();
      return false;
    }

    data.triangles.emplace_back(tri);
  }

  const uint32_t triangle_end = static_cast<uint32_t>(data.triangles.size());
  const uint32_t triangle_count = triangle_end - triangle_start;
  if (triangle_count == 0u) {
    rollback();
    return false;
  }

  const std::string mesh_name = unique_mesh_name(data, definition.id, "sphere");
  data.add_mesh(mesh_name.c_str(), triangle_start, triangle_count, bbox_min, bbox_max);
  return true;
}

bool append_plane_mesh(SceneData& data, const ProceduralGeometryDefinition& definition) {
  if ((finite_float3(definition.center) == false) || (finite_float3(definition.dimensions) == false) || (definition.dimensions.x <= kEpsilon) ||
      (definition.dimensions.z <= kEpsilon)) {
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
  if ((finite_float3(bbox_min) == false) || (finite_float3(bbox_max) == false)) {
    log::warning("Procedural plane `%s` has non-finite bounds - skipped", definition.id.c_str());
    return false;
  }

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

    if (validate_procedural_triangle(tri, data.vertices.pos) == false) {
      data.vertices.pos.resize(vertex_start);
      data.vertices.nrm.resize(vertex_start);
      data.vertices.tan.resize(vertex_start);
      data.vertices.btn.resize(vertex_start);
      data.vertices.tex.resize(vertex_start);
      data.triangles.resize(triangle_start);
      return false;
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
  const float3 projected_radius = definition.radius * float3{
                                                        sqrtf(max(0.0f, 1.0f - normal.x * normal.x)),
                                                        sqrtf(max(0.0f, 1.0f - normal.y * normal.y)),
                                                        sqrtf(max(0.0f, 1.0f - normal.z * normal.z)),
                                                      };
  const float3 projected_half_thickness = 0.5f * definition.thickness * float3{abs(normal.x), abs(normal.y), abs(normal.z)};
  const float3 bbox_min = definition.center - projected_radius - projected_half_thickness;
  const float3 bbox_max = definition.center + projected_radius + projected_half_thickness;
  if ((finite_float3(bbox_min) == false) || (finite_float3(bbox_max) == false)) {
    log::warning("Procedural disk `%s` has non-finite bounds - skipped", definition.id.c_str());
    return false;
  }
  const uint32_t vertex_start = static_cast<uint32_t>(data.vertices.pos.size());
  const uint32_t triangle_start = static_cast<uint32_t>(data.triangles.size());
  bool geometry_valid = true;

  auto append_triangle = [&](const uint32_t i0, const uint32_t i1, const uint32_t i2) {
    Triangle triangle = {};
    triangle.i[0] = i0;
    triangle.i[1] = i1;
    triangle.i[2] = i2;
    triangle.material_index = material_index;
    if (validate_procedural_triangle(triangle, data.vertices.pos)) {
      data.triangles.emplace_back(triangle);
    } else {
      geometry_valid = false;
    }
  };

  auto append_surface_vertex = [&](const float radius, const float axial_offset, const float radial_normal, const float axial_normal, const float cosine, const float sine,
                                 const float3& surface_tangent, const float2 texcoord) {
    const float3 radial_direction = normalize(cosine * tangent_axis + sine * bitangent_axis);
    const float3 surface_normal = normalize(radial_normal * radial_direction + axial_normal * normal);
    data.vertices.pos.emplace_back(definition.center + radius * radial_direction + axial_offset * normal);
    data.vertices.nrm.emplace_back(surface_normal);
    data.vertices.tan.emplace_back(normalize(surface_tangent));
    data.vertices.btn.emplace_back(normalize(cross(surface_normal, surface_tangent)));
    data.vertices.tex.emplace_back(texcoord);
  };

  auto append_strip = [&](const float radius0, const float offset0, const float radial_normal0, const float axial_normal0, const float radius1, const float offset1,
                        const float radial_normal1, const float axial_normal1) {
    const uint32_t strip_start = static_cast<uint32_t>(data.vertices.pos.size());
    for (uint32_t segment = 0u; segment <= segments; ++segment) {
      const UnitCircleSample sample = unit_circle_sample(segment, segments);
      const float3 azimuth_tangent = -sample.sine * tangent_axis + sample.cosine * bitangent_axis;
      append_surface_vertex(radius0, offset0, radial_normal0, axial_normal0, sample.cosine, sample.sine, azimuth_tangent, float2{sample.u, 1.0f});
      append_surface_vertex(radius1, offset1, radial_normal1, axial_normal1, sample.cosine, sample.sine, azimuth_tangent, float2{sample.u, 0.0f});
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
        const UnitCircleSample sample = unit_circle_sample(segment, segments);
        const float radius0 = (axial_normal > 0.0f) ? inner_radius : outer_radius;
        const float radius1 = (axial_normal > 0.0f) ? outer_radius : inner_radius;
        const float normalized_radius0 = radius0 / definition.radius;
        const float normalized_radius1 = radius1 / definition.radius;
        const float oriented_sine = axial_normal * sample.sine;
        append_surface_vertex(radius0, axial_offset, 0.0f, axial_normal, sample.cosine, sample.sine, tangent_axis,
          float2{0.5f + 0.5f * normalized_radius0 * sample.cosine, 0.5f + 0.5f * normalized_radius0 * oriented_sine});
        append_surface_vertex(radius1, axial_offset, 0.0f, axial_normal, sample.cosine, sample.sine, tangent_axis,
          float2{0.5f + 0.5f * normalized_radius1 * sample.cosine, 0.5f + 0.5f * normalized_radius1 * oriented_sine});
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
      const UnitCircleSample sample = unit_circle_sample(segment, segments);
      append_surface_vertex(outer_radius, axial_offset, 0.0f, axial_normal, sample.cosine, sample.sine, tangent_axis,
        float2{0.5f + 0.5f * sample.cosine, 0.5f + 0.5f * axial_normal * sample.sine});
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
    struct BevelProfileSample {
      float radius = 0.0f;
      float offset = 0.0f;
      float radial_normal = 0.0f;
      float axial_normal = 0.0f;
    };
    auto bevel_profile_sample = [&](uint32_t sample_index, int32_t start_quarter_turn, int32_t end_quarter_turn, float center_radius, float center_offset, float start_radius,
                                  float start_offset, float end_radius, float end_offset) {
      const UnitCircleSample direction = unit_circle_arc_sample(sample_index, bevel_segments, start_quarter_turn, end_quarter_turn);
      const float radius = sample_index == 0u ? start_radius : (sample_index == bevel_segments ? end_radius : center_radius + bevel * direction.cosine);
      const float offset = sample_index == 0u ? start_offset : (sample_index == bevel_segments ? end_offset : center_offset + bevel * direction.sine);
      return BevelProfileSample{radius, offset, direction.cosine, direction.sine};
    };
    append_cap(half_thickness, 1.0f, top_inner_radius, top_outer_radius);

    if (bevel > 0.0f) {
      const float outer_center_radius = definition.radius - bevel;
      const float top_bevel_center = half_thickness - bevel;
      for (uint32_t i = 0u; i < bevel_segments; ++i) {
        const BevelProfileSample sample0 =
          bevel_profile_sample(i, 1, 0, outer_center_radius, top_bevel_center, top_outer_radius, half_thickness, definition.radius, top_bevel_center);
        const BevelProfileSample sample1 =
          bevel_profile_sample(i + 1u, 1, 0, outer_center_radius, top_bevel_center, top_outer_radius, half_thickness, definition.radius, top_bevel_center);
        append_strip(sample0.radius, sample0.offset, sample0.radial_normal, sample0.axial_normal, sample1.radius, sample1.offset, sample1.radial_normal, sample1.axial_normal);
      }
    }

    append_strip(definition.radius, half_thickness - bevel, 1.0f, 0.0f, definition.radius, -half_thickness + bevel, 1.0f, 0.0f);

    if (bevel > 0.0f) {
      const float outer_center_radius = definition.radius - bevel;
      const float bottom_bevel_center = -half_thickness + bevel;
      for (uint32_t i = 0u; i < bevel_segments; ++i) {
        const BevelProfileSample sample0 =
          bevel_profile_sample(i, 0, -1, outer_center_radius, bottom_bevel_center, definition.radius, bottom_bevel_center, top_outer_radius, -half_thickness);
        const BevelProfileSample sample1 =
          bevel_profile_sample(i + 1u, 0, -1, outer_center_radius, bottom_bevel_center, definition.radius, bottom_bevel_center, top_outer_radius, -half_thickness);
        append_strip(sample0.radius, sample0.offset, sample0.radial_normal, sample0.axial_normal, sample1.radius, sample1.offset, sample1.radial_normal, sample1.axial_normal);
      }
    }

    const float bottom_inner_radius = has_center_hole ? definition.inner_radius + bevel : 0.0f;
    append_cap(-half_thickness, -1.0f, bottom_inner_radius, top_outer_radius);

    if (has_center_hole) {
      if (bevel > 0.0f) {
        const float inner_center_radius = definition.inner_radius + bevel;
        const float bottom_bevel_center = -half_thickness + bevel;
        for (uint32_t i = 0u; i < bevel_segments; ++i) {
          const BevelProfileSample sample0 =
            bevel_profile_sample(i, -1, -2, inner_center_radius, bottom_bevel_center, bottom_inner_radius, -half_thickness, definition.inner_radius, bottom_bevel_center);
          const BevelProfileSample sample1 =
            bevel_profile_sample(i + 1u, -1, -2, inner_center_radius, bottom_bevel_center, bottom_inner_radius, -half_thickness, definition.inner_radius, bottom_bevel_center);
          append_strip(sample0.radius, sample0.offset, sample0.radial_normal, sample0.axial_normal, sample1.radius, sample1.offset, sample1.radial_normal, sample1.axial_normal);
        }
      }

      append_strip(definition.inner_radius, -half_thickness + bevel, -1.0f, 0.0f, definition.inner_radius, half_thickness - bevel, -1.0f, 0.0f);

      if (bevel > 0.0f) {
        const float inner_center_radius = definition.inner_radius + bevel;
        const float top_bevel_center = half_thickness - bevel;
        for (uint32_t i = 0u; i < bevel_segments; ++i) {
          const BevelProfileSample sample0 =
            bevel_profile_sample(i, 2, 1, inner_center_radius, top_bevel_center, definition.inner_radius, top_bevel_center, top_inner_radius, half_thickness);
          const BevelProfileSample sample1 =
            bevel_profile_sample(i + 1u, 2, 1, inner_center_radius, top_bevel_center, definition.inner_radius, top_bevel_center, top_inner_radius, half_thickness);
          append_strip(sample0.radius, sample0.offset, sample0.radial_normal, sample0.axial_normal, sample1.radius, sample1.offset, sample1.radial_normal, sample1.axial_normal);
        }
      }
    }
  }

  const uint32_t triangle_end = static_cast<uint32_t>(data.triangles.size());
  const uint32_t triangle_count = triangle_end - triangle_start;
  if ((geometry_valid == false) || (triangle_count == 0u)) {
    data.vertices.pos.resize(vertex_start);
    data.vertices.nrm.resize(vertex_start);
    data.vertices.tan.resize(vertex_start);
    data.vertices.btn.resize(vertex_start);
    data.vertices.tex.resize(vertex_start);
    data.triangles.resize(triangle_start);
    return false;
  }

  const std::string mesh_name = unique_mesh_name(data, definition.id, "disk");
  data.add_mesh(mesh_name.c_str(), triangle_start, triangle_count, bbox_min, bbox_max);
  return true;
}

bool append_box_mesh(SceneData& data, const ProceduralGeometryDefinition& definition) {
  if ((finite_float3(definition.center) == false) || (finite_float3(definition.dimensions) == false) || (definition.dimensions.x <= kEpsilon) ||
      (definition.dimensions.y <= kEpsilon) || (definition.dimensions.z <= kEpsilon)) {
    log::warning("Procedural box `%s` has invalid center or dimensions - skipped", definition.id.c_str());
    return false;
  }

  const uint32_t material_index = resolve_material_index(data, definition);
  if (material_index == kInvalidIndex) {
    log::warning("Procedural box `%s` has no valid material - skipped", definition.id.c_str());
    return false;
  }

  const float3 half_dimensions = 0.5f * definition.dimensions;
  const std::array<float3, 8u> vertices = {
    definition.center + float3{-half_dimensions.x, -half_dimensions.y, -half_dimensions.z},
    definition.center + float3{+half_dimensions.x, -half_dimensions.y, -half_dimensions.z},
    definition.center + float3{+half_dimensions.x, +half_dimensions.y, -half_dimensions.z},
    definition.center + float3{-half_dimensions.x, +half_dimensions.y, -half_dimensions.z},
    definition.center + float3{-half_dimensions.x, -half_dimensions.y, +half_dimensions.z},
    definition.center + float3{+half_dimensions.x, -half_dimensions.y, +half_dimensions.z},
    definition.center + float3{+half_dimensions.x, +half_dimensions.y, +half_dimensions.z},
    definition.center + float3{-half_dimensions.x, +half_dimensions.y, +half_dimensions.z},
  };
  static constexpr uint32_t faces[6][4] = {
    {0u, 1u, 2u, 3u},
    {4u, 7u, 6u, 5u},
    {0u, 4u, 5u, 1u},
    {3u, 2u, 6u, 7u},
    {0u, 3u, 7u, 4u},
    {1u, 5u, 6u, 2u},
  };

  BakedMeshBuilder builder(data, material_index);
  builder.reserve(24u, 12u);
  for (const auto& face : faces) {
    const float3 positions[4] = {
      vertices[face[0]],
      vertices[face[1]],
      vertices[face[2]],
      vertices[face[3]],
    };
    if (append_flat_polygon(builder, positions, 4u, definition.center) == false) {
      builder.rollback();
      return false;
    }
  }
  return builder.finish(definition, "box");
}

bool append_cone_mesh(SceneData& data, const ProceduralGeometryDefinition& definition) {
  const float height = definition.dimensions.y;
  const float axis_length = length(definition.normal);
  if ((finite_float3(definition.center) == false) || (std::isfinite(definition.radius) == false) || (definition.radius <= kEpsilon) || (std::isfinite(height) == false) ||
      (height <= kEpsilon) || (finite_float3(definition.normal) == false) || (std::isfinite(axis_length) == false) || (axis_length <= kEpsilon)) {
    log::warning("Procedural cone `%s` has invalid center, radius, height, or normal - skipped", definition.id.c_str());
    return false;
  }

  uint32_t segments = definition.segments;
  if (segments < kMinimumRoundSegments) {
    log::warning("Procedural cone `%s` segments %u is below minimum %u - clamped", definition.id.c_str(), segments, kMinimumRoundSegments);
    segments = kMinimumRoundSegments;
  } else if (segments > kMaximumRoundSegments) {
    log::warning("Procedural cone `%s` segments %u exceeds maximum %u - clamped", definition.id.c_str(), segments, kMaximumRoundSegments);
    segments = kMaximumRoundSegments;
  }

  const uint32_t material_index = resolve_material_index(data, definition);
  if (material_index == kInvalidIndex) {
    log::warning("Procedural cone `%s` has no valid material - skipped", definition.id.c_str());
    return false;
  }

  BakedMeshBuilder builder(data, material_index);
  builder.reserve(4u * segments + 2u, 2u * segments);
  const float half_height = 0.5f * height;
  const float3 axis = definition.normal / axis_length;
  const float3 radial_tangent = stable_tangent(axis);
  const float3 radial_bitangent = normalize(cross(radial_tangent, axis));
  const float3 apex = definition.center + half_height * axis;
  for (uint32_t segment = 0u; segment < segments; ++segment) {
    const UnitCircleSample sample0 = unit_circle_sample(segment, segments);
    const UnitCircleSample sample1 = unit_circle_sample(segment + 1u, segments);
    const float u0 = sample0.u;
    const float u1 = sample1.u;
    const float angle0 = kDoublePi * u0;
    const float angle1 = kDoublePi * u1;
    const float middle_angle = 0.5f * (angle0 + angle1);
    const float3 radial0 = sample0.cosine * radial_tangent + sample0.sine * radial_bitangent;
    const float3 radial1 = sample1.cosine * radial_tangent + sample1.sine * radial_bitangent;
    const float3 radial_middle = cosf(middle_angle) * radial_tangent + sinf(middle_angle) * radial_bitangent;
    const float3 normal0 = normalize(height * radial0 + definition.radius * axis);
    const float3 normal1 = normalize(height * radial1 + definition.radius * axis);
    const float3 normal_middle = normalize(height * radial_middle + definition.radius * axis);
    const float3 tangent0 = -sample0.sine * radial_tangent + sample0.cosine * radial_bitangent;
    const float3 tangent1 = -sample1.sine * radial_tangent + sample1.cosine * radial_bitangent;
    const float3 tangent_middle = -sinf(middle_angle) * radial_tangent + cosf(middle_angle) * radial_bitangent;
    const uint32_t base0 = builder.append_vertex(definition.center + definition.radius * radial0 - half_height * axis, normal0, tangent0, {u0, 1.0f});
    const uint32_t tip = builder.append_vertex(apex, normal_middle, tangent_middle, {0.5f * (u0 + u1), 0.0f});
    const uint32_t base1 = builder.append_vertex(definition.center + definition.radius * radial1 - half_height * axis, normal1, tangent1, {u1, 1.0f});
    if (builder.append_outward_triangle(base0, tip, base1, definition.center) == false) {
      builder.rollback();
      return false;
    }
  }

  const float3 cap_normal = -1.0f * axis;
  const uint32_t cap_center = builder.append_vertex(definition.center - half_height * axis, cap_normal, radial_tangent, {0.5f, 0.5f});
  const uint32_t cap_ring_start = static_cast<uint32_t>(data.vertices.pos.size());
  for (uint32_t segment = 0u; segment <= segments; ++segment) {
    const UnitCircleSample sample = unit_circle_sample(segment, segments);
    const float3 radial = sample.cosine * radial_tangent + sample.sine * radial_bitangent;
    const float3 position = definition.center + definition.radius * radial - half_height * axis;
    builder.append_vertex(position, cap_normal, radial_tangent, {0.5f + 0.5f * sample.cosine, 0.5f + 0.5f * sample.sine});
  }
  for (uint32_t segment = 0u; segment < segments; ++segment) {
    if (builder.append_outward_triangle(cap_center, cap_ring_start + segment, cap_ring_start + segment + 1u, definition.center) == false) {
      builder.rollback();
      return false;
    }
  }

  return builder.finish(definition, "cone");
}

bool append_capsule_mesh(SceneData& data, const ProceduralGeometryDefinition& definition) {
  const float height = definition.dimensions.y;
  const float axis_length = length(definition.normal);
  if ((finite_float3(definition.center) == false) || (std::isfinite(definition.radius) == false) || (definition.radius <= kEpsilon) || (std::isfinite(height) == false) ||
      (height < (2.0f * definition.radius)) || (finite_float3(definition.normal) == false) || (std::isfinite(axis_length) == false) || (axis_length <= kEpsilon)) {
    log::warning("Procedural capsule `%s` requires a valid normal, positive radius, and height at least twice the radius - skipped", definition.id.c_str());
    return false;
  }

  uint32_t segments = definition.segments;
  if (segments < kMinimumRoundSegments) {
    log::warning("Procedural capsule `%s` segments %u is below minimum %u - clamped", definition.id.c_str(), segments, kMinimumRoundSegments);
    segments = kMinimumRoundSegments;
  } else if (segments > kMaximumRoundSegments) {
    log::warning("Procedural capsule `%s` segments %u exceeds maximum %u - clamped", definition.id.c_str(), segments, kMaximumRoundSegments);
    segments = kMaximumRoundSegments;
  }
  uint32_t hemisphere_subdivisions = max(1u, definition.subdivisions);
  if (hemisphere_subdivisions > kMaximumRoundSubdivisions) {
    log::warning("Procedural capsule `%s` subdivisions %u exceeds maximum %u - clamped", definition.id.c_str(), hemisphere_subdivisions, kMaximumRoundSubdivisions);
    hemisphere_subdivisions = kMaximumRoundSubdivisions;
  }

  const uint32_t material_index = resolve_material_index(data, definition);
  if (material_index == kInvalidIndex) {
    log::warning("Procedural capsule `%s` has no valid material - skipped", definition.id.c_str());
    return false;
  }

  struct CapsuleRing {
    float y = 0.0f;
    float ring_radius = 0.0f;
    float normal_y = 0.0f;
    float normal_radius = 0.0f;
  };

  const float half_height = 0.5f * height;
  const float cylinder_half_height = max(0.0f, half_height - definition.radius);
  std::vector<CapsuleRing> rings;
  rings.reserve(2u * hemisphere_subdivisions);
  for (uint32_t ring = 1u; ring <= hemisphere_subdivisions; ++ring) {
    const float angle = -0.5f * kPi + 0.5f * kPi * float(ring) / float(hemisphere_subdivisions);
    const float cosine = max(0.0f, cosf(angle));
    rings.push_back({-cylinder_half_height + definition.radius * sinf(angle), definition.radius * cosine, sinf(angle), cosine});
  }
  if (cylinder_half_height > 0.0f) {
    rings.push_back({cylinder_half_height, definition.radius, 0.0f, 1.0f});
  }
  for (uint32_t ring = 1u; ring < hemisphere_subdivisions; ++ring) {
    const float angle = 0.5f * kPi * float(ring) / float(hemisphere_subdivisions);
    const float cosine = max(0.0f, cosf(angle));
    rings.push_back({cylinder_half_height + definition.radius * sinf(angle), definition.radius * cosine, sinf(angle), cosine});
  }

  BakedMeshBuilder builder(data, material_index);
  const float3 axis = definition.normal / axis_length;
  const float3 radial_tangent = stable_tangent(axis);
  const float3 radial_bitangent = normalize(cross(radial_tangent, axis));
  const uint32_t row_width = segments + 1u;
  const uint32_t pole_vertex_count = 2u * segments;
  const uint32_t strip_triangle_count = 2u * segments * (static_cast<uint32_t>(rings.size()) - 1u);
  builder.reserve(static_cast<uint32_t>(rings.size()) * row_width + pole_vertex_count, strip_triangle_count + pole_vertex_count);
  for (const CapsuleRing& ring : rings) {
    const float v = (half_height - ring.y) / height;
    for (uint32_t segment = 0u; segment <= segments; ++segment) {
      const UnitCircleSample sample = unit_circle_sample(segment, segments);
      const float3 radial = sample.cosine * radial_tangent + sample.sine * radial_bitangent;
      const float3 position = definition.center + ring.ring_radius * radial + ring.y * axis;
      const float3 normal = ring.normal_radius * radial + ring.normal_y * axis;
      const float3 tangent = -sample.sine * radial_tangent + sample.cosine * radial_bitangent;
      builder.append_vertex(position, normal, tangent, {sample.u, v});
    }
  }

  for (uint32_t row = 0u; row + 1u < rings.size(); ++row) {
    const uint32_t row_start = builder.vertex_start + row * row_width;
    const uint32_t next_row_start = row_start + row_width;
    for (uint32_t segment = 0u; segment < segments; ++segment) {
      const uint32_t current = row_start + segment;
      const uint32_t next_segment = current + 1u;
      const uint32_t next_row = next_row_start + segment;
      const uint32_t next_row_segment = next_row + 1u;
      const bool first_valid = builder.append_outward_triangle(current, next_row_segment, next_segment, definition.center);
      const bool second_valid = builder.append_outward_triangle(current, next_row, next_row_segment, definition.center);
      if ((first_valid == false) || (second_valid == false)) {
        builder.rollback();
        return false;
      }
    }
  }

  const uint32_t bottom_ring_start = builder.vertex_start;
  const uint32_t top_ring_start = builder.vertex_start + (static_cast<uint32_t>(rings.size()) - 1u) * row_width;
  for (uint32_t segment = 0u; segment < segments; ++segment) {
    const float u = (float(segment) + 0.5f) / float(segments);
    const float angle = kDoublePi * u;
    const float3 tangent = -sinf(angle) * radial_tangent + cosf(angle) * radial_bitangent;
    const uint32_t bottom_pole = builder.append_vertex(definition.center - half_height * axis, -1.0f * axis, tangent, {u, 1.0f});
    const uint32_t top_pole = builder.append_vertex(definition.center + half_height * axis, axis, tangent, {u, 0.0f});
    const bool bottom_valid = builder.append_outward_triangle(bottom_pole, bottom_ring_start + segment, bottom_ring_start + segment + 1u, definition.center);
    const bool top_valid = builder.append_outward_triangle(top_ring_start + segment, top_pole, top_ring_start + segment + 1u, definition.center);
    if ((bottom_valid == false) || (top_valid == false)) {
      builder.rollback();
      return false;
    }
  }

  return builder.finish(definition, "capsule");
}

bool append_torus_mesh(SceneData& data, const ProceduralGeometryDefinition& definition) {
  const float major_radius = definition.radius;
  const float minor_radius = definition.inner_radius;
  const float axis_length = length(definition.normal);
  if ((finite_float3(definition.center) == false) || (std::isfinite(major_radius) == false) || (std::isfinite(minor_radius) == false) || (minor_radius <= kEpsilon) ||
      (major_radius <= minor_radius) || (finite_float3(definition.normal) == false) || (std::isfinite(axis_length) == false) || (axis_length <= kEpsilon)) {
    log::warning("Procedural torus `%s` requires a valid normal and major radius greater than positive minor radius - skipped", definition.id.c_str());
    return false;
  }

  uint32_t major_segments = definition.segments;
  if (major_segments < kMinimumRoundSegments) {
    log::warning("Procedural torus `%s` segments %u is below minimum %u - clamped", definition.id.c_str(), major_segments, kMinimumRoundSegments);
    major_segments = kMinimumRoundSegments;
  } else if (major_segments > kMaximumRoundSegments) {
    log::warning("Procedural torus `%s` segments %u exceeds maximum %u - clamped", definition.id.c_str(), major_segments, kMaximumRoundSegments);
    major_segments = kMaximumRoundSegments;
  }
  uint32_t minor_segments = max(kMinimumRoundSegments, definition.subdivisions);
  if (minor_segments > kMaximumRoundSubdivisions) {
    log::warning("Procedural torus `%s` subdivisions %u exceeds maximum %u - clamped", definition.id.c_str(), minor_segments, kMaximumRoundSubdivisions);
    minor_segments = kMaximumRoundSubdivisions;
  }

  const uint32_t material_index = resolve_material_index(data, definition);
  if (material_index == kInvalidIndex) {
    log::warning("Procedural torus `%s` has no valid material - skipped", definition.id.c_str());
    return false;
  }

  BakedMeshBuilder builder(data, material_index);
  builder.reserve((major_segments + 1u) * (minor_segments + 1u), 2u * major_segments * minor_segments);
  const float3 axis = definition.normal / axis_length;
  const float3 radial_tangent = stable_tangent(axis);
  const float3 radial_bitangent = normalize(cross(radial_tangent, axis));
  const uint32_t row_width = minor_segments + 1u;
  for (uint32_t major_segment = 0u; major_segment <= major_segments; ++major_segment) {
    const UnitCircleSample major_sample = unit_circle_sample(major_segment, major_segments);
    const float3 radial = major_sample.cosine * radial_tangent + major_sample.sine * radial_bitangent;
    const float3 tangent = -major_sample.sine * radial_tangent + major_sample.cosine * radial_bitangent;
    for (uint32_t minor_segment = 0u; minor_segment <= minor_segments; ++minor_segment) {
      const UnitCircleSample minor_sample = unit_circle_sample(minor_segment, minor_segments);
      const float3 normal = minor_sample.cosine * radial + minor_sample.sine * axis;
      const float3 position = definition.center + major_radius * radial + minor_radius * normal;
      builder.append_vertex(position, normal, tangent, {major_sample.u, 1.0f - minor_sample.u});
    }
  }

  for (uint32_t major_segment = 0u; major_segment < major_segments; ++major_segment) {
    const uint32_t row_start = builder.vertex_start + major_segment * row_width;
    const uint32_t next_row_start = row_start + row_width;
    for (uint32_t minor_segment = 0u; minor_segment < minor_segments; ++minor_segment) {
      const uint32_t current = row_start + minor_segment;
      const uint32_t next_minor = current + 1u;
      const uint32_t next_major = next_row_start + minor_segment;
      const uint32_t next_major_minor = next_major + 1u;
      const bool first_valid = builder.append_triangle(current, next_minor, next_major_minor);
      const bool second_valid = builder.append_triangle(current, next_major_minor, next_major);
      if ((first_valid == false) || (second_valid == false)) {
        builder.rollback();
        return false;
      }
    }
  }

  return builder.finish(definition, "torus");
}

bool append_tetrahedron_mesh(SceneData& data, const ProceduralGeometryDefinition& definition) {
  const float inv_sqrt_three = 1.0f / sqrtf(3.0f);
  const std::vector<float3> vertices = {
    inv_sqrt_three * float3{1.0f, 1.0f, 1.0f},
    inv_sqrt_three * float3{-1.0f, -1.0f, 1.0f},
    inv_sqrt_three * float3{-1.0f, 1.0f, -1.0f},
    inv_sqrt_three * float3{1.0f, -1.0f, -1.0f},
  };
  const std::vector<IcosphereFace> faces = {
    {{0u, 1u, 2u}},
    {{0u, 3u, 1u}},
    {{0u, 2u, 3u}},
    {{1u, 3u, 2u}},
  };
  return append_flat_triangular_mesh(data, definition, vertices, faces, "tetrahedron");
}

bool append_octahedron_mesh(SceneData& data, const ProceduralGeometryDefinition& definition) {
  const std::vector<float3> vertices = {
    {0.0f, 1.0f, 0.0f},
    {0.0f, -1.0f, 0.0f},
    {1.0f, 0.0f, 0.0f},
    {0.0f, 0.0f, 1.0f},
    {-1.0f, 0.0f, 0.0f},
    {0.0f, 0.0f, -1.0f},
  };
  const std::vector<IcosphereFace> faces = {
    {{0u, 2u, 3u}},
    {{0u, 3u, 4u}},
    {{0u, 4u, 5u}},
    {{0u, 5u, 2u}},
    {{1u, 3u, 2u}},
    {{1u, 4u, 3u}},
    {{1u, 5u, 4u}},
    {{1u, 2u, 5u}},
  };
  return append_flat_triangular_mesh(data, definition, vertices, faces, "octahedron");
}

bool append_dodecahedron_mesh(SceneData& data, const ProceduralGeometryDefinition& definition) {
  if ((finite_float3(definition.center) == false) || (std::isfinite(definition.radius) == false) || (definition.radius <= kEpsilon)) {
    log::warning("Procedural dodecahedron `%s` has invalid center or radius - skipped", definition.id.c_str());
    return false;
  }

  const uint32_t material_index = resolve_material_index(data, definition);
  if (material_index == kInvalidIndex) {
    log::warning("Procedural dodecahedron `%s` has no valid material - skipped", definition.id.c_str());
    return false;
  }

  std::vector<float3> icosahedron_vertices;
  std::vector<IcosphereFace> icosahedron_faces;
  build_icosahedron(icosahedron_vertices, icosahedron_faces);
  std::vector<float3> dodecahedron_vertices;
  dodecahedron_vertices.reserve(icosahedron_faces.size());
  for (const IcosphereFace& face : icosahedron_faces) {
    dodecahedron_vertices.push_back(normalize(icosahedron_vertices[face.i[0]] + icosahedron_vertices[face.i[1]] + icosahedron_vertices[face.i[2]]));
  }

  BakedMeshBuilder builder(data, material_index);
  builder.reserve(60u, 36u);
  for (uint32_t icosahedron_vertex = 0u; icosahedron_vertex < icosahedron_vertices.size(); ++icosahedron_vertex) {
    std::array<uint32_t, 5u> incident_faces = {};
    uint32_t incident_count = 0u;
    for (uint32_t face_index = 0u; face_index < icosahedron_faces.size(); ++face_index) {
      const IcosphereFace& face = icosahedron_faces[face_index];
      if ((face.i[0] == icosahedron_vertex) || (face.i[1] == icosahedron_vertex) || (face.i[2] == icosahedron_vertex)) {
        if (incident_count >= incident_faces.size()) {
          builder.rollback();
          return false;
        }
        incident_faces[incident_count++] = face_index;
      }
    }
    if (incident_count != incident_faces.size()) {
      builder.rollback();
      return false;
    }

    const float3 face_axis = icosahedron_vertices[icosahedron_vertex];
    const float3 face_tangent = stable_tangent(face_axis);
    const float3 face_bitangent = normalize(cross(face_axis, face_tangent));
    std::sort(incident_faces.begin(), incident_faces.end(), [&](uint32_t lhs, uint32_t rhs) {
      const float lhs_angle = atan2f(dot(dodecahedron_vertices[lhs], face_bitangent), dot(dodecahedron_vertices[lhs], face_tangent));
      const float rhs_angle = atan2f(dot(dodecahedron_vertices[rhs], face_bitangent), dot(dodecahedron_vertices[rhs], face_tangent));
      return lhs_angle < rhs_angle;
    });

    float3 positions[5] = {};
    for (uint32_t i = 0u; i < incident_faces.size(); ++i) {
      positions[i] = definition.center + definition.radius * dodecahedron_vertices[incident_faces[i]];
    }
    if (append_flat_polygon(builder, positions, 5u, definition.center) == false) {
      builder.rollback();
      return false;
    }
  }

  return builder.finish(definition, "dodecahedron");
}

bool append_icosahedron_mesh(SceneData& data, const ProceduralGeometryDefinition& definition) {
  std::vector<float3> vertices;
  std::vector<IcosphereFace> faces;
  build_icosahedron(vertices, faces);
  return append_flat_triangular_mesh(data, definition, vertices, faces, "icosahedron");
}

}  // namespace

bool is_procedural_geometry_entry(const std::string& name) {
  return is_geometry_entry(name) || is_sphere_alias(name) || is_plane_alias(name) || is_disk_alias(name) || is_box_alias(name) || is_cone_alias(name) || is_capsule_alias(name) ||
         is_torus_alias(name) || is_tetrahedron_alias(name) || is_octahedron_alias(name) || is_dodecahedron_alias(name) || is_icosahedron_alias(name);
}

bool parse_procedural_geometry_definition(const MaterialDefinition& material, ProceduralGeometryDefinition& out_definition) {
  out_definition = {};

  if (is_sphere_alias(material.name)) {
    out_definition.cls = ProceduralGeometryDefinition::Class::Sphere;
  } else if (is_plane_alias(material.name)) {
    out_definition.cls = ProceduralGeometryDefinition::Class::Plane;
  } else if (is_disk_alias(material.name)) {
    out_definition.cls = ProceduralGeometryDefinition::Class::Disk;
  } else if (is_box_alias(material.name)) {
    out_definition.cls = ProceduralGeometryDefinition::Class::Box;
  } else if (is_cone_alias(material.name)) {
    out_definition.cls = ProceduralGeometryDefinition::Class::Cone;
  } else if (is_capsule_alias(material.name)) {
    out_definition.cls = ProceduralGeometryDefinition::Class::Capsule;
  } else if (is_torus_alias(material.name)) {
    out_definition.cls = ProceduralGeometryDefinition::Class::Torus;
  } else if (is_tetrahedron_alias(material.name)) {
    out_definition.cls = ProceduralGeometryDefinition::Class::Tetrahedron;
  } else if (is_octahedron_alias(material.name)) {
    out_definition.cls = ProceduralGeometryDefinition::Class::Octahedron;
  } else if (is_dodecahedron_alias(material.name)) {
    out_definition.cls = ProceduralGeometryDefinition::Class::Dodecahedron;
  } else if (is_icosahedron_alias(material.name)) {
    out_definition.cls = ProceduralGeometryDefinition::Class::Icosahedron;
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
    } else if (class_name == "box") {
      out_definition.cls = ProceduralGeometryDefinition::Class::Box;
    } else if (class_name == "cone") {
      out_definition.cls = ProceduralGeometryDefinition::Class::Cone;
    } else if (class_name == "capsule") {
      out_definition.cls = ProceduralGeometryDefinition::Class::Capsule;
    } else if (class_name == "torus") {
      out_definition.cls = ProceduralGeometryDefinition::Class::Torus;
    } else if (class_name == "tetrahedron") {
      out_definition.cls = ProceduralGeometryDefinition::Class::Tetrahedron;
    } else if (class_name == "octahedron") {
      out_definition.cls = ProceduralGeometryDefinition::Class::Octahedron;
    } else if (class_name == "dodecahedron") {
      out_definition.cls = ProceduralGeometryDefinition::Class::Dodecahedron;
    } else if (class_name == "icosahedron") {
      out_definition.cls = ProceduralGeometryDefinition::Class::Icosahedron;
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
  } else if (out_definition.cls == ProceduralGeometryDefinition::Class::Box) {
    out_definition.dimensions = {1.0f, 1.0f, 1.0f};
  } else if (out_definition.cls == ProceduralGeometryDefinition::Class::Cone) {
    out_definition.dimensions = {1.0f, 1.0f, 1.0f};
    out_definition.radius = 0.5f;
    out_definition.segments = 64u;
  } else if (out_definition.cls == ProceduralGeometryDefinition::Class::Capsule) {
    out_definition.dimensions = {1.0f, 2.0f, 1.0f};
    out_definition.radius = 0.5f;
    out_definition.segments = 64u;
    out_definition.subdivisions = 8u;
  } else if (out_definition.cls == ProceduralGeometryDefinition::Class::Torus) {
    out_definition.radius = 0.375f;
    out_definition.inner_radius = 0.125f;
    out_definition.segments = 64u;
    out_definition.subdivisions = 24u;
  } else if ((out_definition.cls == ProceduralGeometryDefinition::Class::Tetrahedron) || (out_definition.cls == ProceduralGeometryDefinition::Class::Octahedron) ||
             (out_definition.cls == ProceduralGeometryDefinition::Class::Dodecahedron) || (out_definition.cls == ProceduralGeometryDefinition::Class::Icosahedron)) {
    out_definition.radius = 0.5f;
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
  } else if ((out_definition.cls == ProceduralGeometryDefinition::Class::Sphere) || (out_definition.cls == ProceduralGeometryDefinition::Class::Plane)) {
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
  } else if (out_definition.cls == ProceduralGeometryDefinition::Class::Box) {
    read_float3_property(material, "dimensions", out_definition.dimensions);
  } else if ((out_definition.cls == ProceduralGeometryDefinition::Class::Cone) || (out_definition.cls == ProceduralGeometryDefinition::Class::Capsule)) {
    const bool has_dimensions = read_float3_property(material, "dimensions", out_definition.dimensions);
    bool has_radius = read_float_property(material, "radius", out_definition.radius);
    if (has_radius == false) {
      float diameter = 0.0f;
      if (read_float_property(material, "diameter", diameter)) {
        out_definition.radius = 0.5f * diameter;
        has_radius = true;
      }
    }
    if (has_dimensions && (has_radius == false)) {
      out_definition.radius = 0.5f * min(out_definition.dimensions.x, out_definition.dimensions.z);
    }
    float height = 0.0f;
    if (read_float_property(material, "height", height)) {
      out_definition.dimensions.y = height;
    }
    read_float3_property(material, "normal", out_definition.normal);
    read_uint32_property(material, "segments", out_definition.segments);
    if (out_definition.cls == ProceduralGeometryDefinition::Class::Capsule) {
      read_uint32_property(material, "subdivisions", out_definition.subdivisions);
    }
  } else if (out_definition.cls == ProceduralGeometryDefinition::Class::Torus) {
    if (read_float_property(material, "major-radius", out_definition.radius) == false) {
      if (read_float_property(material, "major_radius", out_definition.radius) == false) {
        read_float_property(material, "radius", out_definition.radius);
      }
    }
    if (read_float_property(material, "minor-radius", out_definition.inner_radius) == false) {
      if (read_float_property(material, "minor_radius", out_definition.inner_radius) == false) {
        if (read_float_property(material, "tube-radius", out_definition.inner_radius) == false) {
          if (read_float_property(material, "tube_radius", out_definition.inner_radius) == false) {
            if (read_float_property(material, "inner-radius", out_definition.inner_radius) == false) {
              read_float_property(material, "inner_radius", out_definition.inner_radius);
            }
          }
        }
      }
    }
    read_float3_property(material, "normal", out_definition.normal);
    read_uint32_property(material, "segments", out_definition.segments);
    if (read_uint32_property(material, "minor-segments", out_definition.subdivisions) == false) {
      if (read_uint32_property(material, "minor_segments", out_definition.subdivisions) == false) {
        read_uint32_property(material, "subdivisions", out_definition.subdivisions);
      }
    }
  } else {
    if (read_float_property(material, "radius", out_definition.radius) == false) {
      float diameter = 0.0f;
      if (read_float_property(material, "diameter", diameter)) {
        out_definition.radius = 0.5f * diameter;
      }
    }
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
    } else if (definition.cls == ProceduralGeometryDefinition::Class::Box) {
      if (append_box_mesh(data, definition)) {
        ++generated_count;
      }
    } else if (definition.cls == ProceduralGeometryDefinition::Class::Cone) {
      if (append_cone_mesh(data, definition)) {
        ++generated_count;
      }
    } else if (definition.cls == ProceduralGeometryDefinition::Class::Capsule) {
      if (append_capsule_mesh(data, definition)) {
        ++generated_count;
      }
    } else if (definition.cls == ProceduralGeometryDefinition::Class::Torus) {
      if (append_torus_mesh(data, definition)) {
        ++generated_count;
      }
    } else if (definition.cls == ProceduralGeometryDefinition::Class::Tetrahedron) {
      if (append_tetrahedron_mesh(data, definition)) {
        ++generated_count;
      }
    } else if (definition.cls == ProceduralGeometryDefinition::Class::Octahedron) {
      if (append_octahedron_mesh(data, definition)) {
        ++generated_count;
      }
    } else if (definition.cls == ProceduralGeometryDefinition::Class::Dodecahedron) {
      if (append_dodecahedron_mesh(data, definition)) {
        ++generated_count;
      }
    } else if (definition.cls == ProceduralGeometryDefinition::Class::Icosahedron) {
      if (append_icosahedron_mesh(data, definition)) {
        ++generated_count;
      }
    }
  }
  return generated_count;
}

}  // namespace etx
