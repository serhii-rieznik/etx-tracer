#include <etx/std.hxx>
#include "scene_mesh_loader.hxx"
#include "scene_mesh_input.hxx"
#include <numeric>
#include <sstream>
#include <unordered_set>

namespace etx {
namespace {

enum class PlyType { Int8, UInt8, Int16, UInt16, Int32, UInt32, Float32, Float64 };

PlyType ply_type(const std::string& name) {
  if ((name == "char") || (name == "int8"))
    return PlyType::Int8;
  if ((name == "uchar") || (name == "uint8"))
    return PlyType::UInt8;
  if ((name == "short") || (name == "int16"))
    return PlyType::Int16;
  if ((name == "ushort") || (name == "uint16"))
    return PlyType::UInt16;
  if ((name == "int") || (name == "int32"))
    return PlyType::Int32;
  if ((name == "uint") || (name == "uint32"))
    return PlyType::UInt32;
  if ((name == "float") || (name == "float32"))
    return PlyType::Float32;
  if ((name == "double") || (name == "float64"))
    return PlyType::Float64;
  throw std::runtime_error("Unsupported PLY property type: " + name);
}

size_t width(PlyType type) {
  constexpr size_t widths[] = {1u, 1u, 2u, 2u, 4u, 4u, 4u, 8u};
  return widths[static_cast<size_t>(type)];
}

bool integral(PlyType type) {
  return (type != PlyType::Float32) && (type != PlyType::Float64);
}

enum class Attribute { Skip, X, Y, Z, NX, NY, NZ, U, V, Indices, UV };

struct Property {
  std::string name;
  PlyType type;
  PlyType count_type = PlyType::UInt8;
  Attribute attribute = Attribute::Skip;
  bool list = false;
};

struct Element {
  std::string name;
  uint32_t count = 0u;
  std::vector<Property> properties;
};

struct PlyReader {
  explicit PlyReader(const std::filesystem::path& path)
    : input(path) {
  }

  double scalar(PlyType type) {
    double value = 0.0;
    if (ascii) {
      value = input.number();
      constexpr double lower[] = {-128.0, 0.0, -32768.0, 0.0, -2147483648.0, 0.0, -double(std::numeric_limits<float>::max()), -std::numeric_limits<double>::max()};
      constexpr double upper[] = {127.0, 255.0, 32767.0, 65535.0, 2147483647.0, 4294967295.0, double(std::numeric_limits<float>::max()), std::numeric_limits<double>::max()};
      const auto index = static_cast<size_t>(type);
      if ((value < lower[index]) || (value > upper[index]) || (integral(type) && (std::floor(value) != value)))
        throw std::runtime_error("ASCII PLY value does not match its declared type.");
    } else {
      switch (type) {
        case PlyType::Int8:
          value = input.binary<int8_t>(big_endian);
          break;
        case PlyType::UInt8:
          value = input.binary<uint8_t>(big_endian);
          break;
        case PlyType::Int16:
          value = input.binary<int16_t>(big_endian);
          break;
        case PlyType::UInt16:
          value = input.binary<uint16_t>(big_endian);
          break;
        case PlyType::Int32:
          value = input.binary<int32_t>(big_endian);
          break;
        case PlyType::UInt32:
          value = input.binary<uint32_t>(big_endian);
          break;
        case PlyType::Float32:
          value = input.binary<float>(big_endian);
          break;
        case PlyType::Float64:
          value = input.binary<double>(big_endian);
          break;
      }
      if (std::isfinite(value) == false)
        throw std::runtime_error("Non-finite PLY value.");
    }
    return value;
  }

  uint32_t integer(PlyType type) {
    const double value = scalar(type);
    if ((value < 0.0) || (value > std::numeric_limits<uint32_t>::max()) || (std::floor(value) != value))
      throw std::runtime_error("Invalid PLY index or list count.");
    return static_cast<uint32_t>(value);
  }

  void skip(PlyType type, uint32_t count) {
    if (ascii) {
      for (uint32_t index = 0u; index < count; ++index)
        scalar(type);
    } else {
      input.read(nullptr, size_t(count) * width(type));
    }
  }

  MeshInput input;
  bool ascii = false;
  bool big_endian = false;
};

struct Point2 {
  double x, y;
};

double orientation(const Point2& a, const Point2& b, const Point2& c) {
  return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
}

bool on_segment(const Point2& a, const Point2& b, const Point2& p) {
  return (orientation(a, b, p) == 0.0) && (p.x >= std::min(a.x, b.x)) && (p.x <= std::max(a.x, b.x)) && (p.y >= std::min(a.y, b.y)) && (p.y <= std::max(a.y, b.y));
}

void triangulate(const std::vector<uint32_t>& face, const std::vector<float3>& positions, std::vector<uint3>& triangles, std::vector<Point2>& points,
  std::vector<uint32_t>& polygon) {
  if (face.size() == 3u) {
    triangles.push_back({face[0], face[1], face[2]});
    return;
  }
  double normal[3] = {};
  for (size_t index = 0u; index < face.size(); ++index) {
    const auto& a = positions[face[index]];
    const auto& b = positions[face[(index + 1u) % face.size()]];
    normal[0] += (double(a.y) - b.y) * (double(a.z) + b.z);
    normal[1] += (double(a.z) - b.z) * (double(a.x) + b.x);
    normal[2] += (double(a.x) - b.x) * (double(a.y) + b.y);
  }
  const size_t axis =
    (std::abs(normal[0]) > std::abs(normal[1])) ? ((std::abs(normal[0]) > std::abs(normal[2])) ? 0u : 2u) : ((std::abs(normal[1]) > std::abs(normal[2])) ? 1u : 2u);
  if (normal[axis] == 0.0)
    throw std::runtime_error("PLY polygon has no unambiguous orientation.");
  points.clear();
  for (uint32_t index : face) {
    const auto& p = positions[index];
    points.push_back(axis == 0u ? Point2{p.y, p.z} : (axis == 1u ? Point2{p.z, p.x} : Point2{p.x, p.y}));
  }
  for (size_t i = 0u; i < points.size(); ++i) {
    const size_t next_i = (i + 1u) % points.size();
    for (size_t j = i + 1u; j < points.size(); ++j) {
      const size_t next_j = (j + 1u) % points.size();
      if ((next_i == j) || (next_j == i))
        continue;
      const auto& a = points[i];
      const auto& b = points[next_i];
      const auto& c = points[j];
      const auto& d = points[next_j];
      const double ab_c = orientation(a, b, c), ab_d = orientation(a, b, d);
      const double cd_a = orientation(c, d, a), cd_b = orientation(c, d, b);
      if ((((ab_c > 0.0) != (ab_d > 0.0)) && ((cd_a > 0.0) != (cd_b > 0.0))) || on_segment(a, b, c) || on_segment(a, b, d) || on_segment(c, d, a) || on_segment(c, d, b))
        throw std::runtime_error("PLY polygon has intersecting edges or repeated vertices.");
    }
  }
  const double sign = normal[axis] > 0.0 ? 1.0 : -1.0;
  bool convex = true;
  for (size_t index = 0u; index < points.size(); ++index) {
    if ((sign * orientation(points[index], points[(index + 1u) % points.size()], points[(index + 2u) % points.size()])) < 0.0) {
      convex = false;
      break;
    }
  }
  if (convex) {
    for (size_t index = 2u; index < face.size(); ++index)
      triangles.push_back({face[0], face[index - 1u], face[index]});
    return;
  }
  polygon.resize(face.size());
  std::iota(polygon.begin(), polygon.end(), 0u);
  while (polygon.size() > 3u) {
    bool clipped = false;
    for (size_t index = 0u; index < polygon.size(); ++index) {
      const uint32_t a = polygon[(index + polygon.size() - 1u) % polygon.size()];
      const uint32_t b = polygon[index];
      const uint32_t c = polygon[(index + 1u) % polygon.size()];
      if ((sign * orientation(points[a], points[b], points[c])) <= 0.0)
        continue;
      bool contains = false;
      for (uint32_t other : polygon) {
        if ((other == a) || (other == b) || (other == c))
          continue;
        if (((sign * orientation(points[a], points[b], points[other])) >= 0.0) && ((sign * orientation(points[b], points[c], points[other])) >= 0.0) &&
            ((sign * orientation(points[c], points[a], points[other])) >= 0.0)) {
          contains = true;
          break;
        }
      }
      if (contains)
        continue;
      triangles.push_back({face[a], face[b], face[c]});
      polygon.erase(polygon.begin() + index);
      clipped = true;
      break;
    }
    if (clipped == false)
      throw std::runtime_error("PLY polygon cannot be triangulated; check for intersecting edges or repeated vertices.");
  }
  triangles.push_back({face[polygon[0]], face[polygon[1]], face[polygon[2]]});
}

}  // namespace

ImportedMesh load_ply_mesh(const std::filesystem::path& path) {
  PlyReader reader(path);
  if (reader.input.line() != "ply")
    throw std::runtime_error("Invalid PLY signature.");
  std::vector<Element> elements;
  std::unordered_set<std::string> names;
  bool format_seen = false, header_done = false;
  ImportedMesh mesh;
  while (reader.input.remaining() > 0u) {
    const std::string line = reader.input.line();
    std::istringstream tokens(line);
    std::string key, extra;
    tokens >> key;
    if ((key == "comment") || (key == "obj_info")) {
      if (line.find("TextureFile") != std::string::npos)
        mesh.omitted_textures = true;
      continue;
    }
    if (key == "format") {
      std::string format, version;
      if (((tokens >> format >> version).fail()) || format_seen || (version != "1.0") ||
          ((format != "ascii") && (format != "binary_little_endian") && (format != "binary_big_endian")))
        throw std::runtime_error("Unsupported or invalid PLY format.");
      reader.ascii = format == "ascii";
      reader.big_endian = format == "binary_big_endian";
      format_seen = true;
    } else if (key == "element") {
      Element element;
      uint64_t count;
      if (((tokens >> element.name >> count).fail()) || (count > reader.input.size()) || (count > std::numeric_limits<uint32_t>::max()) ||
          (names.insert(element.name).second == false))
        throw std::runtime_error("Invalid PLY element declaration.");
      element.count = static_cast<uint32_t>(count);
      elements.push_back(std::move(element));
    } else if (key == "property") {
      if (elements.empty())
        throw std::runtime_error("PLY property has no element.");
      Property property;
      std::string type;
      tokens >> type;
      if (type == "list") {
        std::string count_type;
        tokens >> count_type >> type >> property.name;
        property.count_type = ply_type(count_type);
        property.list = true;
        if (integral(property.count_type) == false)
          throw std::runtime_error("PLY list counts must use an integer type.");
      } else {
        tokens >> property.name;
      }
      if ((tokens.fail()) || (property.name.empty()))
        throw std::runtime_error("Invalid PLY property declaration.");
      property.type = ply_type(type);
      auto& properties = elements.back().properties;
      if (std::any_of(properties.begin(), properties.end(), [&](const auto& other) {
            return property.name == other.name;
          }))
        throw std::runtime_error("Duplicate PLY property: " + property.name);
      properties.push_back(std::move(property));
    } else if (key == "end_header") {
      header_done = true;
    } else {
      throw std::runtime_error("Unknown PLY header declaration: " + key);
    }
    if (tokens >> extra)
      throw std::runtime_error("Unexpected PLY header token: " + extra);
    if (header_done)
      break;
  }
  if ((format_seen == false) || (header_done == false) || (names.contains("vertex") == false) || (names.contains("face") == false))
    throw std::runtime_error("PLY requires a complete header, vertices, and faces.");
  uint64_t minimum_size = 0u;
  bool face_uv = false;
  for (const auto& element : elements) {
    uint64_t row_size = 0u;
    for (const auto& property : element.properties) {
      row_size += reader.ascii ? 1u : width(property.list ? property.count_type : property.type);
      if ((element.name == "face") && property.list) {
        if ((property.name == "vertex_indices") || (property.name == "vertex_index"))
          row_size += 3u * (reader.ascii ? 1u : width(property.type));
        else if (property.name == "texcoord")
          row_size += 6u * (reader.ascii ? 1u : width(property.type));
      }
    }
    if ((element.count > 0u) && (row_size == 0u))
      throw std::runtime_error("PLY element has no properties.");
    if ((row_size * element.count) > (reader.input.remaining() - minimum_size))
      throw std::runtime_error("PLY element counts exceed the available payload.");
    minimum_size += row_size * element.count;
  }
  for (auto& element : elements) {
    const auto find = [&](std::string_view name) -> Property* {
      for (auto& property : element.properties) {
        if (property.name == name)
          return &property;
      }
      return nullptr;
    };
    const auto bind = [&](const char* name, Attribute attribute) {
      auto* property = find(name);
      if ((property == nullptr) || property->list)
        throw std::runtime_error("Missing scalar PLY vertex attribute: " + std::string(name));
      property->attribute = attribute;
    };
    if (element.name == "vertex") {
      bind("x", Attribute::X);
      bind("y", Attribute::Y);
      bind("z", Attribute::Z);
      mesh.positions.resize(element.count);
      if ((find("nx") != nullptr) || (find("ny") != nullptr) || (find("nz") != nullptr)) {
        bind("nx", Attribute::NX);
        bind("ny", Attribute::NY);
        bind("nz", Attribute::NZ);
        mesh.normals.resize(element.count);
      }
      for (const auto& pair : {std::pair{"u", "v"}, std::pair{"s", "t"}, std::pair{"texture_u", "texture_v"}}) {
        if ((find(pair.first) != nullptr) || (find(pair.second) != nullptr)) {
          bind(pair.first, Attribute::U);
          bind(pair.second, Attribute::V);
          mesh.texcoords.resize(element.count);
          break;
        }
      }
    } else if (element.name == "face") {
      auto* indices = find("vertex_indices");
      if (indices == nullptr)
        indices = find("vertex_index");
      if ((indices == nullptr) || (indices->list == false) || (integral(indices->type) == false))
        throw std::runtime_error("PLY faces require an integer vertex-index list.");
      indices->attribute = Attribute::Indices;
      if (auto* uv = find("texcoord")) {
        if (uv->list == false)
          throw std::runtime_error("PLY face texture coordinates must be a list.");
        uv->attribute = Attribute::UV;
        face_uv = true;
      }
      mesh.indices.reserve(element.count);
    }
    for (const auto& property : element.properties) {
      if ((property.name == "red") || (property.name == "green") || (property.name == "blue") || (property.name == "diffuse_red") || (property.name == "diffuse_green") ||
          (property.name == "diffuse_blue"))
        mesh.omitted_colors = true;
    }
  }
  std::vector<uint32_t> corners, face_ends, face;
  std::vector<float2> corner_uv;
  std::vector<float> uv;
  for (const auto& element : elements) {
    if (element.name == "face") {
      face_ends.reserve(element.count);
      corners.reserve(size_t(element.count) * 3u);
    }
    for (uint32_t row = 0u; row < element.count; ++row) {
      face.clear();
      uv.clear();
      for (const auto& property : element.properties) {
        if (property.list) {
          const uint32_t count = reader.integer(property.count_type);
          if (count > (reader.input.remaining() / (reader.ascii ? 1u : width(property.type))))
            throw std::runtime_error("PLY list exceeds the available payload.");
          if (property.attribute == Attribute::Indices) {
            face.reserve(count);
            for (uint32_t index = 0u; index < count; ++index)
              face.push_back(reader.integer(property.type));
          } else if (property.attribute == Attribute::UV) {
            uv.reserve(count);
            for (uint32_t index = 0u; index < count; ++index)
              uv.push_back(mesh_float(reader.scalar(property.type)));
          } else {
            reader.skip(property.type, count);
          }
        } else if (property.attribute == Attribute::Skip) {
          reader.skip(property.type, 1u);
        } else {
          const float value = mesh_float(reader.scalar(property.type));
          switch (property.attribute) {
            case Attribute::X:
              mesh.positions[row].x = value;
              break;
            case Attribute::Y:
              mesh.positions[row].y = value;
              break;
            case Attribute::Z:
              mesh.positions[row].z = value;
              break;
            case Attribute::NX:
              mesh.normals[row].x = value;
              break;
            case Attribute::NY:
              mesh.normals[row].y = value;
              break;
            case Attribute::NZ:
              mesh.normals[row].z = value;
              break;
            case Attribute::U:
              mesh.texcoords[row].x = value;
              break;
            case Attribute::V:
              mesh.texcoords[row].y = value;
              break;
            default:
              break;
          }
        }
      }
      if (element.name == "face") {
        if ((face.size() < 3u) || (face.size() > (std::numeric_limits<uint32_t>::max() - corners.size())) || (face_uv && (uv.size() != (2u * face.size()))))
          throw std::runtime_error("Invalid PLY face or face texture-coordinate count.");
        for (uint32_t index : face) {
          if (index >= mesh.positions.size())
            throw std::runtime_error("PLY face references an invalid vertex.");
        }
        corners.insert(corners.end(), face.begin(), face.end());
        face_ends.push_back(static_cast<uint32_t>(corners.size()));
        for (size_t index = 0u; index < uv.size(); index += 2u)
          corner_uv.push_back({uv[index], uv[index + 1u]});
      }
    }
  }
  if (reader.ascii) {
    if (reader.input.token().empty() == false)
      throw std::runtime_error("Unexpected trailing ASCII PLY data.");
  } else if (reader.input.remaining() != 0u) {
    throw std::runtime_error("Unexpected trailing binary PLY data.");
  }
  if (face_uv) {
    ImportedMesh expanded;
    expanded.omitted_colors = mesh.omitted_colors;
    expanded.omitted_textures = mesh.omitted_textures;
    expanded.positions.reserve(corners.size());
    if (mesh.normals.empty() == false)
      expanded.normals.reserve(corners.size());
    expanded.texcoords = std::move(corner_uv);
    for (auto& index : corners) {
      expanded.positions.push_back(mesh.positions[index]);
      if (mesh.normals.empty() == false)
        expanded.normals.push_back(mesh.normals[index]);
      index = static_cast<uint32_t>(expanded.positions.size() - 1u);
    }
    mesh = std::move(expanded);
  }
  std::vector<Point2> points;
  std::vector<uint32_t> polygon;
  const size_t triangle_count = corners.size() - 2u * face_ends.size();
  if (triangle_count > std::numeric_limits<uint32_t>::max())
    throw std::runtime_error("PLY triangulation exceeds native index capacity.");
  mesh.indices.reserve(triangle_count);
  uint32_t begin = 0u;
  for (uint32_t end : face_ends) {
    face.assign(corners.begin() + begin, corners.begin() + end);
    if ((face.size() - 2u) > (std::numeric_limits<uint32_t>::max() - mesh.indices.size()))
      throw std::runtime_error("PLY triangulation exceeds native index capacity.");
    triangulate(face, mesh.positions, mesh.indices, points, polygon);
    begin = end;
  }
  if ((mesh.positions.empty()) || (mesh.indices.empty()))
    throw std::runtime_error("PLY contains no triangle geometry.");
  return mesh;
}

}  // namespace etx
