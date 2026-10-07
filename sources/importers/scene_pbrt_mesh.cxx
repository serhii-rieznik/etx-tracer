#include <etx/std.hxx>
#include <etx/core/core.hxx>
#include "scene_pbrt_mesh.hxx"
#include "scene_pbrt_input.hxx"

#include <bit>
#include <array>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>

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

struct Property {
  std::string name;
  PlyType type;
  PlyType count_type = PlyType::UInt8;
  bool list = false;
};

struct Element {
  std::string name;
  size_t count = 0u;
  std::vector<Property> properties;
};

struct PlyReader {
  explicit PlyReader(const std::filesystem::path& path)
    : input(is_pbrt_gzip_file(path) ? static_cast<std::istream&>(decompressed) : file) {
    if (is_pbrt_gzip_file(path)) {
      auto contents = read_pbrt_gzip(path);
      file_size = contents.size();
      decompressed.str(std::move(contents));
    } else {
      file.open(path, std::ios::binary | std::ios::ate);
      if (file.is_open() == false)
        throw std::runtime_error("Cannot open PLY mesh: " + path_to_utf8(path));
      const auto size = file.tellg();
      if (size < 0)
        throw std::runtime_error("Cannot determine PLY input size.");
      file_size = static_cast<size_t>(size);
      file.seekg(0);
    }
  }

  template <class T>
  double binary_value() {
    std::array<uint8_t, sizeof(T)> bytes;
    if (input.read(reinterpret_cast<char*>(bytes.data()), bytes.size()).fail())
      throw std::runtime_error("Truncated PLY mesh.");
    if (big_endian != (std::endian::native == std::endian::big))
      std::reverse(bytes.begin(), bytes.end());
    return static_cast<double>(std::bit_cast<T>(bytes));
  }

  double scalar(PlyType type) {
    double value = 0.0;
    if (ascii) {
      if ((input >> value).fail())
        throw std::runtime_error("Invalid or truncated ASCII PLY data.");
    } else {
      switch (type) {
        case PlyType::Int8:
          value = binary_value<int8_t>();
          break;
        case PlyType::UInt8:
          value = binary_value<uint8_t>();
          break;
        case PlyType::Int16:
          value = binary_value<int16_t>();
          break;
        case PlyType::UInt16:
          value = binary_value<uint16_t>();
          break;
        case PlyType::Int32:
          value = binary_value<int32_t>();
          break;
        case PlyType::UInt32:
          value = binary_value<uint32_t>();
          break;
        case PlyType::Float32:
          value = binary_value<float>();
          break;
        case PlyType::Float64:
          value = binary_value<double>();
          break;
      }
    }
    if (std::isfinite(value) == false)
      throw std::runtime_error("Non-finite PLY value.");
    return value;
  }

  size_t integer(PlyType type) {
    const double value = scalar(type);
    if ((value < 0.0) || (value > std::numeric_limits<uint32_t>::max()) || (std::floor(value) != value))
      throw std::runtime_error("Invalid PLY index or list size.");
    return static_cast<size_t>(value);
  }

  std::ifstream file;
  std::istringstream decompressed;
  std::istream& input;
  size_t file_size = 0u;
  bool ascii = false;
  bool big_endian = false;
};

}  // namespace

PbrtMesh load_pbrt_ply(const std::filesystem::path& path) {
  PlyReader reader(path);
  std::string line;
  if ((std::getline(reader.input, line).fail()) || ((line != "ply") && (line != "ply\r")))
    throw std::runtime_error("Invalid PLY signature: " + path_to_utf8(path));
  std::vector<Element> elements;
  bool format_seen = false;
  bool header_done = false;
  while (std::getline(reader.input, line)) {
    std::istringstream tokens(line);
    std::string key;
    tokens >> key;
    if (key == "format") {
      std::string format, version;
      tokens >> format >> version;
      if ((version != "1.0") || ((format != "ascii") && (format != "binary_little_endian") && (format != "binary_big_endian")))
        throw std::runtime_error("Unsupported PLY format.");
      reader.ascii = format == "ascii";
      reader.big_endian = format == "binary_big_endian";
      format_seen = true;
    } else if (key == "element") {
      Element element;
      if ((tokens >> element.name >> element.count).fail() || (element.count > reader.file_size) || (element.count > std::numeric_limits<uint32_t>::max()))
        throw std::runtime_error("Invalid PLY element count.");
      elements.emplace_back(std::move(element));
    } else if (key == "property") {
      if (elements.empty())
        throw std::runtime_error("PLY property has no element.");
      std::string type, name;
      tokens >> type;
      Property property;
      if (type == "list") {
        tokens >> type;
        property.count_type = ply_type(type);
        tokens >> type;
        property.list = true;
      }
      if ((tokens >> property.name).fail())
        throw std::runtime_error("Invalid PLY property declaration.");
      property.type = ply_type(type);
      elements.back().properties.emplace_back(std::move(property));
    } else if (key == "end_header") {
      header_done = true;
      break;
    } else if ((key != "comment") && (key != "obj_info") && (key.empty() == false))
      throw std::runtime_error("Unknown PLY header declaration: " + key);
  }
  if ((format_seen == false) || (header_done == false))
    throw std::runtime_error("Incomplete PLY header.");

  PbrtMesh mesh;
  for (const auto& element : elements) {
    const auto has = [&](const char* name) {
      return std::any_of(element.properties.begin(), element.properties.end(), [&](const Property& property) {
        return (property.name == name) && (property.list == false);
      });
    };
    const bool vertex = element.name == "vertex";
    const bool has_normals = has("nx") && has("ny") && has("nz");
    const bool has_uv = (has("u") && has("v")) || (has("s") && has("t")) || (has("texture_u") && has("texture_v"));
    if (vertex) {
      if ((has("x") && has("y") && has("z")) == false)
        throw std::runtime_error("PLY vertices require x, y, and z.");
      mesh.positions.reserve(element.count);
      if (has_normals)
        mesh.normals.reserve(element.count);
      if (has_uv)
        mesh.texcoords.reserve(element.count);
    }
    if (element.name == "face")
      mesh.indices.reserve(element.count);
    for (size_t row = 0u; row < element.count; ++row) {
      float3 position = {}, normal = {};
      float2 uv = {};
      std::vector<uint32_t> face;
      for (const auto& property : element.properties) {
        if (property.list) {
          const size_t count = reader.integer(property.count_type);
          if (count > reader.file_size)
            throw std::runtime_error("Invalid PLY list size.");
          const bool indices = (element.name == "face") && ((property.name == "vertex_indices") || (property.name == "vertex_index"));
          if (indices)
            face.reserve(count);
          for (size_t item = 0u; item < count; ++item) {
            if (indices)
              face.push_back(static_cast<uint32_t>(reader.integer(property.type)));
            else
              reader.scalar(property.type);
          }
        } else {
          const double raw = reader.scalar(property.type);
          if (std::abs(raw) > std::numeric_limits<float>::max())
            throw std::runtime_error("PLY attribute exceeds native floating-point range.");
          const float value = static_cast<float>(raw);
          if (vertex) {
            if (property.name == "x")
              position.x = value;
            else if (property.name == "y")
              position.y = value;
            else if (property.name == "z")
              position.z = value;
            else if (property.name == "nx")
              normal.x = value;
            else if (property.name == "ny")
              normal.y = value;
            else if (property.name == "nz")
              normal.z = value;
            else if ((property.name == "u") || (property.name == "s") || (property.name == "texture_u"))
              uv.x = value;
            else if ((property.name == "v") || (property.name == "t") || (property.name == "texture_v"))
              uv.y = value;
          }
        }
      }
      if (vertex) {
        mesh.positions.push_back(position);
        if (has_normals)
          mesh.normals.push_back(normal);
        if (has_uv)
          mesh.texcoords.push_back(uv);
      } else if (element.name == "face") {
        if ((face.size() != 3u) && (face.size() != 4u))
          throw std::runtime_error("PBRT PLY faces require three or four vertex indices.");
        for (size_t item = 2u; item < face.size(); ++item)
          mesh.indices.push_back({face[0], face[item - 1u], face[item]});
      }
    }
  }
  for (const auto& triangle : mesh.indices) {
    if ((triangle.x >= mesh.positions.size()) || (triangle.y >= mesh.positions.size()) || (triangle.z >= mesh.positions.size()))
      throw std::runtime_error("PLY face references an invalid vertex.");
  }
  return mesh;
}

void discard_degenerate_pbrt_triangles(PbrtMesh& mesh) {
  for (const auto& values : {std::cref(mesh.positions), std::cref(mesh.normals)}) {
    for (const float3& value : values.get()) {
      if ((std::isfinite(value.x) == false) || (std::isfinite(value.y) == false) || (std::isfinite(value.z) == false))
        throw std::runtime_error("Non-finite mesh position or normal.");
    }
  }
  for (const float2& uv : mesh.texcoords) {
    if ((std::isfinite(uv.x) == false) || (std::isfinite(uv.y) == false))
      throw std::runtime_error("Non-finite mesh UV.");
  }
  const size_t triangle_count = mesh.indices.size();
  std::erase_if(mesh.indices, [&](const uint3& triangle) {
    if ((triangle.x >= mesh.positions.size()) || (triangle.y >= mesh.positions.size()) || (triangle.z >= mesh.positions.size()))
      throw std::runtime_error("Mesh triangle references an invalid vertex.");
    const float3 normal = cross(mesh.positions[triangle.y] - mesh.positions[triangle.x], mesh.positions[triangle.z] - mesh.positions[triangle.x]);
    if ((std::isfinite(normal.x) == false) || (std::isfinite(normal.y) == false) || (std::isfinite(normal.z) == false) || (std::isfinite(dot(normal, normal)) == false))
      throw std::runtime_error("Mesh triangle exceeds the native floating-point range.");
    return dot(normal, normal) == 0.0f;
  });
  if (mesh.indices.size() != triangle_count) {
    std::vector<uint32_t> remap(mesh.positions.size(), kInvalidIndex);
    for (const uint3& triangle : mesh.indices) {
      remap[triangle.x] = remap[triangle.y] = remap[triangle.z] = 0u;
    }
    uint32_t count = 0u;
    for (size_t index = 0u; index < remap.size(); ++index) {
      if (remap[index] == kInvalidIndex)
        continue;
      remap[index] = count;
      mesh.positions[count] = mesh.positions[index];
      if (mesh.normals.empty() == false)
        mesh.normals[count] = mesh.normals[index];
      if (mesh.texcoords.empty() == false)
        mesh.texcoords[count] = mesh.texcoords[index];
      ++count;
    }
    for (uint3& triangle : mesh.indices)
      triangle = {remap[triangle.x], remap[triangle.y], remap[triangle.z]};
    mesh.positions.resize(count);
    if (mesh.normals.empty() == false)
      mesh.normals.resize(count);
    if (mesh.texcoords.empty() == false)
      mesh.texcoords.resize(count);
  }
}

}  // namespace etx
