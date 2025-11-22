#include <etx/render/host/scene_serialization.hxx>

#include <etx/core/core.hxx>
#include <etx/core/debug.hxx>
#include <etx/core/environment.hxx>
#include <etx/core/platform.hxx>
#include <etx/core/log.hxx>
#include <etx/render/host/scene_data.hxx>
#include <etx/render/host/scene_representation.hxx>
#include <etx/render/shared/base.hxx>
#include <etx/render/shared/math.hxx>
#include <etx/render/shared/scene.hxx>
#include <etx/render/shared/scattering.hxx>
#include <etx/render/shared/ior_database.hxx>

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>

namespace etx {

namespace {

// Chunk ID constants
static constexpr const char* kChunkIdVertexPositions = "Vpos";
static constexpr const char* kChunkIdVertexNormals = "Vnrm";
static constexpr const char* kChunkIdVertexTexCoords = "Vtex";
static constexpr const char* kChunkIdVertexTangents = "Vtan";
static constexpr const char* kChunkIdVertexBitangents = "Vbtn";
static constexpr const char* kChunkIdTriangles = "Tri";
static constexpr const char* kChunkIdMeshes = "Mesh";
static constexpr const char* kChunkIdStringTable = "StrTab";
static constexpr const char* kChunkIdMaterialIndexMapping = "MatIdx";
static constexpr const char* kChunkIdMeshMapping = "MeshMap";

#pragma pack(push, 1)
struct BinaryGeometryFileHeader {
  uint32_t magic = kBinaryGeometryMagic;
  uint32_t version = kBinaryGeometryVersion;
  uint64_t total_size = 0;
};

struct ChunkHeader {
  char id[8] = {};
  uint64_t size = 0;
  uint32_t meta_size = 0;
};
#pragma pack(pop)

inline Material::Class material_string_to_class(const char* s) {
  if (strcmp(s, "diffuse") == 0)
    return Material::Class::Diffuse;
  if (strcmp(s, "translucent") == 0)
    return Material::Class::Translucent;
  else if (strcmp(s, "plastic") == 0)
    return Material::Class::Plastic;
  else if (strcmp(s, "conductor") == 0)
    return Material::Class::Conductor;
  else if (strcmp(s, "dielectric") == 0)
    return Material::Class::Dielectric;
  else if (strcmp(s, "thinfilm") == 0)
    return Material::Class::Thinfilm;
  else if (strcmp(s, "mirror") == 0)
    return Material::Class::Mirror;
  else if (strcmp(s, "boundary") == 0)
    return Material::Class::Boundary;
  else if (strcmp(s, "velvet") == 0)
    return Material::Class::Velvet;
  else if (strcmp(s, "principled") == 0)
    return Material::Class::Principled;
  else if (strcmp(s, "void") == 0)
    return Material::Class::Void;
  else {
    log::error("Undefined BSDF: `%s`", s);
    return Material::Class::Diffuse;
  }
}

// Helper function to compare chunk IDs efficiently
inline bool chunk_id_equals(const char* chunk_id, const char* expected_id) {
  for (int i = 0; i < sizeof(ChunkHeader::id); ++i) {
    char c1 = chunk_id[i];
    char c2 = expected_id[i];
    if (c1 != c2) {
      return false;
    }
    // Stop at null terminator in expected_id
    if (c2 == '\0') {
      return true;
    }
  }
  return true;
}

}  // namespace

struct SceneSerializationImpl {
  struct ChunkInfo {
    char id[sizeof(ChunkHeader::id) + 1] = {};
    std::vector<uint8_t> meta_data;
    std::vector<uint8_t> data;

    ChunkInfo(const char* chunk_id, std::vector<uint8_t> meta = {}, std::vector<uint8_t> chunk_data = {}) {
      memcpy(id, chunk_id, sizeof(ChunkHeader::id));
      id[sizeof(ChunkHeader::id)] = '\0';  // Ensure null termination
      meta_data = std::move(meta);
      data = std::move(chunk_data);
    }
  };

  // State
  std::vector<uint8_t> _buffer;
  std::vector<std::string> _string_table;
  std::vector<ChunkInfo> _chunks;
  std::vector<MaterialIndexMapping> _material_index_mappings;
  bool _is_loaded = false;

  // Helper buffer for parsing
  static constexpr uint32_t kDataBufferSize = 2048u;
  char _data_buffer[kDataBufferSize] = {};

  // Helper functions for material parsing
  bool get_param(const MaterialDefinition& m, const char* param) {
    memset(_data_buffer, 0, kDataBufferSize);
    auto it = m.properties.find(param);
    if (it != m.properties.end()) {
      size_t len = it->second.size();
      if (len < kDataBufferSize) {
        memcpy(_data_buffer, it->second.c_str(), len);
        _data_buffer[len] = 0;
        return true;
      }
    }
    return false;
  }

  bool is_internal_name(const std::string& name) {
    return name.compare(0, 4, "et::") == 0 || name.compare(0, 5, "etx::") == 0;
  }

  void resolve_material_index_mappings(SceneData& scene_data) {
    if (_material_index_mappings.empty()) {
      return;
    }

    // Create translation map: saved_index -> runtime_index
    std::unordered_map<uint32_t, uint32_t> index_translation;

    for (const auto& mapping : _material_index_mappings) {
      if (mapping.name_index >= _string_table.size()) {
        log::warning("Invalid name index %u in material mapping (string table size: %zu)", mapping.name_index, _string_table.size());
        continue;
      }

      const std::string& material_name = _string_table[mapping.name_index];

      // Find the runtime index for this material name
      if (scene_data.material_mapping.count(material_name)) {
        uint32_t runtime_index = scene_data.material_mapping[material_name];
        index_translation[mapping.saved_index] = runtime_index;
      } else {
        log::warning("Material '%s' not found in runtime mapping", material_name.c_str());
      }
    }

    // Apply translations to triangles
    for (auto& tri : scene_data.triangles) {
      if (tri.material_index != kInvalidIndex && index_translation.count(tri.material_index)) {
        tri.material_index = index_translation[tri.material_index];
      }
    }

    // Clear mappings after use
    _material_index_mappings.clear();
  }

  bool get_file(const char* base_dir, const std::string& base) {
    memset(_data_buffer, 0, sizeof(_data_buffer));
    if (base.empty() == false) {
      snprintf(_data_buffer, sizeof(_data_buffer), "%s/%s", base_dir, base.c_str());
      return true;
    }
    return false;
  }

  std::vector<const char*> split_params(char* data) {
    std::vector<const char*> params;
    const char* begin = data;
    char* token = data;
    while (*token != 0) {
      if (*token == 0x20) {
        *token++ = 0;
        params.emplace_back(begin);
        begin = token;
      } else {
        ++token;
      }
    }
    params.emplace_back(begin);
    return params;
  }

  std::filesystem::path locate_spectrum_file(const char* identifier, std::initializer_list<const char*> fallback_folders) const {
    if ((identifier == nullptr) || (identifier[0] == 0))
      return {};

    std::filesystem::path requested(identifier);
    if (requested.has_extension() == false)
      requested.replace_extension(".spd");

    std::error_code ec;
    if (requested.is_absolute()) {
      if (std::filesystem::exists(requested, ec))
        return requested;
      return {};
    }

    std::filesystem::path data_root = std::filesystem::path(env().data_folder()) / "spectrum";

    std::filesystem::path combined = data_root / requested;
    if (std::filesystem::exists(combined, ec))
      return combined;

    for (const char* folder : fallback_folders) {
      std::filesystem::path candidate = data_root / folder / requested.filename();
      if (std::filesystem::exists(candidate, ec))
        return candidate;
    }

    return {};
  }

  bool load_ior_from_identifier(const char* identifier, const IORDatabase& ior_database, SpectralDistribution& eta, SpectralDistribution& k,
    SpectralDistribution::Class& cls) const {
    if ((identifier == nullptr) || (identifier[0] == 0))
      return false;

    if (const IORDefinition* def = ior_database.find_by_name(identifier)) {
      cls = def->cls;
      eta = def->eta;
      k = def->k;
      return true;
    }

    std::filesystem::path candidate = locate_spectrum_file(identifier, {"conductor", "dielectric"});
    if (candidate.empty())
      return false;

    cls = RefractiveIndex::load_from_file(candidate.string().c_str(), eta, k);
    return cls != SpectralDistribution::Class::Invalid;
  }

  uint32_t load_reflectance_spectrum(SceneData& data, char* values) {
    auto params = split_params(values);

    if (params.size() == 1) {
      uint32_t index = data.find_spectrum(params[0]);
      if (index != kInvalidIndex)
        return index;
    }

    if (params.size() == 3) {
      float3 value = gamma_to_linear({
        static_cast<float>(atof(params[0])),
        static_cast<float>(atof(params[1])),
        static_cast<float>(atof(params[2])),
      });
      return data.add_spectrum(SpectralDistribution::rgb_reflectance(value));
    }

    return 0u;
  }

  SpectralDistribution load_illuminant_spectrum(SceneData& data, char* values) {
    auto params = split_params(values);

    if (params.size() == 1) {
      float value = 0.0f;
      if (sscanf(params[0], "%f", &value) == 1) {
        return SpectralDistribution::rgb_luminance({value, value, value});
      }

      auto i = data.find_spectrum(params[0]);
      if (i != kInvalidIndex)
        return data.spectrum_values[i];
    }

    if (params.size() == 3) {
      float3 value = {
        static_cast<float>(atof(params[0])),
        static_cast<float>(atof(params[1])),
        static_cast<float>(atof(params[2])),
      };
      return SpectralDistribution::rgb_luminance(value);
    }

    SpectralDistribution emitter_spectrum = SpectralDistribution::rgb_luminance({1.0f, 1.0f, 1.0f});

    float scale = 1.0f;
    for (uint64_t i = 0, count = params.size(); i < count; ++i) {
      if ((strcmp(params[i], "blackbody") == 0) && (i + 1 < count)) {
        float temperature = static_cast<float>(atof(params[i + 1]));
        emitter_spectrum = SpectralDistribution::from_black_body(temperature, 1.0f);
        i += 1;
      } else if ((strcmp(params[i], "nblackbody") == 0) && (i + 1 < count)) {
        float temperature = static_cast<float>(atof(params[i + 1]));
        emitter_spectrum = SpectralDistribution::from_normalized_black_body(temperature, 1.0f);
        i += 1;
      } else if ((strcmp(params[i], "scale") == 0) && (i + 1 < count)) {
        scale = static_cast<float>(atof(params[i + 1]));
        i += 1;
      }
    }

    emitter_spectrum.scale(scale);
    return emitter_spectrum;
  }

  // Writing methods
  bool prepare_data(const SceneData& data) {
    _buffer.clear();
    _string_table.clear();
    _chunks.clear();

    // Reserve space for file header and initialize it
    _buffer.resize(sizeof(BinaryGeometryFileHeader));
    BinaryGeometryFileHeader* header = reinterpret_cast<BinaryGeometryFileHeader*>(_buffer.data());
    header->magic = kBinaryGeometryMagic;
    header->version = kBinaryGeometryVersion;
    header->total_size = 0;  // Will be updated in save_to_file()

    // Write vertex positions
    if (!data.vertices.pos.empty()) {
      ChunkInfo chunk = {kChunkIdVertexPositions, {}};
      chunk.meta_data.resize(sizeof(uint64_t));
      *reinterpret_cast<uint64_t*>(chunk.meta_data.data()) = data.vertices.pos.size();
      chunk.data.resize(data.vertices.pos.size() * sizeof(float3));
      memcpy(chunk.data.data(), data.vertices.pos.data(), data.vertices.pos.size() * sizeof(float3));
      _chunks.emplace_back(std::move(chunk));
    }

    // Write vertex normals
    if (!data.vertices.nrm.empty()) {
      ChunkInfo chunk = {kChunkIdVertexNormals, {}};
      chunk.meta_data.resize(sizeof(uint64_t));
      *reinterpret_cast<uint64_t*>(chunk.meta_data.data()) = data.vertices.nrm.size();
      chunk.data.resize(data.vertices.nrm.size() * sizeof(float3));
      memcpy(chunk.data.data(), data.vertices.nrm.data(), data.vertices.nrm.size() * sizeof(float3));
      _chunks.emplace_back(std::move(chunk));
    }

    // Write vertex texture coordinates
    if (!data.vertices.tex.empty()) {
      ChunkInfo chunk = {kChunkIdVertexTexCoords, {}};
      chunk.meta_data.resize(sizeof(uint64_t));
      *reinterpret_cast<uint64_t*>(chunk.meta_data.data()) = data.vertices.tex.size();
      chunk.data.resize(data.vertices.tex.size() * sizeof(float2));
      memcpy(chunk.data.data(), data.vertices.tex.data(), data.vertices.tex.size() * sizeof(float2));
      _chunks.emplace_back(std::move(chunk));
    }

    // Write vertex tangents
    if (!data.vertices.tan.empty()) {
      ChunkInfo chunk = {kChunkIdVertexTangents, {}};
      chunk.meta_data.resize(sizeof(uint64_t));
      *reinterpret_cast<uint64_t*>(chunk.meta_data.data()) = data.vertices.tan.size();
      chunk.data.resize(data.vertices.tan.size() * sizeof(float3));
      memcpy(chunk.data.data(), data.vertices.tan.data(), data.vertices.tan.size() * sizeof(float3));
      _chunks.emplace_back(std::move(chunk));
    }

    // Write vertex bitangents
    if (!data.vertices.btn.empty()) {
      ChunkInfo chunk = {kChunkIdVertexBitangents, {}};
      chunk.meta_data.resize(sizeof(uint64_t));
      *reinterpret_cast<uint64_t*>(chunk.meta_data.data()) = data.vertices.btn.size();
      chunk.data.resize(data.vertices.btn.size() * sizeof(float3));
      memcpy(chunk.data.data(), data.vertices.btn.data(), data.vertices.btn.size() * sizeof(float3));
      _chunks.emplace_back(std::move(chunk));
    }

    // Write triangles
    {
      ChunkInfo chunk = {kChunkIdTriangles, {}};

      // Metadata: triangle count
      chunk.meta_data.resize(sizeof(uint64_t));
      *reinterpret_cast<uint64_t*>(chunk.meta_data.data()) = data.triangles.size();

      // Data: triangle array
      chunk.data.resize(data.triangles.size() * sizeof(Triangle));
      memcpy(chunk.data.data(), data.triangles.data(), data.triangles.size() * sizeof(Triangle));

      _chunks.emplace_back(std::move(chunk));
    }

    // Write material index mapping (maps saved triangle indices to material names)
    {
      // Find all unique material indices referenced by triangles
      std::set<uint32_t> referenced_indices;
      for (const auto& tri : data.triangles) {
        if (tri.material_index != kInvalidIndex) {
          referenced_indices.insert(tri.material_index);
        }
      }

      // Create mapping: saved_index -> material_name_index
      std::vector<MaterialIndexMapping> mappings;
      mappings.reserve(referenced_indices.size());

      for (uint32_t saved_idx : referenced_indices) {
        if (saved_idx < data.materials.size()) {
          // Find the material name for this index
          for (const auto& [name, runtime_idx] : data.material_mapping) {
            if (runtime_idx == saved_idx && !is_internal_name(name)) {
              uint32_t name_index = add_string(name);
              mappings.push_back({saved_idx, name_index});
              break;
            }
          }
        } else {
          log::warning("Triangle references invalid material index %u", saved_idx);
        }
      }

      ChunkInfo chunk = {kChunkIdMaterialIndexMapping, {}};

      // Metadata: mapping count
      chunk.meta_data.resize(sizeof(uint64_t));
      *reinterpret_cast<uint64_t*>(chunk.meta_data.data()) = mappings.size();

      // Data: mapping entries
      chunk.data.resize(mappings.size() * sizeof(MaterialIndexMapping));
      memcpy(chunk.data.data(), mappings.data(), mappings.size() * sizeof(MaterialIndexMapping));

      _chunks.emplace_back(std::move(chunk));
    }

    // Write meshes
    {
      ChunkInfo chunk = {kChunkIdMeshes, {}};

      // Metadata: mesh count
      chunk.meta_data.resize(sizeof(uint64_t));
      *reinterpret_cast<uint64_t*>(chunk.meta_data.data()) = data.meshes.size();

      // Data: mesh array
      chunk.data.resize(data.meshes.size() * sizeof(Mesh));
      memcpy(chunk.data.data(), data.meshes.data(), data.meshes.size() * sizeof(Mesh));

      _chunks.emplace_back(std::move(chunk));
    }

    // Write mesh mapping (populates string table)
    {
      ChunkInfo chunk = {kChunkIdMeshMapping, {}};

      // Metadata: mapping count
      chunk.meta_data.resize(sizeof(uint64_t));
      *reinterpret_cast<uint64_t*>(chunk.meta_data.data()) = data.mesh_mapping.size();

      // Data: mapping entries
      chunk.data.resize(data.mesh_mapping.size() * sizeof(MappingEntry));
      MappingEntry* entries = reinterpret_cast<MappingEntry*>(chunk.data.data());
      uint32_t index = 0;
      for (const auto& [name, id] : data.mesh_mapping) {
        entries[index].string_index = add_string(name);
        entries[index].target_index = id;
        ++index;
      }

      _chunks.emplace_back(std::move(chunk));
    }

    // Write string table (after all strings have been added)
    if (!write_string_table()) {
      return false;
    }

    return true;
  }

  uint32_t add_string(const std::string& str) {
    auto it = std::find(_string_table.begin(), _string_table.end(), str);
    if (it != _string_table.end()) {
      return static_cast<uint32_t>(std::distance(_string_table.begin(), it));
    }

    uint32_t index = static_cast<uint32_t>(_string_table.size());
    _string_table.emplace_back(str);
    return index;
  }

  bool write_string_table() {
    if (_string_table.empty()) {
      return true;  // Empty string table is OK
    }

    ChunkInfo chunk = {kChunkIdStringTable, {}};

    // Metadata: string count
    chunk.meta_data.resize(sizeof(uint64_t));
    *reinterpret_cast<uint64_t*>(chunk.meta_data.data()) = _string_table.size();

    // Calculate total data size
    size_t total_size = 0;
    for (const auto& str : _string_table) {
      total_size += sizeof(uint32_t) + str.size() + 1;  // length + string + null terminator
    }

    chunk.data.resize(total_size);

    // Pack strings
    size_t offset = 0;
    for (const auto& str : _string_table) {
      uint32_t length = static_cast<uint32_t>(str.size() + 1);  // include null terminator
      memcpy(chunk.data.data() + offset, &length, sizeof(uint32_t));
      offset += sizeof(uint32_t);

      memcpy(chunk.data.data() + offset, str.c_str(), str.size() + 1);
      offset += str.size() + 1;
    }

    _chunks.emplace_back(std::move(chunk));
    return true;
  }

  bool serialize_chunk(const ChunkInfo& chunk) {
    ChunkHeader header = {};
    memcpy(header.id, chunk.id, sizeof(ChunkHeader::id));
    header.size = chunk.data.size();
    header.meta_size = static_cast<uint32_t>(chunk.meta_data.size());

    // Write chunk header
    size_t offset = _buffer.size();
    _buffer.resize(offset + sizeof(ChunkHeader));
    memcpy(_buffer.data() + offset, &header, sizeof(ChunkHeader));

    // Write metadata
    if (!chunk.meta_data.empty()) {
      offset = _buffer.size();
      _buffer.resize(offset + chunk.meta_data.size());
      memcpy(_buffer.data() + offset, chunk.meta_data.data(), chunk.meta_data.size());
    }

    // Write data
    if (!chunk.data.empty()) {
      offset = _buffer.size();
      _buffer.resize(offset + chunk.data.size());
      memcpy(_buffer.data() + offset, chunk.data.data(), chunk.data.size());
    }

    return true;
  }

  bool write_to_file(const std::filesystem::path& path) {
    // Write string table first (needed by mappings)
    if (!write_string_table()) {
      return false;
    }

    // Serialize all chunks
    for (const auto& chunk : _chunks) {
      if (!serialize_chunk(chunk)) {
        return false;
      }
    }

    // Update file header with total size
    BinaryGeometryFileHeader* header = reinterpret_cast<BinaryGeometryFileHeader*>(_buffer.data());
    header->total_size = _buffer.size();

    // Write to file
    std::ofstream file(path, std::ios::out | std::ios::trunc | std::ios::binary);
    if (!file.is_open()) {
      log::error("Failed to open file for writing: %s", path.string().c_str());
      return false;
    }

    file.write(reinterpret_cast<const char*>(_buffer.data()), _buffer.size());
    file.close();

    if (!file.good()) {
      log::error("Failed to write data to file: %s", path.string().c_str());
      return false;
    }

    log::info("Binary geometry saved to %s (%zu bytes, %zu chunks)", path.string().c_str(), _buffer.size(), _chunks.size());

    return true;
  }

  bool load_from_file(const std::filesystem::path& path, SceneData& data, const char* materials_file, SceneLoaderContext& context, Scene& scene, const IORDatabase& database,
    TaskScheduler& scheduler) {
    char base_dir[2048] = {};
    if (materials_file && materials_file[0]) {
      get_file_folder(materials_file, base_dir, sizeof(base_dir));
    }

    if (!parse_materials_file(materials_file, base_dir, data, context, scene, database, scheduler)) {
      log::error("Failed to load materials from %s", materials_file);
      return false;
    }

    std::ifstream file(path, std::ios::in | std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
      log::error("Failed to open file for reading: %s", path.string().c_str());
      return false;
    }

    size_t file_size = file.tellg();
    file.seekg(0);

    if (file_size < sizeof(BinaryGeometryFileHeader)) {
      log::error("File too small for binary geometry header: %s", path.string().c_str());
      return false;
    }

    _buffer.resize(file_size);
    file.read(reinterpret_cast<char*>(_buffer.data()), file_size);
    file.close();

    if (!file.good()) {
      log::error("Failed to read file: %s", path.string().c_str());
      return false;
    }

    // Clear existing geometry data in SceneData (preserve material mappings if we loaded them)
    data.vertices.pos.clear();
    data.vertices.nrm.clear();
    data.vertices.tex.clear();
    data.vertices.tan.clear();
    data.vertices.btn.clear();
    data.triangles.clear();
    data.meshes.clear();
    data.mesh_mapping.clear();

    return parse_file(data);
  }

  // Reading methods
  bool parse_file(SceneData& data) {
    if (_buffer.size() < sizeof(BinaryGeometryFileHeader)) {
      return false;
    }

    const BinaryGeometryFileHeader* header = reinterpret_cast<const BinaryGeometryFileHeader*>(_buffer.data());
    if (header->magic != kBinaryGeometryMagic) {
      log::error("Invalid magic number in binary geometry file");
      return false;
    }

    if (header->version != kBinaryGeometryVersion) {
      log::error("Unsupported binary geometry version: %u (expected %u)", header->version, kBinaryGeometryVersion);
      return false;
    }

    if (header->total_size != _buffer.size()) {
      log::error("File size mismatch: expected %llu, got %zu", header->total_size, _buffer.size());
      return false;
    }

    // Clear string table for new file
    _string_table.clear();

    // First pass: parse independent chunks (geometry, string table)
    size_t offset = sizeof(BinaryGeometryFileHeader);
    std::vector<std::pair<size_t, std::string>> deferred_chunks;

    while (offset < _buffer.size()) {
      // Peek at chunk ID without advancing offset
      if (offset + sizeof(ChunkHeader) > _buffer.size()) {
        log::error("Unexpected end of file while reading chunk header");
        return false;
      }

      const ChunkHeader* header = reinterpret_cast<const ChunkHeader*>(_buffer.data() + offset);

      // Process independent chunks immediately
      if (chunk_id_equals(header->id, kChunkIdStringTable) || chunk_id_equals(header->id, kChunkIdVertexPositions) || chunk_id_equals(header->id, kChunkIdVertexNormals) ||
          chunk_id_equals(header->id, kChunkIdVertexTexCoords) || chunk_id_equals(header->id, kChunkIdVertexTangents) || chunk_id_equals(header->id, kChunkIdVertexBitangents) ||
          chunk_id_equals(header->id, kChunkIdTriangles) || chunk_id_equals(header->id, kChunkIdMeshes)) {
        if (!parse_chunk(offset, data)) {
          return false;
        }
      } else {
        // Defer dependent chunks (mappings) until string table is loaded
        std::string chunk_id_str(header->id, sizeof(ChunkHeader::id));
        size_t null_pos = chunk_id_str.find('\0');
        if (null_pos != std::string::npos) {
          chunk_id_str = chunk_id_str.substr(0, null_pos);
        }
        deferred_chunks.emplace_back(offset, chunk_id_str);
        size_t chunk_size = sizeof(ChunkHeader) + header->meta_size + header->size;
        offset += chunk_size;
      }
    }

    // Second pass: parse deferred chunks (mappings) now that string table is loaded
    for (const auto& [chunk_offset, chunk_id] : deferred_chunks) {
      size_t temp_offset = chunk_offset;
      if (!parse_chunk(temp_offset, data)) {
        return false;
      }
    }

    resolve_material_index_mappings(data);

    return true;
  }

  bool parse_chunk(size_t& offset, SceneData& data) {
    if (offset + sizeof(ChunkHeader) > _buffer.size()) {
      log::error("Unexpected end of file while reading chunk header");
      return false;
    }

    const ChunkHeader* header = reinterpret_cast<const ChunkHeader*>(_buffer.data() + offset);
    offset += sizeof(ChunkHeader);

    size_t meta_size = header->meta_size;
    size_t data_size = header->size;

    if (offset + meta_size + data_size > _buffer.size()) {
      log::error("Chunk data extends beyond file");
      return false;
    }

    const uint8_t* meta_data = (meta_size > 0) ? _buffer.data() + offset : nullptr;
    offset += meta_size;

    const uint8_t* chunk_data = (data_size > 0) ? _buffer.data() + offset : nullptr;
    offset += data_size;

    // Handle different chunk types
    if (chunk_id_equals(header->id, kChunkIdVertexPositions)) {
      return parse_vertex_positions_chunk(data, meta_data, meta_size, chunk_data, data_size);
    } else if (chunk_id_equals(header->id, kChunkIdVertexNormals)) {
      return parse_vertex_normals_chunk(data, meta_data, meta_size, chunk_data, data_size);
    } else if (chunk_id_equals(header->id, kChunkIdVertexTexCoords)) {
      return parse_vertex_texcoords_chunk(data, meta_data, meta_size, chunk_data, data_size);
    } else if (chunk_id_equals(header->id, kChunkIdVertexTangents)) {
      return parse_vertex_tangents_chunk(data, meta_data, meta_size, chunk_data, data_size);
    } else if (chunk_id_equals(header->id, kChunkIdVertexBitangents)) {
      return parse_vertex_bitangents_chunk(data, meta_data, meta_size, chunk_data, data_size);
    } else if (chunk_id_equals(header->id, kChunkIdTriangles)) {
      return parse_triangle_chunk(data, meta_data, meta_size, chunk_data, data_size);
    } else if (chunk_id_equals(header->id, kChunkIdMeshes)) {
      return parse_mesh_chunk(data, meta_data, meta_size, chunk_data, data_size);
    } else if (chunk_id_equals(header->id, kChunkIdStringTable)) {
      return parse_string_table_chunk(meta_data, meta_size, chunk_data, data_size);
    } else if (chunk_id_equals(header->id, kChunkIdMaterialIndexMapping)) {
      return parse_material_index_mapping_chunk(data, meta_data, meta_size, chunk_data, data_size);
    } else if (chunk_id_equals(header->id, kChunkIdMeshMapping)) {
      return parse_mesh_mapping_chunk(data, meta_data, meta_size, chunk_data, data_size);
    } else {
      // Create a string for logging unknown chunk types
      std::string chunk_id_str(header->id, sizeof(ChunkHeader::id));
      size_t null_pos = chunk_id_str.find('\0');
      if (null_pos != std::string::npos) {
        chunk_id_str = chunk_id_str.substr(0, null_pos);
      }
      log::warning("Unknown chunk type: %s - skipping", chunk_id_str.c_str());
      // Skip unknown chunks for forward compatibility
      return true;
    }

    // This should never be reached, but add return false to satisfy compiler
    return false;
  }

  bool parse_vertex_positions_chunk(SceneData& scene_data, const uint8_t* meta_data, size_t meta_size, const uint8_t* data, size_t data_size) {
    if (meta_size != sizeof(uint64_t)) {
      log::error("Invalid metadata size for VPOS chunk");
      return false;
    }

    uint64_t vertex_count = *reinterpret_cast<const uint64_t*>(meta_data);
    size_t expected_size = vertex_count * sizeof(float3);
    if (data_size != expected_size) {
      log::error("Data size mismatch for VPOS chunk: expected %zu, got %zu", expected_size, data_size);
      return false;
    }

    scene_data.vertices.pos.resize(vertex_count);
    memcpy(scene_data.vertices.pos.data(), data, vertex_count * sizeof(float3));
    return true;
  }

  bool parse_vertex_normals_chunk(SceneData& scene_data, const uint8_t* meta_data, size_t meta_size, const uint8_t* data, size_t data_size) {
    if (meta_size != sizeof(uint64_t)) {
      log::error("Invalid metadata size for VNOR chunk");
      return false;
    }

    uint64_t vertex_count = *reinterpret_cast<const uint64_t*>(meta_data);
    size_t expected_size = vertex_count * sizeof(float3);
    if (data_size != expected_size) {
      log::error("Data size mismatch for VNOR chunk: expected %zu, got %zu", expected_size, data_size);
      return false;
    }

    scene_data.vertices.nrm.resize(vertex_count);
    memcpy(scene_data.vertices.nrm.data(), data, vertex_count * sizeof(float3));
    return true;
  }

  bool parse_vertex_texcoords_chunk(SceneData& scene_data, const uint8_t* meta_data, size_t meta_size, const uint8_t* data, size_t data_size) {
    if (meta_size != sizeof(uint64_t)) {
      log::error("Invalid metadata size for VTEX chunk");
      return false;
    }

    uint64_t vertex_count = *reinterpret_cast<const uint64_t*>(meta_data);
    size_t expected_size = vertex_count * sizeof(float2);
    if (data_size != expected_size) {
      log::error("Data size mismatch for VTEX chunk: expected %zu, got %zu", expected_size, data_size);
      return false;
    }

    scene_data.vertices.tex.resize(vertex_count);
    memcpy(scene_data.vertices.tex.data(), data, vertex_count * sizeof(float2));
    return true;
  }

  bool parse_vertex_tangents_chunk(SceneData& scene_data, const uint8_t* meta_data, size_t meta_size, const uint8_t* data, size_t data_size) {
    if (meta_size != sizeof(uint64_t)) {
      log::error("Invalid metadata size for VTAN chunk");
      return false;
    }

    uint64_t vertex_count = *reinterpret_cast<const uint64_t*>(meta_data);
    size_t expected_size = vertex_count * sizeof(float3);
    if (data_size != expected_size) {
      log::error("Data size mismatch for VTAN chunk: expected %zu, got %zu", expected_size, data_size);
      return false;
    }

    scene_data.vertices.tan.resize(vertex_count);
    memcpy(scene_data.vertices.tan.data(), data, vertex_count * sizeof(float3));
    return true;
  }

  bool parse_vertex_bitangents_chunk(SceneData& scene_data, const uint8_t* meta_data, size_t meta_size, const uint8_t* data, size_t data_size) {
    if (meta_size != sizeof(uint64_t)) {
      log::error("Invalid metadata size for VBTN chunk");
      return false;
    }

    uint64_t vertex_count = *reinterpret_cast<const uint64_t*>(meta_data);
    size_t expected_size = vertex_count * sizeof(float3);
    if (data_size != expected_size) {
      log::error("Data size mismatch for VBTN chunk: expected %zu, got %zu", expected_size, data_size);
      return false;
    }

    scene_data.vertices.btn.resize(vertex_count);
    memcpy(scene_data.vertices.btn.data(), data, vertex_count * sizeof(float3));
    return true;
  }

  bool parse_triangle_chunk(SceneData& scene_data, const uint8_t* meta_data, size_t meta_size, const uint8_t* data, size_t data_size) {
    if (meta_size != sizeof(uint64_t)) {
      log::error("Invalid metadata size for TRIS chunk");
      return false;
    }

    uint64_t triangle_count = *reinterpret_cast<const uint64_t*>(meta_data);

    size_t expected_size = triangle_count * sizeof(Triangle);
    if (data_size != expected_size) {
      log::error("Data size mismatch for TRIS chunk: expected %zu, got %zu", expected_size, data_size);
      return false;
    }

    scene_data.triangles.resize(triangle_count);
    memcpy(scene_data.triangles.data(), data, data_size);

    // Validate triangle data
    size_t vertex_count = scene_data.vertices.pos.size();
    for (uint64_t i = 0; i < triangle_count; ++i) {
      const auto& tri = scene_data.triangles[i];

      // Validate vertex indices
      for (int j = 0; j < 3; ++j) {
        if (tri.i[j] >= vertex_count) {
          log::error("Triangle %llu vertex %d references invalid vertex index %u (total vertices: %zu)", i, j, tri.i[j], vertex_count);
          return false;
        }
      }
    }

    return true;
  }

  bool parse_mesh_chunk(SceneData& scene_data, const uint8_t* meta_data, size_t meta_size, const uint8_t* data, size_t data_size) {
    if (meta_size != sizeof(uint64_t)) {
      log::error("Invalid metadata size for MESH chunk");
      return false;
    }

    uint64_t mesh_count = *reinterpret_cast<const uint64_t*>(meta_data);

    size_t expected_size = mesh_count * sizeof(Mesh);
    if (data_size != expected_size) {
      log::error("Data size mismatch for MESH chunk: expected %zu, got %zu", expected_size, data_size);
      return false;
    }

    scene_data.meshes.resize(mesh_count);
    memcpy(scene_data.meshes.data(), data, data_size);

    // Validate mesh triangle ranges
    size_t triangle_count = scene_data.triangles.size();
    for (uint64_t i = 0; i < mesh_count; ++i) {
      const auto& mesh = scene_data.meshes[i];
      if (mesh.triangle_offset >= triangle_count) {
        log::error("Mesh %llu has invalid triangle offset %u (total triangles: %zu)", i, mesh.triangle_offset, triangle_count);
        return false;
      }
      if (mesh.triangle_offset + mesh.triangle_count > triangle_count) {
        log::error("Mesh %llu triangle range [%u, %u) exceeds triangle count %zu", i, mesh.triangle_offset, mesh.triangle_offset + mesh.triangle_count, triangle_count);
        return false;
      }
    }

    return true;
  }

  bool parse_string_table_chunk(const uint8_t* meta_data, size_t meta_size, const uint8_t* data, size_t data_size) {
    if (meta_size != sizeof(uint64_t)) {
      log::error("Invalid metadata size for STRTAB chunk");
      return false;
    }

    uint64_t string_count = *reinterpret_cast<const uint64_t*>(meta_data);

    _string_table.clear();
    _string_table.reserve(string_count);

    size_t offset = 0;
    for (uint64_t i = 0; i < string_count; ++i) {
      if (offset + sizeof(uint32_t) > data_size) {
        log::error("Unexpected end of string table data");
        return false;
      }

      uint32_t length = *reinterpret_cast<const uint32_t*>(data + offset);
      offset += sizeof(uint32_t);

      if (offset + length > data_size) {
        log::error("String data extends beyond chunk");
        return false;
      }

      std::string str(reinterpret_cast<const char*>(data + offset), length - 1);  // exclude null terminator
      _string_table.emplace_back(std::move(str));
      offset += length;
    }

    return true;
  }

  bool parse_material_index_mapping_chunk(SceneData& scene_data, const uint8_t* meta_data, size_t meta_size, const uint8_t* data, size_t data_size) {
    if (meta_size != sizeof(uint64_t)) {
      log::error("Invalid metadata size for MIDXMP chunk");
      return false;
    }

    uint64_t mapping_count = *reinterpret_cast<const uint64_t*>(meta_data);

    size_t expected_size = mapping_count * sizeof(MaterialIndexMapping);
    if (data_size != expected_size) {
      log::error("Data size mismatch for MIDXMP chunk: expected %zu, got %zu", expected_size, data_size);
      return false;
    }

    const MaterialIndexMapping* mappings = reinterpret_cast<const MaterialIndexMapping*>(data);
    _material_index_mappings.assign(mappings, mappings + mapping_count);

    return true;
  }

  bool parse_mesh_mapping_chunk(SceneData& scene_data, const uint8_t* meta_data, size_t meta_size, const uint8_t* data, size_t data_size) {
    if (meta_size != sizeof(uint64_t)) {
      log::error("Invalid metadata size for MSHMAP chunk");
      return false;
    }

    uint64_t mapping_count = *reinterpret_cast<const uint64_t*>(meta_data);

    size_t expected_size = mapping_count * sizeof(MappingEntry);
    if (data_size != expected_size) {
      log::error("Data size mismatch for MSHMAP chunk: expected %zu, got %zu", expected_size, data_size);
      return false;
    }

    const MappingEntry* entries = reinterpret_cast<const MappingEntry*>(data);
    for (uint64_t i = 0; i < mapping_count; ++i) {
      const MappingEntry& entry = entries[i];

      if (entry.string_index >= _string_table.size()) {
        log::error("Invalid string index in mesh mapping: %u", entry.string_index);
        return false;
      }

      if (entry.target_index >= scene_data.meshes.size()) {
        log::error("Invalid mesh index in mesh mapping: %u (total meshes: %zu)", entry.target_index, scene_data.meshes.size());
        return false;
      }

      scene_data.mesh_mapping[_string_table[entry.string_index]] = entry.target_index;
    }

    return true;
  }

  bool parse_materials_file(const std::filesystem::path& path, const char* base_dir, SceneData& data, SceneLoaderContext& context, Scene& scene, const IORDatabase& database,
    TaskScheduler& scheduler) {
    std::ifstream file(path);
    if (!file.is_open()) {
      log::error("Failed to open materials file: %s", path.string().c_str());
      return false;
    }

    std::vector<MaterialDefinition> materials;
    MaterialDefinition* current_material = nullptr;

    std::string line;
    while (std::getline(file, line)) {
      if (line.empty() || line[0] == '#') {
        continue;
      }

      line.erase(line.begin(), std::find_if(line.begin(), line.end(), [](unsigned char ch) {
        return !std::isspace(ch);
      }));
      line.erase(std::find_if(line.rbegin(), line.rend(),
                   [](unsigned char ch) {
                     return !std::isspace(ch);
                   })
                   .base(),
        line.end());

      if (line.empty()) {
        continue;
      }

      std::istringstream iss(line);
      std::string key;
      iss >> key;

      if (key == "newmtl") {
        std::string material_name;
        iss >> material_name;
        materials.push_back({material_name, {}});
        current_material = &materials.back();
      } else if (current_material) {
        std::string value;
        std::getline(iss, value);
        if (!value.empty() && value[0] == ' ') {
          value.erase(0, 1);
        }
        if (!value.empty()) {
          current_material->properties[key] = value;
        }
      }
    }

    file.close();

    // Use the unified parsing approach
    SceneSerialization temp_serialization;
    temp_serialization.parse_material_definitions(base_dir, materials, data, context, scene, database, scheduler);

    return true;
  }

  void parse_camera(const char* base_dir, const MaterialDefinition& material, SceneData& data, SceneLoaderContext& context, Scene& scene, const IORDatabase& database) {
    auto& entry = data.cameras.emplace_back();

    if (get_param(material, "class")) {
      entry.cam.cls = (strcmp(_data_buffer, "eq") == 0) ? Camera::Class::Equirectangular : Camera::Class::Perspective;
    }

    if (get_param(material, "viewport")) {
      uint32_t val[2] = {};
      if (sscanf(_data_buffer, "%u %u", val + 0, val + 1) == 2) {
        entry.cam.film_size = {val[0], val[1]};
      }
    }

    if (get_param(material, "origin")) {
      float val[3] = {};
      if (sscanf(_data_buffer, "%f %f %f", val + 0, val + 1, val + 2) == 3) {
        entry.cam.position = {val[0], val[1], val[2]};
      }
    }

    float3 target = entry.cam.position + kWorldForward;
    if (get_param(material, "target")) {
      float val[3] = {};
      if (sscanf(_data_buffer, "%f %f %f", val + 0, val + 1, val + 2) == 3) {
        target = {val[0], val[1], val[2]};
      }
    }

    // Check for direction parameter (backward compatibility)
    if (get_param(material, "direction")) {
      float val[3] = {};
      if (sscanf(_data_buffer, "%f %f %f", val + 0, val + 1, val + 2) == 3) {
        entry.cam.direction = normalize(float3{val[0], val[1], val[2]});
      }
    } else {
      entry.cam.direction = normalize(target - entry.cam.position);
    }

    entry.cam.up = kWorldUp;
    if (get_param(material, "up")) {
      float val[3] = {};
      if (sscanf(_data_buffer, "%f %f %f", val + 0, val + 1, val + 2) == 3) {
        entry.cam.up = {val[0], val[1], val[2]};
      }
    }

    float fov = 50.0f;
    if (get_param(material, "fov")) {
      float val = 0.0f;
      if (sscanf(_data_buffer, "%f", &val) == 1) {
        fov = val;
      }
    }

    if (get_param(material, "focal-length")) {
      float val = 0.0f;
      if (sscanf(_data_buffer, "%f", &val) == 1) {
        fov = focal_length_to_fov(val) * 180.0f / kPi;
      }
    }

    if (get_param(material, "lens-radius")) {
      float val = 0.0f;
      if (sscanf(_data_buffer, "%f", &val) == 1) {
        entry.cam.lens_radius = val;
      }
    }

    if (get_param(material, "focal-distance")) {
      float val = 0.0f;
      if (sscanf(_data_buffer, "%f", &val) == 1) {
        entry.cam.focal_distance = val;
      }
    }

    if (get_param(material, "clip-near")) {
      float val = 0.0f;
      if (sscanf(_data_buffer, "%f", &val) == 1) {
        entry.cam.clip_near = val;
      }
    }

    if (get_param(material, "clip-far")) {
      float val = 0.0f;
      if (sscanf(_data_buffer, "%f", &val) == 1) {
        entry.cam.clip_far = val;
      }
    }

    if (get_param(material, "shape")) {
      char tmp_buffer[2048] = {};
      snprintf(tmp_buffer, sizeof(tmp_buffer), "%s/%s", base_dir, _data_buffer);
      entry.cam.lens_image = context.add_image(tmp_buffer, Image::BuildSamplingTable | Image::UniformSamplingTable, {}, {1.0f, 1.0f});
    }

    if (get_param(material, "ext_medium")) {
      auto m = context.mediums.find(_data_buffer);
      if (m == kInvalidIndex) {
        log::warning("Medium %s was not declared, but used in camera %s as external medium\n", _data_buffer, material.name.c_str());
      }
      entry.cam.medium_index = m;
    }

    if (get_param(material, "id")) {
      entry.id = _data_buffer;
    }

    int active_flag = 0;
    if (get_param(material, "active")) {
      if (sscanf(_data_buffer, "%d", &active_flag) == 1) {
        entry.active = (active_flag != 0);
      }
    }

    build_camera(entry.cam, entry.cam.position, entry.cam.direction, entry.cam.up, entry.cam.film_size, fov);
  }

  void parse_medium(const char* base_dir, const MaterialDefinition& material, SceneData& data, SceneLoaderContext& context, Scene& scene, const IORDatabase& database) {
    if (get_param(material, "id") == false) {
      log::warning("Medium does not have identifier - skipped");
      return;
    }

    std::string name = _data_buffer;

    float anisotropy = 0.0f;
    if (get_param(material, "g")) {
      float val = {};
      if (sscanf(_data_buffer, "%f", &val) == 1) {
        anisotropy = val;
      }
    }

    if (get_param(material, "anisotropy")) {
      float val = {};
      if (sscanf(_data_buffer, "%f", &val) == 1) {
        anisotropy = val;
      }
    }

    SpectralDistribution s_a = SpectralDistribution::null();
    if (get_param(material, "absorption")) {
      float val[3] = {};
      int params_read = sscanf(_data_buffer, "%f %f %f", val + 0, val + 1, val + 2);
      if (params_read == 3) {
        s_a = SpectralDistribution::rgb_reflectance({val[0], val[1], val[2]});
      } else if (params_read == 1) {
        s_a = SpectralDistribution::rgb_reflectance({val[0], val[0], val[0]});
      }
    }

    if (get_param(material, "absorbtion")) {
      log::warning("absorBtion used in medium: %s", name.c_str());
      float val[3] = {};
      int params_read = sscanf(_data_buffer, "%f %f %f", val + 0, val + 1, val + 2);
      if (params_read == 3) {
        s_a = SpectralDistribution::rgb_reflectance({val[0], val[1], val[2]});
      } else if (params_read == 1) {
        s_a = SpectralDistribution::rgb_reflectance({val[0], val[0], val[0]});
      }
    }

    SpectralDistribution s_t = SpectralDistribution::null();
    if (get_param(material, "scattering")) {
      float val[3] = {};
      int params_read = sscanf(_data_buffer, "%f %f %f", val + 0, val + 1, val + 2);
      if (params_read == 3) {
        s_t = SpectralDistribution::rgb_reflectance({val[0], val[1], val[2]});
      } else if (params_read == 1) {
        s_t = SpectralDistribution::rgb_reflectance({val[0], val[0], val[0]});
      }
    }

    if (get_param(material, "rayleigh")) {
      s_t = context.scattering_spectrums.rayleigh;

      float scale = 1.0f;
      char buffer[kDataBufferSize] = {};
      memcpy(buffer, _data_buffer, kDataBufferSize);
      auto params = split_params(buffer);
      for (uint64_t i = 0, e = params.size(); i < e; ++i) {
        if ((strcmp(params[i], "scale") == 0) && (i + 1 < e)) {
          scale = static_cast<float>(atof(params[i + 1]));
        }
      }
      s_t.scale(scale / s_t.maximum_spectral_power());
    }

    if (get_param(material, "mie")) {
      s_t = context.scattering_spectrums.mie;

      float scale = 1.0f;
      char buffer[kDataBufferSize] = {};
      memcpy(buffer, _data_buffer, kDataBufferSize);
      auto params = split_params(buffer);
      for (uint64_t i = 0, e = params.size(); i < e; ++i) {
        if ((strcmp(params[i], "scale") == 0) && (i + 1 < e)) {
          scale = static_cast<float>(atof(params[i + 1]));
        }
      }
      s_t.scale(scale / s_t.maximum_spectral_power());
    }

    if (get_param(material, "parametric")) {
      float3 color = {1.0f, 1.0f, 1.0f};
      float3 scattering_distances = {0.25f, 0.25f, 0.25f};

      float scale = 1.0f;
      char buffer[kDataBufferSize] = {};
      memcpy(buffer, _data_buffer, kDataBufferSize);
      auto params = split_params(buffer);
      for (uint64_t i = 0, e = params.size(); i < e; ++i) {
        if ((strcmp(params[i], "color") == 0) && (i + 3 < e)) {
          color = {
            static_cast<float>(atof(params[i + 1])),
            static_cast<float>(atof(params[i + 2])),
            static_cast<float>(atof(params[i + 3])),
          };
          i += 3;
        }
        if ((strcmp(params[i], "distance") == 0) && (i + 1 < e)) {
          float value = static_cast<float>(atof(params[i + 1]));
          scattering_distances = {value, value, value};
          i += 1;
        }
        if ((strcmp(params[i], "distances") == 0) && (i + 1 < e)) {
          scattering_distances = {
            static_cast<float>(atof(params[i + 1])),
            static_cast<float>(atof(params[i + 2])),
            static_cast<float>(atof(params[i + 3])),
          };
          i += 3;
        }
        if ((strcmp(params[i], "scale") == 0) && (i + 1 < e)) {
          scale = static_cast<float>(atof(params[i + 1]));
          i += 1;
        }
      }

      float3 albedo = {};
      float3 extinction = {};
      float3 scattering = {};
      subsurface::remap(color, scale * scattering_distances, albedo, extinction, scattering);

      float3 absorption = max({}, extinction - scattering);
      ETX_VALIDATE(absorption);

      s_t = SpectralDistribution::rgb_reflectance(scattering);
      s_a = SpectralDistribution::rgb_reflectance(absorption);
    }

    bool explicit_connections = true;
    if (get_param(material, "enclosed")) {
      explicit_connections = false;
    }

    Medium::Class cls = Medium::Class::Homogeneous;

    char tmp_buffer[2048] = {};

    if (get_param(material, "volume")) {
      if (strlen(_data_buffer) > 0) {
        snprintf(tmp_buffer, sizeof(tmp_buffer), "%s%s", base_dir, _data_buffer);
        cls = Medium::Class::Heterogeneous;
      }
    }

    context.add_medium(scene, data, cls, name.c_str(), tmp_buffer, s_a, s_t, anisotropy, explicit_connections);
  }

  void parse_directional_light(const char* base_dir, const MaterialDefinition& material, SceneData& data, SceneLoaderContext& context, Scene& scene, const IORDatabase& database) {
    auto& instance = data.emitter_instances.emplace_back(EmitterProfile::Class::Directional);
    instance.profile = uint32_t(data.emitter_profiles.size());

    auto& e = data.emitter_profiles.emplace_back(EmitterProfile::Class::Directional);
    if (get_param(material, "color")) {
      char buffer[kDataBufferSize] = {};
      memcpy(buffer, _data_buffer, kDataBufferSize);
      e.emission.spectrum_index = data.add_spectrum(load_illuminant_spectrum(data, buffer));
    } else {
      e.emission.spectrum_index = data.add_spectrum(SpectralDistribution::rgb_luminance({1.0f, 1.0f, 1.0f}));
    }

    e.direction = float3{1.0f, 1.0f, 1.0f};
    if (get_param(material, "direction")) {
      float value[3] = {};
      if (sscanf(_data_buffer, "%f %f %f", value + 0, value + 1, value + 2) == 3) {
        e.direction = {value[0], value[1], value[2]};
      }
    }
    e.direction = normalize(e.direction);

    if (get_param(material, "image")) {
      char tmp_buffer[2048] = {};
      snprintf(tmp_buffer, sizeof(tmp_buffer), "%s/%s", base_dir, _data_buffer);
      e.emission.image_index = context.add_image(tmp_buffer, Image::Regular, {}, {1.0f, 1.0f});
    }

    if (get_param(material, "angular_diameter")) {
      float val = {};
      if (sscanf(_data_buffer, "%f", &val) == 1) {
        e.angular_size = val * kPi / 180.0f;
      }
    }

    if (get_param(material, "ext_medium")) {
      auto m = context.mediums.find(_data_buffer);
      if (m == kInvalidIndex) {
        log::warning("Medium %s was not declared, but used in directional emitter as external medium\n", _data_buffer);
      }
      e.medium_index = m;
    }
  }

  void parse_env_light(const char* base_dir, const MaterialDefinition& material, SceneData& data, SceneLoaderContext& context, Scene& scene, const IORDatabase& database) {
    auto& instance = data.emitter_instances.emplace_back(EmitterProfile::Class::Environment);
    instance.profile = uint32_t(data.emitter_profiles.size());

    auto& e = data.emitter_profiles.emplace_back(EmitterProfile::Class::Environment);

    char tmp_buffer[2048] = {};
    if (get_param(material, "image")) {
      snprintf(tmp_buffer, sizeof(tmp_buffer), "%s/%s", base_dir, _data_buffer);
    }

    float rotation = 0.0f;
    if (get_param(material, "rotation")) {
      rotation = -static_cast<float>(atof(_data_buffer)) / 360.0f;
    }

    float u_scale = 1.0f;
    if (get_param(material, "scale")) {
      float val = {};
      if (sscanf(_data_buffer, "%f", &val) == 1) {
        u_scale = val;
      }
    }
    e.emission.image_index = context.add_image(tmp_buffer, Image::BuildSamplingTable | Image::RepeatU, {rotation, 0.0f}, {u_scale, 1.0f});

    if (get_param(material, "color")) {
      char buffer[kDataBufferSize] = {};
      memcpy(buffer, _data_buffer, kDataBufferSize);
      e.emission.spectrum_index = data.add_spectrum(load_illuminant_spectrum(data, buffer));
    } else {
      e.emission.spectrum_index = data.add_spectrum(SpectralDistribution::rgb_luminance({1.0f, 1.0f, 1.0f}));
    }

    if (get_param(material, "ext_medium")) {
      auto m = context.mediums.find(_data_buffer);
      if (m == kInvalidIndex) {
        log::warning("Medium %s was not declared, but used in environment emitter as external medium\n", _data_buffer);
      }
      e.medium_index = m;
    }
  }

  void parse_atmosphere_light(const char* base_dir, const MaterialDefinition& material, SceneData& data, SceneLoaderContext& context, Scene& scene, const IORDatabase& database,
    TaskScheduler& scheduler) {
    float quality = 1.0f;
    if (get_param(material, "quality")) {
      float val = {};
      if (sscanf(_data_buffer, "%f", &val) == 1) {
        quality = val;
      }
    }

    float scale = 1.0f;
    if (get_param(material, "scale")) {
      float val = {};
      if (sscanf(_data_buffer, "%f", &val) == 1) {
        scale = val;
      }
    }
    float sun_scale = 1.0f;
    if (get_param(material, "sun_scale")) {
      float val = {};
      if (sscanf(_data_buffer, "%f", &val) == 1) {
        sun_scale = val;
      }
    }
    float sky_scale = 1.0f;
    if (get_param(material, "sky_scale")) {
      float val = {};
      if (sscanf(_data_buffer, "%f", &val) == 1) {
        sky_scale = val;
      }
    }

    float3 direction = normalize(float3{1.0f, 1.0f, 1.0f});
    float angular_size = 0.5422f * (kPi / 180.0f);

    scattering::Parameters scattering_parameters = {};

    if (get_param(material, "direction")) {
      float value[3] = {};
      if (sscanf(_data_buffer, "%f %f %f", value + 0, value + 1, value + 2) == 3) {
        direction = normalize(float3{value[0], value[1], value[2]});
      }
    }
    if (get_param(material, "angular_diameter")) {
      float val = {};
      if (sscanf(_data_buffer, "%f", &val) == 1) {
        angular_size = val * (kPi / 180.0f);
      }
    }
    if (get_param(material, "anisotropy")) {
      float val = {};
      if (sscanf(_data_buffer, "%f", &val) == 1) {
        scattering_parameters.anisotropy = val;
      }
    }
    if (get_param(material, "altitude")) {
      float val = {};
      if (sscanf(_data_buffer, "%f", &val) == 1) {
        scattering_parameters.altitude = val;
      }
    }
    if (get_param(material, "rayleigh")) {
      float val = {};
      if (sscanf(_data_buffer, "%f", &val) == 1) {
        scattering_parameters.rayleigh_scale = val;
      }
    }
    if (get_param(material, "mie")) {
      float val = {};
      if (sscanf(_data_buffer, "%f", &val) == 1) {
        scattering_parameters.mie_scale = val;
      }
    }
    if (get_param(material, "ozone")) {
      float val = {};
      if (sscanf(_data_buffer, "%f", &val) == 1) {
        scattering_parameters.ozone_scale = val;
      }
    }

    float radiance_scale = scale * (kDoublePi * (1.0f - cosf(0.5f * angular_size)));
    auto sun_spectrum = SpectralDistribution::from_black_body(5900.0f, radiance_scale);

    constexpr uint2 kSunImageDimensions = uint2{128u, 128u};
    constexpr uint32_t kSkyImageBaseDimensions = 1024u;

    uint2 sky_image_dimensions = uint2{2u * kSkyImageBaseDimensions, kSkyImageBaseDimensions};
    sky_image_dimensions.x = max(64u, uint32_t(sky_image_dimensions.x * quality));
    sky_image_dimensions.y = max(64u, uint32_t(sky_image_dimensions.y * quality));

    {
      auto& instance = data.emitter_instances.emplace_back(EmitterProfile::Class::Directional);
      instance.profile = uint32_t(data.emitter_profiles.size());

      auto& d = data.emitter_profiles.emplace_back(EmitterProfile::Class::Directional);
      d.emission.spectrum_index = data.add_spectrum(sun_spectrum);
      data.spectrum_values[d.emission.spectrum_index].scale(sun_scale);
      d.angular_size = angular_size;
      d.direction = direction;

      if (angular_size > 0.0f) {
        d.emission.image_index = context.add_image(nullptr, kSunImageDimensions, Image::Delay, {}, {1.0f, 1.0f});
        auto& img = context.images.get(d.emission.image_index);
        scattering::generate_sun_image(scattering_parameters, kSunImageDimensions, direction, angular_size, img.pixels.f32.a, context.scattering_spectrums, scheduler);
      }
    }

    {
      auto& instance = data.emitter_instances.emplace_back(EmitterProfile::Class::Environment);
      instance.profile = uint32_t(data.emitter_profiles.size());

      auto& e = data.emitter_profiles.emplace_back(EmitterProfile::Class::Environment);
      e.emission.spectrum_index = data.add_spectrum(sun_spectrum);
      data.spectrum_values[e.emission.spectrum_index].scale(sky_scale);
      uint32_t image_options = Image::BuildSamplingTable | Image::Delay;
      e.emission.image_index = context.add_image(nullptr, sky_image_dimensions, image_options, {}, {1.0f, 1.0f});
      e.direction = direction;

      auto& img = context.images.get(e.emission.image_index);
      scattering::generate_sky_image(scattering_parameters, sky_image_dimensions, direction, data.atmosphere_extinction, img.pixels.f32.a, context.scattering_spectrums, scheduler);
    }
  }

  void parse_spectrum(const char* base_dir, const MaterialDefinition& material, SceneData& data, SceneLoaderContext& context, Scene& scene, const IORDatabase& database) {
    if (get_param(material, "id") == false) {
      log::warning("Spectrum does not have identifier - skipped");
      return;
    }
    std::string name = _data_buffer;

    bool initialized = false;
    bool illuminant = false;

    float scale = 1.0f;

    if (get_param(material, "scale")) {
      scale = static_cast<float>(atof(_data_buffer));
    }

    if (get_param(material, "illuminant")) {
      illuminant = true;
    }

    if (get_param(material, "rgb")) {
      char buffer[kDataBufferSize] = {};
      memcpy(buffer, _data_buffer, kDataBufferSize);
      auto params = split_params(buffer);

      if (params.size() < 3) {
        log::warning("Spectrum `%s` uses RGB but did not provide all values - skipped", name.c_str());
        return;
      }

      float3 value = {
        static_cast<float>(atof(params[0])),
        static_cast<float>(atof(params[1])),
        static_cast<float>(atof(params[2])),
      };
      value = gamma_to_linear(value);

      data.add_spectrum(name.c_str(), illuminant ? SpectralDistribution::rgb_luminance(value) : SpectralDistribution::rgb_reflectance(value));
      initialized = true;
    } else if (get_param(material, "blackbody")) {
      char buffer[kDataBufferSize] = {};
      memcpy(buffer, _data_buffer, kDataBufferSize);
      auto params = split_params(buffer);
      if (params.size() < 1) {
        log::warning("Spectrum `%s` uses blackbody but did not provide temperature value - skipped", name.c_str());
        return;
      }

      float t = static_cast<float>(atof(params[0]));
      data.add_spectrum(name.c_str(), SpectralDistribution::from_black_body(t, scale));
      initialized = true;
    } else if (get_param(material, "nblackbody")) {
      char buffer[kDataBufferSize] = {};
      memcpy(buffer, _data_buffer, kDataBufferSize);
      auto params = split_params(buffer);
      if (params.size() < 1) {
        log::warning("Spectrum `%s` uses nblackbody but did not provide temperature value - skipped", name.c_str());
        return;
      }

      float scale = 1.0f;
      for (uint64_t i = 0, e = params.size(); i < e; ++i) {
        if ((i + 1 < e) && (strcmp(params[i], "scale") == 0)) {
          scale = static_cast<float>(atof(params[i + 1]));
          ++i;
        }
      }

      float t = static_cast<float>(atof(params[0]));
      data.add_spectrum(name.c_str(), SpectralDistribution::from_normalized_black_body(t, scale));
      initialized = true;
    }

    bool have_samples = get_param(material, "samples");

    if ((have_samples == false) && (initialized == false)) {
      log::warning("Spectrum `%s` does not have samples or RBG or (n)blackbody - skipped", name.c_str());
      return;
    } else if (initialized && have_samples) {
      log::warning("Spectrum `%s` uses both RGB or (n)blackbody and samples set - samples will be used", name.c_str());
    } else if (initialized == false) {
      char buffer[kDataBufferSize] = {};
      memcpy(buffer, _data_buffer, kDataBufferSize);
      auto params = split_params(buffer);
      if (params.size() % 2) {
        log::warning("Spectrum `%s` have uneven number samples - skipped", name.c_str());
        return;
      }

      std::vector<float2> samples;
      samples.reserve(params.size() / 2 + 1);

      for (uint64_t i = 0, e = params.size(); i < e; i += 2) {
        float2& smp = samples.emplace_back();
        smp.x = static_cast<float>(atof(params[i + 0]));
        smp.y = static_cast<float>(atof(params[i + 1]));
      }

      if (samples.empty() == false) {
        log::warning("Spectrum `%s` sample set is empty - skipped", name.c_str());
      }

      auto spectrum = SpectralDistribution::from_samples(samples.data(), samples.size());

      if (get_param(material, "normalize")) {
        float3 xyz = spectrum.integrate_to_xyz();
        bool normalize_rgb = strcmp(_data_buffer, "luminance") != 0;
        float3 rgb = spectrum::xyz_to_rgb(xyz);
        float lum = normalize_rgb ? fmaxf(fmaxf(0.0f, rgb.x), fmaxf(rgb.y, rgb.z)) : xyz.y;
        if (lum > kEpsilon) {
          spectrum.scale(1.0f / lum);
        }
      }
      data.add_spectrum(name.c_str(), spectrum);
    }

    uint32_t i = data.find_spectrum(name.c_str());
    if (i != kInvalidIndex) {
      data.spectrum_values[i].scale(scale);
    }
  }

  void parse_material(const char* base_dir, const MaterialDefinition& material, SceneData& data, SceneLoaderContext& context, Scene& scene, const IORDatabase& database) {
    auto& material_mapping = data.material_mapping;

    uint32_t material_index = kInvalidIndex;

    if (material_mapping.count(material.name) == 0) {
      material_index = data.add_material(material.name.c_str());
    } else {
      material_index = material_mapping.at(material.name);
    }

    auto& mtl = data.materials[material_index];

    mtl.cls = Material::Class::Diffuse;
    mtl.emission = {};
    mtl.emission_collimation = 0.0f;

    if (get_param(material, "base")) {
      auto i = material_mapping.find(_data_buffer);
      if (i != material_mapping.end()) {
        mtl = data.materials[i->second];
      }
    }

    if (get_param(material, "Kd")) {
      char buffer[kDataBufferSize] = {};
      memcpy(buffer, _data_buffer, kDataBufferSize);
      mtl.scattering.spectrum_index = load_reflectance_spectrum(data, buffer);
    }

    if (get_param(material, "Ks")) {
      char buffer[kDataBufferSize] = {};
      memcpy(buffer, _data_buffer, kDataBufferSize);
      mtl.reflectance.spectrum_index = load_reflectance_spectrum(data, buffer);
    }

    if (get_param(material, "Kt")) {
      char buffer[kDataBufferSize] = {};
      memcpy(buffer, _data_buffer, kDataBufferSize);
      mtl.scattering.spectrum_index = load_reflectance_spectrum(data, buffer);
    }

    SpectralDistribution emission_spd = SpectralDistribution::null();

    float pending_scale = 1.0f;

    bool is_emitter = false;

    bool emission_spd_defined = false;

    float collimation = mtl.emission_collimation;

    if (get_param(material, "Ke")) {
      is_emitter = true;
      emission_spd = load_illuminant_spectrum(data, _data_buffer);
      emission_spd_defined = true;
      auto map_ke_it = material.properties.find("map_Ke");
      if (map_ke_it != material.properties.end() && get_file(base_dir, map_ke_it->second)) {
        mtl.emission.image_index = context.add_image(_data_buffer, Image::RepeatU | Image::RepeatV | Image::BuildSamplingTable, {}, {1.0f, 1.0f});
      }
    }

    if (get_param(material, "emitter")) {
      is_emitter = true;
      auto params = split_params(_data_buffer);
      for (uint64_t i = 0, end = params.size(); i < end; ++i) {
        if ((strcmp(params[i], "image") == 0) && (i + 1 < end) && get_file(base_dir, params[i + 1])) {
          mtl.emission.image_index = context.add_image(_data_buffer, Image::RepeatU | Image::RepeatV | Image::BuildSamplingTable, {}, {1.0f, 1.0f});
        } else if (strcmp(params[i], "twosided") == 0) {
          mtl.two_sided = 1u;
        } else if ((strcmp(params[i], "collimated") == 0) && (i + 1 < end)) {
          collimation = static_cast<float>(atof(params[i + 1]));
          i += 1;
        } else if ((strcmp(params[i], "color") == 0) && (i + 3 < end)) {
          float3 value = {
            static_cast<float>(atof(params[i + 1])),
            static_cast<float>(atof(params[i + 2])),
            static_cast<float>(atof(params[i + 3])),
          };
          emission_spd = SpectralDistribution::rgb_luminance(value);
          emission_spd_defined = true;
          i += 3;
        } else if ((strcmp(params[i], "blackbody") == 0) && (i + 1 < end)) {
          emission_spd = SpectralDistribution::from_black_body(static_cast<float>(atof(params[i + 1])), 1.0f);
          emission_spd_defined = true;
          i += 1;
        } else if ((strcmp(params[i], "nblackbody") == 0) && (i + 1 < end)) {
          emission_spd = SpectralDistribution::from_normalized_black_body(static_cast<float>(atof(params[i + 1])), 1.0f);
          emission_spd_defined = true;
          i += 1;
        } else if ((strcmp(params[i], "scale") == 0) && (i + 1 < end)) {
          pending_scale *= static_cast<float>(atof(params[i + 1]));
          i += 1;
        } else if ((strcmp(params[i], "spectrum") == 0) && (i + 1 < end)) {
          char buffer[2048] = {};
          snprintf(buffer, sizeof(buffer), "%s/%s", base_dir, params[i + 1]);
          auto cls = SpectralDistribution::load_from_file(buffer, emission_spd, nullptr, false);
          if (cls != SpectralDistribution::Class::Illuminant) {
            log::warning("Spectrum %s is not illuminant", buffer);
          }
          emission_spd_defined = true;
          i += 1;
        }
      }

      collimation = clamp(collimation, 0.0f, 1.0f);
    }

    if (is_emitter) {
      emission_spd.scale(pending_scale);
      mtl.emission_collimation = collimation;
      if (emission_spd_defined && (emission_spd.luminance() > 0.0f)) {
        mtl.emission.spectrum_index = data.add_spectrum(emission_spd);
      } else if (emission_spd_defined == false && mtl.emission.spectrum_index != kInvalidIndex) {
        // keep existing spectrum (e.g. inherited from base)
      } else {
        mtl.emission.spectrum_index = kInvalidIndex;
      }

      if (mtl.emission.spectrum_index == kInvalidIndex) {
        mtl.emission.image_index = kInvalidIndex;
      }
    } else if (mtl.emission.spectrum_index == kInvalidIndex) {
      mtl.emission.image_index = kInvalidIndex;
      mtl.emission_collimation = 0.0f;
    }

    if (get_param(material, "two_sided")) {
      int val = 0;
      if (sscanf(_data_buffer, "%d", &val) == 1) {
        mtl.two_sided = (val != 0) ? 1u : 0u;
      } else {
        mtl.two_sided = ((strcmp(_data_buffer, "true") == 0) || (strcmp(_data_buffer, "on") == 0)) ? 1u : 0u;
      }
    }

    if (get_param(material, "opacity")) {
      float val = 1.0f;
      if (sscanf(_data_buffer, "%f", &val) == 1) {
        mtl.opacity = clamp(val, 0.0f, 1.0f);
      }
    }

    if (get_param(material, "Pr")) {
      float4 value = {};
      if (sscanf(_data_buffer, "%f %f", &value.x, &value.y) == 2) {
        mtl.roughness.value = sqr(value);
      } else if (sscanf(_data_buffer, "%f", &value.x) == 1) {
        mtl.roughness = {sqr(value.x), sqr(value.x)};
      }
    }

    if (get_param(material, "metalness")) {
      float val = 0.0f;
      if (sscanf(_data_buffer, "%f", &val) == 1) {
        mtl.metalness.value = {val, val, val, val};
      }
    }

    if (get_param(material, "transmission")) {
      float val = 0.0f;
      if (sscanf(_data_buffer, "%f", &val) == 1) {
        mtl.transmission.value = {val, val, val, val};
      }
    }

    if (get_param(material, "map_Pr")) {
      char buffer[kDataBufferSize] = {};
      memcpy(buffer, _data_buffer, kDataBufferSize);
      auto params = split_params(buffer);
      const char* path = (params.empty() == false) ? params[0] : nullptr;
      int channel = 0;
      for (uint64_t i = 0, e = params.size(); i < e; ++i) {
        if ((strcmp(params[i], "channel") == 0) && (i + 1 < e)) {
          channel = std::max(0, atoi(params[i + 1]));
          ++i;
        }
      }
      if (path && get_file(base_dir, path)) {
        mtl.roughness.image_index = context.add_image(_data_buffer, Image::RepeatU | Image::RepeatV, {}, {1.0f, 1.0f});
        mtl.roughness.channel = static_cast<uint32_t>(channel);
      }
    }

    if (get_param(material, "map_Ml")) {
      char buffer[kDataBufferSize] = {};
      memcpy(buffer, _data_buffer, kDataBufferSize);
      auto params = split_params(buffer);
      const char* path = (params.empty() == false) ? params[0] : nullptr;
      int channel = 0;
      for (uint64_t i = 0, e = params.size(); i < e; ++i) {
        if ((strcmp(params[i], "channel") == 0) && (i + 1 < e)) {
          channel = std::max(0, atoi(params[i + 1]));
          ++i;
        }
      }
      if (path && get_file(base_dir, path)) {
        mtl.metalness.image_index = context.add_image(_data_buffer, Image::RepeatU | Image::RepeatV, {}, {1.0f, 1.0f});
        mtl.metalness.channel = static_cast<uint32_t>(channel);
      }
    }

    if (get_param(material, "map_Tm")) {
      char buffer[kDataBufferSize] = {};
      memcpy(buffer, _data_buffer, kDataBufferSize);
      auto params = split_params(buffer);
      const char* path = (params.empty() == false) ? params[0] : nullptr;
      int channel = 0;
      for (uint64_t i = 0, e = params.size(); i < e; ++i) {
        if ((strcmp(params[i], "channel") == 0) && (i + 1 < e)) {
          channel = std::max(0, atoi(params[i + 1]));
          ++i;
        }
      }
      if (path && get_file(base_dir, path)) {
        mtl.transmission.image_index = context.add_image(_data_buffer, Image::RepeatU | Image::RepeatV, {}, {1.0f, 1.0f});
        mtl.transmission.channel = static_cast<uint32_t>(channel);
      }
    }

    if (get_param(material, "map_Kd")) {
      if (get_file(base_dir, _data_buffer)) {
        mtl.scattering.image_index = context.add_image(_data_buffer, Image::RepeatU | Image::RepeatV, {}, {1.0f, 1.0f});
      }
    }

    if (get_param(material, "map_Ks")) {
      if (get_file(base_dir, _data_buffer)) {
        mtl.reflectance.image_index = context.add_image(_data_buffer, Image::RepeatU | Image::RepeatV, {}, {1.0f, 1.0f});
      }
    }

    if (get_param(material, "map_Kt")) {
      if (get_file(base_dir, _data_buffer)) {
        mtl.scattering.image_index = context.add_image(_data_buffer, Image::RepeatU | Image::RepeatV, {}, {1.0f, 1.0f});
      }
    }

    if (get_param(material, "material")) {
      char buffer[kDataBufferSize] = {};
      memcpy(buffer, _data_buffer, kDataBufferSize);
      auto params = split_params(buffer);
      for (uint64_t i = 0, e = params.size(); i < e; ++i) {
        if ((strcmp(params[i], "class") == 0) && (i + 1 < e)) {
          mtl.cls = material_string_to_class(params[i + 1]);
          i += 1;
        }
      }
    }

    if (get_param(material, "diffuse")) {
      uint32_t var = 0;
      if (sscanf(_data_buffer, "%u", &var) == 1) {
        mtl.diffuse_variation = var;
      }
    }

    auto load_ior = [&](RefractiveIndex& target, const char* buffer) {
      float2 values = {};
      int values_read = sscanf(buffer, "%f %f", &values.x, &values.y);
      target.cls = SpectralDistribution::Class::Dielectric;
      if (values_read == 1) {
        target.eta_index = data.add_spectrum(SpectralDistribution::constant(values.x));
        target.k_index = kInvalidIndex;
      } else if (values_read == 2) {
        target.cls = SpectralDistribution::Class::Conductor;
        target.eta_index = data.add_spectrum(SpectralDistribution::constant(values.x));
        target.k_index = data.add_spectrum(SpectralDistribution::constant(values.y));
      } else {
        SpectralDistribution eta_spd = {};
        SpectralDistribution k_spd = {};
        SpectralDistribution::Class cls = SpectralDistribution::Class::Invalid;
        if (load_ior_from_identifier(buffer, database, eta_spd, k_spd, cls) == false) {
          std::filesystem::path fallback = locate_spectrum_file(buffer, {});
          if (fallback.empty() == false) {
            cls = RefractiveIndex::load_from_file(fallback.string().c_str(), eta_spd, k_spd);
          }
        }

        if (cls == SpectralDistribution::Class::Invalid) {
          log::warning("Unable to load IOR spectrum `%s`, falling back to 1.5 dielectric", buffer);
          cls = SpectralDistribution::Class::Dielectric;
          eta_spd = SpectralDistribution::constant(1.5f);
          k_spd = SpectralDistribution::constant(0.0f);
        }

        target.cls = cls;
        target.eta_index = data.add_spectrum(eta_spd);
        if (cls == SpectralDistribution::Class::Conductor) {
          target.k_index = data.add_spectrum(k_spd);
        } else {
          target.k_index = k_spd.empty() ? data.add_spectrum(SpectralDistribution::constant(0.0f)) : data.add_spectrum(k_spd);
        }
      }
    };

    if (get_param(material, "int_ior")) {
      load_ior(mtl.int_ior, _data_buffer);
    } else {
      mtl.int_ior.cls = SpectralDistribution::Class::Dielectric;
      mtl.int_ior.eta_index = data.add_spectrum(SpectralDistribution::constant(1.5f));
      mtl.int_ior.k_index = data.add_spectrum(SpectralDistribution::constant(0.0f));
    }

    if (get_param(material, "ext_ior")) {
      load_ior(mtl.ext_ior, _data_buffer);
    } else {
      mtl.ext_ior.cls = SpectralDistribution::Class::Dielectric;
      mtl.ext_ior.eta_index = data.add_spectrum(SpectralDistribution::constant(1.0f));
      mtl.ext_ior.k_index = data.add_spectrum(SpectralDistribution::constant(0.0f));
    }

    if (get_param(material, "int_medium")) {
      auto m = context.mediums.find(_data_buffer);
      if (m == kInvalidIndex) {
        log::warning("Medium %s was not declared, but used in material %s as internal medium", _data_buffer, material.name.c_str());
      }
      mtl.int_medium = m;
    }

    if (get_param(material, "ext_medium")) {
      auto m = context.mediums.find(_data_buffer);
      if (m == kInvalidIndex) {
        log::warning("Medium %s was not declared, but used in material %s as external medium\n", _data_buffer, material.name.c_str());
      }
      mtl.ext_medium = m;
    }

    if (get_param(material, "normalmap")) {
      char buffer[kDataBufferSize] = {};
      memcpy(buffer, _data_buffer, kDataBufferSize);
      auto params = split_params(buffer);
      for (uint64_t i = 0, e = params.size(); i < e; ++i) {
        if ((strcmp(params[i], "image") == 0) && (i + 1 < e)) {
          char tmp_buffer[1024] = {};
          snprintf(tmp_buffer, sizeof(tmp_buffer), "%s/%s", base_dir, params[i + 1]);
          mtl.normal_image_index = context.add_image(tmp_buffer, Image::RepeatU | Image::RepeatV | Image::SkipSRGBConversion, {}, {1.0f, 1.0f});
          i += 1;
        }
        if ((strcmp(params[i], "scale") == 0) && (i + 1 < e)) {
          mtl.normal_scale = static_cast<float>(atof(params[i + 1]));
          i += 1;
        }
      }
    }

    if (get_param(material, "thinfilm")) {
      char buffer[kDataBufferSize] = {};
      memcpy(buffer, _data_buffer, kDataBufferSize);
      auto params = split_params(buffer);

      for (uint64_t i = 0, e = params.size(); i < e; ++i) {
        if ((strcmp(params[i], "image") == 0) && (i + 1 < e)) {
          char tmp_buffer[1024] = {};
          snprintf(tmp_buffer, sizeof(tmp_buffer), "%s/%s", base_dir, params[i + 1]);
          mtl.thinfilm.thinkness_image = context.add_image(tmp_buffer, Image::RepeatU | Image::RepeatV, {}, {1.0f, 1.0f});
          i += 1;
        }

        if ((strcmp(params[i], "range") == 0) && (i + 2 < e)) {
          mtl.thinfilm.min_thickness = static_cast<float>(atof(params[i + 1]));
          mtl.thinfilm.max_thickness = static_cast<float>(atof(params[i + 2]));
          i += 2;
        }

        if ((strcmp(params[i], "ior") == 0) && (i + 1 < e)) {
          float value = 0.0f;
          if (sscanf(params[i + 1], "%f", &value) == 1) {
            mtl.thinfilm.ior.cls = SpectralDistribution::Class::Dielectric;
            mtl.thinfilm.ior.eta_index = data.add_spectrum(SpectralDistribution::constant(value));
            mtl.thinfilm.ior.k_index = kInvalidIndex;
          } else {
            SpectralDistribution eta_spd = {};
            SpectralDistribution k_spd = {};
            SpectralDistribution::Class cls = SpectralDistribution::Class::Invalid;
            if (load_ior_from_identifier(params[i + 1], database, eta_spd, k_spd, cls) == false) {
              std::filesystem::path fallback = locate_spectrum_file(params[i + 1], {});
              if (fallback.empty() == false) {
                cls = RefractiveIndex::load_from_file(fallback.string().c_str(), eta_spd, k_spd);
              }
            }

            if (cls == SpectralDistribution::Class::Invalid) {
              log::warning("Unable to load thinfilm IOR `%s`, using dielectric 1.5", params[i + 1]);
              cls = SpectralDistribution::Class::Dielectric;
              eta_spd = SpectralDistribution::constant(1.5f);
              k_spd = SpectralDistribution::constant(0.0f);
            }

            mtl.thinfilm.ior.cls = cls;
            mtl.thinfilm.ior.eta_index = data.add_spectrum(eta_spd);
            if (cls == SpectralDistribution::Class::Conductor) {
              mtl.thinfilm.ior.k_index = data.add_spectrum(k_spd);
            } else {
              mtl.thinfilm.ior.k_index = k_spd.empty() ? kInvalidIndex : data.add_spectrum(k_spd);
            }
          }
        }
      }
    }

    if (get_param(material, "subsurface")) {
      mtl.subsurface.cls = SubsurfaceMaterial::Class::RandomWalk;

      float subsurface_scale = 1.0f;
      float3 scattering_distances = {1.0f, 0.2f, 0.04f};

      char buffer[kDataBufferSize] = {};
      memcpy(buffer, _data_buffer, kDataBufferSize);
      auto params = split_params(buffer);
      for (uint64_t i = 0, e = params.size(); i < e; ++i) {
        if ((strcmp(params[i], "path") == 0) && (i + 1 < e)) {
          bool is_refraction = (strcmp(params[i + 1], "refracted") == 0) || (strcmp(params[i + 1], "refraction") == 0) || (strcmp(params[i + 1], "refract") == 0);
          mtl.subsurface.path = is_refraction ? SubsurfaceMaterial::Path::Refracted : SubsurfaceMaterial::Path::Diffuse;
        }

        if ((strcmp(params[i], "distances") == 0) && (i + 3 < e)) {
          scattering_distances.x = static_cast<float>(atof(params[i + 1]));
          scattering_distances.y = static_cast<float>(atof(params[i + 2]));
          scattering_distances.z = static_cast<float>(atof(params[i + 3]));
          i += 3;
        }
      }
    }
  }
};

void SceneSerialization::parse_material_definitions(const char* base_dir, const std::vector<MaterialDefinition>& materials, SceneData& data, SceneLoaderContext& context,
  Scene& scene, const IORDatabase& database, TaskScheduler& scheduler) {
  for (const auto& material : materials) {
    if (material.name == "et::camera") {
      _private->parse_camera(base_dir, material, data, context, scene, database);
    } else if (material.name == "et::medium") {
      _private->parse_medium(base_dir, material, data, context, scene, database);
    } else if (material.name == "et::dir") {
      _private->parse_directional_light(base_dir, material, data, context, scene, database);
    } else if (material.name == "et::env") {
      _private->parse_env_light(base_dir, material, data, context, scene, database);
    } else if (material.name == "et::atmosphere") {
      _private->parse_atmosphere_light(base_dir, material, data, context, scene, database, scheduler);
    } else if (material.name == "et::spectrum") {
      _private->parse_spectrum(base_dir, material, data, context, scene, database);
    } else {
      _private->parse_material(base_dir, material, data, context, scene, database);
    }
  }
}

ETX_IMPLEMENT_PIMPL(SceneSerialization);

SceneSerialization::SceneSerialization() {
  ETX_PIMPL_INIT(SceneSerialization);
}

SceneSerialization::~SceneSerialization() {
  ETX_PIMPL_CLEANUP(SceneSerialization);
}

bool SceneSerialization::save_to_file(const SceneData& data, const std::filesystem::path& path) {
  if (!_private->prepare_data(data)) {
    return false;
  }
  return _private->write_to_file(path);
}

bool SceneSerialization::load_from_file(const std::filesystem::path& path, SceneData& data, const char* materials_file, SceneLoaderContext& context, Scene& scene,
  const IORDatabase& database, TaskScheduler& scheduler) {
  return _private->load_from_file(path, data, materials_file, context, scene, database, scheduler);
}

}  // namespace etx
