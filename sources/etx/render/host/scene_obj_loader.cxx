#include <etx/render/host/scene_obj_loader.hxx>

#include <etx/core/core.hxx>
#include <etx/core/log.hxx>
#include <etx/render/host/scene_data.hxx>
#include <etx/render/shared/scene.hxx>
#include <etx/render/shared/ior_database.hxx>
#include <etx/render/shared/vertex_utils.hxx>
#include <etx/render/host/tasks.hxx>
#include <etx/render/host/scene_serialization.hxx>
#include <etx/render/host/scene_loader_utils.hxx>

#include <algorithm>
#include <tiny_obj_loader.hxx>

namespace etx {

namespace {
bool load_materials(SceneData& data, const IORDatabase& ior_database, TaskScheduler& scheduler, const char* materials_file, const char* base_dir,
  const std::vector<etx::MaterialDefinition>& embedded_materials) {
  SceneSerialization serialization;

  if (materials_file && materials_file[0]) {
    if (!serialization.parse_materials_file(materials_file, base_dir, data, ior_database, scheduler)) {
      log::warning("Failed to parse materials from %s", materials_file);
      return false;
    }
  } else if (embedded_materials.empty() == false) {
    serialization.parse_material_definitions(base_dir, embedded_materials, data, ior_database, scheduler);
  }

  data.images.load_images(scheduler);
  return true;
}

etx::MaterialDefinition convert_tinyobj_to_material_definition(const tinyobj::material_t& material) {
  etx::MaterialDefinition result;
  result.name = material.name;
  trim_whitespace(result.name);
  std::transform(result.name.begin(), result.name.end(), result.name.begin(), ::tolower);

  for (const auto& param : material.unknown_parameter) {
    result.properties[param.first] = param.second;
  }

  char buffer[256] = {};
  if (material.shininess != 0.0f) {
    snprintf(buffer, sizeof(buffer), "%.6f", material.shininess);
    result.properties["Ns"] = buffer;
  }
  if (material.ior != 1.0f) {
    snprintf(buffer, sizeof(buffer), "%.6f", material.ior);
    result.properties["Ni"] = buffer;
  }
  if (material.dissolve != 1.0f) {
    snprintf(buffer, sizeof(buffer), "%.6f", material.dissolve);
    result.properties["d"] = buffer;
  }
  if (!material.diffuse_texname.empty()) {
    result.properties["map_Kd"] = material.diffuse_texname;
  }
  if (!material.specular_texname.empty()) {
    result.properties["map_Ks"] = material.specular_texname;
  }
  if (!material.transmittance_texname.empty()) {
    result.properties["map_Kt"] = material.transmittance_texname;
  }
  if (!material.emissive_texname.empty()) {
    result.properties["map_Ke"] = material.emissive_texname;
  }
  if (!material.roughness_texname.empty()) {
    result.properties["map_Pr"] = material.roughness_texname;
  }
  if (!material.metallic_texname.empty()) {
    result.properties["map_Ml"] = material.metallic_texname;
  }
  if (!material.normal_texname.empty()) {
    result.properties["normalmap"] = std::string("image ") + material.normal_texname;
  }

  return result;
}

}  // namespace

struct SceneObjLoaderImpl {};

uint32_t load_from_obj_file(const char* obj_file_name, const char* mtl_file_name, SceneData& data, const IORDatabase& ior_database, TaskScheduler& scheduler) {
  auto start_time = std::chrono::high_resolution_clock::now();

  auto& triangles = data.triangles;
  auto& vertices = data.vertices;
  auto& material_mapping = data.material_mapping;

  tinyobj::attrib_t obj_attrib;
  std::vector<tinyobj::shape_t> obj_shapes;
  std::vector<tinyobj::material_t> obj_materials;
  std::string warnings;
  std::string errors;

  constexpr auto kDataBufferSize = 2048llu;
  static char base_dir[kDataBufferSize] = {};
  get_base_directory(obj_file_name, base_dir, sizeof(base_dir));

  std::string materials_to_load = {};

  if ((mtl_file_name == nullptr) || (mtl_file_name[0] == 0)) {
    std::ifstream obj_file(obj_file_name);
    std::string line;
    while (std::getline(obj_file, line)) {
      if (line.substr(0, 6) == "mtllib") {
        std::istringstream iss(line.substr(7));
        std::string mtl_path;
        iss >> mtl_path;
        std::filesystem::path obj_path(obj_file_name);
        std::filesystem::path mtl_full_path = obj_path.parent_path() / mtl_path;
        materials_to_load = mtl_full_path.string().c_str();
        break;
      }
    }
  } else {
    materials_to_load = mtl_file_name;
  }

  if (tinyobj::LoadObj(&obj_attrib, &obj_shapes, &obj_materials, &warnings, &errors, obj_file_name, base_dir, materials_to_load.c_str()) == false) {
    log::error("Failed to load OBJ from file: `%s`\n%s", obj_file_name, errors.c_str());
    return SceneLoadFailed;
  }

  if (warnings.empty() == false) {
    log::warning("Loaded OBJ from file: `%s`\n%s", obj_file_name, warnings.c_str());
  }

  if (materials_to_load.empty() == false) {
    load_materials(data, ior_database, scheduler, materials_to_load.c_str(), base_dir, {});
  } else {
    std::vector<etx::MaterialDefinition> material_definitions;
    material_definitions.reserve(obj_materials.size());
    for (const auto& material : obj_materials) {
      material_definitions.emplace_back(convert_tinyobj_to_material_definition(material));
    }
    load_materials(data, ior_database, scheduler, nullptr, base_dir, material_definitions);
  }

  std::unordered_map<etx::VertexKey, uint32_t, etx::VertexKeyHash> vertex_map;
  size_t cache_hits = 0;

  uint64_t total_triangles = 0;
  for (const auto& shape : obj_shapes) {
    total_triangles += shape.mesh.num_face_vertices.size();
  }

  triangles.reserve(total_triangles);

  const uint64_t total_count = std::min(total_triangles * 3, obj_attrib.vertices.size() / 3);
  vertices.pos.reserve(total_count);
  vertices.nrm.reserve(total_count);
  vertices.tan.reserve(total_count);
  vertices.btn.reserve(total_count);
  vertices.tex.reserve(total_count);

  auto processing_start = std::chrono::high_resolution_clock::now();

  for (const auto& shape : obj_shapes) {
    uint64_t index_offset = 0;

    std::unordered_map<uint32_t, std::pair<uint32_t, uint32_t>> material_to_triangle_range;

    for (uint64_t face = 0, face_e = shape.mesh.num_face_vertices.size(); face < face_e; ++face) {
      uint64_t face_size = shape.mesh.num_face_vertices[face];
      ETX_ASSERT(face_size == 3);

      uint32_t triangle_index = static_cast<uint32_t>(triangles.size());
      auto& tri = triangles.emplace_back();

      std::string material_name;
      int material_id = shape.mesh.material_ids[face];
      if ((material_id >= 0) && (static_cast<size_t>(material_id) < shape.mesh.material_names.size())) {
        material_name = shape.mesh.material_names[material_id];
      }

      if (!material_name.empty()) {
        auto material_it = material_mapping.find(material_name);
        if (material_it != material_mapping.end()) {
          tri.material_index = material_it->second;
        } else {
          log::warning("Material '%s' referenced in OBJ file but not found in materials, using missing material", material_name.c_str());
          tri.material_index = data.defaults.missing_material;
        }
      } else {
        tri.material_index = data.defaults.missing_material;
      }

      for (uint64_t vertex_index = 0; vertex_index < face_size; ++vertex_index) {
        const auto& index = shape.mesh.indices[index_offset + vertex_index];

        float3 position = make_float3(obj_attrib.vertices.data() + (3 * index.vertex_index));
        bool has_normal = (index.normal_index >= 0);
        float3 normal = has_normal ? make_float3(obj_attrib.normals.data() + (3 * index.normal_index)) : float3{0.0f, 1.0f, 0.0f};
        bool has_uv = (index.texcoord_index >= 0);
        float2 uv = has_uv ? make_float2(obj_attrib.texcoords.data() + (2 * index.texcoord_index)) : float2{0.0f, 0.0f};

        VertexKey key = {position, normal, uv, has_normal, has_uv};

        auto it = vertex_map.find(key);
        if (it != vertex_map.end()) {
          tri.i[vertex_index] = it->second;
          cache_hits++;
        } else {
          uint32_t vertex_index_new = static_cast<uint32_t>(vertices.pos.size());
          tri.i[vertex_index] = vertex_index_new;
          vertex_map[key] = vertex_index_new;

          vertices.pos.emplace_back(position);
          vertices.nrm.emplace_back(normal);
          vertices.tex.emplace_back(uv);
        }
      }
      index_offset += face_size;

      if (validate_triangle(tri, vertices.pos) == false) {
        triangles.pop_back();
        continue;
      }

      auto& range = material_to_triangle_range[tri.material_index];
      if (range.second == 0) {
        range.first = triangle_index;
      }
      range.second++;  // count
    }

    uint32_t material_counter = 0;
    for (const auto& [material_index, triangle_range] : material_to_triangle_range) {
      if (triangle_range.second == 0)
        continue;

      std::string mesh_name = shape.name;
      if (material_to_triangle_range.size() > 1) {
        mesh_name += "_" + std::to_string(material_counter++);
      }

      float3 mesh_bbox_min = {kMaxFloat, kMaxFloat, kMaxFloat};
      float3 mesh_bbox_max = {-kMaxFloat, -kMaxFloat, -kMaxFloat};
      for (uint32_t i = 0; i < triangle_range.second; ++i) {
        const auto& tri = triangles[triangle_range.first + i];
        mesh_bbox_min = min(mesh_bbox_min, vertices.pos[tri.i[0]]);
        mesh_bbox_min = min(mesh_bbox_min, vertices.pos[tri.i[1]]);
        mesh_bbox_min = min(mesh_bbox_min, vertices.pos[tri.i[2]]);
        mesh_bbox_max = max(mesh_bbox_max, vertices.pos[tri.i[0]]);
        mesh_bbox_max = max(mesh_bbox_max, vertices.pos[tri.i[1]]);
        mesh_bbox_max = max(mesh_bbox_max, vertices.pos[tri.i[2]]);
      }

      data.add_mesh(mesh_name.c_str(), triangle_range.first, triangle_range.second, mesh_bbox_min, mesh_bbox_max);
    }
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
  log::info("OBJ loading total: %lld ms", total_duration.count());

  return SceneLoadSucceeded;
}

}  // namespace etx
