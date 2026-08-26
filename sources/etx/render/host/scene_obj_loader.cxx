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
  result.name = etx::normalize_material_name(material.name);

  for (const auto& param : material.unknown_parameter) {
    result.properties[param.first] = param.second;
  }

  // Special handling for normal map (add "image " prefix as expected by the material system)
  auto normal_it = result.properties.find("norm");
  if (normal_it != result.properties.end()) {
    result.properties["normalmap"] = std::string("image ") + normal_it->second;
  }

  return result;
}

struct ObjFileData {
  tinyobj::attrib_t attrib;
  std::vector<tinyobj::shape_t> shapes;
  std::vector<tinyobj::material_t> materials;
  std::string base_dir;
};

bool load_obj_file_data(const char* obj_file_name, const char* mtl_file_name, ObjFileData& result) {
  constexpr auto kDataBufferSize = 2048llu;
  static char base_dir[kDataBufferSize] = {};
  get_base_directory(obj_file_name, base_dir, sizeof(base_dir));
  result.base_dir = base_dir;

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

  std::string warnings;
  std::string errors;

  if (tinyobj::LoadObj(&result.attrib, &result.shapes, &result.materials, &warnings, &errors, obj_file_name, base_dir, materials_to_load.c_str()) == false) {
    log::error("Failed to load OBJ from file: `%s`\n%s", obj_file_name, errors.c_str());
    return false;
  }

  if (warnings.empty() == false) {
    log::warning("Loaded OBJ from file: `%s`\n%s", obj_file_name, warnings.c_str());
  }

  return true;
}

void setup_materials_for_obj(const ObjFileData& obj_data, const char* mtl_file_name, const char* obj_file_name, SceneData& data, const IORDatabase& ior_database,
  TaskScheduler& scheduler) {
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

  if (materials_to_load.empty() == false) {
    load_materials(data, ior_database, scheduler, materials_to_load.c_str(), obj_data.base_dir.c_str(), {});
  } else {
    std::vector<etx::MaterialDefinition> material_definitions;
    material_definitions.reserve(obj_data.materials.size());
    for (const auto& material : obj_data.materials) {
      material_definitions.emplace_back(convert_tinyobj_to_material_definition(material));
    }
    load_materials(data, ior_database, scheduler, nullptr, obj_data.base_dir.c_str(), material_definitions);
  }
}

void process_obj_shape(const tinyobj::shape_t& shape, const tinyobj::attrib_t& obj_attrib, const std::vector<tinyobj::material_t>& obj_materials, SceneData& data,
  size_t& total_vertices_processed, size_t& cache_hits) {
  auto& triangles = data.triangles;
  auto& vertices = data.vertices;
  auto& material_mapping = data.material_mapping;

  using VertexMap = std::unordered_map<etx::VertexKey, uint32_t, etx::VertexKeyHash>;
  VertexMap explicit_normal_vertices;
  std::unordered_map<uint32_t, VertexMap> smoothing_group_vertices;

  uint64_t index_offset = 0;

  struct FaceData {
    int material_id;
    uint32_t smoothing_group_id;
    tinyobj::index_t indices[3];
  };

  std::unordered_map<int, std::vector<FaceData>> material_to_faces;

  for (uint64_t face = 0, face_e = shape.mesh.num_face_vertices.size(); face < face_e; ++face) {
    uint64_t face_size = shape.mesh.num_face_vertices[face];
    ETX_ASSERT(face_size == 3);

    int material_id = -1;
    if (face < shape.face_material_ids.size()) {
      material_id = shape.face_material_ids[face];
    }

    FaceData face_data = {
      .material_id = material_id,
      .smoothing_group_id = (face < shape.mesh.smoothing_group_ids.size()) ? shape.mesh.smoothing_group_ids[face] : 0u,
    };
    for (uint64_t vertex_index = 0; vertex_index < face_size; ++vertex_index) {
      face_data.indices[vertex_index] = shape.mesh.indices[index_offset + vertex_index];
    }

    material_to_faces[material_id].push_back(face_data);
    index_offset += face_size;
  }

  uint32_t material_counter = 0;
  for (const auto& [material_id, faces] : material_to_faces) {
    if (faces.empty())
      continue;

    uint32_t triangle_start = static_cast<uint32_t>(triangles.size());
    uint32_t valid_triangle_count = 0;

    float3 mesh_bbox_min = {kMaxFloat, kMaxFloat, kMaxFloat};
    float3 mesh_bbox_max = {-kMaxFloat, -kMaxFloat, -kMaxFloat};

    uint32_t material_index = data.defaults.missing_material;
    if (faces.empty() == false) {
      const auto& first_face = faces[0];
      if ((first_face.material_id >= 0) && (static_cast<size_t>(first_face.material_id) < obj_materials.size())) {
        std::string material_name = obj_materials[first_face.material_id].name;
        auto material_it = material_mapping.find(material_name);
        if (material_it != material_mapping.end()) {
          material_index = material_it->second;
        } else {
          log::warning("Material '%s' referenced in OBJ file but not found in materials, using missing material", material_name.c_str());
        }
      }
    }

    for (const auto& face_data : faces) {
      Triangle tri = {};
      tri.material_index = material_index;

      bool valid_triangle = true;
      float3 face_bbox_min = {kMaxFloat, kMaxFloat, kMaxFloat};
      float3 face_bbox_max = {-kMaxFloat, -kMaxFloat, -kMaxFloat};

      for (uint64_t vertex_index = 0; vertex_index < 3; ++vertex_index) {
        ++total_vertices_processed;
        const auto& index = face_data.indices[vertex_index];

        float3 position = {static_cast<float>(obj_attrib.vertex_x[index.vertex_index]), static_cast<float>(obj_attrib.vertex_y[index.vertex_index]),
          static_cast<float>(obj_attrib.vertex_z[index.vertex_index])};

        bool has_normal = (index.normal_index >= 0) && (static_cast<size_t>(index.normal_index) < obj_attrib.normal_x.size());
        float3 normal = has_normal ? float3{static_cast<float>(obj_attrib.normal_x[index.normal_index]), static_cast<float>(obj_attrib.normal_y[index.normal_index]),
                                       static_cast<float>(obj_attrib.normal_z[index.normal_index])}
                                   : float3{};

        bool has_uv = (index.texcoord_index >= 0) && (static_cast<size_t>(index.texcoord_index) < obj_attrib.texcoord_u.size());
        float2 uv =
          has_uv ? float2{static_cast<float>(obj_attrib.texcoord_u[index.texcoord_index]), static_cast<float>(obj_attrib.texcoord_v[index.texcoord_index])} : float2{0.0f, 0.0f};

        face_bbox_min = min(face_bbox_min, position);
        face_bbox_max = max(face_bbox_max, position);

        VertexKey key = {position, normal, uv, has_normal, has_uv};

        VertexMap* vertex_map = nullptr;
        if (has_normal) {
          vertex_map = &explicit_normal_vertices;
        } else if (face_data.smoothing_group_id != 0u) {
          vertex_map = &smoothing_group_vertices[face_data.smoothing_group_id];
        }

        const uint32_t vertex_index_new = static_cast<uint32_t>(vertices.pos.size());
        if (vertex_map != nullptr) {
          auto [it, inserted] = vertex_map->emplace(key, vertex_index_new);
          if (inserted == false) {
            tri.i[vertex_index] = it->second;
            cache_hits++;
            continue;
          }
        }

        tri.i[vertex_index] = vertex_index_new;
        vertices.pos.emplace_back(position);
        vertices.nrm.emplace_back(normal);
        vertices.tex.emplace_back(uv);
      }

      if (validate_triangle(tri, vertices.pos) == false) {
        continue;
      }

      triangles.push_back(tri);
      valid_triangle_count++;

      mesh_bbox_min = min(mesh_bbox_min, face_bbox_min);
      mesh_bbox_max = max(mesh_bbox_max, face_bbox_max);
    }

    if (valid_triangle_count > 0) {
      std::string mesh_name = shape.name;
      if (material_to_faces.size() > 1) {
        mesh_name += "_" + std::to_string(material_counter++);
      }

      data.add_mesh(mesh_name.c_str(), triangle_start, valid_triangle_count, mesh_bbox_min, mesh_bbox_max);
    }
  }
}

void process_obj_shapes(const ObjFileData& obj_data, SceneData& data) {
  size_t total_vertices_processed = 0;
  size_t cache_hits = 0;

  uint64_t total_triangles = 0;
  for (const auto& shape : obj_data.shapes) {
    total_triangles += shape.mesh.num_face_vertices.size();
  }

  auto& triangles = data.triangles;
  auto& vertices = data.vertices;

  triangles.reserve(total_triangles);

  const uint64_t total_count = min(static_cast<uint64_t>(total_triangles) * 3, static_cast<uint64_t>(obj_data.attrib.vertex_x.size()));
  vertices.pos.reserve(total_count);
  vertices.nrm.reserve(total_count);
  vertices.tan.reserve(total_count);
  vertices.btn.reserve(total_count);
  vertices.tex.reserve(total_count);

  for (const auto& shape : obj_data.shapes) {
    process_obj_shape(shape, obj_data.attrib, obj_data.materials, data, total_vertices_processed, cache_hits);
  }

  size_t unique_vertices = total_vertices_processed - cache_hits;
  log::info("Vertex deduplication: %llu total processed, %llu unique (%.1f%% reduction)", total_vertices_processed, unique_vertices,
    total_vertices_processed > 0 ? (1.0f - float(unique_vertices) / total_vertices_processed) * 100.0f : 0.0f);
}

}  // namespace

struct SceneObjLoaderImpl {};

uint32_t load_from_obj_file(const char* obj_file_name, const char* mtl_file_name, SceneData& data, const IORDatabase& ior_database, TaskScheduler& scheduler) {
  auto start_time = std::chrono::high_resolution_clock::now();

  ObjFileData obj_data;
  if (!load_obj_file_data(obj_file_name, mtl_file_name, obj_data)) {
    return SceneLoadFailed;
  }

  setup_materials_for_obj(obj_data, mtl_file_name, obj_file_name, data, ior_database, scheduler);

  auto processing_start = std::chrono::high_resolution_clock::now();
  process_obj_shapes(obj_data, data);

  auto end_time = std::chrono::high_resolution_clock::now();
  auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
  log::info("OBJ loading total: %lld ms", total_duration.count());

  return SceneLoadSucceeded;
}

}  // namespace etx
