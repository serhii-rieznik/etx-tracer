#include <array>

#include <etx/render/host/emitter_packing.hxx>
#include <etx/render/host/scene_data.hxx>
#include <etx/render/host/scene_representation.hxx>
#include <etx/render/host/scene_serialization.hxx>
#include <etx/render/host/tasks.hxx>
#include <etx/render/shared/ior_database.hxx>

#include <filesystem>
#include <fstream>
#include <cstdio>

namespace {

constexpr float kTestEpsilon = 1.0e-4f;

bool check_condition(bool condition, const char* message) {
  if (condition == false) {
    std::printf("FAILED: %s\n", message);
    return false;
  }
  return true;
}

bool nearly_equal(float a, float b) {
  return fabsf(a - b) <= kTestEpsilon;
}

bool nearly_zero(float value) {
  return fabsf(value) <= kTestEpsilon;
}

etx::MaterialDefinition make_definition(const char* name) {
  etx::MaterialDefinition result = {};
  result.name = etx::normalize_material_name(name);
  return result;
}

void add_property(etx::MaterialDefinition& definition, const char* name, const char* value) {
  definition.properties[name] = value;
}

struct TestContext {
  etx::TaskScheduler scheduler = {};
  etx::SceneData data;
  etx::IORDatabase ior_database = {};

  TestContext()
    : data(scheduler) {
    data.images.init(16u);
    data.mediums.init(16u);
    data.defaults.missing_material = data.add_material("__missing");
  }
};

bool parse_definitions(TestContext& context, const std::vector<etx::MaterialDefinition>& definitions) {
  etx::SceneSerialization serialization;
  serialization.parse_material_definitions("", definitions, context.data, context.ior_database, context.scheduler);
  return true;
}

etx::MaterialDefinition make_basic_material(const char* name) {
  etx::MaterialDefinition material = make_definition(name);
  add_property(material, "Kd", "0.8 0.2 0.1");
  return material;
}

etx::MaterialDefinition make_sphere(const char* entry_name, const char* id, const char* material_name) {
  etx::MaterialDefinition sphere = make_definition(entry_name);
  if ((sphere.name == "et::geometry") || (sphere.name == "etx::geometry")) {
    add_property(sphere, "class", "sphere");
  }
  add_property(sphere, "id", id);
  add_property(sphere, "center", "1 2 3");
  add_property(sphere, "dimensions", "4 4 4");
  add_property(sphere, "subdivisions", "1");
  add_property(sphere, "material", material_name);
  return sphere;
}

etx::MaterialDefinition make_plane(const char* entry_name, const char* id, const char* material_name) {
  etx::MaterialDefinition plane = make_definition(entry_name);
  if ((plane.name == "et::geometry") || (plane.name == "etx::geometry")) {
    add_property(plane, "class", "plane");
  }
  add_property(plane, "id", id);
  add_property(plane, "center", "1 2 3");
  add_property(plane, "dimensions", "4 0 6");
  add_property(plane, "material", material_name);
  return plane;
}

bool test_sphere_generation() {
  TestContext context;
  std::vector<etx::MaterialDefinition> definitions;
  definitions.emplace_back(make_sphere("et::geometry", "ball", "red"));
  definitions.emplace_back(make_basic_material("red"));

  parse_definitions(context, definitions);

  if (check_condition(context.data.meshes.size() == 1u, "one mesh generated") == false) {
    return false;
  }
  if (check_condition(context.data.triangles.size() == 80u, "subdivision 1 sphere has 80 triangles") == false) {
    return false;
  }
  if (check_condition(context.data.vertices.pos.empty() == false, "vertices generated") == false) {
    return false;
  }
  if (check_condition(context.data.mesh_mapping.count("ball") == 1u, "mesh name registered") == false) {
    return false;
  }

  const uint32_t material_index = context.data.material_mapping.at("red");
  for (const Triangle& triangle : context.data.triangles) {
    if (check_condition(triangle.material_index == material_index, "triangle material resolved by later material") == false) {
      return false;
    }
  }

  for (uint32_t i = 0u, e = static_cast<uint32_t>(context.data.vertices.nrm.size()); i < e; ++i) {
    const float3& nrm = context.data.vertices.nrm[i];
    const float3& tan = context.data.vertices.tan[i];
    const float3& btn = context.data.vertices.btn[i];
    if (check_condition(nearly_equal(length(tan), 1.0f), "sphere tangent has unit length") == false) {
      return false;
    }
    if (check_condition(nearly_equal(length(btn), 1.0f), "sphere bitangent has unit length") == false) {
      return false;
    }
    if (check_condition(nearly_zero(dot(nrm, tan)), "sphere tangent is orthogonal to normal") == false) {
      return false;
    }
    if (check_condition(nearly_zero(dot(nrm, btn)), "sphere bitangent is orthogonal to normal") == false) {
      return false;
    }
    if (check_condition(nearly_zero(dot(tan, btn)), "sphere tangent and bitangent are orthogonal") == false) {
      return false;
    }
  }

  const etx::Mesh& mesh = context.data.meshes[0];
  if (check_condition(nearly_equal(mesh.bbox_min.x, -1.0f) && nearly_equal(mesh.bbox_min.y, 0.0f) && nearly_equal(mesh.bbox_min.z, 1.0f), "bbox minimum matches analytic sphere") ==
      false) {
    return false;
  }
  if (check_condition(nearly_equal(mesh.bbox_max.x, 3.0f) && nearly_equal(mesh.bbox_max.y, 4.0f) && nearly_equal(mesh.bbox_max.z, 5.0f), "bbox maximum matches analytic sphere") ==
      false) {
    return false;
  }

  return true;
}

bool test_plane_generation() {
  TestContext context;
  std::vector<etx::MaterialDefinition> definitions;
  definitions.emplace_back(make_plane("et::geometry", "floor", "matte"));
  definitions.emplace_back(make_basic_material("matte"));

  parse_definitions(context, definitions);

  if (check_condition(context.data.meshes.size() == 1u, "one plane mesh generated") == false) {
    return false;
  }
  if (check_condition(context.data.triangles.size() == 2u, "plane has two triangles") == false) {
    return false;
  }
  if (check_condition(context.data.vertices.pos.size() == 4u, "plane has four vertices") == false) {
    return false;
  }
  if (check_condition(context.data.mesh_mapping.count("floor") == 1u, "plane mesh name registered") == false) {
    return false;
  }

  const uint32_t material_index = context.data.material_mapping.at("matte");
  for (const Triangle& triangle : context.data.triangles) {
    if (check_condition(triangle.material_index == material_index, "plane material resolved by later material") == false) {
      return false;
    }
  }

  const etx::Mesh& mesh = context.data.meshes[0];
  if (check_condition(nearly_equal(mesh.bbox_min.x, -1.0f) && nearly_equal(mesh.bbox_min.y, 2.0f) && nearly_equal(mesh.bbox_min.z, 0.0f), "plane bbox minimum matches size") ==
      false) {
    return false;
  }
  if (check_condition(nearly_equal(mesh.bbox_max.x, 3.0f) && nearly_equal(mesh.bbox_max.y, 2.0f) && nearly_equal(mesh.bbox_max.z, 6.0f), "plane bbox maximum matches size") ==
      false) {
    return false;
  }

  const float3 normal = cross(context.data.vertices.pos[context.data.triangles[0].i[1]] - context.data.vertices.pos[context.data.triangles[0].i[0]],
    context.data.vertices.pos[context.data.triangles[0].i[2]] - context.data.vertices.pos[context.data.triangles[0].i[0]]);
  if (check_condition(normal.y > 0.0f, "plane triangles face upward") == false) {
    return false;
  }

  return true;
}

bool test_etx_alias_and_material_index() {
  TestContext context;
  const uint32_t blue_index = context.data.add_material("blue");

  etx::MaterialDefinition sphere = make_definition("etx::sphere");
  add_property(sphere, "id", "alias");
  add_property(sphere, "dimensions", "2 2 2");
  add_property(sphere, "subdivisions", "0");
  add_property(sphere, "material-index", std::to_string(blue_index).c_str());

  std::vector<etx::MaterialDefinition> definitions;
  definitions.emplace_back(sphere);

  parse_definitions(context, definitions);

  if (check_condition(context.data.meshes.size() == 1u, "etx::sphere alias generated mesh") == false) {
    return false;
  }
  if (check_condition(context.data.triangles.size() == 20u, "subdivision 0 sphere has 20 triangles") == false) {
    return false;
  }
  for (const Triangle& triangle : context.data.triangles) {
    if (check_condition(triangle.material_index == blue_index, "material-index applied") == false) {
      return false;
    }
  }

  return true;
}

bool test_plane_alias() {
  TestContext context;
  std::vector<etx::MaterialDefinition> definitions;
  definitions.emplace_back(make_basic_material("matte"));
  definitions.emplace_back(make_plane("etx::plane", "alias-plane", "matte"));

  parse_definitions(context, definitions);

  if (check_condition(context.data.meshes.size() == 1u, "etx::plane alias generated mesh") == false) {
    return false;
  }
  if (check_condition(context.data.triangles.size() == 2u, "etx::plane alias has two triangles") == false) {
    return false;
  }
  if (check_condition(context.data.mesh_mapping.count("alias-plane") == 1u, "etx::plane alias mesh name registered") == false) {
    return false;
  }

  return true;
}

bool test_duplicate_ids() {
  TestContext context;
  std::vector<etx::MaterialDefinition> definitions;
  definitions.emplace_back(make_basic_material("red"));
  definitions.emplace_back(make_sphere("et::geometry", "duplicate", "red"));
  definitions.emplace_back(make_sphere("et::geometry", "duplicate", "red"));

  parse_definitions(context, definitions);

  if (check_condition(context.data.meshes.size() == 2u, "duplicate ids generate two meshes") == false) {
    return false;
  }
  if (check_condition(context.data.mesh_mapping.count("duplicate") == 1u, "first duplicate keeps requested name") == false) {
    return false;
  }
  if (check_condition(context.data.mesh_mapping.count("duplicate-1") == 1u, "second duplicate gets suffix") == false) {
    return false;
  }

  return true;
}

bool test_invalid_dimensions_skips() {
  TestContext context;
  std::vector<etx::MaterialDefinition> definitions;
  definitions.emplace_back(make_basic_material("red"));

  etx::MaterialDefinition sphere = make_sphere("et::geometry", "bad", "red");
  sphere.properties["dimensions"] = "0 0 0";
  definitions.emplace_back(sphere);

  parse_definitions(context, definitions);

  return check_condition(context.data.triangles.empty(), "invalid dimensions skip geometry");
}

bool test_invalid_material_uses_missing() {
  TestContext context;
  std::vector<etx::MaterialDefinition> definitions;
  definitions.emplace_back(make_sphere("et::geometry", "fallback", "not_defined"));

  parse_definitions(context, definitions);

  if (check_condition(context.data.triangles.empty() == false, "unknown material still generates geometry") == false) {
    return false;
  }
  for (const Triangle& triangle : context.data.triangles) {
    if (check_condition(triangle.material_index == context.data.defaults.missing_material, "unknown material uses missing material") == false) {
      return false;
    }
  }

  return true;
}

bool test_binary_load_skips_baked_procedural_geometry() {
  TestContext saved_context;
  const uint32_t material_index = saved_context.data.add_material("matte");

  saved_context.data.vertices.pos.emplace_back(float3{0.0f, 0.0f, 0.0f});
  saved_context.data.vertices.pos.emplace_back(float3{1.0f, 0.0f, 0.0f});
  saved_context.data.vertices.pos.emplace_back(float3{0.0f, 0.0f, 1.0f});

  Triangle tri = {};
  tri.i[0] = 0u;
  tri.i[1] = 2u;
  tri.i[2] = 1u;
  tri.material_index = material_index;
  saved_context.data.triangles.emplace_back(tri);
  saved_context.data.add_mesh("already_baked", 0u, 1u, float3{0.0f, 0.0f, 0.0f}, float3{1.0f, 0.0f, 1.0f});

  const std::filesystem::path geometry_path = std::filesystem::path("build") / "procedural_geometry_binary_test.etx";
  const std::filesystem::path materials_path = std::filesystem::path("build") / "procedural_geometry_binary_test.etx.materials";
  std::filesystem::create_directories(geometry_path.parent_path());

  etx::SceneSerialization save_serialization;
  if (check_condition(save_serialization.save_to_file(saved_context.data, geometry_path), "binary geometry saved") == false) {
    return false;
  }

  std::ofstream materials_file(materials_path, std::ios::out | std::ios::trunc);
  if (check_condition(materials_file.is_open(), "binary sidecar materials opened") == false) {
    return false;
  }

  materials_file << "newmtl matte\n";
  materials_file << "Kd 0.8 0.8 0.8\n\n";
  materials_file << "newmtl et::geometry\n";
  materials_file << "class plane\n";
  materials_file << "id already_baked\n";
  materials_file << "center 0 0 0\n";
  materials_file << "dimensions 4 0 4\n";
  materials_file << "material matte\n";
  materials_file.close();

  TestContext loaded_context;
  etx::SceneSerialization load_serialization;
  const std::string materials_path_string = materials_path.string();
  if (check_condition(load_serialization.load_from_file(geometry_path, loaded_context.data, materials_path_string.c_str(), loaded_context.ior_database, loaded_context.scheduler),
        "binary geometry loaded with procedural sidecar") == false) {
    return false;
  }

  if (check_condition(loaded_context.data.meshes.size() == 1u, "binary load skips baked procedural mesh") == false) {
    return false;
  }
  if (check_condition(loaded_context.data.triangles.size() == 1u, "binary load skips baked procedural triangles") == false) {
    return false;
  }
  if (check_condition(loaded_context.data.vertices.pos.size() == 3u, "binary load skips baked procedural vertices") == false) {
    return false;
  }

  std::filesystem::remove(geometry_path);
  std::filesystem::remove(materials_path);
  return true;
}

bool test_native_json_loads_text_only_procedural_geometry() {
  const std::filesystem::path json_path = std::filesystem::path("build") / "procedural_geometry_text_only.etx.json";
  const std::filesystem::path materials_path = std::filesystem::path("build") / "procedural_geometry_text_only.etx.materials";
  std::filesystem::create_directories(json_path.parent_path());

  std::ofstream json_file(json_path, std::ios::out | std::ios::trunc);
  if (check_condition(json_file.is_open(), "text-only json opened") == false) {
    return false;
  }
  json_file << "{\n";
  json_file << "  \"materials\": \"procedural_geometry_text_only.etx.materials\",\n";
  json_file << "  \"samples\": 1\n";
  json_file << "}\n";
  json_file.close();

  std::ofstream materials_file(materials_path, std::ios::out | std::ios::trunc);
  if (check_condition(materials_file.is_open(), "text-only materials opened") == false) {
    return false;
  }
  materials_file << "newmtl matte\n";
  materials_file << "Kd 0.8 0.8 0.8\n\n";
  materials_file << "newmtl et::geometry\n";
  materials_file << "class sphere\n";
  materials_file << "id text_sphere\n";
  materials_file << "center 0 0 0\n";
  materials_file << "dimensions 2 2 2\n";
  materials_file << "subdivisions 0\n";
  materials_file << "material matte\n";
  materials_file.close();

  TestContext context;
  etx::SceneRepresentation scene(context.scheduler, context.ior_database);
  const std::string json_path_string = json_path.string();
  if (check_condition(scene.load_from_file(json_path_string.c_str(), etx::SceneRepresentation::LoadGeometry, nullptr), "text-only procedural scene loaded") == false) {
    return false;
  }

  if (check_condition(scene.data().triangles.size() == 20u, "text-only procedural scene generated triangles") == false) {
    return false;
  }
  if (check_condition(scene.data().meshes.size() == 1u, "text-only procedural scene generated mesh") == false) {
    return false;
  }
  if (check_condition(scene.mesh_mapping().count("text_sphere") == 1u, "text-only procedural mesh named") == false) {
    return false;
  }

  std::filesystem::remove(json_path);
  std::filesystem::remove(materials_path);
  return true;
}

bool test_empty_scene_environment_emitter_packs() {
  TestContext context;

  auto& emitter = context.data.emitter_profiles.emplace_back(etx::EmitterProfile::Class::Environment);
  emitter.emission.spectrum_index = context.data.add_spectrum(etx::SpectralDistribution::rgb_luminance({1.0f, 1.0f, 1.0f}));

  const float4 pixel = {1.0f, 1.0f, 1.0f, 1.0f};
  emitter.emission.image_index = context.data.add_image(&pixel, {1u, 1u}, etx::Image::BuildSamplingTable | etx::Image::RepeatU, {}, {1.0f, 1.0f});

  context.data.images.load_images(context.scheduler);
  const etx::Image& image = context.data.images.get(emitter.emission.image_index);
  if (check_condition(image.x_distributions.count == 1u, "empty scene environment image has x sampling table") == false) {
    return false;
  }
  if (check_condition(image.y_distribution.values.count == 1u, "empty scene environment image has y sampling table") == false) {
    return false;
  }

  const etx::PackedEmitterData packed_emitters = etx::build_packed_emitters(context.data);

  if (check_condition(packed_emitters.emitter_profiles.size() == 1u, "empty scene keeps environment profile") == false) {
    return false;
  }
  if (check_condition(packed_emitters.emitter_instances.size() == 1u, "empty scene creates environment instance") == false) {
    return false;
  }
  if (check_condition(packed_emitters.active_emitter_indices.size() == 1u, "empty scene environment is active") == false) {
    return false;
  }
  if (check_condition(packed_emitters.environment_emitters.count == 1u, "empty scene environment is tracked as distant emitter") == false) {
    return false;
  }
  if (check_condition(std::isfinite(packed_emitters.emitter_instances[0].additional_weight), "empty scene environment has finite weight") == false) {
    return false;
  }
  if (check_condition(packed_emitters.emitter_instances[0].additional_weight > 0.0f, "empty scene environment has positive weight") == false) {
    return false;
  }

  return true;
}

bool test_raw_obj_load_adds_default_camera_and_lighting() {
  const std::filesystem::path obj_path = std::filesystem::path("build") / "raw_model_defaults.obj";
  std::filesystem::create_directories(obj_path.parent_path());

  std::ofstream obj_file(obj_path, std::ios::out | std::ios::trunc);
  if (check_condition(obj_file.is_open(), "raw obj opened") == false) {
    return false;
  }
  obj_file << "o triangle\n";
  obj_file << "v -1 0 -1\n";
  obj_file << "v 1 0 -1\n";
  obj_file << "v 0 0 1\n";
  obj_file << "f 1 2 3\n";
  obj_file.close();

  TestContext context;
  etx::SceneRepresentation scene(context.scheduler, context.ior_database);
  const std::string obj_path_string = obj_path.string();
  if (check_condition(scene.load_from_file(obj_path_string.c_str(), etx::SceneRepresentation::LoadEverything, nullptr), "raw obj scene loaded") == false) {
    return false;
  }

  if (check_condition(scene.data().cameras.size() == 1u, "raw obj creates one default camera") == false) {
    return false;
  }
  if (check_condition(scene.data().cameras[0].active, "raw obj default camera is active") == false) {
    return false;
  }
  if (check_condition(scene.camera().film_size.x == 1280u, "raw obj default camera width") == false) {
    return false;
  }
  if (check_condition(scene.camera().film_size.y == 720u, "raw obj default camera height") == false) {
    return false;
  }
  if (check_condition(length(scene.camera().direction) > 0.9f, "raw obj default camera direction is valid") == false) {
    return false;
  }

  uint32_t atmosphere_count = 0u;
  uint32_t directional_count = 0u;
  uint32_t atmosphere_index = kInvalidIndex;
  uint32_t sun_reference = kInvalidIndex;
  for (uint32_t i = 0u, e = static_cast<uint32_t>(scene.data().emitter_profiles.size()); i < e; ++i) {
    const etx::EmitterProfile& profile = scene.data().emitter_profiles[i];
    if ((profile.cls == etx::EmitterProfile::Class::Environment) && ((profile.meta & etx::EmitterProfile::Meta::Atmosphere) != 0u)) {
      atmosphere_index = i;
      ++atmosphere_count;
    } else if (profile.cls == etx::EmitterProfile::Class::Directional) {
      sun_reference = profile.reference_emitter_index;
      ++directional_count;
    }
  }

  if (check_condition(atmosphere_count == 1u, "raw obj creates atmosphere emitter") == false) {
    return false;
  }
  if (check_condition(directional_count == 1u, "raw obj creates directional sun emitter") == false) {
    return false;
  }
  if (check_condition(sun_reference == atmosphere_index, "raw obj sun references atmosphere") == false) {
    return false;
  }

  std::filesystem::remove(obj_path);
  return true;
}

bool test_directional_use_as_sun_parse() {
  TestContext context;

  etx::MaterialDefinition atmosphere = make_definition("et::atmosphere");
  add_property(atmosphere, "quality", "0.125");

  etx::MaterialDefinition directional_definition = make_definition("et::dir");
  add_property(directional_definition, "direction", "0 1 0");
  add_property(directional_definition, "use_as_sun", "1");

  std::vector<etx::MaterialDefinition> definitions = {atmosphere, directional_definition};
  parse_definitions(context, definitions);

  uint32_t atmosphere_index = kInvalidIndex;
  uint32_t directional_index = kInvalidIndex;
  for (uint32_t i = 0u, e = static_cast<uint32_t>(context.data.emitter_profiles.size()); i < e; ++i) {
    const etx::EmitterProfile& profile = context.data.emitter_profiles[i];
    if ((profile.cls == etx::EmitterProfile::Class::Environment) && ((profile.meta & etx::EmitterProfile::Meta::Atmosphere) != 0u)) {
      atmosphere_index = i;
    } else if (profile.cls == etx::EmitterProfile::Class::Directional) {
      directional_index = i;
    }
  }

  if (check_condition(atmosphere_index != kInvalidIndex, "use_as_sun scene has atmosphere") == false) {
    return false;
  }
  if (check_condition(directional_index != kInvalidIndex, "use_as_sun scene has directional emitter") == false) {
    return false;
  }

  const etx::EmitterProfile& directional = context.data.emitter_profiles[directional_index];
  return check_condition(directional.reference_emitter_index == atmosphere_index, "use_as_sun directional references atmosphere");
}

bool test_gltf_double_sided_material() {
  const std::filesystem::path gltf_path = std::filesystem::path("build") / "double_sided_material.gltf";
  const std::filesystem::path buffer_path = std::filesystem::path("build") / "double_sided_material.bin";
  std::filesystem::create_directories(gltf_path.parent_path());

  const std::array<float, 9u> positions = {
    0.0f,
    0.0f,
    0.0f,
    1.0f,
    0.0f,
    0.0f,
    0.0f,
    1.0f,
    0.0f,
  };
  const std::array<uint16_t, 3u> indices = {0u, 1u, 2u};
  std::ofstream buffer_file(buffer_path, std::ios::out | std::ios::binary | std::ios::trunc);
  if (check_condition(buffer_file.is_open(), "glTF material test buffer opened") == false) {
    return false;
  }
  buffer_file.write(reinterpret_cast<const char*>(positions.data()), static_cast<std::streamsize>(sizeof(positions)));
  buffer_file.write(reinterpret_cast<const char*>(indices.data()), static_cast<std::streamsize>(sizeof(indices)));
  buffer_file.close();

  std::ofstream gltf_file(gltf_path, std::ios::out | std::ios::trunc);
  if (check_condition(gltf_file.is_open(), "glTF material test scene opened") == false) {
    std::filesystem::remove(buffer_path);
    return false;
  }
  gltf_file << R"({
  "asset": {"version": "2.0"},
  "buffers": [{"uri": "double_sided_material.bin", "byteLength": 42}],
  "bufferViews": [
    {"buffer": 0, "byteOffset": 0, "byteLength": 36, "target": 34962},
    {"buffer": 0, "byteOffset": 36, "byteLength": 6, "target": 34963}
  ],
  "accessors": [
    {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3", "min": [0, 0, 0], "max": [1, 1, 0]},
    {"bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR"}
  ],
  "materials": [{"name": "two-sided", "doubleSided": true}],
  "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1, "material": 0}]}],
  "nodes": [{"mesh": 0}],
  "scenes": [{"nodes": [0]}],
  "scene": 0
})";
  gltf_file.close();

  TestContext context;
  etx::SceneRepresentation scene(context.scheduler, context.ior_database);
  const std::string gltf_path_string = gltf_path.string();
  bool result = check_condition(scene.load_from_file(gltf_path_string.c_str(), etx::SceneRepresentation::LoadEverything, nullptr), "glTF material test scene loaded");
  result &= check_condition(scene.data().triangles.size() == 1u, "glTF material test triangle loaded");
  result &= check_condition(scene.data().gltf_material_mapping.count(0) == 1u, "glTF material index mapped");
  if (scene.data().gltf_material_mapping.count(0) == 1u) {
    const uint32_t material_index = scene.data().gltf_material_mapping.at(0);
    result &= check_condition(scene.data().materials[material_index].two_sided == 1u, "glTF doubleSided maps to two_sided material flag");
  }

  std::filesystem::remove(gltf_path);
  std::filesystem::remove(buffer_path);
  return result;
}

bool test_native_scene_hierarchy_round_trip() {
  const std::filesystem::path obj_path = std::filesystem::path("build") / "hierarchy_round_trip_source.obj";
  const std::filesystem::path save_path = std::filesystem::path("build") / "hierarchy_round_trip.etx.json";
  std::filesystem::create_directories(obj_path.parent_path());
  std::ofstream obj_file(obj_path, std::ios::out | std::ios::trunc);
  obj_file << "o triangle\n";
  obj_file << "v 0 0 0\n";
  obj_file << "v 1 0 0\n";
  obj_file << "v 0 1 0\n";
  obj_file << "f 1 2 3\n";
  obj_file.close();

  TestContext context;
  etx::SceneRepresentation source(context.scheduler, context.ior_database);
  const std::string obj_path_string = obj_path.string();
  if (check_condition(source.load_from_file(obj_path_string.c_str(), etx::SceneRepresentation::LoadEverything, nullptr), "hierarchy source loaded") == false) {
    return false;
  }

  etx::SceneHierarchy& hierarchy = source.data().hierarchy;
  const etx::AffineTransform parent_transform = {
    .rows =
      {
        float4{2.0f, 0.0f, 0.0f, 3.0f},
        float4{0.0f, 3.0f, 0.0f, 4.0f},
        float4{0.0f, 0.0f, -1.0f, 5.0f},
      },
  };
  const uint32_t parent_index = hierarchy.add_node("round-trip-parent", kInvalidIndex, parent_transform);
  if (check_condition((hierarchy.nodes.empty() == false) && hierarchy.set_parent(0u, parent_index), "hierarchy parent assigned") == false) {
    return false;
  }
  if (check_condition(source.data().cameras.empty() == false, "default camera is available for hierarchy round trip") == false) {
    return false;
  }
  const uint32_t first_camera_node = hierarchy.add_node("first-camera", parent_index, {});
  hierarchy.add_attachment(first_camera_node, {etx::SceneAttachment::Type::Camera, 0u, 0u, 0u});
  etx::SceneData::CameraInfo second_camera = source.data().cameras[0];
  second_camera.id = "second-camera";
  second_camera.active = false;
  source.data().cameras.emplace_back(second_camera);
  const uint32_t second_camera_node = hierarchy.add_node("second-camera", parent_index, {});
  hierarchy.add_attachment(second_camera_node, {etx::SceneAttachment::Type::Camera, 1u, 0u, 0u});
  if (check_condition(source.data().resolve_hierarchy(), "source hierarchy resolved") == false) {
    return false;
  }

  const std::string saved_file = source.save_to_file(save_path.string().c_str());
  if (check_condition(saved_file.empty() == false, "native hierarchy scene saved") == false) {
    return false;
  }

  etx::SceneRepresentation loaded(context.scheduler, context.ior_database);
  if (check_condition(loaded.load_from_file(saved_file.c_str(), etx::SceneRepresentation::LoadEverything, nullptr), "native hierarchy scene loaded") == false) {
    return false;
  }
  const etx::SceneHierarchy& loaded_hierarchy = loaded.data().hierarchy;
  if (check_condition(loaded_hierarchy.nodes.size() == hierarchy.nodes.size(), "hierarchy node count round trips") == false) {
    return false;
  }
  if (check_condition(loaded_hierarchy.nodes[0].parent_index == parent_index, "hierarchy parent round trips") == false) {
    return false;
  }
  if (check_condition(loaded.data().cameras.size() == 2u, "all hierarchy-attached cameras round trip") == false) {
    return false;
  }
  if (check_condition(nearly_equal(loaded_hierarchy.nodes[parent_index].local_transform.rows[0].w, 3.0f) &&
                        nearly_equal(loaded_hierarchy.nodes[parent_index].local_transform.rows[1].y, 3.0f) &&
                        nearly_equal(loaded_hierarchy.nodes[parent_index].local_transform.rows[2].z, -1.0f),
        "hierarchy affine transform round trips") == false) {
    return false;
  }
  if (check_condition((loaded_hierarchy.mesh_instances.size() == 1u) && nearly_equal(loaded_hierarchy.mesh_instances[0].bbox_min.x, 3.0f) &&
                        nearly_equal(loaded_hierarchy.mesh_instances[0].bbox_max.y, 7.0f),
        "round-tripped instance resolves in world space") == false) {
    return false;
  }

  std::filesystem::remove(obj_path);
  std::filesystem::remove(std::filesystem::path(saved_file));
  std::filesystem::remove(std::filesystem::path(saved_file).replace_extension(".materials"));
  std::filesystem::remove(std::filesystem::path(saved_file).replace_extension(""));
  return true;
}

}  // namespace

int main() {
  struct TestCase {
    const char* name = nullptr;
    bool (*function)() = nullptr;
  };

  const TestCase tests[] = {
    {"sphere_generation", test_sphere_generation},
    {"plane_generation", test_plane_generation},
    {"etx_alias_and_material_index", test_etx_alias_and_material_index},
    {"plane_alias", test_plane_alias},
    {"duplicate_ids", test_duplicate_ids},
    {"invalid_dimensions_skips", test_invalid_dimensions_skips},
    {"invalid_material_uses_missing", test_invalid_material_uses_missing},
    {"binary_load_skips_baked_procedural_geometry", test_binary_load_skips_baked_procedural_geometry},
    {"native_json_loads_text_only_procedural_geometry", test_native_json_loads_text_only_procedural_geometry},
    {"empty_scene_environment_emitter_packs", test_empty_scene_environment_emitter_packs},
    {"raw_obj_load_adds_default_camera_and_lighting", test_raw_obj_load_adds_default_camera_and_lighting},
    {"directional_use_as_sun_parse", test_directional_use_as_sun_parse},
    {"gltf_double_sided_material", test_gltf_double_sided_material},
    {"native_scene_hierarchy_round_trip", test_native_scene_hierarchy_round_trip},
  };

  uint32_t passed_count = 0u;
  for (const TestCase& test : tests) {
    std::printf("Running %s...\n", test.name);
    if (test.function()) {
      ++passed_count;
      std::printf("PASSED: %s\n", test.name);
    }
  }

  const uint32_t test_count = static_cast<uint32_t>(sizeof(tests) / sizeof(tests[0]));
  std::printf("%u/%u tests passed\n", passed_count, test_count);
  return (passed_count == test_count) ? 0 : 1;
}
