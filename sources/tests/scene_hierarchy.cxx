#include <etx/render/host/scene_hierarchy.hxx>
#include <etx/render/host/film.hxx>
#include <etx/render/host/emitter_packing.hxx>
#include <etx/render/host/scene_data.hxx>
#include <etx/render/host/scene_global.hxx>
#include <etx/render/shared/medium.hxx>
#include <etx/render/shared/sampler.hxx>
#include <etx/rt/rt.hxx>

#include <cstdio>

namespace {

constexpr float kTestEpsilon = 1.0e-5f;

bool check_condition(bool condition, const char* message) {
  if (condition == false) {
    std::printf("FAILED: %s\n", message);
    return false;
  }
  return true;
}

bool nearly_equal(float value, float expected) {
  return std::abs(value - expected) <= kTestEpsilon;
}

etx::AffineTransform make_translation_scale(const float3& translation, const float3& scale) {
  etx::AffineTransform result = {};
  result.rows[0] = {scale.x, 0.0f, 0.0f, translation.x};
  result.rows[1] = {0.0f, scale.y, 0.0f, translation.y};
  result.rows[2] = {0.0f, 0.0f, scale.z, translation.z};
  return result;
}

bool test_deep_hierarchy_is_iterative() {
  etx::SceneHierarchy hierarchy;
  constexpr uint32_t kNodeCount = 100000u;
  uint32_t parent_index = kInvalidIndex;
  const etx::AffineTransform translation = make_translation_scale({1.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f});
  for (uint32_t node_index = 0u; node_index < kNodeCount; ++node_index) {
    parent_index = hierarchy.add_node(nullptr, parent_index, translation);
  }

  if (check_condition(hierarchy.update_world_transforms(), "deep hierarchy resolves") == false) {
    return false;
  }
  if (check_condition(hierarchy.evaluation_order.size() == kNodeCount, "all deep hierarchy nodes evaluated") == false) {
    return false;
  }
  return check_condition(nearly_equal(hierarchy.world_transforms.back().rows[0].w, static_cast<float>(kNodeCount)), "deep hierarchy transform accumulated");
}

bool test_reparent_rejects_cycles() {
  etx::SceneHierarchy hierarchy;
  const etx::AffineTransform identity = {};
  const uint32_t root = hierarchy.add_node("root", kInvalidIndex, identity);
  const uint32_t child = hierarchy.add_node("child", root, identity);
  const uint32_t grandchild = hierarchy.add_node("grandchild", child, identity);

  if (check_condition(hierarchy.set_parent(root, grandchild) == false, "cycle rejected") == false) {
    return false;
  }
  if (check_condition(hierarchy.set_parent(child, child) == false, "self parent rejected") == false) {
    return false;
  }
  return check_condition(hierarchy.update_world_transforms(), "valid topology retained after rejected cycle");
}

bool test_transform_resolution_and_instances() {
  etx::SceneHierarchy hierarchy;
  const etx::AffineTransform root_transform = make_translation_scale({3.0f, 4.0f, 5.0f}, {-2.0f, 3.0f, 4.0f});
  const etx::AffineTransform child_transform = make_translation_scale({1.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f});
  const uint32_t root = hierarchy.add_node("root", kInvalidIndex, root_transform);
  const uint32_t child = hierarchy.add_node("child", root, child_transform);
  const etx::SceneAttachment attachment = {etx::SceneAttachment::Type::Mesh, 0u, 0u, 0u};
  if (check_condition(hierarchy.add_attachment(child, attachment), "mesh attachment added") == false) {
    return false;
  }

  Mesh mesh = {};
  mesh.bbox_min = {-1.0f, -1.0f, -1.0f};
  mesh.bbox_max = {1.0f, 1.0f, 1.0f};
  const std::vector<Mesh> meshes = {mesh};
  if (check_condition(hierarchy.resolve_mesh_instances(meshes), "mesh instances resolve") == false) {
    return false;
  }
  if (check_condition(hierarchy.mesh_instances.size() == 1u, "one mesh instance resolved") == false) {
    return false;
  }

  const etx::ResolvedMeshInstance& instance = hierarchy.mesh_instances[0];
  const float3 point = etx::transform_point(instance.object_to_world, {1.0f, 2.0f, 3.0f});
  const float3 local_point = etx::transform_point(instance.world_to_object, point);
  if (check_condition(nearly_equal(local_point.x, 1.0f) && nearly_equal(local_point.y, 2.0f) && nearly_equal(local_point.z, 3.0f), "inverse transform round trip") == false) {
    return false;
  }
  if (check_condition((instance.flags & etx::ResolvedMeshInstance::Mirrored) != 0u, "mirrored transform flagged") == false) {
    return false;
  }
  if (check_condition(nearly_equal(instance.bbox_min.x, -1.0f) && nearly_equal(instance.bbox_max.x, 3.0f), "transformed bounds use center and extent") == false) {
    return false;
  }

  etx::AffineTransform singular = {};
  singular.rows[2].z = 0.0f;
  if (check_condition(hierarchy.set_local_transform(root, singular), "singular transform accepted for authoring") == false) {
    return false;
  }
  if (check_condition(hierarchy.resolve_mesh_instances(meshes), "singular hierarchy still resolves") == false) {
    return false;
  }
  return check_condition((hierarchy.mesh_instances.size() == 1u) && ((hierarchy.mesh_instances[0].flags & etx::ResolvedMeshInstance::Enabled) == 0u),
    "singular instance retained but masked from rendering");
}

bool test_attachment_ranges_remain_valid() {
  etx::SceneHierarchy hierarchy;
  const uint32_t node_a = hierarchy.add_node("a", kInvalidIndex, {});
  const uint32_t node_b = hierarchy.add_node("b", kInvalidIndex, {});
  const etx::SceneAttachment mesh_0 = {etx::SceneAttachment::Type::Mesh, 0u, 0u, 0u};
  const etx::SceneAttachment mesh_1 = {etx::SceneAttachment::Type::Mesh, 1u, 0u, 0u};
  const etx::SceneAttachment mesh_2 = {etx::SceneAttachment::Type::Mesh, 2u, 0u, 0u};
  hierarchy.add_attachment(node_a, mesh_0);
  hierarchy.add_attachment(node_b, mesh_1);
  hierarchy.add_attachment(node_a, mesh_2);

  const etx::SceneNode& a = hierarchy.nodes[node_a];
  const etx::SceneNode& b = hierarchy.nodes[node_b];
  if (check_condition((a.attachment_count == 2u) && (b.attachment_count == 1u), "attachment counts retained") == false) {
    return false;
  }
  if (check_condition(hierarchy.attachments[a.attachment_offset + 1u].resource_index == 2u, "inserted attachment remains contiguous") == false) {
    return false;
  }
  return check_condition(hierarchy.attachments[b.attachment_offset].resource_index == 1u, "later attachment offset repaired");
}

bool test_visibility_is_inherited() {
  etx::SceneHierarchy hierarchy;
  const uint32_t root = hierarchy.add_node("root", kInvalidIndex, {});
  const uint32_t child = hierarchy.add_node("child", root, {});
  const etx::SceneAttachment attachment = {etx::SceneAttachment::Type::Mesh, 0u, 0u, 0u};
  hierarchy.add_attachment(child, attachment);

  const std::vector<Mesh> meshes(1u);
  if (check_condition(hierarchy.set_enabled(root, false), "root disabled") == false) {
    return false;
  }
  if (check_condition(hierarchy.resolve_mesh_instances(meshes), "disabled hierarchy resolves") == false) {
    return false;
  }
  if (check_condition((hierarchy.mesh_instances.size() == 1u) && ((hierarchy.mesh_instances[0].flags & etx::ResolvedMeshInstance::Enabled) == 0u),
        "disabled parent masks child attachment without changing instance topology") == false) {
    return false;
  }
  if (check_condition(hierarchy.set_enabled(root, true), "root enabled") == false) {
    return false;
  }
  if (check_condition(hierarchy.resolve_mesh_instances(meshes), "enabled hierarchy resolves") == false) {
    return false;
  }
  return check_condition((hierarchy.mesh_instances.size() == 1u) && ((hierarchy.mesh_instances[0].flags & etx::ResolvedMeshInstance::Enabled) != 0u),
    "enabled subtree restores child attachment");
}

bool test_mirrored_tangent_frame_handedness() {
  SceneInstance instance = {};
  instance.object_to_world.rows[0].x = -1.0f;
  instance.world_to_object.rows[0].x = -1.0f;
  instance.flags = SceneInstance::Mirrored | SceneInstance::Enabled;

  Vertex vertex = {};
  vertex.nrm = {0.0f, 0.0f, 1.0f};
  vertex.tan = {1.0f, 0.0f, 0.0f};
  vertex.btn = {0.0f, 1.0f, 0.0f};
  const Vertex transformed = etx::scene_instance_transform_vertex(instance, vertex);
  if (check_condition(nearly_equal(transformed.nrm.z, -1.0f) && nearly_equal(transformed.tan.x, -1.0f), "mirrored normal and tangent follow transformed winding") == false) {
    return false;
  }
  return check_condition(nearly_equal(transformed.btn.y, 1.0f) && (dot(cross(transformed.nrm, transformed.tan), transformed.btn) > 0.0f),
    "mirrored bitangent preserves the source tangent-frame handedness");
}

bool test_disabled_attached_emitter_is_not_global() {
  etx::TaskScheduler scheduler = {};
  etx::SceneData scene_data(scheduler);
  const uint32_t emitter_index = static_cast<uint32_t>(scene_data.emitter_profiles.size());
  etx::EmitterProfile& emitter = scene_data.emitter_profiles.emplace_back(etx::EmitterProfile::Class::Directional);
  emitter.directional.direction = {0.0f, 0.0f, -1.0f};
  const uint32_t node_index = scene_data.hierarchy.add_node("light", kInvalidIndex, {});
  scene_data.hierarchy.add_attachment(node_index, {etx::SceneAttachment::Type::Emitter, emitter_index, 0u, 0u});
  scene_data.hierarchy.set_enabled(node_index, false);
  if (check_condition(scene_data.resolve_hierarchy(), "disabled emitter hierarchy resolves") == false) {
    return false;
  }
  if (check_condition(etx::build_packed_emitters(scene_data).emitter_instances.empty(), "disabled attached emitter is omitted") == false) {
    return false;
  }

  scene_data.hierarchy.set_enabled(node_index, true);
  if (check_condition(scene_data.resolve_hierarchy(), "enabled emitter hierarchy resolves") == false) {
    return false;
  }
  return check_condition(etx::build_packed_emitters(scene_data).emitter_instances.size() == 1u, "enabled attached emitter is packed once");
}

bool test_hierarchy_hashes_content() {
  etx::TaskScheduler scheduler = {};
  etx::SceneData first(scheduler);
  etx::SceneData second(scheduler);
  first.hierarchy.add_node("node", kInvalidIndex, make_translation_scale({1.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f}));
  second.hierarchy.add_node("node", kInvalidIndex, make_translation_scale({2.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f}));
  const etx::SceneHashes first_hashes = first.compute_hashes();
  const etx::SceneHashes second_hashes = second.compute_hashes();
  return check_condition(first_hashes.transforms_hash != second_hashes.transforms_hash, "equal revision counts cannot hide different hierarchy transform content");
}

bool test_attachment_removal_repairs_ranges() {
  etx::SceneHierarchy hierarchy;
  const uint32_t node_a = hierarchy.add_node("a", kInvalidIndex, {});
  const uint32_t node_b = hierarchy.add_node("b", kInvalidIndex, {});
  hierarchy.add_attachment(node_a, {etx::SceneAttachment::Type::Mesh, 0u, 0u, 0u});
  hierarchy.add_attachment(node_b, {etx::SceneAttachment::Type::Mesh, 1u, 0u, 0u});
  hierarchy.add_attachment(node_a, {etx::SceneAttachment::Type::Mesh, 2u, 0u, 0u});

  if (check_condition(hierarchy.remove_attachment(node_a, 0u), "first attachment removed") == false) {
    return false;
  }
  const etx::SceneNode& a = hierarchy.nodes[node_a];
  const etx::SceneNode& b = hierarchy.nodes[node_b];
  if (check_condition((a.attachment_count == 1u) && (hierarchy.attachments[a.attachment_offset].resource_index == 2u), "remaining attachment retained") == false) {
    return false;
  }
  return check_condition((b.attachment_count == 1u) && (hierarchy.attachments[b.attachment_offset].resource_index == 1u), "other attachment range repaired");
}

bool test_transformed_medium_coordinates_and_bounds() {
  etx::Medium medium = {};
  medium.local_bounds = {{-1.0f, -1.0f, -1.0f}, 0.0f, {1.0f, 1.0f, 1.0f}, 0.0f};
  medium.bounds = {{3.0f, -1.0f, -1.0f}, 0.0f, {7.0f, 1.0f, 1.0f}, 0.0f};
  medium.world_to_object.rows[0] = {0.5f, 0.0f, 0.0f, -2.5f};
  medium.set_grid_type(etx::DensityGrid::Type::NoiseFunction);
  medium.set_noise_type(etx::NoiseFunction::Uniform);

  const float3 center = medium_world_to_local(medium.world_to_object, medium.local_bounds, {5.0f, 0.0f, 0.0f});
  if (check_condition(nearly_equal(center.x, 0.5f) && nearly_equal(center.y, 0.5f) && nearly_equal(center.z, 0.5f), "medium world-to-local mapping") == false) {
    return false;
  }
  if (check_condition(nearly_equal(medium.sample_density_world({5.0f, 0.0f, 0.0f}), 1.0f), "transformed medium samples inside local bounds") == false) {
    return false;
  }
  if (check_condition(nearly_equal(medium.sample_density_world({8.0f, 0.0f, 0.0f}), 0.0f), "transformed medium rejects points outside local bounds") == false) {
    return false;
  }

  etx::MediumSharedIntersection intersection = {};
  if (check_condition(etx::medium_shared_intersects_bounds(medium.bounds.p_min, medium.bounds.p_max, {0.0f, 0.0f, 0.0f}, {2.0f, 0.0f, 0.0f}, 10.0f, intersection),
        "world-space medium bounds intersect") == false) {
    return false;
  }
  return check_condition(nearly_equal(intersection.t_min, 3.0f) && nearly_equal(intersection.t_max, 7.0f), "medium entry and exit distances stay in world units");
}

bool test_embree_transform_only_commit() {
  struct SceneGlobalGuard {
    SceneGlobalGuard() {
      etx::scene_global_init();
    }
    ~SceneGlobalGuard() {
      etx::scene_global_deinit();
    }
  } scene_global_guard;

  etx::TaskScheduler scheduler = {};
  etx::SceneData scene_data(scheduler);
  scene_data.vertices.pos = {{-0.5f, -0.5f, 0.0f}, {0.5f, -0.5f, 0.0f}, {0.0f, 0.5f, 0.0f}};
  scene_data.vertices.nrm.assign(3u, {0.0f, 0.0f, 1.0f});
  scene_data.vertices.tan.assign(3u, {1.0f, 0.0f, 0.0f});
  scene_data.vertices.btn.assign(3u, {0.0f, 1.0f, 0.0f});
  scene_data.vertices.tex.assign(3u, {});

  Triangle triangle = {};
  triangle.i[0] = 0u;
  triangle.i[1] = 1u;
  triangle.i[2] = 2u;
  triangle.material_index = 0u;
  triangle.geo_n = {0.0f, 0.0f, 1.0f};
  scene_data.triangles.emplace_back(triangle);

  Material material = {};
  material.cls = MaterialClass::Diffuse;
  material.opacity = 1.0f;
  scene_data.materials.emplace_back(material);
  scene_data.add_mesh("triangle", 0u, 1u, {-0.5f, -0.5f, 0.0f}, {0.5f, 0.5f, 0.0f});
  if (check_condition(scene_data.resolve_hierarchy(), "Embree validation hierarchy resolves") == false) {
    return false;
  }

  Camera camera = {};
  camera.film_size = {1u, 1u};
  etx::Film film(scheduler);
  etx::Raytracing raytracing(scheduler, film);
  const etx::SceneHashes initial_hashes = scene_data.compute_hashes();
  raytracing.commit(scene_data, camera, initial_hashes.compare({}));

  etx::Sampler sampler(1u, 2u);
  Intersection intersection = {};
  if (check_condition(raytracing.trace(raytracing.scene(), Ray{{0.0f, 0.0f, 1.0f}, {0.0f, 0.0f, -1.0f}}, intersection, sampler), "initial Embree instance hit") == false) {
    return false;
  }

  const etx::AffineTransform translated = make_translation_scale({3.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f});
  if (check_condition(scene_data.hierarchy.set_local_transform(0u, translated) && scene_data.resolve_hierarchy(), "transformed Embree hierarchy resolves") == false) {
    return false;
  }
  const etx::SceneHashes transformed_hashes = scene_data.compute_hashes();
  const etx::UpdateFlags transform_changes = transformed_hashes.compare(initial_hashes);
  if (check_condition(transform_changes[etx::UpdateFlags::Transforms] && (transform_changes[etx::UpdateFlags::AnyGeometryStructure] == false),
        "transform-only update classified for refit") == false) {
    return false;
  }
  raytracing.commit(scene_data, camera, transform_changes);

  etx::Sampler old_position_sampler(3u, 4u);
  Intersection old_position_intersection = {};
  if (check_condition(raytracing.trace(raytracing.scene(), Ray{{0.0f, 0.0f, 1.0f}, {0.0f, 0.0f, -1.0f}}, old_position_intersection, old_position_sampler) == false,
        "Embree refit removes old instance position") == false) {
    return false;
  }
  etx::Sampler new_position_sampler(5u, 6u);
  Intersection new_position_intersection = {};
  return check_condition(raytracing.trace(raytracing.scene(), Ray{{3.0f, 0.0f, 1.0f}, {0.0f, 0.0f, -1.0f}}, new_position_intersection, new_position_sampler),
    "Embree refit exposes transformed instance position");
}

}  // namespace

int main() {
  struct TestCase {
    const char* name = nullptr;
    bool (*function)() = nullptr;
  };
  const TestCase tests[] = {
    {"deep_hierarchy_is_iterative", test_deep_hierarchy_is_iterative},
    {"reparent_rejects_cycles", test_reparent_rejects_cycles},
    {"transform_resolution_and_instances", test_transform_resolution_and_instances},
    {"attachment_ranges_remain_valid", test_attachment_ranges_remain_valid},
    {"visibility_is_inherited", test_visibility_is_inherited},
    {"mirrored_tangent_frame_handedness", test_mirrored_tangent_frame_handedness},
    {"disabled_attached_emitter_is_not_global", test_disabled_attached_emitter_is_not_global},
    {"hierarchy_hashes_content", test_hierarchy_hashes_content},
    {"attachment_removal_repairs_ranges", test_attachment_removal_repairs_ranges},
    {"transformed_medium_coordinates_and_bounds", test_transformed_medium_coordinates_and_bounds},
    {"embree_transform_only_commit", test_embree_transform_only_commit},
  };

  uint32_t passed = 0u;
  for (const TestCase& test : tests) {
    std::printf("Running %s...\n", test.name);
    if (test.function()) {
      ++passed;
      std::printf("PASSED: %s\n", test.name);
    }
  }
  std::printf("%u/%zu tests passed\n", passed, std::size(tests));
  return passed == std::size(tests) ? 0 : 1;
}
