#include <etx/render/host/scene_hierarchy.hxx>
#include <etx/render/host/buffer_pool.hxx>
#include <etx/render/host/film.hxx>
#include <etx/render/host/emitter_packing.hxx>
#include <etx/render/host/scene_data.hxx>
#include <etx/render/host/scene_global.hxx>
#include <etx/render/host/scene_representation.hxx>
#include <etx/render/interop/camera_film_shared.hxx>
#include <etx/render/shared/ior_database.hxx>
#include <etx/render/shared/medium.hxx>
#include <etx/render/shared/sampler.hxx>
#include <etx/rt/rt.hxx>
#include <etx/rt/integrators/integrator.hxx>
#include <etx/rt/shared/vcm_shared.hxx>
#include <raytracer/renderer.hxx>

#include <cstdio>
#include <limits>

namespace {

constexpr float kTestEpsilon = 1.0e-5f;

struct RendererProbe : etx::Renderer {
  using Renderer::Renderer;

  void initialize_camera(etx::SceneRepresentation& scene) {
    _camera_controller = std::make_unique<etx::CameraController>(scene.mutable_camera());
    _camera_controller->enable_inertia = false;
    reset_preview_state();
  }

  void on_camera_changed(etx::SceneRepresentation& scene) override {
    (void)scene;
    camera_changed_count += 1u;
  }

  void on_camera_become_steady(etx::SceneRepresentation& scene) override {
    (void)scene;
    camera_steady_count += 1u;
  }

  const char* name() const override {
    return "probe";
  }

  etx::RendererMode mode() const override {
    return etx::RendererMode::CPURaytracing;
  }

  uint32_t camera_changed_count = 0u;
  uint32_t camera_steady_count = 0u;
};

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

bool nearly_equal(const float3& value, const float3& expected) {
  return nearly_equal(value.x, expected.x) && nearly_equal(value.y, expected.y) && nearly_equal(value.z, expected.z);
}

float3 rotate_by_quaternion(const float4& rotation, const float3& direction) {
  const float3 imaginary = {rotation.x, rotation.y, rotation.z};
  const float3 twice_cross = 2.0f * cross(imaginary, direction);
  return direction + rotation.w * twice_cross + cross(imaginary, twice_cross);
}

bool affine_nearly_equal(const etx::AffineTransform& value, const etx::AffineTransform& expected) {
  for (uint32_t row = 0u; row < 3u; ++row) {
    const float4& a = value.rows[row];
    const float4& b = expected.rows[row];
    if ((nearly_equal(a.x, b.x) == false) || (nearly_equal(a.y, b.y) == false) || (nearly_equal(a.z, b.z) == false) || (nearly_equal(a.w, b.w) == false)) {
      return false;
    }
  }
  return true;
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

bool test_reparent_preserves_world_transform() {
  etx::SceneHierarchy hierarchy;
  etx::AffineTRS first_parent_trs = {};
  first_parent_trs.translation = {2.0f, 3.0f, 4.0f};
  first_parent_trs.rotation_radians = {0.2f, 0.1f, -0.3f};
  first_parent_trs.scale = {2.0f, 3.0f, 4.0f};
  etx::AffineTRS second_parent_trs = {};
  second_parent_trs.translation = {-5.0f, 1.0f, 7.0f};
  second_parent_trs.rotation_radians = {-0.4f, 0.3f, 0.15f};
  second_parent_trs.scale = {0.5f, 2.0f, 1.5f};
  etx::AffineTRS child_trs = {};
  child_trs.translation = {1.0f, -2.0f, 0.5f};
  child_trs.rotation_radians = {0.1f, 0.2f, 0.3f};

  const uint32_t first_parent = hierarchy.add_node("first", kInvalidIndex, etx::affine_from_trs(first_parent_trs));
  const uint32_t second_parent = hierarchy.add_node("second", kInvalidIndex, etx::affine_from_trs(second_parent_trs));
  const uint32_t child = hierarchy.add_node("child", first_parent, etx::affine_from_trs(child_trs));
  if (check_condition(hierarchy.update_world_transforms(), "reparent test hierarchy resolves") == false) {
    return false;
  }
  const etx::AffineTransform original_world = hierarchy.world_transforms[child];
  if (check_condition(hierarchy.reparent_preserve_world(child, second_parent), "world-preserving reparent succeeds") == false ||
      check_condition(hierarchy.update_world_transforms(), "reparented hierarchy resolves") == false) {
    return false;
  }
  if (check_condition(affine_nearly_equal(hierarchy.world_transforms[child], original_world), "reparent keeps the child world transform") == false) {
    return false;
  }

  etx::AffineTransform singular_parent_transform = {};
  singular_parent_transform.rows[1] = {};
  const uint32_t singular_parent = hierarchy.add_node("singular", kInvalidIndex, singular_parent_transform);
  const uint32_t current_parent = hierarchy.nodes[child].parent_index;
  return check_condition((hierarchy.reparent_preserve_world(child, singular_parent) == false) && (hierarchy.nodes[child].parent_index == current_parent),
    "reparent to a singular parent is rejected without mutation");
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

bool test_affine_trs_round_trip_and_shear_detection() {
  etx::AffineTRS source = {};
  source.translation = {1.0f, -2.0f, 3.0f};
  source.rotation_radians = {20.0f * kPi / 180.0f, -35.0f * kPi / 180.0f, 70.0f * kPi / 180.0f};
  source.scale = {-2.0f, 3.0f, 4.0f};
  const etx::AffineTransform transform = etx::affine_from_trs(source);

  etx::AffineTRS decomposed = {};
  if (check_condition(etx::affine_to_trs(transform, decomposed), "TRS transform decomposes") == false) {
    return false;
  }
  if (check_condition(affine_nearly_equal(etx::affine_from_trs(decomposed), transform), "TRS decomposition preserves reflected transform") == false) {
    return false;
  }

  etx::AffineTransform sheared = transform;
  sheared.rows[0].y += 0.25f;
  return check_condition(etx::affine_to_trs(sheared, decomposed) == false, "sheared affine transform is not silently converted to TRS");
}

bool test_affine_inverse_is_scale_aware() {
  for (const float uniform_scale : {1.0e-8f, 1.0e8f}) {
    const etx::AffineTransform transform = make_translation_scale({}, {uniform_scale, uniform_scale, uniform_scale});
    etx::AffineTransform inverse = {};
    double determinant = 0.0;
    if (check_condition(etx::invert_affine(transform, inverse, determinant), "well-conditioned uniform scale is invertible regardless of magnitude") == false) {
      return false;
    }
    const float3 point = {2.0f, -3.0f, 4.0f};
    const float3 round_trip = etx::transform_point(inverse, etx::transform_point(transform, point));
    if (check_condition(nearly_equal(round_trip.x, point.x) && nearly_equal(round_trip.y, point.y) && nearly_equal(round_trip.z, point.z), "scale-aware inverse round trip") ==
        false) {
      return false;
    }
  }

  const etx::AffineTransform ill_conditioned = make_translation_scale({}, {1.0e-8f, 1.0f, 1.0f});
  etx::AffineTransform inverse = {};
  double determinant = 0.0;
  return check_condition(etx::invert_affine(ill_conditioned, inverse, determinant) == false, "numerically ill-conditioned affine transform is rejected");
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

bool test_shading_frame_finalization() {
  etx::AffineTRS trs = {};
  trs.rotation_radians = {0.25f, -0.4f, 0.1f};
  trs.scale = {-2.0f, 3.0f, 0.5f};

  SceneInstance instance = {};
  instance.object_to_world = etx::affine_from_trs(trs);
  double determinant = 0.0;
  if (check_condition(etx::invert_affine(instance.object_to_world, instance.world_to_object, determinant), "shading-frame transform is invertible") == false) {
    return false;
  }
  instance.flags = SceneInstance::Mirrored | SceneInstance::Enabled;

  Vertex vertex = {};
  vertex.nrm = normalize(float3{0.2f, 0.3f, 1.0f});
  vertex.tan = normalize(float3{1.0f, 0.0f, -0.2f});
  vertex.btn = -normalize(cross(vertex.nrm, vertex.tan));
  const Vertex transformed = etx::scene_instance_transform_vertex(instance, vertex);

  const float3 geo_normal = transformed.nrm;
  const float3 incoming_direction = -geo_normal;
  const float3 conflicting_normal = -geo_normal;
  float3 normal = {};
  float3 tangent = {};
  float3 bitangent = {};
  scene_math_shared_finalize_shading_frame(conflicting_normal, transformed.nrm, transformed.tan, transformed.btn, geo_normal, incoming_direction, normal, tangent, bitangent);

  const bool orthonormal = nearly_equal(dot(normal, normal), 1.0f) && nearly_equal(dot(tangent, tangent), 1.0f) && nearly_equal(dot(bitangent, bitangent), 1.0f) &&
                           nearly_equal(dot(normal, tangent), 0.0f) && nearly_equal(dot(normal, bitangent), 0.0f) && nearly_equal(dot(tangent, bitangent), 0.0f);
  if (check_condition(orthonormal, "finalized shading frame is orthonormal after mirrored non-uniform scaling") == false) {
    return false;
  }
  if (check_condition((dot(incoming_direction, normal) * dot(incoming_direction, geo_normal)) > 0.0f, "shading normal is oriented to the geometric hemisphere") == false) {
    return false;
  }
  if (check_condition(dot(cross(normal, tangent), bitangent) < 0.0f, "shading-frame finalization preserves negative handedness") == false) {
    return false;
  }

  const float3 missing_hint = {};
  scene_math_shared_build_sampling_frame(geo_normal, missing_hint, missing_hint, normal, tangent, bitangent);
  if (check_condition(nearly_equal(dot(normal, tangent), 0.0f) && nearly_equal(dot(normal, bitangent), 0.0f) && nearly_equal(dot(tangent, bitangent), 0.0f) &&
                        nearly_equal(dot(tangent, tangent), 1.0f) && nearly_equal(dot(bitangent, bitangent), 1.0f),
        "missing tangent data receives a valid fallback frame") == false) {
    return false;
  }

  Vertex interpolated = {};
  const float3 position_0 = {};
  const float3 position_1 = {1.0f, 0.0f, 0.0f};
  const float3 position_2 = {0.0f, 1.0f, 0.0f};
  const float3 source_normal = {0.0f, 0.0f, 1.0f};
  const float3 parallel_tangent = source_normal;
  surface_point_shared_interpolate_vertex(position_0, position_1, position_2, source_normal, source_normal, source_normal, parallel_tangent, parallel_tangent, parallel_tangent,
    missing_hint, missing_hint, missing_hint, float2{}, float2{}, float2{}, float3{0.25f, 0.25f, 0.5f}, true, false, interpolated);
  return check_condition(nearly_equal(dot(interpolated.nrm, interpolated.tan), 0.0f) && nearly_equal(dot(interpolated.nrm, interpolated.btn), 0.0f) &&
                           nearly_equal(dot(interpolated.tan, interpolated.btn), 0.0f) && nearly_equal(dot(interpolated.tan, interpolated.tan), 1.0f) &&
                           nearly_equal(dot(interpolated.btn, interpolated.btn), 1.0f),
    "degenerate interpolated tangent hints receive a valid fallback frame");
}

bool test_vcm_vertex_restores_world_tangent_frame() {
  const float3 positions[] = {{0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}};
  const float3 normals[] = {{0.0f, 0.0f, 1.0f}, {0.0f, 0.0f, 1.0f}, {0.0f, 0.0f, 1.0f}};
  const float3 tangents[] = {{1.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}};
  const float3 bitangents[] = {{0.0f, 1.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 1.0f, 0.0f}};
  const float2 texcoords[] = {{0.0f, 0.0f}, {1.0f, 0.0f}, {0.0f, 1.0f}};
  Triangle triangle = {};
  triangle.i[0] = 0u;
  triangle.i[1] = 1u;
  triangle.i[2] = 2u;

  etx::AffineTRS trs = {};
  trs.rotation_radians.y = kHalfPi;
  trs.scale = {2.0f, 3.0f, 4.0f};
  SceneInstance instance = {};
  instance.object_to_world = etx::affine_from_trs(trs);
  double determinant = 0.0;
  if (check_condition(etx::invert_affine(instance.object_to_world, instance.world_to_object, determinant), "VCM test transform is invertible") == false) {
    return false;
  }

  etx::Scene scene = {};
  scene.vertices.pos = etx::ArrayView<float3>{positions, std::size(positions)};
  scene.vertices.nrm = etx::ArrayView<float3>{normals, std::size(normals)};
  scene.vertices.tan = etx::ArrayView<float3>{tangents, std::size(tangents)};
  scene.vertices.btn = etx::ArrayView<float3>{bitangents, std::size(bitangents)};
  scene.vertices.tex = etx::ArrayView<float2>{texcoords, std::size(texcoords)};
  scene.triangles = etx::ArrayView<Triangle>{&triangle, 1u};
  scene.instances = etx::ArrayView<SceneInstance>{&instance, 1u};

  etx::VCMLightVertex light_vertex = {};
  light_vertex.triangle_index = 0u;
  light_vertex.instance_index = 0u;
  light_vertex.bc = {1.0f, 0.0f, 0.0f};
  light_vertex.pos = {5.0f, 6.0f, 7.0f};
  light_vertex.nrm = normalize(float3{1.0f, 0.5f, 0.0f});
  const Vertex restored = light_vertex.vertex(scene);
  if (check_condition(nearly_equal(restored.pos.x, 5.0f) && nearly_equal(restored.pos.y, 6.0f) && nearly_equal(restored.pos.z, 7.0f),
        "VCM connection vertex preserves its stored world position") == false) {
    return false;
  }
  if (check_condition(nearly_equal(restored.tan.x, 0.0f) && nearly_equal(restored.tan.y, 0.0f) && nearly_equal(restored.tan.z, -1.0f),
        "VCM connection tangent follows the instance transform") == false) {
    return false;
  }
  return check_condition(nearly_equal(dot(restored.nrm, restored.tan), 0.0f) && nearly_equal(length(restored.btn), 1.0f),
    "VCM connection frame is orthonormalized around the stored shading normal");
}

bool test_equirectangular_camera_uses_orientation() {
  Camera camera = {};
  camera.cls = Camera::Class::Equirectangular;
  camera.position = {2.0f, 3.0f, 4.0f};
  camera.direction = {0.0f, 0.0f, -1.0f};
  camera.up = {0.0f, 1.0f, 0.0f};
  camera.side = {1.0f, 0.0f, 0.0f};

  const Ray center_ray = camera_generate_ray(camera, {0.0f, 0.0f}, {});
  if (check_condition(nearly_equal(center_ray.d.x, 0.0f) && nearly_equal(center_ray.d.y, 0.0f) && nearly_equal(center_ray.d.z, -1.0f),
        "equirectangular center ray follows camera direction") == false) {
    return false;
  }

  const Ray quarter_turn_ray = camera_generate_ray(camera, {0.5f, 0.0f}, {});
  if (check_condition(nearly_equal(quarter_turn_ray.d.x, 1.0f) && nearly_equal(quarter_turn_ray.d.y, 0.0f) && nearly_equal(quarter_turn_ray.d.z, 0.0f),
        "equirectangular horizontal angle follows camera side") == false) {
    return false;
  }

  const float3 world_point = camera.position + 4.0f * camera.direction;
  const CameraFilmSampleShared film_sample = camera_film_shared_evaluate(camera, world_point, camera.position);
  if (check_condition(nearly_equal(film_sample.uv.x, 0.0f) && nearly_equal(film_sample.uv.y, 0.0f), "world direction maps back to oriented equirectangular film center") == false) {
    return false;
  }

  const CameraFilmEvalShared film_eval = camera_film_shared_evaluate_out(camera, center_ray);
  return check_condition(std::isfinite(film_eval.pdf_dir) && (film_eval.pdf_dir > 0.0f), "oriented equirectangular ray has a valid solid-angle PDF");
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

bool test_directional_and_area_emitters_use_node_transforms() {
  etx::TaskScheduler scheduler = {};
  etx::SceneData scene_data(scheduler);

  etx::EmitterProfile& directional = scene_data.emitter_profiles.emplace_back(etx::EmitterProfile::Class::Directional);
  directional.directional.direction = {0.0f, 0.0f, -1.0f};
  etx::AffineTRS light_parent_trs = {};
  light_parent_trs.rotation_radians.z = kHalfPi;
  light_parent_trs.scale = {2.0f, 3.0f, 4.0f};
  const uint32_t light_parent = scene_data.hierarchy.add_node("directional-parent", kInvalidIndex, etx::affine_from_trs(light_parent_trs));
  etx::AffineTRS light_trs = {};
  light_trs.rotation_radians.y = 0.25f * kPi;
  const uint32_t light_node = scene_data.hierarchy.add_node("directional", light_parent, etx::affine_from_trs(light_trs));
  scene_data.hierarchy.add_attachment(light_node, {etx::SceneAttachment::Type::Emitter, 0u, 0u, 0u});

  scene_data.vertices.pos = {{0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}};
  Triangle triangle = {};
  triangle.i[0] = 0u;
  triangle.i[1] = 1u;
  triangle.i[2] = 2u;
  triangle.material_index = 0u;
  triangle.emitter_index = 1u;
  scene_data.triangles.push_back(triangle);
  scene_data.materials.emplace_back();
  scene_data.emitter_profiles.emplace_back(etx::EmitterProfile::Class::Area);
  scene_data.add_mesh_asset("area", 0u, 1u, {}, {1.0f, 1.0f, 0.0f});
  const uint32_t area_node = scene_data.hierarchy.add_node("area", kInvalidIndex, make_translation_scale({}, {2.0f, 3.0f, 1.0f}));
  scene_data.hierarchy.add_attachment(area_node, {etx::SceneAttachment::Type::Mesh, 0u, 0u, 0u});

  if (check_condition(scene_data.resolve_hierarchy(), "emitter hierarchy resolves") == false) {
    return false;
  }
  etx::PackedEmitterTopology topology = {};
  const etx::PackedEmitterData packed = etx::build_packed_emitters(scene_data, topology);
  if (check_condition(packed.emitter_instances.size() == 2u, "directional and area emitters are packed") == false) {
    return false;
  }

  const etx::Emitter& packed_directional = packed.emitter_instances[0];
  const float3 direction = packed.emitter_profiles[packed_directional.profile].directional.direction;
  const float expected_component = -std::sqrt(0.5f);
  if (check_condition(nearly_equal(direction.x, 0.0f) && nearly_equal(direction.y, expected_component) && nearly_equal(direction.z, expected_component),
        "directional emitter composes node rotations without scale-induced skew") == false) {
    return false;
  }
  if (check_condition(nearly_equal(packed.emitter_instances[1].triangle_area, 3.0f), "area emitter sampling area uses transformed triangle") == false) {
    return false;
  }

  scene_data.hierarchy.set_local_transform(area_node, make_translation_scale({}, {4.0f, 3.0f, 1.0f}));
  if (check_condition(scene_data.resolve_hierarchy(), "transformed area emitter hierarchy resolves") == false) {
    return false;
  }
  const etx::PackedEmitterData transform_packed = etx::build_packed_emitters_for_transforms(scene_data, topology);
  const etx::PackedEmitterData reference_packed = etx::build_packed_emitters(scene_data);
  if (check_condition(transform_packed.triangles.empty(), "transform-only emitter packing does not copy static triangles") == false) {
    return false;
  }
  if (check_condition(
        (transform_packed.instances.size() == reference_packed.instances.size()) && (transform_packed.emitter_instances.size() == reference_packed.emitter_instances.size()),
        "transform-only emitter packing preserves instance and emitter counts") == false) {
    return false;
  }
  if (check_condition(affine_nearly_equal(transform_packed.instances[0].object_to_world, reference_packed.instances[0].object_to_world),
        "transform-only emitter packing preserves instance transforms") == false) {
    return false;
  }
  return check_condition(nearly_equal(transform_packed.emitter_instances[1].triangle_area, reference_packed.emitter_instances[1].triangle_area) &&
                           nearly_equal(transform_packed.emitter_instances[1].triangle_area, 6.0f),
    "transform-only emitter packing recomputes transformed area-light weights");
}

bool test_scene_update_scope_coalescing() {
  etx::TaskScheduler scheduler = {};
  RendererProbe renderer(scheduler);
  if (check_condition(renderer.consume_scene_update_request() == etx::SceneUpdateScope::Full, "renderer starts with a full scene update") == false) {
    return false;
  }
  renderer.request_scene_transform_update();
  if (check_condition(renderer.consume_scene_update_request() == etx::SceneUpdateScope::Transforms, "transform request selects the fast update scope") == false) {
    return false;
  }
  renderer.request_scene_transform_update();
  renderer.request_scene_update();
  renderer.request_scene_transform_update();
  if (check_condition(renderer.consume_scene_update_request() == etx::SceneUpdateScope::Full, "transform requests cannot downgrade a pending full update") == false) {
    return false;
  }
  return check_condition(renderer.consume_scene_update_request() == etx::SceneUpdateScope::None, "consuming an update request clears the pending scope");
}

bool test_camera_interaction_queues_updates_until_input_released() {
  etx::TaskScheduler scheduler = {};
  etx::IORDatabase ior_database = {};
  etx::SceneRepresentation scene(scheduler, ior_database);
  RendererProbe renderer(scheduler);
  renderer.initialize_camera(scene);
  renderer.consume_scene_update_request();

  etx::CameraController* controller = renderer.camera_controller();
  controller->set_mouse_button_state(etx::CameraController::MouseLeft, true);
  controller->add_mouse_delta(8.0f, 0.0f);
  renderer.update_camera(scene, 1.0f / 60.0f);
  if (check_condition((renderer.camera_changed_count == 1u) && (renderer.camera_steady_count == 0u), "camera motion starts one preview interaction") == false ||
      check_condition(renderer.consume_scene_update_request() == etx::SceneUpdateScope::None, "camera-only motion does not queue a scene-resource update") == false) {
    return false;
  }

  renderer.update_camera(scene, 1.0f / 60.0f);
  if (check_condition(renderer.camera_steady_count == 0u, "held navigation input keeps the preview interaction active between mouse deltas") == false) {
    return false;
  }

  controller->set_mouse_button_state(etx::CameraController::MouseLeft, false);
  renderer.update_camera(scene, 1.0f / 60.0f);
  return check_condition(renderer.camera_steady_count == 1u, "releasing navigation input finishes the preview interaction");
}

bool test_environment_emitter_uses_node_rotation() {
  etx::TaskScheduler scheduler = {};
  etx::SceneData scene_data(scheduler);
  etx::EmitterProfile& source_profile = scene_data.emitter_profiles.emplace_back(etx::EmitterProfile::Class::Environment);
  const float half_angle_sine = std::sqrt(0.5f);
  source_profile.set_environment_rotation({half_angle_sine, 0.0f, 0.0f, half_angle_sine});

  etx::AffineTRS parent_trs = {};
  parent_trs.translation = {10.0f, 20.0f, 30.0f};
  parent_trs.rotation_radians.z = kHalfPi;
  parent_trs.scale = {2.0f, 3.0f, 4.0f};
  const uint32_t parent_index = scene_data.hierarchy.add_node("environment-parent", kInvalidIndex, etx::affine_from_trs(parent_trs));
  etx::AffineTRS environment_trs = {};
  environment_trs.rotation_radians.y = kHalfPi;
  const uint32_t node_index = scene_data.hierarchy.add_node("environment", parent_index, etx::affine_from_trs(environment_trs));
  scene_data.hierarchy.add_attachment(node_index, {etx::SceneAttachment::Type::Emitter, 0u, 0u, 0u});
  if (check_condition(scene_data.resolve_hierarchy(), "environment hierarchy resolves") == false) {
    return false;
  }

  const etx::PackedEmitterData packed = etx::build_packed_emitters(scene_data);
  if (check_condition(packed.emitter_instances.size() == 1u, "attached environment emitter is packed once") == false) {
    return false;
  }
  const etx::EmitterProfile& profile = packed.emitter_profiles[packed.emitter_instances[0].profile];
  const float4 rotation = profile.environment_rotation();
  const float3 basis_directions[] = {{1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f, 1.0f}};
  const float4 source_rotation = source_profile.environment_rotation();
  for (const float3& basis_direction : basis_directions) {
    const float3 profile_direction = rotate_by_quaternion(source_rotation, basis_direction);
    const float3 expected = normalize(etx::transform_vector(scene_data.hierarchy.world_orientations[node_index], profile_direction));
    const float3 actual = rotate_by_quaternion(rotation, basis_direction);
    if (check_condition(nearly_equal(actual.x, expected.x) && nearly_equal(actual.y, expected.y) && nearly_equal(actual.z, expected.z),
          "environment lookup composes profile and node rotations through a sheared world transform") == false) {
      return false;
    }
  }
  return true;
}

bool test_active_camera_uses_node_transform() {
  etx::TaskScheduler scheduler = {};
  etx::IORDatabase ior_database = {};
  etx::SceneRepresentation scene(scheduler, ior_database);
  etx::SceneData& data = scene.data();
  data.cameras.clear();

  etx::SceneData::CameraInfo camera_info = {};
  camera_info.id = "camera";
  camera_info.active = true;
  etx::build_camera(camera_info.cam, {}, {0.0f, 0.0f, -1.0f}, {0.0f, 1.0f, 0.0f}, {64u, 32u}, 60.0f);
  data.cameras.push_back(camera_info);

  etx::AffineTRS camera_parent_trs = {};
  camera_parent_trs.translation = {2.0f, 3.0f, 4.0f};
  camera_parent_trs.scale = {2.0f, 3.0f, 4.0f};
  const uint32_t parent_index = data.hierarchy.add_node("camera-parent", kInvalidIndex, etx::affine_from_trs(camera_parent_trs));
  etx::AffineTRS camera_trs = {};
  camera_trs.rotation_radians.y = 0.25f * kPi;
  const uint32_t node_index = data.hierarchy.add_node("camera", parent_index, etx::affine_from_trs(camera_trs));
  data.hierarchy.add_attachment(node_index, {etx::SceneAttachment::Type::Camera, 0u, 0u, 0u});
  etx::Medium& camera_medium = data.mediums_vector.emplace_back();
  camera_medium.bounds = {{-1.0f, -1.0f, -1.0f}, 0.0f, {1.0f, 1.0f, 1.0f}, 0.0f};
  const uint32_t medium_node_index = data.hierarchy.add_node("camera-medium", node_index, {});
  data.hierarchy.add_attachment(medium_node_index, {etx::SceneAttachment::Type::Medium, 0u, 0u, 0u});
  if (check_condition(data.resolve_hierarchy(), "camera hierarchy resolves") == false) {
    return false;
  }

  scene.update_active_camera();
  const Camera& camera = scene.camera();
  if (check_condition(nearly_equal(camera.position.x, 2.0f) && nearly_equal(camera.position.y, 3.0f) && nearly_equal(camera.position.z, 4.0f),
        "camera position follows node translation") == false) {
    return false;
  }
  const float expected_component = -std::sqrt(0.5f);
  if (check_condition(nearly_equal(camera.direction.x, expected_component) && nearly_equal(camera.direction.y, 0.0f) && nearly_equal(camera.direction.z, expected_component),
        "camera direction composes node rotations without scale-induced skew") == false) {
    return false;
  }
  scene.store_active_camera();
  const Camera& stored_camera = data.cameras[0].cam;
  if (check_condition(nearly_equal(stored_camera.direction.x, 0.0f) && nearly_equal(stored_camera.direction.y, 0.0f) && nearly_equal(stored_camera.direction.z, -1.0f),
        "storing an attached camera applies the inverse orientation without scale-induced skew") == false) {
    return false;
  }

  scene.update_active_camera();
  const float3 edited_position = {5.0f, 6.0f, 7.0f};
  const float3 edited_direction = normalize(float3{-0.25f, 0.1f, -1.0f});
  etx::build_camera(scene.mutable_camera(), edited_position, edited_direction, {0.0f, 1.0f, 0.0f}, {64u, 32u}, 55.0f);
  const Camera expected_world_camera = scene.camera();
  if (check_condition(scene.store_active_camera(), "camera navigation reports transform-dependent rig resources") == false) {
    return false;
  }
  scene.update_medium_bounds();
  const Camera& edited_local_camera = data.cameras[0].cam;
  if (check_condition(
        nearly_equal(edited_local_camera.position.x, 0.0f) && nearly_equal(edited_local_camera.position.y, 0.0f) && nearly_equal(edited_local_camera.position.z, 0.0f),
        "viewport navigation keeps the canonical camera pose in the resource") == false) {
    return false;
  }
  const etx::AffineTransform& edited_camera_world = data.hierarchy.world_transforms[node_index];
  if (check_condition(nearly_equal(edited_camera_world.rows[0].w, edited_position.x) && nearly_equal(edited_camera_world.rows[1].w, edited_position.y) &&
                        nearly_equal(edited_camera_world.rows[2].w, edited_position.z),
        "viewport navigation writes the attached camera node transform through its parent") == false) {
    return false;
  }
  const etx::Medium& moved_medium = data.mediums_vector[0];
  const float3 moved_medium_center = 0.5f * (moved_medium.bounds.p_min + moved_medium.bounds.p_max);
  if (check_condition(nearly_equal(moved_medium_center, edited_position), "camera-driven transform refreshes descendant medium bounds") == false) {
    return false;
  }
  scene.update_active_camera();
  const Camera& restored_world_camera = scene.camera();
  const bool world_edit_preserved =
    nearly_equal(restored_world_camera.position.x, expected_world_camera.position.x) && nearly_equal(restored_world_camera.position.y, expected_world_camera.position.y) &&
    nearly_equal(restored_world_camera.position.z, expected_world_camera.position.z) && nearly_equal(restored_world_camera.direction.x, expected_world_camera.direction.x) &&
    nearly_equal(restored_world_camera.direction.y, expected_world_camera.direction.y) && nearly_equal(restored_world_camera.direction.z, expected_world_camera.direction.z) &&
    nearly_equal(restored_world_camera.up.x, expected_world_camera.up.x) && nearly_equal(restored_world_camera.up.y, expected_world_camera.up.y) &&
    nearly_equal(restored_world_camera.up.z, expected_world_camera.up.z);
  return check_condition(world_edit_preserved, "editing an attached active camera round-trips through local storage without a second node transform");
}

bool test_hierarchy_hashes_content() {
  etx::TaskScheduler scheduler = {};
  etx::SceneData first(scheduler);
  etx::SceneData second(scheduler);
  first.hierarchy.add_node("node", kInvalidIndex, make_translation_scale({1.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f}));
  second.hierarchy.add_node("node", kInvalidIndex, make_translation_scale({2.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f}));
  return check_condition(first.compute_transforms_hash() != second.compute_transforms_hash(), "equal revision counts cannot hide different hierarchy transform content");
}

bool test_enabled_state_is_an_instance_update() {
  etx::TaskScheduler scheduler = {};
  etx::SceneData data(scheduler);
  data.add_mesh("mesh", 0u, 0u, {}, {});
  if (check_condition(data.resolve_hierarchy(), "visibility hash hierarchy resolves") == false) {
    return false;
  }
  const etx::SceneHashes enabled_hashes = data.compute_hashes();
  if (check_condition(data.hierarchy.set_enabled(0u, false) && data.resolve_hierarchy(), "disabled hierarchy resolves") == false) {
    return false;
  }
  etx::SceneHashes disabled_hashes = enabled_hashes;
  disabled_hashes.transforms_hash = data.compute_transforms_hash();
  const etx::UpdateFlags changes = disabled_hashes.compare(enabled_hashes);
  return check_condition(changes[etx::UpdateFlags::Transforms] && (changes[etx::UpdateFlags::AnyGeometryStructure] == false) && changes[etx::UpdateFlags::EmbreeScene],
    "enabled-state changes update instance masks without rebuilding geometry structure");
}

bool test_failed_resolution_remains_dirty_and_recovers() {
  etx::TaskScheduler scheduler = {};
  etx::SceneData data(scheduler);
  data.cameras.emplace_back();
  const uint32_t first_node = data.hierarchy.add_node("first-camera", kInvalidIndex, {});
  const uint32_t second_node = data.hierarchy.add_node("second-camera", kInvalidIndex, {});
  data.hierarchy.add_attachment(first_node, {etx::SceneAttachment::Type::Camera, 0u, 0u, 0u});
  data.hierarchy.add_attachment(second_node, {etx::SceneAttachment::Type::Camera, 0u, 0u, 0u});
  data.hierarchy.set_enabled(second_node, false);
  if (check_condition(data.resolve_hierarchy() && data.hierarchy.resolved_state_current(), "valid hierarchy resolution becomes current") == false) {
    return false;
  }

  data.hierarchy.set_enabled(second_node, true);
  if (check_condition((data.resolve_hierarchy() == false) && (data.hierarchy.resolved_state_current() == false), "failed duplicate attachment resolution remains dirty") == false) {
    return false;
  }

  data.hierarchy.set_enabled(second_node, false);
  return check_condition(data.resolve_hierarchy() && data.hierarchy.resolved_state_current(), "hierarchy recovers after invalid visibility edit is reverted");
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

bool test_attached_medium_bounds_follow_visibility_and_transform() {
  etx::TaskScheduler scheduler = {};
  etx::IORDatabase ior_database = {};
  etx::SceneRepresentation scene(scheduler, ior_database);
  etx::SceneData& data = scene.data();

  etx::Medium& medium = data.mediums_vector.emplace_back();
  medium.bounds = {{-1.0f, -2.0f, -3.0f}, 0.0f, {1.0f, 2.0f, 3.0f}, 0.0f};
  const etx::AffineTransform transform = make_translation_scale({5.0f, 7.0f, 11.0f}, {2.0f, 3.0f, 4.0f});
  const uint32_t node_index = data.hierarchy.add_node("medium", kInvalidIndex, transform);
  data.hierarchy.add_attachment(node_index, {etx::SceneAttachment::Type::Medium, 0u, 0u, 0u});
  if (check_condition(data.resolve_hierarchy(), "attached medium hierarchy resolves") == false) {
    return false;
  }
  scene.update_medium_bounds();

  const etx::Medium& transformed = data.mediums_vector[0];
  const float3 local_center = medium_world_to_local(transformed.world_to_object, transformed.local_bounds, {5.0f, 7.0f, 11.0f});
  if (check_condition(nearly_equal(local_center.x, 0.5f) && nearly_equal(local_center.y, 0.5f) && nearly_equal(local_center.z, 0.5f),
        "attached medium world-to-local transform follows its node") == false) {
    return false;
  }
  if (check_condition(nearly_equal(transformed.bounds.p_min.x, 3.0f) && nearly_equal(transformed.bounds.p_min.y, 1.0f) && nearly_equal(transformed.bounds.p_min.z, -1.0f) &&
                        nearly_equal(transformed.bounds.p_max.x, 7.0f) && nearly_equal(transformed.bounds.p_max.y, 13.0f) && nearly_equal(transformed.bounds.p_max.z, 23.0f),
        "attached medium world bounds follow non-uniform scale and translation") == false) {
    return false;
  }

  data.hierarchy.set_enabled(node_index, false);
  if (check_condition(data.resolve_hierarchy(), "disabled medium hierarchy resolves") == false) {
    return false;
  }
  scene.update_medium_bounds();
  const etx::Medium& disabled = data.mediums_vector[0];
  return check_condition(nearly_equal(disabled.bounds.p_min.x, -1.0f) && nearly_equal(disabled.bounds.p_max.z, 3.0f) && nearly_equal(disabled.world_to_object.rows[0].x, 1.0f) &&
                           nearly_equal(disabled.world_to_object.rows[0].w, 0.0f),
    "disabled medium attachment restores authored bounds and identity mapping");
}

bool test_bake_node_transform_isolates_geometry_and_preserves_children() {
  etx::TaskScheduler scheduler = {};
  etx::IORDatabase ior_database = {};
  etx::SceneRepresentation scene(scheduler, ior_database);
  etx::SceneData& data = scene.data();
  data.vertices.pos = {{0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}};
  data.vertices.nrm.assign(3u, {0.0f, 0.0f, 1.0f});
  data.vertices.tan.assign(3u, {1.0f, 0.0f, 0.0f});
  data.vertices.btn.assign(3u, {0.0f, 1.0f, 0.0f});
  data.vertices.tex.assign(3u, {});
  Triangle triangle = {};
  triangle.i[0] = 0u;
  triangle.i[1] = 1u;
  triangle.i[2] = 2u;
  triangle.geo_n = {0.0f, 0.0f, 1.0f};
  data.triangles.push_back(triangle);
  const uint32_t mesh_index = data.add_mesh_asset("shared", 0u, 1u, {0.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 0.0f});

  const etx::AffineTransform local_transform = make_translation_scale({3.0f, 4.0f, 5.0f}, {-2.0f, 3.0f, 4.0f});
  const uint32_t edited_node = data.hierarchy.add_node("edited", kInvalidIndex, local_transform);
  const uint32_t other_node = data.hierarchy.add_node("other", kInvalidIndex, {});
  const etx::AffineTransform child_local = make_translation_scale({1.0f, 2.0f, 3.0f}, {1.0f, 1.0f, 1.0f});
  const uint32_t child_node = data.hierarchy.add_node("child", edited_node, child_local);
  data.hierarchy.add_attachment(edited_node, {etx::SceneAttachment::Type::Mesh, mesh_index, 0u, 0u});
  data.hierarchy.add_attachment(other_node, {etx::SceneAttachment::Type::Mesh, mesh_index, 0u, 0u});
  if (check_condition(data.resolve_hierarchy(), "bake test hierarchy resolves") == false) {
    return false;
  }
  const etx::SceneHashes original_hashes = data.compute_hashes();
  const etx::AffineTransform original_child_world = data.hierarchy.world_transforms[child_node];
  const float3 original_source_position = data.vertices.pos[1];

  if (check_condition(scene.edit_node_geometry(edited_node, etx::NodeGeometryOperation::BakeLocalTransform) == etx::NodeGeometryEditResult::Success,
        "local transform bakes into geometry") == false) {
    return false;
  }
  const etx::SceneNode& edited = data.hierarchy.nodes[edited_node];
  const uint32_t edited_mesh_index = data.hierarchy.attachments[edited.attachment_offset].resource_index;
  const etx::SceneNode& other = data.hierarchy.nodes[other_node];
  const uint32_t other_mesh_index = data.hierarchy.attachments[other.attachment_offset].resource_index;
  const etx::UpdateFlags geometry_changes = data.compute_hashes().compare(original_hashes);
  if (check_condition(edited_mesh_index != mesh_index, "bake creates private mesh geometry") == false ||
      check_condition(other_mesh_index == mesh_index, "bake leaves another shared instance on its source mesh") == false ||
      check_condition(affine_nearly_equal(edited.local_transform, {}), "bake resets the node local transform") == false ||
      check_condition(affine_nearly_equal(data.hierarchy.world_transforms[child_node], original_child_world), "bake preserves direct child world transforms") == false ||
      check_condition(nearly_equal(data.vertices.pos[1], original_source_position), "bake leaves source vertices unchanged") == false ||
      check_condition(geometry_changes[etx::UpdateFlags::AnyGeometryStructure], "bake requests renderer geometry and acceleration-structure rebuilds") == false) {
    return false;
  }

  const Mesh& edited_mesh = data.meshes[edited_mesh_index];
  const Triangle& edited_triangle = data.triangles[edited_mesh.triangle_offset];
  for (uint32_t corner = 0u; corner < 3u; ++corner) {
    const float3 expected = etx::transform_point(local_transform, data.vertices.pos[triangle.i[corner]]);
    if (check_condition(nearly_equal(data.vertices.pos[edited_triangle.i[corner]], expected), "baked vertex matches the former local transform") == false) {
      return false;
    }
  }
  return check_condition(nearly_equal(edited_triangle.geo_n, {0.0f, 0.0f, -1.0f}) && nearly_equal(data.vertices.nrm[edited_triangle.i[0]], {0.0f, 0.0f, -1.0f}),
    "bake preserves mirrored geometric and shading-normal orientation");
}

bool test_center_node_pivot_preserves_geometry_and_children() {
  etx::TaskScheduler scheduler = {};
  etx::IORDatabase ior_database = {};
  etx::SceneRepresentation scene(scheduler, ior_database);
  etx::SceneData& data = scene.data();
  data.vertices.pos = {{0.0f, 0.0f, 0.0f}, {6.0f, 0.0f, 0.0f}, {0.0f, 3.0f, 0.0f}};
  Triangle triangle = {};
  triangle.i[0] = 0u;
  triangle.i[1] = 1u;
  triangle.i[2] = 2u;
  triangle.geo_n = {0.0f, 0.0f, 1.0f};
  data.triangles.push_back(triangle);
  const uint32_t mesh_index = data.add_mesh_asset("pivot-source", 0u, 1u, {0.0f, 0.0f, 0.0f}, {6.0f, 3.0f, 0.0f});

  const etx::AffineTransform local_transform = make_translation_scale({5.0f, 7.0f, 0.0f}, {2.0f, 3.0f, 1.0f});
  const uint32_t edited_node = data.hierarchy.add_node("pivot", kInvalidIndex, local_transform);
  const etx::AffineTransform child_local = make_translation_scale({1.0f, 2.0f, 3.0f}, {1.0f, 1.0f, 1.0f});
  const uint32_t child_node = data.hierarchy.add_node("child", edited_node, child_local);
  data.hierarchy.add_attachment(edited_node, {etx::SceneAttachment::Type::Mesh, mesh_index, 0u, 0u});
  if (check_condition(data.resolve_hierarchy(), "pivot test hierarchy resolves") == false) {
    return false;
  }
  const etx::AffineTransform original_child_world = data.hierarchy.world_transforms[child_node];
  float3 original_world_positions[3] = {};
  for (uint32_t corner = 0u; corner < 3u; ++corner) {
    original_world_positions[corner] = etx::transform_point(local_transform, data.vertices.pos[triangle.i[corner]]);
  }

  if (check_condition(scene.edit_node_geometry(edited_node, etx::NodeGeometryOperation::CenterPivot) == etx::NodeGeometryEditResult::Success,
        "pivot moves to surface center of mass") == false) {
    return false;
  }
  const etx::SceneNode& edited = data.hierarchy.nodes[edited_node];
  const uint32_t edited_mesh_index = data.hierarchy.attachments[edited.attachment_offset].resource_index;
  const Mesh& edited_mesh = data.meshes[edited_mesh_index];
  const Triangle& edited_triangle = data.triangles[edited_mesh.triangle_offset];
  const float3 expected_center = {2.0f, 1.0f, 0.0f};
  if (check_condition(nearly_equal(float3{edited.local_transform.rows[0].w, edited.local_transform.rows[1].w, edited.local_transform.rows[2].w}, {9.0f, 10.0f, 0.0f}),
        "pivot moves to the transformed triangle center") == false ||
      check_condition(affine_nearly_equal(data.hierarchy.world_transforms[child_node], original_child_world), "pivot edit preserves direct child world transforms") == false ||
      check_condition(nearly_equal(data.vertices.pos[0], {0.0f, 0.0f, 0.0f}), "pivot edit leaves source vertices unchanged") == false) {
    return false;
  }
  for (uint32_t corner = 0u; corner < 3u; ++corner) {
    const float3 edited_local_position = data.vertices.pos[edited_triangle.i[corner]];
    if (check_condition(nearly_equal(edited_local_position, data.vertices.pos[triangle.i[corner]] - expected_center), "pivot recenters private local geometry") == false ||
        check_condition(nearly_equal(etx::transform_point(edited.local_transform, edited_local_position), original_world_positions[corner]),
          "pivot edit preserves rendered vertex positions") == false) {
      return false;
    }
  }
  return true;
}

bool test_node_geometry_edits_reject_mixed_attachments() {
  etx::TaskScheduler scheduler = {};
  etx::IORDatabase ior_database = {};
  etx::SceneRepresentation scene(scheduler, ior_database);
  etx::SceneData& data = scene.data();
  data.vertices.pos = {{0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}};
  Triangle triangle = {};
  triangle.i[0] = 0u;
  triangle.i[1] = 1u;
  triangle.i[2] = 2u;
  data.triangles.push_back(triangle);
  data.meshes.push_back({{0.0f, 0.0f, 0.0f}, 0u, {1.0f, 1.0f, 0.0f}, 1u});
  data.cameras.emplace_back();
  const uint32_t node_index = data.hierarchy.add_node("mixed", kInvalidIndex, {});
  data.hierarchy.add_attachment(node_index, {etx::SceneAttachment::Type::Mesh, 0u, 0u, 0u});
  data.hierarchy.add_attachment(node_index, {etx::SceneAttachment::Type::Camera, 0u, 0u, 0u});
  const size_t original_mesh_count = data.meshes.size();
  const size_t original_vertex_count = data.vertices.pos.size();
  const etx::NodeGeometryEditResult result = scene.edit_node_geometry(node_index, etx::NodeGeometryOperation::BakeLocalTransform);
  return check_condition(result == etx::NodeGeometryEditResult::NonMeshAttachments, "mixed node attachments reject geometry-only edits") &&
         check_condition((data.meshes.size() == original_mesh_count) && (data.vertices.pos.size() == original_vertex_count), "rejected geometry edit does not mutate scene data");
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
  etx::SceneHashes transformed_hashes = initial_hashes;
  transformed_hashes.transforms_hash = scene_data.compute_transforms_hash();
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
  if (check_condition(raytracing.trace(raytracing.scene(), Ray{{3.0f, 0.0f, 1.0f}, {0.0f, 0.0f, -1.0f}}, new_position_intersection, new_position_sampler),
        "Embree refit exposes transformed instance position") == false) {
    return false;
  }

  if (check_condition(scene_data.hierarchy.set_enabled(0u, false) && scene_data.resolve_hierarchy(), "disabled Embree hierarchy resolves") == false) {
    return false;
  }
  etx::SceneHashes disabled_hashes = transformed_hashes;
  disabled_hashes.transforms_hash = scene_data.compute_transforms_hash();
  const etx::UpdateFlags visibility_changes = disabled_hashes.compare(transformed_hashes);
  if (check_condition(visibility_changes[etx::UpdateFlags::Transforms] && (visibility_changes[etx::UpdateFlags::AnyGeometryStructure] == false),
        "visibility-only Embree update is classified as an instance refit") == false) {
    return false;
  }
  raytracing.commit(scene_data, camera, visibility_changes);
  etx::Sampler disabled_sampler(7u, 8u);
  Intersection disabled_intersection = {};
  return check_condition(raytracing.trace(raytracing.scene(), Ray{{3.0f, 0.0f, 1.0f}, {0.0f, 0.0f, -1.0f}}, disabled_intersection, disabled_sampler) == false,
    "Embree instance-mask update hides a disabled node without rebuilding geometry");
}

bool test_preview_resolution_adapts_with_hysteresis() {
  etx::PreviewResolutionController controller(4u, 4u);
  controller.begin();
  if (check_condition(controller.active() && (controller.pixel_size() == 4u), "preview adaptation starts at its retained resolution") == false ||
      check_condition(controller.update(0.050, true) == false, "one slow output does not immediately reduce preview resolution") == false ||
      check_condition((controller.update(0.050, true) == false) && (controller.pixel_size() == 4u), "slow output cannot exceed the maximum preview pixel block") == false) {
    return false;
  }

  for (uint32_t sample = 0u; sample < 5u; ++sample) {
    if (check_condition(controller.update(0.010, true) == false, "fast preview output respects refinement hysteresis") == false) {
      return false;
    }
  }
  if (check_condition(controller.update(0.010, true) && (controller.pixel_size() == 2u), "sustained fast output refines preview resolution") == false) {
    return false;
  }

  controller.end();
  controller.begin();
  if (check_condition(controller.pixel_size() == 4u, "each preview interaction starts with the configured coarse pixel block") == false ||
      check_condition(controller.update(0.050, true) == false, "one slow output at maximum keeps the preview resolution") == false ||
      check_condition((controller.update(0.050, true) == false) && (controller.pixel_size() == 4u), "preview coarsening remains capped at a four-pixel block") == false) {
    return false;
  }

  controller.end();
  return check_condition((controller.active() == false) && (controller.update(1.0, false) == false) && (controller.pixel_size() == 4u), "inactive preview does not adapt");
}

bool test_preview_iteration_completes_before_pending_scene_commit() {
  struct IntegratorProbe : etx::Integrator {
    using Integrator::Integrator;

    void run() override {
      current_state = State::Running;
      current_status = {};
      update_count = 0u;
    }

    void update() override {
      if ((current_state != State::Running) || (update_count >= 2u)) {
        return;
      }
      update_count += 1u;
      if (update_count == 2u) {
        current_status.completed_iterations = 1u;
        current_status.last_iteration_time = 0.01;
      }
    }

    void stop(Stop) override {
      current_state = State::Stopped;
    }

    const Status& status() const override {
      return current_status;
    }

    Status current_status = {};
    uint32_t update_count = 0u;
  };

  etx::TaskScheduler scheduler = {};
  etx::IORDatabase ior_database = {};
  etx::SceneRepresentation scene(scheduler, ior_database);
  etx::Film film(scheduler);
  etx::Raytracing raytracing(scheduler, film);
  IntegratorProbe integrator(raytracing);
  etx::IntegratorThread integrator_thread(scene, raytracing);
  integrator_thread.start(&integrator);
  integrator_thread.run();

  if (check_condition(integrator_thread.update_integrator() == false, "preview polling starts the coarse iteration without reporting an output") == false ||
      check_condition(integrator_thread.scene_changes_pending(), "initial scene commit remains queued while the coarse iteration runs") == false) {
    return false;
  }

  if (check_condition(integrator_thread.update_integrator(), "completed coarse iteration is observable before scene changes are committed") == false) {
    return false;
  }
  if (check_condition(integrator_thread.scene_changes_pending(), "completed coarse output remains publishable while the newest scene commit stays queued") == false) {
    return false;
  }

  integrator_thread.stop(etx::Integrator::Stop::Immediate);
  return check_condition(integrator.state() == etx::Integrator::State::Stopped, "immediate stop is consumed without committing a queued scene update");
}

bool test_external_medium_change_is_detected_without_notification() {
  struct IntegratorProbe : etx::Integrator {
    using Integrator::Integrator;

    void run() override {
      run_count += 1u;
      current_state = State::Running;
      current_status = {};
    }

    void update() override {
    }

    void stop(Stop) override {
      stop_count += 1u;
      current_state = State::Stopped;
    }

    const Status& status() const override {
      return current_status;
    }

    Status current_status = {};
    uint32_t run_count = 0u;
    uint32_t stop_count = 0u;
  };

  struct SceneGlobalGuard {
    SceneGlobalGuard() {
      etx::scene_global_init();
    }
    ~SceneGlobalGuard() {
      etx::scene_global_deinit();
    }
  } scene_global_guard;

  etx::TaskScheduler scheduler = {};
  etx::IORDatabase ior_database = {};
  etx::SceneRepresentation scene(scheduler, ior_database);
  etx::Film film(scheduler);
  etx::Raytracing raytracing(scheduler, film);
  IntegratorProbe integrator(raytracing);
  etx::IntegratorThread integrator_thread(scene, raytracing);

  scene.camera().film_size = {1u, 1u};
  const uint32_t medium_index = scene.add_medium("Regression Medium");
  Material material = {};
  material.cls = MaterialClass::Diffuse;
  material.opacity = 1.0f;
  scene.data().materials.push_back(material);
  integrator_thread.start(&integrator);
  integrator_thread.commit_scene_changes();

  const uint64_t initial_revision = integrator_thread.scene_revision();
  const uint32_t initial_run_count = integrator.run_count;
  integrator_thread.commit_scene_changes();
  if (check_condition(integrator.run_count == initial_run_count, "unchanged scene polling does not restart a running integrator") == false ||
      check_condition(integrator.stop_count == 0u, "unchanged scene polling does not stop a running integrator") == false ||
      check_condition(integrator_thread.scene_revision() == initial_revision, "unchanged scene polling preserves the committed revision") == false) {
    return false;
  }
  scene.data().materials.front().ext_medium = medium_index;
  if (check_condition(raytracing.scene().materials[0u].ext_medium != medium_index, "active CPU render snapshot remains immutable until the next scene commit") == false) {
    return false;
  }
  integrator_thread.commit_scene_changes();

  const bool result = check_condition(integrator_thread.scene_revision() == (initial_revision + 1u), "external-medium edit advances the detected scene revision") &&
                      check_condition(integrator.stop_count == 1u, "external-medium edit stops the active integrator before replacing its render snapshot") &&
                      check_condition(integrator.run_count == (initial_run_count + 1u), "external-medium edit restarts the active integrator without an explicit notification") &&
                      check_condition(raytracing.scene().materials[0u].ext_medium == medium_index, "external-medium edit is published in the next CPU render snapshot");
  integrator_thread.stop(etx::Integrator::Stop::Immediate);
  return result;
}

bool test_buffer_view_rejects_recycled_slot() {
  etx::BufferPool pool = {};
  const etx::BufferHandle first_handle = pool.create();
  const etx::BufferView stale_view = pool.allocate_elements<uint32_t>(first_handle, 1u);
  const uint32_t first_value = 17u;
  if (check_condition(pool.write(stale_view, &first_value, sizeof(first_value)), "initial buffer view accepts writes") == false) {
    return false;
  }

  pool.destroy(first_handle);
  const etx::BufferHandle second_handle = pool.create();
  const etx::BufferView current_view = pool.allocate_elements<uint32_t>(second_handle, 1u);
  const uint32_t second_value = 31u;
  if (check_condition(second_handle.index == first_handle.index, "buffer pool recycles the destroyed slot") == false ||
      check_condition(second_handle.generation != first_handle.generation, "recycled buffer slot advances its generation") == false ||
      check_condition(pool.write(current_view, &second_value, sizeof(second_value)), "current buffer view accepts writes") == false) {
    return false;
  }

  const uint32_t stale_value = 47u;
  const uint32_t* current_data = pool.map<uint32_t>(current_view);
  return check_condition(pool.map(stale_view) == nullptr, "stale buffer view cannot map a recycled slot") &&
         check_condition(pool.write(stale_view, &stale_value, sizeof(stale_value)) == false, "stale buffer view cannot overwrite a recycled slot") &&
         check_condition((current_data != nullptr) && (*current_data == second_value), "rejected stale access leaves the current allocation unchanged") &&
         check_condition(pool.allocate_elements<uint64_t>(second_handle, std::numeric_limits<uint64_t>::max()).valid() == false, "overflowing element allocation is rejected");
}

bool test_cpu_render_snapshot_owns_medium_density() {
  struct SceneGlobalGuard {
    SceneGlobalGuard() {
      etx::scene_global_init();
    }
    ~SceneGlobalGuard() {
      etx::scene_global_deinit();
    }
  } scene_global_guard;

  etx::TaskScheduler scheduler = {};
  etx::IORDatabase ior_database = {};
  etx::SceneRepresentation scene(scheduler, ior_database);
  etx::Film film(scheduler);
  etx::Raytracing raytracing(scheduler, film);
  scene.camera().film_size = {1u, 1u};

  const uint32_t medium_index = scene.add_medium("Density Snapshot Medium");
  etx::Medium& medium = scene.data().mediums.get(medium_index);
  medium.cls = etx::Medium::Heterogeneous;
  medium.grid.dimensions = {1u, 1u, 1u};
  medium.density_buffer = scene.data().buffer_pool.create();
  medium.density_data = scene.data().buffer_pool.allocate_elements<float>(medium.density_buffer, 1u);
  const float initial_density = 0.25f;
  if (check_condition(scene.data().buffer_pool.write(medium.density_data, &initial_density, sizeof(initial_density)), "medium density payload is initialized") == false) {
    return false;
  }
  medium.density_view = {scene.data().buffer_pool.map<float>(medium.density_data), 1u};

  raytracing.commit(scene.data(), scene.camera(), scene.data().compute_hashes().compare({}));
  const etx::Medium& committed_medium = raytracing.scene().mediums[medium_index];
  if (check_condition((committed_medium.density_view.a != nullptr) && (committed_medium.density_view[0u] == initial_density), "CPU snapshot owns the committed medium density") ==
      false) {
    return false;
  }

  const float edited_density = 0.75f;
  if (check_condition(scene.data().buffer_pool.write(medium.density_data, &edited_density, sizeof(edited_density)), "source medium density payload accepts an edit") == false) {
    return false;
  }
  return check_condition(committed_medium.density_view[0u] == initial_density, "active CPU snapshot is unaffected by source density edits");
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
    {"reparent_preserves_world_transform", test_reparent_preserves_world_transform},
    {"transform_resolution_and_instances", test_transform_resolution_and_instances},
    {"affine_trs_round_trip_and_shear_detection", test_affine_trs_round_trip_and_shear_detection},
    {"affine_inverse_is_scale_aware", test_affine_inverse_is_scale_aware},
    {"attachment_ranges_remain_valid", test_attachment_ranges_remain_valid},
    {"visibility_is_inherited", test_visibility_is_inherited},
    {"mirrored_tangent_frame_handedness", test_mirrored_tangent_frame_handedness},
    {"shading_frame_finalization", test_shading_frame_finalization},
    {"vcm_vertex_restores_world_tangent_frame", test_vcm_vertex_restores_world_tangent_frame},
    {"equirectangular_camera_uses_orientation", test_equirectangular_camera_uses_orientation},
    {"disabled_attached_emitter_is_not_global", test_disabled_attached_emitter_is_not_global},
    {"directional_and_area_emitters_use_node_transforms", test_directional_and_area_emitters_use_node_transforms},
    {"scene_update_scope_coalescing", test_scene_update_scope_coalescing},
    {"camera_interaction_queues_updates_until_input_released", test_camera_interaction_queues_updates_until_input_released},
    {"environment_emitter_uses_node_rotation", test_environment_emitter_uses_node_rotation},
    {"active_camera_uses_node_transform", test_active_camera_uses_node_transform},
    {"hierarchy_hashes_content", test_hierarchy_hashes_content},
    {"enabled_state_is_an_instance_update", test_enabled_state_is_an_instance_update},
    {"failed_resolution_remains_dirty_and_recovers", test_failed_resolution_remains_dirty_and_recovers},
    {"attachment_removal_repairs_ranges", test_attachment_removal_repairs_ranges},
    {"transformed_medium_coordinates_and_bounds", test_transformed_medium_coordinates_and_bounds},
    {"attached_medium_bounds_follow_visibility_and_transform", test_attached_medium_bounds_follow_visibility_and_transform},
    {"bake_node_transform_isolates_geometry_and_preserves_children", test_bake_node_transform_isolates_geometry_and_preserves_children},
    {"center_node_pivot_preserves_geometry_and_children", test_center_node_pivot_preserves_geometry_and_children},
    {"node_geometry_edits_reject_mixed_attachments", test_node_geometry_edits_reject_mixed_attachments},
    {"embree_transform_only_commit", test_embree_transform_only_commit},
    {"preview_resolution_adapts_with_hysteresis", test_preview_resolution_adapts_with_hysteresis},
    {"preview_iteration_completes_before_pending_scene_commit", test_preview_iteration_completes_before_pending_scene_commit},
    {"external_medium_change_is_detected_without_notification", test_external_medium_change_is_detected_without_notification},
    {"buffer_view_rejects_recycled_slot", test_buffer_view_rejects_recycled_slot},
    {"cpu_render_snapshot_owns_medium_density", test_cpu_render_snapshot_owns_medium_density},
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
