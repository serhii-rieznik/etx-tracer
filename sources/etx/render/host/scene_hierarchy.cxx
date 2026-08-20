#include <etx/render/host/scene_hierarchy.hxx>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace etx {
namespace {

struct TraversalEntry {
  uint32_t node_index = kInvalidIndex;
  uint32_t next_child = 0u;
};

bool affine_is_finite(const AffineTransform& transform) {
  for (const float4& row : transform.rows) {
    if ((std::isfinite(row.x) == false) || (std::isfinite(row.y) == false) || (std::isfinite(row.z) == false) || (std::isfinite(row.w) == false)) {
      return false;
    }
  }
  return true;
}

bool finite_float(double value) {
  return std::isfinite(value) && (std::abs(value) <= static_cast<double>(std::numeric_limits<float>::max()));
}

bool affine_decompose_axes(const AffineTransform& transform, float3 (&axes)[3], float (&scales)[3]) {
  if (affine_is_finite(transform) == false) {
    return false;
  }

  axes[0] = {transform.rows[0].x, transform.rows[1].x, transform.rows[2].x};
  axes[1] = {transform.rows[0].y, transform.rows[1].y, transform.rows[2].y};
  axes[2] = {transform.rows[0].z, transform.rows[1].z, transform.rows[2].z};
  auto axis_length = [](const float3& axis) {
    return static_cast<float>(std::sqrt(static_cast<double>(axis.x) * axis.x + static_cast<double>(axis.y) * axis.y + static_cast<double>(axis.z) * axis.z));
  };
  scales[0] = axis_length(axes[0]);
  scales[1] = axis_length(axes[1]);
  scales[2] = axis_length(axes[2]);
  const float minimum_scale = std::numeric_limits<float>::min();
  if ((scales[0] <= minimum_scale) || (scales[1] <= minimum_scale) || (scales[2] <= minimum_scale)) {
    return false;
  }

  axes[0] /= scales[0];
  axes[1] /= scales[1];
  axes[2] /= scales[2];
  const float orthogonal_tolerance = 64.0f * std::numeric_limits<float>::epsilon();
  if ((std::abs(dot(axes[0], axes[1])) > orthogonal_tolerance) || (std::abs(dot(axes[0], axes[2])) > orthogonal_tolerance) ||
      (std::abs(dot(axes[1], axes[2])) > orthogonal_tolerance)) {
    return false;
  }

  const float orientation = dot(cross(axes[0], axes[1]), axes[2]);
  if (std::abs(std::abs(orientation) - 1.0f) > orthogonal_tolerance) {
    return false;
  }
  if (orientation < 0.0f) {
    uint32_t reflection_axis = 0u;
    if (scales[1] > scales[reflection_axis]) {
      reflection_axis = 1u;
    }
    if (scales[2] > scales[reflection_axis]) {
      reflection_axis = 2u;
    }
    axes[reflection_axis] = -axes[reflection_axis];
    scales[reflection_axis] = -scales[reflection_axis];
  }
  return true;
}

bool affine_extract_orientation(const AffineTransform& transform, AffineTransform& orientation) {
  float3 axes[3] = {};
  float scales[3] = {};
  if (affine_decompose_axes(transform, axes, scales) == false) {
    return false;
  }
  orientation.rows[0] = {axes[0].x, axes[1].x, axes[2].x, 0.0f};
  orientation.rows[1] = {axes[0].y, axes[1].y, axes[2].y, 0.0f};
  orientation.rows[2] = {axes[0].z, axes[1].z, axes[2].z, 0.0f};
  return true;
}

}  // namespace

AffineTransform affine_from_matrix(const float4x4& matrix) {
  AffineTransform result = {};
  result.rows[0] = {matrix.col[0].x, matrix.col[1].x, matrix.col[2].x, matrix.col[3].x};
  result.rows[1] = {matrix.col[0].y, matrix.col[1].y, matrix.col[2].y, matrix.col[3].y};
  result.rows[2] = {matrix.col[0].z, matrix.col[1].z, matrix.col[2].z, matrix.col[3].z};
  return result;
}

float4x4 matrix_from_affine(const AffineTransform& transform) {
  float4x4 result = {};
  result.col[0] = {transform.rows[0].x, transform.rows[1].x, transform.rows[2].x, 0.0f};
  result.col[1] = {transform.rows[0].y, transform.rows[1].y, transform.rows[2].y, 0.0f};
  result.col[2] = {transform.rows[0].z, transform.rows[1].z, transform.rows[2].z, 0.0f};
  result.col[3] = {transform.rows[0].w, transform.rows[1].w, transform.rows[2].w, 1.0f};
  return result;
}

AffineTransform affine_from_trs(const AffineTRS& trs) {
  const float cx = std::cos(trs.rotation_radians.x);
  const float sx = std::sin(trs.rotation_radians.x);
  const float cy = std::cos(trs.rotation_radians.y);
  const float sy = std::sin(trs.rotation_radians.y);
  const float cz = std::cos(trs.rotation_radians.z);
  const float sz = std::sin(trs.rotation_radians.z);

  AffineTransform result = {};
  result.rows[0] = {cz * cy * trs.scale.x, (cz * sy * sx - sz * cx) * trs.scale.y, (cz * sy * cx + sz * sx) * trs.scale.z, trs.translation.x};
  result.rows[1] = {sz * cy * trs.scale.x, (sz * sy * sx + cz * cx) * trs.scale.y, (sz * sy * cx - cz * sx) * trs.scale.z, trs.translation.y};
  result.rows[2] = {-sy * trs.scale.x, cy * sx * trs.scale.y, cy * cx * trs.scale.z, trs.translation.z};
  return result;
}

bool affine_to_trs(const AffineTransform& transform, AffineTRS& trs) {
  trs.translation = {transform.rows[0].w, transform.rows[1].w, transform.rows[2].w};
  float3 axes[3] = {};
  float scales[3] = {};
  if (affine_decompose_axes(transform, axes, scales) == false) {
    return false;
  }
  const float orthogonal_tolerance = 64.0f * std::numeric_limits<float>::epsilon();

  const float r00 = axes[0].x;
  const float r10 = axes[0].y;
  const float r20 = axes[0].z;
  const float r01 = axes[1].x;
  const float r11 = axes[1].y;
  const float r21 = axes[1].z;
  const float r22 = axes[2].z;
  const float rotation_y = std::asin(clamp(-r20, -1.0f, 1.0f));
  const float cos_y = std::cos(rotation_y);
  float rotation_x = 0.0f;
  float rotation_z = 0.0f;
  if (std::abs(cos_y) > orthogonal_tolerance) {
    rotation_x = std::atan2(r21, r22);
    rotation_z = std::atan2(r10, r00);
  } else {
    rotation_x = (r20 < 0.0f) ? std::atan2(r01, r11) : std::atan2(-r01, r11);
  }

  trs.rotation_radians = {rotation_x, rotation_y, rotation_z};
  trs.scale = {scales[0], scales[1], scales[2]};
  return true;
}

float3 transform_point(const AffineTransform& transform, const float3& point) {
  return {
    transform.rows[0].x * point.x + transform.rows[0].y * point.y + transform.rows[0].z * point.z + transform.rows[0].w,
    transform.rows[1].x * point.x + transform.rows[1].y * point.y + transform.rows[1].z * point.z + transform.rows[1].w,
    transform.rows[2].x * point.x + transform.rows[2].y * point.y + transform.rows[2].z * point.z + transform.rows[2].w,
  };
}

float3 transform_vector(const AffineTransform& transform, const float3& vector) {
  return {
    transform.rows[0].x * vector.x + transform.rows[0].y * vector.y + transform.rows[0].z * vector.z,
    transform.rows[1].x * vector.x + transform.rows[1].y * vector.y + transform.rows[1].z * vector.z,
    transform.rows[2].x * vector.x + transform.rows[2].y * vector.y + transform.rows[2].z * vector.z,
  };
}

AffineTransform multiply_affine(const AffineTransform& parent, const AffineTransform& local) {
  AffineTransform result = {};
  for (uint32_t row = 0u; row < 3u; ++row) {
    const float4& p = parent.rows[row];
    result.rows[row] = {
      p.x * local.rows[0].x + p.y * local.rows[1].x + p.z * local.rows[2].x,
      p.x * local.rows[0].y + p.y * local.rows[1].y + p.z * local.rows[2].y,
      p.x * local.rows[0].z + p.y * local.rows[1].z + p.z * local.rows[2].z,
      p.x * local.rows[0].w + p.y * local.rows[1].w + p.z * local.rows[2].w + p.w,
    };
  }
  return result;
}

bool invert_affine(const AffineTransform& transform, AffineTransform& result, double& determinant) {
  const double a00 = transform.rows[0].x;
  const double a01 = transform.rows[0].y;
  const double a02 = transform.rows[0].z;
  const double a10 = transform.rows[1].x;
  const double a11 = transform.rows[1].y;
  const double a12 = transform.rows[1].z;
  const double a20 = transform.rows[2].x;
  const double a21 = transform.rows[2].y;
  const double a22 = transform.rows[2].z;

  const double c00 = a11 * a22 - a12 * a21;
  const double c01 = a12 * a20 - a10 * a22;
  const double c02 = a10 * a21 - a11 * a20;
  determinant = a00 * c00 + a01 * c01 + a02 * c02;
  if ((determinant == 0.0) || (std::isfinite(determinant) == false)) {
    return false;
  }

  const double inverse_determinant = 1.0 / determinant;
  const double inverse[3][3] = {
    {c00 * inverse_determinant, (a02 * a21 - a01 * a22) * inverse_determinant, (a01 * a12 - a02 * a11) * inverse_determinant},
    {c01 * inverse_determinant, (a00 * a22 - a02 * a20) * inverse_determinant, (a02 * a10 - a00 * a12) * inverse_determinant},
    {c02 * inverse_determinant, (a01 * a20 - a00 * a21) * inverse_determinant, (a00 * a11 - a01 * a10) * inverse_determinant},
  };
  const double matrix_norm = std::max({std::abs(a00) + std::abs(a01) + std::abs(a02), std::abs(a10) + std::abs(a11) + std::abs(a12),
    std::abs(a20) + std::abs(a21) + std::abs(a22)});
  const double inverse_norm = std::max({std::abs(inverse[0][0]) + std::abs(inverse[0][1]) + std::abs(inverse[0][2]),
    std::abs(inverse[1][0]) + std::abs(inverse[1][1]) + std::abs(inverse[1][2]), std::abs(inverse[2][0]) + std::abs(inverse[2][1]) + std::abs(inverse[2][2])});
  if ((matrix_norm * inverse_norm * static_cast<double>(std::numeric_limits<float>::epsilon())) >= 1.0) {
    return false;
  }

  const double translation[3] = {transform.rows[0].w, transform.rows[1].w, transform.rows[2].w};
  const double inverse_translation[3] = {
    -(inverse[0][0] * translation[0] + inverse[0][1] * translation[1] + inverse[0][2] * translation[2]),
    -(inverse[1][0] * translation[0] + inverse[1][1] * translation[1] + inverse[1][2] * translation[2]),
    -(inverse[2][0] * translation[0] + inverse[2][1] * translation[1] + inverse[2][2] * translation[2]),
  };
  for (uint32_t row = 0u; row < 3u; ++row) {
    if ((finite_float(inverse[row][0]) == false) || (finite_float(inverse[row][1]) == false) || (finite_float(inverse[row][2]) == false) ||
        (finite_float(inverse_translation[row]) == false)) {
      return false;
    }
    result.rows[row] = {static_cast<float>(inverse[row][0]), static_cast<float>(inverse[row][1]), static_cast<float>(inverse[row][2]), static_cast<float>(inverse_translation[row])};
  }
  return true;
}

BoundingBox transform_bounding_box(const AffineTransform& transform, const BoundingBox& bounds) {
  const float3 center = 0.5f * (bounds.p_min + bounds.p_max);
  const float3 extent = 0.5f * (bounds.p_max - bounds.p_min);
  const float3 world_center = transform_point(transform, center);
  const float3 world_extent = {
    std::abs(transform.rows[0].x) * extent.x + std::abs(transform.rows[0].y) * extent.y + std::abs(transform.rows[0].z) * extent.z,
    std::abs(transform.rows[1].x) * extent.x + std::abs(transform.rows[1].y) * extent.y + std::abs(transform.rows[1].z) * extent.z,
    std::abs(transform.rows[2].x) * extent.x + std::abs(transform.rows[2].y) * extent.y + std::abs(transform.rows[2].z) * extent.z,
  };
  return {world_center - world_extent, 0.0f, world_center + world_extent, 0.0f};
}

void SceneHierarchy::clear() {
  nodes.clear();
  node_names.clear();
  attachments.clear();
  evaluation_order.clear();
  order_position.clear();
  subtree_end_position.clear();
  world_transforms.clear();
  world_orientations.clear();
  orientation_valid.clear();
  effective_enabled.clear();
  mesh_instances.clear();
  _dirty_begin = 0u;
  _dirty_end = kInvalidIndex;
  _topology_dirty = true;
  _resolved_state_dirty = true;
}

uint32_t SceneHierarchy::add_node(const char* name, uint32_t parent_index, const AffineTransform& local_transform) {
  if (((parent_index != kInvalidIndex) && (parent_index >= nodes.size())) || (affine_is_finite(local_transform) == false)) {
    return kInvalidIndex;
  }

  const uint32_t node_index = static_cast<uint32_t>(nodes.size());
  SceneNode& node = nodes.emplace_back();
  node.local_transform = local_transform;
  node.parent_index = parent_index;
  node_names.emplace_back((name != nullptr) && (name[0] != 0) ? name : ("node-" + std::to_string(node_index)));
  _topology_dirty = true;
  _resolved_state_dirty = true;
  return node_index;
}

bool SceneHierarchy::set_parent(uint32_t node_index, uint32_t parent_index) {
  if ((node_index >= nodes.size()) || (parent_index == node_index) || ((parent_index != kInvalidIndex) && (parent_index >= nodes.size()))) {
    return false;
  }
  if (nodes[node_index].parent_index == parent_index) {
    return true;
  }

  uint32_t ancestor = parent_index;
  while (ancestor != kInvalidIndex) {
    if (ancestor == node_index) {
      return false;
    }
    ancestor = nodes[ancestor].parent_index;
  }

  nodes[node_index].parent_index = parent_index;
  _topology_dirty = true;
  _resolved_state_dirty = true;
  return true;
}

bool SceneHierarchy::reparent_preserve_world(uint32_t node_index, uint32_t parent_index) {
  if ((node_index >= nodes.size()) || ((parent_index != kInvalidIndex) && (parent_index >= nodes.size()))) {
    return false;
  }
  if (nodes[node_index].parent_index == parent_index) {
    return true;
  }
  if (update_world_transforms() == false) {
    return false;
  }

  AffineTransform new_local_transform = world_transforms[node_index];
  if (parent_index != kInvalidIndex) {
    AffineTransform parent_to_local = {};
    double determinant = 0.0;
    if (invert_affine(world_transforms[parent_index], parent_to_local, determinant) == false) {
      return false;
    }
    new_local_transform = multiply_affine(parent_to_local, world_transforms[node_index]);
  }
  if ((affine_is_finite(new_local_transform) == false) || (set_parent(node_index, parent_index) == false)) {
    return false;
  }
  nodes[node_index].local_transform = new_local_transform;
  return true;
}

bool SceneHierarchy::set_local_transform(uint32_t node_index, const AffineTransform& local_transform) {
  if ((node_index >= nodes.size()) || (affine_is_finite(local_transform) == false)) {
    return false;
  }
  if (memcmp(&nodes[node_index].local_transform, &local_transform, sizeof(AffineTransform)) == 0) {
    return true;
  }
  nodes[node_index].local_transform = local_transform;
  mark_subtree_dirty(node_index);
  _resolved_state_dirty = true;
  return true;
}

bool SceneHierarchy::set_enabled(uint32_t node_index, bool enabled) {
  if (node_index >= nodes.size()) {
    return false;
  }
  const uint32_t enabled_flag = enabled ? SceneNode::Enabled : 0u;
  if ((nodes[node_index].flags & SceneNode::Enabled) == enabled_flag) {
    return true;
  }
  nodes[node_index].flags = (nodes[node_index].flags & (~SceneNode::Enabled)) | enabled_flag;
  _resolved_state_dirty = true;
  return true;
}

bool SceneHierarchy::add_attachment(uint32_t node_index, const SceneAttachment& attachment) {
  if (node_index >= nodes.size()) {
    return false;
  }

  SceneNode& node = nodes[node_index];
  const uint32_t insertion_position = node.attachment_offset + node.attachment_count;
  if ((node.attachment_count == 0u) && (insertion_position != attachments.size())) {
    node.attachment_offset = static_cast<uint32_t>(attachments.size());
  }

  if ((node.attachment_offset + node.attachment_count) == attachments.size()) {
    attachments.emplace_back(attachment);
  } else {
    attachments.insert(attachments.begin() + insertion_position, attachment);
    for (uint32_t other_index = 0u; other_index < nodes.size(); ++other_index) {
      if ((other_index != node_index) && (nodes[other_index].attachment_count > 0u) && (nodes[other_index].attachment_offset >= insertion_position)) {
        ++nodes[other_index].attachment_offset;
      }
    }
  }
  ++node.attachment_count;
  _resolved_state_dirty = true;
  return true;
}

bool SceneHierarchy::remove_attachment(uint32_t node_index, uint32_t local_attachment_index) {
  if ((node_index >= nodes.size()) || (local_attachment_index >= nodes[node_index].attachment_count)) {
    return false;
  }

  SceneNode& node = nodes[node_index];
  const uint32_t removal_position = node.attachment_offset + local_attachment_index;
  attachments.erase(attachments.begin() + removal_position);
  --node.attachment_count;
  for (uint32_t other_index = 0u; other_index < nodes.size(); ++other_index) {
    if ((other_index != node_index) && (nodes[other_index].attachment_count > 0u) && (nodes[other_index].attachment_offset > removal_position)) {
      --nodes[other_index].attachment_offset;
    }
  }
  if (node.attachment_count == 0u) {
    node.attachment_offset = 0u;
  }
  _resolved_state_dirty = true;
  return true;
}

bool SceneHierarchy::rebuild_topology() {
  const uint32_t node_count = static_cast<uint32_t>(nodes.size());
  evaluation_order.clear();
  evaluation_order.reserve(node_count);
  order_position.assign(node_count, kInvalidIndex);
  subtree_end_position.assign(node_count, kInvalidIndex);
  world_transforms.resize(node_count);
  world_orientations.resize(node_count);
  orientation_valid.assign(node_count, 0u);

  std::vector<uint32_t> child_counts(node_count, 0u);
  uint32_t root_count = 0u;
  for (uint32_t node_index = 0u; node_index < node_count; ++node_index) {
    const uint32_t parent_index = nodes[node_index].parent_index;
    if (parent_index == kInvalidIndex) {
      ++root_count;
      continue;
    }
    if ((parent_index >= node_count) || (parent_index == node_index)) {
      return false;
    }
    ++child_counts[parent_index];
  }

  std::vector<uint32_t> child_offsets(node_count + 1u, 0u);
  for (uint32_t node_index = 0u; node_index < node_count; ++node_index) {
    child_offsets[node_index + 1u] = child_offsets[node_index] + child_counts[node_index];
  }
  std::vector<uint32_t> children(child_offsets[node_count], kInvalidIndex);
  std::vector<uint32_t> write_offsets = child_offsets;
  for (uint32_t node_index = 0u; node_index < node_count; ++node_index) {
    const uint32_t parent_index = nodes[node_index].parent_index;
    if (parent_index != kInvalidIndex) {
      children[write_offsets[parent_index]++] = node_index;
    }
  }

  std::vector<TraversalEntry> stack;
  stack.reserve(node_count);
  for (uint32_t root_index = 0u; root_index < node_count; ++root_index) {
    if (nodes[root_index].parent_index != kInvalidIndex) {
      continue;
    }
    order_position[root_index] = static_cast<uint32_t>(evaluation_order.size());
    evaluation_order.emplace_back(root_index);
    stack.push_back({root_index, child_offsets[root_index]});

    while (stack.empty() == false) {
      TraversalEntry& entry = stack.back();
      const uint32_t child_end = child_offsets[entry.node_index + 1u];
      if (entry.next_child < child_end) {
        const uint32_t child_index = children[entry.next_child++];
        order_position[child_index] = static_cast<uint32_t>(evaluation_order.size());
        evaluation_order.emplace_back(child_index);
        stack.push_back({child_index, child_offsets[child_index]});
        continue;
      }
      subtree_end_position[entry.node_index] = static_cast<uint32_t>(evaluation_order.size());
      stack.pop_back();
    }
  }

  if ((evaluation_order.size() != node_count) || ((node_count > 0u) && (root_count == 0u))) {
    return false;
  }

  _topology_dirty = false;
  _dirty_begin = 0u;
  _dirty_end = node_count;
  return true;
}

bool SceneHierarchy::update_world_transforms() {
  if (_topology_dirty && (rebuild_topology() == false)) {
    return false;
  }
  if (_dirty_end == kInvalidIndex) {
    return true;
  }

  const uint32_t update_end = min(_dirty_end, static_cast<uint32_t>(evaluation_order.size()));
  for (uint32_t position = _dirty_begin; position < update_end; ++position) {
    const uint32_t node_index = evaluation_order[position];
    const SceneNode& node = nodes[node_index];
    AffineTransform local_orientation = {};
    const bool local_orientation_valid = affine_extract_orientation(node.local_transform, local_orientation);
    if (node.parent_index == kInvalidIndex) {
      world_transforms[node_index] = node.local_transform;
      orientation_valid[node_index] = local_orientation_valid ? 1u : 0u;
      world_orientations[node_index] = local_orientation;
    } else {
      world_transforms[node_index] = multiply_affine(world_transforms[node.parent_index], node.local_transform);
      const bool world_orientation_valid = local_orientation_valid && (orientation_valid[node.parent_index] != 0u);
      orientation_valid[node_index] = world_orientation_valid ? 1u : 0u;
      world_orientations[node_index] = world_orientation_valid ? multiply_affine(world_orientations[node.parent_index], local_orientation) : AffineTransform{};
    }
  }
  _dirty_begin = 0u;
  _dirty_end = kInvalidIndex;
  return true;
}

bool SceneHierarchy::resolve_mesh_instances(const std::vector<Mesh>& meshes) {
  if (update_world_transforms() == false) {
    return false;
  }

  mesh_instances.clear();
  mesh_instances.reserve(attachments.size());
  effective_enabled.assign(nodes.size(), 0u);
  for (uint32_t node_index : evaluation_order) {
    const SceneNode& node = nodes[node_index];
    const bool parent_enabled = (node.parent_index == kInvalidIndex) || (effective_enabled[node.parent_index] != 0u);
    const bool node_enabled = parent_enabled && ((node.flags & SceneNode::Enabled) != 0u);
    effective_enabled[node_index] = node_enabled ? 1u : 0u;

    const uint32_t attachment_end = node.attachment_offset + node.attachment_count;
    if (attachment_end > attachments.size()) {
      return false;
    }
    AffineTransform world_to_object = {};
    double determinant = 0.0;
    bool transform_evaluated = false;
    bool transform_valid = false;
    for (uint32_t attachment_index = node.attachment_offset; attachment_index < attachment_end; ++attachment_index) {
      const SceneAttachment& attachment = attachments[attachment_index];
      if (attachment.type != SceneAttachment::Type::Mesh) {
        continue;
      }
      if (attachment.resource_index >= meshes.size()) {
        return false;
      }
      if (transform_evaluated == false) {
        transform_valid = invert_affine(world_transforms[node_index], world_to_object, determinant);
        transform_evaluated = true;
      }

      const Mesh& mesh = meshes[attachment.resource_index];
      ResolvedMeshInstance& instance = mesh_instances.emplace_back();
      instance.node_index = node_index;
      instance.mesh_index = attachment.resource_index;
      if (transform_valid == false) {
        continue;
      }

      const BoundingBox local_bounds = {mesh.bbox_min, 0.0f, mesh.bbox_max, 0.0f};
      const BoundingBox world_bounds = transform_bounding_box(world_transforms[node_index], local_bounds);
      instance.object_to_world = world_transforms[node_index];
      instance.world_to_object = world_to_object;
      instance.bbox_min = world_bounds.p_min;
      instance.bbox_max = world_bounds.p_max;
      instance.flags = determinant < 0.0f ? ResolvedMeshInstance::Mirrored : 0u;
      if (node_enabled) {
        instance.flags |= ResolvedMeshInstance::Enabled;
      }
    }
  }
  return true;
}

bool SceneHierarchy::resolved_state_current() const {
  return _resolved_state_dirty == false;
}

void SceneHierarchy::mark_resolved_state_current() {
  _resolved_state_dirty = false;
}

void SceneHierarchy::mark_subtree_dirty(uint32_t node_index) {
  if (_topology_dirty || (node_index >= order_position.size()) || (order_position[node_index] == kInvalidIndex)) {
    _dirty_begin = 0u;
    _dirty_end = static_cast<uint32_t>(nodes.size());
    return;
  }

  const uint32_t begin = order_position[node_index];
  const uint32_t end = subtree_end_position[node_index];
  if (_dirty_end == kInvalidIndex) {
    _dirty_begin = begin;
    _dirty_end = end;
    return;
  }
  _dirty_begin = min(_dirty_begin, begin);
  _dirty_end = max(_dirty_end, end);
}

}  // namespace etx
