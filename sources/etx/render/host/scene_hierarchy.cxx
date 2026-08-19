#include <etx/render/host/scene_hierarchy.hxx>

#include <cmath>
#include <cstring>

namespace etx {
namespace {

constexpr float kMinimumAffineDeterminant = 1.0e-12f;

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

bool invert_affine(const AffineTransform& transform, AffineTransform& result, float& determinant) {
  const float a00 = transform.rows[0].x;
  const float a01 = transform.rows[0].y;
  const float a02 = transform.rows[0].z;
  const float a10 = transform.rows[1].x;
  const float a11 = transform.rows[1].y;
  const float a12 = transform.rows[1].z;
  const float a20 = transform.rows[2].x;
  const float a21 = transform.rows[2].y;
  const float a22 = transform.rows[2].z;

  const float c00 = a11 * a22 - a12 * a21;
  const float c01 = a12 * a20 - a10 * a22;
  const float c02 = a10 * a21 - a11 * a20;
  determinant = a00 * c00 + a01 * c01 + a02 * c02;
  if (std::isfinite(determinant) == false || (std::abs(determinant) <= kMinimumAffineDeterminant)) {
    return false;
  }

  const float inverse_determinant = 1.0f / determinant;
  result.rows[0] = {c00 * inverse_determinant, (a02 * a21 - a01 * a22) * inverse_determinant, (a01 * a12 - a02 * a11) * inverse_determinant, 0.0f};
  result.rows[1] = {c01 * inverse_determinant, (a00 * a22 - a02 * a20) * inverse_determinant, (a02 * a10 - a00 * a12) * inverse_determinant, 0.0f};
  result.rows[2] = {c02 * inverse_determinant, (a01 * a20 - a00 * a21) * inverse_determinant, (a00 * a11 - a01 * a10) * inverse_determinant, 0.0f};

  const float3 translation = {transform.rows[0].w, transform.rows[1].w, transform.rows[2].w};
  const float3 inverse_translation = transform_vector(result, -translation);
  result.rows[0].w = inverse_translation.x;
  result.rows[1].w = inverse_translation.y;
  result.rows[2].w = inverse_translation.z;
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
  effective_enabled.clear();
  mesh_instances.clear();
  _dirty_begin = 0u;
  _dirty_end = kInvalidIndex;
  _topology_dirty = true;
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
  return true;
}

bool SceneHierarchy::rebuild_topology() {
  const uint32_t node_count = static_cast<uint32_t>(nodes.size());
  evaluation_order.clear();
  evaluation_order.reserve(node_count);
  order_position.assign(node_count, kInvalidIndex);
  subtree_end_position.assign(node_count, kInvalidIndex);
  world_transforms.resize(node_count);

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
    if (node.parent_index == kInvalidIndex) {
      world_transforms[node_index] = node.local_transform;
    } else {
      world_transforms[node_index] = multiply_affine(world_transforms[node.parent_index], node.local_transform);
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

    AffineTransform world_to_object = {};
    float determinant = 0.0f;
    const bool transform_valid = invert_affine(world_transforms[node_index], world_to_object, determinant);

    const uint32_t attachment_end = node.attachment_offset + node.attachment_count;
    if (attachment_end > attachments.size()) {
      return false;
    }
    for (uint32_t attachment_index = node.attachment_offset; attachment_index < attachment_end; ++attachment_index) {
      const SceneAttachment& attachment = attachments[attachment_index];
      if (attachment.type != SceneAttachment::Type::Mesh) {
        continue;
      }
      if (attachment.resource_index >= meshes.size()) {
        return false;
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
