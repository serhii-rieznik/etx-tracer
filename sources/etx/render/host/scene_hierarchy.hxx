#pragma once

#include <etx/render/interop/interop_base.hxx>
#include <etx/render/shared/base.hxx>

#include <string>
#include <vector>

namespace etx {

using AffineTransform = ::AffineTransform;
struct SceneData;

struct SceneNode {
  enum : uint32_t {
    Enabled = 1u << 0u,
  };

  AffineTransform local_transform = {};
  uint32_t parent_index = kInvalidIndex;
  uint32_t attachment_offset = 0u;
  uint32_t attachment_count = 0u;
  uint32_t flags = Enabled;
};

struct SceneAttachment {
  enum class Type : uint32_t {
    Mesh,
    Camera,
    Emitter,
    Medium,
  };

  Type type = Type::Mesh;
  uint32_t resource_index = kInvalidIndex;
  uint32_t flags = 0u;
  uint32_t reserved = 0u;
};

struct ResolvedMeshInstance {
  enum : uint32_t {
    Mirrored = 1u << 0u,
    Enabled = 1u << 1u,
  };

  AffineTransform object_to_world = {};
  AffineTransform world_to_object = {};
  float3 bbox_min = {};
  uint32_t node_index = kInvalidIndex;
  float3 bbox_max = {};
  uint32_t mesh_index = kInvalidIndex;
  uint32_t flags = 0u;
};

struct AffineTRS {
  float3 translation = {};
  float3 rotation_radians = {};
  float3 scale = {1.0f, 1.0f, 1.0f};
};

struct SceneHierarchy {
  std::vector<SceneNode> nodes;
  std::vector<std::string> node_names;
  std::vector<SceneAttachment> attachments;

  std::vector<uint32_t> evaluation_order;
  std::vector<uint32_t> order_position;
  std::vector<uint32_t> subtree_end_position;
  std::vector<AffineTransform> world_transforms;
  std::vector<AffineTransform> world_orientations;
  std::vector<uint8_t> orientation_valid;
  std::vector<uint8_t> effective_enabled;
  std::vector<ResolvedMeshInstance> mesh_instances;

  void clear();

  uint32_t add_node(const char* name, uint32_t parent_index, const AffineTransform& local_transform);
  bool set_parent(uint32_t node_index, uint32_t parent_index);
  bool reparent_preserve_world(uint32_t node_index, uint32_t parent_index);
  bool set_local_transform(uint32_t node_index, const AffineTransform& local_transform);
  bool set_enabled(uint32_t node_index, bool enabled);
  bool add_attachment(uint32_t node_index, const SceneAttachment& attachment);
  bool remove_attachment(uint32_t node_index, uint32_t local_attachment_index);

  bool rebuild_topology();
  bool update_world_transforms();
  bool resolve_mesh_instances(const std::vector<Mesh>& meshes);
  bool resolved_state_current() const;

 private:
  friend struct SceneData;
  uint32_t _dirty_begin = 0u;
  uint32_t _dirty_end = kInvalidIndex;
  bool _topology_dirty = true;
  bool _resolved_state_dirty = true;

  void mark_subtree_dirty(uint32_t node_index);
  void mark_resolved_state_current();
};

AffineTransform affine_from_matrix(const float4x4& matrix);
float4x4 matrix_from_affine(const AffineTransform& transform);
AffineTransform affine_from_trs(const AffineTRS& trs);
bool affine_to_trs(const AffineTransform& transform, AffineTRS& trs);
AffineTransform multiply_affine(const AffineTransform& parent, const AffineTransform& local);
bool invert_affine(const AffineTransform& transform, AffineTransform& result, double& determinant);
float3 transform_point(const AffineTransform& transform, const float3& point);
float3 transform_vector(const AffineTransform& transform, const float3& vector);
BoundingBox transform_bounding_box(const AffineTransform& transform, const BoundingBox& bounds);

static_assert(sizeof(AffineTransform) == 48u);
static_assert(sizeof(SceneNode) == 64u);
static_assert(sizeof(SceneAttachment) == 16u);
static_assert(sizeof(SceneInstance) == 112u);

}  // namespace etx
